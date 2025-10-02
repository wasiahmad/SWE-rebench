# This file contains logic for running evaluations on TractoAI: <https://tracto.ai/>.

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Iterable

import docker
import pydantic
import yt.wrapper as yt

from swebench.harness.constants import (
    LOG_INSTANCE,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    PATCH_DIFF,
    SWEbenchInstance,
)
from swebench.harness.eval import get_log_dir, run_instance
from swebench.harness.reporting import make_run_report
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec
from swebench.harness.tracto_eval.utils import (
    configure_podman_storage,
    get_tracto_registry_creds_from_env,
    get_tracto_registry_url,
    logging_basic_config,
)

YT_PROXY_ENV = "YT_PROXY"
YT_TOKEN_ENV = "YT_TOKEN"
TRACTO_TENANT_ENV = "TRACTO_TENANT"
TRACTO_EVAL_IMAGE_ENV = "TRACTO_EVAL_IMAGE"
TRACTO_EVAL_RUNS_DIR_ENV = "TRACTO_EVAL_RUNS_DIR"
TRACTO_EVAL_MAX_PARALLEL_JOBS = int(os.getenv("TRACTO_EVAL_MAX_PARALLEL_JOBS", "100"))
TRACTO_EVAL_TMPFS_SIZE_GB = int(os.getenv("TRACTO_EVAL_TMPFS_SIZE_GB", "16"))

yt.config["pickling"]["ignore_system_modules"] = True
yt.config["pickling"]["dynamic_libraries"]["enable_auto_collection"] = False

logger = logging.getLogger(__name__)


def validate_tracto_env_vars():
    for env in (YT_PROXY_ENV, YT_TOKEN_ENV, TRACTO_EVAL_IMAGE_ENV):
        if env not in os.environ:
            raise RuntimeError(
                f"{env} environment variable is not set, "
                "check tracto_eval/README.md for details"
            )


def get_tracto_eval_run_dir(run_id: str) -> str | None:
    if TRACTO_EVAL_RUNS_DIR_ENV in os.environ:
        return f"{os.environ[TRACTO_EVAL_RUNS_DIR_ENV]}/{run_id}"

    logger.warning(
        f"{TRACTO_EVAL_RUNS_DIR_ENV} environment variable is not set, "
        "consider setting it to keep your eval inputs/outputs after eval run."
    )

    return None


class TestInput(pydantic.BaseModel):
    instance_id: str
    test_spec: TestSpec
    prediction: dict
    run_id: str
    timeout: int


class TestOutput(pydantic.BaseModel):
    instance_id: str
    test_output: str | None
    report_json_str: str | None
    run_instance_log: str | None
    patch_diff: str | None
    log_dir: str
    errored: bool
    exception: str | None

    # job metadata
    operation_id: str
    job_id: str


class PodmanDaemon:
    def __init__(
        self,
        socket_path: Path = Path("/run/podman/podman.sock"),
    ):
        self.proc = None
        self.socket_path = Path(socket_path)

    @property
    def socket_url(self) -> str:
        return f"unix://{self.socket_path}"

    def __enter__(self):
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        self.proc = subprocess.Popen(
            [
                "podman",
                "system",
                "service",
                "--time=0",
                self.socket_url,
            ],
        )

        return self

    def wait_for_podman(self):
        for _ in range(5):
            time.sleep(1)

            if (exitcode := self.proc.poll()) is not None:
                raise RuntimeError(f"Podman daemon exited prematurely, {exitcode=}")

            if self.socket_path.exists():
                logger.info(f"Podman socket {self.socket_path} appeared")
                return

        raise RuntimeError(f"Podman socket {self.socket_path} did not appear in time")

    def __exit__(self, exc_type, exc_val, exc_tb):
        # podman may spawn multiple processes
        subprocess.run(["pkill", "-SIGTERM", "podman"], check=True)

        try:
            self.proc.wait(1)
        except subprocess.TimeoutExpired:
            subprocess.run(["pkill", "-SIGKILL", "podman"], check=True)
            self.proc.wait()


class RunInstanceTracto:
    def __call__(self, test_input_raw: dict) -> Iterable[dict]:
        logging_basic_config()
        configure_podman_storage()

        test_input = TestInput.model_validate(test_input_raw)

        log_dir = get_log_dir(
            test_input.prediction, test_input.run_id, test_input.test_spec.instance_id
        )

        with PodmanDaemon() as podman_daemon:
            try:
                podman_daemon.wait_for_podman()

                docker_client = docker.DockerClient(base_url=podman_daemon.socket_url)

                tracto_registry_creds = get_tracto_registry_creds_from_env()
                docker_client.login(
                    username=tracto_registry_creds["username"],
                    password=tracto_registry_creds["password"],
                    registry=os.environ["TRACTO_REGISTRY_URL"],
                )

                logger.info("Running run_instance...")

                run_instance(
                    test_spec=test_input.test_spec,
                    pred=test_input.prediction,
                    rm_image=False,
                    force_rebuild=False,
                    client=docker_client,
                    run_id=test_input.run_id,
                    timeout=test_input.timeout,
                    rewrite_reports=False,
                    container_kwargs={
                        "network_mode": "host",
                    },
                    add_stderr_logger=True,
                )
            except Exception as e:
                errored = True
                if isinstance(e, subprocess.CalledProcessError):
                    exception_text = f"{str(e)}\nstderr:\n{e.stderr}"
                else:
                    exception_text = str(e)
                logger.exception("Exception occured:")
            else:
                errored = False
                exception_text = None

            logger.info("Finished run_instance")
        logger.info("PodmanDaemon context exited")

        test_output = TestOutput(
            instance_id=test_input.test_spec.instance_id,
            test_output=self._maybe_read_text(log_dir / LOG_TEST_OUTPUT),
            report_json_str=self._maybe_read_text(log_dir / LOG_REPORT),
            run_instance_log=self._maybe_read_text(log_dir / LOG_INSTANCE),
            patch_diff=self._maybe_read_text(log_dir / PATCH_DIFF),
            log_dir=str(log_dir),
            errored=errored,
            exception=exception_text,
            operation_id=os.environ["YT_OPERATION_ID"],
            job_id=os.environ["YT_JOB_ID"],
        )
        yield test_output.model_dump(mode="json")

    @staticmethod
    def _maybe_read_text(path: Path) -> str | None:
        if path.exists():
            return path.read_text()
        return None


def run_instances_tracto(
    predictions: dict[str, dict],
    instances: list[SWEbenchInstance],
    full_dataset: list[SWEbenchInstance],
    run_id: str,
    timeout: int,
    namespace: str | None,
    tracto_run_dir: str | None = None,
    instance_image_tag: str = "latest",
):
    """
    Run all instances for the given predictions on Tracto.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        run_id (str): Run ID
        timeout (int): Timeout for running tests
        namespace (str | None): Prefix for instance images, i.e registry URL + subpath.
        tracto_run_dir: (str | None): Directory on Tracto where to store eval inputs/outputs.
    """
    logger.info("Creating test specs...")
    test_specs = [
        make_test_spec(instance, namespace, instance_image_tag=instance_image_tag)
        for instance in instances
    ]

    run_test_specs: list[TestSpec] = []

    # Check for instances that have already been run
    for test_spec in test_specs:
        log_dir = get_log_dir(
            predictions[test_spec.instance_id], run_id, test_spec.instance_id
        )
        if log_dir.exists():
            continue
        run_test_specs.append(test_spec)

    if run_test_specs:
        if tracto_run_dir is None:
            input_table_path = yt.create_temp_table(prefix="input")
            output_table_path = yt.create_temp_table(prefix="output")
            stderr_table_path = yt.create_temp_table(prefix="stderr")
        else:
            if yt.exists(tracto_run_dir):
                raise RuntimeError(f"{tracto_run_dir=} already exists on Tracto")
            yt.create("map_node", tracto_run_dir, recursive=True)

            input_table_path = f"{tracto_run_dir}/input"
            output_table_path = f"{tracto_run_dir}/output"
            stderr_table_path = f"{tracto_run_dir}/stderr"

        source_table_rows = [
            TestInput(
                instance_id=test_spec.instance_id,
                test_spec=test_spec,
                prediction=predictions[test_spec.instance_id],
                run_id=run_id,
                timeout=timeout,
                instance_image_tag=instance_image_tag,
            )
            for test_spec in run_test_specs
        ]
        logger.info(f"Writing input table to Tracto at {input_table_path}...")
        yt.write_table(
            input_table_path,
            [row.model_dump(mode="json") for row in source_table_rows],
        )

        logger.info("Running map job on Tracto...")
        yt.run_map(
            RunInstanceTracto(),
            input_table_path,
            output_table_path,
            stderr_table=stderr_table_path,
            table_writer={"max_row_weight": 128 * 1024 * 1024},  # 128 MB
            spec={
                "mapper": {
                    "docker_image": os.environ[TRACTO_EVAL_IMAGE_ENV],
                    "tmpfs_size": TRACTO_EVAL_TMPFS_SIZE_GB * 1024**3,
                    # now resources are autoscaled based on CPU requests
                    "cpu_limit": max(TRACTO_EVAL_TMPFS_SIZE_GB / 4, 1),
                    "environment": {
                        "TRACTO_REGISTRY_URL": get_tracto_registry_url(),
                    },
                },
                "job_count": len(run_test_specs),  # 1 job per instance
                "resource_limits": {
                    "user_slots": TRACTO_EVAL_MAX_PARALLEL_JOBS,
                },
                "max_failed_job_count": 1,
            },
        )

        logger.info(f"Collecting job outputs at {output_table_path}...")
        for result_raw in yt.read_table(output_table_path):
            result = TestOutput.model_validate(result_raw)

            if result.errored:
                logger.warning(
                    f"Instance {result.instance_id} errored, "
                    f"see {result.log_dir} for details. "
                    f"Exception:\n{result.exception}"
                )

            log_dir = Path(result.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            for text, subpath in [
                (result.test_output, LOG_TEST_OUTPUT),
                (result.report_json_str, LOG_REPORT),
                (result.run_instance_log, LOG_INSTANCE),
                (result.patch_diff, PATCH_DIFF),
            ]:
                path = log_dir / subpath

                if text is not None:
                    path.write_text(text)

    logger.info("Generating run report...")
    make_run_report(predictions, full_dataset, run_id)
