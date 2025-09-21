# This file contains logic for running evaluations on TractoAI: <https://tracto.ai/>.

import dataclasses
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Annotated, Iterable, cast

import docker
import yt.type_info as ti
import yt.wrapper as yt
from yt import yson

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

TRACTO_EVAL_HOME = os.environ["TRACTO_EVAL_HOME"]
TRACTO_EVAL_IMAGE = os.getenv(
    "TRACTO_EVAL_IMAGE",
    "cr.turing.yt.nebius.yt/home/llm/sbkarasik/registry/swebench-fork:2025-09-14",
)
TRACTO_EVAL_MAX_PARALLEL_JOBS = int(os.getenv("TRACTO_EVAL_MAX_PARALLEL_JOBS", "100"))
TRACTO_EVAL_TMPFS_SIZE_GB = int(os.getenv("TRACTO_EVAL_TMPFS_SIZE_GB", "32"))
TRACTO_PODMAN_WORKDIR = Path("/slot/sandbox/tmpfs/podman")

yt.config["pickling"]["ignore_system_modules"] = True
yt.config["pickling"]["dynamic_libraries"]["enable_auto_collection"] = False

logger = logging.getLogger(__name__)


@yt.yt_dataclass
class TestInput:
    test_spec: (
        Annotated[
            bytes,
            yt.schema.types.Annotation(
                ti_type=ti.Yson,
                to_yt_type=lambda x: yson.dumps(dataclasses.asdict(x)),
                from_yt_type=lambda x: TestSpec(**yson.loads(x)),
            ),
        ]
        | None
    )
    prediction: (
        Annotated[
            bytes,
            yt.schema.types.Annotation(
                ti_type=ti.Yson,
                to_yt_type=lambda x: yson.dumps(x),
                from_yt_type=lambda x: yson.loads(x),
            ),
        ]
        | None
    )
    run_id: str
    timeout: int


@yt.yt_dataclass
class TestOutput:
    instance_id: str
    test_output: str | None
    report_json_str: str | None
    run_instance_log: str
    patch_diff: str | None
    log_dir: str
    errored: bool

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

        podman_root = TRACTO_PODMAN_WORKDIR / "root"
        podman_root.mkdir(parents=True, exist_ok=True)
        podman_runroot = TRACTO_PODMAN_WORKDIR / "runroot"
        podman_runroot.mkdir(parents=True, exist_ok=True)

        # TODO: login to tracto registry
        # available env vars in a job:
        # YT_SECURE_VAULT_docker_auth={username="XXX"; password="XXX"}
        # YT_SECURE_VAULT_YT_TOKEN

        # TODO: configure tracto registry and docker.io as well-known registries in
        # podman, with tracto registry having priority.

        self.proc = subprocess.Popen(
            [
                "podman",
                "--storage-driver=vfs",
                f"--root={podman_root}",
                f"--runroot={podman_runroot}",
                "system",
                "service",
                "--time=0",
                self.socket_url,
            ],
        )

        self._wait_for_podman()

        return self

    def _wait_for_podman(self):
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

        self.proc.wait()


class RunInstanceTracto(yt.TypedJob):
    def __call__(self, test_input: TestInput) -> Iterable[TestOutput]:
        logging.basicConfig(level=logging.INFO)

        test_spec = cast(TestSpec, test_input.test_spec)
        prediction = cast(dict, test_input.prediction)

        log_dir = get_log_dir(prediction, test_input.run_id, test_spec.instance_id)

        with PodmanDaemon() as podman_daemon:
            docker_client = docker.DockerClient(base_url=podman_daemon.socket_url)

            logger.info("Running run_instance...")

            # TODO: manually pull the image first with proper retries

            try:
                run_instance(
                    test_spec=test_spec,
                    pred=prediction,
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
            except Exception:
                errored = True
            else:
                errored = False

            logger.info("Finished run_instance")
        logger.info("PodmanDaemon context exited")

        yield TestOutput(
            instance_id=test_spec.instance_id,
            test_output=self._maybe_read_text(log_dir / LOG_TEST_OUTPUT),
            report_json_str=self._maybe_read_text(log_dir / LOG_REPORT),
            run_instance_log=self._maybe_read_text(log_dir / LOG_INSTANCE),
            patch_diff=self._maybe_read_text(log_dir / PATCH_DIFF),
            log_dir=str(log_dir),
            errored=errored,
            operation_id=os.environ["YT_OPERATION_ID"],
            job_id=os.environ["YT_JOB_ID"],
        )

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
):
    """
    Run all instances for the given predictions on Tracto.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        run_id (str): Run ID
        timeout (int): Timeout for running tests
        namespace (str | None):
    """
    test_specs = [make_test_spec(instance, namespace) for instance in instances]

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
        run_dir = f"{TRACTO_EVAL_HOME}/{run_id}"
        input_table_path = f"{run_dir}/input"
        output_table_path = f"{run_dir}/output"
        stderr_table_path = f"{run_dir}/stderr"

        logger.info(f"Tracto run_dir={run_dir}")

        if yt.exists(run_dir):
            raise RuntimeError(f"Tracto run_dir={run_dir} already exists on Tracto")
        yt.create("map_node", run_dir, recursive=True)

        source_table_rows = [
            TestInput(
                test_spec=test_spec,
                prediction=predictions[test_spec.instance_id],
                run_id=run_id,
                timeout=timeout,
            )
            for test_spec in run_test_specs
        ]
        yt.write_table_structured(
            table=input_table_path,
            row_type=TestInput,
            input_stream=source_table_rows,
        )

        yt.run_map(
            RunInstanceTracto(),
            input_table_path,
            output_table_path,
            stderr_table=stderr_table_path,
            table_writer={"max_row_weight": 128 * 1024 * 1024},  # 128 MB
            spec={
                "mapper": {
                    "docker_image": TRACTO_EVAL_IMAGE,
                    "tmpfs_size": TRACTO_EVAL_TMPFS_SIZE_GB * 1024**3,
                },
                "job_count": len(run_test_specs),  # 1 job per instance
                "resource_limits": {
                    "user_slots": TRACTO_EVAL_MAX_PARALLEL_JOBS,
                },
                "max_failed_job_count": 1,
            },
        )

        for result in yt.read_table_structured(output_table_path, TestOutput):
            result = cast(TestOutput, result)

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

    make_run_report(predictions, full_dataset, run_id)
