# This file contains logic for running evaluations on TractoAI: <https://tracto.ai/>.

import dataclasses
import datetime
import json
import subprocess
import time
from pathlib import Path
from typing import Annotated, Iterable, cast

import docker
import yt.type_info as ti
import yt.wrapper as yt
from yt import yson

from swebench.harness.constants import SWEbenchInstance
from swebench.harness.eval import get_log_dir, run_instance
from swebench.harness.reporting import make_run_report
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec
from swebench.harness.constants import (
    LOG_REPORT,
    LOG_INSTANCE,
    LOG_TEST_OUTPUT,
    PATCH_DIFF,
)

TRACTO_PODMAN_WORKDIR = Path("/slot/sandbox/tmpfs/podman")
TMPFS_SIZE_GB = 8
yt.config["pickling"]["ignore_system_modules"] = True
yt.config["pickling"]["dynamic_libraries"]["enable_auto_collection"] = False


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
    test_output: str
    report_json_str: str
    run_instance_log: str
    patch_diff: str
    log_dir: str
    errored: bool


class PodmanDaemon:
    def __init__(self):
        self.proc = None
        self.socket = None

    def __enter__(self):
        podman_socket_path = "unix:///run/podman/podman.sock"
        Path("/run/podman").mkdir(parents=True, exist_ok=True)

        podman_root = TRACTO_PODMAN_WORKDIR / "root"
        podman_root.mkdir(parents=True, exist_ok=True)
        podman_runroot = TRACTO_PODMAN_WORKDIR / "runroot"
        podman_runroot.mkdir(parents=True, exist_ok=True)

        self.proc = subprocess.Popen(
            [
                "podman",
                "--storage-driver=vfs",
                f"--root={podman_root}",
                f"--runroot={podman_runroot}",
                "system",
                "service",
                "--time=0",
                podman_socket_path,
            ],
        )

        time.sleep(1)  # Give Podman time to start
        if (exitcode := self.proc.poll()) is not None:
            raise ValueError(f"Podman service failed to start, exitcode={exitcode}")

        self.socket = podman_socket_path

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.proc.terminate()
        self.proc.wait()


class RunInstanceTracto(yt.TypedJob):
    def __call__(self, test_input: TestInput) -> Iterable[TestOutput]:
        import logging

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        test_spec = cast(TestSpec, test_input.test_spec)
        prediction = cast(dict, test_input.prediction)

        log_dir = get_log_dir(prediction, test_input.run_id, test_spec.instance_id)

        with PodmanDaemon() as podman_daemon:
            docker_client = docker.DockerClient(base_url=podman_daemon.socket)

            logger.info("Running run_instance...")
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
            logger.info("Finished run_instance")
        logger.info("PodmanDaemon context exited")

        # time.sleep(3600 * 100)

        yield TestOutput(
            instance_id=test_spec.instance_id,
            test_output=self._maybe_read_text(log_dir / LOG_TEST_OUTPUT),
            report_json_str=self._maybe_read_text(log_dir / LOG_REPORT),
            run_instance_log=self._maybe_read_text(log_dir / LOG_INSTANCE),
            patch_diff=self._maybe_read_text(log_dir / PATCH_DIFF),
            log_dir=str(log_dir),
            errored=False,
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
        run_dir = (
            "//home/llm/sbkarasik/tracto-swe-bench/"
            f"{run_id}-{datetime.datetime.now().isoformat()}"
        )
        print(f"{run_dir=}")

        yt.create("map_node", run_dir, recursive=True)

        input_table_path = f"{run_dir}/input"
        output_table_path = f"{run_dir}/output"
        print(f"{input_table_path=}")

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
            spec={
                "mapper": {
                    "docker_image": "cr.turing.yt.nebius.yt/home/llm/sbkarasik/registry/swebench-fork:2025-09-14",
                    "tmpfs_size": TMPFS_SIZE_GB * 1024**3,
                },
                "max_failed_job_count": 1,
            },
        )

        for result in yt.read_table_structured(output_table_path, TestOutput):
            result = cast(TestOutput, result)

            # Save logs locally
            log_dir = Path(result.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / LOG_INSTANCE, "w") as f:
                f.write(result.run_instance_log)
            with open(log_dir / LOG_TEST_OUTPUT, "w") as f:
                f.write(result.test_output)
            with open(log_dir / PATCH_DIFF, "w") as f:
                f.write(result.patch_diff)
            with open(log_dir / LOG_REPORT, "w") as f:
                try:
                    report_json = json.loads(result.report_json_str)
                    json.dump(report_json, f, indent=4)
                except Exception:
                    # This happens if the test fails with any exception
                    print(f"{result.instance_id}: no report.json")

    make_run_report(predictions, full_dataset, run_id)
