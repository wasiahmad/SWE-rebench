"""
Script to copy instance images to Tracto registry for better evaluation performance.

Example: import swe-rebench-leaderboard images
```bash
python -m swebench.harness.tracto_eval.import_images_to_tracto \
    --dataset_name nebius/SWE-rebench-leaderboard  \
    --namespace docker.io/swerebench \
    --tracto-namespace <<tracto registry url>/<<your-subpath>>/swerebench
```

Example: import SWE-bench_Verified images
```bash
python -m swebench.harness.tracto_eval.import_images_to_tracto \
    --dataset_name SWE-bench/SWE-bench_Verified  \
    --namespace docker.io/swebench \
    --tracto-namespace <<tracto registry url>/<<your-subpath>>/swebench
```
"""

import argparse
import logging
import os
import subprocess
from typing import Iterable

import pydantic
import yt.wrapper as yt

from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.tracto_eval.utils import (
    configure_podman_storage,
    get_tracto_registry_creds_from_env,
    logging_basic_config,
)
from swebench.harness.tracto_eval.run_evaluation_tracto import (
    TRACTO_EVAL_IMAGE_ENV,
    validate_tracto_env_vars,
)
from swebench.harness.utils import load_swebench_dataset

logger = logging.getLogger(__name__)

# As of 2025-09-25, overlayfs can't be used when we run podman/buildah inside jobs,
# so we have to use VFS storage driver. But this results in
# about x^2 increase in storage usage, because layers are duplicated.
# To overcome this, we:
# 1) squash all image layers into one during import
# 2) have to set very large tmpfs size to handle this x^2 storage usage during import
TRACTO_IMPORT_TMPFS_SIZE_GB = int(os.environ.get("TRACTO_IMPORT_TMPFS_SIZE_GB", 64))


class InputRow(pydantic.BaseModel):
    instance_id: str
    source_image: str
    target_image: str


class OutputRow(pydantic.BaseModel):
    instance_id: str
    source_image: str
    target_image: str
    errored: bool
    exception: str | None


class ImportImageToTracto:
    def __init__(self, squash_layers: bool):
        self.squash_layers = squash_layers

    def __call__(self, row_dict: dict) -> Iterable[dict]:
        configure_podman_storage()

        row = InputRow.model_validate(row_dict)

        try:
            self._import_image(
                instance_id=row.instance_id,
                source_image=row.source_image,
                target_image=row.target_image,
            )
        except subprocess.CalledProcessError as e:
            errored = True
            if isinstance(e, subprocess.CalledProcessError):
                exception_text = f"{str(e)}\nstderr:\n{e.stderr}"
            else:
                exception_text = str(e)
            logger.exception(e)
        else:
            errored = False
            exception_text = None

        yield OutputRow(
            instance_id=row.instance_id,
            source_image=row.source_image,
            target_image=row.target_image,
            errored=errored,
            exception=exception_text,
        ).model_dump(mode="json")

    def _import_image(self, instance_id: str, source_image: str, target_image: str):
        logger.info(
            f"Importing image for instance {instance_id}: "
            f"{source_image} -> {target_image}"
        )

        logger.info("Logging into Tracto registry...")

        tracto_registry_creds = get_tracto_registry_creds_from_env()
        self._run_command(
            [
                "buildah",
                "login",
                "--username",
                tracto_registry_creds["username"],
                "--password-stdin",
                os.environ["TRACTO_REGISTRY_URL"],
            ],
            input=tracto_registry_creds["password"],
        )

        logger.info(f"Pulling image {source_image}...")

        if self.squash_layers:
            logger.info("Squashing layers...")

            self._run_command(
                ["buildah", "from", "--name", "stage", source_image],
            )

            self._run_command(
                ["buildah", "commit", "--squash", "stage", target_image],
            )
        else:
            self._run_command(
                ["buildah", "tag", source_image, target_image],
            )

        logger.info(f"Pushing image {target_image}...")
        self._run_command(["buildah", "push", target_image])
        logger.info("Done")

    @staticmethod
    def _run_command(command: list[str], input: str | None = None) -> str:
        return subprocess.check_output(command, text=True, input=input)


def main(
    dataset_name: str,
    split: str,
    instance_ids: list[str],
    instance_image_tag: str,
    tracto_instance_image_tag: str,
    namespace: str,
    tracto_namespace: str,
    max_workers: int,
    squash_layers: bool,
    tracto_workdir: str | None = None,
):
    logger.info(f"Loading dataset={dataset_name}, split={split}")

    validate_tracto_env_vars()

    instances = load_swebench_dataset(dataset_name, split, instance_ids)

    input_rows: list[InputRow] = []

    logger.info("Crafting data for input table")
    for instance in instances:
        test_spec = make_test_spec(
            instance, namespace, instance_image_tag=instance_image_tag
        )
        tracto_test_spec = make_test_spec(
            instance, tracto_namespace, instance_image_tag=tracto_instance_image_tag
        )

        input_rows.append(
            InputRow(
                instance_id=test_spec.instance_id,
                source_image=test_spec.instance_image_key,
                target_image=tracto_test_spec.instance_image_key,
            )
        )

    if tracto_workdir is None:
        input_table = yt.create_temp_table(prefix="input")
        output_table = yt.create_temp_table(prefix="output")
        stderr_table = yt.create_temp_table(prefix="stderr")
    else:
        yt.create("map_node", tracto_workdir, recursive=True, ignore_existing=True)
        input_table = f"{tracto_workdir}/input"
        output_table = f"{tracto_workdir}/output"
        stderr_table = f"{tracto_workdir}/stderr"

    logger.info(f"input_table={input_table}")
    logger.info(f"output_table={output_table}")
    logger.info(f"stderr_table={stderr_table}")
    logger.info("Writing input table")
    yt.write_table(input_table, [row.model_dump(mode="json") for row in input_rows])

    logger.info("Running map operation to import images")
    yt.run_map(
        ImportImageToTracto(squash_layers=squash_layers),
        input_table,
        output_table,
        stderr_table=stderr_table,
        spec={
            "mapper": {
                "docker_image": os.environ[TRACTO_EVAL_IMAGE_ENV],
                "tmpfs_size": TRACTO_IMPORT_TMPFS_SIZE_GB * 1024**3,
                "environment": {
                    "TRACTO_REGISTRY_URL": tracto_namespace,
                },
                # resources are autoscaled based on CPU requests
                "cpu_limit": max(TRACTO_IMPORT_TMPFS_SIZE_GB / 4, 1),
            },
            "job_count": len(input_rows),
            "resource_limits": {
                "user_slots": max_workers,
            },
            "max_failed_job_count": 1,
        },
    )

    failed_count = 0
    for row_dict in yt.read_table(output_table):
        row = OutputRow.model_validate(row_dict)

        if row.errored:
            failed_count += 1
            logger.error(
                f"Failed to copy image for instance_id={row.instance_id}, "
                f"source={row.source_image}, "
                f"target={row.target_image}. Exception:\n{row.exception}"
            )

    if failed_count == 0:
        logger.info("Done, all images copied successfully")
    else:
        logger.info(f"Done, {failed_count}/{len(input_rows)} images failed to copy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import images to Tracto registry for better eval performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset_name",
        default="SWE-bench/SWE-bench_Lite",
        type=str,
        help="Name of dataset or path to JSON file.",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split of the dataset"
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs images to copy (space separated)",
    )
    parser.add_argument(
        "--instance_image_tag", type=str, default="latest", help="Instance image tag"
    )
    parser.add_argument(
        "--tracto_instance_image_tag", type=str, default=None, help="Instance image tag"
    )
    parser.add_argument(
        "--namespace", type=str, default="swebench", help="Namespace for images"
    )
    parser.add_argument(
        "--tracto-namespace",
        type=str,
        required=True,
        help="Target namespace for images on Tracto",
    )
    parser.add_argument("--max-workers", type=int, default=32)

    args = parser.parse_args()

    logging_basic_config()

    main(
        dataset_name=args.dataset_name,
        split=args.split,
        instance_ids=args.instance_ids,
        instance_image_tag=args.instance_image_tag,
        tracto_instance_image_tag=(
            args.tracto_instance_image_tag or args.instance_image_tag
        ),
        namespace=args.namespace,
        tracto_namespace=args.tracto_namespace,
        max_workers=args.max_workers,
        squash_layers=True,
    )
