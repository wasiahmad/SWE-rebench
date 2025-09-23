"""
Script to copy instance images to Tracto registry for better evaluation performance.

Example: import swe-rebench-leaderboard images
```bash
python -m swebench.harness.tracto_eval.copy_images_to_tracto \
    --dataset_name nebius/SWE-rebench-leaderboard  \
    --namespace swerebench \
    --tracto-namespace <<tracto registry url>/<<your-subpath>>/swerebench
```

Example: import swe-bench-verified images
```bash
python -m swebench.harness.tracto_eval.copy_images_to_tracto \
    --dataset_name SWE-bench/SWE-bench_Verified  \
    --namespace swebench \
    --tracto-namespace <<tracto registry url>/<<your-subpath>>/swebench
```
"""

import argparse
import logging

import yt.wrapper as yt

from swebench.harness.tracto_eval.utils import (
    logging_basic_config,
)
from swebench.harness.utils import (
    load_swebench_dataset,
)
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.tracto_eval import copy_images_to_tracto_script

logger = logging.getLogger(__name__)


def main(
    dataset_name: str,
    split: str,
    instance_ids: list[str],
    instance_image_tag: str,
    tracto_instance_image_tag: str,
    namespace: str,
    tracto_namespace: str,
    max_workers: int,
    tracto_workdir: str | None = None,
):
    logger.info(f"Loading dataset={dataset_name}, split={split}")
    instances = load_swebench_dataset(dataset_name, split, instance_ids)

    input_rows = []

    logger.info("Crafting data for input table")
    for instance in instances:
        test_spec = make_test_spec(
            instance, namespace, instance_image_tag=instance_image_tag
        )
        tracto_test_spec = make_test_spec(
            instance, tracto_namespace, instance_image_tag=tracto_instance_image_tag
        )

        input_rows.append(
            {
                "instance_id": test_spec.instance_id,
                "instance_image_key": test_spec.instance_image_key,
                "tracto_instance_image_key": tracto_test_spec.instance_image_key,
            }
        )

    if tracto_workdir is None:
        input_table = yt.create_temp_table(prefix="input")
        output_table = yt.create_temp_table(prefix="output")
    else:
        yt.create("map_node", tracto_workdir, recursive=True, ignore_existing=True)
        input_table = f"{tracto_workdir}/input"
        output_table = f"{tracto_workdir}/output"

    logger.info(f"input_table={input_table}")
    logger.info(f"output_table={output_table}")
    logger.info("Writing input table")
    yt.write_table(input_table, input_rows)

    logger.info("Running map operation to copy images")
    yt.run_map(
        "python3 copy_images_to_tracto_script.py",
        input_table,
        output_table,
        format="json",
        spec={
            "mapper": {
                "docker_image": "quay.io/skopeo/stable:latest",
                "tmpfs_size": 32 * 1024**3,
                "environment": {
                    "TRACTO_REGISTRY_URL": tracto_namespace,
                },
            },
            "job_count": max_workers,
            "secure_vault": {
                "TRACTO_REGISTRY_USERNAME": yt.get_user_name(),
                "TRACTO_REGISTRY_PASSWORD": yt.config["token"],
            },
        },
        local_files=[yt.LocalFile(copy_images_to_tracto_script.__file__)],
    )

    failed_count = 0
    for row in yt.read_table(output_table):
        if not row["success"]:
            failed_count += 1
            logger.error(
                f"Failed to copy image for instance_id={row['instance_id']}, "
                f"source={row['instance_image_key']}, "
                f"target={row['tracto_instance_image_key']}"
            )

    if failed_count == 0:
        logger.info("Done, all images copied successfully")
    else:
        logger.info(f"Done, {failed_count}/{len(input_rows)} images failed to copy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy images to Tracto for better eval performance",
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
    )
