import logging
import os
import urllib.parse
from pathlib import Path

import yt.wrapper as yt
from yt import yson

logger = logging.getLogger(__name__)


TRACTO_PODMAN_WORKDIR = Path("/slot/sandbox/tmpfs/podman")


def get_tracto_registry_url() -> str:
    tracto_url = urllib.parse.urlparse(yt.config["proxy"]["url"]).netloc
    return f"cr.{tracto_url}"


def logging_basic_config(level: int = logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def configure_podman_storage():
    storage_conf = f'''
    [storage]
    driver = "vfs"
    runroot = "{TRACTO_PODMAN_WORKDIR}/runroot"
    graphroot = "{TRACTO_PODMAN_WORKDIR}/root"
    '''

    Path("/etc/containers/storage.conf").write_text(storage_conf)

    logger.info(f"podman will store data in {TRACTO_PODMAN_WORKDIR} ")


def get_tracto_registry_creds_from_env() -> dict[str, str]:
    return yson.loads(os.environ["YT_SECURE_VAULT_docker_auth"].encode("utf-8"))
