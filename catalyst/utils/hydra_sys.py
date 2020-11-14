import json
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from catalyst.contrib.tools.tensorboard import SummaryWriter
from catalyst.utils.config import save_config
from catalyst.utils.sys import (
    get_environment_vars,
    list_conda_packages,
    list_pip_packages,
)

logger = logging.getLogger(__name__)


def dump_environment(cfg: DictConfig, logdir: str) -> None:
    """
    Saves config, environment variables and package list in JSON into logdir.

    Args:
        cfg: (DictConfig) config
        logdir: (str) path to logdir

    """
    config_dir = Path(logdir) / "configs"
    config_dir.mkdir(exist_ok=True, parents=True)

    environment = get_environment_vars()

    save_config(
        OmegaConf.to_container(cfg, resolve=True), config_dir / "_config.json"
    )
    save_config(environment, config_dir / "_environment.json")

    pip_pkg = list_pip_packages()
    (config_dir / "pip-packages.txt").write_text(pip_pkg)
    conda_pkg = list_conda_packages()
    if conda_pkg:
        (config_dir / "conda-packages.txt").write_text(conda_pkg)

    with open(config_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    config_str = json.dumps(
        OmegaConf.to_container(cfg, resolve=True), indent=2, ensure_ascii=False
    )
    config_str = config_str.replace("\n", "\n\n")

    environment_str = json.dumps(environment, indent=2, ensure_ascii=False)
    environment_str = environment_str.replace("\n", "\n\n")

    pip_pkg = pip_pkg.replace("\n", "\n\n")
    conda_pkg = conda_pkg.replace("\n", "\n\n")
    with SummaryWriter(config_dir) as writer:
        writer.add_text("_config", config_str, 0)
        writer.add_text("_environment", environment_str, 0)
        writer.add_text("pip-packages", pip_pkg, 0)
        if conda_pkg:
            writer.add_text("conda-packages", conda_pkg, 0)


__all__ = ["dump_environment"]
