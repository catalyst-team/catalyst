import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from catalyst.utils.distributed import get_rank
from catalyst.utils.misc import set_global_seed
from catalyst.utils.sys import dump_code, dump_environment, import_module
from catalyst.utils.torch import prepare_cudnn

logger = logging.getLogger(__name__)


def prepare_hydra_config(cfg: DictConfig) -> DictConfig:
    """
    Prepare config. Add required parameters.

    Args:
        cfg: (DictConfig) config

    Returns:
        DictConfig: config

    """
    OmegaConf.set_readonly(cfg, False)
    OmegaConf.set_struct(cfg, False)

    # cfg.setdefault("vals", DictConfig({}))

    cfg.setdefault("args", DictConfig({}))
    cfg.args.setdefault("expdir", ".")
    cfg.args.setdefault("resume", None)
    cfg.args.setdefault("autoresume", None)
    cfg.args.setdefault("seed", 42)
    cfg.args.setdefault("distributed", os.getenv("USE_DDP", "0") == "1")
    cfg.args.setdefault("apex", os.getenv("USE_APEX", "0") == "1")
    cfg.args.setdefault("amp", os.getenv("USE_AMP", "0") == "1")
    cfg.args.setdefault("verbose", False)
    cfg.args.setdefault("timeit", False)
    cfg.args.setdefault("check", False)
    cfg.args.setdefault("overfit", False)
    cfg.args.setdefault("deterministic", False)
    cfg.args.setdefault("benchmark", False)

    # cfg.setdefault("distributed", DictConfig({}))
    # cfg.distributed.setdefault("apex", cfg.args.apex)
    # cfg.distributed.setdefault("amp", cfg.args.amp)

    cfg.setdefault("runner", DictConfig({}))

    cfg.setdefault("model", DictConfig({}))

    cfg.setdefault("stages", DictConfig({}))

    return cfg


@hydra.main()
def main(cfg: DictConfig):
    """
    Hydra config catalyst-dl run entry point

    Args:
        cfg: (DictConfig) configuration

    """
    cfg = prepare_hydra_config(cfg)
    set_global_seed(cfg.args.seed)
    prepare_cudnn(cfg.args.deterministic, cfg.args.benchmark)

    import_module(hydra.utils.to_absolute_path(cfg.args.expdir))
    runner = hydra.utils.instantiate(cfg.runner, cfg=cfg)

    if get_rank() <= 0:
        dump_environment(logdir=runner.logdir, config=cfg)
        dump_code(expdir=hydra.utils.to_absolute_path(cfg.args.expdir), logdir=runner.logdir)

    runner.run()


__all__ = ["main"]
