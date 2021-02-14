import logging

import hydra
from omegaconf import DictConfig

from catalyst.utils.distributed import get_rank
from catalyst.utils.hydra_config import prepare_hydra_config
from catalyst.utils.misc import set_global_seed
from catalyst.utils.scripts import dump_code, import_module
from catalyst.utils.sys import dump_environment
from catalyst.utils.torch import prepare_cudnn

logger = logging.getLogger(__name__)


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
