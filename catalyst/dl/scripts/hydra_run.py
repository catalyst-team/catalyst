import logging

import hydra
from omegaconf import DictConfig

from catalyst.utils.distributed import get_rank
from catalyst.utils.hydra_config import prepare_hydra_config
from catalyst.utils.misc import set_global_seed
from catalyst.utils.scripts import distributed_cmd_run, dump_code, import_module
from catalyst.utils.sys import dump_environment
from catalyst.utils.torch import prepare_cudnn

logger = logging.getLogger(__name__)


def main_worker(cfg: DictConfig):
    set_global_seed(cfg.args.seed)
    prepare_cudnn(cfg.args.deterministic, cfg.args.benchmark)

    import_module(hydra.utils.to_absolute_path(cfg.args.expdir))

    experiment = hydra.utils.instantiate(cfg.experiment, cfg=cfg)
    runner = hydra.utils.instantiate(cfg.runner)

    if experiment.logdir is not None and get_rank() <= 0:
        dump_environment(cfg, experiment.logdir)
        dump_code(hydra.utils.to_absolute_path(cfg.args.expdir), experiment.logdir)

    runner.run(experiment)


@hydra.main()
def main(cfg: DictConfig):
    """
    Hydra config catalyst-dl run entry point

    Args:
        cfg: (DictConfig) configuration

    """
    cfg = prepare_hydra_config(cfg)
    distributed_cmd_run(main_worker, cfg.args.distributed, cfg)


__all__ = ["main"]
