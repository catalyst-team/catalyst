import logging

import hydra
from omegaconf import DictConfig

from catalyst.dl import utils

logger = logging.getLogger(__name__)


def main_worker(cfg: DictConfig):
    utils.set_global_seed(cfg.args.seed)
    utils.prepare_cudnn(cfg.args.deterministic, cfg.args.benchmark)

    utils.import_module(hydra.utils.to_absolute_path(cfg.args.expdir))

    experiment = hydra.utils.instantiate(cfg.experiment, cfg=cfg)
    runner = hydra.utils.instantiate(cfg.runner)

    if experiment.logdir is not None and utils.get_rank() <= 0:
        utils.hydra_dump_environment(cfg, experiment.logdir)
        utils.dump_code(
            hydra.utils.to_absolute_path(cfg.args.expdir), experiment.logdir
        )

    runner.run_experiment(experiment)


@hydra.main()
def main(cfg: DictConfig):
    """
    Hydra config catalyst-dl run entry point

    Args:
        cfg: (DictConfig) configuration

    """
    cfg = utils.hydra_prepare_config(cfg)
    utils.hydra_distributed_cmd_run(main_worker, cfg.args.distributed, cfg)


__all__ = ["main"]
