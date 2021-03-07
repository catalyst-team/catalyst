import os

from omegaconf import DictConfig, OmegaConf


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

    cfg.setdefault("vals", DictConfig({}))

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

    cfg.setdefault("distributed", DictConfig({}))
    cfg.distributed.setdefault("apex", cfg.args.apex)
    cfg.distributed.setdefault("amp", cfg.args.amp)

    cfg.setdefault("experiment", DictConfig({}))

    cfg.setdefault("runner", DictConfig({}))

    cfg.setdefault("models", DictConfig({}))

    cfg.setdefault("stages", DictConfig({}))

    return cfg


__all__ = ["prepare_hydra_config"]
