from collections import OrderedDict, Mapping
from pathlib import Path
import safitty

from catalyst.dl import utils
from catalyst.utils.config import parse_args_uargs


class ConfigError(Exception):
    pass


def _err(msg, prefix="Config error", postfix=""):
    err_msg = f"{prefix}:\n    {msg}"
    if postfix:
        err_msg += f"\n{postfix}"
    raise ConfigError(err_msg)


def config_stage_validator(stage_name, stage):
    pre_err = "Stage `{stage_name}` config invalid"

    # Common validations
    if not isinstance(stage, Mapping):
        _err(pre_err, "Config itself is not a dict, but {type(stage)}")
    if "data_params" not in stage:
        _err(pre_err, "You need to configure `data_params` and set "
                      "`num_workers` there")

    # Infer stage validations
    if stage["_mode"] == "infer":
        pass
    # Other type stage validations
    else:
        pass


class Config():
    STAGES = "stages"

    STAGE_KEYWORDS = [
        "criterion_params",
        "optimizer_params",
        "scheduler_params",
        "data_params",
        "state_params",
        "callbacks_params",
    ]

    DEFAULT_SEED = 42

    def __init__(self, args, uargs):
        self.args, self._config = parse_args_uargs(args, uargs)
        self.__stages = self.__compile_stages()

    def __compile_stages(self):
        defaults = {}
        result = OrderedDict()
        stages = safitty.get(self._config, "stages", default={}, copy=True)
        for key in self.STAGE_KEYWORDS:
            defaults[key] = safitty.get(stages, key, default={}, copy=True)
        for stage in stages:
            if stage in self.STAGE_KEYWORDS or stages.get(stage) is None:
                continue
            result[stage] = {}
            for key in self.STAGE_KEYWORDS:
                result[stage][key] = utils.merge_dicts(
                    safitty.get(defaults, key, default={}, copy=True),
                    safitty.get(stages, stage, key, default={}, copy=True)
                )

        if not result:
            raise ConfigError("You didn't setup any stage to run")

        return result

    @property
    def seed(self) -> int:
        return safitty.get(
            self._config, "args", "seed", default=self.DEFAULT_SEED)

    @property
    def verbose(self) -> bool:
        return safitty.get(
            self._config, "args", "verbose", default=False)

    @property
    def distributed_params(self):
        return safitty.get(
            self._config, "distributed_params", default={})

    @property
    def monitoring_params(self):
        return safitty.get(
            self._config, "monitoring_params", default={})

    def __generate_logdir(self):
        timestamp = utils.get_utcnow_time()
        config_hash = utils.get_short_hash(self._config)
        logdir = f"{timestamp}.{config_hash}"
        distributed_rank = self.distributed_params.get("rank", -1)
        if distributed_rank > -1:
            logdir = f"{logdir}.rank{distributed_rank:02d}"
        return logdir

    @property
    def logdir(self):
        logdir = safitty.get(
            self._config, "args", "logdir", default=None)
        if str(logdir).lower() != "none":
            return logdir

        baselogdir = safitty.get(
            self._config, "args", "baselogdir", default=None)
        if str(baselogdir).lower() != "none":
            return f"{baselogdir}/{self.__generate_logdir()}"
        else:
            # Maybe better to provide some default?
            # Like f"logs/{self.__generate_logidr()}"
            raise ConfigError("`logdir` not set")

    @property
    def args_params(self):
        return safitty.get(
            self._config, "args", default={}, copy=True)

    @property
    def common_stages_params(self):
        return safitty.get(
            self._config, "stages", default={}, copy=True)

    @property
    def stages(self):
        return list(self.__stages.keys())

    @property
    def state_params(self):
        return utils.merge_dicts(
            safitty.get(self.common_stages_params, "state_params", default={}),
            self.args_params, {"logdir": self.logdir}
        )

    def stage_params(self, stage):
        return self.__stages[stage]

    def stage_state_params(self, stage):
        return safitty.get(
            self.stage_params(stage), "state_params", default={})
