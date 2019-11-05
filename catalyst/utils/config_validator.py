from schema import Schema, And, Optional, Forbidden, SchemaError
from catalyst.dl.registry import (
    CALLBACKS, CRITERIONS, MODELS, OPTIMIZERS, SCHEDULERS
)
from catalyst.utils.registry import RegistryException

TOP_LEVEL_PARAMS = {
    'model_params': object,
    'args': object,
    'stages': object
}

MODEL_PARAMS = {
    'model': And(str, len)
}
ARGS_PARAMS = {
    'expdir': And(str, len),
    Forbidden('logdir'): '.',
    'logdir': And(str, len),
    'configs': object,
    Optional(object): object
}
STAGES_PARAMS = {
    'stage1': object,
    'data_params': {
        'num_workers': int,
        Optional(object): object
    },
    Optional(object): object
}

DL_CONFIG = {
    'model_params': MODEL_PARAMS,
    'args': ARGS_PARAMS,
    'stages': STAGES_PARAMS
}


def validate_dl_config(config):
    return Schema(DL_CONFIG).validate(config)


class ConfigError(Exception):
    pass


class ConfigValidator():
    def __init__(self, config: dict):
        self.config = config

    def validate(self):
        self._validate_top_level()
        self._validate_model()
        self._validate_args()
        self._validate_stages()

    def _validate_top_level(self):
        try:
            Schema(TOP_LEVEL_PARAMS).validate(self.config)
        except SchemaError as e:
            raise ConfigError(
                f'You missed one of required sections in your config file:'
                f'\n\n{e}')

    def _validate_model(self):
        try:
            params = self.config['model_params']
            Schema(MODEL_PARAMS).validate(params)
            # From where I run this code now - we don't load all from expdir,
            # so it will always throw error
            # MODELS.get(params['model'])
        except SchemaError as e:
            raise ConfigError(
                f'Your `model_params` section is not valid:\n\n{e}')
        except RegistryException as e:
            raise ConfigError(
                f'You misspelled your `model` name'
                f' in `model_params` section:\n\n{e}')

    def _validate_args(self):
        try:
            Schema(ARGS_PARAMS).validate(self.config['args'])
        except SchemaError as e:
            raise ConfigError(
                f'Your `args` section is not valid:\n\n{e}')

    def _validate_stages(self):
        try:
            Schema(STAGES_PARAMS).validate(self.config['stages'])
        except SchemaError as e:
            raise ConfigError(
                f'Your `stages` section is not valid:\n\n{e}')
