from schema import Schema, And, Optional, Forbidden

dl_config_schema = Schema({
    'model_params': {
        'model': And(str, len)
    },
    'args': {
        'expdir': And(str, len),
        Forbidden('logdir'): '.',
        'logdir': And(str, len),
        'configs': object,
        Optional(object): object
    },
    'stages': {
        'stage1': object,
        Optional(object): object
    }
})


def validate_dl_config(config):
    return dl_config_schema.validate(config)
