# flake8: noqa
from catalyst.dl.callbacks import \
    core as core_callbacks, \
    metrics as metrics_callbacks, \
    schedulers as schedulers_callbacks

from catalyst.dl.callbacks.core import *
from catalyst.dl.callbacks.metrics import *
from catalyst.dl.callbacks.schedulers import *

CALLBACKS = {
    **core_callbacks.__dict__,
    **metrics_callbacks.__dict__,
    **schedulers_callbacks.__dict__,
}
