# flake8: noqa
from . import core, metrics, schedulers

from .core import *
from .metrics import *
from .schedulers import *

CALLBACKS = {
    **core.__dict__,
    **metrics.__dict__,
    **schedulers.__dict__,
}
