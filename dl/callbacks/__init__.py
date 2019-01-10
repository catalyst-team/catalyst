# flake8: noqa
from . import core, metrics, schedulers

from .core import *  # yapf: disable
from .metrics import *  # yapf: disable
from .schedulers import *  # yapf: disable

CALLBACKS = {
    **core.__dict__,
    **metrics.__dict__,
    **schedulers.__dict__,
}
