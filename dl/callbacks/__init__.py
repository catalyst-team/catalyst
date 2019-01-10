from . import core, metrics, schedulers

# yapf: disable
from .core import *
from .metrics import *
from .schedulers import *
# yapf: enable

CALLBACKS = {
    **core.__dict__,
    **metrics.__dict__,
    **schedulers.__dict__,
}
