from . import actor
from . import critic

AGENTS = {**actor.__dict__, **critic.__dict__}
