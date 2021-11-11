# flake8: noqa
from torch.optim.lr_scheduler import *

from catalyst.contrib.schedulers.base import BaseScheduler, BatchScheduler
from catalyst.contrib.schedulers.onecycle import OneCycleLRWithWarmup
