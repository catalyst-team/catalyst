# flake8: noqa
from torch.optim.lr_scheduler import *

from catalyst.contrib.nn.schedulers.base import BaseScheduler, BatchScheduler
from catalyst.contrib.nn.schedulers.onecycle import OneCycleLRWithWarmup
