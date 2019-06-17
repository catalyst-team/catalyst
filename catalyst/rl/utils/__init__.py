# flake8: noqa

from .criterion import categorical_loss, quantile_loss

from .buffers import OffpolicyReplayBuffer, OnpolicyRolloutBuffer
from .gamma import hyperbolic_gammas
from .samplers import OffpolicyReplaySampler, OnpolicyRolloutSampler
from .torch import get_trainer_components, \
    get_network_weights, set_network_weights

from catalyst.utils import *
