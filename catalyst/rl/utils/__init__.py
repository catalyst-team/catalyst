# flake8: noqa

from .criterion import categorical_loss, quantile_loss

from .buffer import OffpolicyReplayBuffer, OnpolicyRolloutBuffer
from .gamma import hyperbolic_gammas
from .gym import extend_space
from .sampler import OffpolicyReplaySampler, OnpolicyRolloutSampler
from .torch import get_trainer_components, \
    get_network_weights, set_network_weights
from .trajectory import structed2dict_trajectory, dict2structed_trajectory

from catalyst.utils import *
