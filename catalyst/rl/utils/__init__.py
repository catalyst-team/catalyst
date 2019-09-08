# flake8: noqa

from .criterion import categorical_loss, quantile_loss

from .agent import get_observation_net, \
    process_state_ff, process_state_ff_kv, \
    process_state_temporal, process_state_temporal_kv
from .buffer import OffpolicyReplayBuffer, OnpolicyRolloutBuffer
from .gamma import hyperbolic_gammas
from .gym import extend_space
from .sampler import OffpolicyReplaySampler, OnpolicyRolloutSampler
from .torch import get_trainer_components, \
    get_network_weights, set_network_weights
from .trajectory import structed2dict_trajectory, dict2structed_trajectory

from catalyst.utils import *
