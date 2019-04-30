from catalyst.rl import registry

from .env import AtariEnvWrapper
from .critic import ConvActionCritic

registry.Environment(AtariEnvWrapper)
registry.Agent(ConvActionCritic)
