from catalyst.rl import registry

from .env import AtariEnvWrapper
from .actor import ConvActor
from .critic import ConvCritic, ConvQCritic

registry.Environment(AtariEnvWrapper)
registry.Agent(ConvActor)
registry.Agent(ConvCritic)
registry.Agent(ConvQCritic)
