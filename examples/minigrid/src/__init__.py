from catalyst.rl import registry

from .env import MiniGridEnvWrapper
from .actor import ConvActor
from .critic import ConvCritic, ConvQCritic

registry.Environment(MiniGridEnvWrapper)
registry.Agent(ConvActor)
registry.Agent(ConvCritic)
registry.Agent(ConvQCritic)
