from catalyst.rl import registry
from .actor import ConvActor
from .critic import ConvCritic, ConvQCritic
from .env import MiniGridEnvWrapper

registry.Environment(MiniGridEnvWrapper)
registry.Agent(ConvActor)
registry.Agent(ConvCritic)
registry.Agent(ConvQCritic)
