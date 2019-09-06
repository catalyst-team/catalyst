from catalyst.rl import registry

from .env import AtariEnvWrapper

registry.Environment(AtariEnvWrapper)
