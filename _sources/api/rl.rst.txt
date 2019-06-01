RL
================================================

.. automodule:: catalyst.rl
    :members:
    :undoc-members:
    :show-inheritance:


Agents
--------------------

.. currentmodule:: catalyst.rl.agents

.. autoclass:: ActorSpec
    :members:
    :undoc-members:

.. autoclass:: CriticSpec
    :members:
    :undoc-members:

.. autoclass:: Actor
    :members:
    :undoc-members:

.. autoclass:: StateCritic
    :members:
    :undoc-members:

.. autoclass:: ActionCritic
    :members:
    :undoc-members:

.. autoclass:: StateActionCritic
    :members:
    :undoc-members:

.. autoclass:: ValueHead
    :members:
    :undoc-members:

.. autoclass:: PolicyHead
    :members:
    :undoc-members:

.. autoclass:: TemporalAttentionPooling
    :members:
    :undoc-members:

.. autoclass:: LamaPooling
    :members:
    :undoc-members:

.. autoclass:: CouplingLayer
    :members:
    :undoc-members:

.. autoclass:: SquashingLayer
    :members:
    :undoc-members:

.. autoclass:: StateNet
    :members:
    :undoc-members:

.. autoclass:: StateActionNet
    :members:
    :undoc-members:

.. autoclass:: CategoricalPolicy
    :members:
    :undoc-members:

.. autoclass:: GaussPolicy
    :members:
    :undoc-members:

.. autoclass:: RealNVPPolicy
    :members:
    :undoc-members:


Agent utils
~~~~~~~~~~~~~~~~

.. automodule:: catalyst.rl.agents.utils
    :members:
    :undoc-members:
    :show-inheritance:


DB
--------------------

.. currentmodule:: catalyst.rl.db

.. autoclass:: DBSpec
    :members:
    :undoc-members:

.. autoclass:: MongoDB
    :members:
    :undoc-members:

.. autoclass:: RedisDB
    :members:
    :undoc-members:


Environments
--------------------

.. currentmodule:: catalyst.rl.environments

.. autoclass:: EnvironmentSpec
    :members:
    :undoc-members:

.. autoclass:: GymWrapper
    :members:
    :undoc-members:


Exploration
--------------------

.. currentmodule:: catalyst.rl.exploration

.. autoclass:: ExplorationHandler
    :members:
    :undoc-members:

.. autoclass:: ExplorationStrategy
    :members:
    :undoc-members:

.. autoclass:: Greedy
    :members:
    :undoc-members:

.. autoclass:: EpsilonGreedy
    :members:
    :undoc-members:

.. autoclass:: Boltzmann
    :members:
    :undoc-members:

.. autoclass:: NoExploration
    :members:
    :undoc-members:

.. autoclass:: GaussNoise
    :members:
    :undoc-members:

.. autoclass:: OrnsteinUhlenbeckProcess
    :members:
    :undoc-members:

.. autoclass:: ParameterSpaceNoise
    :members:
    :undoc-members:


Offpolicy
--------------------

.. currentmodule:: catalyst.rl.offpolicy


Trainer
~~~~~~~~~~~~~~~~

.. automodule:: catalyst.rl.offpolicy.trainer
    :members:
    :undoc-members:
    :show-inheritance:


Sampler
~~~~~~~~~~~~~~~~

.. automodule:: catalyst.rl.offpolicy.sampler
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: get_actor_weights, set_actor_weights, set_params_noise

Algorithms
~~~~~~~~~~~~~~~~

.. automodule:: catalyst.rl.offpolicy.algorithms
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.rl.offpolicy.algorithms.core
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.rl.offpolicy.algorithms.core_continuous
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.rl.offpolicy.algorithms.core_discrete
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.rl.offpolicy.algorithms.ddpg
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.rl.offpolicy.algorithms.sac
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.rl.offpolicy.algorithms.td3
    :members:
    :undoc-members:
    :show-inheritance:


Onpolicy
--------------------

.. currentmodule:: catalyst.rl.onpolicy


Trainer
~~~~~~~~~~~~~~~~

.. automodule:: catalyst.rl.onpolicy.trainer
    :members:
    :undoc-members:
    :show-inheritance:


Sampler
~~~~~~~~~~~~~~~~

.. automodule:: catalyst.rl.onpolicy.sampler
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: get_actor_weights, set_actor_weights, set_params_noise

Algorithms
~~~~~~~~~~~~~~~~

.. automodule:: catalyst.rl.onpolicy.algorithms
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.rl.onpolicy.algorithms.actor
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.rl.onpolicy.algorithms.actor_critic
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.rl.onpolicy.algorithms.core
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.rl.onpolicy.algorithms.ppo
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.rl.onpolicy.algorithms.reinforce
    :members:
    :undoc-members:
    :show-inheritance:

