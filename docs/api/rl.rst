RL
===========

.. automodule:: catalyst.rl
    :members:
    :undoc-members:
    :show-inheritance:


Random process
-----------------

.. automodule:: catalyst.rl.random_process
    :members:
    :undoc-members:
    :show-inheritance:

Agents
-------

.. currentmodule:: catalyst.rl.agents

.. autoclass:: Actor
    :members:
    :undoc-members:

.. autoclass:: Critic
    :members:
    :undoc-members:

.. autoclass:: ValueCritic
    :members:
    :undoc-members:


Layers
~~~~~~~~~~

.. automodule:: catalyst.rl.agents.layers
    :members:
    :undoc-members:
    :show-inheritance:


Utils
~~~~~~~~~~

.. automodule:: catalyst.rl.agents.utils
    :members:
    :undoc-members:
    :show-inheritance:


Offpolicy
-----------

.. currentmodule:: catalyst.rl.offpolicy


Trainer
~~~~~~~~~~

.. automodule:: catalyst.rl.offpolicy.trainer
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: redis2queue_loop


Sampler
~~~~~~~~~~

.. automodule:: catalyst.rl.offpolicy.sampler
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: get_actor_weights, set_actor_weights, set_params_noise

Algorithms
~~~~~~~~~~~

.. automodule:: catalyst.rl.offpolicy.algorithms
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: catalyst.rl.offpolicy.algorithms.core
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
