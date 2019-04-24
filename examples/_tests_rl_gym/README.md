## Test - Catalyst.RL: OpenAI Gym MountainCarContinuous

This example is needed for CI test of Catalyst.RL.

#### How to run

##### Examples RL – Discrete
```bash
redis-server --port 12000

catalyst-rl run-trainer --config=./examples/_tests_rl_gym/config_dqn.yml

catalyst-rl run-samplers --config=./examples/_tests_rl_gym/config_dqn.yml
```

##### Examples RL – Continuous
```bash
redis-server --port 12000

catalyst-rl run-trainer --config=./examples/_tests_rl_gym/config_sac.yml

catalyst-rl run-samplers --config=./examples/_tests_rl_gym/config_sac.yml
```
