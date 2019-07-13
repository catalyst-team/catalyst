## Test - Catalyst.RL: OpenAI Gym

This example is needed for CI test of Catalyst.RL.

#### How to run

1. System requirements – redis

    `sudo apt install redis-server`

2. Start DB service

    ```bash
    redis-server --port 12000
    ```

3. Select config
    ```bash
    # DQN        – off-policy algorithm on discrete action space
    export CONFIG=./_tests_rl_gym/config_dqn_base.yml

    # DDPG       – off-policy algorithm on continuous action space environment
    export CONFIG=./_tests_rl_gym/config_ddpg_base.yml

    # SAC        – off-policy algorithm on continuous action space environment
    export CONFIG=./_tests_rl_gym/config_sac_base.yml

    # TD3        – off-policy algorithm on continuous action space environment
    export CONFIG=./_tests_rl_gym/config_td3_base.yml


    # REINFORCE  – on-policy algorithm on discrete action space
    export CONFIG=./_tests_rl_gym/config_reinforce.yml

    # PPO        – on-policy algorithm on discrete action space
    export CONFIG=./_tests_rl_gym/config_ppo_discrete.yml
    ```

4. Run trainer

    ```bash
    export GPUS=""  # like GPUS="0" or GPUS="0,1" for multi-gpu training
    CUDA_VISIBLE_DEVICES="$GPUS" catalyst-rl run-trainer --config="$CONFIG"
    ```

5. Run samplers

    ```bash
    CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers --config="$CONFIG"
    ```
