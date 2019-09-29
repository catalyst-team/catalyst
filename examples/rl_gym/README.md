## Catalyst.RL – OpenAI Gym LunarLander example

1. System requirements – redis

    `sudo apt install redis-server`

2. Python requirements – OpenAI Gym box2d

    ```bash
    pip install gym['box2d']
    ```

3. Start DB service

    ```bash
    redis-server --port 12000
    ```
    
4. Select config

    ```bash
    # DQN        – off-policy algorithm on discrete LunarLander
    export CONFIG=./rl_gym/config_dqn.yml

    # DDPG       – off-policy algorithm on continuous LunarLander
    export CONFIG=./rl_gym/config_ddpg.yml
    # SAC        – off-policy algorithm on continuous LunarLander
    export CONFIG=./rl_gym/config_sac.yml
    # TD3        – off-policy algorithm on continuous LunarLander
    export CONFIG=./rl_gym/config_td3.yml

    # PPO        – on-policy algorithm on discrete LunarLander
    export CONFIG=./rl_gym/config_ppo_discrete.yml
    # PPO        – on-policy algorithm on continuous LunarLander
    export CONFIG=./rl_gym/config_ppo_continuous.yml
    # REINFORCE  – on-policy algorithm on discrete LunarLander
    export CONFIG=./rl_gym/config_reinforce_discrete.yml
    # REINFORCE  – on-policy algorithm on continuous LunarLander
    export CONFIG=./rl_gym/config_reinforce_continuous.yml
    ```

3. Run trainer

    ```bash
    export GPUS=""  # like GPUS="0" or GPUS="0,1" for multi-gpu training
    CUDA_VISIBLE_DEVICES="$GPUS" catalyst-rl run-trainer --config="$CONFIG"
    ```

4. Run samplers

    ```bash
    CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers --config="$CONFIG"
    ```

5. For logs visualization, use

    ```bash
    CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
    ```

## Additional links

[NeurIPS'18 Catalyst.RL solution](https://github.com/Scitator/neurips-18-prosthetics-challenge)
