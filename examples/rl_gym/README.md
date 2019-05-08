## Catalyst.RL – OpenAI Gym LunarLander example

1. System requirements – redis

    `sudo apt install redis-server`

2. Python requirements – OpenAI Gym box2d

    ```bash
    pip install gym['box2d']
    ```

3. Run DQN – discrete action space environment

    ```bash
    redis-server --port 12000
    export GPUS=""  # like GPUS="0" or GPUS="0,1" for multi-gpu training
 
    CUDA_VISIBLE_DEVICES="$GPUS" catalyst-rl run-trainer \
       --config=./rl_gym/config_dqn.yml
    
    CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers \
       --config=./rl_gym/config_dqn.yml
    
    CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
    ```

4. Run DDPG – continuous action space environment

    ```bash
    redis-server --port 12000
    export GPUS=""  # like GPUS="0" or GPUS="0,1" for multi-gpu training
 
    CUDA_VISIBLE_DEVICES="$GPUS" catalyst-rl run-trainer \
       --config=./rl_gym/config_ddpg.yml
    
    CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers \
       --config=./rl_gym/config_ddpg.yml
    
    CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
    ```
 
5. Run TD3 – continuous action space environment

    ```bash
    redis-server --port 12000
    export GPUS=""  # like GPUS="0" or GPUS="0,1" for multi-gpu training
 
    CUDA_VISIBLE_DEVICES="$GPUS" catalyst-rl run-trainer \
       --config=./rl_gym/config_td3.yml
    
    CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers \
       --config=./rl_gym/config_td3.yml
    
    CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
    ```

6. Run SAC – continuous action space environment

    ```bash
    redis-server --port 12000
    export GPUS=""  # like GPUS="0" or GPUS="0,1" for multi-gpu training
 
    CUDA_VISIBLE_DEVICES="$GPUS" catalyst-rl run-trainer \
       --config=./rl_gym/config_sac.yml
    
    CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers \
       --config=./rl_gym/config_sac.yml
    
    CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
    ```

7. Run PPO – on-policy algorithm for discrete action space environment

    ```bash
    redis-server --port 12000
    export GPUS=""  # like GPUS="0" or GPUS="0,1" for multi-gpu training

    CUDA_VISIBLE_DEVICES="$GPUS" catalyst-rl run-trainer \
       --config=./rl_gym/config_ppo.yml --setup=on-policy

    CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers \
       --config=./rl_gym/config_ppo.yml --setup=on-policy

    CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
    ```

## Additional links

[NeurIPS'18 Catalyst.RL solution](https://github.com/Scitator/neurips-18-prosthetics-challenge)
