## Catalyst.RL – MiniGrid example

This example shows how to use Catalyst.RL with
- your custom environment, like [GymMiniGrid](https://github.com/maximecb/gym-minigrid) with Dict observations,
- and custom agent, like CNN-based actor/critic,
- MongoDB and memmap replay buffer for efficient resource management.


1. System requirements – Mongo/Redis

    - [Mongo installation](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/)
    - Redis installation: 
        `sudo apt install redis-server`

2. Python requirements – [GymMiniGrid](https://github.com/maximecb/gym-minigrid)

    ```bash
    pip install gym-minigrid
    ```

3. Run DQN on MiniGrid-Empty-8x8-v0

    ```bash
    redis-server --port 12000  # or mongod --config mongod.conf --port 12000
    # check config.yml for correct db/db specification

    export GPUS=""  # like GPUS="0" or GPUS="0,1" for multi-gpu training

    CUDA_VISIBLE_DEVICES="$GPUS" catalyst-rl run-trainer --config=./config_dqn.yml

    CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers --config=./config_dqn.yml

    CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
    ```

4. Run PPO on MiniGrid-Empty-8x8-v0

    ```bash
    redis-server --port 12001  # or mongod --config mongod.conf --port 12001
    # check config.yml for correct db/db specification
 
    export GPUS=""  # like GPUS="0" or GPUS="0,1" for multi-gpu training
 
    CUDA_VISIBLE_DEVICES="$GPUS" catalyst-rl run-trainer --config=./config_ppo.yml
    
    CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers --config=./config_ppo.yml
    
    CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
    ```
