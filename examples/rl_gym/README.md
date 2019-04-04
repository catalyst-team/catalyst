## Catalyst.RL – OpenAI Gym LunarLander example

1. System requirements – redis

    `sudo apt install redis-server`

2. Python requirements – box2d

    ```bash
    pip install gym['box2d']
    pip install tensorflow # for visualization
    ```

3. Run

    ```bash
    redis-server --port 12000
    export GPUS=""  # like GPUS="0" or GPUS="0,1" for multi-gpu training
 
    CUDA_VISIBLE_DEVICES="$GPUS" catalyst-rl run-trainer \
       --config=./rl_gym/config_dqn.yml
    
    CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers \
       --config=./rl_gym/config_dqn.yml
    
    CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs/rl_gym
    ```


## Additional links

[NeurIPS'18 Catalyst.RL solution](https://github.com/Scitator/neurips-18-prosthetics-challenge)
