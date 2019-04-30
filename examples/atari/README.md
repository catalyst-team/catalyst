
1. System requirements – redis

    `sudo apt install redis-server`

2. Python requirements – box2d

    ```bash
    pip install gym['atari']
    ```

3. Run DQN – discrete action space environment

    ```bash
    redis-server --port 12000
    export GPUS=""  # like GPUS="0" or GPUS="0,1" for multi-gpu training
 
    CUDA_VISIBLE_DEVICES="$GPUS" catalyst-rl run-trainer --config=./config.yml
    
    CUDA_VISIBLE_DEVICES="" catalyst-rl run-samplers --config=./config.yml
    
    CUDA_VISIBLE_DEVICE="" tensorboard --logdir=./logs
    ```