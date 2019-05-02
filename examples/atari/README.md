## Catalyst.RL – Atari example

This example shows how to use Catalyst.RL with
- your custom environment, like Atari with a bunch of wrappers,
- and custom agent, like CNN-based critic.


1. System requirements – redis

    `sudo apt install redis-server`

2. Python requirements – OpenAI Gym  Atari

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
