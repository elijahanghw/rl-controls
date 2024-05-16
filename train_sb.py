import os
import time
import numpy as np
import torch

import stable_baselines3
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from rlgym.env.boxy2 import Boxy2
from rlgym.env.droney import Droney
from rlgym.utils import *

start_time = time.time()

# Path Variables
log_path = os.path.join('Training', 'Logs', 'PPO_Droney')
save_path = os.path.join('Training', 'Saved_Models', 'PPO_Droney')
load_path = os.path.join('Training', 'Saved_Models', 'PPO_Droney')


# Initialise environment, memory and agent
env = make_vec_env(lambda: Droney(), n_envs=20)
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 128, 128])
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_path)

model = PPO.load(load_path, env=env)

TIMESTEPS = 10_000
for i in range(1, 100):
    print(f"Iteration {i}")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(save_path)
    avg_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print(f"Average reward for 50 Episodes: {avg_reward[0]}")
    print(f"Time elapsed: {time.time() - start_time:.2f}")
    print("-----------------------------------------------")
    if avg_reward[0] >= 90:
        print("Reward threshold reached, stopping training")
        break