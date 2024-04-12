import os
import numpy as np
import matplotlib.pyplot as plt

import stable_baselines3
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from rlgym.env.boxy import Boxy
from rlgym.env.boxy3 import Boxy3
from rlgym.utils import *

# Variables
log_path = os.path.join('Training', 'Logs')
load_path = os.path.join('Training', 'Saved_Models', 'PPO_Boxy3_2')

# Initialise environment, memory and agent
env = DummyVecEnv([lambda: Boxy3()])
#noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=-.1 * np.ones(env.action_space.shape[0]))
model = PPO.load(load_path, env=env)

episodes = 1
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    episode_states = [obs[0]]
    action_history = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward   
        
        if done:
            episode_states.append(info[0]['terminal_observation'])
        else:
            episode_states.append(obs[0])

        action_history.append(action[0])

    print("Episode:{} Score:{}".format(episode, score))

    episode_states = np.asarray(episode_states)
    #print(episode_states)

    plt.plot(episode_states[:,0], episode_states[:,1])
    plt.scatter(episode_states[0,0], episode_states[0,1])

plt.scatter(0 ,0, marker='x')
plt.grid()

action_history = np.asarray(action_history)
plt.figure()
plt.plot(action_history[:,0])
plt.plot(action_history[:,1])
plt.xlabel('timesteps')
plt.ylabel('action')
plt.grid()

plt.figure()
plt.plot(episode_states[:,0])
plt.plot(episode_states[:,1])
plt.xlabel('timesteps')
plt.ylabel('position')
plt.plot()

plt.figure()
plt.plot(episode_states[:,2])
plt.plot(episode_states[:,3])
plt.xlabel('timesteps')
plt.ylabel('velocity')
plt.plot()

plt.figure()
plt.plot(episode_states[:,4])
plt.plot(episode_states[:,5])
plt.xlabel('timesteps')
plt.ylabel('acceleration')
plt.plot()

plt.show()

