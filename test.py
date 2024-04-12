import time
import numpy as np
import matplotlib.pyplot as plt

from rlgym.env.boxy import Boxy
from rlgym.ddpg import DDPG
from rlgym.utils import *

DEVICE = 'cpu'

# Training parameters
gamma = 0.99
batch_size = 64
tau = 0.001

lr_actor = 2.5e-5
lr_critic = 2.5e-5

num_episodes = 10

# Initialise environment, memory and agent
env = Boxy()
agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], 200000, batch_size, tau, gamma, device=DEVICE)
agent.load_models()

score_history = []

for i in range(num_episodes):
    # Initialise episode
    state = env.reset()
    score = 0
    done = False
    episode_states = []

    while not done:
        action = agent.get_action(state, 0.0)
        ## Add scaling action
        next_state, reward, done, _ = env.step(action)
        episode_states.append(state)
        state = next_state
        score += reward   

    episode_states = np.asarray(episode_states)
    score_history.append(score)
    print(f'Episode: {i+1}/{num_episodes}  Score: {score:.2f}')

    plt.plot(episode_states[:,0], episode_states[:,1])
    plt.scatter(episode_states[0,0], episode_states[0,1])

plt.scatter(0 ,0, marker='x')
plt.grid()
plt.show()
