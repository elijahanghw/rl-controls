import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from rlgym.env.boxy import Boxy
from rlgym.ddpg import DDPG
from rlgym.utils import *

print("Training DDPG agent on Boxy environment")
# Training device
# DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
print(f'Training on {DEVICE}')
print('---------------------------------------------------------------------')

# Training parameters
gamma = 0.99
batch_size = 64
tau = 0.01

lr_actor = 0.001
lr_critic = 0.001

num_episodes = 3000
print_freq = 10
save_freq = 100

# Initialise environment, memory and agent
env = Boxy()
agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], 1000000, batch_size, tau, gamma, device=DEVICE)
noise = OUNoise(mu=np.zeros(env.action_space.shape[0]))

np.random.seed(0)

score_history = []
score_rolling_average = []
start_time = time.time()
for i in range(num_episodes):
    # Initialise episode
    state, _ = env.reset()
    score = 0
    done = False

    while not done:
        action = agent.get_action(state, noise())
        ## Add scaling action
        next_state, reward, done, _, _ = env.step(action)
        agent.store(state, action, reward, next_state, int(done))
        agent.learn()
        state = next_state
        score += reward   

    score_history.append(score)
    score_rolling_average.append(np.mean(score_history[-100:]))
    if (i+1) % print_freq == 0:
        print(f'Episode: {i+1}/{num_episodes}  Score: {score:.2f}    100 game average: {score_rolling_average[-1]:.2f}   Elapsed time: {time.time() - start_time:.2f}s') 
    
    if (i+1) % save_freq == 0:
        agent.save_models()

plt.plot(list(range(num_episodes)), score_history, color='tab:blue', alpha=0.3)
plt.plot(list(range(num_episodes)), score_rolling_average, color='tab:blue')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.grid()
plt.savefig('boxy_training.png')

print('Training done')