import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from rlgym.utils import *

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate, name, chkpt_dir='tmp/ddpg', hidden_dim=256):
        super(Actor, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        action = self.actor(state)
        return action
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate, name, chkpt_dir='tmp/ddpg', hidden_dim=256):
        super(Critic, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        self.critic2 = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, state, action):
        x = self.critic1(state)
        value = self.critic2(torch.cat([x, action], dim=1))

        return value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    

class DDPG:
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, tau, gamma=0.99, lr_actor=1e-4, lr_critic=1e-3, device='cpu'):
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.device = device

        self.actor = Actor(state_dim, action_dim, lr_actor, name="Actor").to(self.device)
        self.actor_target = Actor(state_dim, action_dim, lr_actor, name="TargetActor").to(self.device)

        self.critic = Critic(state_dim, action_dim, lr_critic, name='Critic').to(self.device)
        self.critic_target = Critic(state_dim, action_dim, lr_critic, name='TargetCritic').to(self.device)

        self.memory = ReplayBuffer(buffer_size)

        self.noise = OUNoise(mu=np.zeros(action_dim))

        self.update_network_parameters(tau=1)

    def get_action(self, state, noise=0.0):
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = self.actor.forward(state).to(self.device)
        action = action + torch.tensor(noise, dtype=torch.float32).to(self.device)
        self.actor.train().to(self.device)
        return action.cpu().detach().numpy()
    
    def store(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(np.asarray(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.asarray(actions), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.actor_target.eval()
        self.critic_target.eval()
        self.critic.eval()

        # Compute current and target state-action values
        with torch.no_grad():
            target_actions = self.actor_target.forward(next_states)
            target_q_values = self.critic_target.forward(next_states, target_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * target_q_values # Bellman equation

        current_q_values = self.critic.forward(states, actions)

        # Train critic
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()

        # Update actor
        self.actor.optimizer.zero_grad()

        mu = self.actor.forward(states)
        self.actor.train()

        actor_loss = -self.critic(states, mu).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters(tau=None)


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_models(self):
        print('... saving checkpoint ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic_target.save_checkpoint()
    
    def load_models(self):
        print('... loading checkpoint ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic_target.load_checkpoint()

        

