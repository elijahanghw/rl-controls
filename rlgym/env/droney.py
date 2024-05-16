import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Droney(gym.Env):
    def __init__(self):
        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high =1, shape=(2,))
        self.observation_space = spaces.Box(low =-np.inf, high=np.inf, shape=(6,))

        # Define environment parameters
        self.max_steps = 1000
        self.dt = 0.01
        self.goal_threshold = 0.1
        self.goal_time = 0

        # Define variables
        self.state = np.zeros(4)
        self.step_count = 0

    def reset(self, seed=None):
        # Initialize state and step count
        x0 = np.random.uniform(-5,5)
        z0 = np.random.uniform(-5,5)
        u0 = np.random.uniform(-1,1)
        w0 = np.random.uniform(-1,1)
        theta0 = np.random.uniform(-1,1)
        q0 = np.random.uniform(-1,1)

        self.state = np.array([x0, z0, u0, w0, theta0, q0]) # Initial state of boxy
        self.step_count = 0
        return self.state, {}
    
    def step(self, action):
        # Update the state based on the action
        T1, T2 = action
        x, z, u, w, theta, q = self.state

        x += u * self.dt
        z += w * self.dt
        u += (-0.5 * u + (-T1 -T2 - 9)* np.sin(theta)) * self.dt
        w += (-0.5 * w + (T1 + T2 + 9) * np.cos(theta) - 9.81) * self.dt
        theta += q * self.dt
        q += (-30 * T1 + 30 * T2) * self.dt
        # Update state and step count
        self.state = np.array([x, z, u, w, theta, q])
        self.step_count += 1

        # Update action
        self.action = action

        ## Calculate the reward for reaching target
        reward = -self.dt * np.sqrt(x**2 + z**2) + self.dt

        goal_reached = np.linalg.norm(self.state) < self.goal_threshold
        if goal_reached:
            reward = 100
            # reward = self.dt

        out_of_bound = np.abs(x) > 10 or np.abs(z) > 10
        if out_of_bound:
            reward = -100

        # Check if the episode is done
        done = self.step_count >= self.max_steps or goal_reached or out_of_bound

        return self.state, reward, done, False, {}
