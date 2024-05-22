import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Goland(gym.Env):
    def __init__(self):
        # Define action and observation space
        self.action_space = spaces.Box(low=-10*np.pi/180, high=10*np.pi/180, shape=(1,))
        self.observation_space = spaces.Box(low =-np.inf, high=np.inf, shape=(1,))
        
        # Aeroelastic parameters
        ds = 1/8
        V_inf = 160
        ref_chord = 1.8288
        
        # Import Goland Wing
        self.A_sys = np.load("rlgym/env/goland_wing/A_sys.npy")
        self.B_sys = np.load("rlgym/env/goland_wing/B_sys.npy")
        self.C_sys = np.load("rlgym/env/goland_wing/C_sys.npy")
        self.Disturbance = np.load("rlgym/env/goland_wing/Disturbance.npy")

        # Define environment parameters
        self.max_time = 1
        self.dt = ds*ref_chord/(V_inf)
        self.max_steps = int(self.max_time/self.dt)
        self.goal_threshold = 0.01
        self.goal_time = 0

        # Define variables
        self.state = np.zeros(1)
        self.x = np.zeros(self.A_sys.shape[0])
        self.step_count = 0

    def reset(self, seed=None):
        # Initialize state and step count
        self.state = np.zeros(1)
        self.x = np.zeros(self.A_sys.shape[0])
        self.step_count = 0
        return self.state, {}
    
    def step(self, action):
        # Update the state based on the action
        beta = action
        x_new = self.A_sys @ self.x + self.Disturbance[:,self.step_count] + self.B_sys[:,0]*beta 
        y = self.C_sys @ x_new
        self.x = x_new
        self.state = y
      
        # Update state and step count
        self.step_count += 1

        # Update action
        self.action = action

        ## Calculate the reward for reaching target
        reward = -10*np.sqrt(y**2)

        # Check if the episode is done
        done = self.step_count >= self.max_steps

        return self.state, reward, done, False, {}
