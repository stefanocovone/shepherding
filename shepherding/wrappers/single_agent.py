import gymnasium as gym
from gymnasium import Wrapper, spaces
import numpy as np
from shepherding.envs import ShepherdingEnv
class SingleAgent(Wrapper):
    def __init__(self, env, k_1=0.5, k_2=1, k_3=20, k_4=50, k_5=100):
        super().__init__(env)

        # Set reward weigths
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3
        self.k_4 = k_4
        self.k_5 = k_5

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward()
        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        reward = - self.k_1 * np.linalg.norm(self.unwrapped.target_pos - self.unwrapped.herder_pos) \
                    - self.k_2 * np.linalg.norm(self.unwrapped.target_pos) \
                    - self.k_3 * self._compute_r_term(self.unwrapped.target_pos) \
                    + self.k_4 * self._compute_r_term(self.unwrapped.herder_pos)
        return reward

    def _compute_r_term(self, x):
        return (1 / (1 - np.exp(-self.k_5 * (np.linalg.norm(x) - self.unwrapped.rho_g))) -1)





