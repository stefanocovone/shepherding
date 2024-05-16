import gymnasium as gym
from gymnasium import Wrapper, spaces
import numpy as np
from shepherding.envs import ShepherdingEnv
class SingleAgentReward(Wrapper):
    def __init__(self, env, k_1=0.5, k_2=1, k_3=20, k_4=50, k_5=100):
        super().__init__(env)
        self.env = env
        self.env.compute_reward = False  # Disable reward computation in the original environment

        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3
        self.k_4 = k_4
        self.k_5 = k_5

        self.k5_rho_g = self.k_5 * self.unwrapped.rho_g

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward()
        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        target_pos = self.unwrapped.target_pos
        herder_pos = self.unwrapped.herder_pos

        target_distance = np.linalg.norm(target_pos, axis=1)
        herder_distance = np.linalg.norm(herder_pos, axis=1)
        diff_distance = np.linalg.norm(target_pos - herder_pos, axis=1)

        reward = (
                - self.k_1 * diff_distance
                - self.k_2 * target_distance
                - self.k_3 * (1 / (1 - np.exp(-self.k5_rho_g + target_distance)) - 1 + 1e-10)
                + self.k_4 * (1 / (1 - np.exp(-self.k5_rho_g + herder_distance)) - 1 + 1e-10)
        )

        return reward