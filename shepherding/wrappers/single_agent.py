from gymnasium import Wrapper
import numpy as np


class SingleAgentReward(Wrapper):
    def __init__(self, env, k_1=0.5, k_2=1, k_3=20, k_4=40, k_5=100):
        super().__init__(env)
        self.env = env
        self.env.compute_reward = False  # Disable reward computation in the original environment

        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3
        self.k_4 = k_4
        self.k_5 = k_5

    def step(self, action):
        _, _, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward()

        target_pos = self.env.unwrapped.target_pos
        herder_pos = self.env.unwrapped.herder_pos
        diff_pos = - herder_pos + target_pos

        obs = np.array([diff_pos, target_pos])/self.env.unwrapped.region_length

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        target_pos = self.env.unwrapped.target_pos
        herder_pos = self.env.unwrapped.herder_pos

        target_distance = np.linalg.norm(target_pos, axis=1)
        herder_distance = np.linalg.norm(herder_pos, axis=1)
        diff_distance = np.linalg.norm(target_pos - herder_pos, axis=1)

        reward = (
                - self.k_1 * diff_distance
                - self.k_2 * target_distance
                - self.k_3 * np.where(target_distance > self.env.unwrapped.rho_g, 0, -1)
                + self.k_4 * np.where(herder_distance > self.env.unwrapped.rho_g, 0, -1)
        )
        return reward/10
