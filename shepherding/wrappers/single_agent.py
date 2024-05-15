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
        # Precompute the positions
        target_pos = self.unwrapped.target_pos
        herder_pos = self.unwrapped.herder_pos

        # Precompute distances
        target_distance = np.linalg.norm(target_pos)
        herder_distance = np.linalg.norm(herder_pos)

        # Precompute the difference vector
        diff_vector = target_pos - herder_pos
        diff_distance = np.linalg.norm(diff_vector)

        # Compute the reward components
        reward = (
                - self.k_1 * diff_distance
                - self.k_2 * target_distance
                - self.k_3 * self._compute_r_term(target_distance)
                + self.k_4 * self._compute_r_term(herder_distance)
        )

        return reward

    def _compute_r_term(self, x):
        rho_g = self.unwrapped.rho_g
        k_5 = self.k_5
        exp_term = np.exp(-k_5 * (x - rho_g))

        # Prevent division by zero or near-zero by adding a small epsilon
        epsilon = 1e-10
        return 1 / (1 - exp_term + epsilon) - 1






