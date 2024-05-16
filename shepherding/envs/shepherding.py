from typing import Optional

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces, Wrapper


class ShepherdingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode: Optional[str] = None, parameters=None, compute_reward: bool = True):
        self.compute_reward = compute_reward
        self.parameters = {
            'num_targets': 4,
            'target_max_vel': 1,
            'noise_strength': 1,
            'beta': 3,
            'lambda': 2.5,
            'num_herders': 2,
            'herder_max_vel': 8,
            'xi': 500,
            'region_length': 60,
            'max_steps': 2000,
            'dt': 0.05,
            'rho_g': 5,
        }

        if parameters is not None:
            self.parameters.update(parameters)

        self.num_targets = self.parameters['num_targets']
        self.target_max_vel = self.parameters['target_max_vel']
        self.noise_strength = self.parameters['noise_strength']
        self.beta = self.parameters['beta']
        self.lmbda = self.parameters['lambda']
        self.num_herders = self.parameters['num_herders']
        self.herder_max_vel = self.parameters['herder_max_vel']
        self.xi = self.parameters['xi']
        self.region_length = self.parameters['region_length']
        self.max_steps = self.parameters['max_steps']
        self.dt = self.parameters['dt']
        self.rho_g = self.parameters['rho_g']

        self.num_agents = self.num_herders + self.num_targets

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=-self.region_length / 2, high=self.region_length / 2,
                                            shape=(self.num_herders + self.num_targets, 2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.herder_max_vel, high=self.herder_max_vel, shape=(self.num_herders, 2))

        self.herder_pos = np.zeros((self.num_herders, 2))
        self.target_pos = np.zeros((self.num_targets, 2))
        self.herder_pos_new = np.zeros((self.num_herders, 2))
        self.target_pos_new = np.zeros((self.num_targets, 2))

        self.window_size = 600
        self.window = None
        self.clock = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        all_positions = self._random_positions(self.num_herders + self.num_targets)
        self.herder_pos = all_positions[:self.num_herders]
        self.target_pos = all_positions[self.num_herders:]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, {}

    def step(self, action):
        herder_vel = np.clip(action, -self.herder_max_vel, self.herder_max_vel)
        self.herder_pos_new = np.clip(self.herder_pos + herder_vel * self.dt, -self.region_length / 2,
                                      self.region_length / 2)

        noise = np.sqrt(2 * self.noise_strength * self.dt) * self.np_random.normal(size=(self.num_targets, 2))
        repulsion = self._repulsion() * self.dt
        self.target_pos_new = np.clip(self.target_pos + noise + repulsion, -self.region_length / 2,
                                      self.region_length / 2)

        self.herder_pos = self.herder_pos_new
        self.target_pos = self.target_pos_new

        target_radii = np.linalg.norm(self.target_pos, axis=1)
        reward = self._compute_reward(target_radii, k_t=1) if self.compute_reward else 0.0
        terminated = False
        truncated = False
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, info

    def _compute_reward(self, target_radii, k_t):
        distance_from_goal = target_radii - self.rho_g
        reward_vector = np.where(distance_from_goal < 0, distance_from_goal - k_t, distance_from_goal)
        reward = -np.sum(reward_vector)
        return reward

    def _repulsion(self):
        differences = self.herder_pos[:, np.newaxis, :] - self.target_pos[np.newaxis, :, :]
        distances = np.linalg.norm(differences, axis=2)
        nearby_agents = distances < self.lmbda
        nearby_differences = np.where(nearby_agents[:, :, np.newaxis], differences, 0)
        distances_with_min = np.maximum(distances[:, :, np.newaxis], 1e-6)
        nearby_unit_vector = nearby_differences / distances_with_min
        repulsion = -self.beta * np.sum((self.lmbda - distances[:, :, np.newaxis]) * nearby_unit_vector, axis=0)
        return repulsion

    def _get_obs(self):
        state = np.concatenate((self.herder_pos, self.target_pos)).astype(np.float32)
        return state

    def _get_info(self):
        return {"num_herders": self.num_herders, "num_targets": self.num_targets}

    def _random_positions(self, num_agents):
        return self.np_random.uniform(-self.region_length / 2, self.region_length / 2, size=(num_agents, 2))

    def render(self):
        if self.render_mode == "rgb_array":
            self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        for herder_pos in self.herder_pos:
            pygame.draw.circle(self.window, (0, 0, 255), self._rescale_position(herder_pos), 5)

        for target_pos in self.target_pos:
            pygame.draw.circle(self.window, (255, 0, 0), self._rescale_position(target_pos), 5)

        pygame.draw.circle(self.window, (0, 255, 0), self._rescale_position((0, 0)), int(self.rho_g * 10), 2)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _rescale_position(self, pos):
        return int(pos[0] * self.window_size / self.region_length) + self.window_size // 2, int(pos[1] * self.window_size / self.region_length) + self.window_size // 2

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
