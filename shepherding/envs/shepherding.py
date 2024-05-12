from typing import Optional

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class ShepherdingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: Optional[str] = None, parameters=None):

        self.parameters = {

            # Parameters of the dynamics of the targets
            'num_targets': 4,  # Number of targets
            'target_max_vel': 1,  # Maximum speed achieved by one target
            'noise_strength': 1,  # Regulates noise strength
            'beta': 3,  # Regulates the strength of the repulsion exerted onto a target by one herder
            'lambda': 2.5,  # Maximum distance at which a target is repelled by a herder

            # Parameters of the dynamics of the herders
            'num_herders': 2,  # Number of herders
            'herder_max_vel': 8,  # Maximum speed achieved by one herder
            'xi': 500,  # Herders sensing range

            # Parameters of the environment
            'region_length': 60,  # Side-length of the region where the agent are confined to move
            'max_steps': 2000,  # Duration of the simulation in arbitrary units
            'dt': 0.05,  # Time step of the integration scheme
            'rho_g': 5,  # Size of the goal region centered around the origin
            # where you want to collect the targets
        }

        # Update parameters with custom ones
        if parameters is not None:
            self.parameters.update(parameters)

        # Save targets parameters
        self.num_targets = self.parameters['num_targets']
        self.target_max_vel = self.parameters['target_max_vel']
        self.noise_strength = self.parameters['noise_strength']
        self.beta = self.parameters['beta']
        self.lmbda = self.parameters['lambda']
        # Save herders parameters
        self.num_herders = self.parameters['num_herders']
        self.herder_max_vel = self.parameters['herder_max_vel']
        self.xi = self.parameters['xi']
        # Save environment parameters
        self.region_length = self.parameters['region_length']
        self.max_steps = self.parameters['max_steps']
        self.dt = self.parameters['dt']
        self.rho_g = self.parameters['rho_g']

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Define observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.num_herders + self.num_targets, 2), dtype=np.float64)
        # Define action space
        self.action_space = spaces.Box(low=-self.herder_max_vel, high=self.herder_max_vel, shape=(self.num_herders, 2))

        # Initialize herders and targets state
        self.herder_pos = np.zeros((self.num_herders, 2))
        self.target_pos = np.zeros((self.num_targets, 2))
        # Initialize herders and targets next state
        self.herder_pos_new = np.zeros((self.num_herders, 2))
        self.target_pos_new = np.zeros((self.num_targets, 2))

        # Define Pygame variables
        self.window_size = 600
        self.window = None
        self.clock = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # DEAL WITH OPTIONS

        # Initialize herders to random positions with zero velocity 
        self.herder_pos = self.np_random.uniform(-self.region_length / 2,
                                                 self.region_length / 2, size=(self.num_herders, 2))
        # Initialize targets to random positions with zero velocity
        self.target_pos = self.np_random.uniform(-self.region_length / 2,
                                                 self.region_length / 2, size=(self.num_targets, 2))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Take a step in the environment

        # Update herder positions and velocities
        herder_vel = np.clip(action, -self.herder_max_vel, self.herder_max_vel)
        self.herder_pos_new = self.herder_pos + herder_vel * self.dt
        self.herder_pos_new = np.clip(self.herder_pos_new, -self.region_length / 2, self.region_length / 2)

        # Update target positions
        self.target_pos_new = self.target_pos + \
                              np.sqrt(2 * self.noise_strength * self.dt) * \
                              self.np_random.normal(size=(self.num_targets, 2)) + \
                              self._repulsion(self.target_pos, self.herder_pos) * self.dt
        self.target_pos_new = np.clip(self.target_pos_new, -self.region_length / 2, self.region_length / 2)

        # Compute rewards and check for termination
        # Calculate target radii
        target_radii = np.linalg.norm(self.target_pos, axis=1)
        terminated = self._check_termination(target_radii)
        reward = self._compute_reward(target_radii, k_t=1)
        truncated = False
        info = self._get_info()

        # Update state
        self.herder_pos = self.herder_pos_new
        self.target_pos = self.target_pos_new

        if self.render_mode == "human":
            self._render_frame()

        # Return observation, reward, termination flag, and additional info
        return self._get_obs(), reward, terminated, truncated, info

    def _check_termination(self, target_radii):
        # Check if all radii are less than rho_g
        return np.all(target_radii < self.rho_g)

    def _compute_reward(self, target_radii, k_t):
        # Compute distance from goal
        distance_from_goal = target_radii - self.rho_g
        # Compute reward
        reward_vector = np.where(distance_from_goal < 0, distance_from_goal - k_t, distance_from_goal)
        reward = np.sum(reward_vector)
        return reward

    def _repulsion(self, target_pos, herder_pos):
        # Compute the repulsion force of herders on targets
        repulsion = np.zeros((len(target_pos), 2))  # Initialize repulsion array

        for i, target_pos_i in enumerate(target_pos):
            repulsion_i = np.zeros(2)
            distances = np.linalg.norm(herder_pos - target_pos_i, axis=1)
            for herder_pos_i, distance in zip(herder_pos, distances):
                if distance <= self.lmbda:
                    repulsion_i += (self.lmbda - distance) * (herder_pos_i - target_pos_i) / distance
            # Assign repulsion force to the corresponding row of the output array
            repulsion[i] = -repulsion_i

        return repulsion

    def _get_obs(self):
        # Get observation of herders and targets positions and velocities
        # state = np.concatenate((self.herder_pos.flatten(), self.target_pos.flatten()), dtype=np.float64)
        state = np.concatenate((self.herder_pos, self.target_pos))
        return state

    def _get_info(self):
        return {
            "num_herders": self.num_herders,
            "num_targets": self.num_targets
        }

    def _random_positions(self, num_agents):
        # Generate random positions for agents within the square environment
        return np.random.uniform(-self.region_length / 2, self.region_length / 2, size=(num_agents, 2))

    def render(self):
        if self.render_mode == "rgb_array":
            self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        # Rescale and draw herders
        for herder_pos in self.herder_pos:
            pygame.draw.circle(self.window, (0, 0, 255), self._rescale_position(herder_pos), 5)

        # Rescale and draw targets
        for target_pos in self.target_pos:
            pygame.draw.circle(self.window, (255, 0, 0), self._rescale_position(target_pos), 5)

        # Draw goal region
        pygame.draw.circle(self.window, (0, 255, 0), self._rescale_position((0, 0)), int(self.rho_g * 10), 2)

        pygame.display.flip()
        self.clock.tick(30)

    def _rescale_position(self, pos):
        return int(pos[0] * self.window_size / self.region_length) + self.window_size // 2, int(
            pos[1] * self.window_size / self.region_length) + self.window_size // 2

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
