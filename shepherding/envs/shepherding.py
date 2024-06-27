from typing import Optional

import numpy as np
import pygame
import pygame.freetype

import gymnasium as gym
from gymnasium import spaces, Wrapper


class ShepherdingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode: Optional[str] = None, parameters=None,
                 compute_reward: bool = True, rand_target: bool = False):
        self.compute_reward = compute_reward
        self.parameters = {
            'num_targets_min': 2,
            'num_targets_max': 10,
            'num_targets': 5,
            'noise_strength': 1,
            'k_T': 3,
            'lambda': 2.5,
            'num_herders': 2,
            'herder_max_vel': 8,
            'xi': 10,
            'alpha': 5,
            'region_length': 50,
            'max_steps': 2000,
            'dt': 0.05,
            'rho_g': 10,
        }

        if parameters is not None:
            self.parameters.update(parameters)

        self.num_targets = self.parameters['num_targets']
        if rand_target:
            self.num_targets_min = self.parameters['num_targets_min']
            self.num_targets_max = self.parameters['num_targets_max']
        else:
            self.num_targets_min = self.num_targets
            self.num_targets_max = self.num_targets

        self.noise_strength = self.parameters['noise_strength']
        self.k_T = self.parameters['k_T']
        self.lmbda = self.parameters['lambda']
        self.num_herders = self.parameters['num_herders']
        self.herder_max_vel = self.parameters['herder_max_vel']
        self.xi = self.parameters['xi']
        self.alpha = self.parameters['alpha']
        self.region_length = self.parameters['region_length']
        self.max_steps = self.parameters['max_steps']
        self.dt = self.parameters['dt']
        self.rho_g = self.parameters['rho_g']

        self.num_agents = self.num_herders + self.num_targets_max

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=-self.region_length / 2, high=self.region_length / 2,
                                            shape=(self.num_herders + self.num_targets_max, 2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.herder_max_vel, high=self.herder_max_vel, shape=(self.num_herders, 2))

        self.herder_pos = np.zeros((self.num_herders, 2))
        self.target_pos = np.zeros((self.num_targets_max, 2))
        self.herder_pos_new = np.zeros((self.num_herders, 2))
        self.target_pos_new = np.zeros((self.num_targets_max, 2))

        self.episode_step = 0

        self.window_size = 600
        self.window = None
        self.clock = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.num_targets = self.np_random.integers(self.num_targets_min, self.num_targets_max + 1)

        all_positions = self._random_positions(self.num_herders + self.num_targets)
        self.herder_pos = all_positions[:self.num_herders]
        self.target_pos = all_positions[self.num_herders:self.num_herders + self.num_targets]

        observation = self._get_obs()
        info = self._get_info()

        self.episode_step = 0

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
        reward = self._compute_reward(target_radii, k_t=10) if self.compute_reward else 0.0
        terminated = False
        truncated = False
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.episode_step += 1

        return self._get_obs(), reward, terminated, truncated, info

    def _compute_reward(self, target_radii, k_t):
        distance_from_goal = target_radii - self.rho_g
        reward_vector = np.where(distance_from_goal < 0, -k_t, distance_from_goal)
        reward = -np.sum(reward_vector) / 100
        return reward

    def _repulsion(self):
        differences = self.herder_pos[:, np.newaxis, :] - self.target_pos[np.newaxis, :, :]
        distances = np.linalg.norm(differences, axis=2)
        nearby_agents = distances < self.lmbda
        nearby_differences = np.where(nearby_agents[:, :, np.newaxis], differences, 0)
        distances_with_min = np.maximum(distances[:, :, np.newaxis], 1e-6)
        nearby_unit_vector = nearby_differences / distances_with_min
        repulsion = -self.k_T * np.sum((self.lmbda - distances[:, :, np.newaxis]) * nearby_unit_vector, axis=0)
        return repulsion

    def _get_obs(self):
        # Create an array of zeros for the target positions
        zero_filled_targets = np.zeros((self.num_targets_max, 2), dtype=np.float32)

        # Fill the first 'num_targets' positions with the actual target positions
        zero_filled_targets[:self.num_targets] = self.target_pos[:self.num_targets]

        # Concatenate herder positions and zero-filled target positions
        state = np.concatenate((self.herder_pos, zero_filled_targets)).astype(np.float32)

        return state

    def _get_info(self):
        return {"num_herders": self.num_herders, "num_targets": self.num_targets}

    def _random_positions(self, num_agents):
        radius = self.np_random.uniform(self.rho_g + 1, 0.9 * self.region_length / 2, num_agents)
        angle = self.np_random.uniform(0, 2 * np.pi, num_agents)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        position = np.column_stack((x, y))
        return position

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        else:
            self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.freetype.init()
            self.window = pygame.display.set_mode((self.window_size + 20, self.window_size + 80))  # Increased window height
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        pygame.event.get()

        # Define the center and radius of circles
        center = self._rescale_position((0, 0))
        goal_radius = int(self.rho_g * self.window_size / self.region_length)
        domain_radius = int(0.5 * self.window_size)

        # Draw the initial region as a shaded circle (yellow with transparency)
        s = pygame.Surface((2 * domain_radius, 2 * domain_radius), pygame.SRCALPHA)
        pygame.draw.circle(s, (255, 255, 0, 77), (domain_radius, domain_radius),
                           0.9 * domain_radius)  # Alpha = 77 for transparency

        # pygame.draw.rect(s, (255, 255, 0, 57), (0, 0, 2*domain_radius, 2*domain_radius))  # Alpha = 77 for transparency
        pygame.draw.rect(s, (0, 0, 0), (0, 0, 2 * domain_radius, 2 * domain_radius), 1)  # Black border
        self.window.blit(s, (center[0] - domain_radius, center[1] - domain_radius))

        # Draw the goal region as a solid blue circle
        pygame.draw.circle(self.window, (0, 116, 187), center, goal_radius, 5)

        # Draw targets and count those in the goal region
        p_in = 0  # Counter for targets in the goal region
        for target_pos in self.target_pos[:self.num_targets]:
            pygame.draw.circle(self.window, (255, 0, 255), self._rescale_position(target_pos),
                               5)  # Smaller circle for targets
            pygame.draw.circle(self.window, (0, 0, 0), self._rescale_position(target_pos),
                               5, 1)  # Smaller circle for targets
            if np.linalg.norm(target_pos) < 5:
                p_in += 1

        # Draw herders as diamonds
        for herder_pos in self.herder_pos:
            rescaled_pos = self._rescale_position(herder_pos)
            pygame.draw.polygon(self.window, (0, 0, 255), [
                (rescaled_pos[0], rescaled_pos[1] - 10),  # Top
                (rescaled_pos[0] + 8, rescaled_pos[1]),  # Right
                (rescaled_pos[0], rescaled_pos[1] + 10),  # Bottom
                (rescaled_pos[0] - 8, rescaled_pos[1]),  # Left
            ])
            pygame.draw.polygon(self.window, (0, 0, 0), [
                (rescaled_pos[0], rescaled_pos[1] - 10),  # Top
                (rescaled_pos[0] + 8, rescaled_pos[1]),  # Right
                (rescaled_pos[0], rescaled_pos[1] + 10),  # Bottom
                (rescaled_pos[0] - 8, rescaled_pos[1]),  # Left
            ],1)

        # Display fraction of captured targets and time
        font = pygame.freetype.SysFont("cmb10.ttf", 24)

        # Specify the path to your Computer Modern font file (cmr10.ttf or any variant)
        font_path = r'C:\Users\stefa\anaconda3\pkgs\matplotlib-base-3.8.0-py311hf62ec03_0\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\cmr10.ttf'


        # Load the font file
        font = pygame.font.Font(font_path, 24)  # Adjust the font size (24 in this case)

        fraction_text = f'Fraction of captured targets X={p_in / len(self.target_pos[:self.num_targets]):.2f}'
        current_time = self.episode_step * self.dt
        time_text = f't={current_time:.2f}'

        # Render text
        fraction_rendered = font.render(fraction_text, True, (0, 0, 0))
        time_rendered = font.render(time_text, True, (0, 0, 0))

        # Calculate centering positions
        fraction_x = (self.window.get_width() - fraction_rendered.get_width()) // 2
        time_x = (self.window.get_width() - time_rendered.get_width()) // 2

        # Display text centered horizontally
        self.window.blit(fraction_rendered, (fraction_x, 10))  # Adjust y-coordinate as needed
        self.window.blit(time_rendered, (time_x, 40))  # Adjust y-coordinate as needed

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            frame = pygame.surfarray.array3d(self.window)
            frame = np.transpose(frame,
                                 (1, 0, 2))  # Convert from (width, height, channels) to (height, width, channels)
            return frame

    def _rescale_position(self, pos):
        return int(pos[0] * self.window_size / self.region_length) + (self.window_size+20) // 2, int(
            pos[1] * self.window_size / self.region_length) + (self.window_size+140) // 2

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
