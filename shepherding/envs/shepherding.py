from typing import Optional

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class ShepherdingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: Optional[str] = None, parameters = None):

        self.parameters = {

            # Parameters of the dynamics of the targets
            'M': 4,             # Number of targets
            'VmaxT': 1,         # Maximum speed achieved by one target
            'D': 1,             # Regulates noise strength
            'beta': 3,          # Regulates the strenght of the repulsion exerted onto a target by one herder
            'lambda': 2.5,      # Maximum distance at which a target is repelled by a herder

            # Parameters of the dynamics of the herders
            'N': 2,             # Number of herders
            'VmaxH': 8,         # Maximum speed achieved by one herder
            'xi': 500,          # Herders sensing range

            # Parameters of the environment
            'L': 60,            # Side-length of the region where the agent are confined to move
            'Tf': 2000,         # Duration of the simulation in arbitrary units
            'dt': 0.05,         # Time step of the integration scheme
            'rho_g': 5,         # Size of the goal region centered arounf the origin where you want to collect the targets 
        }

        # Update parameters with custom ones
        if parameters is not None: self.parameters.update(parameters)

        # Save targets parameters
        self.M = self.parameters['M']
        self.VmaxT = self.parameters['VmaxT']
        self.D = self.parameters['D']
        self.beta = self.parameters['beta']
        self.lambbda = self.parameters['lambda']
        # Save herders parameters
        self.N = self.parameters['N']
        self.VmaxH = self.parameters['VmaxH']
        self.xi = self.parameters['xi']
        # Save environment parameters
        self.L = self.parameters['L']
        self.Tf = self.parameters['Tf']
        self.dt = self.parameters['dt']
        self.rho_g = self.parameters['rho_g']

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Define observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.N + self.M, 2), dtype=np.float64)
        # Define action space
        self.action_space = spaces.Box(low=-self.VmaxH, high=self.VmaxH, shape=(self.N, 2))

        # Initialize herders and targets state
        self.H_pos = np.zeros((self.N, 2))
        self.T_pos = np.zeros((self.M, 2))
        # Initialize herders and targets next state
        self.H_pos_new = np.zeros((self.N, 2))
        self.T_pos_new = np.zeros((self.M, 2))

        # Define Pygame variables
        self.window_size = 600
        self.window = None
        self.clock = None
        

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # DEAL WITH OPTIONS

        # Initialize herders to random positions with zero velocity 
        self.H_pos = self.np_random.uniform(-self.L/2, self.L/2, size=(self.N,2))
        # Initialize targets to random positions with zero velocity
        self.T_pos = self.np_random.uniform(-self.L/2, self.L/2, size=(self.M,2))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info




    def step(self, action):
        # Take a step in the environment

        # Update herder positions and velocities
        H_vel = np.clip(action, -self.VmaxH, self.VmaxH)
        self.H_pos_new = self.H_pos + H_vel*self.dt
        self.H_pos_new = np.clip(self.H_pos_new, -self.L/2, self.L/2)

        # Update target positions
        self.T_pos_new = self.T_pos + np.sqrt(2*self.D*self.dt) * self.np_random.normal(size=(self.M, 2)) + \
                         self._repulsion(self.T_pos, self.H_pos)*self.dt
        self.T_pos_new = np.clip(self.T_pos_new, -self.L/2, self.L/2)


        # Compute rewards and check for termination
        terminated = self._check_termination(self.T_pos_new, self.rho_g)
        reward = self._compute_reward(terminated)
        truncated = False
        info = self._get_info()

        # Update state
        self.H_pos = self.H_pos_new
        self.T_pos = self.T_pos_new

        if self.render_mode == "human":
            self._render_frame()

        # Return observation, reward, termination flag, and additional info
        return self._get_obs(), reward, terminated, truncated, info





    def _check_termination(self, T, rho_g):
        # Compute distances from the origin to all targets
        distances = np.linalg.norm(T, axis=1)
        
        # Check if all distances are less than rho_g
        all_distances_less_than_rho_g = np.all(distances < rho_g)
        
        return all_distances_less_than_rho_g
    



    def _compute_reward(self, terminated):
        if terminated:
            return 10
        else:
            return 0




    def _repulsion(self, T, H):
        # Compute the repulsion force of herders on targets
        repulsion = np.zeros((len(T), 2))  # Initialize repulsion array

        for i, target_pos in enumerate(T):
            repulsion_T = np.zeros(2)
            distances = np.linalg.norm(H - target_pos, axis=1)
            for herder_pos, distance in zip(H, distances):
                if distance <= self.lambbda:
                    repulsion_T += (self.lambbda - distance) * (herder_pos - target_pos) / distance
            # Assign repulsion force to the corresponding row of the output array
            repulsion[i] = -repulsion_T

        return repulsion





        

    


    def _get_obs(self):
        # Get observation of herders and targets positions and velocities
        # state = np.concatenate((self.H_pos.flatten(), self.T_pos.flatten()), dtype=np.float64)
        state = np.concatenate((self.H_pos, self.T_pos))
        return state
    
    def _get_info(self):
        return {
            "info": "info"
        }




    def _random_positions(self, num_agents):
        # Generate random positions for agents within the square environment
        return np.random.uniform(-self.L/2, self.L/2, size=(num_agents, 2))


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
        for herder_pos in self.H_pos:
            pygame.draw.circle(self.window, (0, 0, 255), self._rescale_position(herder_pos), 5)

        # Rescale and draw targets
        for target_pos in self.T_pos:
            pygame.draw.circle(self.window, (255, 0, 0), self._rescale_position(target_pos), 5)

        # Draw goal region
        pygame.draw.circle(self.window, (0, 255, 0), self._rescale_position((0, 0)), int(self.rho_g * 10), 2)

        pygame.display.flip()
        self.clock.tick(30)

    def _rescale_position(self, pos):
        return int(pos[0] * self.window_size / self.L) + self.window_size // 2, int(pos[1] * self.window_size / self.L) + self.window_size // 2


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()