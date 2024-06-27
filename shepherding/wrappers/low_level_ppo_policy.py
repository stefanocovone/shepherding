import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
from gymnasium import Wrapper

# Import your neural network
from shepherding.utils import ActorCritic


class LowLevelPPOPolicy(Wrapper):
    def __init__(self, env, update_frequency = 1):
        super().__init__(env)
        self.env = env
        # Action space is now discrete, each herder selects a target index
        self.action_space = spaces.MultiDiscrete([env.num_targets_max] * env.num_herders)
        # Observation space remains the same
        self.observation_space = env.observation_space
        # Initialize the neural network
        self.model = ActorCritic.ActorCritic()
        # Define the update frequency
        self.update_frequency = update_frequency

    def step(self, action):

        cumulative_reward = 0
        for step in range(self.update_frequency):

            # Prepare batched inputs for the neural network
            herder_positions = self.env.herder_pos / self.env.unwrapped.region_length
            if action > 100 + self.env.num_targets:
                target_positions = self.env.target_pos[action] / self.env.unwrapped.region_length
            else:
                target_positions = np.array([[0, 0]])

            relative_positions = target_positions - herder_positions
            # Stack relative positions and target positions to form the batch
            batch_inputs = np.hstack((relative_positions, target_positions)).astype(np.float32)

            # Convert the batch inputs to a tensor
            batch_tensor = torch.tensor(batch_inputs)

            # Get batched actions from the neural network
            batched_herder_actions = self.model.get_action_mean(batch_tensor).cpu().numpy()

            # Call the episode_step function of the original environment
            obs, reward, terminated, truncated, info = self.env.step(batched_herder_actions)
            cumulative_reward += reward

            if terminated or truncated:
                break

        obs = obs / self.env.unwrapped.region_length
        reward = cumulative_reward

        return obs, reward, terminated, truncated, info
