import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
from gymnasium import Wrapper

# Import your neural network
from shepherding.utils import ActorCritic


class MultiAgentPPO(Wrapper):
    def __init__(self, env, update_frequency=1):
        super().__init__(env)
        self.env = env

        self.num_closest_targets = 5  # Number of closest targets to include in the observation
        self.closest_targets_indices = []

        # Action space is now discrete, each herder selects a target index
        self.action_space = spaces.MultiDiscrete([self.num_closest_targets] * env.unwrapped.num_herders)

        # Observation space:
        # For each herder:
        # - own position (2)
        # - position of closest herder (2)
        # - positions of 5 closest targets (5 * 2)
        # Total features per herder: 2 + 2 + 10 = 14

        num_features = 2 + 2 + self.num_closest_targets * 2  # 14 features per herder
        num_herders = self.env.unwrapped.num_herders
        region_length = self.env.unwrapped.region_length

        # Define observation space
        obs_low = np.full((num_herders, num_features), -region_length / 2, dtype=np.float32)
        obs_high = np.full((num_herders, num_features), region_length / 2, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Initialize the neural network
        self.model = ActorCritic.ActorCritic()
        # Define the update frequency
        self.update_frequency = update_frequency

        self.obs_batch = np.zeros(shape=((self.update_frequency,) + env.unwrapped.observation_space.shape))


    def step(self, action):
        """
        Processes the action and steps through the environment.

        Args:
            action (np.ndarray): Array of length num_herders, where each element is an integer in [0, 4],
                                 representing which of the 5 closest targets the herder should chase.

        Returns:
            obs (np.ndarray): Processed observations.
            reward (float): Cumulative reward.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information.
        """

        num_herders = self.env.unwrapped.num_herders

        # Map action indices to actual target indices in env.target_pos
        target_indices_to_chase = self.closest_targets_indices[np.arange(num_herders), action]

        cumulative_reward = 0
        self.obs_batch = np.zeros(shape=((self.update_frequency,) + self.env.unwrapped.observation_space.shape))

        for step in range(self.update_frequency):

            # Prepare batched inputs for the neural network
            herder_positions = self.env.unwrapped.herder_pos / self.env.unwrapped.region_length

            # Assuming action is a NumPy array of integers
            target_indices_to_chase = np.array(target_indices_to_chase)

            # Initialize the target_positions array with zeros
            target_positions = np.zeros((len(target_indices_to_chase), 2), dtype=np.float32)

            target_positions = self.env.unwrapped.target_pos[
                                   target_indices_to_chase] / self.env.unwrapped.region_length

            relative_positions = target_positions - herder_positions
            # Stack relative positions and target positions to form the batch
            batch_inputs = np.hstack((relative_positions, target_positions)).astype(np.float32)

            # Convert the batch inputs to a tensor
            batch_tensor = torch.tensor(batch_inputs)

            # Get batched actions from the neural network
            batched_herder_actions = self.model.get_action_mean(batch_tensor).cpu().numpy()

            # Call the episode_step function of the original environment
            obs_unw, reward_unw, terminated, truncated, info = self.env.step(batched_herder_actions)
            cumulative_reward += reward_unw

            self.obs_batch[step] = obs_unw

            if terminated or truncated:
                break

        info = self._get_info(info)

        # Return the processed observations
        obs = self._process_observations()
        reward = cumulative_reward

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed_obs = self._process_observations()
        return processed_obs, info

    def _process_observations(self):
        """
        Processes the observations to include:
        - Own position
        - Position of closest herder
        - Positions of 5 closest targets

        Returns:
            processed_obs (np.ndarray): Array of shape (num_herders, 14).
            closest_target_indices (np.ndarray): Indices of the 5 closest targets for each herder.
        """
        herder_positions = self.env.unwrapped.herder_pos  # Shape: (num_herders, 2)
        target_positions = self.env.unwrapped.target_pos  # Shape: (num_targets, 2)
        num_herders = self.env.unwrapped.num_herders
        num_targets = self.env.unwrapped.num_targets

        # Normalize positions
        herder_positions_normalized = herder_positions / self.env.unwrapped.region_length
        target_positions_normalized = target_positions / self.env.unwrapped.region_length

        # Compute distances between herders (excluding self)
        distances_between_herders = np.linalg.norm(
            herder_positions_normalized[:, np.newaxis, :] - herder_positions_normalized[np.newaxis, :, :],
            axis=2
        )
        np.fill_diagonal(distances_between_herders, np.inf)  # Exclude self

        # Find the closest herder for each herder
        closest_herder_indices = np.argmin(distances_between_herders, axis=1)
        closest_herder_positions = herder_positions_normalized[closest_herder_indices]

        # Compute distances from herders to targets
        distances_to_targets = np.linalg.norm(
            herder_positions_normalized[:, np.newaxis, :] - target_positions_normalized[np.newaxis, :, :],
            axis=2
        )

        # Get indices of the 5 closest targets for each herder
        closest_target_indices = np.argsort(distances_to_targets, axis=1)[:, :self.num_closest_targets]

        # Sort closest_target_indices to maintain original order
        self.closest_targets_indices = np.sort(closest_target_indices, axis=1)

        # Get positions of the 5 closest targets in the original order
        closest_targets = target_positions_normalized[self.closest_targets_indices]

        # Flatten the closest target positions
        closest_targets_flat = closest_targets.reshape(num_herders, -1)  # Shape: (num_herders, 10)

        # Concatenate own position, closest herder position, and closest target positions
        processed_obs = np.concatenate(
            [herder_positions_normalized, closest_herder_positions, closest_targets_flat],
            axis=1
        )  # Shape: (num_herders, 14)

        return processed_obs

    def _get_info(self, info):
        info['obs_batch'] = self.obs_batch
        return info
