import gymnasium as gym
import numpy as np
from shepherding.wrappers import SingleAgentReward, DeterministicReset

import cProfile
import pstats


def herder_actions(env, observation, random=False):
    if random:
        actions = np.random.uniform(-1, 1, size=env.action_space.shape)
    else:

        # Assuming observation is a numpy array and env.num_herders is the number of herders
        # xi is the distance threshold, v_h is the herder velocity scalar, alpha and delta are given parameters
        xi = env.xi
        v_h = env.herder_max_vel
        alpha = env.alpha
        delta = env.lmbda / 2
        rho_g = env.rho_g

        # Extract herder and target positions from the observation
        herder_pos = observation[0:env.num_herders, :]  # Shape (N, 2)
        target_pos = observation[env.num_herders:, :]  # Shape (M, 2)

        # Expand dimensions of herder_pos and target_pos to enable broadcasting
        herder_pos_exp = herder_pos[:, np.newaxis, :]  # Shape (N, 1, 2)
        target_pos_exp = target_pos[np.newaxis, :, :]  # Shape (1, M, 2)

        # Calculate the relative positions
        relative_positions = target_pos_exp - herder_pos_exp  # Shape (N, M, 2)

        # Compute the Euclidean distances between herders and targets
        distances = np.linalg.norm(relative_positions, axis=2)  # Shape (N, M)

        # Find the index of the closest herder for each target
        closest_herders = np.argmin(distances, axis=0)  # Shape (M,)

        # Create a boolean mask where each target is only considered if it's closer to the current herder
        closest_mask = np.zeros_like(distances, dtype=bool)
        np.put_along_axis(closest_mask, closest_herders[np.newaxis, :], True, axis=0)

        # Create a boolean mask where distances are less than xi and the herder is the closest one
        mask = (distances < xi) & closest_mask  # Shape (N, M)

        # Calculate the absolute distances from the origin for the targets
        absolute_distances = np.linalg.norm(target_pos, axis=1)  # Shape (M,)

        # Use broadcasting to expand the absolute distances to match the shape of the mask
        expanded_absolute_distances = np.tile(absolute_distances, (env.num_herders, 1))  # Shape (N, M)

        # Apply the mask to get valid distances only
        valid_absolute_distances = np.where(mask, expanded_absolute_distances, -np.inf)  # Shape (N, M)

        # Find the index of the target with the maximum absolute distance from the origin for each herder
        selected_target_indices = np.argmax(valid_absolute_distances, axis=1)  # Shape (N,)

        # Create a mask to identify herders that have no valid targets
        no_valid_target_mask = np.all(~mask, axis=1)

        # Replace invalid indices with -1 (indicating no target)
        selected_target_indices = np.where(no_valid_target_mask, -1, selected_target_indices)

        # Create a vector (N, 2) to store the absolute position of the selected target for each herder
        selected_target_positions = np.zeros((env.num_herders, 2))
        selected_target_positions[~no_valid_target_mask] = target_pos[selected_target_indices[~no_valid_target_mask]]

        # Calculate unit vectors for herders and selected targets
        herder_unit_vectors = herder_pos / np.linalg.norm(herder_pos, axis=1, keepdims=True)  # Shape (N, 2)
        selected_target_unit_vectors = np.zeros((env.num_herders, 2))
        selected_target_unit_vectors[~no_valid_target_mask] = (
                target_pos[selected_target_indices[~no_valid_target_mask]] / np.linalg.norm(
            target_pos[selected_target_indices[~no_valid_target_mask]], axis=1, keepdims=True
        )
        )

        # Calculate actions for each herder
        actions = np.zeros((env.num_herders, 2))
        herder_abs_distances = np.linalg.norm(herder_pos, axis=1)  # Absolute distances of herders from the origin

        # If no target is selected and the herder's distance is less than rho_g, action is zero
        # Otherwise, action is v_h * herder_unit_vector
        no_target_selected = no_valid_target_mask & (herder_abs_distances < rho_g)
        actions[no_valid_target_mask & ~no_target_selected] = - v_h * herder_unit_vectors[
            no_valid_target_mask & ~no_target_selected]

        # If a target is selected, action is
        # alpha * (herder_pos - (selected_target_pos + delta * selected_target_unit_vector))
        actions[~no_valid_target_mask] = - alpha * (
                herder_pos[~no_valid_target_mask] - (
                selected_target_positions[~no_valid_target_mask] +
                delta * selected_target_unit_vectors[~no_valid_target_mask]
        )
        )

    return actions


profiler = cProfile.Profile()
profiler.enable()

parameters = {
    'num_herders': 10,
    'num_targets': 50,
    'noise_strength': 1,
    'rho_g': 5,
    'region_length': 60,
    'xi': 1000,
    'dt': 0.05,
    'k_T': 3,
    'k_rep': 100,
}
env = gym.make('Shepherding-v0', render_mode='rgb_array', parameters=parameters)
env._max_episode_steps = 1000
# env = SingleAgentReward(env)
# env = DeterministicReset(env)

# Run the simulation for a certain number of steps
truncated = False
terminated = False

for episode in range(1, 1 + 1):
    # Reset the environment to get the initial observation
    observation, info = env.reset(seed=1)
    step = 0
    cum_reward = 0
    truncated = False
    terminated = False
    while not (terminated or truncated):
        step += 1
        # Choose a random action (here, randomly setting velocities for herders)
        action = herder_actions(env, observation, random=False)
        # Take a episode_step in the environment by applying the chosen action
        observation, reward, terminated, truncated, _ = env.step(action)
        # print(step)
        cum_reward += reward

    print("episode: ", episode, "reward: ", cum_reward)

# Close the environment (optional)
env.close()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('tottime').print_stats(2)
