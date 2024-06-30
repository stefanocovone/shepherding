import gymnasium as gym
from shepherding.wrappers import TerminateWhenSuccessful, LowLevelPPOPolicy
from shepherding.utils.control_rules import herder_actions, select_targets
import numpy as np
import os

# Base parameters for the environment
base_parameters = {
    'noise_strength': 1,
    'rho_g': 5,
    'region_length': 50,
    'xi': 1000,
    'dt': 0.05,
    'k_T': 3,
    'k_rep': 100,
    'simulation_dt': 0.01,
    'solver': 'Euler',
}

use_learning = True

# Range of herders and targets
herders_range = range(2, 26)
targets_range = range(2, 26)

# Initialize a numpy array to store the average chi values
average_chi_array = np.zeros((len(herders_range), len(targets_range)))

for herder_idx, num_herders in enumerate(herders_range):
    for target_idx, num_targets in enumerate(targets_range):
        # Update parameters with the current number of herders and targets
        parameters = base_parameters.copy()
        parameters['num_herders'] = num_herders
        parameters['num_targets'] = num_targets**2

        # Create the environment with the updated parameters
        env = gym.make('Shepherding-v0', render_mode='rgb_array', parameters=parameters)
        env._max_episode_steps = 1000*20
        env = TerminateWhenSuccessful(env)
        if use_learning:
            env = LowLevelPPOPolicy(env, update_frequency=1)

        # List to store chi values for the current combination of herders and targets
        chi_list = []

        # Run 50 simulations
        for episode in range(1, 1 + 1):
            # Reset the environment to get the initial observation
            observation, info = env.reset(seed=episode)
            step = 0
            cum_reward = 0
            chi = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                step += 1
                # Compute herders' actions according to the desired policy
                if use_learning:
                    action = select_targets(env, observation)
                else:
                    action = herder_actions(env, observation)
                # Take a step in the environment by applying the chosen action
                observation, reward, terminated, truncated, info = env.step(action)
                cum_reward += reward
                chi = max(info.get("fraction_captured_targets", 0), chi)

            # Store the chi value for this episode
            chi_list.append(chi)

        # Calculate the average chi value for the current configuration
        average_chi = np.mean(chi_list)

        # Store the average chi value in the numpy array
        average_chi_array[herder_idx, target_idx] = average_chi

        print(f"Herders: {num_herders}, Targets: {num_targets}, Chi: {average_chi}")

        # Close the environment (optional)
        env.close()

# Create the directory if it doesn't exist
output_dir = 'herdability_results'
os.makedirs(output_dir, exist_ok=True)

# Save the average chi array to a file
if use_learning:
    output_file = os.path.join(output_dir, 'chi_learning.npy')
else:
    output_file = os.path.join(output_dir, 'chi_rule.npy')
np.save(output_file, average_chi_array)

print(f"Average chi array saved to {output_file}")
