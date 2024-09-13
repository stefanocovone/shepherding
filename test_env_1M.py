import gymnasium as gym
from shepherding.wrappers import LowLevelPPOPolicy
from shepherding.utils.control_rules import select_targets, select_targets_learning
from shepherding.utils import ActorCriticDiscrete
from gymnasium.wrappers import RecordVideo

use_learning = True

parameters = {
    'num_herders': 1,
    'num_targets': 20,
    'num_targets_min': 2,
    'num_targets_max': 7,
    'noise_strength': .1,
    'rho_g': 5,
    'region_length': 50,
    'xi': 2000,
    'dt': 0.05,
    'k_T': 3,
    'k_rep': 100,
    'simulation_dt': 0.001,
    'solver': 'Euler',
}
env = gym.make('Shepherding-v0', render_mode='rgb_array', parameters=parameters, rand_target=False)
env._max_episode_steps = 2000
env = RecordVideo(env, video_folder="videos")
env = LowLevelPPOPolicy(env, 20)

model = ActorCriticDiscrete.ActorCritic()

# Run the simulation for a certain number of steps
truncated = False
terminated = False

for episode in range(1, 1 + 1):
    # Reset the environment to get the initial observation
    observation, info = env.reset(seed=1)
    action = env.action_space.sample()  # Example action
    step = 0
    cum_reward = 0
    truncated = False
    terminated = False

    while not (terminated or truncated):

        step += 1
        if use_learning:
            action = select_targets_learning(env, observation, model)
        else:
            action = select_targets(env, observation)
        # action = env.action_space.sample()  # Example action
        # Take a episode_step in the environment by applying the chosen action
        observation, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward

    print("episode: ", episode, "reward: ", cum_reward)

# Close the environment (optional)
env.close()
