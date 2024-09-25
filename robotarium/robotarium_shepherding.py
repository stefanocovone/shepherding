# Imports needed by Robotarium
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

# Imports needed by the learning-based control
# from shepherding.utils import *
from robotarium.utils import *
from shepherding.utils import ActorCritic, ActorCriticMultiDiscrete
import torch


def _random_positions(num_agents):
    radius = np.random.uniform(0.6, 0.9, num_agents)
    angle = np.random.uniform(0, 2 * np.pi, num_agents)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    alpha = np.zeros_like(x)
    position = np.vstack([x, y, alpha])
    return position

# Define number of agents
num_herders = 2
num_targets = 4
num_agents = num_targets + num_herders

# Define arena parameters (must be scaled)
scaling_rate = 1 / 50 * 5  # robotarium_length / virtual_length * scaling_factor
rho_g = 5 * scaling_rate
arena_length = 1.1

# Define target parameters
noise_strength = .1
dt = 1
k_T = 3
lmbda = 2.5 * scaling_rate




herder_max_vel = 8

# define episode length
num_steps = 5000

# Set initial conditions
# initial_conditions = np.array(np.asmatrix('1 0.5 -0.5 0 0.28 0.6 -.2; 0.8 -0.3 -0.75 0.1 0.34 0.6 -.2; 0 0 0 0 0 0 0'))
# initial_conditions =generate_initial_conditions(N=num_agents, width=2, height=2, spacing=0.4)
initial_conditions = _random_positions(num_agents)
# initial_conditions = np.array(np.asmatrix('0.5 0.5 -0.5 0.7 -0.6 0.9 -0.9; 0.5 -0.5 0.5 -0.7 -0.6 0.5 0.5; 0 0 0 0 0 0 0'))

# Instantiate Robotarium object
r = robotarium.Robotarium(number_of_robots=num_agents, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=False)

# Create barrier certificates to avoid collision
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary(safety_radius=0.15, boundary_points=np.array([-1.6, 1.6, -1, 1]))

_, uni_to_si_states = create_si_to_uni_mapping()

# Create mapping from single integrator velocity commands to unicycle velocity commands
si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

# define x initially
x = r.get_poses()
x_si = uni_to_si_states(x)

# Plot the goal region
goal_marker_size_m = rho_g
marker_size_goal = determine_marker_size(r,goal_marker_size_m)
line_width = 5
goal_markers = r.axes.scatter(0, 0, s=marker_size_goal, marker='o', facecolors='none',edgecolors='blue',linewidth=line_width,zorder=-2)

# Plot the robot markers
robot_marker_size_m = 0.12
marker_size_robot = determine_marker_size(r, robot_marker_size_m)
target_markers = [r.axes.scatter(x[0,ii], x[1,ii], s=marker_size_robot, marker='o', facecolors='none',edgecolors='green',linewidth=line_width)
for ii in range(num_targets)]
herder_markers = [r.axes.scatter(x[0,ii], x[1,ii], s=marker_size_robot, marker='o', facecolors='none',edgecolors='red',linewidth=line_width)
for ii in range(num_targets, num_targets + num_herders)]


# Define the Actor Critic model
model = ActorCritic.ActorCritic()
modelL2 = ActorCriticMultiDiscrete.ActorCritic()


# Data saving
herder_positions_save = np.zeros((num_herders, 2, num_steps))
target_positions_save = np.zeros((num_targets, 2, num_steps))

r.step()

for step in range(num_steps):

    # Get poses of agents
    x = r.get_poses()
    x_si = uni_to_si_states(x)
    agent_pos = x_si.transpose()

    # Update Robot Marker Plotted Visualization
    for i in range(num_targets):
        target_markers[i].set_offsets(x[:2, i].T)
    for i in range(num_herders):
        herder_markers[i].set_offsets(x[:2, i + num_targets].T)

    # compute agents position
    target_pos = agent_pos[0:num_targets]
    herder_pos = agent_pos[num_targets:num_agents]

    # target_positions_save = target_pos[...,step]
    # herder_positions_save = herder_pos[..., step]



    # TARGET DYNAMICS
    noise = np.sqrt(2 * noise_strength * dt) * np.random.normal(size=(num_targets, 2))

    # repulsion force computation
    differences = herder_pos[:, np.newaxis, :] - target_pos[np.newaxis, :, :]
    distances = np.linalg.norm(differences, axis=2)
    nearby_agents = distances < lmbda
    nearby_differences = np.where(nearby_agents[:, :, np.newaxis], differences, 0)
    distances_with_min = np.maximum(distances[:, :, np.newaxis], 1e-6)
    nearby_unit_vector = nearby_differences / distances_with_min
    repulsion = -k_T * np.sum((lmbda - distances[:, :, np.newaxis]) * nearby_unit_vector, axis=0)

    target_vel = (noise + repulsion) / 5



    # HERDER DYNAMICS


    # Target selection rule
    if (step % 50) == 0:
        # Rule-based target selection
        # selected_targets = select_targets(herder_pos, target_pos, num_herders, num_targets)
        # Learning-based target selection
        selected_targets = select_targets_learning(herder_pos, target_pos, num_herders, num_targets, modelL2)


    # Assuming action is a NumPy array of integers
    selected_targets = np.array(selected_targets)

    # Initialize the target_positions array with zeros
    target_positions = np.zeros((len(selected_targets), 2), dtype=np.float32)

    # Filter the action elements that are within the valid range
    valid_indices = selected_targets < num_targets

    # Only fetch positions for valid actions
    target_positions[valid_indices] = target_pos[
                                          selected_targets[valid_indices]]

    relative_positions = target_positions - herder_pos
    # Stack relative positions and target positions to form the batch
    batch_inputs = np.hstack((relative_positions, target_positions)).astype(np.float32)

    # Convert the batch inputs to a tensor
    batch_tensor = torch.tensor(batch_inputs /  5)

    # Get batched actions from the neural network
    actions = model.get_action_mean(batch_tensor).cpu().numpy()

    # RULE BASED CONTROL
    # actions_rule = herder_actions(herder_pos, target_pos, num_herders, num_targets, v_h=0.2, rho_g=rho_g, delta=lmbda/2, alpha=1)
    #
    # print("actions: ", actions)
    # print("actions rule: ", actions_rule)

    herder_vel = np.zeros((num_herders,2))

    herder_vel = actions / 8 / 5

    # Compute agents velocity
    dxi = np.concatenate((target_vel, herder_vel), axis=0)
    dxi = dxi.transpose()

    # Create safe control inputs (i.e., no collisions)
    dxi = si_barrier_cert(dxi, x_si)

    # Transform single integrator velocity commands to unicycle
    dxu = si_to_uni_dyn(dxi, x)


    # Set the velocities by mapping the single-integrator inputs to unciycle inputs
    r.set_velocities(np.arange(num_agents), dxu)
    # Iterate the simulation
    r.step()

# Save data locally as numpy
# np.save

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
