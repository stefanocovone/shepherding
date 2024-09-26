import numpy as np

def select_targets(herder_pos, target_pos, num_herders, num_targets):

    # Assuming observation is a numpy array and env.num_herders is the number of herders
    # xi is the distance threshold, v_h is the herder velocity scalar, alpha and delta are given parameters
    xi = 1000

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
    expanded_absolute_distances = np.tile(absolute_distances, (num_herders, 1))  # Shape (N, M)

    # Apply the mask to get valid distances only
    valid_absolute_distances = np.where(mask, expanded_absolute_distances, -np.inf)  # Shape (N, M)

    # Find the index of the target with the maximum absolute distance from the origin for each herder
    selected_target_indices = np.argmax(valid_absolute_distances, axis=1)  # Shape (N,)

    # Create a mask to identify herders that have no valid targets
    no_valid_target_mask = np.all(~mask, axis=1)

    # Replace invalid indices with -1 (indicating no target)
    selected_target_indices = np.where(no_valid_target_mask, -1, selected_target_indices)

    return selected_target_indices




def herder_actions(herder_pos, target_pos, num_herders, num_targets, v_h, rho_g, delta, alpha):

    # Assuming observation is a numpy array and env.num_herders is the number of herders
    # xi is the distance threshold, v_h is the herder velocity scalar, alpha and delta are given parameters
    xi = 1000

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
    expanded_absolute_distances = np.tile(absolute_distances, (num_herders, 1))  # Shape (N, M)

    # Apply the mask to get valid distances only
    valid_absolute_distances = np.where(mask, expanded_absolute_distances, -np.inf)  # Shape (N, M)

    # Find the index of the target with the maximum absolute distance from the origin for each herder
    selected_target_indices = np.argmax(valid_absolute_distances, axis=1)  # Shape (N,)

    # Create a mask to identify herders that have no valid targets
    no_valid_target_mask = np.all(~mask, axis=1)

    # Replace invalid indices with -1 (indicating no target)
    selected_target_indices = np.where(no_valid_target_mask, -1, selected_target_indices)

    # Create a vector (N, 2) to store the absolute position of the selected target for each herder
    selected_target_positions = np.zeros((num_herders, 2))
    selected_target_positions[~no_valid_target_mask] = target_pos[selected_target_indices[~no_valid_target_mask]]

    # Calculate unit vectors for herders and selected targets
    herder_unit_vectors = herder_pos / np.linalg.norm(herder_pos, axis=1, keepdims=True)  # Shape (N, 2)
    selected_target_unit_vectors = np.zeros((num_herders, 2))
    selected_target_unit_vectors[~no_valid_target_mask] = (
            target_pos[selected_target_indices[~no_valid_target_mask]] / np.linalg.norm(
        target_pos[selected_target_indices[~no_valid_target_mask]], axis=1, keepdims=True
    )
    )

    # Calculate actions for each herder
    actions = np.zeros((num_herders, 2))
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






import numpy as np
import torch

def select_targets_learning(herder_pos, target_pos, num_herders, num_targets, model):
    # Assuming observation is a numpy array and env.num_herders is the number of herders
    # xi is the distance threshold, v_h is the herder velocity scalar, alpha and delta are given parameters
    xi = 1000


    if num_targets < 7:
        # Pad target_pos with [0, 0] to ensure there are at least 7 targets
        padding = np.zeros((7 - num_targets, 2))
        target_pos = np.vstack((target_pos, padding))

    # Calculate the distances from each herder to each target
    distances = np.linalg.norm(herder_pos[:, np.newaxis, :] - target_pos[np.newaxis, :, :], axis=2)  # Shape (N, M)

    # Get the indices of the 7 closest targets for each herder
    closest_indices = np.argsort(distances, axis=1)[:, :7]  # Shape (N, 7)

    # Sort the indices of the closest targets to match the original order
    sorted_indices = np.sort(closest_indices, axis=1)

    # Prepare the input vector for the model
    herders_repeated = np.repeat(herder_pos, 7, axis=0).reshape(-1, 2)  # Shape (N*7, 2)
    closest_targets = target_pos[sorted_indices].reshape(num_herders, -1)  # Shape (N, 7, 2) -> Shape (N, 14)

    inputs = np.hstack((herder_pos, closest_targets))  # Shape (N, 16)

    # inputs = np.vstack((herder_pos, target_pos)).reshape(-1).reshape(1,18)  # Shape (N+M, 2)
    inputs = np.vstack((herder_pos, target_pos)).flatten()  # Shape (N+M, 2)
    inputs = (inputs[np.newaxis, :] / 5)

    # Convert inputs to tensor
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

    # Get actions from the model
    with torch.no_grad():
        selected_targets = model.get_action(inputs_tensor)

    selected_targets = selected_targets.cpu().numpy()  # Shape (N, 1)

    # Map the selected targets onto the sorted_indices
    final_selected_targets = np.zeros(num_herders, dtype=int)
    for i in range(num_herders):
        final_selected_targets[i] = sorted_indices[i][selected_targets[0][i]]

    return final_selected_targets

