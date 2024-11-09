import numpy as np
import matplotlib.pyplot as plt

# Load the chi_rule and chi_learning matrices
file_path_rule = 'herdability_results/chi_rule.npy'
file_path_learning = 'herdability_results/chi_learning.npy'

chi_rule = np.load(file_path_rule)
chi_learning = np.load(file_path_learning)

# Define the specific ticks for herders and targets
specific_ticks = [2, 5, 10, 15, 20, 25]

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot chi_rule in the first subplot with interpolation
im1 = axs[0].imshow(chi_rule, origin='lower', cmap='jet', aspect='equal', interpolation='bilinear')
axs[0].set_title(r'Heuristic')
axs[0].set_xticks(ticks=[x-2 for x in specific_ticks])  # Adjust the ticks to match the range index
axs[0].set_xticklabels(specific_ticks)
axs[0].set_yticks(ticks=[x-2 for x in specific_ticks])  # Adjust the ticks to match the range index
axs[0].set_yticklabels(specific_ticks)
axs[0].set_xlabel(r'$\sqrt{M}$', fontsize=16)
axs[0].set_ylabel(r'$N$', fontsize=16)

# Extract the contour line data for chi_rule using allsegs
contour1_coords = np.vstack(axs[0].contour(chi_rule, levels=[0.99], colors='none').allsegs[0])
x_contour1, y_contour1 = contour1_coords[:, 0], contour1_coords[:, 1]

# Fit a linear regression to the contour line data for chi_rule
m_rule, q_rule = np.polyfit(x_contour1, y_contour1, 1)
axs[0].plot(x_contour1, m_rule * x_contour1 + q_rule, color='white', linestyle='-', linewidth=2,
            label=f'$N = {m_rule:.2f} \, \sqrt{{M}} + {q_rule:.2f}$')
axs[0].legend()

# Plot chi_learning in the second subplot with interpolation
im2 = axs[1].imshow(chi_learning, origin='lower', cmap='jet', aspect='equal', interpolation='bilinear')
axs[1].set_title(r'Hybrid')
axs[1].set_xticks(ticks=[x-2 for x in specific_ticks])  # Adjust the ticks to match the range index
axs[1].set_xticklabels(specific_ticks)
axs[1].set_yticks(ticks=[x-2 for x in specific_ticks])  # Adjust the ticks to match the range index
axs[1].set_yticklabels(specific_ticks)
axs[1].set_xlabel(r'$\sqrt{M}$', fontsize=16)
axs[1].set_ylabel(r'$N$', fontsize=16)

# Extract the contour line data for chi_learning using allsegs
contour2_coords = np.vstack(axs[1].contour(chi_learning, levels=[0.99], colors='none').allsegs[0])
x_contour2, y_contour2 = contour2_coords[:, 0], contour2_coords[:, 1]

# Fit a linear regression to the contour line data for chi_learning
m_learning, q_learning = np.polyfit(x_contour2, y_contour2, 1)
axs[1].plot(x_contour2, m_learning * x_contour2 + q_learning, color='white', linestyle='-', linewidth=2,
            label=f'$N = {m_learning:.2f} \, \sqrt{{M}} + {q_learning:.2f}$')
axs[1].legend()

# Create a common colorbar
cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label(r'$\chi$')

# Adjust layout to avoid overlap
plt.subplots_adjust(wspace=0.2, hspace=0.2, right=0.85)  # Reduced wspace and hspace

# Save the figure if needed
plt.savefig('comparison_plot.pdf', format='pdf', dpi=300)

# Display the plot
plt.show()
