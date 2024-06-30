import numpy as np
import matplotlib.pyplot as plt
import os

# Define the file path for the precomputed average chi array
use_learning = True
output_dir = 'herdability_results'
if use_learning:
    output_file = os.path.join(output_dir, 'chi_noise_learning.npy')
else:
    output_file = os.path.join(output_dir, 'chi_noise_rule.npy')

# Load the precomputed average chi array
average_chi_array = np.load(output_file)

# Define the noise levels and targets range
noise_levels = np.arange(-2, 0.1, 0.1)
targets_range = range(1, 21)

# Define the specific ticks for targets
specific_ticks = [1, 5, 10, 15, 20, 25]

# Create a heatmap plot
plt.figure(figsize=(7, 5))
plt.imshow(average_chi_array, origin='lower', cmap='jet', aspect='auto',
           extent=[min(targets_range), max(targets_range), min(noise_levels), max(noise_levels)], interpolation='bilinear')
plt.colorbar(label=r'$\chi$')

# Set plot labels and titles
plt.xlabel(r'$M$', fontsize=16)
plt.ylabel(r'$\log_{10}D$', fontsize=16)
plt.title('Single-Agent Herdability')

# Set specific ticks and labels for y-axis (noise levels)
noise_ticks = np.arange(-2, 0.1, 0.5)
noise_labels = [f'$10^{{{tick}}}$' for tick in noise_ticks]
plt.yticks(noise_ticks, noise_ticks)

# Set x-axis ticks
plt.xticks(ticks=specific_ticks, labels=specific_ticks)

# Add a white contour line at chi = 0.99
contour = plt.contour(average_chi_array, levels=[0.99], colors='white', linewidths=2,
                      extent=[min(targets_range), max(targets_range), min(noise_levels), max(noise_levels)])
plt.clabel(contour, fmt='%1.1f', colors='white', fontsize=12)

# Save the heatmap
heatmap_file = 'herdability_noise_heatmap.png'
plt.savefig(heatmap_file)
print(f"Heatmap saved to {heatmap_file}")

# Display the heatmap
plt.show()
