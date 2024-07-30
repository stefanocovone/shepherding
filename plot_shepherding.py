import matplotlib.pyplot as plt
import numpy as np

# Create figure and axis
fig, ax = plt.subplots()

# Plot the main circle
main_circle = plt.Circle((0, 0), .5, color='blue', fill=False, linewidth=3)
ax.add_artist(main_circle)

# Plot the inner circle (yellow)
inner_circle = plt.Circle((0, 0), 0.5, color='yellow', fill=True, alpha=0.5)
ax.add_artist(inner_circle)

# Plot the outer circle (green)
outer_circle = plt.Circle((0, 0), 1.3, color='lightgreen', fill=True, alpha=0.5)
ax.add_artist(outer_circle)

# Plotting points Hi and Ta
plt.plot(-0.7, 0.7, 'bD', markersize=10, label='$H_i$')
plt.plot(1, 0.3, 'mp', markersize=10, label='$T_a$')

# Adding radius lines and angles
ax.plot([0, -0.7], [0, 0.7], 'k--', linewidth=1)  # ρ_i line
ax.plot([0, 1], [0, 0.3], 'k-', linewidth=1)  # r_a line

# Adding radius labels
ax.text(-0.35, 0.35, '$\\rho_i$', fontsize=12, verticalalignment='bottom')
ax.text(0.5, 0.15, '$r_a$', fontsize=12, verticalalignment='bottom')

# Adding angle labels
ax.text(0.3, 0.1, '$\\varphi_a$', fontsize=12, verticalalignment='bottom')
ax.text(-0.2, 0.4, '$\\theta_i$', fontsize=12, verticalalignment='bottom')

# Adding O point and labels for circles
ax.plot(0, 0, 'ko')  # Origin O
ax.text(0.05, 0.05, '$O$', fontsize=12, verticalalignment='bottom')

# Adding text for regions
ax.text(-0.15, -0.15, '$\\Omega_G$', fontsize=12, verticalalignment='bottom')
ax.text(.7, .7, '$\\Omega_{0}$', fontsize=12, verticalalignment='bottom')

# Axis settings
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.axhline(0, color='black',linewidth=0.5)
ax.axvline(0, color='black',linewidth=0.5)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])

# Display plot
plt.show()
