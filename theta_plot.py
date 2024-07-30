import numpy as np
import matplotlib.pyplot as plt


def Theta(r, x):
    mask = np.abs(x) <= r
    theta = np.zeros_like(x)
    theta[mask] = (r - np.abs(x[mask]))
    return theta


# Parameters
r = 1
x_range = np.linspace(0, 2, 400)
values = Theta(r, x_range)

# Plotting
plt.figure(figsize=(6, 4))
plt.plot(x_range, values)
plt.xlim([0, 1.2])
plt.ylim([-0.2, 1.2])
plt.title(r'Plot of $\mathbf{\Theta}(r, \mathbf{x})$')
plt.xlabel(r'$|x| / \lambda$')
plt.ylabel(r'$\mathbf{\Theta}$')
plt.grid(True)
# plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('theta_function_plot.eps', format='eps')
plt.show()
