import numpy as np
import matplotlib.pyplot as plt

def piecewise_function(x, k_T, k_rep, lambda_, sigma):
    abs_x = np.abs(x)
    if abs_x <= 1:
        return (k_T * (lambda_ - abs_x) + k_rep * (sigma - abs_x)) * np.sign(x)
    elif 1 < abs_x <= 2.5:
        return k_T * (lambda_ - abs_x) * np.sign(x)
    else:
        return 0

# Parameters
k_T = 3.0
k_rep = 100.0
lambda_ = 2.5
sigma = 1

# Create a range of x values
x_vals = np.linspace(-3, 3, 400)
y_vals = np.array([piecewise_function(x, k_T, k_rep, lambda_, sigma) for x in x_vals])

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label=r'repulsion')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title('Cumulative Target-Herder repulsion')
plt.xlabel('x')
plt.ylabel(r'$\Theta (x)$')
plt.grid(True)
plt.show()
