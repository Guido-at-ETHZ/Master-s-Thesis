import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Logistic function (sigmoid)
def logistic(x, L, k, x0):
    """
    Logistic function parameters:
    L: the curve's maximum value
    k: the logistic growth rate or steepness of the curve
    x0: the x-value of the sigmoid's midpoint
    """
    return L / (1 + np.exp(-k * (x - x0)))

# First set of points
x1 = np.array([3, 6, 10])
y1 = np.array([0.09, 0.20, 0.45])
y1_err = np.array([0.002, 0.004, 0.005])

# Second set of points
x2 = np.array([10, 14, 18])
y2 = np.array([0.45, 0.65, 0.91])
y2_err = np.array([0.005, 0.003, 0.003])

# Fit sigmoid for first set of points
popt1, pcov1 = curve_fit(logistic, x1, y1,
                         p0=[1, 1, 6],  # Initial guess for [L, k, x0]
                         sigma=y1_err,  # Measurement errors
                         absolute_sigma=True)

# Fit sigmoid for second set of points
popt2, pcov2 = curve_fit(logistic, x2, y2,
                         p0=[1, 1, 14],  # Initial guess for [L, k, x0]
                         sigma=y2_err,  # Measurement errors
                         absolute_sigma=True)

# Generate smooth curves for plotting
x1_smooth = np.linspace(0, 15, 100)
x2_smooth = np.linspace(5, 25, 100)

# Plot results
plt.figure(figsize=(10, 6))

# First sigmoid
plt.errorbar(x1, y1, yerr=y1_err, fmt='ro', label='First Set Data Points')
plt.plot(x1_smooth, logistic(x1_smooth, *popt1), 'r-', label='First Sigmoid Fit')

# Second sigmoid
plt.errorbar(x2, y2, yerr=y2_err, fmt='bo', label='Second Set Data Points')
plt.plot(x2_smooth, logistic(x2_smooth, *popt2), 'b-', label='Second Sigmoid Fit')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sigmoid Fits with Measurement Errors')
plt.legend()
plt.grid(True)
plt.show()

# Print fit parameters with their uncertainties
def print_fit_params(popt, pcov, param_names):
    perr = np.sqrt(np.diag(pcov))
    for i, (param, value, error) in enumerate(zip(param_names, popt, perr)):
        print(f"{param}: {value:.4f} ± {error:.4f}")

print("\nFirst Sigmoid Fit Parameters:")
print_fit_params(popt1, pcov1, ['L', 'k', 'x0'])

print("\nSecond Sigmoid Fit Parameters:")
print_fit_params(popt2, pcov2, ['L', 'k', 'x0'])

# Goodness of fit (reduced chi-squared)
def reduced_chi_squared(y_obs, y_pred, y_err, num_params):
    chi_sq = np.sum(((y_obs - y_pred) / y_err)**2)
    dof = len(y_obs) - num_params
    return chi_sq / dof

print("\nReduced Chi-Squared:")
print(f"First Sigmoid: {reduced_chi_squared(y1, logistic(x1, *popt1), y1_err, 3):.4f}")
print(f"Second Sigmoid: {reduced_chi_squared(y2, logistic(x2, *popt2), y2_err, 3):.4f}")