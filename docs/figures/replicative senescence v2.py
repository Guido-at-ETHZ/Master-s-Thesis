import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Logistic function (sigmoid) with numerical stability improvements
def logistic(x, L, k, x0):
    """
    Logistic function parameters:
    L: the curve's maximum value
    k: the logistic growth rate or steepness of the curve
    x0: the x-value of the sigmoid's midpoint
    """
    z = np.clip(-k * (x - x0), -500, 500)  # Prevent overflow
    return L / (1 + np.exp(z))

# First set of points
x1 = np.array([4, 15, 31])  # PDL values
y1 = np.array([0.09, 0.20, 0.45])  # Percentage values
y1_err = np.array([0.002, 0.004, 0.005])

# Second set of points
x2 = np.array([31, 46, 61])  # PDL values
y2 = np.array([0.45, 0.65, 0.91])  # Percentage values
y2_err = np.array([0.005, 0.003, 0.003])

# Third set of points (control data converted from days to PDL)
x3 = np.array([8.7, 23.3, 29.1, 35.2])  # Time converted to PDL
y3 = np.array([0.01, 0.04, 0.17, 0.30])  # Converting percentages to decimal
y3_err = np.array([0.015, 0.020, 0.015, 0.015])  # Standard deviations in decimal

# Fit sigmoid for first set of points
popt1, pcov1 = curve_fit(logistic, x1, y1,
                        p0=[0.5, 0.05, 20],
                        sigma=y1_err,
                        absolute_sigma=True,
                        bounds=([0, 0, 0], [2, 1, 100]))

# Fit sigmoid for second set of points
popt2, pcov2 = curve_fit(logistic, x2, y2,
                        p0=[1.0, 0.05, 50],
                        sigma=y2_err,
                        absolute_sigma=True,
                        bounds=([0, 0, 0], [5, 1, 200]))

# Fit sigmoid for third set of points (control)
popt3, pcov3 = curve_fit(logistic, x3, y3,
                        p0=[0.5, 0.05, 25],
                        sigma=y3_err,
                        absolute_sigma=True,
                        bounds=([0, 0, 0], [2, 1, 100]))

# Generate smooth curves for plotting
x1_smooth = np.linspace(0, 40, 100)
x2_smooth = np.linspace(25, 65, 100)
x3_smooth = np.linspace(0, 40, 100)

plt.figure(figsize=(10, 6))

# First sigmoid
plt.errorbar(x1, y1*100, yerr=y1_err*100, fmt='ro', label='First Set Data Points')
plt.plot(x1_smooth, logistic(x1_smooth, *popt1)*100, 'r-', label='First Sigmoid Fit')

# Second sigmoid
plt.errorbar(x2, y2*100, yerr=y2_err*100, fmt='bo', label='Second Set Data Points')
plt.plot(x2_smooth, logistic(x2_smooth, *popt2)*100, 'b-', label='Second Sigmoid Fit')

# Third sigmoid (control)
plt.errorbar(x3, y3*100, yerr=y3_err*100, fmt='gs', label='Control Data Points')
plt.plot(x3_smooth, logistic(x3_smooth, *popt3)*100, 'g-', label='Control Sigmoid Fit')

plt.xlabel('Population Doubling Level (PDL)')
plt.ylabel('Percentage (%)')
plt.title('Sigmoid Fits with Measurement Errors')
plt.legend()
plt.grid(True)
plt.savefig('sigmoid_plot.png', dpi=300, bbox_inches='tight')
plt.ylim(0, 100)

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

print("\nControl Sigmoid Fit Parameters:")
print_fit_params(popt3, pcov3, ['L', 'k', 'x0'])

