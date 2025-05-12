import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# Experimental data from Table 1
tau_exp = np.array([15, 25, 45])
Amax_exp = np.array([1.4, 3.4, 5.7])

# Set the fixed threshold
fixed_threshold = 5.0


# Define the continuous Amax linear model function
def Amax_linear_model(tau, m):
    """Calculate Amax using the linear model with fixed threshold."""
    tau = np.asarray(tau)
    result = np.ones_like(tau, dtype=float)  # Initialize with baseline value 1
    above_threshold = tau >= fixed_threshold
    result[above_threshold] = 1 + m * (tau[above_threshold] - fixed_threshold)
    return result


# Define the fitting function for points above threshold
def linear_above_threshold(tau, m):
    """Function for fitting: only applies to points above threshold."""
    return 1 + m * (tau - fixed_threshold)


# Define a function for weighted linear regression with fixed intercept
def weighted_linear_regression(x, y, weights=None):
    """Perform weighted linear regression through a point with fixed_intercept=1."""
    # Shift x values by threshold and y values by 1 to force the line through (threshold, 1)
    x_shifted = x - fixed_threshold
    y_shifted = y - 1

    # Calculate weighted mean of x and y
    if weights is None:
        weights = np.ones_like(x)

    # Normalize weights
    weights = weights / np.sum(weights)

    # Calculate slope for regression through origin
    numerator = np.sum(weights * x_shifted * y_shifted)
    denominator = np.sum(weights * x_shifted * x_shifted)

    if denominator == 0:
        slope = 0
    else:
        slope = numerator / denominator

    # Calculate R-squared
    y_pred = 1 + slope * (x - fixed_threshold)
    residuals = y - y_pred
    ss_residuals = np.sum(weights * residuals * residuals)
    y_mean = np.sum(weights * y)
    ss_total = np.sum(weights * (y - y_mean) * (y - y_mean))

    if ss_total == 0:
        r_squared = 1.0  # Perfect fit
    else:
        r_squared = 1 - (ss_residuals / ss_total)

    # Calculate weighted MSE
    weighted_mse = np.sum(weights * residuals * residuals)

    return slope, r_squared, weighted_mse


# Function to explore different weight combinations and find best fit
def explore_weight_scenarios(tau, amax, num_scenarios=7):
    scenarios = []

    # Only include points above threshold
    mask = tau >= fixed_threshold
    tau_above = tau[mask]
    amax_above = amax[mask]

    # Scenario 1: Equal weights (unweighted)
    weights1 = np.ones_like(tau_above)
    m1, r2_1, mse_1 = weighted_linear_regression(tau_above, amax_above, weights1)
    scenarios.append({
        'name': 'Equal weights',
        'weights': weights1,
        'slope': m1,
        'r_squared': r2_1,
        'mse': mse_1
    })

    # Scenario 2: Weight by inverse of tau (give more weight to points closer to threshold)
    weights2 = 1.0 / tau_above
    weights2 = weights2 / np.sum(weights2)  # Normalize
    m2, r2_2, mse_2 = weighted_linear_regression(tau_above, amax_above, weights2)
    scenarios.append({
        'name': 'Weight by 1/τ',
        'weights': weights2,
        'slope': m2,
        'r_squared': r2_2,
        'mse': mse_2
    })

    # Scenario 3: Weight by tau (give more weight to points farther from threshold)
    weights3 = tau_above
    weights3 = weights3 / np.sum(weights3)  # Normalize
    m3, r2_3, mse_3 = weighted_linear_regression(tau_above, amax_above, weights3)
    scenarios.append({
        'name': 'Weight by τ',
        'weights': weights3,
        'slope': m3,
        'r_squared': r2_3,
        'mse': mse_3
    })

    # Scenario 4: Custom weights focusing on initial point
    weights4 = np.array([0.4, 0.4, 0.2]) if len(tau_above) == 3 else np.ones_like(tau_above)
    weights4 = weights4 / np.sum(weights4)  # Normalize
    m4, r2_4, mse_4 = weighted_linear_regression(tau_above, amax_above, weights4)
    scenarios.append({
        'name': 'Focus on initial points',
        'weights': weights4,
        'slope': m4,
        'r_squared': r2_4,
        'mse': mse_4
    })

    # Scenario 5: Custom weights focusing on endpoints
    weights5 = np.array([0.6, 0.3, 0.1]) if len(tau_above) == 3 else np.ones_like(tau_above)
    weights5 = weights5 / np.sum(weights5)  # Normalize
    m5, r2_5, mse_5 = weighted_linear_regression(tau_above, amax_above, weights5)
    scenarios.append({
        'name': 'Focus on first point',
        'weights': weights5,
        'slope': m5,
        'r_squared': r2_5,
        'mse': mse_5
    })

    # NEW Scenario 6: Emphasize 15 dyn/cm² point
    weights6 = np.ones_like(tau_above)
    # Find index of the 15 dyn/cm² point
    idx_15 = np.where(tau_above == 15)[0]
    if len(idx_15) > 0:
        weights6[idx_15[0]] = 5.0  # Give 5x weight to the 15 dyn/cm² point
    weights6 = weights6 / np.sum(weights6)  # Normalize
    m6, r2_6, mse_6 = weighted_linear_regression(tau_above, amax_above, weights6)
    scenarios.append({
        'name': 'Emphasize τ=15',
        'weights': weights6,
        'slope': m6,
        'r_squared': r2_6,
        'mse': mse_6
    })

    # NEW Scenario 7: Only use first and last points (15 and 45 dyn/cm²)
    weights7 = np.zeros_like(tau_above)
    # Find indices of the 15 and 45 dyn/cm² points
    idx_15 = np.where(tau_above == 15)[0]
    idx_45 = np.where(tau_above == 45)[0]
    if len(idx_15) > 0 and len(idx_45) > 0:
        weights7[idx_15[0]] = 1.0
        weights7[idx_45[0]] = 1.0
    weights7 = weights7 / np.sum(weights7)  # Normalize
    m7, r2_7, mse_7 = weighted_linear_regression(tau_above, amax_above, weights7)
    scenarios.append({
        'name': 'Only τ=15 and τ=45',
        'weights': weights7,
        'slope': m7,
        'r_squared': r2_7,
        'mse': mse_7
    })

    return scenarios


# Run weight scenarios
scenarios = explore_weight_scenarios(tau_exp, Amax_exp)

# Find the best scenario by R-squared
best_scenario_r2 = max(scenarios, key=lambda x: x['r_squared'])
print(f"Best scenario by R-squared: {best_scenario_r2['name']}")
print(f"Slope (m): {best_scenario_r2['slope']:.4f}")
print(f"R-squared: {best_scenario_r2['r_squared']:.4f}")
print(f"MSE: {best_scenario_r2['mse']:.4f}")

# Find the best scenario by MSE
best_scenario_mse = min(scenarios, key=lambda x: x['mse'])
print(f"\nBest scenario by MSE: {best_scenario_mse['name']}")
print(f"Slope (m): {best_scenario_mse['slope']:.4f}")
print(f"R-squared: {best_scenario_mse['r_squared']:.4f}")
print(f"MSE: {best_scenario_mse['mse']:.4f}")

# Now visualization with the best model (by R-squared)
best_m = best_scenario_r2['slope']
best_weights = best_scenario_r2['weights']

# Generate data for plotting the model
tau_plot = np.linspace(0, 50, 500)
amax_plot = Amax_linear_model(tau_plot, best_m)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot experimental data
plt.scatter(tau_exp, Amax_exp, color='red', s=80, zorder=5, label='Experimental Data')

# Plot all scenarios
colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']  # Added colors for new scenarios
for i, scenario in enumerate(scenarios):
    m = scenario['slope']
    amax_scenario = Amax_linear_model(tau_plot, m)
    plt.plot(tau_plot, amax_scenario,
             color=colors[i % len(colors)],
             linestyle='-' if scenario == best_scenario_r2 else '--',
             linewidth=3 if scenario == best_scenario_r2 else 1.5,
             alpha=1.0 if scenario == best_scenario_r2 else 0.6,
             label=f"{scenario['name']} (m={m:.4f}, R²={scenario['r_squared']:.4f})")

# Highlight best fit
plt.plot(tau_plot, amax_plot, color='blue', linewidth=3.5,
         label=f"Best Fit: {best_scenario_r2['name']} (m={best_m:.4f})")

# Annotate experimental points
for i, (x, y) in enumerate(zip(tau_exp, Amax_exp)):
    # Show point weights if they're above threshold
    if x >= fixed_threshold:
        idx = np.where(tau_exp[tau_exp >= fixed_threshold] == x)[0][0]
        weight_text = f"w={best_weights[idx]:.2f}"
        plt.annotate(f"({x}, {y})\n{weight_text}", (x, y),
                     xytext=(5, 5), textcoords='offset points')
    else:
        plt.annotate(f"({x}, {y})", (x, y),
                     xytext=(5, 5), textcoords='offset points')

# Plot baseline and threshold
plt.axhline(y=1, color='gray', linestyle='--', label='Baseline Amax=1')
plt.axvline(x=fixed_threshold, color='red', linestyle=':', alpha=0.8,
            label=f'Threshold τ={fixed_threshold} dyn/cm²')

# Configure the plot
plt.xlabel('Shear Stress τ (dyn/cm²)', fontsize=12)
plt.ylabel('Normalized Amax', fontsize=12)
plt.title('Weighted Linear Approximation Fits for Amax(τ)', fontsize=14)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 50)
plt.ylim(0, 7)

# Add model equation
equation = f"Amax(τ) = 1 for τ < {fixed_threshold}\n"
equation += f"Amax(τ) = 1 + {best_m:.4f}×(τ-{fixed_threshold}) for τ ≥ {fixed_threshold}"

# Show the scenario comparison table
scenario_table = "Weighting Scenarios Comparison:\n"
scenario_table += "-----------------------------------------\n"
scenario_table += "Scenario | Slope (m) | R² | MSE\n"
scenario_table += "-----------------------------------------\n"

for s in scenarios:
    scenario_table += f"{s['name']:<15} | {s['slope']:.4f} | {s['r_squared']:.4f} | {s['mse']:.4f}\n"

# Position the text boxes
plt.figtext(0.15, 0.02, equation, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.figtext(0.6, 0.3, scenario_table, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

print("\nWeighted Model Equation (Best Fit):")
print(f"Amax(τ) = 1 for τ < {fixed_threshold}")
print(f"Amax(τ) = 1 + {best_m:.4f}×(τ-{fixed_threshold}) for τ ≥ {fixed_threshold}")