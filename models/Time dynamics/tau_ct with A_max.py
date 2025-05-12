import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

# Your provided data
P_values = np.array([15, 25, 45])
A_max_map = {15: 1.5, 25: 3.7, 45: 5.3}
y0 = 1.0

# Experimental data
exp_data = {
    15: {'t': np.array([0, 1, 2, 6]), 'y': np.array([1.0, 1.4, 1.0, 1.1])},
    25: {'t': np.array([0, 1, 2, 6]), 'y': np.array([1.0, 3.6, 3.0, 2.9])},
    45: {'t': np.array([0, 1, 2, 6]), 'y': np.array([1.0, 3.0, 3.5, 5.3])}
}
colors = {15: 'blue', 25: 'green', 45: 'red'}
markers = {15: 's', 25: 'o', 45: '^'}

# Define threshold model parameters
threshold = 5.0
base_value = 1.0
slope = 0.1093


# MODIFIED: Define the threshold model for A_max calculation
def calculate_A_max(P, threshold=threshold, base_value=base_value, slope=slope):
    """
    Calculate A_max using a threshold model:
    A_max = base_value for P < threshold
    A_max = base_value + slope * (P - threshold) for P >= threshold
    """
    # For known P values, use the original A_max
    if P in A_max_map:
        return A_max_map[P]
    else:
        # For unknown P values, use the threshold model
        if P < threshold:
            return base_value
        else:
            return base_value + slope * (P - threshold)


# Calculate the linear model parameters for reference (not used in the new model)
P_known = np.array(list(A_max_map.keys()))
A_max_known = np.array(list(A_max_map.values()))
linear_slope, linear_intercept = np.polyfit(P_known, A_max_known, 1)
print(f"Original Linear A_max model: A_max = {linear_slope:.4f} * P + {linear_intercept:.4f}")
print(
    f"New Threshold Model: A_max = {base_value} for P < {threshold}, A_max = {base_value} + {slope} * (P - {threshold}) for P >= {threshold}")


# Define a model where the time constant scales with the steady-state value
def model(y, t, P, V_max, K_m, tau_base, scaling_factor):
    # Calculate production rate using Michaelis-Menten kinetics
    v = V_max * P / (K_m + P)

    # Get A_max for this P value using the threshold model
    A_max = calculate_A_max(P)

    # Calculate K to achieve the desired A_max
    K = A_max / v

    # Calculate steady-state value
    steady_state = K * v  # This equals A_max

    # Calculate tau to scale with the steady-state value
    # Handle cases where steady_state is zero or negative
    if steady_state <= 0:
        tau = tau_base
    else:
        tau = tau_base * (steady_state / 1.0) ** scaling_factor

    # Differential equation: dy/dt = (K*v - y) / tau
    dydt = (K * v - y) / tau

    return dydt


# Function to calculate the sum of squared errors between model and data
def objective_function(params):
    V_max, K_m, tau_base, scaling_factor = params

    total_error = 0

    for P in P_values:
        # Get experimental data for this P value
        t_exp = exp_data[P]['t']
        y_exp = exp_data[P]['y']

        # Solve the model for this P value
        solution = odeint(model, y0, t_exp, args=(P, V_max, K_m, tau_base, scaling_factor))
        y_model = solution.flatten()

        # Calculate squared error
        error = np.sum((y_model - y_exp) ** 2)
        total_error += error

    return total_error


# Initial parameter guess
initial_params = [2.0, 10.0, 0.5, 0.8]  # V_max, K_m, tau_base, scaling_factor

# Parameter bounds (min, max) for each parameter
bounds = [(0.1, 20.0), (0.1, 100.0), (0.1, 5.0), (0.1, 2.0)]

# Optimize the parameters
result = minimize(
    objective_function,
    initial_params,
    method='L-BFGS-B',
    bounds=bounds
)

# Get the optimized parameters
V_max_opt, K_m_opt, tau_base_opt, scaling_factor_opt = result.x

print("\nOptimized Parameters:")
print(f"V_max = {V_max_opt:.3f}")
print(f"K_m = {K_m_opt:.3f}")
print(f"tau_base = {tau_base_opt:.3f}")
print(f"scaling_factor = {scaling_factor_opt:.3f}")

# Time points for solution
t = np.linspace(0, 8, 100)

# Create plot for original P values and additional P values
plt.figure(figsize=(12, 8))

# Define original and new P values
original_P_values = P_values
new_P_values = [1, 3, 5, 10, 20, 30, 35, 40, 50, 60, 75]  # Added more values around threshold
all_P_values = np.concatenate((original_P_values, new_P_values))

# Define additional colors for new P values
import matplotlib.cm as cm

all_colors = {}
all_markers = {}

# Colors for original values
all_colors.update(colors)

# Colors for new values - use colormap for gradient
new_cmap = cm.plasma
for i, P in enumerate(new_P_values):
    all_colors[P] = new_cmap(i / len(new_P_values))
    all_markers[P] = 'x'  # Use 'x' for all new P values

# Updating markers for original values
all_markers.update(markers)

# Solve and plot for original P values using optimized parameters
for P in original_P_values:
    # Solve the ODE with optimized parameters
    solution = odeint(model, y0, t, args=(P, V_max_opt, K_m_opt, tau_base_opt, scaling_factor_opt))

    # Plot the model solution
    plt.plot(t, solution, color=all_colors[P], label=f'Model P={P}')

    # Plot the experimental data points
    plt.scatter(
        exp_data[P]['t'],
        exp_data[P]['y'],
        color=all_colors[P],
        marker=all_markers[P],
        s=100,
        label=f'Data P={P}'
    )

# Solve and plot for new P values using optimized parameters with dashed lines
for P in new_P_values:
    # Solve the ODE with optimized parameters
    solution = odeint(model, y0, t, args=(P, V_max_opt, K_m_opt, tau_base_opt, scaling_factor_opt))

    # Plot the model solution with dashed line
    plt.plot(t, solution, color=all_colors[P], label=f'Model P={P}', linestyle='--')

plt.xlabel('Time', fontsize=14)
plt.ylabel('Response', fontsize=14)
plt.title('Time-Scaling Response Model with Threshold at P=5', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Calculate and plot steady-state vs pressure relationship
plt.figure(figsize=(10, 6))

# Range of pressure values for curve
P_range = np.linspace(0, 80, 100)
threshold_model_ss = []
linear_model_ss = []

# Calculate steady states for each pressure value using both models
for P in P_range:
    if P > 0:  # Avoid division by zero
        # Threshold model
        A_max_threshold = calculate_A_max(P)
        v = V_max_opt * P / (K_m_opt + P)
        K = A_max_threshold / v
        steady_state_threshold = K * v  # This equals A_max_threshold
        threshold_model_ss.append(steady_state_threshold)

        # Linear model (for comparison)
        A_max_linear = linear_slope * P + linear_intercept
        K_linear = A_max_linear / v
        steady_state_linear = K_linear * v
        linear_model_ss.append(steady_state_linear)
    else:
        threshold_model_ss.append(0)
        linear_model_ss.append(0)

# Plot both models for comparison
plt.plot(P_range, threshold_model_ss, 'b-', linewidth=2, label='Threshold Model')
plt.plot(P_range, linear_model_ss, 'r--', linewidth=2, label='Original Linear Model')

# Plot the known steady-state values
for P in P_values:
    plt.scatter(P, A_max_map[P], color='green', s=100)

# Indicate the threshold
plt.axvline(x=5, color='k', linestyle=':', label='Threshold P=5')

plt.xlabel('Pressure (P)', fontsize=14)
plt.ylabel('Steady-State Response', fontsize=14)
plt.title('Steady-State Response vs. Pressure (Threshold Model)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()

# Plot time constants vs steady state for different pressures
plt.figure(figsize=(12, 8))

# Calculate time constants for all P values
tau_values = {}
steady_state_values = {}

for P in all_P_values:
    A_max = calculate_A_max(P)
    v = V_max_opt * P / (K_m_opt + P)
    K = A_max / v
    steady_state = K * v  # This equals A_max

    if steady_state <= 0:
        tau = tau_base_opt
    else:
        tau = tau_base_opt * (steady_state / 1.0) ** scaling_factor_opt

    steady_state_values[P] = steady_state
    tau_values[P] = tau

# Sort P values by steady state for plotting
sorted_P = sorted(all_P_values, key=lambda P: steady_state_values[P])

# Create lists for plotting
sorted_steady_states = [steady_state_values[P] for P in sorted_P]
sorted_taus = [tau_values[P] for P in sorted_P]
sorted_colors = [all_colors[P] for P in sorted_P]

# Plot tau vs steady state
plt.scatter(sorted_steady_states, sorted_taus, c=sorted_colors, s=100)

# Add P value labels to points
for i, P in enumerate(sorted_P):
    plt.annotate(f'P={P}',
                 (sorted_steady_states[i], sorted_taus[i]),
                 xytext=(5, 5),
                 textcoords='offset points')

# Add curve showing the relationship
ss_range = np.linspace(max(0.1, min(sorted_steady_states) * 0.9), max(sorted_steady_states) * 1.1, 100)
tau_curve = [tau_base_opt * (ss / 1.0) ** scaling_factor_opt if ss > 0 else tau_base_opt for ss in ss_range]
plt.plot(ss_range, tau_curve, 'k--', label=f'τ = {tau_base_opt:.3f} × (SS)^{scaling_factor_opt:.3f}')

plt.xlabel('Steady-State Value', fontsize=14)
plt.ylabel('Time Constant (τ)', fontsize=14)
plt.title('Time Constant vs. Steady-State Value', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()

# Plot time to 95% steady state vs pressure
plt.figure(figsize=(10, 6))

# Calculate times to reach various percentages of steady state
percentages = [50, 63, 90, 95]
P_for_curve = np.linspace(1, 80, 100)
times_to_reach = {}

for pct in percentages:
    times_to_reach[pct] = []

    for P in P_for_curve:
        A_max = calculate_A_max(P)
        v = V_max_opt * P / (K_m_opt + P)
        K = A_max / v
        steady_state = K * v  # This equals A_max

        # Handle negative or zero steady state values
        if steady_state <= 0:
            tau = tau_base_opt
            # For negative steady states, set time to reach to NaN (will not be plotted)
            time = np.nan
        else:
            tau = tau_base_opt * (steady_state / 1.0) ** scaling_factor_opt
            # Time to reach percentage = -tau * ln(1 - percentage/100)
            time = -tau * np.log(1 - pct / 100)

        times_to_reach[pct].append(time)

    plt.plot(P_for_curve, times_to_reach[pct], label=f'{pct}% of steady state')

# Indicate the threshold
plt.axvline(x=5, color='k', linestyle=':', label='Threshold P=5')

plt.xlabel('Pressure (P)', fontsize=14)
plt.ylabel('Time to Reach (units)', fontsize=14)
plt.title('Time Required to Reach Steady State vs. Pressure', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

# Print predicted values for all P values
print("\nPredicted values for all pressures with threshold model:")
print("-" * 70)
print(f"{'Pressure':^10} | {'A_max':^15} | {'Steady State':^15} | {'Time Constant':^15} | {'Time to 95%':^15}")
print("-" * 70)

for P in sorted(all_P_values):
    A_max = calculate_A_max(P)
    v = V_max_opt * P / (K_m_opt + P)
    K = A_max / v
    steady_state = K * v  # Should equal A_max

    if steady_state <= 0:
        tau = tau_base_opt
        time_to_95 = np.nan
    else:
        tau = tau_base_opt * (steady_state / 1.0) ** scaling_factor_opt
        time_to_95 = -tau * np.log(1 - 0.95)

    print(f"{P:^10.1f} | {A_max:^15.3f} | {steady_state:^15.3f} | {tau:^15.3f} | {time_to_95:^15.3f}")

plt.show()