import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.cm as cm  # Kept for now, but specific call changed

# Your provided data
P_values = np.array([15, 25, 45])
A_max_map = {15: 1.5, 25: 3.7, 45: 5.3}  # Target steady-state for given P
y0 = 1.0  # Initial condition for y

# Experimental data
exp_data = {
    15: {'t': np.array([0, 1, 2, 6]), 'y': np.array([1.0, 1.4, 1.0, 1.1])},
    25: {'t': np.array([0, 1, 2, 6]), 'y': np.array([1.0, 3.6, 3.0, 2.9])},
    45: {'t': np.array([0, 1, 2, 6]), 'y': np.array([1.0, 3.0, 3.5, 5.3])}
}
colors = {15: 'blue', 25: 'green', 45: 'red'}
markers = {15: 's', 25: 'o', 45: '^'}


# Define a linear model for A_max calculation
def calculate_A_max(P, slope, intercept):
    """
    Calculates A_max based on pressure P using a linear model.
    For known P values in A_max_map, it uses the mapped value.
    Otherwise, it uses linear interpolation/extrapolation.
    Ensures A_max is at least 1.
    """
    if P in A_max_map:
        return A_max_map[P]
    else:
        # For unknown P values, use linear interpolation/extrapolation
        return max(1, slope * P + intercept)  # Ensure non-negative A_max (min 1)


# Calculate the linear model parameters for A_max based on the provided map
P_known = np.array(list(A_max_map.keys()))
A_max_known = np.array(list(A_max_map.values()))
# Fit a 1st degree polynomial (line) to the known P and A_max values
slope, intercept = np.polyfit(P_known, A_max_known, 1)
print(f"Linear A_max model: A_max = {slope:.4f} * P + {intercept:.4f}")


# Define the simplified model differential equation
def model_simplified(y, t, P, tau_base, scaling_factor, slope_A_max, intercept_A_max):
    """
    Simplified ODE model: dy/dt = (A_max - y) / tau
    A_max is determined by P.
    tau scales with A_max.
    """
    # Get A_max for this P value using the pre-calculated linear model
    A_max = calculate_A_max(P, slope_A_max, intercept_A_max)

    # Calculate tau, which scales with the steady-state value (A_max)
    if A_max <= 0:  # Should not happen with current calculate_A_max logic (min 1)
        tau = tau_base  # Fallback tau
    else:
        # Tau scales with A_max, normalized by 1.0 (can be adjusted if needed)
        tau = tau_base * (A_max / 1.0) ** scaling_factor

    # Differential equation: dy/dt = (A_max - y) / tau
    dydt = (A_max - y) / tau
    return dydt


# Function to calculate the sum of squared errors between model and data
def objective_function_simplified(params):
    """
    Objective function to minimize.
    Calculates total squared error between model predictions and experimental data.
    """
    tau_base, scaling_factor = params  # Parameters to optimize
    total_error = 0

    for P_val in P_values:  # Iterate through each experimental condition
        # Get experimental data for this P value
        t_exp = exp_data[P_val]['t']
        y_exp = exp_data[P_val]['y']

        # Solve the ODE model for this P value with current parameters
        solution = odeint(model_simplified, y0, t_exp, args=(P_val, tau_base, scaling_factor, slope, intercept))
        y_model = solution.flatten()

        # Calculate squared error for this condition
        error = np.sum((y_model - y_exp) ** 2)
        total_error += error

    return total_error


# Initial parameter guess for [tau_base, scaling_factor]
initial_params_simplified = [0.5, 0.8]

# Parameter bounds (min, max) for each parameter
bounds_simplified = [(0.01, 5.0), (-2.0, 2.0)]

# Optimize the parameters using L-BFGS-B method
result_simplified = minimize(
    objective_function_simplified,
    initial_params_simplified,
    method='L-BFGS-B',
    bounds=bounds_simplified
)

# Get the optimized parameters
tau_base_opt, scaling_factor_opt = result_simplified.x

print("\nOptimized Parameters (Simplified Model):")
print(f"tau_base = {tau_base_opt:.4f}")
print(f"scaling_factor = {scaling_factor_opt:.4f}")
print(f"Final SSE = {result_simplified.fun:.4f}")

# --- Plotting Section ---

# Time points for smooth solution curves
t_plot = np.linspace(0, 8, 100)

# Create plot for original P values and additional P values
plt.figure(figsize=(12, 8))

# Define original and new P values for simulation
original_P_values = P_values
new_P_values = [5, 10, 20, 30, 35, 40, 50, 60, 75]
all_P_values_plot = np.unique(np.concatenate((original_P_values, new_P_values)))

# Define colors and markers for plotting
all_colors_plot = {}
all_markers_plot = {}

all_colors_plot.update(colors)
all_markers_plot.update(markers)

# Corrected: Use plt.get_cmap() for Matplotlib 3.7+
# new_cmap = cm.get_cmap('plasma', len(new_P_values))
new_cmap = plt.get_cmap('plasma', len(new_P_values))
for i, P_val in enumerate(new_P_values):
    all_colors_plot[P_val] = new_cmap(i)
    all_markers_plot[P_val] = 'x'

# Plot experimental data and model solutions for original P values
for P_val in original_P_values:
    solution_opt = odeint(model_simplified, y0, t_plot,
                          args=(P_val, tau_base_opt, scaling_factor_opt, slope, intercept))
    plt.plot(t_plot, solution_opt, color=all_colors_plot.get(P_val, 'gray'), label=f'Model P={P_val}')
    if P_val in exp_data:
        plt.scatter(
            exp_data[P_val]['t'],
            exp_data[P_val]['y'],
            color=all_colors_plot.get(P_val, 'black'),
            marker=all_markers_plot.get(P_val, '.'),
            s=100,
            label=f'Data P={P_val}'
        )

# Plot model solutions for new P values (dashed lines)
for P_val in new_P_values:
    solution_new_P = odeint(model_simplified, y0, t_plot,
                            args=(P_val, tau_base_opt, scaling_factor_opt, slope, intercept))
    plt.plot(t_plot, solution_new_P, color=all_colors_plot.get(P_val, 'gray'), label=f'Model P={P_val} (Pred.)',
             linestyle='--')

plt.xlabel('Time', fontsize=14)
plt.ylabel('Response (y)', fontsize=14)
plt.title('Simplified Time-Scaling Response Model', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# Plot: Steady-state (A_max) vs Pressure
plt.figure(figsize=(10, 6))
P_range_plot = np.linspace(min(0, min(all_P_values_plot) * 0.8), max(all_P_values_plot) * 1.1, 100)
A_max_curve = [calculate_A_max(P_val, slope, intercept) for P_val in P_range_plot]
plt.plot(P_range_plot, A_max_curve, 'b-', linewidth=2, label='Calculated $A_{max}(P)$ (Linear Model)')
plt.scatter(P_known, A_max_known, color='red', s=100, zorder=5, label='Input $A_{max}$ points')
plt.xlabel('Pressure (P)', fontsize=14)
plt.ylabel('Steady-State Response ($A_{max}$)', fontsize=14)
plt.title('$A_{max}$ vs. Pressure', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Plot: Time Constant (tau) vs Steady-State (A_max)
plt.figure(figsize=(12, 8))
calculated_A_max_values = []
calculated_tau_values = []
p_values_for_tau_plot = sorted(list(all_P_values_plot))

for P_val in p_values_for_tau_plot:
    A_max_val = calculate_A_max(P_val, slope, intercept)
    calculated_A_max_values.append(A_max_val)
    if A_max_val <= 0:
        tau_val = tau_base_opt
    else:
        tau_val = tau_base_opt * (A_max_val / 1.0) ** scaling_factor_opt
    calculated_tau_values.append(tau_val)

scatter_colors = [all_colors_plot.get(P_val, 'grey') for P_val in p_values_for_tau_plot]
plt.scatter(calculated_A_max_values, calculated_tau_values, c=scatter_colors, s=100, alpha=0.8)
for i, P_val in enumerate(p_values_for_tau_plot):
    plt.annotate(f'P={P_val}', (calculated_A_max_values[i], calculated_tau_values[i]), xytext=(5, 5),
                 textcoords='offset points')

A_max_range_for_curve = np.linspace(max(0.1, min(calculated_A_max_values) * 0.9), max(calculated_A_max_values) * 1.1,
                                    100)
tau_curve_plot = [tau_base_opt * (ss / 1.0) ** scaling_factor_opt if ss > 0 else tau_base_opt for ss in
                  A_max_range_for_curve]
plt.plot(A_max_range_for_curve, tau_curve_plot, 'k--',
         label=f'$\\tau = {tau_base_opt:.3f} \\times (A_{{max}})^{{{scaling_factor_opt:.3f}}}$')
plt.xlabel('Steady-State Value ($A_{max}$)', fontsize=14)
plt.ylabel('Time Constant ($\\tau$)', fontsize=14)
plt.title('Time Constant vs. $A_{max}$', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Plot: Time to 95% of Steady State vs Pressure
plt.figure(figsize=(10, 6))
percentages = [50, 63.2, 90, 95]
P_for_time_curve = np.linspace(1, max(all_P_values_plot) * 1.1, 100)
times_to_reach_plot = {pct: [] for pct in percentages}

for P_val in P_for_time_curve:
    A_max_val = calculate_A_max(P_val, slope, intercept)
    tau_val = tau_base_opt  # Default tau if A_max is invalid

    if A_max_val > 0:  # Calculate specific tau only if A_max is positive
        tau_val = tau_base_opt * (A_max_val / 1.0) ** scaling_factor_opt

    for pct in percentages:
        time_val = np.nan  # Default to NaN if conditions aren't met for a valid time

        if A_max_val <= 0 and pct > 0:  # Cannot reach a percentage of a non-positive steady state
            time_val = np.nan
        elif pct == 0:  # 0% of change takes 0 time
            time_val = 0.0
        elif A_max_val == y0 and pct > 0:  # Already at steady state
            time_val = 0.0
        elif A_max_val > 0:  # Proceed with calculation only if A_max is valid
            # Standard formula for time to reach pct of change for a first-order system
            # t = -tau * ln(1 - fraction_of_change)
            # This applies whether y0 is above or below A_max_val,
            # as pct represents the fraction of the total |A_max_val - y0| covered.
            if pct >= 100:  # Theoretically infinite time to reach 100%
                time_val = np.inf
            else:
                log_arg = 1.0 - (pct / 100.0)
                if log_arg <= 0:  # Should be caught by pct >= 100, but as a safeguard
                    time_val = np.inf
                else:
                    time_val = -tau_val * np.log(log_arg)

        times_to_reach_plot[pct].append(time_val)

for pct in percentages:
    label_pct = f'{pct}% of $\\Delta y$'
    if pct == 63.2: label_pct = f'$\\approx 1\\tau$ ({pct}%)'
    plt.plot(P_for_time_curve, times_to_reach_plot[pct], label=label_pct)

plt.xlabel('Pressure (P)', fontsize=14)
plt.ylabel('Time to Reach Percentage of Change', fontsize=14)
plt.title('Time to Reach Towards $A_{max}$ vs. Pressure', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

# Print predicted values for all P values
print("\nPredicted values for all pressures (Simplified Model):")
print("-" * 80)
print(f"{'Pressure':^10} | {'A_max':^15} | {'Tau (τ)':^15} | {'Time to 95%':^20}")
print("-" * 80)

for P_val in sorted(list(all_P_values_plot)):
    A_max_val = calculate_A_max(P_val, slope, intercept)
    time_to_95 = np.nan  # Default
    tau_val = tau_base_opt  # Default

    if A_max_val <= 0:
        tau_val = tau_base_opt  # Use base if A_max is not positive
    else:
        tau_val = tau_base_opt * (A_max_val / 1.0) ** scaling_factor_opt
        if A_max_val == y0:
            time_to_95 = 0.0
        else:
            # Time to reach 95% of the change from y0 towards A_max
            log_arg_95 = 1.0 - 0.95
            if log_arg_95 > 0:  # Ensure log argument is positive
                time_to_95 = -tau_val * np.log(log_arg_95)

    print(f"{P_val:^10.1f} | {A_max_val:^15.3f} | {tau_val:^15.3f} | {time_to_95:^20.3f}")
print("-" * 80)

