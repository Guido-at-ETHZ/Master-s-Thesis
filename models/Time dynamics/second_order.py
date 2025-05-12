import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

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


# Define a hybrid model for A_max calculation (uses map if P is known, else linear)
def calculate_A_max_hybrid(P, slope, intercept, a_max_map_ref):
    """
    Calculates A_max based on pressure P.
    For P values in a_max_map_ref, it uses the corresponding mapped value.
    Otherwise, it uses linear interpolation/extrapolation.
    Ensures A_max is at least 1.
    """
    if P in a_max_map_ref:
        return a_max_map_ref[P]
    else:
        return max(1, slope * P + intercept)


# Calculate the linear model parameters for A_max based on the provided map
P_known = np.array(list(A_max_map.keys()))
A_max_known = np.array(list(A_max_map.values()))
slope_fit, intercept_fit = np.polyfit(P_known, A_max_known, 1)
print(f"Linear A_max model (fit to all known points): A_max = {slope_fit:.4f} * P + {intercept_fit:.4f}")


# Define the second-order system model
def second_order_model(y, t, P_val, omega_n, zeta, slope_param, intercept_param, a_max_calc_func, a_max_map_ref):
    """
    Second-order ODE model represented as two first-order ODEs:
    y[0] = position, y[1] = velocity

    d²y/dt² + 2*zeta*omega_n*dy/dt + omega_n²*y = omega_n²*A_max

    Converted to:
    dy[0]/dt = y[1]
    dy[1]/dt = omega_n²*A_max - omega_n²*y[0] - 2*zeta*omega_n*y[1]
    """
    # Get A_max based on pressure
    A_max = a_max_calc_func(P_val, slope_param, intercept_param, a_max_map_ref)

    # Extract position and velocity
    position, velocity = y

    # Calculate derivatives
    dposition_dt = velocity
    dvelocity_dt = omega_n ** 2 * A_max - omega_n ** 2 * position - 2 * zeta * omega_n * velocity

    return [dposition_dt, dvelocity_dt]


# Function to calculate the sum of squared errors
def objective_function(params, current_P_values, current_y0, current_exp_data, current_slope,
                       current_intercept, current_a_max_map_ref):
    omega_n, zeta = params
    total_error = 0

    for P_val_opt in current_P_values:
        t_exp = current_exp_data[P_val_opt]['t']
        y_exp = current_exp_data[P_val_opt]['y']

        # Initial conditions for second-order system: [position, velocity]
        initial_conditions = [current_y0, 0.0]  # Starting position y0, zero initial velocity

        # Solve ODE system
        solution = odeint(second_order_model, initial_conditions, t_exp,
                          args=(P_val_opt, omega_n, zeta, current_slope, current_intercept,
                                calculate_A_max_hybrid, current_a_max_map_ref))

        # Extract position values (first column of solution)
        y_model = solution[:, 0]

        # Calculate error
        error = np.sum((y_model - y_exp) ** 2)
        total_error += error

    return total_error


# Initial parameter guess and bounds
# omega_n (natural frequency), zeta (damping ratio)
initial_params = [1.0, 0.7]  # Common starting values
bounds = [(0.1, 10.0), (0.01, 2.0)]  # omega_n > 0, 0 < zeta < 2 (covers under, critical, and over-damping)

# Optimize the parameters
result = minimize(
    objective_function,
    initial_params,
    args=(P_values, y0, exp_data, slope_fit, intercept_fit, A_max_map),
    method='L-BFGS-B',
    bounds=bounds
)
omega_n_opt, zeta_opt = result.x

print("\nOptimized Parameters (Second-Order System):")
print(f"omega_n (natural frequency) = {omega_n_opt:.4f}")
print(f"zeta (damping ratio) = {zeta_opt:.4f}")
print(f"Final SSE = {result.fun:.4f}")

# Classify the system based on damping ratio
if zeta_opt < 1.0:
    system_type = "Underdamped (oscillatory response)"
    td = np.pi / (omega_n_opt * np.sqrt(1 - zeta_opt ** 2))  # Damped period
    print(f"Damped period = {td:.4f}")
elif zeta_opt == 1.0:
    system_type = "Critically damped"
elif zeta_opt > 1.0:
    system_type = "Overdamped"
print(f"System type: {system_type}")

# --- Plotting Section ---
t_plot = np.linspace(0, 8, 100)
original_P_values_plot = P_values
new_P_values_plot = [5, 10, 20, 30, 35, 40, 50, 60, 75]
all_P_values_for_plotting = np.unique(np.concatenate((original_P_values_plot, new_P_values_plot)))

all_plot_colors = {}
all_plot_markers = {}
all_plot_colors.update(colors)
all_plot_markers.update(markers)

new_cmap = plt.get_cmap('plasma')
num_new_P = len(new_P_values_plot)
cmap_colors = [new_cmap(i / (num_new_P - 1 if num_new_P > 1 else 1)) for i in range(num_new_P)]

for i, P_val_new in enumerate(new_P_values_plot):
    all_plot_colors[P_val_new] = cmap_colors[i]
    all_plot_markers[P_val_new] = 'x'

# Plot 1: Second-Order System Response Comparison
plt.figure(figsize=(14, 9))

# Plot experimental data
for P_val_exp in original_P_values_plot:
    if P_val_exp in exp_data:
        plt.scatter(
            exp_data[P_val_exp]['t'], exp_data[P_val_exp]['y'],
            color=all_plot_colors.get(P_val_exp, 'black'), marker=all_plot_markers.get(P_val_exp, '.'),
            s=120, zorder=10, label=f'Data P={P_val_exp}'
        )

# Plot second-order model solutions
for P_val_model in all_P_values_for_plotting:
    initial_state = [y0, 0.0]  # Position y0, zero initial velocity
    solution = odeint(second_order_model, initial_state, t_plot,
                      args=(P_val_model, omega_n_opt, zeta_opt, slope_fit, intercept_fit,
                            calculate_A_max_hybrid, A_max_map))

    y_position = solution[:, 0]  # Extract position (first column)

    linestyle = '--' if P_val_model in new_P_values_plot else '-'
    label = f'Model P={P_val_model} (Second-Order)'
    if P_val_model in new_P_values_plot: label += ' Pred.'

    plt.plot(t_plot, y_position, color=all_plot_colors.get(P_val_model, 'gray'),
             linestyle=linestyle, label=label, linewidth=2, zorder=5)

plt.xlabel('Time', fontsize=14)
plt.ylabel('Response (y)', fontsize=14)
plt.title(f'Second-Order System Response (ωn={omega_n_opt:.2f}, ζ={zeta_opt:.2f})', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.80, 1])
plt.show()

# Plot 2: Steady-state (A_max) vs Pressure
plt.figure(figsize=(10, 6))
P_range_plot_ax = np.linspace(min(0, min(all_P_values_for_plotting) * 0.8), max(all_P_values_for_plotting) * 1.1, 200)

A_max_hybrid_curve = [calculate_A_max_hybrid(P_val_calc, slope_fit, intercept_fit, A_max_map) for P_val_calc in
                      P_range_plot_ax]
plt.plot(P_range_plot_ax, A_max_hybrid_curve, 'm-', linewidth=2,
         label='Effective $A_{max}(P)$ (Hybrid Model)', zorder=5)

plt.scatter(P_known, A_max_known, color='red', s=100, zorder=10, label='Input $A_{max}$ points (from map)')

plt.xlabel('Pressure (P)', fontsize=14)
plt.ylabel('Steady-State Response ($A_{max}$)', fontsize=14)
plt.title('$A_{max}$ vs. Pressure', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 3: First-Order vs Second-Order System Comparison
plt.figure(figsize=(14, 8))


# Function to represent the original first-order model for comparison
def first_order_model_for_comparison(y, t, P_val, tau_base, scaling_factor, slope_param, intercept_param,
                                     a_max_calc_func, a_max_map_ref):
    A_max = a_max_calc_func(P_val, slope_param, intercept_param, a_max_map_ref)
    if A_max <= 0:
        tau = tau_base
    else:
        tau = tau_base * (A_max / 1.0) ** scaling_factor
    dydt = (A_max - y) / tau
    return dydt


# Use the parameters from your first-order model for comparison
# These values are taken from your original code output
tau_base_orig = 0.5  # Example value - replace with your actual optimized value
scaling_factor_orig = 0.8  # Example value - replace with your actual optimized value

# Plot for a few key pressure values to compare first vs second order
key_pressures = [15, 25, 45]  # Original experimental pressures

for P_val in key_pressures:
    # Second-order model solution
    initial_state_2nd = [y0, 0.0]
    solution_2nd = odeint(second_order_model, initial_state_2nd, t_plot,
                          args=(P_val, omega_n_opt, zeta_opt, slope_fit, intercept_fit,
                                calculate_A_max_hybrid, A_max_map))
    y_position_2nd = solution_2nd[:, 0]

    # First-order model solution
    solution_1st = odeint(first_order_model_for_comparison, y0, t_plot,
                          args=(P_val, tau_base_orig, scaling_factor_orig, slope_fit, intercept_fit,
                                calculate_A_max_hybrid, A_max_map))

    # Plot both models
    plt.plot(t_plot, y_position_2nd,
             color=all_plot_colors.get(P_val, 'black'),
             linestyle='-',
             linewidth=2.5,
             label=f'Second-Order P={P_val}')

    plt.plot(t_plot, solution_1st,
             color=all_plot_colors.get(P_val, 'black'),
             linestyle=':',
             linewidth=2,
             label=f'First-Order P={P_val}')

    # Plot experimental data if available
    if P_val in exp_data:
        plt.scatter(
            exp_data[P_val]['t'], exp_data[P_val]['y'],
            color=all_plot_colors.get(P_val, 'black'),
            marker=all_plot_markers.get(P_val, '.'),
            s=120, zorder=10,
            label=f'Data P={P_val}'
        )

plt.xlabel('Time', fontsize=14)
plt.ylabel('Response (y)', fontsize=14)
plt.title('First-Order vs Second-Order System Comparison', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 4: Step Response Characteristics
plt.figure(figsize=(12, 8))

# Generate step response for P=25 (middle value)
P_step = 25
initial_state = [y0, 0.0]
solution_step = odeint(second_order_model, initial_state, t_plot,
                       args=(P_step, omega_n_opt, zeta_opt, slope_fit, intercept_fit,
                             calculate_A_max_hybrid, A_max_map))
y_position_step = solution_step[:, 0]
y_velocity_step = solution_step[:, 1]

# Calculate A_max for this pressure
A_max_step = calculate_A_max_hybrid(P_step, slope_fit, intercept_fit, A_max_map)

plt.plot(t_plot, y_position_step, 'b-', linewidth=3, label='Position (y)')
plt.plot(t_plot, y_velocity_step, 'r--', linewidth=2, label='Velocity (dy/dt)')
plt.axhline(y=A_max_step, color='k', linestyle=':', label=f'Steady-State (A_max={A_max_step:.2f})')

# Mark key response characteristics (if underdamped)
if zeta_opt < 1.0:
    # Find first peak (if it exists)
    peak_indices = []
    for i in range(1, len(t_plot) - 1):
        if y_velocity_step[i - 1] > 0 and y_velocity_step[i + 1] < 0:
            peak_indices.append(i)

    if peak_indices:
        first_peak_idx = peak_indices[0]
        first_peak_time = t_plot[first_peak_idx]
        first_peak_value = y_position_step[first_peak_idx]

        # Mark peak time and overshoot
        overshoot_percent = (first_peak_value - A_max_step) / (A_max_step - y0) * 100
        plt.scatter(first_peak_time, first_peak_value, color='g', s=100, zorder=10)
        plt.annotate(f'Peak Time={first_peak_time:.2f}s\nOvershoot={overshoot_percent:.1f}%',
                     (first_peak_time, first_peak_value), xytext=(10, -30),
                     textcoords='offset points', arrowprops=dict(arrowstyle='->'))

# Calculate rise time (10% to 90% of final value)
rise_start = y0 + 0.1 * (A_max_step - y0)
rise_end = y0 + 0.9 * (A_max_step - y0)

# Find times when response crosses these values
rise_start_idx = np.where(y_position_step >= rise_start)[0][0]
rise_end_idx = np.where(y_position_step >= rise_end)[0][0]

rise_start_time = t_plot[rise_start_idx]
rise_end_time = t_plot[rise_end_idx]
rise_time = rise_end_time - rise_start_time

# Mark rise time
plt.plot([rise_start_time, rise_end_time], [rise_start, rise_end], 'g-', linewidth=2)
plt.annotate(f'Rise Time={rise_time:.2f}s',
             ((rise_start_time + rise_end_time) / 2, (rise_start + rise_end) / 2),
             xytext=(30, 0), textcoords='offset points', arrowprops=dict(arrowstyle='->'))

plt.xlabel('Time', fontsize=14)
plt.ylabel('Response', fontsize=14)
plt.title(f'Second-Order Step Response (P={P_step}, ωn={omega_n_opt:.2f}, ζ={zeta_opt:.2f})', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Print predicted steady-state values and response characteristics
print("\nPredicted values and step response characteristics (Second-Order Model):")
print("-" * 115)
print(
    f"{'Pressure':^10} | {'A_max (Hybrid)':^15} | {'ωn':^10} | {'ζ':^10} | {'Rise Time (10-90%)':^20} | {'Settling Time (95%)':^20} | {'Overshoot (%)':^15}")
print("-" * 115)

for P_val_table in sorted(list(all_P_values_for_plotting)):
    A_max_val_table = calculate_A_max_hybrid(P_val_table, slope_fit, intercept_fit, A_max_map)

    # Calculate step response
    initial_state_table = [y0, 0.0]
    t_table = np.linspace(0, 20, 500)  # Longer time for settling time calculation
    solution_table = odeint(second_order_model, initial_state_table, t_table,
                            args=(P_val_table, omega_n_opt, zeta_opt, slope_fit, intercept_fit,
                                  calculate_A_max_hybrid, A_max_map))
    y_position_table = solution_table[:, 0]

    # Calculate rise time (10% to 90%)
    rise_start_table = y0 + 0.1 * (A_max_val_table - y0)
    rise_end_table = y0 + 0.9 * (A_max_val_table - y0)

    try:
        rise_start_idx_table = np.where(y_position_table >= rise_start_table)[0][0]
        rise_end_idx_table = np.where(y_position_table >= rise_end_table)[0][0]
        rise_time_table = t_table[rise_end_idx_table] - t_table[rise_start_idx_table]
    except IndexError:
        rise_time_table = np.nan

    # Calculate settling time (95% of final value)
    settling_band = 0.05 * (A_max_val_table - y0)
    settling_min = A_max_val_table - settling_band
    settling_max = A_max_val_table + settling_band

    # Find when the response permanently enters the settling band
    settling_time = np.nan
    for i in range(len(t_table) - 1, 0, -1):
        if not (settling_min <= y_position_table[i] <= settling_max):
            settling_time = t_table[i + 1] if i + 1 < len(t_table) else np.nan
            break

    # Calculate overshoot
    overshoot = 0.0
    if zeta_opt < 1.0:  # Underdamped systems can have overshoot
        max_response = np.max(y_position_table)
        if max_response > A_max_val_table:
            overshoot = (max_response - A_max_val_table) / (A_max_val_table - y0) * 100

    print(f"{P_val_table:^10.1f} | {A_max_val_table:^15.3f} | {omega_n_opt:^10.3f} | {zeta_opt:^10.3f} | "
          f"{rise_time_table:^20.3f} | {settling_time:^20.3f} | {overshoot:^15.2f}")

print("-" * 115)