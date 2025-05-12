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
    if P in a_max_map_ref:
        return a_max_map_ref[P]
    else:
        return max(1, slope * P + intercept)


# Define a strictly linear model for A_max calculation
def calculate_A_max_strictly_linear(P, slope, intercept,
                                    dummy_map_arg=None):  # Added dummy_map_arg for consistent signature
    return max(1, slope * P + intercept)


# Calculate the linear model parameters for A_max based on the provided map
P_known = np.array(list(A_max_map.keys()))
A_max_known = np.array(list(A_max_map.values()))
slope_fit, intercept_fit = np.polyfit(P_known, A_max_known, 1)
print(f"Linear A_max model (fit to all known points): A_max = {slope_fit:.4f} * P + {intercept_fit:.4f}")


# --- Second-Order System Model ---
def model_core_logic_second_order(S, t, P_val,
                                  omega_n_base, omega_n_sf,  # Natural frequency base and scaling factor
                                  zeta_base, zeta_sf,  # Damping ratio base and scaling factor
                                  slope_param, intercept_param,
                                  a_max_calc_func, a_max_map_ref,
                                  ref_A_max_for_scaling=1.0):  # Reference A_max for scaling omega_n and zeta
    """
    Core ODE model logic for a second-order system:
    dS[0]/dt = S[1]  (dy/dt = y_dot)
    dS[1]/dt = omega_n^2 * A_max - 2*zeta*omega_n*S[1] - omega_n^2*S[0]
    A_max is determined by P using the provided a_max_calc_func.
    omega_n and zeta can scale with A_max.
    S = [y, y_dot]
    """
    y, y_dot = S
    A_max = a_max_calc_func(P_val, slope_param, intercept_param, a_max_map_ref)

    # Calculate omega_n and zeta, potentially scaled by A_max
    # Ensure A_max for scaling is positive; if A_max is 0 or negative, use base values.
    current_A_max_for_scaling = A_max
    if current_A_max_for_scaling <= 0:  # Avoid issues with log or power of non-positive A_max
        omega_n = omega_n_base
        zeta = zeta_base
    else:
        omega_n = omega_n_base * (current_A_max_for_scaling / ref_A_max_for_scaling) ** omega_n_sf
        zeta = zeta_base * (current_A_max_for_scaling / ref_A_max_for_scaling) ** zeta_sf

    # Ensure omega_n and zeta are physically meaningful
    omega_n = max(1e-6, omega_n)  # Must be positive
    zeta = max(1e-6, zeta)  # Must be positive (though can be interesting if slightly negative for instability)

    # Second-order system equations
    # dS[0]/dt = y_dot
    # dS[1]/dt = d(y_dot)/dt = omega_n^2 * (A_max - y) - 2 * zeta * omega_n * y_dot
    dydt = y_dot
    dy_dot_dt = (omega_n ** 2 * A_max) - (2 * zeta * omega_n * y_dot) - (omega_n ** 2 * y)

    return [dydt, dy_dot_dt]


# Function to calculate the sum of squared errors for the second-order model
def objective_function_second_order(params, current_P_values, current_y0_scalar, current_exp_data,
                                    current_slope, current_intercept, current_a_max_map_ref):
    omega_n_base, omega_n_sf, zeta_base, zeta_sf = params
    total_error = 0
    initial_dy0 = 0.0  # Assuming system starts with zero derivative

    for P_val_opt in current_P_values:
        t_exp = current_exp_data[P_val_opt]['t']
        y_exp = current_exp_data[P_val_opt]['y']

        # Initial conditions for the second-order system: [y(0), dy/dt(0)]
        S0 = [current_y0_scalar, initial_dy0]

        # Optimization uses the hybrid A_max model
        solution = odeint(model_core_logic_second_order, S0, t_exp,
                          args=(P_val_opt, omega_n_base, omega_n_sf, zeta_base, zeta_sf,
                                current_slope, current_intercept,
                                calculate_A_max_hybrid, current_a_max_map_ref))
        y_model = solution[:, 0]  # We only care about y (the first state variable) for error calculation
        error = np.sum((y_model - y_exp) ** 2)
        total_error += error
    return total_error


# Initial parameter guess and bounds for the second-order model
# params = [omega_n_base, omega_n_sf, zeta_base, zeta_sf]
initial_params_second_order = [1.5, 0.1, 0.8, 0.1]  # Initial guesses
bounds_second_order = [
    (0.01, 10.0),  # omega_n_base (natural frequency)
    (-1.0, 1.0),  # omega_n_sf (scaling factor for omega_n)
    (0.05, 5.0),  # zeta_base (damping ratio) - allow underdamped to very overdamped
    (-1.0, 1.0)  # zeta_sf (scaling factor for zeta)
]

print("\nStarting optimization for Second-Order Model...")
# Optimize the parameters for the second-order model
result_second_order = minimize(
    objective_function_second_order,
    initial_params_second_order,
    args=(P_values, y0, exp_data, slope_fit, intercept_fit, A_max_map),
    method='L-BFGS-B',  # or 'SLSQP' or other bounded methods
    bounds=bounds_second_order
)
omega_n_base_opt, omega_n_sf_opt, zeta_base_opt, zeta_sf_opt = result_second_order.x

print("\nOptimized Parameters (Second-Order Model, Hybrid A_max for fitting):")
print(f"omega_n_base = {omega_n_base_opt:.4f}")
print(f"omega_n_scaling_factor = {omega_n_sf_opt:.4f}")
print(f"zeta_base = {zeta_base_opt:.4f}")
print(f"zeta_scaling_factor = {zeta_sf_opt:.4f}")
print(f"Final SSE = {result_second_order.fun:.4f}")

# --- Plotting Section ---
t_plot = np.linspace(0, 8, 200)  # More points for smoother curves
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

# Plot 1: Time-Response Comparison (Second-Order Model)
plt.figure(figsize=(14, 9))
initial_dy0_plot = 0.0  # Derivative at t=0 for plotting

# Plot experimental data
for P_val_exp in original_P_values_plot:
    if P_val_exp in exp_data:
        plt.scatter(
            exp_data[P_val_exp]['t'], exp_data[P_val_exp]['y'],
            color=all_plot_colors.get(P_val_exp, 'black'), marker=all_plot_markers.get(P_val_exp, '.'),
            s=120, zorder=10, label=f'Data P={P_val_exp}'
        )

# Plot second-order model solutions (optimized using hybrid A_max)
for P_val_model in all_P_values_for_plotting:
    S0_plot = [y0, initial_dy0_plot]
    solution_so_hybrid_A_max = odeint(model_core_logic_second_order, S0_plot, t_plot,
                                      args=(P_val_model, omega_n_base_opt, omega_n_sf_opt,
                                            zeta_base_opt, zeta_sf_opt,
                                            slope_fit, intercept_fit,
                                            calculate_A_max_hybrid, A_max_map))
    linestyle = '--' if P_val_model in new_P_values_plot else '-'
    label = f'Model P={P_val_model} (2nd Order, Hybrid A_max)'
    if P_val_model in new_P_values_plot: label += ' Pred.'
    plt.plot(t_plot, solution_so_hybrid_A_max[:, 0], color=all_plot_colors.get(P_val_model, 'gray'),
             linestyle=linestyle, label=label, linewidth=2, zorder=5)

# Plot second-order model solutions using strictly linear A_max for comparison
P_values_for_strict_linear_A_max_comparison = sorted(list(original_P_values_plot))
for P_val_strict_comp in P_values_for_strict_linear_A_max_comparison:
    S0_plot = [y0, initial_dy0_plot]
    solution_so_strict_linear_A_max = odeint(model_core_logic_second_order, S0_plot, t_plot,
                                             args=(P_val_strict_comp, omega_n_base_opt, omega_n_sf_opt,
                                                   zeta_base_opt, zeta_sf_opt,
                                                   slope_fit, intercept_fit,
                                                   calculate_A_max_strictly_linear,
                                                   A_max_map))  # A_max_map is dummy here
    plt.plot(t_plot, solution_so_strict_linear_A_max[:, 0],
             color=all_plot_colors.get(P_val_strict_comp, 'cyan'),
             linestyle=':', linewidth=2.5, alpha=0.9,
             label=f'Model P={P_val_strict_comp} (2nd Order, Strictly Linear A_max)')

plt.xlabel('Time', fontsize=14)
plt.ylabel('Response (y)', fontsize=14)
plt.title('Second-Order System Response Model Comparison', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.80, 1])
plt.show()

# Plot 2: Steady-state (A_max) vs Pressure (This plot remains the same as it's about A_max definition)
plt.figure(figsize=(10, 6))
P_range_plot_ax = np.linspace(min(0, min(all_P_values_for_plotting) * 0.8), max(all_P_values_for_plotting) * 1.1, 200)
A_max_hybrid_curve = [calculate_A_max_hybrid(P_val_calc, slope_fit, intercept_fit, A_max_map) for P_val_calc in
                      P_range_plot_ax]
plt.plot(P_range_plot_ax, A_max_hybrid_curve, 'm-', linewidth=2,
         label='Effective $A_{max}(P)$ (Hybrid: Map then Linear)', zorder=5)
A_max_strictly_linear_curve = [calculate_A_max_strictly_linear(P_val_calc, slope_fit, intercept_fit) for P_val_calc in
                               P_range_plot_ax]
plt.plot(P_range_plot_ax, A_max_strictly_linear_curve, 'g--', linewidth=2, label='Strictly Linear $A_{max}(P)$ Fit',
         zorder=3)
plt.scatter(P_known, A_max_known, color='red', s=100, zorder=10, label='Input $A_{max}$ points (from map)')
plt.xlabel('Pressure (P)', fontsize=14)
plt.ylabel('Steady-State Response ($A_{max}$)', fontsize=14)
plt.title('$A_{max}$ vs. Pressure: Model Comparison', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 3: Effective Omega_n and Zeta vs A_max
plt.figure(figsize=(12, 8))
calculated_A_max_values_plot = []
calculated_omega_n_values_plot = []
calculated_zeta_values_plot = []
p_values_for_params_plot_sorted = sorted(list(all_P_values_for_plotting))
ref_A_max_for_scaling = 1.0  # As used in the model

for P_val_param in p_values_for_params_plot_sorted:
    A_max_val = calculate_A_max_hybrid(P_val_param, slope_fit, intercept_fit, A_max_map)
    calculated_A_max_values_plot.append(A_max_val)

    current_A_max_for_scaling = A_max_val
    if current_A_max_for_scaling <= 0:
        omega_n_val_calc = omega_n_base_opt
        zeta_val_calc = zeta_base_opt
    else:
        omega_n_val_calc = omega_n_base_opt * (current_A_max_for_scaling / ref_A_max_for_scaling) ** omega_n_sf_opt
        zeta_val_calc = zeta_base_opt * (current_A_max_for_scaling / ref_A_max_for_scaling) ** zeta_sf_opt

    omega_n_val_calc = max(1e-6, omega_n_val_calc)
    zeta_val_calc = max(1e-6, zeta_val_calc)
    calculated_omega_n_values_plot.append(omega_n_val_calc)
    calculated_zeta_values_plot.append(zeta_val_calc)

fig, ax1 = plt.subplots(figsize=(12, 8))
scatter_plot_colors_params = [all_plot_colors.get(P_val_p_c, 'grey') for P_val_p_c in p_values_for_params_plot_sorted]

# Plot Omega_n
color = 'tab:red'
ax1.set_xlabel('Steady-State Value ($A_{max}$ from Hybrid Model)', fontsize=14)
ax1.set_ylabel('Natural Frequency ($\\omega_n$)', color=color, fontsize=14)
ax1.scatter(calculated_A_max_values_plot, calculated_omega_n_values_plot, c=scatter_plot_colors_params, s=100,
            alpha=0.8, zorder=5, marker='o')
ax1.tick_params(axis='y', labelcolor=color)
# Add annotations for P values for omega_n
for i, P_val_annot in enumerate(p_values_for_params_plot_sorted):
    ax1.annotate(f'P={P_val_annot}', (calculated_A_max_values_plot[i], calculated_omega_n_values_plot[i]),
                 xytext=(5, -10), textcoords='offset points', color=color)

# Create a second y-axis for Zeta
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Damping Ratio ($\\zeta$)', color=color, fontsize=14)
ax2.scatter(calculated_A_max_values_plot, calculated_zeta_values_plot, c=scatter_plot_colors_params, s=100, alpha=0.8,
            zorder=5, marker='^')
ax2.tick_params(axis='y', labelcolor=color)
# Add annotations for P values for zeta
for i, P_val_annot in enumerate(p_values_for_params_plot_sorted):
    ax2.annotate(f'P={P_val_annot}', (calculated_A_max_values_plot[i], calculated_zeta_values_plot[i]),
                 xytext=(5, 10), textcoords='offset points', color=color)

# Plot curves for omega_n and zeta if A_max varies continuously
min_A_max_p = min(m for m in calculated_A_max_values_plot if m > 0) if any(
    m > 0 for m in calculated_A_max_values_plot) else 0.1
max_A_max_p = max(calculated_A_max_values_plot) if calculated_A_max_values_plot else 1.0
A_max_range_for_curve = np.linspace(max(0.01, min_A_max_p * 0.9), max_A_max_p * 1.1, 100)  # Ensure positive for scaling

omega_n_curve_vals = [
    omega_n_base_opt * (ss_val / ref_A_max_for_scaling) ** omega_n_sf_opt if ss_val > 0 else omega_n_base_opt for ss_val
    in A_max_range_for_curve]
zeta_curve_vals = [zeta_base_opt * (ss_val / ref_A_max_for_scaling) ** zeta_sf_opt if ss_val > 0 else zeta_base_opt for
                   ss_val in A_max_range_for_curve]
omega_n_curve_vals = [max(1e-6, val) for val in omega_n_curve_vals]
zeta_curve_vals = [max(1e-6, val) for val in zeta_curve_vals]

ax1.plot(A_max_range_for_curve, omega_n_curve_vals, color='tab:red', linestyle='--',
         label=f'$\\omega_n = {omega_n_base_opt:.2f} \\times (A_{{max}}/{ref_A_max_for_scaling})^{{{omega_n_sf_opt:.2f}}}$')
ax2.plot(A_max_range_for_curve, zeta_curve_vals, color='tab:blue', linestyle=':',
         label=f'$\\zeta = {zeta_base_opt:.2f} \\times (A_{{max}}/{ref_A_max_for_scaling})^{{{zeta_sf_opt:.2f}}}$')

fig.suptitle('Second-Order Parameters $\\omega_n$ and $\\zeta$ vs. $A_{max}$ (Hybrid Model)', fontsize=16)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)  # Combined legend
fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for title and legend
plt.show()

# Print predicted values for Second-Order Hybrid A_max model
print("\nPredicted values (Second-Order Model, Hybrid A_max - used for fitting):")
print("-" * 105)
print(f"{'Pressure':^10} | {'A_max (Hybrid)':^18} | {'Omega_n (ωn)':^18} | {'Zeta (ζ)':^15} | {'Comment':^30}")
print("-" * 105)
for P_val_table in sorted(list(all_P_values_for_plotting)):
    A_max_val_table = calculate_A_max_hybrid(P_val_table, slope_fit, intercept_fit, A_max_map)

    current_A_max_for_scaling_table = A_max_val_table
    if current_A_max_for_scaling_table <= 0:
        omega_n_table = omega_n_base_opt
        zeta_table = zeta_base_opt
    else:
        omega_n_table = omega_n_base_opt * (current_A_max_for_scaling_table / ref_A_max_for_scaling) ** omega_n_sf_opt
        zeta_table = zeta_base_opt * (current_A_max_for_scaling_table / ref_A_max_for_scaling) ** zeta_sf_opt

    omega_n_table = max(1e-6, omega_n_table)
    zeta_table = max(1e-6, zeta_table)

    comment = ""
    if zeta_table < 1:
        comment = "Underdamped (overshoot likely)"
    elif zeta_table == 1:
        comment = "Critically damped"
    else:
        comment = "Overdamped (no overshoot)"

    print(
        f"{P_val_table:^10.1f} | {A_max_val_table:^18.3f} | {omega_n_table:^18.3f} | {zeta_table:^15.3f} | {comment:^30}")
print("-" * 105)

# Note: Plot 4 for "Time to 95% of Steady State" is more complex for a second-order system
# as it depends on both omega_n and zeta, and doesn't have a simple analytical formula like -tau*ln(0.05).
# It would require finding the time by checking the simulation or using approximations.
# For simplicity, I'm omitting the direct adaptation of Plot 4.
# You can infer steepness from omega_n and overshoot characteristics from zeta.

print("\nConsiderations for Second-Order Model:")
print("1. Steepness (Rise Time): Primarily influenced by omega_n. Higher omega_n generally leads to a faster rise.")
print("2. Overshoot/Oscillations: Determined by zeta. zeta < 1 implies overshoot.")
print(
    "3. The experimental data for P=15 and P=25 shows non-monotonic behavior that might be challenging for a standard LTI second-order system to capture perfectly if A_max is a simple step target.")
print(
    "4. The 'Time to 95%' plot is not directly translated as it's more complex for second-order systems. Analyze omega_n for speed and zeta for shape.")