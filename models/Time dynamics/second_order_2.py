import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize, Bounds

# Your provided data
P_values = np.array([15, 25, 45])
A_max_map = {15: 1.5, 25: 3.7, 45: 5.3}  # Target steady-state for given P
y0_initial_value = 1.0  # Initial condition for y

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
def calculate_A_max_strictly_linear(P, slope, intercept, dummy_map_arg=None):
    return max(1, slope * P + intercept)


# Calculate the linear model parameters for A_max based on the provided map
P_known = np.array(list(A_max_map.keys()))
A_max_known = np.array(list(A_max_map.values()))
slope_fit, intercept_fit = np.polyfit(P_known, A_max_known, 1)
print(f"Linear A_max model (fit to all known points): A_max = {slope_fit:.4f} * P + {intercept_fit:.4f}")


# --- SECOND-ORDER SYSTEM ---
def model_second_order(Y, t, P_val, omega_n_base, scaling_factor_omega, zeta,
                       slope_param, intercept_param, a_max_calc_func, a_max_map_ref):
    """
    Second-order ODE model:
    dY1/dt = Y2
    dY2/dt = omega_n^2 * (A_max - Y1) - 2*zeta*omega_n*Y2
    where Y = [Y1, Y2] = [y, dy/dt]
    """
    y_val, dy_dt_val = Y
    A_max = a_max_calc_func(P_val, slope_param, intercept_param, a_max_map_ref)

    if A_max <= 0:  # Or some other condition for omega_n
        omega_n = omega_n_base  # Default or minimum omega_n
    else:
        omega_n = omega_n_base * (A_max / 1.0) ** scaling_factor_omega

    if omega_n <= 1e-6:  # Avoid division by zero or extremely stiff system if omega_n is too small
        omega_n = 1e-6

    dY1dt = dy_dt_val
    dY2dt = (omega_n ** 2) * (A_max - y_val) - 2 * zeta * omega_n * dy_dt_val
    return [dY1dt, dY2dt]


# Function to calculate the sum of squared errors for the second-order system
def objective_function_second_order(params, current_P_values, current_y0_val, current_exp_data,
                                    current_slope, current_intercept, current_a_max_map_ref):
    omega_n_base_opt, scaling_factor_omega_opt, zeta_opt = params
    total_error = 0

    # Initial conditions for the second-order system [y(0), dy/dt(0)]
    y0_system = [current_y0_val, 0.0]

    for P_val_opt in current_P_values:
        t_exp = current_exp_data[P_val_opt]['t']
        y_exp = current_exp_data[P_val_opt]['y']

        solution = odeint(model_second_order, y0_system, t_exp,
                          args=(P_val_opt, omega_n_base_opt, scaling_factor_omega_opt, zeta_opt,
                                current_slope, current_intercept,
                                calculate_A_max_hybrid, current_a_max_map_ref))
        y_model = solution[:, 0]  # We only care about the y value (first component)
        error = np.sum((y_model - y_exp) ** 2)
        total_error += error
    return total_error


# Initial parameter guess and bounds for the second-order system
# params: [omega_n_base, scaling_factor_omega, zeta]
initial_params_so = [1.0, 0.5, 0.7]  # Guess: omega_n_base=1, some scaling, zeta=0.7 (slight overshoot)
bounds_so = Bounds(
    [0.01, -2.0, 0.1],  # Lower bounds: omega_n_base > 0, zeta > 0 (e.g. >0.1 to avoid extreme oscillations)
    [10.0, 2.0, 5.0]  # Upper bounds: zeta can be >1 for overdamped
)

print("\nOptimizing for Second-Order System Parameters...")
result_so = minimize(
    objective_function_second_order,
    initial_params_so,
    args=(P_values, y0_initial_value, exp_data, slope_fit, intercept_fit, A_max_map),
    method='L-BFGS-B',  # or 'SLSQP' if Bounds object is used with older scipy
    bounds=bounds_so
)
omega_n_base_opt, scaling_factor_omega_opt, zeta_opt = result_so.x

print("\nOptimized Second-Order Parameters (based on Hybrid A_max for fitting):")
print(f"omega_n_base = {omega_n_base_opt:.4f}")
print(f"scaling_factor_omega = {scaling_factor_omega_opt:.4f}")
print(f"zeta = {zeta_opt:.4f}")
print(f"Final SSE (Second-Order) = {result_so.fun:.4f}")

# --- Plotting Section for Second-Order System ---
t_plot = np.linspace(0, 8, 200)  # More points for smoother second-order curves
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

plt.figure(figsize=(14, 9))
y0_system_plot = [y0_initial_value, 0.0]  # [y(0), dy/dt(0)]

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
    solution_so_hybrid = odeint(model_second_order, y0_system_plot, t_plot,
                                args=(P_val_model, omega_n_base_opt, scaling_factor_omega_opt, zeta_opt,
                                      slope_fit, intercept_fit,
                                      calculate_A_max_hybrid, A_max_map))
    linestyle = '--' if P_val_model in new_P_values_plot else '-'
    label = f'SO Model P={P_val_model} (Hybrid A_max, $\\zeta={zeta_opt:.2f}$)'
    if P_val_model in new_P_values_plot: label += ' Pred.'
    plt.plot(t_plot, solution_so_hybrid[:, 0], color=all_plot_colors.get(P_val_model, 'gray'), linestyle=linestyle,
             label=label, linewidth=2, zorder=5)

# Comparison with strictly linear A_max (optional, can be added if needed)
# Comparison with strictly linear A_max (optional, can be added if needed)
# Plot second-order model solutions using strictly linear A_max for comparison
P_values_for_strict_linear_A_max_comparison = sorted(list(original_P_values_plot)) # Show for original P values

print("\nPlotting Second-Order Model with Strictly Linear A_max for comparison...")

for P_val_strict_comp in P_values_for_strict_linear_A_max_comparison:
    # Ensure A_max calculation uses the strictly linear model
    # The optimized omega_n and zeta parameters are still used, as they were fit to the data
    # using the hybrid A_max for the target during optimization.
    solution_so_strict_linear_A_max = odeint(model_second_order, y0_system_plot, t_plot,
                                             args=(P_val_strict_comp,
                                                   omega_n_base_opt,
                                                   scaling_factor_omega_opt,
                                                   zeta_opt,
                                                   slope_fit,  # slope for linear A_max
                                                   intercept_fit, # intercept for linear A_max
                                                   calculate_A_max_strictly_linear, # KEY CHANGE HERE
                                                   A_max_map)) # A_max_map is a dummy here for strictly_linear

    # Choose a distinct style for these comparison lines
    plt.plot(t_plot, solution_so_strict_linear_A_max[:, 0],
             color=all_plot_colors.get(P_val_strict_comp, 'cyan'), # Use existing color, maybe vary alpha/style
             linestyle=':', # Dotted line for distinction
             linewidth=2.5,
             alpha=0.8, # Slightly transparent
             label=f'SO Model P={P_val_strict_comp} (Strictly Linear A_max, $\\zeta={zeta_opt:.2f}$)')

# Make sure the legend is updated if new labels are added
# handles, labels = plt.gca().get_legend_handles_labels()
# # ... any custom legend ordering if needed ...
# plt.legend(handles, labels, bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
# No, the existing legend call later will pick these up.

plt.xlabel('Time', fontsize=14)
plt.ylabel('Response (y)', fontsize=14)
plt.title(f'Second-Order System Response ($\omega_n$ scaled, $\\zeta={zeta_opt:.2f}$)', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.80, 1])  # Adjust for legend
plt.show()

# Plot 2: Steady-state (A_max) vs Pressure (This plot remains the same as A_max logic hasn't changed)
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


# --- Helper function to find time to X% of steady state for second order system numerically ---
def find_time_to_x_percent_so(t_eval, y_eval, y_initial, target_A_max, percentage=0.95):
    if abs(target_A_max - y_initial) < 1e-6:  # Already at target
        return 0.0

    target_y_val = y_initial + (target_A_max - y_initial) * percentage

    # Check if target_A_max is above or below y_initial
    if target_A_max > y_initial:  # Approaching from below
        # Find first time y_eval crosses or equals target_y_val
        indices = np.where(y_eval >= target_y_val)[0]
        if len(indices) > 0:
            # Simple interpolation for better accuracy (optional)
            idx = indices[0]
            if idx == 0: return t_eval[0]  # Reached immediately or at start
            # Linear interpolation between t_eval[idx-1] and t_eval[idx]
            t1, y1 = t_eval[idx - 1], y_eval[idx - 1]
            t2, y2 = t_eval[idx], y_eval[idx]
            if abs(y2 - y1) < 1e-6:  # Avoid division by zero
                return t1 if abs(target_y_val - y1) < abs(target_y_val - y2) else t2
            time_to_reach = t1 + (t2 - t1) * (target_y_val - y1) / (y2 - y1)
            return time_to_reach
        else:
            return np.nan  # Did not reach
    else:  # Approaching from above (decaying towards target_A_max)
        indices = np.where(y_eval <= target_y_val)[0]
        if len(indices) > 0:
            idx = indices[0]
            if idx == 0: return t_eval[0]
            t1, y1 = t_eval[idx - 1], y_eval[idx - 1]
            t2, y2 = t_eval[idx], y_eval[idx]
            if abs(y2 - y1) < 1e-6:
                return t1 if abs(target_y_val - y1) < abs(target_y_val - y2) else t2
            time_to_reach = t1 + (t2 - t1) * (target_y_val - y1) / (y2 - y1)
            return time_to_reach
        else:
            return np.nan  # Did not reach


# Plot 3: Natural Frequency (omega_n) vs Steady-State (A_max)
plt.figure(figsize=(12, 8))
calculated_A_max_values_hybrid_plot_so = []
calculated_omega_n_values_plot = []
p_values_for_omega_n_plot_sorted = sorted(list(all_P_values_for_plotting))

for P_val_omega_n in p_values_for_omega_n_plot_sorted:
    A_max_val_hybrid = calculate_A_max_hybrid(P_val_omega_n, slope_fit, intercept_fit, A_max_map)
    calculated_A_max_values_hybrid_plot_so.append(A_max_val_hybrid)

    if A_max_val_hybrid <= 0:
        omega_n_val_calc = omega_n_base_opt
    else:
        omega_n_val_calc = omega_n_base_opt * (A_max_val_hybrid / 1.0) ** scaling_factor_omega_opt
    calculated_omega_n_values_plot.append(omega_n_val_calc)

scatter_plot_colors_omega_n = [all_plot_colors.get(P_val_tau_c, 'grey') for P_val_tau_c in
                               p_values_for_omega_n_plot_sorted]
plt.scatter(calculated_A_max_values_hybrid_plot_so, calculated_omega_n_values_plot, c=scatter_plot_colors_omega_n,
            s=100, alpha=0.8, zorder=5)

for i, P_val_annot in enumerate(p_values_for_omega_n_plot_sorted):
    plt.annotate(f'P={P_val_annot}',
                 (calculated_A_max_values_hybrid_plot_so[i], calculated_omega_n_values_plot[i]),
                 xytext=(5, 5), textcoords='offset points')

min_A_max_p_so = min(m for m in calculated_A_max_values_hybrid_plot_so if m > 0) if any(
    m > 0 for m in calculated_A_max_values_hybrid_plot_so) else 0.1
max_A_max_p_so = max(calculated_A_max_values_hybrid_plot_so) if calculated_A_max_values_hybrid_plot_so else 1.0
A_max_range_for_omega_n_curve = np.linspace(max(0.1, min_A_max_p_so * 0.9), max_A_max_p_so * 1.1, 100)

omega_n_curve_plot_vals = [
    omega_n_base_opt * (ss_val / 1.0) ** scaling_factor_omega_opt if ss_val > 0 else omega_n_base_opt
    for ss_val in A_max_range_for_omega_n_curve]
plt.plot(A_max_range_for_omega_n_curve, omega_n_curve_plot_vals, 'k--',
         label=f'$\\omega_n = {omega_n_base_opt:.3f} \\times (A_{{max}})^{{{scaling_factor_omega_opt:.3f}}}$')

plt.xlabel('Steady-State Value ($A_{max}$ from Hybrid Model)', fontsize=14)
plt.ylabel('Natural Frequency ($\\omega_n$)', fontsize=14)
plt.title('Natural Frequency vs. $A_{max}$ (Second-Order Model)', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 4: Time to 95% of Steady State vs Pressure (Second-Order)
plt.figure(figsize=(10, 6))
percentage_to_reach = 0.95  # 95%
P_for_time_curve_plot_so = np.unique(
    np.sort(np.concatenate([P_range_plot_ax, all_P_values_for_plotting])))

times_to_reach_hybrid_A_max_so = []
y0_system_calc = [y0_initial_value, 0.0]
t_eval_fine = np.linspace(0, max(t_plot) * 1.5, 500)  # Finer time for accurate detection

for P_val_time in P_for_time_curve_plot_so:
    A_max_val_h = calculate_A_max_hybrid(P_val_time, slope_fit, intercept_fit, A_max_map)

    # Simulate to get y_eval for this P_val
    sol = odeint(model_second_order, y0_system_calc, t_eval_fine,
                 args=(P_val_time, omega_n_base_opt, scaling_factor_omega_opt, zeta_opt,
                       slope_fit, intercept_fit, calculate_A_max_hybrid, A_max_map))
    y_sim_h = sol[:, 0]

    time_val_h = find_time_to_x_percent_so(t_eval_fine, y_sim_h, y0_initial_value, A_max_val_h,
                                           percentage=percentage_to_reach)
    times_to_reach_hybrid_A_max_so.append(time_val_h)

plt.plot(P_for_time_curve_plot_so, times_to_reach_hybrid_A_max_so,
         label=f'{percentage_to_reach * 100:.0f}% $\\Delta y$ (SO Hybrid A_max, $\\zeta={zeta_opt:.2f}$)',
         linestyle='-')

plt.xlabel('Pressure (P)', fontsize=14)
plt.ylabel(f'Time to Reach {percentage_to_reach * 100:.0f}% of Change', fontsize=14)
plt.title(f'SO Model: Time to Reach Towards $A_{{max}}$ vs. P ($\zeta={zeta_opt:.2f}$)', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

# Print predicted values for Second-Order Hybrid A_max model
print("\nPredicted values (Second-Order Hybrid A_max model):")
print("-" * 100)
print(f"{'Pressure':^10} | {'A_max (Hybrid)':^18} | {'Omega_n':^15} | {'Zeta':^10} | {'Time to 95% (SO)':^25}")
print("-" * 100)
for P_val_table in sorted(list(all_P_values_for_plotting)):
    A_max_val_table = calculate_A_max_hybrid(P_val_table, slope_fit, intercept_fit, A_max_map)
    omega_n_val_table = omega_n_base_opt * (
                A_max_val_table / 1.0) ** scaling_factor_omega_opt if A_max_val_table > 0 else omega_n_base_opt

    sol_table = odeint(model_second_order, y0_system_plot, t_eval_fine,
                       args=(P_val_table, omega_n_base_opt, scaling_factor_omega_opt, zeta_opt,
                             slope_fit, intercept_fit, calculate_A_max_hybrid, A_max_map))
    y_sim_table = sol_table[:, 0]
    time_to_95_table = find_time_to_x_percent_so(t_eval_fine, y_sim_table, y0_initial_value, A_max_val_table,
                                                 percentage=percentage_to_reach)

    print(
        f"{P_val_table:^10.1f} | {A_max_val_table:^18.3f} | {omega_n_val_table:^15.3f} | {zeta_opt:^10.3f} | {time_to_95_table:^25.3f}")
print("-" * 100)