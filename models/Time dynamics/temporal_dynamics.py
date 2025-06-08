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


# Define a strictly linear model for A_max calculation
def calculate_A_max_strictly_linear(P, slope, intercept,
                                    dummy_map_arg=None):  # Added dummy_map_arg for consistent signature if needed by wrapper
    """
    Calculates A_max based on pressure P using ONLY a linear model.
    Ensures A_max is at least 1.
    """
    return max(1, slope * P + intercept)


# Calculate the linear model parameters for A_max based on the provided map
P_known = np.array(list(A_max_map.keys()))
A_max_known = np.array(list(A_max_map.values()))
slope_fit, intercept_fit = np.polyfit(P_known, A_max_known, 1)
print(f"Linear A_max model (fit to all known points): A_max = {slope_fit:.4f} * P + {intercept_fit:.4f}")
print(
    f"Note: This linear model predicts A_max(P=25) = {calculate_A_max_strictly_linear(25, slope_fit, intercept_fit):.4f}, while A_max_map[25] = {A_max_map[25]}")


# Define the core model differential equation logic
def model_core_logic(y, t, P_val, tau_base, scaling_factor, slope_param, intercept_param, a_max_calc_func,
                     a_max_map_ref):
    """
    Core ODE model logic: dy/dt = (A_max - y) / tau
    A_max is determined by P using the provided a_max_calc_func.
    tau scales with A_max.
    """
    A_max = a_max_calc_func(P_val, slope_param, intercept_param, a_max_map_ref)

    if A_max <= 0:
        tau = tau_base
    else:
        tau = tau_base * (A_max / 1.0) ** scaling_factor

    dydt = (A_max - y) / tau
    return dydt


# Function to calculate the sum of squared errors
def objective_function_simplified(params, current_P_values, current_y0, current_exp_data, current_slope,
                                  current_intercept, current_a_max_map_ref):
    tau_base, scaling_factor = params
    total_error = 0

    for P_val_opt in current_P_values:
        t_exp = current_exp_data[P_val_opt]['t']
        y_exp = current_exp_data[P_val_opt]['y']

        # Optimization uses the hybrid A_max model
        solution = odeint(model_core_logic, current_y0, t_exp,
                          args=(P_val_opt, tau_base, scaling_factor, current_slope, current_intercept,
                                calculate_A_max_hybrid, current_a_max_map_ref))
        y_model = solution.flatten()
        error = np.sum((y_model - y_exp) ** 2)
        total_error += error
    return total_error


# Initial parameter guess and bounds
initial_params_simplified = [0.5, 0.8]
bounds_simplified = [(0.01, 5.0), (-2.0, 2.0)]

# Optimize the parameters
result_simplified = minimize(
    objective_function_simplified,
    initial_params_simplified,
    args=(P_values, y0, exp_data, slope_fit, intercept_fit, A_max_map),
    method='L-BFGS-B',
    bounds=bounds_simplified
)
tau_base_opt, scaling_factor_opt = result_simplified.x

print("\nOptimized Parameters (based on Hybrid A_max for fitting):")
print(f"tau_base = {tau_base_opt:.4f}")
print(f"scaling_factor = {scaling_factor_opt:.4f}")
print(f"Final SSE = {result_simplified.fun:.4f}")

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

# Plot 1: Time-Scaling Response Model Comparison
plt.figure(figsize=(14, 9))

# Plot experimental data
for P_val_exp in original_P_values_plot:
    if P_val_exp in exp_data:
        plt.scatter(
            exp_data[P_val_exp]['t'], exp_data[P_val_exp]['y'],
            color=all_plot_colors.get(P_val_exp, 'black'), marker=all_plot_markers.get(P_val_exp, '.'),
            s=120, zorder=10, label=f'Data P={P_val_exp}'
        )

# Plot model solutions (optimized using hybrid A_max)
for P_val_model in all_P_values_for_plotting:
    solution_hybrid_A_max = odeint(model_core_logic, y0, t_plot,
                                   args=(P_val_model, tau_base_opt, scaling_factor_opt, slope_fit, intercept_fit,
                                         calculate_A_max_hybrid, A_max_map))
    linestyle = '--' if P_val_model in new_P_values_plot else '-'
    label = f'Model P={P_val_model} (Hybrid A_max)'
    if P_val_model in new_P_values_plot: label += ' Pred.'
    plt.plot(t_plot, solution_hybrid_A_max, color=all_plot_colors.get(P_val_model, 'gray'), linestyle=linestyle,
             label=label, linewidth=2, zorder=5)

# Plot model solutions using strictly linear A_max for comparison
P_values_for_strict_linear_A_max_comparison = sorted(list(original_P_values_plot))   # Show for originals
for P_val_strict_comp in P_values_for_strict_linear_A_max_comparison:
    solution_strict_linear_A_max = odeint(model_core_logic, y0, t_plot,
                                          args=(
                                          P_val_strict_comp, tau_base_opt, scaling_factor_opt, slope_fit, intercept_fit,
                                          calculate_A_max_strictly_linear, A_max_map))  # A_max_map is dummy here
    plt.plot(t_plot, solution_strict_linear_A_max,
             color=all_plot_colors.get(P_val_strict_comp, 'cyan'),
             linestyle=':', linewidth=2.5, alpha=0.9,
             label=f'Model P={P_val_strict_comp} (Strictly Linear A_max)')

plt.xlabel('Time', fontsize=14)
plt.ylabel('Response (y)', fontsize=14)
plt.title('Time-Scaling Response Model Comparison', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
handles, labels = plt.gca().get_legend_handles_labels()  # For ordering legend
# Simple way to somewhat group labels - not perfect, manual ordering might be better
# For now, standard legend
plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.80, 1])
plt.savefig('model_plots_time.png', dpi=300, bbox_inches='tight')
print("Plots saved to model_plots.png")

# Plot 2: Steady-state (A_max) vs Pressure
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
plt.savefig('model_plots_response.png', dpi=300, bbox_inches='tight')
print("Plots saved to model_plots.png")

# Plot 3: Time Constant (tau) vs Steady-State (A_max)
plt.figure(figsize=(12, 8))
calculated_A_max_values_hybrid_plot = []
calculated_tau_values_for_hybrid_A_max_plot = []
p_values_for_tau_plot_sorted = sorted(list(all_P_values_for_plotting))

for P_val_tau in p_values_for_tau_plot_sorted:
    A_max_val_hybrid = calculate_A_max_hybrid(P_val_tau, slope_fit, intercept_fit, A_max_map)
    calculated_A_max_values_hybrid_plot.append(A_max_val_hybrid)

    if A_max_val_hybrid <= 0:
        tau_val_calc = tau_base_opt
    else:
        tau_val_calc = tau_base_opt * (A_max_val_hybrid / 1.0) ** scaling_factor_opt
    calculated_tau_values_for_hybrid_A_max_plot.append(tau_val_calc)

scatter_plot_colors_tau = [all_plot_colors.get(P_val_tau_c, 'grey') for P_val_tau_c in p_values_for_tau_plot_sorted]
plt.scatter(calculated_A_max_values_hybrid_plot, calculated_tau_values_for_hybrid_A_max_plot, c=scatter_plot_colors_tau,
            s=100, alpha=0.8, zorder=5)

for i, P_val_annot in enumerate(p_values_for_tau_plot_sorted):
    plt.annotate(f'P={P_val_annot}',
                 (calculated_A_max_values_hybrid_plot[i], calculated_tau_values_for_hybrid_A_max_plot[i]),
                 xytext=(5, 5), textcoords='offset points')

min_A_max_p = min(m for m in calculated_A_max_values_hybrid_plot if m > 0) if any(
    m > 0 for m in calculated_A_max_values_hybrid_plot) else 0.1
max_A_max_p = max(calculated_A_max_values_hybrid_plot) if calculated_A_max_values_hybrid_plot else 1.0
A_max_range_for_tau_curve = np.linspace(max(0.1, min_A_max_p * 0.9), max_A_max_p * 1.1, 100)

tau_curve_plot_vals = [tau_base_opt * (ss_val / 1.0) ** scaling_factor_opt if ss_val > 0 else tau_base_opt for ss_val in
                       A_max_range_for_tau_curve]
plt.plot(A_max_range_for_tau_curve, tau_curve_plot_vals, 'k--',
         label=f'$\\tau = {tau_base_opt:.3f} \\times (A_{{max}})^{{{scaling_factor_opt:.3f}}}$ (using Hybrid A_max for points)')

plt.xlabel('Steady-State Value ($A_{max}$ from Hybrid Model)', fontsize=14)
plt.ylabel('Time Constant ($\\tau$)', fontsize=14)
plt.title('Time Constant vs. $A_{max}$ (based on Hybrid Model)', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('model_plots_steady.png', dpi=300, bbox_inches='tight')
print("Plots saved to model_plots.png")

# Plot 4: Time to 95% of Steady State vs Pressure
plt.figure(figsize=(10, 6))
percentage_to_reach = 95
P_for_time_curve_plot = np.unique(
    np.sort(np.concatenate([P_range_plot_ax, all_P_values_for_plotting])))  # Use a dense range including key points

times_to_reach_hybrid_A_max = []
times_to_reach_strict_linear_A_max = []

for P_val_time in P_for_time_curve_plot:
    # Using Hybrid A_max
    A_max_val_h = calculate_A_max_hybrid(P_val_time, slope_fit, intercept_fit, A_max_map)
    tau_val_h = tau_base_opt * (A_max_val_h / 1.0) ** scaling_factor_opt if A_max_val_h > 0 else tau_base_opt
    time_val_h = -tau_val_h * np.log(1.0 - percentage_to_reach / 100.0) if A_max_val_h > 0 and A_max_val_h != y0 else (
        0.0 if A_max_val_h == y0 else np.nan)
    if P_val_time == 0 and y0 == 0 and A_max_val_h == 0:
        time_val_h = 0.0  # Special case if y0 is 0 and A_max is 0 at P=0
    elif P_val_time == 0 and y0 != 0 and A_max_val_h == 0:
        time_val_h = -tau_val_h * np.log(0.05)  # Time to decay to 5% if target is 0 from non-zero y0

    # Using Strictly Linear A_max
    A_max_val_sl = calculate_A_max_strictly_linear(P_val_time, slope_fit, intercept_fit)
    tau_val_sl = tau_base_opt * (A_max_val_sl / 1.0) ** scaling_factor_opt if A_max_val_sl > 0 else tau_base_opt
    time_val_sl = -tau_val_sl * np.log(
        1.0 - percentage_to_reach / 100.0) if A_max_val_sl > 0 and A_max_val_sl != y0 else (
        0.0 if A_max_val_sl == y0 else np.nan)
    if P_val_time == 0 and y0 == 0 and A_max_val_sl == 0:
        time_val_sl = 0.0
    elif P_val_time == 0 and y0 != 0 and A_max_val_sl == 0:
        time_val_sl = -tau_val_sl * np.log(0.05)

    times_to_reach_hybrid_A_max.append(time_val_h)
    times_to_reach_strict_linear_A_max.append(time_val_sl)

plt.plot(P_for_time_curve_plot, times_to_reach_hybrid_A_max, label=f'{percentage_to_reach}% $\\Delta y$ (Hybrid A_max)',
         linestyle='-')
plt.plot(P_for_time_curve_plot, times_to_reach_strict_linear_A_max,
         label=f'{percentage_to_reach}% $\\Delta y$ (Strictly Linear A_max)', linestyle=':')

plt.xlabel('Pressure (P)', fontsize=14)
plt.ylabel(f'Time to Reach {percentage_to_reach}% of Change', fontsize=14)
plt.title(f'Time to Reach Towards $A_{{max}}$ vs. Pressure ({percentage_to_reach}% $\\Delta y$)', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.ylim(bottom=0)  # Ensure y-axis starts at 0 for time
plt.tight_layout()
plt.savefig('model_plots.png', dpi=300, bbox_inches='tight')
print("Plots saved to model_plots.png")

# Print predicted values for Hybrid A_max model
print("\nPredicted values (Hybrid A_max model - used for fitting):")
print("-" * 85)
print(f"{'Pressure':^10} | {'A_max (Hybrid)':^18} | {'Tau (τ)':^15} | {'Time to 95% (Hybrid)':^25}")
print("-" * 85)
for P_val_table in sorted(list(all_P_values_for_plotting)):
    A_max_val_table = calculate_A_max_hybrid(P_val_table, slope_fit, intercept_fit, A_max_map)
    tau_val_table = tau_base_opt * (
                A_max_val_table / 1.0) ** scaling_factor_opt if A_max_val_table > 0 else tau_base_opt
    time_to_95_table = -tau_val_table * np.log(1.0 - 0.95) if A_max_val_table > 0 and A_max_val_table != y0 else (
        0.0 if A_max_val_table == y0 else np.nan)
    if P_val_table == 0 and y0 != 0 and A_max_val_table == 0: time_to_95_table = -tau_val_table * np.log(0.05)
    print(f"{P_val_table:^10.1f} | {A_max_val_table:^18.3f} | {tau_val_table:^15.3f} | {time_to_95_table:^25.3f}")
print("-" * 85)

# Print predicted values for Strictly Linear A_max model
print("\nPredicted values (Strictly Linear A_max model - for comparison):")
print("-" * 95)
print(
    f"{'Pressure':^10} | {'A_max (Strict Linear)':^22} | {'Tau (τ, using same opt params)':^30} | {'Time to 95% (Strict Linear)':^28}")
print("-" * 95)
for P_val_table_sl in sorted(list(all_P_values_for_plotting)):
    A_max_val_table_sl = calculate_A_max_strictly_linear(P_val_table_sl, slope_fit, intercept_fit)
    tau_val_table_sl = tau_base_opt * (
                A_max_val_table_sl / 1.0) ** scaling_factor_opt if A_max_val_table_sl > 0 else tau_base_opt
    time_to_95_table_sl = -tau_val_table_sl * np.log(
        1.0 - 0.95) if A_max_val_table_sl > 0 and A_max_val_table_sl != y0 else (
        0.0 if A_max_val_table_sl == y0 else np.nan)
    if P_val_table_sl == 0 and y0 != 0 and A_max_val_table_sl == 0: time_to_95_table_sl = -tau_val_table_sl * np.log(
        0.05)
    print(
        f"{P_val_table_sl:^10.1f} | {A_max_val_table_sl:^22.3f} | {tau_val_table_sl:^30.3f} | {time_to_95_table_sl:^28.3f}")
print("-" * 95)