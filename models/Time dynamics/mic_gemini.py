import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.cm as cm # Added import for cm

# Your provided data
P_values = np.array([15, 25, 45])
A_max_map = {15: 1.5, 25: 3.7, 45: 5.3}
y0 = 1.0

# Experimental data
exp_data = {
    15: {'t': np.array([0, 1, 3, 6]), 'y': np.array([1.0, 1.4, 1.0, 1.1])},
    25: {'t': np.array([0, 1, 3, 6]), 'y': np.array([1.0, 3.6, 3.0, 2.9])},
    45: {'t': np.array([0, 1, 2, 6]), 'y': np.array([1.0, 3.0, 3.5, 5.3])}
}
colors = {15: 'blue', 25: 'green', 45: 'red'}
markers = {15: 's', 25: 'o', 45: '^'}


# Define a linear model for A_max calculation
# This function uses the globally defined slope and intercept once calculated
def calculate_A_max(P, slope, intercept):
    # Simple linear model: A_max = slope * P + intercept
    # For known P values, use the original A_max map for consistency in optimization
    # However, for general calculation and plotting, rely purely on the linear model
    # Let's refine this: The optimization uses the model, plots should too.
    # The override below might cause discrepancies if calculate_A_max is used
    # outside the ODE solver where the specific A_max_map values are expected.
    # For plotting K vs P, we definitely want the linear model result.
    # Let's stick to the linear model consistently after fitting slope/intercept.

    # return max(y0, slope * P + intercept) # Ensure A_max >= y0 or positive
    # Let's use the version from the original code for now, but be aware of this.
     if P in A_max_map:
         # This ensures the ODE solver uses the exact target values during optimization
         # for the points being fitted. But may not be best for plotting general trends.
         # Let's use the linear model everywhere *after* fitting for consistency in plots.
         pass # We will calculate slope/intercept first

     # For plotting and general calculation outside the optimization loop for known P:
     # We use the fitted linear model. Ensure A_max doesn't go below a reasonable floor.
     # A floor of 0 or slightly above might be more logical than y0. Let's use 0.1
     return max(0.1, slope * P + intercept)


# Calculate the linear model parameters for A_max
P_known = np.array(list(A_max_map.keys()))
A_max_known = np.array(list(A_max_map.values()))
slope, intercept = np.polyfit(P_known, A_max_known, 1)
print(f"Linear A_max model: A_max = {slope:.4f} * P + {intercept:.4f}")


# Define a model where the time constant scales with the steady-state value
# Make sure slope and intercept are accessible, either as args or global
def model(y, t, P, V_max, K_m, tau_base, scaling_factor, slope_param, intercept_param):
    # Calculate production rate using Michaelis-Menten kinetics
    # Ensure K_m + P is not zero
    if K_m + P == 0:
       v = 0 # Or handle appropriately
    else:
       v = V_max * P / (K_m + P)

    # Get A_max for this P value using the passed slope and intercept
    # Use the exact values from map for optimization points if needed, or stick to linear model
    if P in A_max_map:
         A_max = A_max_map[P] # Use mapped value for optimization cost calculation consistency
    else:
         A_max = max(0.1, slope_param * P + intercept_param) # Use linear model otherwise

    # Calculate K to achieve the desired A_max
    # Avoid division by zero if v is zero
    if v == 0:
        # If v is 0, K is technically undefined or infinite unless A_max is also 0.
        # This situation occurs at P=0.
        # If P=0, A_max = intercept. If intercept > 0, K -> inf. If intercept = 0, K has limit.
        # The ODE solver might not start at t=0 exactly, or P might be > 0.
        # Let's assume for P > 0, v > 0. If P=0, v=0.
        if P == 0:
            steady_state = max(0.1, intercept_param) # Steady state at P=0 is the intercept
            K = np.inf # Or handle based on intercept value if needed elsewhere
            dydt = (steady_state - y) / tau_base # Simplified dynamics at P=0? Check model logic.
                                                 # Let's assume P>0 in practice for the ODE part.
                                                 # If P can be 0, the model definition needs care.
            # Fallback: if K is needed and v=0, maybe set dydt differently?
            # Original model structure breaks if v=0 and K is used.
            # Let's assume we only simulate for P > 0 or where v > 0.
            # If P=0, dydt = (intercept - y) / tau_base perhaps?
            # For now, we rely on the optimizer and odeint handling P > 0 cases primarily.
            # Let's return 0 derivative if v=0 and P>0 (shouldn't happen)
            # If P=0, use intercept as target.
            A_max_at_P0 = max(0.1, intercept_param)
            tau_at_P0 = tau_base * (A_max_at_P0 / 1.0) ** scaling_factor if A_max_at_P0 > 0 else tau_base
            return (A_max_at_P0 - y) / tau_at_P0

        else: # v=0 for P>0, implies V_max=0? Problematic case.
             K = np.nan
             steady_state = 0 # Or A_max? If V_max=0, A_max should also be 0?
    else:
        K = A_max / v
        steady_state = K * v  # This equals A_max by definition if v!=0

    # Calculate tau to scale with the steady-state value
    # Handle cases where steady_state is zero or negative (should be >= 0.1 now)
    if steady_state <= 0: # Should not happen if A_max floor is > 0
        tau = tau_base
    else:
        tau = tau_base * (steady_state / 1.0) ** scaling_factor

    # Differential equation: dy/dt = (K*v - y) / tau = (steady_state - y) / tau
    # Ensure tau is not zero
    if tau == 0:
        # Handle division by zero - depends on desired behavior (e.g., infinite rate?)
        # Perhaps return a very large number or handle based on sign of (steady_state - y)
        # For now, assume tau > 0 from bounds.
        dydt = np.sign(steady_state - y) * np.inf
    else:
        dydt = (steady_state - y) / tau

    return dydt


# Function to calculate the sum of squared errors between model and data
def objective_function(params):
    V_max, K_m, tau_base, scaling_factor = params
    # We use the globally fitted slope and intercept here
    global slope, intercept

    total_error = 0

    for P in P_values:
        # Get experimental data for this P value
        t_exp = exp_data[P]['t']
        y_exp = exp_data[P]['y']

        # Solve the model for this P value, passing slope and intercept
        try:
             solution = odeint(model, y0, t_exp, args=(P, V_max, K_m, tau_base, scaling_factor, slope, intercept))
             y_model = solution.flatten()
             # Calculate squared error
             error = np.sum((y_model - y_exp) ** 2)
             total_error += error
        except Exception as e:
             # Penalize if ODE solver fails for these parameters
             # print(f"ODE solver failed for params {params} at P={P}: {e}")
             return np.inf # Return a large error

    return total_error


# Initial parameter guess
initial_params = [2.0, 10.0, 0.5, 0.8]  # V_max, K_m, tau_base, scaling_factor

# Parameter bounds (min, max) for each parameter
# Ensure tau_base lower bound is slightly > 0
bounds = [(0.1, 20.0), (0.1, 100.0), (0.01, 5.0), (0.1, 2.0)]

# Optimize the parameters
result = minimize(
    objective_function,
    initial_params,
    method='L-BFGS-B', # Changed method if needed, e.g., 'TNC'
    bounds=bounds
)

# Check if optimization was successful
if not result.success:
    print(f"Optimization failed: {result.message}")
    # Handle failure, e.g., use initial guess or stop
    # For now, print warning and continue with potentially suboptimal params
    V_max_opt, K_m_opt, tau_base_opt, scaling_factor_opt = initial_params
else:
    # Get the optimized parameters
    V_max_opt, K_m_opt, tau_base_opt, scaling_factor_opt = result.x

print("Optimized Parameters:")
print(f"V_max = {V_max_opt:.3f}")
print(f"K_m = {K_m_opt:.3f}")
print(f"tau_base = {tau_base_opt:.3f}")
print(f"scaling_factor = {scaling_factor_opt:.3f}")

# --- Plotting Section ---

# Time points for plotting smooth curves
t = np.linspace(0, 8, 100)

# Create plot for model vs data
plt.figure(figsize=(12, 8))

# Define original and new P values for plotting
original_P_values = P_values
new_P_values = [5, 10, 20, 30, 35, 40, 50, 60, 75]  # Additional P values to simulate
all_P_values = np.sort(np.unique(np.concatenate((original_P_values, new_P_values)))) # Sort and unique

# Define colors and markers for all P values
all_colors = {}
all_markers = {}
# Original colors and markers
all_colors.update(colors)
all_markers.update(markers)
# New colors using a colormap
new_cmap = cm.plasma
for i, P in enumerate(new_P_values):
    if P not in all_colors: # Avoid overwriting original colors if P is reused
      all_colors[P] = new_cmap(i / len(new_P_values))
      all_markers[P] = 'x' # Use 'x' for all new P values

# Solve and plot for all P values using optimized parameters
for P in all_P_values:
    # Solve the ODE with optimized parameters
    try:
         # Pass the globally fitted slope and intercept to the model function
         solution = odeint(model, y0, t, args=(P, V_max_opt, K_m_opt, tau_base_opt, scaling_factor_opt, slope, intercept))
         linestyle = '--' if P in new_P_values else '-'
         label_prefix = 'Model'
         # Plot the model solution
         plt.plot(t, solution, color=all_colors[P], label=f'{label_prefix} P={P}', linestyle=linestyle)
    except Exception as e:
         print(f"ODE plotting failed for P={P}: {e}")

    # Plot the experimental data points if P is in original data
    if P in exp_data:
        plt.scatter(
            exp_data[P]['t'],
            exp_data[P]['y'],
            color=all_colors[P],
            marker=all_markers[P],
            s=100,
            label=f'Data P={P}',
            zorder=5 # Ensure data points are on top
        )

plt.xlabel('Time', fontsize=14)
plt.ylabel('Response (y)', fontsize=14)
plt.title('Model Simulation vs. Experimental Data', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Calculate and plot steady-state vs pressure relationship
plt.figure(figsize=(10, 6))

# Range of pressure values for curve
P_range = np.linspace(0, 80, 100)
steady_states = []

# Calculate steady states for each pressure value using the linear model
for P_plot in P_range:
    # Use the linear model for A_max consistently for plotting the trend
    ss_val = calculate_A_max(P_plot, slope, intercept)
    steady_states.append(ss_val)

# Plot the steady-state vs pressure curve based on linear A_max model
plt.plot(P_range, steady_states, 'b-', linewidth=2, label=f'A_max = {slope:.2f}*P + {intercept:.2f}')

# Plot the original A_max target values used in optimization
P_orig = list(A_max_map.keys())
Amax_orig = list(A_max_map.values())
plt.scatter(P_orig, Amax_orig, color='red', s=100, label='Target A_max Points', zorder=5)

plt.xlabel('Pressure (P)', fontsize=14)
plt.ylabel('Steady-State Response (A_max)', fontsize=14)
plt.title('Steady-State Response vs. Pressure', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()

# Plot time constants vs steady state for different pressures
plt.figure(figsize=(12, 8))

# Calculate time constants and steady states for all P values
tau_values = {}
steady_state_values = {}

for P in all_P_values:
    # Use the linear model A_max for calculating steady state and tau consistently
    A_max = calculate_A_max(P, slope, intercept)
    steady_state = A_max # Since K*v = A_max

    if steady_state <= 0: # Should be prevented by max(0.1, ...) in calculate_A_max
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
# Ensure all P values have colors, generate if missing (shouldn't happen with current logic)
sorted_colors = [all_colors.get(P, 'gray') for P in sorted_P]

# Plot tau vs steady state
plt.scatter(sorted_steady_states, sorted_taus, c=sorted_colors, s=100)

# Add P value labels to points
for i, P in enumerate(sorted_P):
    plt.annotate(f' P={P}',
                 (sorted_steady_states[i], sorted_taus[i]),
                 xytext=(5, 5),
                 textcoords='offset points')

# Add curve showing the relationship: tau = tau_base * (SS)^scaling_factor
# Use a range based on plotted steady states
min_ss = min(s for s in sorted_steady_states if s > 0) # Find min positive steady state
max_ss = max(sorted_steady_states)
ss_range = np.linspace(min_ss * 0.9, max_ss * 1.1, 100)
tau_curve = [tau_base_opt * (ss / 1.0) ** scaling_factor_opt if ss > 0 else tau_base_opt for ss in ss_range]
plt.plot(ss_range, tau_curve, 'k--', label=f'τ = {tau_base_opt:.3f} × (SS)^{scaling_factor_opt:.3f}')

plt.xlabel('Steady-State Value (A_max)', fontsize=14)
plt.ylabel('Time Constant (τ)', fontsize=14)
plt.title('Time Constant vs. Steady-State Value', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()

# Plot time to 95% steady state vs pressure
plt.figure(figsize=(10, 6))

# Calculate times to reach various percentages of steady state
percentages = [50, 63, 90, 95]
P_for_curve = np.linspace(1, 80, 100) # Start P > 0
times_to_reach = {pct: [] for pct in percentages}

for P_plot in P_for_curve:
    # Use linear model A_max consistently
    A_max = calculate_A_max(P_plot, slope, intercept)
    steady_state = A_max

    # Calculate tau based on this A_max
    if steady_state <= 0:
        tau = tau_base_opt
    else:
        tau = tau_base_opt * (steady_state / 1.0) ** scaling_factor_opt

    # Calculate time to reach percentage = -tau * ln(1 - percentage/100)
    for pct in percentages:
        if steady_state <= y0 : # If target is below start, time is tricky, maybe 0 or NaN
             # Or if pct calculation makes no sense (e.g. reaching 95% of a value below start)
             # For simplicity, let's show NaN if target <= start, requires adjustment if y0 != target
             # A more robust way handles distance: frac = (y(t)-y0)/(A_max-y0)
             # t = -tau * ln(1 - frac) => t = -tau * ln(1-pct/100) if y0=0
             # If y0!=0, time to reach y_target = y0 + pct/100 * (A_max - y0)
             # y_target = A_max - (A_max-y0)*exp(-t/tau)
             # (y_target - A_max) / (y0 - A_max) = exp(-t/tau)
             # (y0 + pct/100*(A_max-y0) - A_max) / (y0 - A_max) = exp(-t/tau)
             # ( (pct/100 - 1)*(A_max-y0) ) / -(A_max-y0) = exp(-t/tau)
             # (1 - pct/100) = exp(-t/tau)
             # t = -tau * ln(1 - pct/100) # Formula holds regardless of y0 for % of total change
             time = -tau * np.log(1 - pct / 100) if tau > 0 else np.inf # Check tau>0
        else:
             time = -tau * np.log(1 - pct / 100) if tau > 0 else np.inf

        times_to_reach[pct].append(time)

# Plot the curves
for pct in percentages:
    plt.plot(P_for_curve, times_to_reach[pct], label=f'{pct}% of Δy') # Label reflects % of change

plt.xlabel('Pressure (P)', fontsize=14)
plt.ylabel('Time to Reach % of Steady State Change', fontsize=14)
plt.title('Time Required vs. Pressure', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()


# --- ADDED: Plot K vs P ---
plt.figure(figsize=(10, 6))

# Calculate K values over a range of P
P_range_K = np.linspace(0.1, 80, 200) # Start slightly above 0
K_values = []

for P_k in P_range_K:
    # Calculate A_max using the fitted linear model consistently
    A_max_k = calculate_A_max(P_k, slope, intercept)

    # Calculate v using optimized parameters
    if K_m_opt + P_k == 0: v_k = 0
    else: v_k = V_max_opt * P_k / (K_m_opt + P_k)

    # Calculate K = A_max / v
    if v_k == 0:
        # Handle P=0 case based on intercept
        if P_k == 0: K_val = np.inf if intercept > 0 else (slope * K_m_opt / V_max_opt if V_max_opt != 0 else np.inf)
        else: K_val = np.nan # v=0 for P>0 is problematic
    else:
        K_val = A_max_k / v_k

    K_values.append(K_val)

# Plot the K vs P curve
plt.plot(P_range_K, K_values, 'm-', linewidth=2, label='K = A_max / v')

# Add points for the original P values used in the experiment
for P in original_P_values:
     A_max_p = calculate_A_max(P, slope, intercept) # Use linear model consistently
     v_p = V_max_opt * P / (K_m_opt + P) if (K_m_opt + P) != 0 else 0
     if v_p != 0:
         K_p = A_max_p / v_p
         plt.scatter(P, K_p, color=colors[P], s=100, zorder=5, label=f'_nolegend_') # Use nolegend trick
         plt.annotate(f' P={P}', (P, K_p), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('Pressure (P)', fontsize=14)
plt.ylabel('Scaling Factor (K)', fontsize=14)
plt.title('Scaling Factor K vs. Pressure', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()
# Optional: Adjust y-axis if K values are very large near P=0
# Check if K_values contains infinity or very large numbers
finite_K = [k for k in K_values if np.isfinite(k)]
if finite_K:
     # Set ylim based on finite values, maybe add some padding
     # Or set a specific reasonable upper limit if K starts near infinity
     # plt.ylim(bottom=0, top=max(finite_K)*1.1) # Example: scale to max finite value
     if intercept > 0 and P_range_K[0] < 1: # If intercept is positive and we plot near P=0
           reasonable_top = np.percentile(finite_K, 95) # Show most of the range, clip infinity
           plt.ylim(bottom=0, top=reasonable_top * 1.5)


# --- End of Added Plot ---


# Adjust layout to prevent overlapping titles/labels
plt.tight_layout()


# Print predicted values table
print("\nPredicted values for all pressures (using linear A_max model consistently):")
print("-" * 70)
print(f"{'Pressure':^10} | {'A_max (linear)':^15} | {'Time Constant':^15} | {'Time to 95%':^15}")
print("-" * 70)

for P in sorted(all_P_values):
    A_max = calculate_A_max(P, slope, intercept) # Use linear model
    steady_state = A_max # K*v = A_max

    if steady_state <= 0:
        tau = tau_base_opt
        time_to_95 = np.nan
    else:
        tau = tau_base_opt * (steady_state / 1.0) ** scaling_factor_opt
        # Formula holds for % of change regardless of y0
        time_to_95 = -tau * np.log(1 - 0.95) if tau > 0 else np.inf

    # Removed redundant columns from printout for clarity, added K
    # v = V_max_opt * P / (K_m_opt + P) if (K_m_opt+P)!=0 else 0
    # K_val = A_max / v if v!=0 else np.nan
    print(f"{P:^10.1f} | {A_max:^15.3f} | {tau:^15.3f} | {time_to_95:^15.3f}")

# Display all generated plots
plt.show()