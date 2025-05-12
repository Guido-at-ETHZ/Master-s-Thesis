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
    15: {'t': np.array([0, 1, 3, 6]), 'y': np.array([1.0, 1.4, 1.0, 1.1])},
    25: {'t': np.array([0, 1, 3, 6]), 'y': np.array([1.0, 3.6, 3.0, 2.9])},
    45: {'t': np.array([0, 1, 2, 6]), 'y': np.array([1.0, 3.0, 3.5, 5.3])}
}
colors = {15: 'blue', 25: 'green', 45: 'red'}
markers = {15: 's', 25: 'o', 45: '^'}


# Define a model where the time constant scales with the steady-state value
def model(y, t, P, V_max, K_m, tau_base, scaling_factor):
    # Calculate production rate using Michaelis-Menten kinetics
    v = V_max * P / (K_m + P)

    # Calculate steady-state value that the system is approaching
    K = A_max_map[P] / v
    steady_state = K * v  # This equals A_max_map[P]

    # Calculate tau to scale with the steady-state value
    # tau = tau_base * (steady_state/reference_point)^scaling_factor
    # Using a reference point of 1.0
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

print("Optimized Parameters:")
print(f"V_max = {V_max_opt:.3f}")
print(f"K_m = {K_m_opt:.3f}")
print(f"tau_base = {tau_base_opt:.3f}")
print(f"scaling_factor = {scaling_factor_opt:.3f}")

# Time points for solution
t = np.linspace(0, 8, 100)

# Create plot
plt.figure(figsize=(12, 8))

# Solve and plot for each P value using optimized parameters
for P in P_values:
    # Solve the ODE with optimized parameters
    solution = odeint(model, y0, t, args=(P, V_max_opt, K_m_opt, tau_base_opt, scaling_factor_opt))

    # Plot the model solution
    plt.plot(t, solution, color=colors[P], label=f'Model P={P}')

    # Plot the experimental data points
    plt.scatter(
        exp_data[P]['t'],
        exp_data[P]['y'],
        color=colors[P],
        marker=markers[P],
        s=100,
        label=f'Data P={P}'
    )

plt.xlabel('Time', fontsize=14)
plt.ylabel('Response', fontsize=14)
plt.title('Time-Scaling Response Model', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()

# Plot time to reach various percentages of steady state
plt.figure(figsize=(10, 6))
percentages = [50, 63, 75, 90, 95]  # Percentages of steady state to reach
steady_states = np.linspace(1.0, 6.0, 50)  # Range of steady state values
times = {}

# Calculate times for each percentage
for pct in percentages:
    times[pct] = []
    for ss in steady_states:
        # Calculate the tau for this steady state
        tau = tau_base_opt * (ss / 1.0) ** scaling_factor_opt

        # Time to reach percentage = -tau * ln(1 - percentage/100)
        time_to_reach = -tau * np.log(1 - pct / 100)
        times[pct].append(time_to_reach)

    plt.plot(steady_states, times[pct], label=f'{pct}% of steady state')

plt.xlabel('Steady State Value', fontsize=14)
plt.ylabel('Time to Reach (units)', fontsize=14)
plt.title('Time Required vs. Steady State Magnitude', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()

# Print derived parameters for each P value with optimized parameters
for P in P_values:
    v = V_max_opt * P / (K_m_opt + P)
    K = A_max_map[P] / v
    steady_state = K * v  # Should equal A_max_map[P]
    tau = tau_base_opt * (steady_state / 1.0) ** scaling_factor_opt

    # Calculate characteristic time to reach 63% of steady state
    time_to_63pct = -tau * np.log(1 - 0.63)
    # Calculate time to reach 95% of steady state
    time_to_95pct = -tau * np.log(1 - 0.95)

    print(f"\nFor P = {P} (optimized):")
    print(f"  Production rate (v) = {v:.3f}")
    print(f"  Steady state value = {steady_state:.3f}")
    print(f"  Time constant (tau) = {tau:.3f}")
    print(f"  Time to reach 63% of steady state = {time_to_63pct:.3f}")
    print(f"  Time to reach 95% of steady state = {time_to_95pct:.3f}")
    print(f"  Expected steady state (A_max) = {A_max_map[P]}")

plt.show()