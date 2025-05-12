import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Define the shear stress-dependent senescence rate function
def senescence_rate_shear(tau):
    """
    Models how shear stress affects senescence rate.
    Low shear stress: minimal effect
    High shear stress: rapidly increases senescence

    Parameters:
    tau: Shear stress value

    Returns:
    Senescence rate due to shear stress
    """
    tau_threshold = 15  # Threshold beyond which senescence rapidly increases
    k = 0.05  # Steepness of the response

    # Sigmoid function
    return 0.01 + 0.2 / (1 + np.exp(-k * (tau - tau_threshold)))


# Define the cell dynamics model with telomere-based replicative senescence
def endothelial_cell_dynamics(t, y, params):
    """
    System of differential equations for endothelial cell dynamics with
    telomere-based replicative senescence.

    Parameters:
    t: Time point
    y: Current state [E, S, T_avg]
        E: Healthy endothelial cells
        S: Senescent cells
        T_avg: Average telomere length (replication potential)

    Returns:
    Derivatives [dE/dt, dS/dt, dT_avg/dt]
    """
    E, S, T_avg = y  # Unpack the state vector

    # Unpack parameters
    r = params['r']  # Base proliferation rate
    K = params['K']  # Carrying capacity
    d_E = params['d_E']  # Death rate of healthy cells
    d_S = params['d_S']  # Death rate of senescent cells
    gamma_S = params['gamma_S']  # Senescence induction by senescent cells
    tau = params['tau']  # Shear stress
    T_crit = params['T_crit']  # Critical telomere length for senescence
    T_loss = params['T_loss']  # Telomere loss per division

    # Calculate total cells
    total_cells = E + S

    # Calculate carrying capacity limited proliferation rate
    proliferation_factor = 1 - total_cells / K

    # Calculate shear stress effect on senescence
    gamma_tau = senescence_rate_shear(tau)

    # Calculate replicative senescence rate based on telomere length
    # As T_avg approaches T_crit, more cells become senescent when dividing
    if T_avg > T_crit:
        # Sigmoid function for transition from healthy division to senescence
        rep_sen_factor = np.exp(-(T_avg - T_crit)) / (1 + np.exp(-(T_avg - T_crit)))
    else:
        rep_sen_factor = 1.0  # All divisions lead to senescence when below critical length

    # Effective division rate (affected by carrying capacity)
    effective_div_rate = r * proliferation_factor

    # Proportion of divisions that become senescent due to telomere shortening
    gamma_R = effective_div_rate * rep_sen_factor

    # Proportion of divisions that produce healthy daughters (not senescent)
    healthy_div_factor = 1 - rep_sen_factor

    # Differential equations

    # Healthy cells: gain from division, lose from death and all senescence types
    dEdt = (effective_div_rate * E * healthy_div_factor  # Successful divisions
            - d_E * E  # Death
            - gamma_tau * E  # Shear stress senescence
            - gamma_S * S * E  # Senescence induction by other senescent cells
            - gamma_R * E)  # Replicative senescence

    # Senescent cells: gain from all types of senescence, lose from death
    dSdt = (gamma_tau * E  # Shear stress senescence
            + gamma_S * S * E  # Senescence induction
            + gamma_R * E  # Replicative senescence
            - d_S * S)  # Death

    # Telomere length: decreases with each cell division
    # When cells divide successfully, their average telomere length decreases
    if E > 0:
        dT_avgdt = -T_loss * effective_div_rate * healthy_div_factor
    else:
        dT_avgdt = 0  # No change if no healthy cells

    return [dEdt, dSdt, dT_avgdt]


# Run simulation with different parameters
def run_simulation(tau_value, initial_conditions=None, t_span=(0, 300), t_points=1000):
    """
    Runs a simulation of the endothelial cell dynamics model.

    Parameters:
    tau_value: Value of shear stress
    initial_conditions: Initial values [E0, S0, T_avg0]
    t_span: Time span for simulation
    t_points: Number of time points

    Returns:
    t_eval: Time points
    solution: Solution object from solve_ivp
    """
    # Default initial conditions if none provided
    if initial_conditions is None:
        initial_conditions = [
            100,  # Initial healthy cells
            0,  # Initial senescent cells
            100,  # Initial telomere length (arbitrary units)
        ]

    params = {
        'r': 0.1,  # Proliferation rate
        'K': 10000,  # Carrying capacity
        'd_E': 0.01,  # Death rate of healthy cells
        'd_S': 0.05,  # Death rate of senescent cells
        'gamma_S': 0.0001,  # Senescence induction by senescent cells
        'tau': tau_value,  # Shear stress
        'T_crit': 20,  # Critical telomere length for senescence
        'T_loss': 1.0  # Telomere loss per division
    }

    # Time points for evaluation
    t_eval = np.linspace(t_span[0], t_span[1], t_points)

    # Solve the system of differential equations
    solution = solve_ivp(
        lambda t, y: endothelial_cell_dynamics(t, y, params),
        t_span,
        initial_conditions,
        t_eval=t_eval,
        method='RK45'
    )

    return t_eval, solution


# Run simulations for different shear stress values
tau_values = [0, 5, 10, 15, 20]
results = []

for tau in tau_values:
    t_eval, solution = run_simulation(tau)
    results.append((tau, t_eval, solution))

# Plot the results
plt.figure(figsize=(15, 10))

# Plot healthy cells
plt.subplot(2, 2, 1)
for tau, t_eval, solution in results:
    plt.plot(t_eval, solution.y[0], label=f'τ={tau}')
plt.xlabel('Time')
plt.ylabel('Cell Count')
plt.title('Healthy Endothelial Cells')
plt.legend()
plt.grid(True)

# Plot senescent cells
plt.subplot(2, 2, 2)
for tau, t_eval, solution in results:
    plt.plot(t_eval, solution.y[1], label=f'τ={tau}')
plt.xlabel('Time')
plt.ylabel('Cell Count')
plt.title('Senescent Cells')
plt.legend()
plt.grid(True)

# Plot telomere length
plt.subplot(2, 2, 3)
for tau, t_eval, solution in results:
    plt.plot(t_eval, solution.y[2], label=f'τ={tau}')
plt.xlabel('Time')
plt.ylabel('Telomere Length')
plt.title('Average Telomere Length')
plt.legend()
plt.grid(True)
plt.axhline(y=20, color='r', linestyle='--', label='Critical Length')

# Plot total cells
plt.subplot(2, 2, 4)
for tau, t_eval, solution in results:
    total_cells = solution.y[0] + solution.y[1]
    plt.plot(t_eval, total_cells, label=f'τ={tau}')
plt.xlabel('Time')
plt.ylabel('Cell Count')
plt.title('Total Cell Population')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot percentage of senescent cells
plt.figure(figsize=(10, 6))
for tau, t_eval, solution in results:
    total_cells = solution.y[0] + solution.y[1]
    senescent_percentage = 100 * solution.y[1] / np.maximum(total_cells, 1)  # Avoid division by zero
    plt.plot(t_eval, senescent_percentage, label=f'τ={tau}')
plt.xlabel('Time')
plt.ylabel('Percentage')
plt.title('Percentage of Senescent Cells Over Time')
plt.legend()
plt.grid(True)
plt.show()


# Explore parameter sensitivity
def analyze_parameter_sensitivity():
    """
    Analyzes how changing key parameters affects the cell dynamics.
    """
    # Define parameter variations
    # We'll vary one parameter at a time while keeping others at baseline
    baseline_params = {
        'r': 0.1,  # Proliferation rate
        'K': 10000,  # Carrying capacity
        'd_E': 0.01,  # Death rate of healthy cells
        'd_S': 0.05,  # Death rate of senescent cells
        'gamma_S': 0.0001,  # Senescence induction by senescent cells
        'tau': 10,  # Moderate shear stress
        'T_crit': 20,  # Critical telomere length for senescence
        'T_loss': 1.0  # Telomere loss per division
    }

    # Parameters to vary and their ranges
    parameter_variations = {
        'gamma_S': [0.00005, 0.0001, 0.0002],  # Senescence induction strength
        'T_loss': [0.5, 1.0, 2.0],  # Telomere loss per division
        'd_S': [0.02, 0.05, 0.1]  # Death rate of senescent cells
    }

    # Initial conditions
    initial_conditions = [100, 0, 100]

    # Time span
    t_span = (0, 300)
    t_points = 500
    t_eval = np.linspace(t_span[0], t_span[1], t_points)

    # Analyze each parameter
    for param_name, param_values in parameter_variations.items():
        plt.figure(figsize=(15, 10))

        # Plot healthy cells
        plt.subplot(2, 2, 1)
        for param_value in param_values:
            # Create a copy of baseline parameters and update the specific parameter
            params = baseline_params.copy()
            params[param_name] = param_value

            # Run simulation
            solution = solve_ivp(
                lambda t, y: endothelial_cell_dynamics(t, y, params),
                t_span,
                initial_conditions,
                t_eval=t_eval,
                method='RK45'
            )

            # Plot results
            plt.plot(t_eval, solution.y[0], label=f'{param_name}={param_value}')

        plt.xlabel('Time')
        plt.ylabel('Cell Count')
        plt.title(f'Healthy Cells vs {param_name}')
        plt.legend()
        plt.grid(True)

        # Plot senescent cells
        plt.subplot(2, 2, 2)
        for param_value in param_values:
            # Create a copy of baseline parameters and update the specific parameter
            params = baseline_params.copy()
            params[param_name] = param_value

            # Run simulation
            solution = solve_ivp(
                lambda t, y: endothelial_cell_dynamics(t, y, params),
                t_span,
                initial_conditions,
                t_eval=t_eval,
                method='RK45'
            )

            # Plot results
            plt.plot(t_eval, solution.y[1], label=f'{param_name}={param_value}')

        plt.xlabel('Time')
        plt.ylabel('Cell Count')
        plt.title(f'Senescent Cells vs {param_name}')
        plt.legend()
        plt.grid(True)

        # Plot telomere length
        plt.subplot(2, 2, 3)
        for param_value in param_values:
            # Create a copy of baseline parameters and update the specific parameter
            params = baseline_params.copy()
            params[param_name] = param_value

            # Run simulation
            solution = solve_ivp(
                lambda t, y: endothelial_cell_dynamics(t, y, params),
                t_span,
                initial_conditions,
                t_eval=t_eval,
                method='RK45'
            )

            # Plot results
            plt.plot(t_eval, solution.y[2], label=f'{param_name}={param_value}')

        plt.xlabel('Time')
        plt.ylabel('Telomere Length')
        plt.title(f'Telomere Length vs {param_name}')
        plt.legend()
        plt.grid(True)
        plt.axhline(y=20, color='r', linestyle='--', label='Critical Length')

        # Plot percentage of senescent cells
        plt.subplot(2, 2, 4)
        for param_value in param_values:
            # Create a copy of baseline parameters and update the specific parameter
            params = baseline_params.copy()
            params[param_name] = param_value

            # Run simulation
            solution = solve_ivp(
                lambda t, y: endothelial_cell_dynamics(t, y, params),
                t_span,
                initial_conditions,
                t_eval=t_eval,
                method='RK45'
            )

            # Calculate percentage
            total_cells = solution.y[0] + solution.y[1]
            senescent_percentage = 100 * solution.y[1] / np.maximum(total_cells, 1)

            # Plot results
            plt.plot(t_eval, senescent_percentage, label=f'{param_name}={param_value}')

        plt.xlabel('Time')
        plt.ylabel('Percentage')
        plt.title(f'Senescent Cell Percentage vs {param_name}')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# Run the parameter sensitivity analysis
analyze_parameter_sensitivity()


# Comprehensive analysis of shear stress effects
def analyze_shear_stress_effects():
    """
    Analyzes how different levels of shear stress affect the model variables.
    """
    tau_range = np.linspace(0, 30, 31)
    final_healthy_cells = []
    final_senescent_cells = []
    final_telomere_length = []
    senescent_percentage = []

    for tau in tau_range:
        t_eval, solution = run_simulation(tau, t_span=(0, 500))

        # Get the final state
        final_E = solution.y[0][-1]
        final_S = solution.y[1][-1]
        final_T = solution.y[2][-1]

        # Calculate percentage
        total_cells = final_E + final_S
        if total_cells > 0:
            sen_pct = 100 * final_S / total_cells
        else:
            sen_pct = 0

        # Store results
        final_healthy_cells.append(final_E)
        final_senescent_cells.append(final_S)
        final_telomere_length.append(final_T)
        senescent_percentage.append(sen_pct)

    # Plot results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(tau_range, final_healthy_cells, 'o-', color='blue')
    plt.plot(tau_range, final_senescent_cells, 'o-', color='red')
    plt.xlabel('Shear Stress (τ)')
    plt.ylabel('Cell Count')
    plt.title('Final Cell Counts vs. Shear Stress')
    plt.legend(['Healthy Cells', 'Senescent Cells'])
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(tau_range, np.array(final_healthy_cells) + np.array(final_senescent_cells), 'o-', color='green')
    plt.xlabel('Shear Stress (τ)')
    plt.ylabel('Cell Count')
    plt.title('Total Cell Count vs. Shear Stress')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(tau_range, senescent_percentage, 'o-', color='purple')
    plt.xlabel('Shear Stress (τ)')
    plt.ylabel('Percentage')
    plt.title('Percentage of Senescent Cells vs. Shear Stress')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(tau_range, final_telomere_length, 'o-', color='orange')
    plt.xlabel('Shear Stress (τ)')
    plt.ylabel('Telomere Length')
    plt.title('Final Telomere Length vs. Shear Stress')
    plt.grid(True)
    plt.axhline(y=20, color='r', linestyle='--', label='Critical Length')

    plt.tight_layout()
    plt.show()


# Run the analysis of shear stress effects
analyze_shear_stress_effects()