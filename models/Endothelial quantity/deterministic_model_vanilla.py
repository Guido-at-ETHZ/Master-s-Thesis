import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Define a more realistic shear stress effect function
def shear_stress_effect(tau):
    """
    Models how shear stress affects senescence rate.
    Low shear stress: minimal effect
    Moderate shear stress: slightly increased effect
    High shear stress: significantly increased effect

    Returns a value between 0 and 1 representing the effect
    """
    # Baseline rate with no shear stress
    base_rate = 0.002

    # Additional rate due to shear stress (non-linear relationship)
    if tau <= 10:  # Low shear stress
        additional_rate = tau * 0.0005  # Very minor effect
    elif tau <= 20:  # Moderate shear stress
        additional_rate = 0.005 + (tau - 10) * 0.001  # Slight increase
    else:  # High shear stress
        additional_rate = 0.015 + (tau - 20) * 0.005  # More rapid increase

    return base_rate + additional_rate


# Define the cell dynamics model with realistic growth and telomere-based replicative senescence
def endothelial_cell_dynamics(t, y, params):
    """
    System of differential equations for endothelial cell dynamics with
    division-based population structure.

    This model tracks:
    - Cell populations at different division stages (E_0, E_1, ..., E_N)
    - Senescent cells from telomere shortening (S_tel)
    - Senescent cells from stress-induced senescence (S_stress)
    """
    # Unpack parameters
    r = params['r']  # Base proliferation rate
    K = params['K']  # Carrying capacity
    d_E = params['d_E']  # Base death rate of healthy cells
    d_S_tel = params['d_S_tel']  # Death rate of telomere-induced senescent cells
    d_S_stress = params['d_S_stress']  # Death rate of stress-induced senescent cells
    gamma_S = params['gamma_S']  # Senescence induction by senescent cells
    tau = params['tau']  # Shear stress
    max_divisions = params['max_divisions']  # Maximum divisions before mandatory senescence

    # Calculate stress-induced senescence rate based on shear stress
    gamma_tau = shear_stress_effect(tau)

    # Extract state variables
    # Healthy cells at different division stages
    E = y[:max_divisions + 1]
    # Senescent cells by cause
    S_tel = y[max_divisions + 1]  # Telomere-induced senescent cells
    S_stress = y[max_divisions + 2]  # Stress-induced senescent cells

    # Total number of cells (for carrying capacity calculation)
    total_cells = np.sum(E) + S_tel + S_stress

    # Total number of senescent cells (for SASP effects)
    total_senescent = S_tel + S_stress

    # Initialize derivatives array
    dy = np.zeros_like(y)

    # Modified density-dependent inhibition that allows initial growth
    # Use a sigmoid function that allows growth until approaching capacity
    density_factor = 1 / (1 + np.exp(10 * (total_cells / K - 0.7)))

    # Equations for healthy cells at each division stage
    for i in range(max_divisions + 1):
        # Age-dependent death rate (increases with division age)
        death_rate = d_E * (1 + 0.03 * i)

        # Stress-induced senescence rate (slightly increases with division age)
        stress_senescence_rate = gamma_tau * (1 + 0.05 * i)

        # Senescence induction by SASP
        sasp_senescence_rate = gamma_S * total_senescent

        # Division capacity decreases with division age
        division_capacity = 1.0
        if i > max_divisions * 0.7:  # Reduction starts after 70% of max divisions
            division_capacity = 1.0 - 0.5 * ((i - max_divisions * 0.7) / (max_divisions * 0.3))

        # Calculate division terms
        if i == 0:
            # First group (E_0) - undivided cells
            division_out = r * E[i] * density_factor * division_capacity
            division_in = 0  # No division in for the first stage
        elif i < max_divisions:
            # Middle groups
            division_in = 2 * r * E[i - 1] * density_factor * (1.0 if i - 1 < max_divisions * 0.7 else
                                                               1.0 - 0.5 * ((i - 1 - max_divisions * 0.7) / (
                                                                           max_divisions * 0.3)))
            division_out = r * E[i] * density_factor * division_capacity
        else:
            # Last group - at maximum division count
            division_in = 2 * r * E[i - 1] * density_factor * (1.0 if i - 1 < max_divisions * 0.7 else
                                                               1.0 - 0.5 * ((i - 1 - max_divisions * 0.7) / (
                                                                           max_divisions * 0.3)))
            division_out = r * E[i] * density_factor * division_capacity  # These cells enter senescence

        # Cell loss due to death
        death_term = death_rate * E[i]

        # Cell loss due to stress-induced senescence
        stress_senescence_term = stress_senescence_rate * E[i]

        # Cell loss due to SASP-induced senescence
        sasp_senescence_term = sasp_senescence_rate * E[i]

        # Combine all terms for this division stage
        if i == max_divisions:
            # Last division stage - all divisions lead to senescence
            dy[i] = division_in - death_term - stress_senescence_term - sasp_senescence_term - division_out
        else:
            dy[i] = division_in - division_out - death_term - stress_senescence_term - sasp_senescence_term

        # Add contributions to senescent populations
        if i == max_divisions:
            # Telomere-induced senescence from cells at max divisions
            dy[max_divisions + 1] += division_out

        # Stress-induced senescence from all division stages
        dy[max_divisions + 2] += stress_senescence_term + sasp_senescence_term

    # Equation for telomere-induced senescent cells
    dy[max_divisions + 1] += -d_S_tel * S_tel  # Death of telomere-senescent cells

    # Equation for stress-induced senescent cells
    dy[max_divisions + 2] += -d_S_stress * S_stress  # Death of stress-senescent cells

    return dy


# Run simulation with different parameters
def run_simulation(tau_value, max_divisions=15, initial_conditions=None, t_span=(0, 600), t_points=1000):
    """
    Runs a simulation of the endothelial cell dynamics model.
    """
    # Default initial conditions if none provided
    if initial_conditions is None:
        # Start with all cells in the undivided state (E_0)
        initial_conditions = np.zeros(max_divisions + 3)
        initial_conditions[0] = 200  # Initial population in E_0

    params = {
        'r': 0.06,  # Proliferation rate (increased for faster growth)
        'K': 3000,  # Carrying capacity
        'd_E': 0.008,  # Death rate of healthy cells (reduced for longer survival)
        'd_S_tel': 0.02,  # Death rate of telomere-induced senescent cells
        'd_S_stress': 0.025,  # Death rate of stress-induced senescent cells
        'gamma_S': 0.00005,  # Senescence induction by senescent cells (reduced)
        'tau': tau_value,  # Shear stress
        'max_divisions': max_divisions,  # Max divisions before mandatory senescence
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


# Function to analyze results
def analyze_results(t_eval, solution, max_divisions):
    """
    Analyze the results of the simulation.
    """
    # Total healthy cells across all division stages
    E_total = np.sum(solution.y[:max_divisions + 1], axis=0)

    # Senescent cells by cause
    S_tel = solution.y[max_divisions + 1]  # Telomere-induced senescent cells
    S_stress = solution.y[max_divisions + 2]  # Stress-induced senescent cells
    S_total = S_tel + S_stress  # Total senescent cells

    # Total cell population
    total_cells = E_total + S_total

    # Calculate statistics
    senescent_fraction = S_total / np.maximum(total_cells, 1e-10)  # Avoid division by zero
    tel_fraction_of_senescent = S_tel / np.maximum(S_total, 1e-10)

    # Calculate average division age of healthy cells
    weighted_sum = np.zeros_like(E_total)
    for i in range(max_divisions + 1):
        weighted_sum += i * solution.y[i]
    avg_division_age = weighted_sum / np.maximum(E_total, 1e-10)

    # Calculate "telomere length" based on average division age
    # Start with maximum length and decrease based on average divisions
    max_telomere = 100
    min_telomere = 20
    telomere_length = max_telomere - (max_telomere - min_telomere) * (avg_division_age / max_divisions)

    return {
        't': t_eval,
        'E_total': E_total,
        'S_tel': S_tel,
        'S_stress': S_stress,
        'S_total': S_total,
        'total_cells': total_cells,
        'senescent_fraction': senescent_fraction,
        'tel_fraction_of_senescent': tel_fraction_of_senescent,
        'avg_division_age': avg_division_age,
        'telomere_length': telomere_length
    }


# Run simulations for different shear stress values
def run_multiple_simulations(tau_values, max_divisions=15):
    """
    Run simulations for different shear stress values and analyze results.
    """
    results = []

    for tau in tau_values:
        t_eval, solution = run_simulation(tau, max_divisions=max_divisions)
        result = analyze_results(t_eval, solution, max_divisions)
        result['tau'] = tau
        results.append(result)

    return results


# Plot the simulation results
def plot_results(results, max_divisions=15):
    """
    Plot comprehensive results from multiple simulations.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Colors for different tau values
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    # Plot total cells
    ax = axes[0, 0]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['total_cells'], color=colors[i],
                label=f'τ={result["tau"]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Total Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot healthy cells
    ax = axes[0, 1]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['E_total'], color=colors[i],
                label=f'τ={result["tau"]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Healthy Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot senescent cells
    ax = axes[0, 2]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['S_total'], color=colors[i],
                label=f'τ={result["tau"]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Senescent Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot senescent fraction
    ax = axes[1, 0]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['senescent_fraction'], color=colors[i],
                label=f'τ={result["tau"]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')
    ax.set_title('Fraction of Senescent Cells')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # Plot telomere-induced vs. stress-induced senescent cells
    ax = axes[1, 1]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['S_tel'], color=colors[i], linestyle='-',
                label=f'Telomere τ={result["tau"]}')
        ax.plot(result['t'], result['S_stress'], color=colors[i], linestyle='--',
                label=f'Stress τ={result["tau"]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Senescent Cells by Cause')
    ax.legend()
    ax.grid(True)

    # Plot fraction of senescent cells that are telomere-induced
    ax = axes[1, 2]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['tel_fraction_of_senescent'], color=colors[i],
                label=f'τ={result["tau"]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')
    ax.set_title('Fraction of Senescent Cells from Telomere Shortening')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # Plot telomere length
    ax = axes[2, 0]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['telomere_length'], color=colors[i],
                label=f'τ={result["tau"]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Telomere Length')
    ax.set_title('Average Telomere Length of Healthy Cells')
    ax.axhline(y=20, color='r', linestyle='--', label='Critical Length')
    ax.legend()
    ax.grid(True)

    # Plot average division age
    ax = axes[2, 1]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['avg_division_age'], color=colors[i],
                label=f'τ={result["tau"]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Divisions')
    ax.set_title('Average Division Age of Healthy Cells')
    ax.legend()
    ax.grid(True)

    # Plot distribution of cells by division age at final time point for one tau value
    ax = axes[2, 2]
    tau_to_show = results[len(results) // 2]['tau']  # Middle tau value

    t_eval, solution = run_simulation(tau_to_show, max_divisions=max_divisions)

    # Get final distribution
    final_distribution = solution.y[:max_divisions + 1, -1]
    division_ages = np.arange(max_divisions + 1)

    # Normalize to percentage
    total_healthy = np.sum(final_distribution)
    if total_healthy > 0:
        normalized_distribution = 100 * final_distribution / total_healthy
    else:
        normalized_distribution = np.zeros_like(final_distribution)

    ax.bar(division_ages, normalized_distribution, color='blue')
    ax.set_xlabel('Division Age')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Distribution of Healthy Cells by Division Age (τ={tau_to_show})')
    ax.grid(True)

    plt.suptitle('Endothelial Cell Dynamics with Different Shear Stress Levels', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    return fig


# Run simulations and plot results
max_divisions = 15
tau_values = [0, 5, 10, 15, 20]
results = run_multiple_simulations(tau_values, max_divisions)
fig = plot_results(results, max_divisions)
plt.show()


# Plot just cell populations to focus on growth-decline pattern
def plot_cell_populations(results):
    """
    Focus on plotting cell population dynamics.
    """
    plt.figure(figsize=(12, 8))

    # Colors for different tau values
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    # Plot healthy, senescent, and total cells for each tau value
    for i, result in enumerate(results):
        tau = result['tau']
        plt.plot(result['t'], result['E_total'], color=colors[i], linestyle='-',
                 label=f'Healthy τ={tau}')
        plt.plot(result['t'], result['S_total'], color=colors[i], linestyle='--',
                 label=f'Senescent τ={tau}')
        plt.plot(result['t'], result['total_cells'], color=colors[i], linestyle=':',
                 label=f'Total τ={tau}')

    plt.xlabel('Time')
    plt.ylabel('Cell Count')
    plt.title('Endothelial Cell Population Dynamics')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Plot cell populations
plot_cell_populations(results)