import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


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


# Define senolytic effect function
def senolytic_effect(concentration, efficacy_factor=1.0):
    """
    Models how senolytic concentration affects senescent cell death rate.

    Args:
        concentration: Concentration of senolytics
        efficacy_factor: Efficacy multiplier for the given senescent cell type

    Returns:
        Additional death rate induced by senolytics
    """
    # No effect at zero concentration
    if concentration <= 0:
        return 0

    # Sigmoid response curve to model drug effect
    # Max effect approaches 0.15 at high concentrations (increased from 0.1)
    max_effect = 0.15 * efficacy_factor

    # EC50 = 5 (concentration at which half the maximum effect is achieved)
    ec50 = 5

    # Hill coefficient = 3 (increased from 2 for more non-linearity)
    hill = 3

    # Calculate effect using Hill equation
    effect = max_effect * (concentration ** hill) / (ec50 ** hill + concentration ** hill)

    return effect


# Define the cell dynamics model with senolytic effects
def endothelial_cell_dynamics(t, y, params):
    """
    System of differential equations for endothelial cell dynamics with
    division-based population structure and senolytic effects.

    This model tracks:
    - Cell populations at different division stages (E_0, E_1, ..., E_N)
    - Senescent cells from telomere shortening (S_tel)
    - Senescent cells from stress-induced senescence (S_stress)
    """
    # Unpack parameters
    r = params['r']  # Base proliferation rate
    K = params['K']  # Carrying capacity
    d_E = params['d_E']  # Base death rate of healthy cells
    d_S_tel = params['d_S_tel']  # Base death rate of telomere-induced senescent cells
    d_S_stress = params['d_S_stress']  # Base death rate of stress-induced senescent cells
    gamma_S = params['gamma_S']  # Senescence induction by senescent cells
    tau = params['tau']  # Shear stress
    max_divisions = params['max_divisions']  # Maximum divisions before mandatory senescence
    senolytic_conc = params['senolytic_conc']  # Concentration of senolytics
    sen_efficacy_tel = params['sen_efficacy_tel']  # Efficacy of senolytics on telomere-induced senescent cells
    sen_efficacy_stress = params['sen_efficacy_stress']  # Efficacy of senolytics on stress-induced senescent cells

    # Calculate stress-induced senescence rate based on shear stress
    gamma_tau = shear_stress_effect(tau)

    # Calculate senolytic effects on senescent cell death rates
    senolytic_effect_tel = senolytic_effect(senolytic_conc, sen_efficacy_tel)
    senolytic_effect_stress = senolytic_effect(senolytic_conc, sen_efficacy_stress)

    # Add toxicity to healthy cells (small effect compared to senescent cells)
    healthy_cell_toxicity = 0.001 * senolytic_conc

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
        # Age-dependent death rate (increases with division age) + senolytic toxicity
        death_rate = d_E * (1 + 0.03 * i) + healthy_cell_toxicity

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
            division_capacity_prev = 1.0
            if (i - 1) > max_divisions * 0.7:  # For previous group
                division_capacity_prev = 1.0 - 0.5 * ((i - 1 - max_divisions * 0.7) / (max_divisions * 0.3))
            division_in = 2 * r * E[i - 1] * density_factor * division_capacity_prev
            division_out = r * E[i] * density_factor * division_capacity
        else:
            # Last group - at maximum division count
            division_capacity_prev = 1.0
            if (i - 1) > max_divisions * 0.7:  # For previous group
                division_capacity_prev = 1.0 - 0.5 * ((i - 1 - max_divisions * 0.7) / (max_divisions * 0.3))
            division_in = 2 * r * E[i - 1] * density_factor * division_capacity_prev
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

    # Equation for telomere-induced senescent cells with senolytic effects
    dy[max_divisions + 1] += -(d_S_tel + senolytic_effect_tel) * S_tel  # Death of telomere-senescent cells

    # Equation for stress-induced senescent cells with senolytic effects
    dy[max_divisions + 2] += -(d_S_stress + senolytic_effect_stress) * S_stress  # Death of stress-senescent cells

    return dy


# Run simulation with different parameters
def run_simulation(tau_value, senolytic_conc=0, max_divisions=15, initial_conditions=None, t_span=(0, 600),
                   t_points=1000):
    """
    Runs a simulation of the endothelial cell dynamics model with senolytic effects.
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
        'senolytic_conc': senolytic_conc,  # Concentration of senolytics
        'sen_efficacy_tel': 1.0,  # Efficacy of senolytics on telomere-induced senescent cells
        'sen_efficacy_stress': 1.2,  # Efficacy of senolytics on stress-induced senescent cells (slightly higher)
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

    return t_eval, solution, params


# Run simulation with pulsed senolytic treatment - UPDATED VERSION
def run_pulsed_treatment_simulation(tau_value, senolytic_conc=10, pulse_interval=50, pulse_duration=10,
                                    max_divisions=15, t_span=(0, 600), t_points=1000):
    """
    Runs a simulation with pulsed senolytic treatment.

    Args:
        tau_value: Shear stress value
        senolytic_conc: Peak concentration during pulses
        pulse_interval: Time between start of each pulse
        pulse_duration: Duration of each pulse
        max_divisions: Maximum number of divisions
        t_span: Time span for simulation
        t_points: Number of time points for evaluation

    Returns:
        Tuple of (t_eval, solution, params)
    """
    # Define the final time points we want to evaluate at
    t_eval = np.linspace(t_span[0], t_span[1], t_points)

    # Initialize arrays to accumulate results from segments
    initial_conditions = np.zeros(max_divisions + 3)
    initial_conditions[0] = 200  # Initial population in E_0

    # Create a time history of senolytic concentration
    senolytic_history = np.zeros(t_points)

    # Set the senolytic concentration for each time point
    for i in range(t_points):
        # Calculate the corresponding time
        t = t_eval[i]
        # Determine if this time is during a pulse
        is_pulse_period = (t % pulse_interval) < pulse_duration
        senolytic_history[i] = senolytic_conc if is_pulse_period else 0

    # Create a modified version of endothelial_cell_dynamics that takes time-varying senolytic concentration
    def dynamics_with_time_varying_senolytics(t, y, params):
        # Find the closest index in t_eval
        idx = np.argmin(np.abs(t_eval - t))
        # Set the senolytic concentration for this time point
        params = params.copy()  # Create a copy to avoid modifying the original
        params['senolytic_conc'] = senolytic_history[idx]
        # Call the original dynamics function
        return endothelial_cell_dynamics(t, y, params)

    # Set up parameters
    params = {
        'r': 0.06,
        'K': 3000,
        'd_E': 0.008,
        'd_S_tel': 0.02,
        'd_S_stress': 0.025,
        'gamma_S': 0.00005,
        'tau': tau_value,
        'max_divisions': max_divisions,
        'senolytic_conc': 0,  # This will be overridden in dynamics_with_time_varying_senolytics
        'sen_efficacy_tel': 1.0,
        'sen_efficacy_stress': 1.2,
    }

    # Solve the system of differential equations
    solution = solve_ivp(
        lambda t, y: dynamics_with_time_varying_senolytics(t, y, params),
        t_span,
        initial_conditions,
        t_eval=t_eval,
        method='RK45'
    )

    return t_eval, solution, params


# Function to analyze results
def analyze_results(t_eval, solution, max_divisions, params=None):
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

    result = {
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

    # Add parameters to results if provided
    if params:
        result.update({
            'tau': params['tau'],
            'senolytic_conc': params['senolytic_conc']
        })

    return result


# Run simulations for different senolytic concentrations
def run_multiple_senolytic_simulations(tau_value, senolytic_concs, max_divisions=15):
    """
    Run simulations for different senolytic concentrations and analyze results.
    """
    results = []

    for conc in senolytic_concs:
        t_eval, solution, params = run_simulation(tau_value, senolytic_conc=conc, max_divisions=max_divisions)
        result = analyze_results(t_eval, solution, max_divisions, params)
        results.append(result)

    return results


# Define optimization function
def optimize_senolytic_concentration(tau_value, max_divisions=15, t_span=(0, 600)):
    """
    Finds the optimal senolytic concentration for a given shear stress value.

    Returns:
        Tuple of (optimal concentration, optimization results)
    """

    def objective_function(senolytic_conc):
        """
        Objective function to minimize.

        We want to:
        1. Maximize healthy cells
        2. Minimize senescent cells
        3. Consider long-term cell population sustainability

        Returns a value to minimize (negative of the "health score")
        """
        # Run simulation with the given senolytic concentration
        t_eval, solution, _ = run_simulation(
            tau_value,
            senolytic_conc=senolytic_conc,
            max_divisions=max_divisions,
            t_span=t_span
        )

        # Analyze results
        result = analyze_results(t_eval, solution, max_divisions)

        # Extract key metrics
        healthy_cells = result['E_total'][-1]  # Final healthy cell count
        senescent_fraction = result['senescent_fraction'][-1]  # Final senescent fraction

        # Calculate area under the curve for healthy cells (approximation)
        healthy_auc = np.trapezoid(result['E_total'], result['t'])

        # Calculate average healthy cells in the last 10% of the simulation
        last_index = int(0.9 * len(result['t']))
        avg_late_healthy = np.mean(result['E_total'][last_index:])

        # Composite health score (to be maximized)
        # Weight factors can be adjusted based on what's most important
        health_score = (
                1.0 * avg_late_healthy +  # Weight for final healthy cell population
                0.5 * healthy_auc / t_span[1] +  # Weight for healthy cell AUC (normalized by time)
                -2.0 * senescent_fraction * avg_late_healthy
        # Penalty for senescent fraction (proportional to population)
        )

        # Return negative score since we're minimizing
        return -health_score

    # Run optimization to find the best concentration
    # Bounds: 0 to 20 units of senolytic concentration
    result = minimize_scalar(
        objective_function,
        bounds=(0, 20),
        method='bounded',
        options={'xatol': 0.1}  # Tolerance for termination
    )

    return result.x, result


# Plot the simulation results with senolytic effects
def plot_senolytic_results(results, max_divisions=15):
    """
    Plot comprehensive results from multiple simulations with different senolytic concentrations.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Colors for different senolytic concentrations
    colors = plt.cm.plasma(np.linspace(0, 1, len(results)))

    # Plot total cells
    ax = axes[0, 0]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['total_cells'], color=colors[i],
                label=f'Senolytic={result["senolytic_conc"]:.2f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title(f'Total Cell Population (τ={results[0]["tau"]})')
    ax.legend()
    ax.grid(True)

    # Plot healthy cells
    ax = axes[0, 1]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['E_total'], color=colors[i],
                label=f'Senolytic={result["senolytic_conc"]:.2f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Healthy Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot senescent cells
    ax = axes[0, 2]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['S_total'], color=colors[i],
                label=f'Senolytic={result["senolytic_conc"]:.2f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Senescent Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot senescent fraction
    ax = axes[1, 0]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['senescent_fraction'], color=colors[i],
                label=f'Senolytic={result["senolytic_conc"]:.2f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')
    ax.set_title('Fraction of Senescent Cells')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # Plot telomere-induced vs. stress-induced senescent cells for middle concentration
    ax = axes[1, 1]
    middle_index = len(results) // 2
    result = results[middle_index]
    ax.plot(result['t'], result['S_tel'], color='blue', linestyle='-',
            label=f'Telomere-induced (Senolytic={result["senolytic_conc"]:.2f})')
    ax.plot(result['t'], result['S_stress'], color='red', linestyle='-',
            label=f'Stress-induced (Senolytic={result["senolytic_conc"]:.2f})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Senescent Cells by Cause')
    ax.legend()
    ax.grid(True)

    # Plot fraction of senescent cells that are telomere-induced
    ax = axes[1, 2]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['tel_fraction_of_senescent'], color=colors[i],
                label=f'Senolytic={result["senolytic_conc"]:.2f}')
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
                label=f'Senolytic={result["senolytic_conc"]:.2f}')
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
                label=f'Senolytic={result["senolytic_conc"]:.2f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Divisions')
    ax.set_title('Average Division Age of Healthy Cells')
    ax.legend()
    ax.grid(True)

    # Plot distribution of cells by division age at final time point for optimal senolytic
    ax = axes[2, 2]
    optimal_index = np.argmax([result['E_total'][-1] for result in results])
    optimal_conc = results[optimal_index]['senolytic_conc']

    t_eval, solution, _ = run_simulation(results[0]['tau'], senolytic_conc=optimal_conc, max_divisions=max_divisions)

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
    ax.set_title(f'Distribution of Healthy Cells (Senolytic={optimal_conc:.2f})')
    ax.grid(True)

    plt.suptitle(f'Endothelial Cell Dynamics with Senolytics (τ={results[0]["tau"]})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])


# Function to optimize pulsed treatment parameters
# Function to optimize pulsed treatment parameters
def optimize_pulsed_treatment(tau_value, max_divisions=15):
    """
    Finds optimal parameters for pulsed senolytic treatment.

    Args:
        tau_value: Shear stress value
        max_divisions: Maximum number of cell divisions

    Returns:
        Tuple of (optimal_conc, optimal_interval, optimal_duration, fig)
    """
    # Define parameter ranges
    conc_range = [5, 10, 15]
    interval_range = [30, 50, 70]
    duration_range = [5, 10, 15]

    # Initialize results matrix
    results = np.zeros((len(conc_range), len(interval_range), len(duration_range)))

    # Run simulations for all parameter combinations
    for i, conc in enumerate(conc_range):
        for j, interval in enumerate(interval_range):
            for k, duration in enumerate(duration_range):
                # Run simulation with these parameters
                t_eval, solution, _ = run_pulsed_treatment_simulation(
                    tau_value,
                    senolytic_conc=conc,
                    pulse_interval=interval,
                    pulse_duration=duration,
                    max_divisions=max_divisions
                )

                # Analyze results
                result = analyze_results(t_eval, solution, max_divisions)

                # Calculate health score
                healthy_cells = result['E_total'][-1]  # Final healthy cell count
                senescent_fraction = result['senescent_fraction'][-1]  # Final senescent fraction

                # Calculate area under the curve for healthy cells (approximation)
                healthy_auc = np.trapezoid(result['E_total'], result['t'])

                # Calculate average healthy cells in the last 10% of the simulation
                last_index = int(0.9 * len(result['t']))
                avg_late_healthy = np.mean(result['E_total'][last_index:])

                # Composite health score (to be maximized)
                health_score = (
                        1.0 * avg_late_healthy +  # Weight for final healthy cell population
                        0.5 * healthy_auc / t_eval[-1] +  # Weight for healthy cell AUC (normalized by time)
                        -2.0 * senescent_fraction * avg_late_healthy  # Penalty for senescent fraction
                )

                # Store health score
                results[i, j, k] = health_score

    # Find optimal parameters
    max_idx = np.unravel_index(np.argmax(results), results.shape)
    optimal_conc = conc_range[max_idx[0]]
    optimal_interval = interval_range[max_idx[1]]
    optimal_duration = duration_range[max_idx[2]]

    # Create heatmap for visualization (averaging over duration)
    avg_results = np.mean(results, axis=2)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(avg_results, cmap='viridis', origin='lower')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Average Health Score', rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(interval_range)))
    ax.set_yticks(np.arange(len(conc_range)))
    ax.set_xticklabels(interval_range)
    ax.set_yticklabels(conc_range)

    # Add labels and title
    ax.set_xlabel('Pulse Interval')
    ax.set_ylabel('Senolytic Concentration')
    ax.set_title(f'Optimization of Pulsed Treatment (τ={tau_value})')

    # Add optimal point marker
    ax.plot(max_idx[1], max_idx[0], 'r*', markersize=15,
            label=f'Optimal: C={optimal_conc}, I={optimal_interval}, D={optimal_duration}')
    ax.legend(loc='upper left')

    # Return the optimal parameters and the figure
    return optimal_conc, optimal_interval, optimal_duration, fig
def plot_optimization_results(tau_value, max_divisions=15, conc_range=np.linspace(0, 20, 21)):
    """
    Plot the optimization landscape for senolytic concentration.
    """
    scores = []
    healthy_counts = []
    senescent_fractions = []

    for conc in conc_range:
        # Run simulation
        t_eval, solution, _ = run_simulation(tau_value, senolytic_conc=conc, max_divisions=max_divisions)
        result = analyze_results(t_eval, solution, max_divisions)

        # Extract metrics
        healthy_cells = result['E_total'][-1]
        senescent_fraction = result['senescent_fraction'][-1]

        # Calculate area under the curve for healthy cells
        healthy_auc = np.trapezoid(result['E_total'], result['t'])

        # Calculate average healthy cells in the last 10% of the simulation
        last_index = int(0.9 * len(result['t']))
        avg_late_healthy = np.mean(result['E_total'][last_index:])

        # Calculate health score
        health_score = (
                1.0 * avg_late_healthy +
                0.5 * healthy_auc / t_eval[-1] +
                -2.0 * senescent_fraction * avg_late_healthy
        )

        scores.append(health_score)
        healthy_counts.append(healthy_cells)
        senescent_fractions.append(senescent_fraction)

    # Find optimal concentration
    optimal_idx = np.argmax(scores)
    optimal_conc = conc_range[optimal_idx]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot health score
    ax = axes[0]
    ax.plot(conc_range, scores, 'o-', color='blue')
    ax.axvline(x=optimal_conc, color='red', linestyle='--',
               label=f'Optimal = {optimal_conc:.2f}')
    ax.set_xlabel('Senolytic Concentration')
    ax.set_ylabel('Health Score')
    ax.set_title('Optimization Landscape')
    ax.legend()
    ax.grid(True)

    # Plot healthy cell count
    ax = axes[1]
    ax.plot(conc_range, healthy_counts, 'o-', color='green')
    ax.axvline(x=optimal_conc, color='red', linestyle='--')
    ax.set_xlabel('Senolytic Concentration')
    ax.set_ylabel('Healthy Cell Count')
    ax.set_title('Final Healthy Cell Count')
    ax.grid(True)

    # Plot senescent fraction
    ax = axes[2]
    ax.plot(conc_range, senescent_fractions, 'o-', color='purple')
    ax.axvline(x=optimal_conc, color='red', linestyle='--')
    ax.set_xlabel('Senolytic Concentration')
    ax.set_ylabel('Senescent Cell Fraction')
    ax.set_title('Final Senescent Cell Fraction')
    ax.grid(True)

    plt.suptitle(f'Optimization of Senolytic Concentration (τ={tau_value})', fontsize=16)
    plt.tight_layout()

    return fig, optimal_conc


# Function to run all analyses and generate plots
# Function to run all analyses and generate plots
def analyze_senolytic_effects(tau_values=[5, 15], senolytic_concs=[0, 2, 5, 10], max_divisions=15):
    """
    Run comprehensive analysis of senolytic effects for different shear stress values.
    """
    results = {}
    optimals = {}

    # Run simulations and create plots for each tau value
    for tau in tau_values:
        # Run simulations with different senolytic concentrations
        sim_results = run_multiple_senolytic_simulations(tau, senolytic_concs, max_divisions)
        results[tau] = sim_results

        # Create a figure and plot results for this tau value
        plt.figure(figsize=(18, 15))  # Create figure explicitly with the same size as in plot_senolytic_results
        plot_senolytic_results(sim_results, max_divisions)
        plt.savefig(f'senolytic_results_tau_{tau}.png')
        plt.close()  # Close the figure to free memory

        # Optimize senolytic concentration
        opt_conc, opt_result = optimize_senolytic_concentration(tau, max_divisions)
        optimals[tau] = (opt_conc, opt_result)

        # Plot optimization results
        fig_opt, _ = plot_optimization_results(tau, max_divisions)
        fig_opt.savefig(f'senolytic_optimization_tau_{tau}.png')
        plt.close(fig_opt)  # Close the figure to free memory

        print(f"Optimal senolytic concentration for τ={tau}: {opt_conc:.2f}")

    return results, optimals

# Function to compare continuous vs. pulsed senolytic treatment
def compare_treatment_strategies(tau_value, max_divisions=15):
    """
    Compares continuous senolytic treatment with pulsed treatment.

    Args:
        tau_value: Shear stress value
        max_divisions: Maximum number of cell divisions

    Returns:
        Figure object with the comparison plots
    """
    # Find optimal senolytic concentration
    opt_conc, _ = optimize_senolytic_concentration(tau_value, max_divisions)

    # Run simulation with no treatment
    t_no_treat, sol_no_treat, _ = run_simulation(
        tau_value,
        senolytic_conc=0,
        max_divisions=max_divisions
    )

    # Run simulation with continuous treatment
    t_continuous, sol_continuous, _ = run_simulation(
        tau_value,
        senolytic_conc=opt_conc,
        max_divisions=max_divisions
    )

    # Run simulation with pulsed treatment (higher peak concentration)
    t_pulsed, sol_pulsed, _ = run_pulsed_treatment_simulation(
        tau_value,
        senolytic_conc=opt_conc * 2,  # Higher peak concentration
        pulse_interval=50,  # Interval between pulses
        pulse_duration=10,  # Duration of each pulse
        max_divisions=max_divisions
    )

    # Analyze results
    res_no_treat = analyze_results(t_no_treat, sol_no_treat, max_divisions)
    res_continuous = analyze_results(t_continuous, sol_continuous, max_divisions)
    res_pulsed = analyze_results(t_pulsed, sol_pulsed, max_divisions)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot healthy cells
    ax = axes[0, 0]
    ax.plot(res_no_treat['t'], res_no_treat['E_total'], 'b-', label='No Treatment')
    ax.plot(res_continuous['t'], res_continuous['E_total'], 'g-', label=f'Continuous ({opt_conc:.2f})')
    ax.plot(res_pulsed['t'], res_pulsed['E_total'], 'r-', label=f'Pulsed ({opt_conc * 2:.2f}, 10/50)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Healthy Cell Count')
    ax.set_title('Healthy Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot senescent cells
    ax = axes[0, 1]
    ax.plot(res_no_treat['t'], res_no_treat['S_total'], 'b-', label='No Treatment')
    ax.plot(res_continuous['t'], res_continuous['S_total'], 'g-', label=f'Continuous ({opt_conc:.2f})')
    ax.plot(res_pulsed['t'], res_pulsed['S_total'], 'r-', label=f'Pulsed ({opt_conc * 2:.2f}, 10/50)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Senescent Cell Count')
    ax.set_title('Senescent Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot senescent fraction
    ax = axes[1, 0]
    ax.plot(res_no_treat['t'], res_no_treat['senescent_fraction'], 'b-', label='No Treatment')
    ax.plot(res_continuous['t'], res_continuous['senescent_fraction'], 'g-', label=f'Continuous ({opt_conc:.2f})')
    ax.plot(res_pulsed['t'], res_pulsed['senescent_fraction'], 'r-', label=f'Pulsed ({opt_conc * 2:.2f}, 10/50)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')
    ax.set_title('Fraction of Senescent Cells')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # Plot telomere length
    ax = axes[1, 1]
    ax.plot(res_no_treat['t'], res_no_treat['telomere_length'], 'b-', label='No Treatment')
    ax.plot(res_continuous['t'], res_continuous['telomere_length'], 'g-', label=f'Continuous ({opt_conc:.2f})')
    ax.plot(res_pulsed['t'], res_pulsed['telomere_length'], 'r-', label=f'Pulsed ({opt_conc * 2:.2f}, 10/50)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Telomere Length')
    ax.set_title('Average Telomere Length of Healthy Cells')
    ax.legend()
    ax.grid(True)

    plt.suptitle(f'Comparison of Senolytic Treatment Strategies (τ={tau_value})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    return fig


def plot_cell_evolution_comparison(tau_value, max_divisions=15, t_span=(0, 600)):
    """
    Creates a detailed comparison of cell population evolution between continuous
    and pulsed senolytic treatments with equal total dose.

    Args:
        tau_value: Shear stress value
        max_divisions: Maximum number of cell divisions
        t_span: Time span for simulation

    Returns:
        Figure object with the detailed comparison plots
    """
    # Find optimal senolytic concentration for continuous treatment
    opt_conc, _ = optimize_senolytic_concentration(tau_value, max_divisions)

    # Define pulsed treatment parameters that will give equal total dose
    pulse_interval = 50  # Time between start of each pulse
    pulse_duration = 10  # Duration of each pulse

    # Calculate the concentration for pulsed treatment to give equal total dose
    pulsed_conc = opt_conc * pulse_interval / pulse_duration

    print(f"Continuous concentration: {opt_conc:.2f}")
    print(f"Pulsed concentration: {pulsed_conc:.2f} (equal total dose)")

    # Run simulations
    t_no_treat, sol_no_treat, _ = run_simulation(
        tau_value, senolytic_conc=0, max_divisions=max_divisions, t_span=t_span
    )

    t_continuous, sol_continuous, _ = run_simulation(
        tau_value, senolytic_conc=opt_conc, max_divisions=max_divisions, t_span=t_span
    )

    t_pulsed, sol_pulsed, _ = run_pulsed_treatment_simulation(
        tau_value,
        senolytic_conc=pulsed_conc,
        pulse_interval=pulse_interval,
        pulse_duration=pulse_duration,
        max_divisions=max_divisions,
        t_span=t_span
    )

    # Create a figure showing the cell distribution over time
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Title for the entire figure
    fig.suptitle(f'Cell Population Evolution (τ={tau_value}, Equal Total Dose)', fontsize=16)

    # Define colors for division stages
    division_colors = [plt.cm.viridis(x) for x in np.linspace(0, 0.8, max_divisions + 1)]
    senescent_colors = ['#FF5733', '#C70039']  # Colors for senescent cells

    # Time points to show cell distribution
    time_points = [int(t_span[1] * 0.2), int(t_span[1] * 0.5), int(t_span[1] * 0.8)]
    time_point_indices = [np.abs(t_continuous - tp).argmin() for tp in time_points]

    # Row labels
    treatments = ['No Treatment', 'Continuous', 'Pulsed (Equal Dose)']
    solutions = [sol_no_treat, sol_continuous, sol_pulsed]

    # Plot cell distribution at different time points
    for i, (treatment, solution) in enumerate(zip(treatments, solutions)):
        for j, time_idx in enumerate(time_point_indices):
            ax = axes[i, j]

            # Get healthy cell counts at this time point
            healthy_cells = solution.y[:max_divisions + 1, time_idx]

            # Get senescent cell counts
            senescent_tel = solution.y[max_divisions + 1, time_idx]
            senescent_stress = solution.y[max_divisions + 2, time_idx]

            # Calculate total cells for percentage
            total_cells = np.sum(healthy_cells) + senescent_tel + senescent_stress

            # Labels for healthy cells by division stage
            healthy_labels = [f'E_{k}' for k in range(max_divisions + 1)]

            # Labels for senescent cells
            senescent_labels = ['S_tel', 'S_stress']

            # Cell counts
            all_cells = np.concatenate([healthy_cells, [senescent_tel, senescent_stress]])

            # Cell labels
            all_labels = healthy_labels + senescent_labels

            # Colors
            all_colors = division_colors + senescent_colors

            # Only include non-zero populations to avoid cluttering the pie chart
            mask = all_cells > 1e-6  # Filter out very small populations

            if np.sum(mask) > 0:  # Check if there are any cells at all
                # Create pie chart
                wedges, texts, autotexts = ax.pie(
                    all_cells[mask],
                    labels=[all_labels[i] for i in range(len(all_labels)) if mask[i]],
                    autopct='%1.1f%%',
                    colors=[all_colors[i] for i in range(len(all_colors)) if mask[i]],
                    startangle=90
                )

                # Improve text appearance
                plt.setp(autotexts, size=8, weight="bold")
                plt.setp(texts, size=8)
            else:
                ax.text(0.5, 0.5, "No cells", ha='center', va='center', transform=ax.transAxes)

            # Set title
            ax.set_title(f'{treatment} at t={time_points[j]}')

    # Add an overall legend for cell types in the empty subplot
    ax = axes[0, 2]
    ax.axis('off')

    # Create legend entries
    legend_elements = []

    # Add some representative division stages
    for i in [0, 5, 10, max_divisions]:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=division_colors[i], markersize=10,
                                          label=f'E_{i}'))

    # Add senescent cells
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=senescent_colors[0], markersize=10,
                                      label='S_tel'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=senescent_colors[1], markersize=10,
                                      label='S_stress'))

    ax.legend(handles=legend_elements, loc='center', title="Cell Types")

    # Plot the time evolution of total cells and cell fractions
    ax = axes[1, 2]
    res_no_treat = analyze_results(t_no_treat, sol_no_treat, max_divisions)
    res_continuous = analyze_results(t_continuous, sol_continuous, max_divisions)
    res_pulsed = analyze_results(t_pulsed, sol_pulsed, max_divisions)

    ax.plot(res_no_treat['t'], res_no_treat['total_cells'], 'k-', label='No Treatment')
    ax.plot(res_continuous['t'], res_continuous['total_cells'], 'b-',
            label=f'Continuous ({opt_conc:.2f})')
    ax.plot(res_pulsed['t'], res_pulsed['total_cells'], 'r-',
            label=f'Pulsed ({pulsed_conc:.2f})')

    # Mark the time points
    for tp in time_points:
        ax.axvline(x=tp, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Total Cell Count')
    ax.set_title('Total Cell Evolution')
    ax.legend()
    ax.grid(True)

    # Plot the senescent cell fraction
    ax = axes[2, 2]
    ax.plot(res_no_treat['t'], res_no_treat['senescent_fraction'], 'k-', label='No Treatment')
    ax.plot(res_continuous['t'], res_continuous['senescent_fraction'], 'b-',
            label=f'Continuous ({opt_conc:.2f})')
    ax.plot(res_pulsed['t'], res_pulsed['senescent_fraction'], 'r-',
            label=f'Pulsed ({pulsed_conc:.2f})')

    # Mark the time points
    for tp in time_points:
        ax.axvline(x=tp, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Senescent Fraction')
    ax.set_title('Senescent Cell Fraction')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Create a second figure showing the distribution by division age
    fig2, axes2 = plt.subplots(3, 3, figsize=(18, 12))

    # Title for the second figure
    fig2.suptitle(f'Cell Age Distribution Evolution (τ={tau_value}, Equal Total Dose)', fontsize=16)

    # Plot distribution by division age at different time points
    for i, (treatment, solution) in enumerate(zip(treatments, solutions)):
        for j, time_idx in enumerate(time_point_indices):
            ax = axes2[i, j]

            # Get healthy cell counts at this time point
            healthy_cells = solution.y[:max_divisions + 1, time_idx]

            # Only proceed if there are healthy cells
            if np.sum(healthy_cells) > 0:
                # Calculate percentage
                percentage = 100 * healthy_cells / np.sum(healthy_cells)

                # Create bar chart
                x = np.arange(max_divisions + 1)
                ax.bar(x, percentage, color=division_colors)

                # Add labels and title
                ax.set_xlabel('Division Age')
                ax.set_ylabel('Percentage')
                ax.set_title(f'{treatment} at t={time_points[j]}')

                # Show only every 3rd tick for readability
                ax.set_xticks(x[::3])
                ax.set_xticklabels([str(i) for i in x[::3]])

                # Add grid
                ax.grid(True, axis='y')
            else:
                ax.text(0.5, 0.5, "No healthy cells", ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, fig2, res_continuous, res_pulsed



# Function to compare continuous vs. pulsed senolytic treatment with equal total dose
def compare_equal_dose_treatments(tau_value, max_divisions=15, t_span=(0, 600)):
    """
    Compares continuous senolytic treatment with pulsed treatment using equal total drug exposure.

    Args:
        tau_value: Shear stress value
        max_divisions: Maximum number of cell divisions
        t_span: Time span for simulation

    Returns:
        Figure object with the comparison plots
    """
    # Find optimal senolytic concentration for continuous treatment
    opt_conc, _ = optimize_senolytic_concentration(tau_value, max_divisions)

    # Define pulsed treatment parameters that will give equal total dose
    pulse_interval = 50  # Time between start of each pulse
    pulse_duration = 10  # Duration of each pulse

    # Calculate the concentration for pulsed treatment to give equal total dose
    # Total dose continuous = opt_conc * total_time
    # Total dose pulsed = pulsed_conc * pulse_duration * (total_time / pulse_interval)
    # For equal doses: opt_conc * total_time = pulsed_conc * pulse_duration * (total_time / pulse_interval)
    # Therefore: pulsed_conc = opt_conc * pulse_interval / pulse_duration

    pulsed_conc = opt_conc * pulse_interval / pulse_duration

    print(f"Continuous concentration: {opt_conc:.2f}")
    print(f"Pulsed concentration: {pulsed_conc:.2f} (equal total dose)")
    print(f"Pulse duration: {pulse_duration}, Pulse interval: {pulse_interval}")

    # Run simulation with no treatment
    t_no_treat, sol_no_treat, _ = run_simulation(
        tau_value,
        senolytic_conc=0,
        max_divisions=max_divisions,
        t_span=t_span
    )

    # Run simulation with continuous treatment
    t_continuous, sol_continuous, _ = run_simulation(
        tau_value,
        senolytic_conc=opt_conc,
        max_divisions=max_divisions,
        t_span=t_span
    )

    # Run simulation with pulsed treatment (equal total dose)
    t_pulsed, sol_pulsed, _ = run_pulsed_treatment_simulation(
        tau_value,
        senolytic_conc=pulsed_conc,
        pulse_interval=pulse_interval,
        pulse_duration=pulse_duration,
        max_divisions=max_divisions,
        t_span=t_span
    )

    # Analyze results
    res_no_treat = analyze_results(t_no_treat, sol_no_treat, max_divisions)
    res_continuous = analyze_results(t_continuous, sol_continuous, max_divisions)
    res_pulsed = analyze_results(t_pulsed, sol_pulsed, max_divisions)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot healthy cells
    ax = axes[0, 0]
    ax.plot(res_no_treat['t'], res_no_treat['E_total'], 'k-', label='No Treatment')
    ax.plot(res_continuous['t'], res_continuous['E_total'], 'b-', label=f'Continuous ({opt_conc:.2f})')
    ax.plot(res_pulsed['t'], res_pulsed['E_total'], 'r-',
            label=f'Pulsed ({pulsed_conc:.2f}, {pulse_duration}/{pulse_interval})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Healthy Cell Count')
    ax.set_title('Healthy Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot senescent cells
    ax = axes[0, 1]
    ax.plot(res_no_treat['t'], res_no_treat['S_total'], 'k-', label='No Treatment')
    ax.plot(res_continuous['t'], res_continuous['S_total'], 'b-', label=f'Continuous ({opt_conc:.2f})')
    ax.plot(res_pulsed['t'], res_pulsed['S_total'], 'r-',
            label=f'Pulsed ({pulsed_conc:.2f}, {pulse_duration}/{pulse_interval})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Senescent Cell Count')
    ax.set_title('Senescent Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot senescent fraction
    ax = axes[1, 0]
    ax.plot(res_no_treat['t'], res_no_treat['senescent_fraction'], 'k-', label='No Treatment')
    ax.plot(res_continuous['t'], res_continuous['senescent_fraction'], 'b-', label=f'Continuous ({opt_conc:.2f})')
    ax.plot(res_pulsed['t'], res_pulsed['senescent_fraction'], 'r-',
            label=f'Pulsed ({pulsed_conc:.2f}, {pulse_duration}/{pulse_interval})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')
    ax.set_title('Fraction of Senescent Cells')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # Plot telomere length
    ax = axes[1, 1]
    ax.plot(res_no_treat['t'], res_no_treat['telomere_length'], 'k-', label='No Treatment')
    ax.plot(res_continuous['t'], res_continuous['telomere_length'], 'b-', label=f'Continuous ({opt_conc:.2f})')
    ax.plot(res_pulsed['t'], res_pulsed['telomere_length'], 'r-',
            label=f'Pulsed ({pulsed_conc:.2f}, {pulse_duration}/{pulse_interval})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Telomere Length')
    ax.set_title('Average Telomere Length of Healthy Cells')
    ax.legend()
    ax.grid(True)

    plt.suptitle(f'Comparison of Senolytic Treatment Strategies (Equal Total Dose, τ={tau_value})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    return fig, res_continuous, res_pulsed

# Main execution
if __name__ == "__main__":
    # Set parameters
    max_divisions = 15
    tau_values = [5, 15]  # Low and high shear stress
    senolytic_concs = [0, 2, 5, 10]  # Different senolytic concentrations

    print("Running endothelial cell model with senolytics...")

    # Phase 1: Basic senolytic effect analysis
    print("Phase 1: Analyzing basic senolytic effects...")
    results, optimals = analyze_senolytic_effects(tau_values, senolytic_concs, max_divisions)

    # Phase 2: Treatment strategy comparison
    print("Phase 2: Comparing treatment strategies...")
    for tau in tau_values:
        fig = compare_treatment_strategies(tau, max_divisions)
        fig.savefig(f'treatment_comparison_tau_{tau}.png')
        plt.close(fig)

    # Phase 3: Pulsed treatment optimization
    print("Phase 3: Optimizing pulsed treatment parameters...")
    pulsed_optimals = {}
    for tau in tau_values:
        opt_conc, opt_interval, opt_duration, fig = optimize_pulsed_treatment(tau, max_divisions)
        pulsed_optimals[tau] = (opt_conc, opt_interval, opt_duration)
        fig.savefig(f'pulsed_optimization_tau_{tau}.png')
        plt.close(fig)

        print(f"Optimal pulsed treatment for τ={tau}:")
        print(f"  Concentration: {opt_conc}")
        print(f"  Interval: {opt_interval}")
        print(f"  Duration: {opt_duration}")

    # Phase 4: Final comparison of continuous vs. pulsed treatment
    print("Phase 4: Final comparison of treatment approaches...")

    # Create a figure to compare the best continuous and pulsed treatments
    plt.figure(figsize=(15, 10))

    # For each tau value
    for i, tau in enumerate(tau_values):
        # Get optimal parameters
        cont_conc = optimals[tau][0]
        pulsed_conc, pulsed_interval, pulsed_duration = pulsed_optimals[tau]

        # Run simulations
        t_no_treatment, sol_no_treatment, _ = run_simulation(tau, senolytic_conc=0, max_divisions=max_divisions)
        t_continuous, sol_continuous, _ = run_simulation(tau, senolytic_conc=cont_conc, max_divisions=max_divisions)
        t_pulsed, sol_pulsed, _ = run_pulsed_treatment_simulation(
            tau,
            senolytic_conc=pulsed_conc,
            pulse_interval=pulsed_interval,
            pulse_duration=pulsed_duration,
            max_divisions=max_divisions
        )

        # Analyze results
        res_no_treatment = analyze_results(t_no_treatment, sol_no_treatment, max_divisions)
        res_continuous = analyze_results(t_continuous, sol_continuous, max_divisions)
        res_pulsed = analyze_results(t_pulsed, sol_pulsed, max_divisions)

        # Plot healthy cells
        plt.subplot(2, 2, i + 1)
        plt.plot(res_no_treatment['t'], res_no_treatment['E_total'], 'k-',
                 label='No Treatment')
        plt.plot(res_continuous['t'], res_continuous['E_total'], 'b-',
                 label=f'Continuous ({cont_conc:.2f})')
        plt.plot(res_pulsed['t'], res_pulsed['E_total'], 'r-',
                 label=f'Pulsed ({pulsed_conc}/{pulsed_interval}/{pulsed_duration})')
        plt.xlabel('Time')
        plt.ylabel('Healthy Cell Count')
        plt.title(f'Healthy Cells (τ={tau})')
        plt.legend()
        plt.grid(True)

        # Plot senescent cells
        plt.subplot(2, 2, i + 3)
        plt.plot(res_no_treatment['t'], res_no_treatment['S_total'], 'k-',
                 label='No Treatment')
        plt.plot(res_continuous['t'], res_continuous['S_total'], 'b-',
                 label=f'Continuous ({cont_conc:.2f})')
        plt.plot(res_pulsed['t'], res_pulsed['S_total'], 'r-',
                 label=f'Pulsed ({pulsed_conc}/{pulsed_interval}/{pulsed_duration})')
        plt.xlabel('Time')
        plt.ylabel('Senescent Cell Count')
        plt.title(f'Senescent Cells (τ={tau})')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('final_treatment_comparison.png')

    # Create a summary table of all results
    plt.figure(figsize=(12, 6))
    plt.axis('off')

    table_data = []
    table_columns = ['Shear Stress (τ)', 'Approach', 'Final Healthy Cells',
                     'Final Senescent Cells', 'Senescent Fraction', 'Improvement (%)']

    for tau in tau_values:
        # Get parameters
        cont_conc = optimals[tau][0]
        pulsed_conc, pulsed_interval, pulsed_duration = pulsed_optimals[tau]

        # Run simulations
        t_no_treatment, sol_no_treatment, _ = run_simulation(tau, senolytic_conc=0, max_divisions=max_divisions)
        t_continuous, sol_continuous, _ = run_simulation(tau, senolytic_conc=cont_conc, max_divisions=max_divisions)
        t_pulsed, sol_pulsed, _ = run_pulsed_treatment_simulation(
            tau, senolytic_conc=pulsed_conc, pulse_interval=pulsed_interval,
            pulse_duration=pulsed_duration, max_divisions=max_divisions
        )

        # Analyze results
        res_no_treatment = analyze_results(t_no_treatment, sol_no_treatment, max_divisions)
        res_continuous = analyze_results(t_continuous, sol_continuous, max_divisions)
        res_pulsed = analyze_results(t_pulsed, sol_pulsed, max_divisions)

        # Calculate metrics
        no_treat_healthy = res_no_treatment['E_total'][-1]
        cont_healthy = res_continuous['E_total'][-1]
        pulsed_healthy = res_pulsed['E_total'][-1]

        no_treat_senescent = res_no_treatment['S_total'][-1]
        cont_senescent = res_continuous['S_total'][-1]
        pulsed_senescent = res_pulsed['S_total'][-1]

        no_treat_fraction = res_no_treatment['senescent_fraction'][-1]
        cont_fraction = res_continuous['senescent_fraction'][-1]
        pulsed_fraction = res_pulsed['senescent_fraction'][-1]

        cont_improvement = 100 * (cont_healthy - no_treat_healthy) / max(no_treat_healthy, 1)
        pulsed_improvement = 100 * (pulsed_healthy - no_treat_healthy) / max(no_treat_healthy, 1)

        # Add to table data
        table_data.append([tau, 'No Treatment',
                           f"{no_treat_healthy:.1f}",
                           f"{no_treat_senescent:.1f}",
                           f"{no_treat_fraction:.3f}",
                           "-"])

        table_data.append([tau, f'Continuous ({cont_conc:.2f})',
                           f"{cont_healthy:.1f}",
                           f"{cont_senescent:.1f}",
                           f"{cont_fraction:.3f}",
                           f"+{cont_improvement:.1f}%"])

        table_data.append([tau, f'Pulsed ({pulsed_conc}/{pulsed_interval}/{pulsed_duration})',
                           f"{pulsed_healthy:.1f}",
                           f"{pulsed_senescent:.1f}",
                           f"{pulsed_fraction:.3f}",
                           f"+{pulsed_improvement:.1f}%"])

    # Create the table
    table = plt.table(
        cellText=table_data,
        colLabels=table_columns,
        loc='center',
        cellLoc='center'
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title('Summary of Senolytic Treatment Effects', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('treatment_summary_table.png', bbox_inches='tight')

    print("Analysis complete. All results saved as PNG files.")

    # Show all plots
    plt.show()

    # Phase 4.5: Compare treatments with equal total dose
    print("Phase 4.5: Comparing treatments with equal total dose...")
    for tau in tau_values:
        fig, res_cont, res_pulsed = compare_equal_dose_treatments(tau, max_divisions)
        fig.savefig(f'equal_dose_comparison_tau_{tau}.png')
        plt.close(fig)

        # Calculate final metrics for comparison
        cont_healthy = res_cont['E_total'][-1]
        pulsed_healthy = res_pulsed['E_total'][-1]

        print(f"For τ={tau}:")
        print(f"  Continuous treatment final healthy cells: {cont_healthy:.1f}")
        print(f"  Pulsed treatment final healthy cells: {pulsed_healthy:.1f}")
        print(f"  Difference: {((pulsed_healthy - cont_healthy) / cont_healthy * 100):.1f}%")

        # Phase 4.6: Plot detailed cell population evolution
        print("Phase 4.6: Plotting detailed cell population evolution...")
        for tau in tau_values:
            fig1, fig2, res_cont, res_pulsed = plot_cell_evolution_comparison(tau, max_divisions)
            fig1.savefig(f'cell_distribution_comparison_tau_{tau}.png')
            fig2.savefig(f'cell_age_comparison_tau_{tau}.png')
            plt.close(fig1)
            plt.close(fig2)

            # Print key metrics
            print(f"For τ={tau}:")
            print(f"  Final healthy/senescent ratio:")
            print(f"    Continuous: {res_cont['E_total'][-1]:.1f}/{res_cont['S_total'][-1]:.1f}")
            print(f"    Pulsed: {res_pulsed['E_total'][-1]:.1f}/{res_pulsed['S_total'][-1]:.1f}")
# Plot optimization curve
