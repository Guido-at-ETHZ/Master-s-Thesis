import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time


# Define a more realistic shear stress effect function (unchanged from original)
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


# Stochastic version of the endothelial cell dynamics with agent-based division
def stochastic_endothelial_cell_dynamics(t, y, params, dt):
    """
    Hybrid deterministic-stochastic system for endothelial cell dynamics.
    - Cell division is modeled stochastically on a cell-by-cell basis
    - All other processes (death, senescence) remain deterministic

    This produces a single step in the simulation with time step dt.
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

    # Extract state variables and convert to integers for stochastic processes
    # Healthy cells at different division stages
    E = np.array([int(max(0, cell_count)) for cell_count in y[:max_divisions + 1]])
    # Senescent cells by cause (these remain as floats for deterministic processes)
    S_tel = y[max_divisions + 1]  # Telomere-induced senescent cells
    S_stress = y[max_divisions + 2]  # Stress-induced senescent cells

    # Total number of cells (for carrying capacity calculation)
    total_cells = np.sum(E) + S_tel + S_stress

    # Total number of senescent cells (for SASP effects)
    total_senescent = S_tel + S_stress

    # Initialize changes array (will be used instead of derivatives)
    dy = np.zeros_like(y)

    # Modified density-dependent inhibition that allows initial growth
    # Use a sigmoid function that allows growth until approaching capacity
    density_factor = 1 / (1 + np.exp(10 * (total_cells / K - 0.7)))

    # Process each division stage
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

        # Number of cells in this compartment
        n_cells = E[i]

        if n_cells > 0:
            # STOCHASTIC PROCESSES:

            # For truly cell-by-cell behavior, we need to simulate individual cell fates
            # For each cell, determine its fate (division, death, senescence) probabilistically

            # Individual cell probabilities per timestep
            p_division = r * density_factor * division_capacity * dt
            p_death = death_rate * dt
            p_stress_senescence = stress_senescence_rate * dt
            p_sasp_senescence = sasp_senescence_rate * dt

            # Ensure probabilities are valid
            p_division = max(0, min(1, p_division))
            p_death = max(0, min(1, p_death))
            p_stress_senescence = max(0, min(1, p_stress_senescence))
            p_sasp_senescence = max(0, min(1, p_sasp_senescence))

            # Renormalize to ensure the sum doesn't exceed 1
            # (A cell can only undergo one fate in this time step)
            total_p = p_division + p_death + p_stress_senescence + p_sasp_senescence
            if total_p > 1:
                scaling_factor = 1.0 / total_p
                p_division *= scaling_factor
                p_death *= scaling_factor
                p_stress_senescence *= scaling_factor
                p_sasp_senescence *= scaling_factor

            # For large cell numbers, using multinomial distribution is more efficient
            # than simulating each cell individually
            p_no_event = 1.0 - (p_division + p_death + p_stress_senescence + p_sasp_senescence)
            probabilities = [p_no_event, p_division, p_death, p_stress_senescence, p_sasp_senescence]

            # Multinomial returns counts for each event type
            event_counts = np.random.multinomial(n_cells, probabilities)

            # Extract counts for each event
            n_no_event = event_counts[0]
            n_dividing = event_counts[1]
            n_dying = event_counts[2]
            n_stress_senescence = event_counts[3]
            n_sasp_senescence = event_counts[4]

            # Apply the effects of each event

            # Division events
            if i < max_divisions:
                # Cells at earlier divisions produce two cells in next division stage
                dy[i] -= n_dividing  # Remove dividing cells
                dy[i + 1] += 2 * n_dividing  # Add two daughter cells to next stage
            else:
                # Cells at max division move to senescent compartment
                dy[i] -= n_dividing
                dy[max_divisions + 1] += n_dividing  # Add to telomere-induced senescent

            # Death events
            dy[i] -= n_dying

            # Senescence events
            dy[i] -= (n_stress_senescence + n_sasp_senescence)
            dy[max_divisions + 2] += (n_stress_senescence + n_sasp_senescence)

    # Stochastic death for senescent cells
    # Use Poisson approximation for death events of senescent cells

    # Telomere-induced senescent cells death
    p_s_tel_death = d_S_tel * dt
    if p_s_tel_death > 0 and S_tel > 0:
        # For large populations, binomial is more accurate than Poisson
        if S_tel > 100:
            n_s_tel_dying = np.random.binomial(int(S_tel), min(1.0, p_s_tel_death))
        else:
            # For small populations, use Poisson approximation
            n_s_tel_dying = np.random.poisson(S_tel * p_s_tel_death)
            n_s_tel_dying = min(n_s_tel_dying, int(S_tel))  # Can't lose more cells than we have

        dy[max_divisions + 1] -= n_s_tel_dying

    # Stress-induced senescent cells death
    p_s_stress_death = d_S_stress * dt
    if p_s_stress_death > 0 and S_stress > 0:
        if S_stress > 100:
            n_s_stress_dying = np.random.binomial(int(S_stress), min(1.0, p_s_stress_death))
        else:
            n_s_stress_dying = np.random.poisson(S_stress * p_s_stress_death)
            n_s_stress_dying = min(n_s_stress_dying, int(S_stress))

        dy[max_divisions + 2] -= n_s_stress_dying

    return dy


# Run stochastic simulation
def run_stochastic_simulation(tau_value, max_divisions=15, initial_conditions=None,
                              t_span=(0, 600), dt=0.1, n_runs=20):
    """
    Runs multiple stochastic simulations and returns average results with confidence intervals.

    Parameters:
    - tau_value: Shear stress value
    - max_divisions: Maximum number of divisions before mandatory senescence
    - initial_conditions: Initial cell counts
    - t_span: Time span for simulation
    - dt: Time step size
    - n_runs: Number of simulation runs

    Returns:
    - t_points: Time points
    - mean_results: Mean values of all runs
    - ci_lower: Lower confidence interval (10th percentile)
    - ci_upper: Upper confidence interval (90th percentile)
    - all_results: All individual simulation results
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
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t_points = np.linspace(t_start, t_end, n_steps)

    # Storage for multiple runs
    all_results = []
    all_raw_states = []

    # Run multiple simulations
    for run in range(n_runs):
        # Initialize state
        y = initial_conditions.copy()

        # Storage for this run
        states = np.zeros((len(y), len(t_points)))
        states[:, 0] = y

        # Run simulation with fixed time steps
        for step in range(1, len(t_points)):
            # Calculate changes for this step
            dy = stochastic_endothelial_cell_dynamics(t_points[step], y, params, dt)

            # Update state
            y = y + dy

            # Ensure no negative values
            y = np.maximum(y, 0)

            # Store state
            states[:, step] = y

        # Create solution-like object for compatibility with analyze_results
        class StochasticSolution:
            def __init__(self, t, y):
                self.t = t
                self.y = y

        solution = StochasticSolution(t_points, states)
        all_raw_states.append(states)

        # Analyze and store results
        result = analyze_results(t_points, solution, max_divisions)
        result['run'] = run
        all_results.append(result)

    # Calculate statistics across runs
    keys_to_average = ['E_total', 'S_tel', 'S_stress', 'S_total', 'total_cells',
                       'senescent_fraction', 'tel_fraction_of_senescent',
                       'avg_division_age', 'telomere_length']

    mean_results = {'t': t_points, 'tau': tau_value}
    ci_lower = {'t': t_points, 'tau': tau_value}
    ci_upper = {'t': t_points, 'tau': tau_value}
    std_results = {'t': t_points, 'tau': tau_value}

    for key in keys_to_average:
        all_values = np.array([result[key] for result in all_results])
        mean_results[key] = np.mean(all_values, axis=0)
        std_results[key] = np.std(all_values, axis=0)
        # Use wider confidence intervals for more visible stochasticity
        ci_lower[key] = np.percentile(all_values, 5, axis=0)  # 5th percentile
        ci_upper[key] = np.percentile(all_values, 95, axis=0)  # 95th percentile

    # Store all raw states for potential detailed analysis
    mean_results['all_raw_states'] = all_raw_states

    return t_points, mean_results, ci_lower, ci_upper, all_results


# Function to analyze results (unchanged from original)
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


# Run deterministic simulation (function from original model)
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

    # Original deterministic endothelial_cell_dynamics function from the given code
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

    # Solve the system of differential equations
    solution = solve_ivp(
        lambda t, y: endothelial_cell_dynamics(t, y, params),
        t_span,
        initial_conditions,
        t_eval=t_eval,
        method='RK45'
    )

    return t_eval, solution


# Function to compare deterministic and stochastic models
def compare_models(tau_values, max_divisions=15):
    """
    Run both deterministic and stochastic simulations for multiple tau values
    and plot comparisons.
    """
    det_results = []
    stoch_mean_results = []
    stoch_ci_lower = []
    stoch_ci_upper = []
    all_stoch_runs = []

    for tau in tau_values:
        print(f"Running simulations for tau = {tau}...")

        # Run deterministic simulation
        t_eval, det_solution = run_simulation(tau, max_divisions=max_divisions)
        det_result = analyze_results(t_eval, det_solution, max_divisions)
        det_result['tau'] = tau
        det_results.append(det_result)

        # Run stochastic simulations
        t_points, mean_result, ci_lower, ci_upper, all_runs = run_stochastic_simulation(
            tau, max_divisions=max_divisions, n_runs=10, dt=0.2
        )

        # Add tau to each individual run result
        for run in all_runs:
            run['tau'] = tau
            all_stoch_runs.append(run)

        stoch_mean_results.append(mean_result)
        stoch_ci_lower.append(ci_lower)
        stoch_ci_upper.append(ci_upper)

    return det_results, stoch_mean_results, stoch_ci_lower, stoch_ci_upper, all_stoch_runs


# Plot stochastic model results
def plot_stochastic_results(results, ci_lower, ci_upper, all_runs=None, max_divisions=15):
    """
    Plot comprehensive results from stochastic simulations, matching the format
    of the original deterministic model plots.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Colors for different tau values
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    # Plot total cells
    ax = axes[0, 0]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['total_cells'], color=colors[i],
                label=f'τ={result["tau"]}')
        ax.fill_between(result['t'],
                        ci_lower[i]['total_cells'],
                        ci_upper[i]['total_cells'],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Total Cell Population (Stochastic)')
    ax.legend()
    ax.grid(True)

    # Plot healthy cells
    ax = axes[0, 1]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['E_total'], color=colors[i],
                label=f'τ={result["tau"]}')
        ax.fill_between(result['t'],
                        ci_lower[i]['E_total'],
                        ci_upper[i]['E_total'],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Healthy Cell Population (Stochastic)')
    ax.legend()
    ax.grid(True)

    # Plot senescent cells
    ax = axes[0, 2]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['S_total'], color=colors[i],
                label=f'τ={result["tau"]}')
        ax.fill_between(result['t'],
                        ci_lower[i]['S_total'],
                        ci_upper[i]['S_total'],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Senescent Cell Population (Stochastic)')
    ax.legend()
    ax.grid(True)

    # Plot senescent fraction
    ax = axes[1, 0]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['senescent_fraction'], color=colors[i],
                label=f'τ={result["tau"]}')
        ax.fill_between(result['t'],
                        ci_lower[i]['senescent_fraction'],
                        ci_upper[i]['senescent_fraction'],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')
    ax.set_title('Fraction of Senescent Cells (Stochastic)')
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
        ax.fill_between(result['t'],
                        ci_lower[i]['S_tel'],
                        ci_upper[i]['S_tel'],
                        color=colors[i], alpha=0.1)
        ax.fill_between(result['t'],
                        ci_lower[i]['S_stress'],
                        ci_upper[i]['S_stress'],
                        color=colors[i], alpha=0.1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Senescent Cells by Cause (Stochastic)')
    ax.legend()
    ax.grid(True)

    # Plot fraction of senescent cells that are telomere-induced
    ax = axes[1, 2]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['tel_fraction_of_senescent'], color=colors[i],
                label=f'τ={result["tau"]}')
        ax.fill_between(result['t'],
                        ci_lower[i]['tel_fraction_of_senescent'],
                        ci_upper[i]['tel_fraction_of_senescent'],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')
    ax.set_title('Fraction of Senescent Cells from Telomere Shortening (Stochastic)')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # Plot telomere length
    ax = axes[2, 0]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['telomere_length'], color=colors[i],
                label=f'τ={result["tau"]}')
        ax.fill_between(result['t'],
                        ci_lower[i]['telomere_length'],
                        ci_upper[i]['telomere_length'],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Telomere Length')
    ax.set_title('Average Telomere Length of Healthy Cells (Stochastic)')
    ax.axhline(y=20, color='r', linestyle='--', label='Critical Length')
    ax.legend()
    ax.grid(True)

    # Plot average division age
    ax = axes[2, 1]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['avg_division_age'], color=colors[i],
                label=f'τ={result["tau"]}')
        ax.fill_between(result['t'],
                        ci_lower[i]['avg_division_age'],
                        ci_upper[i]['avg_division_age'],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Divisions')
    ax.set_title('Average Division Age of Healthy Cells (Stochastic)')
    ax.legend()
    ax.grid(True)

    # Plot individual runs for one specific tau value to show variability
    ax = axes[2, 2]
    if all_runs is not None:
        # Choose a middle tau value
        tau_idx = len(results) // 2
        tau_value = results[tau_idx]['tau']

        # Find all runs for this tau value
        tau_runs = [run for run in all_runs if run['tau'] == tau_value]

        # Plot individual runs - total cells
        for i, run in enumerate(tau_runs):
            if i < 10:  # Limit to 10 runs for clarity
                ax.plot(run['t'], run['total_cells'], alpha=0.4, linewidth=0.8)

        # Plot mean with thicker line
        ax.plot(results[tau_idx]['t'], results[tau_idx]['total_cells'],
                color='black', linewidth=2, label='Mean')

        ax.set_xlabel('Time')
        ax.set_ylabel('Cell Count')
        ax.set_title(f'Individual Simulation Runs (τ={tau_value})')
        ax.legend()
        ax.grid(True)

    plt.suptitle('Stochastic Endothelial Cell Dynamics with Different Shear Stress Levels', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    return fig


# Plot comparison of deterministic and stochastic models
def plot_comparison(det_results, stoch_means, stoch_ci_lower, stoch_ci_upper):
    """
    Create plots comparing deterministic and stochastic model results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Colors for different tau values
    n_tau = len(det_results)
    colors = plt.cm.viridis(np.linspace(0, 1, n_tau))

    # Plot total cell population
    ax = axes[0, 0]
    for i in range(n_tau):
        tau = det_results[i]['tau']
        # Deterministic result
        ax.plot(det_results[i]['t'], det_results[i]['total_cells'],
                color=colors[i], linestyle='-', label=f'Det τ={tau}')
        # Stochastic result with confidence interval
        ax.plot(stoch_means[i]['t'], stoch_means[i]['total_cells'],
                color=colors[i], linestyle='--', label=f'Stoch τ={tau}')
        ax.fill_between(stoch_means[i]['t'],
                        stoch_ci_lower[i]['total_cells'],
                        stoch_ci_upper[i]['total_cells'],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Total Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot healthy cells
    ax = axes[0, 1]
    for i in range(n_tau):
        tau = det_results[i]['tau']
        # Deterministic result
        ax.plot(det_results[i]['t'], det_results[i]['E_total'],
                color=colors[i], linestyle='-', label=f'Det τ={tau}')
        # Stochastic result with confidence interval
        ax.plot(stoch_means[i]['t'], stoch_means[i]['E_total'],
                color=colors[i], linestyle='--', label=f'Stoch τ={tau}')
        ax.fill_between(stoch_means[i]['t'],
                        stoch_ci_lower[i]['E_total'],
                        stoch_ci_upper[i]['E_total'],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Healthy Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot senescent cells
    ax = axes[0, 2]
    for i in range(n_tau):
        tau = det_results[i]['tau']
        # Deterministic result
        ax.plot(det_results[i]['t'], det_results[i]['S_total'],
                color=colors[i], linestyle='-', label=f'Det τ={tau}')
        # Stochastic result with confidence interval
        ax.plot(stoch_means[i]['t'], stoch_means[i]['S_total'],
                color=colors[i], linestyle='--', label=f'Stoch τ={tau}')
        ax.fill_between(stoch_means[i]['t'],
                        stoch_ci_lower[i]['S_total'],
                        stoch_ci_upper[i]['S_total'],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Senescent Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot senescent fraction
    ax = axes[1, 0]
    for i in range(n_tau):
        tau = det_results[i]['tau']
        # Deterministic result
        ax.plot(det_results[i]['t'], det_results[i]['senescent_fraction'],
                color=colors[i], linestyle='-', label=f'Det τ={tau}')
        # Stochastic result with confidence interval
        ax.plot(stoch_means[i]['t'], stoch_means[i]['senescent_fraction'],
                color=colors[i], linestyle='--', label=f'Stoch τ={tau}')
        ax.fill_between(stoch_means[i]['t'],
                        stoch_ci_lower[i]['senescent_fraction'],
                        stoch_ci_upper[i]['senescent_fraction'],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')
    ax.set_title('Fraction of Senescent Cells')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # Plot telomere length
    ax = axes[1, 1]
    for i in range(n_tau):
        tau = det_results[i]['tau']
        # Deterministic result
        ax.plot(det_results[i]['t'], det_results[i]['telomere_length'],
                color=colors[i], linestyle='-', label=f'Det τ={tau}')
        # Stochastic result with confidence interval
        ax.plot(stoch_means[i]['t'], stoch_means[i]['telomere_length'],
                color=colors[i], linestyle='--', label=f'Stoch τ={tau}')
        ax.fill_between(stoch_means[i]['t'],
                        stoch_ci_lower[i]['telomere_length'],
                        stoch_ci_upper[i]['telomere_length'],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Telomere Length')
    ax.set_title('Average Telomere Length of Healthy Cells')
    ax.legend()
    ax.grid(True)

    # Plot division age
    ax = axes[1, 2]
    for i in range(n_tau):
        tau = det_results[i]['tau']
        # Deterministic result
        ax.plot(det_results[i]['t'], det_results[i]['avg_division_age'],
                color=colors[i], linestyle='-', label=f'Det τ={tau}')
        # Stochastic result with confidence interval
        ax.plot(stoch_means[i]['t'], stoch_means[i]['avg_division_age'],
                color=colors[i], linestyle='--', label=f'Stoch τ={tau}')
        ax.fill_between(stoch_means[i]['t'],
                        stoch_ci_lower[i]['avg_division_age'],
                        stoch_ci_upper[i]['avg_division_age'],
                        color=colors[i], alpha=0.2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Divisions')
    ax.set_title('Average Division Age of Healthy Cells')
    ax.legend()
    ax.grid(True)

    plt.suptitle('Comparison of Deterministic vs. Stochastic Cell Division Models', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    return fig


# Main function to run a complete stochastic model
def main():
    """
    Main function to run the stochastic endothelial cell dynamics model with different shear stress values.
    """
    # Configuration
    max_divisions = 15
    tau_values = [0, 5, 10, 15, 20]

    # Run stochastic simulations
    print("Running stochastic simulations...")
    stoch_means = []
    stoch_lower = []
    stoch_upper = []
    all_stoch_runs = []

    for tau in tau_values:
        print(f"  Processing tau = {tau}...")
        t_points, mean_result, ci_lower, ci_upper, all_runs = run_stochastic_simulation(
            tau, max_divisions=max_divisions, n_runs=20, dt=0.2
        )

        # Add tau to each individual run result
        for run in all_runs:
            run['tau'] = tau
            all_stoch_runs.append(run)

        stoch_means.append(mean_result)
        stoch_lower.append(ci_lower)
        stoch_upper.append(ci_upper)

    # Create stochastic plots (same format as original deterministic plots)
    print("Creating stochastic plots...")
    stoch_fig = plot_stochastic_results(stoch_means, stoch_lower, stoch_upper, all_stoch_runs, max_divisions)
    stoch_fig.savefig('stochastic_model_results.png', dpi=300, bbox_inches='tight')

    # Create a stochastic version of the cell population plot
    stoch_pop_fig = plot_stochastic_cell_populations(stoch_means, stoch_lower, stoch_upper)
    stoch_pop_fig.savefig('stochastic_cell_populations.png', dpi=300, bbox_inches='tight')

    print("Stochastic plots created and saved.")
    plt.show()

    return stoch_means, stoch_lower, stoch_upper, all_stoch_runs


# Plot just cell populations for stochastic model
def plot_stochastic_cell_populations(results, ci_lower, ci_upper):
    """
    Focus on plotting stochastic cell population dynamics.
    """
    plt.figure(figsize=(12, 8))

    # Colors for different tau values
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    # Plot healthy, senescent, and total cells for each tau value
    for i, result in enumerate(results):
        tau = result['tau']
        plt.plot(result['t'], result['E_total'], color=colors[i], linestyle='-',
                 label=f'Healthy τ={tau}')
        plt.fill_between(result['t'], ci_lower[i]['E_total'], ci_upper[i]['E_total'],
                         color=colors[i], alpha=0.1)

        plt.plot(result['t'], result['S_total'], color=colors[i], linestyle='--',
                 label=f'Senescent τ={tau}')
        plt.fill_between(result['t'], ci_lower[i]['S_total'], ci_upper[i]['S_total'],
                         color=colors[i], alpha=0.1)

        plt.plot(result['t'], result['total_cells'], color=colors[i], linestyle=':',
                 label=f'Total τ={tau}')
        plt.fill_between(result['t'], ci_lower[i]['total_cells'], ci_upper[i]['total_cells'],
                         color=colors[i], alpha=0.1)

    plt.xlabel('Time')
    plt.ylabel('Cell Count')
    plt.title('Stochastic Endothelial Cell Population Dynamics')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    # Run the full stochastic model
    main()