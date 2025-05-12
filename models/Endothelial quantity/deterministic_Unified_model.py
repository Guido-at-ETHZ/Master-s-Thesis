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
    # Max effect approaches 0.15 at high concentrations
    max_effect = 0.15 * efficacy_factor

    # EC50 = 5 (concentration at which half the maximum effect is achieved)
    ec50 = 5

    # Hill coefficient = 3 (for non-linearity)
    hill = 3

    # Calculate effect using Hill equation
    effect = max_effect * (concentration ** hill) / (ec50 ** hill + concentration ** hill)

    return effect


# Function to create default stem cell distribution
def create_stem_cell_distribution(max_divisions, distribution_type='exponential', peak_stage=0, width=2):
    """
    Creates a distribution for how stem cells differentiate into different division stages.

    Parameters:
    -----------
    max_divisions : int
        Maximum number of divisions that cells can undergo.
    distribution_type : str
        Type of distribution. Options: 'exponential', 'gaussian', 'single_stage', 'uniform'
    peak_stage : int
        For gaussian distribution, the division stage where most cells appear.
    width : float
        For gaussian distribution, the width of the peak.

    Returns:
    --------
    distribution : ndarray
        Normalized distribution of stem cell differentiation into division stages.
    """
    distribution = np.zeros(max_divisions + 1)

    if distribution_type == 'exponential':
        # Decreasing exponential - most cells go to E_0, fewer to higher stages
        for i in range(max_divisions + 1):
            distribution[i] = np.exp(-0.7 * i)

    elif distribution_type == 'gaussian':
        # Gaussian distribution centered on a specific division stage
        for i in range(max_divisions + 1):
            distribution[i] = np.exp(-(i - peak_stage) ** 2 / (2 * width ** 2))

    elif distribution_type == 'single_stage':
        # All cells go to a single specified stage
        distribution[peak_stage] = 1.0

    elif distribution_type == 'uniform':
        # Equal distribution to all stages
        distribution[:] = 1.0

    # Normalize to sum to 1
    distribution = distribution / np.sum(distribution)

    return distribution


# Define the combined cell dynamics model with stem cell input and senolytics
def endothelial_cell_dynamics(t, y, params):
    """
    System of differential equations for endothelial cell dynamics with
    division-based population structure, stem cell input, and senolytic effects.

    This model tracks:
    - Cell populations at different division stages (E_0, E_1, ..., E_N)
    - Senescent cells from telomere shortening (S_tel)
    - Senescent cells from stress-induced senescence (S_stress)
    - Includes stem cell differentiation as a source of new endothelial cells
    - Includes senolytic effects on senescent cell death rates
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

    # Stem cell parameters
    stem_cell_rate = params['stem_cell_rate']  # Rate of stem cell differentiation
    stem_cell_distribution = params['stem_cell_distribution']  # Distribution across division stages

    # Senolytic parameters
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

        # Add stem cell input according to specified distribution
        stem_cell_input = stem_cell_rate * stem_cell_distribution[i]

        # Combine all terms for this division stage
        if i == max_divisions:
            # Last division stage - all divisions lead to senescence
            dy[
                i] = division_in - death_term - stress_senescence_term - sasp_senescence_term - division_out + stem_cell_input
        else:
            dy[
                i] = division_in - division_out - death_term - stress_senescence_term - sasp_senescence_term + stem_cell_input

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
def run_simulation(tau_value, senolytic_conc=5, stem_cell_rate=10, max_divisions=15,
                   stem_distribution_type='exponential', peak_stage=0, width=2,
                   initial_conditions=None, t_span=(0, 600), t_points=1000):
    """
    Runs a simulation of the endothelial cell dynamics model with stem cell input and senolytic effects.
    """
    # Default initial conditions if none provided
    if initial_conditions is None:
        # Start with all cells in the undivided state (E_0)
        initial_conditions = np.zeros(max_divisions + 3)
        initial_conditions[0] = 200  # Initial population in E_0

    # Create stem cell distribution
    stem_cell_distribution = create_stem_cell_distribution(
        max_divisions, stem_distribution_type, peak_stage, width
    )

    params = {
        'r': 0.06,  # Proliferation rate
        'K': 3000,  # Carrying capacity
        'd_E': 0.008,  # Death rate of healthy cells
        'd_S_tel': 0.02,  # Death rate of telomere-induced senescent cells
        'd_S_stress': 0.025,  # Death rate of stress-induced senescent cells
        'gamma_S': 0.00005,  # Senescence induction by senescent cells
        'tau': tau_value,  # Shear stress
        'max_divisions': max_divisions,  # Max divisions before mandatory senescence
        'stem_cell_rate': stem_cell_rate,  # Rate of stem cell differentiation
        'stem_cell_distribution': stem_cell_distribution,  # Distribution of stem cell input
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


# Function to analyze results
def analyze_results(t_eval, solution, max_divisions):
    """
    Analyze the results of the simulation with FIXED calculations for telomere length
    and average division age.
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

    # Calculate average division age of healthy cells - IMPROVED with better handling of edge cases
    avg_division_age = np.zeros_like(E_total)

    for t in range(len(t_eval)):
        if E_total[t] > 1e-6:  # Only calculate when there are meaningful numbers of cells
            weighted_sum = 0
            for i in range(max_divisions + 1):
                weighted_sum += i * solution.y[i, t]
            avg_division_age[t] = weighted_sum / E_total[t]
        else:
            # When E_total is effectively zero, set to NaN to avoid misleading values
            avg_division_age[t] = np.nan

    # Calculate "telomere length" based on average division age - IMPROVED
    # Start with maximum length and decrease based on average divisions
    max_telomere = 100
    min_telomere = 20

    # Handle potential NaN values in avg_division_age
    telomere_length = np.zeros_like(avg_division_age)
    for i in range(len(avg_division_age)):
        if np.isnan(avg_division_age[i]):
            telomere_length[i] = np.nan
        else:
            # Ensure we don't exceed bounds
            normalized_age = min(max(0, avg_division_age[i] / max_divisions), 1)
            telomere_length[i] = max_telomere - (max_telomere - min_telomere) * normalized_age

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
def run_multiple_shear_stress_simulations(tau_values, senolytic_conc=5, stem_cell_rate=10,
                                          max_divisions=15, stem_distribution_type='exponential'):
    """
    Run simulations for different shear stress values and analyze results.
    """
    results = []

    for tau in tau_values:
        t_eval, solution, params = run_simulation(
            tau,
            senolytic_conc=senolytic_conc,
            stem_cell_rate=stem_cell_rate,
            max_divisions=max_divisions,
            stem_distribution_type=stem_distribution_type
        )
        result = analyze_results(t_eval, solution, max_divisions)
        result['tau'] = tau
        result['senolytic_conc'] = senolytic_conc
        result['stem_cell_rate'] = stem_cell_rate
        results.append(result)

    return results


# Plot the simulation results
def plot_results(results, max_divisions=15):
    """
    Plot comprehensive results from multiple simulations with different shear stress values,
    fixed senolytic concentration, and fixed stem cell input rate.
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

    # Plot telomere length - IMPROVED handling of NaN values
    ax = axes[2, 0]
    for i, result in enumerate(results):
        # Remove NaN values for plotting
        mask = ~np.isnan(result['telomere_length'])
        if np.any(mask):  # Only plot if there are valid values
            ax.plot(result['t'][mask], result['telomere_length'][mask], color=colors[i],
                    label=f'τ={result["tau"]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Telomere Length')
    ax.set_title('Average Telomere Length of Healthy Cells')
    ax.axhline(y=20, color='r', linestyle='--', label='Critical Length')
    ax.set_ylim(0, 110)  # Sensible y-axis limits
    ax.legend()
    ax.grid(True)

    # Plot average division age - IMPROVED handling of NaN values
    ax = axes[2, 1]
    for i, result in enumerate(results):
        # Remove NaN values for plotting
        mask = ~np.isnan(result['avg_division_age'])
        if np.any(mask):  # Only plot if there are valid values
            ax.plot(result['t'][mask], result['avg_division_age'][mask], color=colors[i],
                    label=f'τ={result["tau"]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Divisions')
    ax.set_title('Average Division Age of Healthy Cells')
    ax.set_ylim(0, max_divisions)  # Sensible y-axis limits
    ax.legend()
    ax.grid(True)

    # Plot distribution of cells by division age at final time point for one tau value
    ax = axes[2, 2]
    tau_to_show = results[len(results) // 2]['tau']  # Middle tau value
    senolytic_conc = results[0]['senolytic_conc']
    stem_cell_rate = results[0]['stem_cell_rate']

    t_eval, solution, _ = run_simulation(
        tau_to_show,
        senolytic_conc=senolytic_conc,
        stem_cell_rate=stem_cell_rate,
        max_divisions=max_divisions
    )

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

    senolytic_str = f"Senolytic Concentration: {senolytic_conc}"
    stem_cell_str = f"Stem Cell Rate: {stem_cell_rate} (Exponential Distribution)"
    plt.suptitle(f'Endothelial Cell Dynamics with Different Shear Stress Levels\n{senolytic_str}, {stem_cell_str}',
                 fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    return fig


# Function to compare the effect of adding senolytics and stem cells to the vanilla model
def compare_models(tau_value, senolytic_conc=5, stem_cell_rate=10, max_divisions=15):
    """
    Compare four models:
    1. Vanilla model (no senolytics, no stem cells)
    2. Model with senolytics only
    3. Model with stem cells only
    4. Combined model (senolytics and stem cells)
    """
    # Create stem cell distribution
    stem_cell_distribution = create_stem_cell_distribution(max_divisions, 'exponential')

    # Run vanilla model (no senolytics, no stem cells)
    params_vanilla = {
        'r': 0.06,
        'K': 3000,
        'd_E': 0.008,
        'd_S_tel': 0.02,
        'd_S_stress': 0.025,
        'gamma_S': 0.00005,
        'tau': tau_value,
        'max_divisions': max_divisions,
        'stem_cell_rate': 0,
        'stem_cell_distribution': stem_cell_distribution,
        'senolytic_conc': 0,
        'sen_efficacy_tel': 1.0,
        'sen_efficacy_stress': 1.2,
    }

    # Run model with senolytics only
    params_senolytics = params_vanilla.copy()
    params_senolytics['senolytic_conc'] = senolytic_conc

    # Run model with stem cells only
    params_stem = params_vanilla.copy()
    params_stem['stem_cell_rate'] = stem_cell_rate

    # Run combined model (senolytics and stem cells)
    params_combined = params_vanilla.copy()
    params_combined['senolytic_conc'] = senolytic_conc
    params_combined['stem_cell_rate'] = stem_cell_rate

    # Initial conditions
    initial_conditions = np.zeros(max_divisions + 3)
    initial_conditions[0] = 200  # Initial population in E_0

    # Time span
    t_span = (0, 600)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Solve the systems
    sol_vanilla = solve_ivp(
        lambda t, y: endothelial_cell_dynamics(t, y, params_vanilla),
        t_span,
        initial_conditions,
        t_eval=t_eval,
        method='RK45'
    )

    sol_senolytics = solve_ivp(
        lambda t, y: endothelial_cell_dynamics(t, y, params_senolytics),
        t_span,
        initial_conditions,
        t_eval=t_eval,
        method='RK45'
    )

    sol_stem = solve_ivp(
        lambda t, y: endothelial_cell_dynamics(t, y, params_stem),
        t_span,
        initial_conditions,
        t_eval=t_eval,
        method='RK45'
    )

    sol_combined = solve_ivp(
        lambda t, y: endothelial_cell_dynamics(t, y, params_combined),
        t_span,
        initial_conditions,
        t_eval=t_eval,
        method='RK45'
    )

    # Analyze results
    results_vanilla = analyze_results(t_eval, sol_vanilla, max_divisions)
    results_senolytics = analyze_results(t_eval, sol_senolytics, max_divisions)
    results_stem = analyze_results(t_eval, sol_stem, max_divisions)
    results_combined = analyze_results(t_eval, sol_combined, max_divisions)

    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot total cells
    ax = axes[0, 0]
    ax.plot(t_eval, results_vanilla['total_cells'], 'k-', label='Vanilla')
    ax.plot(t_eval, results_senolytics['total_cells'], 'r-', label='With Senolytics')
    ax.plot(t_eval, results_stem['total_cells'], 'g-', label='With Stem Cells')
    ax.plot(t_eval, results_combined['total_cells'], 'b-', label='Combined')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Total Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot healthy cells
    ax = axes[0, 1]
    ax.plot(t_eval, results_vanilla['E_total'], 'k-', label='Vanilla')
    ax.plot(t_eval, results_senolytics['E_total'], 'r-', label='With Senolytics')
    ax.plot(t_eval, results_stem['E_total'], 'g-', label='With Stem Cells')
    ax.plot(t_eval, results_combined['E_total'], 'b-', label='Combined')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Healthy Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot senescent cells
    ax = axes[0, 2]
    ax.plot(t_eval, results_vanilla['S_total'], 'k-', label='Vanilla')
    ax.plot(t_eval, results_senolytics['S_total'], 'r-', label='With Senolytics')
    ax.plot(t_eval, results_stem['S_total'], 'g-', label='With Stem Cells')
    ax.plot(t_eval, results_combined['S_total'], 'b-', label='Combined')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell Count')
    ax.set_title('Senescent Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot senescent fraction
    ax = axes[1, 0]
    ax.plot(t_eval, results_vanilla['senescent_fraction'], 'k-', label='Vanilla')
    ax.plot(t_eval, results_senolytics['senescent_fraction'], 'r-', label='With Senolytics')
    ax.plot(t_eval, results_stem['senescent_fraction'], 'g-', label='With Stem Cells')
    ax.plot(t_eval, results_combined['senescent_fraction'], 'b-', label='Combined')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')
    ax.set_title('Fraction of Senescent Cells')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # Plot telomere length
    ax = axes[1, 1]
    for label, results in [('Vanilla', results_vanilla),
                           ('With Senolytics', results_senolytics),
                           ('With Stem Cells', results_stem),
                           ('Combined', results_combined)]:
        mask = ~np.isnan(results['telomere_length'])
        if np.any(mask):
            ax.plot(t_eval[mask], results['telomere_length'][mask], label=label)
    ax.set_xlabel('Time')
    ax.set_ylabel('Telomere Length')
    ax.set_title('Average Telomere Length of Healthy Cells')
    ax.axhline(y=20, color='r', linestyle='--', label='Critical Length')
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True)

    # Plot average division age
    ax = axes[1, 2]
    for label, results in [('Vanilla', results_vanilla),
                           ('With Senolytics', results_senolytics),
                           ('With Stem Cells', results_stem),
                           ('Combined', results_combined)]:
        mask = ~np.isnan(results['avg_division_age'])
        if np.any(mask):
            ax.plot(t_eval[mask], results['avg_division_age'][mask], label=label)
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Divisions')
    ax.set_title('Average Division Age of Healthy Cells')
    ax.set_ylim(0, max_divisions)
    ax.legend()
    ax.grid(True)

    plt.suptitle(
        f'Comparison of Model Variations (τ={tau_value}, Senolytic={senolytic_conc}, Stem Cell Rate={stem_cell_rate})',
        fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    return fig


# Function to run simulations and plot results for different parameter combinations
def run_parameter_study(tau_values=[5, 10, 15, 20],
                        senolytic_concs=[0, 5, 10],
                        stem_cell_rates=[0, 10, 20],
                        max_divisions=15):
    """
    Run a comprehensive parameter study across different shear stress values,
    senolytic concentrations, and stem cell input rates.
    """
    # Run simulations for each combination of parameters
    all_results = {}

    for senolytic_conc in senolytic_concs:
        for stem_cell_rate in stem_cell_rates:
            key = f"Seno_{senolytic_conc}_Stem_{stem_cell_rate}"
            results = run_multiple_shear_stress_simulations(
                tau_values,
                senolytic_conc=senolytic_conc,
                stem_cell_rate=stem_cell_rate,
                max_divisions=max_divisions
            )
            all_results[key] = results

            # Create and save plot for this parameter combination
            fig = plot_results(results, max_divisions)
            fig_title = f"plots_general_changes/shear_stress_senolytic_{senolytic_conc}_stem_{stem_cell_rate}.png"
            plt.savefig(fig_title, dpi=300, bbox_inches='tight')
            plt.close(fig)

    # Compare models at a specific shear stress value
    for tau in tau_values:
        fig = compare_models(tau, senolytic_conc=5, stem_cell_rate=10, max_divisions=max_divisions)
        fig_title = f"plots/model_comparison_tau_{tau}.png"
        plt.savefig(fig_title, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return all_results


# Main execution
if __name__ == "__main__":
    # Parameters
    tau_values = [0, 5, 10, 15, 20]
    senolytic_conc = 5  # Fixed senolytic concentration
    stem_cell_rate = 10  # Fixed stem cell input rate
    max_divisions = 15

    # Run simulations with varying shear stress
    results = run_multiple_shear_stress_simulations(
        tau_values,
        senolytic_conc=senolytic_conc,
        stem_cell_rate=stem_cell_rate,
        max_divisions=max_divisions
    )

    # Plot results
    fig = plot_results(results, max_divisions)
    plt.show()

    # Compare with vanilla model at middle tau value
    tau_mid = tau_values[len(tau_values) // 2]
    compare_fig = compare_models(tau_mid, senolytic_conc, stem_cell_rate, max_divisions)
    plt.show()