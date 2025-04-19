import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d
import os

# Create directory for saving plots
if not os.path.exists('plots_with_toxic_seno'):
    os.makedirs('plots_with_toxic_seno')


# Define a more realistic shear stress effect function
def shear_stress_effect(tau):
    """
    Models how shear stress affects senescence rate.
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
    """
    # No effect at zero concentration
    if concentration <= 0:
        return 0

    # Sigmoid response curve to model drug effect
    max_effect = 0.15 * efficacy_factor
    ec50 = 5
    hill = 3
    effect = max_effect * (concentration ** hill) / (ec50 ** hill + concentration ** hill)
    return effect


# Define senolytic toxicity to healthy cells - ADJUSTED FOR BETTER THERAPEUTIC WINDOW
def senolytic_toxicity(concentration):
    """
    Models the toxic effect of senolytics on healthy cells.
    ADJUSTED: Reduced toxicity at low-to-moderate concentrations
    """
    if concentration <= 0:
        return 0

    # Base toxicity with linear component - REDUCED by 2.5x
    base_toxicity = 0.0004 * concentration  # Was 0.001

    # Non-linear component with improved therapeutic window
    max_toxicity = 0.05  # Maximum toxic effect (unchanged)
    ec50_toxicity = 20  # Increased threshold (was 15)
    hill_toxicity = 5  # Steeper curve at high concentrations (was 4)

    non_linear_toxicity = max_toxicity * (concentration ** hill_toxicity) / (
                ec50_toxicity ** hill_toxicity + concentration ** hill_toxicity)

    # Combine linear and non-linear components
    total_toxicity = base_toxicity + non_linear_toxicity

    return total_toxicity


# Function to create default stem cell distribution
def create_stem_cell_distribution(max_divisions, distribution_type='exponential', peak_stage=0, width=2):
    """
    Creates a distribution for how stem cells differentiate into different division stages.
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


# Define the combined cell dynamics model
def endothelial_cell_dynamics(t, y, params):
    """
    System of differential equations for endothelial cell dynamics with
    division-based population structure, stem cell input, and senolytic effects.
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

    # Calculate toxicity to healthy cells with enhanced non-linear effect - ADJUSTED
    healthy_cell_toxicity = senolytic_toxicity(senolytic_conc)

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
        # Older cells (higher division ages) are more sensitive to senolytic toxicity
        # ADJUSTED: Reduced age sensitivity factor
        age_sensitivity_factor = 1.0 + 0.08 * i  # Was 0.1
        death_rate = d_E * (1 + 0.03 * i) + healthy_cell_toxicity * age_sensitivity_factor

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


# Function to analyze results with improved numerical stability
def analyze_results(t_eval, solution, max_divisions):
    """
    Analyze the results of the simulation with improved numerical stability
    for calculations involving small numbers.
    """
    # Total healthy cells across all division stages
    E_total = np.sum(solution.y[:max_divisions + 1], axis=0)

    # Senescent cells by cause
    S_tel = solution.y[max_divisions + 1]  # Telomere-induced senescent cells
    S_stress = solution.y[max_divisions + 2]  # Stress-induced senescent cells
    S_total = S_tel + S_stress  # Total senescent cells

    # Total cell population
    total_cells = E_total + S_total

    # Calculate statistics with improved numerical stability
    # Use a higher threshold for division to avoid numerical issues
    stability_threshold = 1.0  # Minimum number of cells needed for reliable fraction calculation

    # Initialize fractions with zeros
    senescent_fraction = np.zeros_like(total_cells)
    tel_fraction_of_senescent = np.zeros_like(total_cells)

    # Calculate fractions only when there are enough cells
    for i in range(len(total_cells)):
        if total_cells[i] > stability_threshold:
            senescent_fraction[i] = S_total[i] / total_cells[i]
        else:
            # When total population is too small, use the last valid value or 0
            if i > 0:
                senescent_fraction[i] = senescent_fraction[i - 1]
            else:
                senescent_fraction[i] = 0

        if S_total[i] > stability_threshold:
            tel_fraction_of_senescent[i] = S_tel[i] / S_total[i]
        else:
            # When senescent population is too small, use the last valid value or 0
            if i > 0:
                tel_fraction_of_senescent[i] = tel_fraction_of_senescent[i - 1]
            else:
                tel_fraction_of_senescent[i] = 0

    # Apply light smoothing to reduce numerical noise
    if len(senescent_fraction) > 3:  # Only smooth if we have enough data points
        senescent_fraction = gaussian_filter1d(senescent_fraction, sigma=2)
        tel_fraction_of_senescent = gaussian_filter1d(tel_fraction_of_senescent, sigma=2)

    # Calculate average division age of healthy cells with improved handling of edge cases
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

    # Calculate "telomere length" based on average division age
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


# Plot the simulation results with different shear stress values
def plot_results(results, max_divisions=15, fig_size=(15, 13)):
    """
    Plot comprehensive results from multiple simulations with different shear stress values,
    fixed senolytic concentration, and fixed stem cell input rate.
    With improved stability for fraction plots.
    """
    fig, axes = plt.subplots(3, 3, figsize=fig_size)

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

    # Plot senescent fraction - with fixed y-axis limits
    ax = axes[1, 0]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['senescent_fraction'], color=colors[i],
                label=f'τ={result["tau"]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')
    ax.set_title('Fraction of Senescent Cells')
    ax.set_ylim(0, 1)  # Enforce limits between 0 and 1
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

    # Plot fraction of senescent cells that are telomere-induced - with fixed y-axis limits
    ax = axes[1, 2]
    for i, result in enumerate(results):
        ax.plot(result['t'], result['tel_fraction_of_senescent'], color=colors[i],
                label=f'τ={result["tau"]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')
    ax.set_title('Fraction of Senescent Cells from Telomere Shortening')
    ax.set_ylim(0, 1)  # Enforce limits between 0 and 1
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

    # Add suptitle with parameter information
    senolytic_str = f"Senolytic Concentration: {senolytic_conc}"
    stem_cell_str = f"Stem Cell Rate: {stem_cell_rate} (Exponential Distribution)"
    plt.suptitle(f'Endothelial Cell Dynamics with Different Shear Stress Levels\n{senolytic_str}, {stem_cell_str}',
                 fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    return fig


# Plot toxicity curve with improved parameters
def plot_toxicity_curve():
    """
    Plot the dose-response curve for senolytic toxicity effects.
    Shows both the effect on senescent cells and the toxic effect on healthy cells.
    """
    concentrations = np.linspace(0, 30, 100)

    # Calculate effects
    senolytic_effects = [senolytic_effect(c) for c in concentrations]
    toxicity_effects = [senolytic_toxicity(c) for c in concentrations]

    # Calculate therapeutic index (ratio of desired effect to toxic effect)
    therapeutic_index = np.array(senolytic_effects) / np.maximum(np.array(toxicity_effects), 1e-10)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot dose-response curves
    ax1.plot(concentrations, senolytic_effects, 'b-', label='Effect on Senescent Cells')
    ax1.plot(concentrations, toxicity_effects, 'r-', label='Toxicity to Healthy Cells')
    ax1.set_xlabel('Senolytic Concentration')
    ax1.set_ylabel('Effect (Death Rate Increase)')
    ax1.set_title('Senolytic Dose-Response Curves')
    ax1.legend()
    ax1.grid(True)

    # Plot therapeutic index
    ax2.plot(concentrations, therapeutic_index, 'g-')
    ax2.set_xlabel('Senolytic Concentration')
    ax2.set_ylabel('Therapeutic Index\n(Senolytic Effect / Toxicity)')
    ax2.set_title('Therapeutic Index\n(Higher is Better)')
    ax2.grid(True)

    # Add optimal concentration marker
    # Find the concentration with maximum therapeutic index
    optimal_idx = np.argmax(therapeutic_index)
    optimal_conc = concentrations[optimal_idx]
    ax2.axvline(x=optimal_conc, color='k', linestyle='--',
                label=f'Optimal Concentration ≈ {optimal_conc:.1f}')
    ax2.plot(optimal_conc, therapeutic_index[optimal_idx], 'ro', markersize=8)
    ax2.legend()

    plt.suptitle('Senolytic Effects and Toxicity Profiles', fontsize=16)
    plt.tight_layout()

    return fig


# Function to find optimal senolytic concentration for a given stem cell rate and shear stress
def find_optimal_senolytic_concentration(tau_value, stem_cell_rate, max_divisions=15, t_span=(0, 600)):
    """
    Finds the optimal senolytic concentration that maximizes healthy cell count
    while keeping toxicity in check.
    """
    # Define range of senolytic concentrations to test
    senolytic_concs = np.linspace(0, 25, 26)  # 0 to 25 in steps of 1

    # Store results
    healthy_counts = []
    senescent_counts = []
    senescent_fractions = []

    # Run simulations for each concentration
    for conc in senolytic_concs:
        t_eval, solution, _ = run_simulation(
            tau_value,
            senolytic_conc=conc,
            stem_cell_rate=stem_cell_rate,
            max_divisions=max_divisions,
            t_span=t_span
        )

        # Analyze results
        result = analyze_results(t_eval, solution, max_divisions)

        # Store final values (at the end of simulation)
        healthy_counts.append(result['E_total'][-1])
        senescent_counts.append(result['S_total'][-1])
        senescent_fractions.append(result['senescent_fraction'][-1])

    # Find optimal concentration (maximum healthy cells)
    optimal_idx = np.argmax(healthy_counts)
    optimal_conc = senolytic_concs[optimal_idx]

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot healthy cell count
    ax = axes[0]
    ax.plot(senolytic_concs, healthy_counts, 'g-o')
    ax.axvline(x=optimal_conc, color='r', linestyle='--',
               label=f'Optimal = {optimal_conc:.1f}')
    ax.set_xlabel('Senolytic Concentration')
    ax.set_ylabel('Healthy Cell Count')
    ax.set_title('Final Healthy Cell Count')
    ax.legend()
    ax.grid(True)

    # Plot senescent cell count
    ax = axes[1]
    ax.plot(senolytic_concs, senescent_counts, 'r-o')
    ax.axvline(x=optimal_conc, color='r', linestyle='--')
    ax.set_xlabel('Senolytic Concentration')
    ax.set_ylabel('Senescent Cell Count')
    ax.set_title('Final Senescent Cell Count')
    ax.grid(True)

    # Plot senescent fraction
    ax = axes[2]
    ax.plot(senolytic_concs, senescent_fractions, 'b-o')
    ax.axvline(x=optimal_conc, color='r', linestyle='--')
    ax.set_xlabel('Senolytic Concentration')
    ax.set_ylabel('Fraction')
    ax.set_title('Final Senescent Cell Fraction')
    ax.set_ylim(0, 1)
    ax.grid(True)

    plt.suptitle(f'Optimization of Senolytic Concentration (τ={tau_value}, Stem Cell Rate={stem_cell_rate})',
                 fontsize=16)
    plt.tight_layout()

    return optimal_conc, fig


# Function to run parameter study across multiple senolytic and stem cell values
def run_parameter_study():
    """
    Run a parameter study across different senolytic concentrations and stem cell rates.
    Generates multiple plots showing the effects of each parameter combination.
    """
    # Define parameter ranges
    tau_values = [0, 5, 10, 15, 20]  # Shear stress values
    senolytic_concs = [0, 5, 10, 20]  # Senolytic concentrations - including higher concentration
    stem_cell_rates = [0, 5, 10, 20]  # Stem cell input rates
    max_divisions = 15

    # Total number of plots to generate
    total_plots = len(senolytic_concs) * len(stem_cell_rates)
    print(f"Generating {total_plots} plots for different parameter combinations...")

    # First, generate the toxicity curve
    fig_toxicity = plot_toxicity_curve()
    fig_toxicity.savefig("plots/senolytic_toxicity_curve.png", dpi=300, bbox_inches='tight')
    plt.close(fig_toxicity)
    print("Generated senolytic toxicity curve")

    # Generate optimal concentration plots for different conditions
    for stem_cell_rate in stem_cell_rates:
        for tau in [5, 15]:  # Low and high shear stress
            optimal_conc, fig_opt = find_optimal_senolytic_concentration(tau, stem_cell_rate)
            fig_opt.savefig(f"plots/optimal_senolytic_tau_{tau}_stem_{stem_cell_rate}.png", dpi=300,
                            bbox_inches='tight')
            plt.close(fig_opt)
            print(f"Found optimal senolytic concentration for τ={tau}, Stem={stem_cell_rate}: {optimal_conc:.1f}")

    # Run simulations and generate plots for each combination
    for seno_idx, senolytic_conc in enumerate(senolytic_concs):
        for stem_idx, stem_cell_rate in enumerate(stem_cell_rates):
            # Display progress
            plot_num = seno_idx * len(stem_cell_rates) + stem_idx + 1
            print(f"Plot {plot_num}/{total_plots}: Senolytic={senolytic_conc}, Stem Cell Rate={stem_cell_rate}")

            # Run simulations for all tau values with this parameter combination
            results = run_multiple_shear_stress_simulations(
                tau_values,
                senolytic_conc=senolytic_conc,
                stem_cell_rate=stem_cell_rate,
                max_divisions=max_divisions
            )

            # Generate and save plot
            fig = plot_results(results, max_divisions)

            # Save the figure
            filename = f"plots_with_toxic_seno/senolytic_{senolytic_conc}_stem_{stem_cell_rate}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory

            print(f"  Saved plot to {filename}")

    print("Parameter study complete. All plots saved to 'plots' directory.")


# Execute the parameter study
if __name__ == "__main__":
    run_parameter_study()