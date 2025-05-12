import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Define the senolytic effect functions
def senolytic_effect_senescent(c, efficacy=1.0):
    """
    Calculate senolytic effect on senescent cells.

    Parameters:
    -----------
    c : float or array
        Senolytic concentration
    efficacy : float
        Efficacy factor (1.0 for telomere-induced, 1.2 for stress-induced)

    Returns:
    --------
    float or array
        Senolytic effect on senescent cells
    """
    return efficacy * 0.15 * (c ** 3) / (5 ** 3 + c ** 3)


def senolytic_toxicity_healthy(c):
    """
    Calculate senolytic toxicity to healthy cells.

    Parameters:
    -----------
    c : float or array
        Senolytic concentration

    Returns:
    --------
    float or array
        Toxicity effect on healthy cells
    """
    return 0.0004 * c + 0.05 * (c ** 5) / (20 ** 5 + c ** 5)


def division_modulated_toxicity(c, division_stage, max_divisions=15):
    """
    Calculate division-modulated toxicity to healthy cells.

    Parameters:
    -----------
    c : float or array
        Senolytic concentration
    division_stage : int
        Division stage of the cell (0 to N)
    max_divisions : int
        Maximum division count

    Returns:
    --------
    float or array
        Division-stage modulated toxicity
    """
    return senolytic_toxicity_healthy(c) * (1 + 0.08 * division_stage)


# Generate concentration range
concentrations = np.linspace(0, 40, 1000)

# Calculate different effects
senescent_tel = senolytic_effect_senescent(concentrations, 1.0)  # Telomere-induced
senescent_stress = senolytic_effect_senescent(concentrations, 1.2)  # Stress-induced
healthy_base = senolytic_toxicity_healthy(concentrations)  # Base toxicity to healthy

# Calculate toxicity for different division stages
division_stages = [0, 5, 10, 15]
healthy_by_division = [division_modulated_toxicity(concentrations, i) for i in division_stages]

# Calculate selectivity ratio (senescent effect / healthy toxicity)
selectivity_tel = senescent_tel / np.maximum(healthy_base, 1e-10)  # Avoid division by zero
selectivity_stress = senescent_stress / np.maximum(healthy_base, 1e-10)

# Calculate optimal concentrations (maximum selectivity)
optimal_tel_idx = np.argmax(selectivity_tel)
optimal_tel_conc = concentrations[optimal_tel_idx]
optimal_stress_idx = np.argmax(selectivity_stress)
optimal_stress_conc = concentrations[optimal_stress_idx]

# Set up plots
plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, height_ratios=[1, 1])

# Plot 1: Basic comparison of effects
ax1 = plt.subplot(gs[0, 0])
ax1.plot(concentrations, senescent_tel, 'r-', linewidth=2.5, label='Effect on telomere-induced senescent cells')
ax1.plot(concentrations, senescent_stress, 'r--', linewidth=2.5, label='Effect on stress-induced senescent cells')
ax1.plot(concentrations, healthy_base, 'b-', linewidth=2.5, label='Toxicity to healthy cells')
ax1.set_title('Senolytic Effects on Different Cell Types', fontsize=14)
ax1.set_xlabel('Senolytic Concentration', fontsize=12)
ax1.set_ylabel('Effect Rate', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')

# Add annotations to key points
c20_idx = np.argmin(np.abs(concentrations - 20))
ax1.plot([20], [healthy_base[c20_idx]], 'bo', markersize=6)
ax1.annotate(f'At c=20: Toxicity={healthy_base[c20_idx]:.4f}',
             xy=(20, healthy_base[c20_idx]), xytext=(22, healthy_base[c20_idx] + 0.01),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=5))

c5_idx = np.argmin(np.abs(concentrations - 5))
ax1.plot([5], [senescent_tel[c5_idx]], 'ro', markersize=6)
ax1.annotate(f'At c=5: Effect={senescent_tel[c5_idx]:.4f}',
             xy=(5, senescent_tel[c5_idx]), xytext=(7, senescent_tel[c5_idx] + 0.02),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=5))

# Plot 2: Division-stage modulated toxicity
ax2 = plt.subplot(gs[0, 1])
colors = plt.cm.viridis(np.linspace(0, 1, len(division_stages)))
for i, (stage, toxicity) in enumerate(zip(division_stages, healthy_by_division)):
    ax2.plot(concentrations, toxicity, color=colors[i], linewidth=2.5,
             label=f'Division stage {stage} (i={stage}/{max(division_stages)})')
ax2.set_title('Toxicity Modulated by Division Stage', fontsize=14)
ax2.set_xlabel('Senolytic Concentration', fontsize=12)
ax2.set_ylabel('Toxicity Rate', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')

# Plot 3: Selectivity ratio
ax3 = plt.subplot(gs[1, 0])
ax3.plot(concentrations, selectivity_tel, 'g-', linewidth=2.5,
         label='Selectivity for telomere-induced senescent cells')
ax3.plot(concentrations, selectivity_stress, 'g--', linewidth=2.5,
         label='Selectivity for stress-induced senescent cells')
ax3.axvline(x=optimal_tel_conc, color='r', linestyle='--', alpha=0.7)
ax3.axvline(x=optimal_stress_conc, color='b', linestyle='--', alpha=0.7)
ax3.set_title('Selectivity Ratio (Senescent Effect / Healthy Toxicity)', fontsize=14)
ax3.set_xlabel('Senolytic Concentration', fontsize=12)
ax3.set_ylabel('Selectivity Ratio', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best')

# Annotate optimal concentrations
ax3.annotate(f'Optimal for telomere senescent: c={optimal_tel_conc:.2f}',
             xy=(optimal_tel_conc, selectivity_tel[optimal_tel_idx]),
             xytext=(optimal_tel_conc + 2, selectivity_tel[optimal_tel_idx] * 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=5))
ax3.annotate(f'Optimal for stress senescent: c={optimal_stress_conc:.2f}',
             xy=(optimal_stress_conc, selectivity_stress[optimal_stress_idx]),
             xytext=(optimal_stress_conc + 2, selectivity_stress[optimal_stress_idx] * 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=5))

# Plot 4: Component analysis of toxicity function
ax4 = plt.subplot(gs[1, 1])
# Linear component
linear_component = 0.0004 * concentrations
# Nonlinear component
nonlinear_component = 0.05 * (concentrations ** 5) / (20 ** 5 + concentrations ** 5)
# Total toxicity (sum of components)
total_toxicity = linear_component + nonlinear_component

ax4.plot(concentrations, linear_component, 'b-', linewidth=2, label='Linear component: 0.0004·c')
ax4.plot(concentrations, nonlinear_component, 'r-', linewidth=2,
         label='Nonlinear component: 0.05·c⁵/(20⁵+c⁵)')
ax4.plot(concentrations, total_toxicity, 'k-', linewidth=2.5, label='Total toxicity')
ax4.set_title('Components of Toxicity Function', fontsize=14)
ax4.set_xlabel('Senolytic Concentration', fontsize=12)
ax4.set_ylabel('Toxicity Rate', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='best')

# Highlight the crossover point where nonlinear becomes dominant
crossover_idx = np.argmin(np.abs(linear_component - nonlinear_component))
crossover_conc = concentrations[crossover_idx]
ax4.plot([crossover_conc], [linear_component[crossover_idx]], 'ko', markersize=6)
ax4.annotate(f'Crossover at c≈{crossover_conc:.2f}',
             xy=(crossover_conc, linear_component[crossover_idx]),
             xytext=(crossover_conc + 2, linear_component[crossover_idx] + 0.01),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=5))

plt.tight_layout()
plt.show()


# Create a pulse simulation plot
def simulate_pulse_treatment(times, pulse_interval=50, pulse_duration=10, continuous_conc=7, base_death_rate=0.02):
    """Simulate senolytic concentrations and resulting death rates for pulsed vs continuous treatment"""
    # Initialize arrays
    continuous_concs = np.ones_like(times) * continuous_conc
    pulsed_concs = np.zeros_like(times)

    # Calculate equivalent pulse concentration
    pulse_conc = continuous_conc * (pulse_interval / pulse_duration)

    # Fill in pulse concentrations
    for t in range(len(times)):
        cycle_position = times[t] % pulse_interval
        if cycle_position < pulse_duration:
            pulsed_concs[t] = pulse_conc

    # Calculate death rates
    continuous_death_rates_tel = base_death_rate + senolytic_effect_senescent(continuous_concs, 1.0)
    continuous_death_rates_stress = base_death_rate + senolytic_effect_senescent(continuous_concs, 1.2)
    continuous_toxicity = senolytic_toxicity_healthy(continuous_concs)

    pulsed_death_rates_tel = base_death_rate + senolytic_effect_senescent(pulsed_concs, 1.0)
    pulsed_death_rates_stress = base_death_rate + senolytic_effect_senescent(pulsed_concs, 1.2)
    pulsed_toxicity = senolytic_toxicity_healthy(pulsed_concs)

    return {
        'times': times,
        'continuous_concs': continuous_concs,
        'pulsed_concs': pulsed_concs,
        'continuous_death_rates_tel': continuous_death_rates_tel,
        'continuous_death_rates_stress': continuous_death_rates_stress,
        'continuous_toxicity': continuous_toxicity,
        'pulsed_death_rates_tel': pulsed_death_rates_tel,
        'pulsed_death_rates_stress': pulsed_death_rates_stress,
        'pulsed_toxicity': pulsed_toxicity
    }


# Simulation settings
time_range = np.arange(0, 200, 0.5)
results = simulate_pulse_treatment(time_range)

# Create a pulse treatment visualization
plt.figure(figsize=(14, 10))
gs = GridSpec(3, 1, height_ratios=[1, 1, 1])

# Concentration plot
ax1 = plt.subplot(gs[0])
ax1.plot(results['times'], results['continuous_concs'], 'b-', linewidth=2, label='Continuous treatment')
ax1.plot(results['times'], results['pulsed_concs'], 'r-', linewidth=2, label='Pulsed treatment')
ax1.set_title('Senolytic Concentration Profiles', fontsize=14)
ax1.set_ylabel('Concentration', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')

# Death rate for senescent cells
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(results['times'], results['continuous_death_rates_tel'], 'b-', linewidth=1.5,
         label='Continuous - telomere-induced')
ax2.plot(results['times'], results['continuous_death_rates_stress'], 'b--', linewidth=1.5,
         label='Continuous - stress-induced')
ax2.plot(results['times'], results['pulsed_death_rates_tel'], 'r-', linewidth=1.5,
         label='Pulsed - telomere-induced')
ax2.plot(results['times'], results['pulsed_death_rates_stress'], 'r--', linewidth=1.5,
         label='Pulsed - stress-induced')
ax2.axhline(y=0.02, color='k', linestyle=':', label='Base death rate (no senolytic)')
ax2.set_title('Death Rates for Senescent Cells', fontsize=14)
ax2.set_ylabel('Death Rate', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')

# Toxicity to healthy cells
ax3 = plt.subplot(gs[2], sharex=ax1)
ax3.plot(results['times'], results['continuous_toxicity'], 'b-', linewidth=2, label='Continuous toxicity')
ax3.plot(results['times'], results['pulsed_toxicity'], 'r-', linewidth=2, label='Pulsed toxicity')
ax3.set_title('Toxicity to Healthy Cells', fontsize=14)
ax3.set_xlabel('Time', fontsize=12)
ax3.set_ylabel('Toxicity Rate', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best')

plt.tight_layout()
plt.show()