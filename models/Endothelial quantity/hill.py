import numpy as np
import matplotlib.pyplot as plt

# Clean, minimal style
plt.style.use('bmh')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'font.family': 'DejaVu Sans',
    'figure.figsize': (8, 5),
    'figure.dpi': 100
})

# Function to calculate the senolytic effect
def senolytic_effect(C, efficacy_factor=1.0):
    return efficacy_factor * (0.15 * C**3) / (5**3 + C**3)

# Create concentration range
concentrations = np.linspace(0, 40, 500)

# Calculate effects
effect_tel = senolytic_effect(concentrations, 1.0)
effect_stress = senolytic_effect(concentrations, 1.2)

# Create the plot
fig, ax = plt.subplots()

# Plot curves - simple and clean
ax.plot(concentrations, effect_tel, color='#1f77b4', lw=2.5,
        label='Telomere-induced (factor = 1.0)')
ax.plot(concentrations, effect_stress, color='#d62728', lw=2.5,
        label='Stress-induced (factor = 1.2)')

# Key concentrations from the paper
continuous = [7.29, 6.95]  # [low stress, high stress]
pulsed = [36.46, 34.75]    # [low stress, high stress]

# Add markers at continuous dose points
ax.scatter(continuous[0], senolytic_effect(continuous[0], 1.0),
           color='#1f77b4', s=80, zorder=5, marker='o')
ax.scatter(continuous[1], senolytic_effect(continuous[1], 1.2),
           color='#d62728', s=80, zorder=5, marker='o')

# Add markers at pulsed dose points
ax.scatter(pulsed[0], senolytic_effect(pulsed[0], 1.0),
           color='#1f77b4', s=80, zorder=5, marker='s')
ax.scatter(pulsed[1], senolytic_effect(pulsed[1], 1.2),
           color='#d62728', s=80, zorder=5, marker='s')

# Simple labels
ax.set_xlabel('Senolytic Concentration (C)')
ax.set_ylabel('Senolytic Effect (death rate)')
ax.set_title('Hill Function: Senolytic Effect vs Concentration')

# Clean equation
ax.text(20, 0.03, r"$senolytic~effect = factor \cdot \frac{0.15 \cdot C^3}{5^3 + C^3}$",
        fontsize=12, ha='center')

# Simple legend
legend1 = ax.legend(loc='upper right')

# Add second legend for treatment types
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=8, label='Continuous dose'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
           markersize=8, label='Pulsed dose')
]
ax.legend(handles=legend_elements, loc='lower right')
ax.add_artist(legend1)

# Add basic annotation for the key concept
ax.annotate('Diminishing returns\nat higher concentrations',
            xy=(30, 0.06), ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

plt.tight_layout()
plt.show()