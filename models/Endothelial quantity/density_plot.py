import numpy as np
import matplotlib.pyplot as plt


def density_inhibition_factor(total_cells, carrying_capacity=3000):
    """
    Calculate the density-dependent inhibition factor for cell proliferation

    Parameters:
    -----------
    total_cells : float or numpy.ndarray
        Total number of cells (Etotal + Stotal)
    carrying_capacity : float, optional
        Carrying capacity (K), defaults to 3000

    Returns:
    --------
    float or numpy.ndarray
        Density inhibition factor (fdensity)
    """
    return 1 / (1 + np.exp(10 * (total_cells / carrying_capacity - 0.7)))


# Create a range of total cell population values
carrying_capacity = 3000
total_cells = np.linspace(0, carrying_capacity * 1.5, 500)
density_factors = density_inhibition_factor(total_cells, carrying_capacity)

# Create figure
plt.figure(figsize=(10, 6))
plt.plot(total_cells, density_factors, 'g-', linewidth=2.5)

# Add vertical lines at key points
plt.axvline(x=0.7 * carrying_capacity, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=carrying_capacity, color='b', linestyle='--', alpha=0.7)

# Add horizontal line at 0.5
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# Mark specific points
plt.plot(0.7 * carrying_capacity, 0.5, 'ro')
plt.text(0.7 * carrying_capacity + 50, 0.52, 'Total Cells = 70% of K\nfdensity = 0.5', va='bottom')

# Add annotations
plt.annotate('Minimal Inhibition\nfdensity ≈ 1', xy=(0.3 * carrying_capacity, 0.95),
             xytext=(0.3 * carrying_capacity, 0.8), ha='center',
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

plt.annotate('Strong Inhibition\nfdensity ≈ 0', xy=(1.2 * carrying_capacity, 0.05),
             xytext=(1.2 * carrying_capacity, 0.2), ha='center',
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

# Customize plot
plt.title('Density-Dependent Inhibition of Cell Proliferation', fontsize=14)
plt.xlabel('Total Cell Population (Etotal + Stotal)', fontsize=12)
plt.ylabel('Density Inhibition Factor (fdensity)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, carrying_capacity * 1.5)
plt.ylim(0, 1.05)

# Add carrying capacity reference
plt.text(carrying_capacity, -0.05, 'K = 3000\n(Carrying Capacity)', ha='center', va='top')

# Add shaded regions
plt.axvspan(0, 0.5 * carrying_capacity, alpha=0.2, color='green', label='Growth Phase')
plt.axvspan(0.5 * carrying_capacity, 0.9 * carrying_capacity, alpha=0.2, color='yellow', label='Transition Phase')
plt.axvspan(0.9 * carrying_capacity, carrying_capacity * 1.5, alpha=0.2, color='red', label='Inhibition Phase')

plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# Create a second plot showing the effect on proliferation rate
plt.figure(figsize=(10, 6))

# Assume a base proliferation rate of 0.06
base_rate = 0.06
effective_rates = base_rate * density_factors

plt.plot(total_cells, effective_rates, 'b-', linewidth=2.5, label='Effective Proliferation Rate')
plt.plot(total_cells, np.ones_like(total_cells) * base_rate, 'k--', linewidth=1.5, label='Base Rate (r = 0.06)')

# Add vertical lines
plt.axvline(x=0.7 * carrying_capacity, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=carrying_capacity, color='b', linestyle='--', alpha=0.7)

# Customize plot
plt.title('Effect of Density Inhibition on Cell Proliferation Rate', fontsize=14)
plt.xlabel('Total Cell Population (Etotal + Stotal)', fontsize=12)
plt.ylabel('Effective Proliferation Rate', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, carrying_capacity * 1.5)
plt.ylim(0, base_rate * 1.1)

# Add carrying capacity reference
plt.text(carrying_capacity, -0.002, 'K = 3000\n(Carrying Capacity)', ha='center', va='top')

# Add shaded regions
plt.axvspan(0, 0.5 * carrying_capacity, alpha=0.2, color='green')
plt.axvspan(0.5 * carrying_capacity, 0.9 * carrying_capacity, alpha=0.2, color='yellow')
plt.axvspan(0.9 * carrying_capacity, carrying_capacity * 1.5, alpha=0.2, color='red')

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()