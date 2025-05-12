import numpy as np
import matplotlib.pyplot as plt


def shear_stress_effect(tau):
    """
    Calculate the shear stress effect on senescence rate

    Parameters:
    -----------
    tau : float or numpy.ndarray
        Shear stress value(s)

    Returns:
    --------
    float or numpy.ndarray
        Senescence rate effect
    """
    if isinstance(tau, np.ndarray):
        result = np.zeros_like(tau, dtype=float)

        # Low shear stress regime (τ ≤ 10)
        mask_low = tau <= 10
        result[mask_low] = 0.002 + 0.0005 * tau[mask_low]

        # Moderate shear stress regime (10 < τ ≤ 20)
        mask_med = (tau > 10) & (tau <= 20)
        result[mask_med] = 0.007 + 0.001 * (tau[mask_med] - 10)

        # High shear stress regime (τ > 20)
        mask_high = tau > 20
        result[mask_high] = 0.017 + 0.005 * (tau[mask_high] - 20)

        return result
    else:
        # For single value input
        if tau <= 10:
            return 0.002 + 0.0005 * tau
        elif tau <= 20:
            return 0.007 + 0.001 * (tau - 10)
        else:
            return 0.017 + 0.005 * (tau - 20)


# Create a range of shear stress values
tau_values = np.linspace(0, 30, 300)
senescence_rates = shear_stress_effect(tau_values)

# Create figure
plt.figure(figsize=(10, 6))
plt.plot(tau_values, senescence_rates, 'b-', linewidth=2.5)

# Add vertical lines at the transition points
plt.axvline(x=10, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=20, color='r', linestyle='--', alpha=0.7)

# Add regime labels
plt.text(5, 0.025, 'Low Shear\nStress Regime\nSlope = 0.0005', ha='center', fontsize=10)
plt.text(15, 0.025, 'Moderate Shear\nStress Regime\nSlope = 0.001', ha='center', fontsize=10)
plt.text(25, 0.05, 'High Shear\nStress Regime\nSlope = 0.005', ha='center', fontsize=10)

# Customize plot
plt.title('Shear Stress Effect on Endothelial Cell Senescence Rate', fontsize=14)
plt.xlabel('Shear Stress (τ)', fontsize=12)
plt.ylabel('Senescence Rate Effect (γτ)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, 30)
plt.ylim(0, 0.07)

# Mark specific points
plt.plot(10, 0.007, 'ro')
plt.text(10.5, 0.007, 'τ=10, γτ=0.007', va='bottom')
plt.plot(20, 0.017, 'ro')
plt.text(20.5, 0.017, 'τ=20, γτ=0.017', va='bottom')

plt.tight_layout()
plt.show()