#!/usr/bin/env python3
"""
Verification script for stress-induced senescence parameters
Based on experimental data:
- tau = 0: 5% senescence in 18 hours
- tau = 1.4 Pa: 30% senescence in 18 hours
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_stress_parameters():
    """Calculate the linear model parameters from experimental data."""

    print("=" * 60)
    print("STRESS-INDUCED SENESCENCE PARAMETER CALCULATION")
    print("=" * 60)

    # Experimental data
    tau_points = [0, 1.4]  # Pa
    senescence_18h = [0.05, 0.30]  # 5% and 30% in 18 hours

    print("\nExperimental Data:")
    print(f"At tau = {tau_points[0]} Pa: {senescence_18h[0] * 100}% senescence in 18 hours")
    print(f"At tau = {tau_points[1]} Pa: {senescence_18h[1] * 100}% senescence in 18 hours")

    # Convert to hourly rates
    hourly_rates = [rate / 18 for rate in senescence_18h]

    print("\nConverted to hourly rates:")
    for i, (tau, rate) in enumerate(zip(tau_points, hourly_rates)):
        print(f"At tau = {tau} Pa: {rate:.6f} per hour ({rate * 100:.4f}% per hour)")

    # Calculate linear model: rate = a + b * tau
    # Using two points: (0, hourly_rates[0]) and (1.4, hourly_rates[1])
    a = hourly_rates[0]  # intercept (rate at tau = 0)
    b = (hourly_rates[1] - hourly_rates[0]) / (tau_points[1] - tau_points[0])  # slope

    print(f"\nLinear Model: rate = a + b * tau")
    print(f"a (intercept): {a:.6f}")
    print(f"b (slope): {b:.6f}")
    print(f"Formula: rate = {a:.6f} + {b:.6f} * tau")

    return a, b, tau_points, hourly_rates


def verify_model(a, b, tau_points, hourly_rates):
    """Verify the model matches experimental data."""

    print("\n" + "=" * 40)
    print("MODEL VERIFICATION")
    print("=" * 40)

    for tau_exp, rate_exp in zip(tau_points, hourly_rates):
        if tau_exp <= 0:
            rate_model = a
        else:
            rate_model = a + b * tau_exp

        # Convert back to 18-hour percentage
        percent_18h_exp = rate_exp * 18 * 100
        percent_18h_model = rate_model * 18 * 100

        print(f"\nAt tau = {tau_exp} Pa:")
        print(f"  Expected hourly rate: {rate_exp:.6f}")
        print(f"  Model hourly rate:    {rate_model:.6f}")
        print(f"  Expected 18h %:       {percent_18h_exp:.1f}%")
        print(f"  Model 18h %:          {percent_18h_model:.1f}%")
        print(f"  Match: {'✓' if abs(percent_18h_exp - percent_18h_model) < 0.1 else '✗'}")


def test_additional_values(a, b):
    """Test the model at additional tau values."""

    print("\n" + "=" * 40)
    print("PREDICTIONS AT OTHER TAU VALUES")
    print("=" * 40)

    test_tau_values = [0.5, 1.0, 2.0, 3.0, 5.0]

    for tau in test_tau_values:
        if tau <= 0:
            rate = a
        else:
            rate = a + b * tau

        # Calculate percentages for different time periods
        percent_1h = rate * 100
        percent_6h = rate * 6 * 100
        percent_18h = rate * 18 * 100
        percent_24h = rate * 24 * 100

        print(f"\nAt tau = {tau} Pa:")
        print(f"  Hourly rate: {rate:.6f} ({percent_1h:.4f}% per hour)")
        print(f"  6-hour:      {percent_6h:.2f}%")
        print(f"  18-hour:     {percent_18h:.1f}%")
        print(f"  24-hour:     {percent_24h:.1f}%")


def generate_code_template(a, b):
    """Generate the actual code template for implementation."""

    print("\n" + "=" * 40)
    print("IMPLEMENTATION CODE")
    print("=" * 40)

    code = f'''
def _calculate_stress_factor(self, config, dt_hours=1.0):
    """
    Calculate stress-induced senescence rate based on experimental data.

    Parameters derived from:
    - tau = 0: 5% senescence in 18 hours
    - tau = 1.4 Pa: 30% senescence in 18 hours

    Args:
        config: Simulation configuration
        dt_hours: Time step in hours

    Returns:
        Senescence probability for the given time step
    """
    tau = self.local_shear_stress

    if tau <= 0:
        hourly_rate = {a:.6f}  # Base senescence rate per hour
    else:
        hourly_rate = {a:.6f} + tau * {b:.6f}  # Linear increase with tau

    # Apply cell resistance if available
    if hasattr(self, 'cellular_resistance') and self.cellular_resistance > 0:
        hourly_rate /= self.cellular_resistance

    # Convert to probability for the specific time step
    senescence_probability = hourly_rate * dt_hours

    return senescence_probability

# Alternative simplified version:
def _calculate_stress_factor_simple(self, config):
    """Simplified version returning hourly rate."""
    tau = self.local_shear_stress
    if tau <= 0:
        return {a:.6f}
    else:
        return {a:.6f} + tau * {b:.6f}
'''

    print(code)


def plot_stress_response(a, b):
    """Create a plot showing the stress-response relationship."""

    print("\n" + "=" * 40)
    print("GENERATING VISUALIZATION")
    print("=" * 40)

    # Generate tau range
    tau_range = np.linspace(0, 3, 100)

    # Calculate rates
    rates = []
    for tau in tau_range:
        if tau <= 0:
            rate = a
        else:
            rate = a + b * tau
        rates.append(rate)

    rates = np.array(rates)

    # Convert to 18-hour percentages for plotting
    percent_18h = rates * 18 * 100

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(tau_range, percent_18h, 'b-', linewidth=2, label='Linear Model')

    # Mark experimental points
    plt.plot([0, 1.4], [5, 30], 'ro', markersize=8, label='Experimental Data')

    plt.xlabel('Shear Stress τ (Pa)')
    plt.ylabel('Senescence Percentage in 18 Hours (%)')
    plt.title('Stress-Induced Senescence Model')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add annotations
    plt.annotate('5% at τ=0', xy=(0, 5), xytext=(0.5, 10),
                 arrowprops=dict(arrowstyle='->', color='red'))
    plt.annotate('30% at τ=1.4', xy=(1.4, 30), xytext=(2, 25),
                 arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    plt.savefig('stress_senescence_model.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'stress_senescence_model.png'")
    plt.show()


def main():
    """Main verification function."""

    # Calculate parameters
    a, b, tau_points, hourly_rates = calculate_stress_parameters()

    # Verify model
    verify_model(a, b, tau_points, hourly_rates)

    # Test additional values
    test_additional_values(a, b)

    # Generate implementation code
    generate_code_template(a, b)

    # Create visualization
    try:
        plot_stress_response(a, b)
    except ImportError:
        print("\nMatplotlib not available - skipping plot generation")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Final parameters for your implementation:")
    print(f"  Base rate (a): {a:.6f}")
    print(f"  Stress coefficient (b): {b:.6f}")
    print(f"  Formula: rate = {a:.6f} + {b:.6f} * tau")
    print("=" * 60)


if __name__ == "__main__":
    main()