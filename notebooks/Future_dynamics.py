import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for PyCharm compatibility
import matplotlib.pyplot as plt
from typing import Dict, List
import os


def simulate_mpc_dynamics_demo():
    """
    Simplified demo of MPC dynamics visualization without requiring the full MPC controller.
    This demonstrates the key concepts you can apply to your actual controller.
    """

    # Simulate what your predict_future_state might return
    def mock_predict_future_state(current_state, control_sequence):
        """Mock version of your predict_future_state function."""
        predictions = []
        state = current_state.copy()
        dt = 0.1  # time step

        for i, u in enumerate(control_sequence):
            # Mock senescence dynamics (increases with shear stress)
            senescence_rate = 0.01 * (1 + 0.1 * u)
            state['senescence_fraction'] = min(1.0, state['senescence_fraction'] + senescence_rate * dt)

            # Mock hole dynamics (depends on senescence)
            if state['senescence_fraction'] > 0.3:  # threshold
                hole_creation_prob = 0.05 * (state['senescence_fraction'] - 0.3)
                expected_new_holes = hole_creation_prob * dt
                state['hole_count'] += expected_new_holes
                state['hole_area_fraction'] = state['hole_count'] * 0.01  # assume 1% per hole

            # Mock alignment dynamics (responds to shear stress)
            target_alignment = 0.0
            tau = 2.0  # time constant
            alignment_change = (target_alignment - state['mean_alignment_error']) / tau * dt
            state['mean_alignment_error'] = max(0, state['mean_alignment_error'] + alignment_change)

            # Mock response dynamics
            target_response = 2.0
            response_tau = 1.5
            if len(state['responses']) > 0:
                current_response = np.mean(state['responses'])
                response_change = (target_response - current_response) / response_tau * dt
                new_response = current_response + response_change
                state['responses'] = np.array([new_response] * len(state['responses']))

            # Update time
            state['time'] = current_state['time'] + (i + 1) * dt

            # Store prediction
            predictions.append(state.copy())

        return predictions

    # Define example initial state
    current_state = {
        'time': 0.0,
        'senescence_fraction': 0.05,
        'hole_count': 0.0,
        'hole_area_fraction': 0.0,
        'mean_alignment_error': 1.0,
        'responses': np.array([0.5, 0.7, 0.3, 0.8])
    }

    # Define control sequence (shear stress values)
    control_sequence = [0.5, 1.2, 2.0, 1.5, 1.0, 0.8, 0.3]

    # Get predictions
    predictions = mock_predict_future_state(current_state, control_sequence)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Extract time points
    times = [current_state['time']] + [p['time'] for p in predictions]
    control_times = times[1:]

    # 1. State evolution plots
    # Senescence
    senescence_vals = [current_state['senescence_fraction']] + [p['senescence_fraction'] for p in predictions]
    axes[0, 0].plot(times, senescence_vals, 'r-o', linewidth=2)
    axes[0, 0].axhline(y=0.3, color='r', linestyle='--', alpha=0.7, label='Threshold')
    axes[0, 0].set_ylabel('Senescence Fraction')
    axes[0, 0].set_title('Senescence Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Holes
    hole_areas = [current_state['hole_area_fraction']] + [p['hole_area_fraction'] for p in predictions]
    axes[0, 1].plot(times, hole_areas, 'b-o', linewidth=2)
    axes[0, 1].axhline(y=0.05, color='b', linestyle='--', alpha=0.7, label='Threshold (5%)')
    axes[0, 1].set_ylabel('Hole Area Fraction')
    axes[0, 1].set_title('Hole Formation')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Alignment
    alignment_errors = [current_state['mean_alignment_error']] + [p['mean_alignment_error'] for p in predictions]
    axes[0, 2].plot(times, alignment_errors, 'g-o', linewidth=2)
    axes[0, 2].set_ylabel('Alignment Error')
    axes[0, 2].set_title('Flow Alignment')
    axes[0, 2].grid(True, alpha=0.3)

    # Control input
    axes[1, 0].step(control_times, control_sequence, 'k-', linewidth=3, where='post')
    axes[1, 0].fill_between(control_times, control_sequence, alpha=0.3, step='post')
    axes[1, 0].set_ylabel('Shear Stress (Pa)')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_title('Control Input')
    axes[1, 0].grid(True, alpha=0.3)

    # Phase portrait (Senescence vs Hole Area)
    axes[1, 1].plot(senescence_vals, hole_areas, 'mo-', linewidth=2, markersize=6)
    axes[1, 1].plot(senescence_vals[0], hole_areas[0], 'go', markersize=12, label='Start')
    axes[1, 1].plot(senescence_vals[-1], hole_areas[-1], 'ro', markersize=12, label='End')
    # Add constraint boundaries
    axes[1, 1].axvline(x=0.3, color='r', linestyle='--', alpha=0.5, label='Sen. Threshold')
    axes[1, 1].axhline(y=0.05, color='b', linestyle='--', alpha=0.5, label='Hole Threshold')
    axes[1, 1].set_xlabel('Senescence Fraction')
    axes[1, 1].set_ylabel('Hole Area Fraction')
    axes[1, 1].set_title('Phase Portrait')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Response dynamics
    if len(current_state['responses']) > 0:
        response_means = [np.mean(current_state['responses'])] + [np.mean(p['responses']) for p in predictions]
        axes[1, 2].plot(times, response_means, 'm-o', linewidth=2)
        axes[1, 2].axhline(y=2.0, color='m', linestyle='--', alpha=0.7, label='Target (2.0)')
        axes[1, 2].set_ylabel('Mean Cell Response')
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_title('Response Dynamics')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()

    plt.tight_layout()
    plt.suptitle('MPC Dynamics Visualization Demo', fontsize=16, y=1.02)

    # Save the figure instead of showing (PyCharm compatibility)
    output_dir = "mpc_dynamics_plots"
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "mpc_dynamics_evolution.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"📊 Main dynamics plot saved to: {fig_path}")
    plt.close()  # Close to free memory

    # Additional: Show dynamics vector field
    fig2 = plot_dynamics_vector_field(current_state, control_sequence[0])

    return fig


def plot_dynamics_vector_field(current_state, control_input):
    """Plot vector field showing dynamics direction in state space."""

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create grid of states
    sen_range = np.linspace(0, 0.8, 15)
    hole_range = np.linspace(0, 0.15, 15)
    Sen, Hole = np.meshgrid(sen_range, hole_range)

    # Calculate dynamics at each point (simplified model)
    dSen = np.zeros_like(Sen)
    dHole = np.zeros_like(Hole)

    dt = 0.1
    for i in range(Sen.shape[0]):
        for j in range(Sen.shape[1]):
            # Senescence rate (increases with shear stress)
            senescence_rate = 0.01 * (1 + 0.1 * control_input)
            dSen[i, j] = senescence_rate

            # Hole formation rate (depends on senescence)
            if Sen[i, j] > 0.3:  # threshold
                hole_creation_prob = 0.05 * (Sen[i, j] - 0.3)
                dHole[i, j] = hole_creation_prob * 0.01  # convert to area fraction
            else:
                dHole[i, j] = 0

    # Plot vector field
    skip = 2  # Skip every 2nd arrow for clarity
    ax.quiver(Sen[::skip, ::skip], Hole[::skip, ::skip],
              dSen[::skip, ::skip], dHole[::skip, ::skip],
              alpha=0.7, scale=10, width=0.003, color='blue')

    # Mark current state
    ax.plot(current_state['senescence_fraction'],
            current_state['hole_area_fraction'],
            'ro', markersize=15, label='Current State', zorder=5)

    # Mark constraints
    ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Senescence Threshold')
    ax.axhline(y=0.05, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='Hole Area Threshold')

    # Highlight danger zone
    danger_x = np.linspace(0.3, 0.8, 100)
    danger_y = np.linspace(0.05, 0.15, 100)
    X_danger, Y_danger = np.meshgrid(danger_x, danger_y)
    ax.contourf(X_danger, Y_danger, np.ones_like(X_danger), levels=[0, 1], colors=['red'], alpha=0.2)
    ax.text(0.5, 0.1, 'DANGER ZONE\n(High Senescence\n+ High Hole Area)',
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))

    ax.set_xlabel('Senescence Fraction', fontsize=12)
    ax.set_ylabel('Hole Area Fraction', fontsize=12)
    ax.set_title(f'Dynamics Vector Field (Control Input: {control_input:.1f} Pa)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Arrows show direction\nof state evolution',
                xy=(0.1, 0.12), xytext=(0.2, 0.13),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.tight_layout()

    # Save the figure instead of showing (PyCharm compatibility)
    output_dir = "mpc_dynamics_plots"
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "dynamics_vector_field.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"🎯 Vector field plot saved to: {fig_path}")
    plt.close()  # Close to free memory

    return fig


# Run the demo
if __name__ == "__main__":
    print("🚀 Running MPC Dynamics Visualization Demo...")
    fig1 = simulate_mpc_dynamics_demo()
    print("✅ Demo completed!")
    print("\n📊 The visualization shows:")
    print("   • State evolution over prediction horizon")
    print("   • Control input sequence")
    print("   • Phase portrait showing state trajectory")
    print("   • Vector field showing dynamics direction")
    print("   • Constraint boundaries and danger zones")
    print(f"\n💾 All plots saved to 'mpc_dynamics_plots/' folder")
    print("   Open the PNG files to view the visualizations!")