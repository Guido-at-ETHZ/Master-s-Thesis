"""
Main module for running endothelial cell mechanotransduction simulations with step input.
"""
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from config import SimulationConfig, create_temporal_only_config, create_spatial_only_config, create_full_config
from core import Simulator
from visualization import Plotter


def run_simulation(config, initial_value, final_value, step_time, duration=None):
    """
    Run a simulation with the specified configuration and step input pattern.

    Parameters:
        config: SimulationConfig object
        initial_value: Initial shear stress value (Pa)
        final_value: Final shear stress value after step (Pa)
        step_time: Time at which the step occurs (minutes)
        duration: Simulation duration (default: from config)

    Returns:
        Simulator object with results
    """
    # Create simulator
    simulator = Simulator(config)

    # Initialize with cells
    simulator.initialize()

    # Set step input pattern
    simulator.set_step_input(initial_value, final_value, step_time)

    # Run simulation
    results = simulator.run(duration)

    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    simulator.save_results(f"step_input_{timestamp}.npz")

    return simulator


def main():
    """
    Main function for running step input simulations from command line.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Endothelial Cell Mechanotransduction Simulation with Step Input')

    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'temporal', 'spatial', 'minimal'],
                        help='Simulation mode (full, temporal, spatial, minimal)')

    parser.add_argument('--duration', type=float, default=1080,
                        help='Simulation duration in minutes (default: 1080 = 18 hours)')

    parser.add_argument('--initial-value', type=float, default=0.0,
                        help='Initial shear stress value in Pa (default: 0.0)')

    parser.add_argument('--final-value', type=float, default=1.4,
                        help='Final shear stress value after step in Pa (default: 1.4)')

    parser.add_argument('--step-time', type=float, default=60,
                        help='Time at which step change occurs in minutes (default: 60)')

    parser.add_argument('--cells', type=int, default=200,
                        help='Initial cell count (default: 200)')

    parser.add_argument('--plot', action='store_true',
                        help='Show plots after simulation')

    args = parser.parse_args()

    # Create configuration based on mode
    if args.mode == 'full':
        config = create_full_config()
    elif args.mode == 'temporal':
        config = create_temporal_only_config()
    elif args.mode == 'spatial':
        config = create_spatial_only_config()
    elif args.mode == 'minimal':
        config = SimulationConfig()
        config.enable_minimal_population()

    # Override with command-line arguments
    config.simulation_duration = args.duration
    config.initial_cell_count = args.cells

    # Print configuration and planned input pattern
    print(config.describe())
    print("\nStep Input Configuration:")
    print(f"  Initial shear stress: {args.initial_value} Pa")
    print(f"  Final shear stress: {args.final_value} Pa")
    print(f"  Step occurs at: {args.step_time} minutes ({args.step_time/60:.1f} hours)")

    # Run simulation
    print("\nRunning simulation with step input pattern...")
    start_time = time.time()
    simulator = run_simulation(
        config,
        args.initial_value,
        args.final_value,
        args.step_time,
        args.duration
    )
    end_time = time.time()

    print(f"Simulation completed in {end_time - start_time:.2f} seconds")

    # Create plots
    print("Generating visualization plots...")
    plotter = Plotter(config)
    figures = plotter.create_all_plots(simulator)

    # Show plots if requested
    if args.plot:
        plt.show()

    print("Step input simulation completed successfully.")
    print(f"Results saved to: {config.plot_directory}")


if __name__ == "__main__":
    main()