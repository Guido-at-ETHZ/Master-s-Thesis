"""
Main module for running endothelial cell mechanotransduction simulations with step input.
Enhanced with optional multi-step experiment support.
"""
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys

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

    # Multi-step option
    parser.add_argument('--use-multi-step', action='store_true',
                        help='Launch multi-step experiment tool')

    args = parser.parse_args()

    # Handle multi-step delegation
    if args.use_multi_step:
        print("=" * 70)
        print("Multi-Step Experiment Tool")
        print("=" * 70)
        print("Launching multi-step experiment interface...")
        print("\nMulti-step experiments allow multiple stress changes over time.")
        print("\nExamples:")
        print("• Predefined protocols:")
        print("  python multi_step_main.py --protocol acute_stress")
        print("  python multi_step_main.py --protocol chronic_stress --duration 600")
        print("  python multi_step_main.py --protocol stepwise_increase --scale-time 2.0")
        print()
        print("• Custom schedules:")
        print("  python multi_step_main.py --schedule '0,0.0;60,1.4;180,0.5;300,0.0'")
        print("  python multi_step_main.py --schedule '0,0;30,1.0;90,0;150,1.5' --duration 240")
        print()
        print("Available protocols: baseline, acute_stress, chronic_stress,")
        print("                    stepwise_increase, stress_recovery, oscillating")
        print()
        print("Full multi-step tool help:")
        print("-" * 70)

        try:
            # Launch the multi-step tool with help
            subprocess.run([sys.executable, 'multi_step_main.py', '--help'])
        except FileNotFoundError:
            print("\nError: multi_step_main.py not found in current directory.")
            print("\nTo use multi-step experiments:")
            print("1. Create multi_step_main.py using the provided code")
            print("2. Ensure it's in the same directory as main.py")
            print("3. Run: python multi_step_main.py --help")
            print("\nFor now, continuing with single-step simulation...")
            print("Use your regular arguments (--initial-value, --final-value, etc.)")

        return

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