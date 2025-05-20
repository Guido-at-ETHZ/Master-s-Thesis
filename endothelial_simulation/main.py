"""
Main module for running endothelial cell mechanotransduction simulations.
"""
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from config import SimulationConfig, create_temporal_only_config, create_spatial_only_config, create_full_config
from core import Simulator
from visualization import Plotter


def run_simulation(config, input_type, input_params, duration=None):
    """
    Run a simulation with the specified configuration and input pattern.

    Parameters:
        config: SimulationConfig object
        input_type: Type of input pattern ('constant', 'step', 'ramp', 'oscillatory')
        input_params: Dictionary of input pattern parameters
        duration: Simulation duration (default: from config)

    Returns:
        Simulator object with results
    """
    # Create simulator
    simulator = Simulator(config)

    # Initialize with cells
    simulator.initialize()

    # Set input pattern
    if input_type == 'constant':
        simulator.set_constant_input(input_params['value'])
    elif input_type == 'step':
        simulator.set_step_input(
            input_params['initial_value'],
            input_params['final_value'],
            input_params['step_time']
        )
    elif input_type == 'ramp':
        simulator.set_ramp_input(
            input_params['initial_value'],
            input_params['final_value'],
            input_params['ramp_start_time'],
            input_params['ramp_end_time']
        )
    elif input_type == 'oscillatory':
        simulator.set_oscillatory_input(
            input_params['base_value'],
            input_params['amplitude'],
            input_params['frequency'],
            input_params.get('phase', 0)
        )
    else:
        raise ValueError(f"Unknown input type: {input_type}")

    # Run simulation
    results = simulator.run(duration)

    # Save results
    simulator.save_results()

    return simulator


def main():
    """
    Main function for running the simulation from command line.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Endothelial Cell Mechanotransduction Simulation')

    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'temporal', 'spatial', 'minimal'],
                        help='Simulation mode (full, temporal, spatial, minimal)')

    parser.add_argument('--duration', type=float, default=None,
                        help='Simulation duration (in minutes)')

    parser.add_argument('--input-type', type=str, default='constant',
                        choices=['constant', 'step', 'ramp', 'oscillatory'],
                        help='Input pattern type')

    parser.add_argument('--value', type=float, default=1.4,
                        help='Constant input value or base value')

    parser.add_argument('--initial-value', type=float, default=0.0,
                        help='Initial input value for step/ramp')

    parser.add_argument('--final-value', type=float, default=1.4,
                        help='Final input value for step/ramp')

    parser.add_argument('--step-time', type=float, default=60,
                        help='Time for step change (in minutes)')

    parser.add_argument('--ramp-start', type=float, default=60,
                        help='Start time for ramp (in minutes)')

    parser.add_argument('--ramp-end', type=float, default=180,
                        help='End time for ramp (in minutes)')

    parser.add_argument('--amplitude', type=float, default=0.5,
                        help='Oscillation amplitude')

    parser.add_argument('--frequency', type=float, default=0.01,
                        help='Oscillation frequency (Hz)')

    parser.add_argument('--phase', type=float, default=0.0,
                        help='Oscillation phase (radians)')

    parser.add_argument('--cells', type=int, default=None,
                        help='Initial cell count')

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
    if args.duration is not None:
        config.simulation_duration = args.duration

    if args.cells is not None:
        config.initial_cell_count = args.cells

    # Print configuration
    print(config.describe())

    # Prepare input parameters based on input type
    input_params = {}

    if args.input_type == 'constant':
        input_params = {
            'value': args.value
        }
    elif args.input_type == 'step':
        input_params = {
            'initial_value': args.initial_value,
            'final_value': args.final_value,
            'step_time': args.step_time
        }
    elif args.input_type == 'ramp':
        input_params = {
            'initial_value': args.initial_value,
            'final_value': args.final_value,
            'ramp_start_time': args.ramp_start,
            'ramp_end_time': args.ramp_end
        }
    elif args.input_type == 'oscillatory':
        input_params = {
            'base_value': args.value,
            'amplitude': args.amplitude,
            'frequency': args.frequency,
            'phase': args.phase
        }

    # Run simulation
    print(f"Running simulation with {args.input_type} input pattern...")
    simulator = run_simulation(config, args.input_type, input_params)

    # Create plots
    plotter = Plotter(config)
    figures = plotter.create_all_plots(simulator)

    # Show plots if requested
    if args.plot:
        plt.show()

    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()