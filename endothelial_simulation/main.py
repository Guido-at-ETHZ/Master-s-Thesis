"""
Enhanced main module for running endothelial cell mechanotransduction simulations.
Now includes full multi-step input support directly in main.py.
"""
import os
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

# FIXED IMPORTS - use absolute package imports
from endothelial_simulation.config import SimulationConfig, create_temporal_only_config, create_spatial_only_config, create_full_config
from endothelial_simulation.core import Simulator
from endothelial_simulation.visualization import Plotter


def parse_schedule_string(schedule_str):
    """
    Parse a schedule string into a list of (time, value) tuples.

    Format: "time1,value1;time2,value2;time3,value3"
    Example: "0,0.0;60,1.4;180,0.5;300,0.0"

    Parameters:
        schedule_str: String representation of the schedule

    Returns:
        List of (time, value) tuples
    """
    try:
        schedule = []
        pairs = schedule_str.split(';')

        for pair in pairs:
            time_str, value_str = pair.split(',')
            time_val = float(time_str.strip())
            value_val = float(value_str.strip())
            schedule.append((time_val, value_val))

        # Sort by time to ensure proper order
        schedule.sort(key=lambda x: x[0])

        return schedule
    except Exception as e:
        raise ValueError(f"Invalid schedule format. Use 'time1,value1;time2,value2;...' format. Error: {e}")


def run_single_step_simulation(config, initial_value, final_value, step_time, duration=None):
    """Run a simulation with a single step input."""
    simulator = Simulator(config)
    simulator.initialize()
    simulator.set_step_input(initial_value, final_value, step_time)

    print(f"Running single-step simulation:")
    print(f"  Initial: {initial_value} Pa")
    print(f"  Final: {final_value} Pa")
    print(f"  Step at: {step_time} minutes")

    results = simulator.run(duration)
    return simulator


def run_multi_step_simulation(config, schedule, duration=None):
    """Run a simulation with multi-step input."""
    simulator = Simulator(config)
    simulator.initialize()
    simulator.set_multi_step_input(schedule)

    print(f"Running multi-step simulation with {len(schedule)} steps:")
    for time_point, value in schedule:
        print(f"  {time_point:6.1f} min ({time_point/60:5.2f}h): {value:5.2f} Pa")

    results = simulator.run(duration)
    return simulator


def run_protocol_simulation(config, protocol_name, duration=None, **protocol_kwargs):
    """Run a simulation with a predefined protocol."""
    simulator = Simulator(config)
    simulator.initialize()
    simulator.set_protocol_input(protocol_name, **protocol_kwargs)

    print(f"Running predefined protocol: {protocol_name}")
    if protocol_kwargs:
        print(f"  Protocol parameters: {protocol_kwargs}")

    results = simulator.run(duration)
    return simulator


def main():
    """Main function with enhanced argument parsing for all input types."""
    parser = argparse.ArgumentParser(
        description='Endothelial Cell Mechanotransduction Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single step input (default)
  python main.py --initial-value 0.0 --final-value 1.4 --step-time 60
  
  # Multi-step input with custom schedule
  python main.py --multi-step --schedule "0,0.0;60,1.4;180,0.5;300,0.0"
  
  # Predefined protocol
  python main.py --protocol acute_stress
  python main.py --protocol chronic_stress --scale-time 1.5 --scale-stress 1.2
  
  # Different simulation modes
  python main.py --mode temporal --protocol stepwise_increase
  python main.py --mode spatial --multi-step --schedule "0,0;30,1.0;90,0;150,1.5"
        """
    )

    # Simulation mode
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'temporal', 'spatial', 'minimal'],
                        help='Simulation mode (default: full)')

    parser.add_argument('--duration', type=float, default=1080,
                        help='Simulation duration in minutes (default: 1080 = 18 hours)')

    parser.add_argument('--cells', type=int, default=10,
                        help='Initial cell count (default: 200)')

    # Input type selection (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()

    input_group.add_argument('--single-step', action='store_true', default=True,
                            help='Use single step input (default)')

    input_group.add_argument('--multi-step', action='store_true',
                            help='Use multi-step input with custom schedule')

    input_group.add_argument('--protocol', type=str,
                            choices=['baseline', 'acute_stress', 'chronic_stress',
                                   'stepwise_increase', 'stress_recovery', 'oscillating'],
                            help='Use predefined protocol')

    # Single-step input parameters
    parser.add_argument('--initial-value', type=float, default=0.0,
                        help='Initial shear stress value in Pa (default: 0.0)')

    parser.add_argument('--final-value', type=float, default=1.4,
                        help='Final shear stress value in Pa (default: 1.4)')

    parser.add_argument('--step-time', type=float, default=60,
                        help='Time for step change in minutes (default: 60)')

    # Multi-step input parameters
    parser.add_argument('--schedule', type=str,
                        help='Multi-step schedule as "time1,value1;time2,value2;..." (times in minutes, values in Pa)')

    # Protocol scaling parameters
    parser.add_argument('--scale-time', type=float, default=1.0,
                        help='Time scaling factor for protocols (default: 1.0)')

    parser.add_argument('--scale-stress', type=float, default=1.0,
                        help='Stress scaling factor for protocols (default: 1.0)')

    parser.add_argument('--max-stress', type=float,
                        help='Maximum stress limit for protocols (Pa)')

    # Visualization
    parser.add_argument('--plot', action='store_true',
                        help='Show plots after simulation')

    parser.add_argument('--no-save', action='store_true',
                        help='Do not save plots to files')

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
    config.save_plots = not args.no_save

    # Print configuration
    print("=" * 70)
    print("Endothelial Cell Mechanotransduction Simulation")
    print("=" * 70)
    print(config.describe())
    print()

    # Determine input type and run appropriate simulation
    simulator = None

    if args.protocol:
        # Protocol input
        print("INPUT TYPE: Predefined Protocol")
        print("-" * 40)

        protocol_kwargs = {}
        if args.scale_time != 1.0:
            protocol_kwargs['scale_time'] = args.scale_time
        if args.scale_stress != 1.0:
            protocol_kwargs['scale_stress'] = args.scale_stress
        if args.max_stress:
            protocol_kwargs['max_stress'] = args.max_stress

        simulator = run_protocol_simulation(config, args.protocol, args.duration, **protocol_kwargs)

    elif args.multi_step:
        # Multi-step input
        print("INPUT TYPE: Multi-Step")
        print("-" * 40)

        if not args.schedule:
            # Default multi-step schedule if none provided
            default_schedule = [(0, 0.0), (60, 1.4), (180, 0.5), (300, 0.0)]
            print("No schedule provided, using default:")
            schedule = default_schedule
        else:
            try:
                schedule = parse_schedule_string(args.schedule)
            except ValueError as e:
                print(f"Error parsing schedule: {e}")
                return

        simulator = run_multi_step_simulation(config, schedule, args.duration)

    else:
        # Single-step input (default)
        print("INPUT TYPE: Single Step")
        print("-" * 40)

        simulator = run_single_step_simulation(config, args.initial_value, args.final_value,
                                             args.step_time, args.duration)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETED")
    print("=" * 70)

    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = simulator.save_results(f"simulation_{timestamp}.npz")
    print(f"Results saved to: {results_file}")

    # Create visualizations
    print("\nGenerating visualizations...")
    plotter = Plotter(config)
    figures = plotter.create_all_plots(simulator, prefix=f"sim_{timestamp}")
    print(f"Created {len(figures)} plots in: {config.plot_directory}")

    # Display final statistics
    final_stats = simulator.grid.get_grid_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total cells: {final_stats['cell_counts']['total']}")
    print(f"  Healthy: {final_stats['cell_counts']['normal']}")
    print(f"  Senescent: {final_stats['cell_counts']['telomere_senescent'] + final_stats['cell_counts']['stress_senescent']}")
    print(f"  Packing efficiency: {final_stats.get('packing_efficiency', 0):.2f}")

    # Show plots if requested
    if args.plot:
        print("\nDisplaying plots...")
        plt.show()

    print(f"\nSimulation completed successfully!")
    return simulator


if __name__ == "__main__":
    main()