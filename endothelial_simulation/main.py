"""
Complete main module for running endothelial cell mechanotransduction simulations
using the event-driven configuration system.
"""
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

# Import your simulation components
from endothelial_simulation.config import (
    SimulationConfig,
    create_temporal_only_config,
    create_spatial_only_config,
    create_full_config
)
from endothelial_simulation.core.simulator import Simulator
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


def create_population_only_config():
    """Create a configuration focused only on population dynamics."""
    config = SimulationConfig()
    config.enable_population_dynamics = True
    config.enable_spatial_properties = False
    config.enable_temporal_dynamics = False
    config.plot_directory = "results_population"
    return config


def run_single_step_simulation(config, initial_value, final_value, step_time, duration=None):
    """Run a simulation with a single step input using event-driven system."""
    print(f"🚀 Setting up single-step simulation...")

    simulator = Simulator(config)

    # Use multi-configuration initialization for best starting configuration
    config_results = simulator.initialize_with_multiple_configurations(
        cell_count=config.initial_cell_count,
        num_configurations=getattr(config, 'multi_config_count', 10),
        optimization_iterations=getattr(config, 'multi_config_optimization_steps', 3)
    )

    # Store config results for plotting
    simulator._config_results = config_results

    # Set step input pattern
    simulator.set_step_input(initial_value, final_value, step_time)

    print(f"📊 Running single-step simulation:")
    print(f"   Initial pressure: {initial_value} Pa")
    print(f"   Final pressure: {final_value} Pa")
    print(f"   Step time: {step_time} minutes")
    print(f"   Selected config energy: {config_results['best_config']['energy']:.4f}")

    # Run simulation
    results = simulator.run(duration)

    # Extract best configuration parameters
    print("\n" + "=" * 50)
    print("BEST CONFIGURATION PARAMETERS")
    print("=" * 50)

    try:
        best_params = simulator.get_best_config_parameters(save_excel=True)
        if best_params:
            area = best_params['averages']['area']
            aspect_ratio = best_params['averages']['aspect_ratio']
            orientation = best_params['averages']['orientation_degrees']

            print(f"📊 Key metrics:")
            print(f"   Average area: {area:.1f} pixels²")
            print(f"   Average aspect ratio: {aspect_ratio:.2f}")
            print(f"   Average orientation: {orientation:.1f}°")
    except Exception as e:
        print(f"⚠️  Could not extract parameters: {e}")

    print("=" * 50)
    return simulator


def run_multi_step_simulation(config, schedule, duration=None):
    """Run a simulation with multi-step input using event-driven system."""
    print(f"🚀 Setting up multi-step simulation...")

    simulator = Simulator(config)

    # Use smart initialization (chooses multi-config for larger simulations)
    config_results = simulator.initialize_smart(
        cell_count=config.initial_cell_count,
        force_multi_config=True  # Always use multi-config for multi-step
    )

    # Store config results if multi-config was used
    if config_results:
        simulator._config_results = config_results

    # Set multi-step input pattern
    simulator.set_multi_step_input(schedule)

    print(f"📊 Running multi-step simulation with {len(schedule)} steps:")
    for time_point, value in schedule:
        print(f"   {time_point:6.1f} min ({time_point/60:5.2f}h): {value:5.2f} Pa")

    # Run simulation
    results = simulator.run(duration)
    return simulator


def run_protocol_simulation(config, protocol_name, duration=None, **protocol_kwargs):
    """Run a simulation with a predefined protocol using event-driven system."""
    print(f"🚀 Setting up protocol simulation: {protocol_name}")

    simulator = Simulator(config)

    # Use smart initialization
    config_results = simulator.initialize_smart(
        cell_count=config.initial_cell_count,
        force_multi_config=True  # Protocols benefit from multi-config
    )

    # Store config results if multi-config was used
    if config_results:
        simulator._config_results = config_results

    # Set protocol input pattern
    simulator.set_protocol_input(protocol_name, **protocol_kwargs)

    print(f"📊 Running predefined protocol: {protocol_name}")
    if protocol_kwargs:
        print(f"   Protocol parameters: {protocol_kwargs}")

    # Run simulation
    results = simulator.run(duration)
    return simulator


def run_constant_simulation(config, constant_value, duration=None):
    """Run a simulation with constant input."""
    print(f"🚀 Setting up constant simulation...")

    simulator = Simulator(config)

    # Use standard initialization for constant simulations
    simulator.initialize(cell_count=config.initial_cell_count)

    # Set constant input
    simulator.input_pattern = {
        'type': 'constant',
        'value': constant_value,
        'params': {'value': constant_value}
    }

    print(f"📊 Running constant simulation:")
    print(f"   Constant pressure: {constant_value} Pa")

    # Run simulation
    results = simulator.run(duration)
    return simulator


def main():
    """Main function with comprehensive argument parsing for event-driven simulations."""
    parser = argparse.ArgumentParser(
        description='Endothelial Cell Mechanotransduction Simulation (Event-Driven)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single step input (default)
  python main.py --initial-value 0.0 --final-value 1.4 --step-time 60
  
  # Constant input
  python main.py --constant-value 1.4
  
  # Multi-step input with custom schedule
  python main.py --multi-step --schedule "0,0.0;60,1.4;180,0.5;300,0.0"
  
  # Predefined protocols
  python main.py --protocol acute_stress
  python main.py --protocol chronic_stress --scale-time 1.5 --scale-stress 1.2
  
  # Different simulation modes
  python main.py --mode temporal --protocol stepwise_increase
  python main.py --mode spatial --multi-step --schedule "0,0;30,1.0;90,0;150,1.5"
  
  # Event-driven system configuration
  python main.py --pressure-threshold 0.05 --debug-events
  python main.py --min-reconfig-interval 20 --debug-transitions
        """
    )

    # === SIMULATION MODE ===
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'temporal', 'spatial', 'population', 'minimal'],
                        help='Simulation mode (default: full)')

    # === BASIC PARAMETERS ===
    parser.add_argument('--duration', type=float, default=360,
                        help='Simulation duration in minutes (default: 360 = 6 hours)')

    parser.add_argument('--cells', type=int, default=50,
                        help='Initial cell count (default: 50)')

    # === INPUT TYPE SELECTION (mutually exclusive) ===
    input_group = parser.add_mutually_exclusive_group()

    input_group.add_argument('--single-step', action='store_true', default=True,
                            help='Use single step input (default)')

    input_group.add_argument('--constant-value', type=float,
                            help='Use constant input with specified value (Pa)')

    input_group.add_argument('--multi-step', action='store_true',
                            help='Use multi-step input with custom schedule')

    input_group.add_argument('--protocol', type=str,
                            choices=['acute_stress', 'chronic_stress', 'stepwise_increase',
                                   'stress_recovery', 'oscillatory_low', 'high_stress_brief'],
                            help='Use predefined protocol')

    # === SINGLE-STEP INPUT PARAMETERS ===
    parser.add_argument('--initial-value', type=float, default=0.0,
                        help='Initial shear stress value in Pa (default: 0.0)')

    parser.add_argument('--final-value', type=float, default=1.4,
                        help='Final shear stress value in Pa (default: 1.4)')

    parser.add_argument('--step-time', type=float, default=60,
                        help='Time for step change in minutes (default: 60)')

    # === MULTI-STEP INPUT PARAMETERS ===
    parser.add_argument('--schedule', type=str,
                        help='Multi-step schedule as "time1,value1;time2,value2;..." (times in minutes, values in Pa)')

    # === PROTOCOL SCALING PARAMETERS ===
    parser.add_argument('--scale-time', type=float, default=1.0,
                        help='Time scaling factor for protocols (default: 1.0)')

    parser.add_argument('--scale-stress', type=float, default=1.0,
                        help='Stress scaling factor for protocols (default: 1.0)')

    parser.add_argument('--max-stress', type=float,
                        help='Maximum stress limit for protocols (Pa)')

    # === EVENT-DRIVEN SYSTEM CONFIGURATION ===
    parser.add_argument('--pressure-threshold', type=float, default=0.1,
                        help='Pressure change threshold for events (Pa, default: 0.1)')

    parser.add_argument('--min-reconfig-interval', type=float, default=30.0,
                        help='Minimum interval between reconfigurations (minutes, default: 30)')

    parser.add_argument('--max-compression', type=float, default=0.7,
                        help='Maximum compression ratio during transitions (default: 0.7)')

    # === DEBUG OPTIONS ===
    parser.add_argument('--debug-events', action='store_true',
                        help='Enable event debugging output')

    parser.add_argument('--debug-transitions', action='store_true',
                        help='Enable transition debugging output')

    # === VISUALIZATION OPTIONS ===
    parser.add_argument('--plot', action='store_true',
                        help='Show plots after simulation')

    parser.add_argument('--no-save', action='store_true',
                        help='Do not save plots to files')

    parser.add_argument('--create-animations', action='store_true',
                        help='Create animation files (requires ffmpeg)')

    # === MULTI-CONFIGURATION OPTIONS ===
    parser.add_argument('--num-configs', type=int, default=10,
                        help='Number of configurations to test (default: 10)')

    parser.add_argument('--optimization-iterations', type=int, default=3,
                        help='Optimization iterations per configuration (default: 3)')

    # Parse arguments
    args = parser.parse_args()

    # === CREATE CONFIGURATION BASED ON MODE ===
    if args.mode == 'full':
        config = create_full_config()
    elif args.mode == 'temporal':
        config = create_temporal_only_config()
    elif args.mode == 'spatial':
        config = create_spatial_only_config()
    elif args.mode == 'population':
        config = create_population_only_config()
    elif args.mode == 'minimal':
        config = SimulationConfig()
        # Keep only basic functionality
    else:
        config = create_full_config()

    # === APPLY COMMAND-LINE OVERRIDES ===
    config.simulation_duration = args.duration
    config.initial_cell_count = args.cells
    config.save_plots = not args.no_save
    config.create_animations = args.create_animations

    # Event-driven system configuration
    config.use_event_driven_system = True
    config.pressure_change_threshold = args.pressure_threshold
    config.min_reconfiguration_interval = args.min_reconfig_interval
    config.max_compression_ratio = args.max_compression
    config.debug_events = args.debug_events
    config.debug_transitions = args.debug_transitions

    # Multi-configuration parameters
    config.multi_config_count = args.num_configs
    config.multi_config_optimization_steps = args.optimization_iterations

    # === PRINT CONFIGURATION ===
    print("=" * 70)
    print("🧬 ENDOTHELIAL CELL MECHANOTRANSDUCTION SIMULATION")
    print("🔄 Event-Driven Configuration System")
    print("=" * 70)
    print(config.describe())
    print(f"\n🔧 Event-Driven Parameters:")
    print(f"   Pressure threshold: {config.pressure_change_threshold} Pa")
    print(f"   Min reconfig interval: {config.min_reconfiguration_interval} min")
    print(f"   Max compression: {config.max_compression_ratio}")
    print(f"   Debug events: {config.debug_events}")
    print(f"   Debug transitions: {config.debug_transitions}")
    print()

    # === DETERMINE INPUT TYPE AND RUN SIMULATION ===
    simulator = None

    try:
        if args.constant_value is not None:
            # Constant input
            print("INPUT TYPE: Constant")
            print("-" * 40)
            simulator = run_constant_simulation(config, args.constant_value, args.duration)

        elif args.protocol:
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
                for time_point, value in default_schedule:
                    print(f"   {time_point:6.1f} min: {value:5.2f} Pa")
                schedule = default_schedule
            else:
                try:
                    schedule = parse_schedule_string(args.schedule)
                except ValueError as e:
                    print(f"❌ Error parsing schedule: {e}")
                    return

            simulator = run_multi_step_simulation(config, schedule, args.duration)

        else:
            # Single-step input (default)
            print("INPUT TYPE: Single Step")
            print("-" * 40)

            simulator = run_single_step_simulation(
                config, args.initial_value, args.final_value, args.step_time, args.duration
            )

    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # === SIMULATION COMPLETED ===
    print("\n" + "=" * 70)
    print("✅ SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    # === SAVE RESULTS ===
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = simulator.save_results(f"simulation_{timestamp}")
        print(f"💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")

    # === CREATE VISUALIZATIONS ===
    try:
        print("\n📊 Generating visualizations...")
        plotter = Plotter(config)

        # Create comprehensive plots
        figures = plotter.create_all_plots(simulator, prefix=f"comprehensive_{timestamp}")
        print(f"   Created {len(figures)} comprehensive plots")

        # Create all plots (includes configuration analysis if available)
        if hasattr(plotter, 'create_all_plots'):
            all_figures = plotter.create_all_plots(simulator, prefix=f"sim_{timestamp}")
            print(f"   Created {len(all_figures)} total plots in: {config.plot_directory}")

        # Create animations if requested and data is available
        if args.create_animations and hasattr(simulator, 'frame_data') and simulator.frame_data:
            print("🎬 Creating animations...")
            simulator._create_animations()

    except Exception as e:
        print(f"⚠️  Visualization creation failed: {e}")

    # === DISPLAY FINAL STATISTICS ===
    try:
        # Use safe statistics method instead of get_grid_statistics
        if hasattr(simulator, 'get_safe_final_statistics'):
            final_stats = simulator.get_safe_final_statistics()
        else:
            # Fallback to basic stats if method doesn't exist
            final_stats = {
                'total_cells': len(simulator.grid.cells),
                'healthy_cells': 0,
                'senescent_cells': 0,
                'biological_energy': 0.0,
                'packing_efficiency': 0.0,
                'reconfigurations_count': 0,
                'events_count': 0
            }

        print(f"\n📈 Final Statistics:")
        print(f"   Total cells: {final_stats.get('total_cells', 0):,}")
        print(f"   Healthy cells: {final_stats.get('healthy_cells', 0):,}")
        print(f"   Senescent cells: {final_stats.get('senescent_cells', 0):,}")
        print(f"   Packing efficiency: {final_stats.get('packing_efficiency', 0):.2f}")
        print(f"   Biological energy: {final_stats.get('biological_energy', 0):.4f}")
        print(f"   Reconfigurations triggered: {final_stats.get('reconfigurations_count', 0)}")
        print(f"   Events detected: {final_stats.get('events_count', 0)}")

    except Exception as e:
        print(f"⚠️  Could not display final statistics: {e}")
        # Minimal fallback
        try:
            total_cells = len(simulator.grid.cells)
            print(f"\n📈 Basic Statistics:")
            print(f"   Total cells: {total_cells:,}")
        except:
            print(f"\n⚠️  No statistics available")


    # === SHOW PLOTS IF REQUESTED ===
    if args.plot:
        print("\n🖼️  Displaying plots...")
        plt.show()

    # === DISPLAY SUMMARY ===
    print(f"\n🎉 Simulation completed successfully!")
    print(f"📁 Results directory: {config.plot_directory}")
    print(f"⏱️  Final simulation time: {simulator.time:.1f} minutes ({simulator.time/60:.1f} hours)")
    print(f"🔄 Event-driven system: {config.use_event_driven_system}")

    return simulator


if __name__ == "__main__":
    main()