"""
Examples of how to use multi-step inputs in the endothelial simulation.
This script demonstrates different ways to set up time-varying shear stress patterns.
"""
import numpy as np
import matplotlib.pyplot as plt

# FIXED IMPORTS - use absolute package imports
from endothelial_simulation.config import SimulationConfig, create_full_config
from endothelial_simulation.core import Simulator
from endothelial_simulation.visualization import Plotter


def example_1_custom_multi_step():
    """Example 1: Custom multi-step schedule."""
    print("=" * 60)
    print("Example 1: Custom Multi-Step Schedule")
    print("=" * 60)
    
    # Create configuration
    config = create_full_config()
    config.simulation_duration = 360  # 6 hours
    config.initial_cell_count = 100
    
    # Create simulator
    simulator = Simulator(config)
    simulator.initialize()
    
    # Define custom schedule: (time_in_minutes, shear_stress_in_Pa)
    custom_schedule = [
        (0, 0.0),      # Start with no stress
        (60, 1.0),     # Increase to 1 Pa at 1 hour
        (120, 1.4),    # Peak stress at 2 hours
        (180, 0.7),    # Reduce stress at 3 hours
        (240, 1.2),    # Increase again at 4 hours
        (300, 0.0)     # Return to baseline at 5 hours
    ]
    
    # Set the multi-step input
    simulator.set_multi_step_input(custom_schedule)
    
    # Run simulation
    print("Running custom multi-step simulation...")
    results = simulator.run()
    
    # Create plots
    plotter = Plotter(config)
    figures = plotter.create_all_plots(simulator, prefix="example1_custom")
    
    print(f"Example 1 completed. Check {config.plot_directory} for results.")
    return simulator


def example_2_predefined_protocols():
    """Example 2: Using predefined protocols."""
    print("\n" + "=" * 60)
    print("Example 2: Predefined Protocols")
    print("=" * 60)
    
    # Test different protocols
    protocols_to_test = [
        ('acute_stress', {}),
        ('chronic_stress', {}),
        ('stepwise_increase', {}),
        ('oscillating', {}),
        ('acute_stress', {'scale_time': 1.5, 'scale_stress': 1.2})  # Scaled version
    ]
    
    simulators = {}
    
    for protocol_name, kwargs in protocols_to_test:
        print(f"\nTesting protocol: {protocol_name}")
        if kwargs:
            print(f"  Parameters: {kwargs}")
            
        # Create fresh config and simulator for each protocol
        config = create_full_config()
        config.simulation_duration = 240  # 4 hours
        config.initial_cell_count = 50
        
        simulator = Simulator(config)
        simulator.initialize()
        
        # Set protocol
        simulator.set_protocol_input(protocol_name, **kwargs)
        
        # Run simulation
        results = simulator.run()
        
        # Store result
        key = f"{protocol_name}" + (f"_scaled" if kwargs else "")
        simulators[key] = simulator
        
        # Create plots
        plotter = Plotter(config)
        figures = plotter.create_all_plots(simulator, prefix=f"example2_{key}")
        
        print(f"  Protocol {protocol_name} completed.")
    
    print(f"\nExample 2 completed. Tested {len(simulators)} protocols.")
    return simulators


def example_3_complex_pattern():
    """Example 3: Complex oscillating pattern with multiple phases."""
    print("\n" + "=" * 60)
    print("Example 3: Complex Oscillating Pattern")
    print("=" * 60)
    
    config = create_full_config()
    config.simulation_duration = 480  # 8 hours
    config.initial_cell_count = 75
    
    simulator = Simulator(config)
    simulator.initialize()
    
    # Create a complex pattern with multiple phases
    # Phase 1: Gradual increase (0-2 hours)
    # Phase 2: High stress oscillations (2-4 hours)  
    # Phase 3: Gradual decrease (4-6 hours)
    # Phase 4: Low stress maintenance (6-8 hours)
    
    complex_schedule = [
        # Phase 1: Gradual increase
        (0, 0.0),
        (30, 0.2),
        (60, 0.4),
        (90, 0.6),
        (120, 0.8),
        
        # Phase 2: High stress oscillations
        (150, 1.4),    # Peak
        (165, 0.8),    # Dip
        (180, 1.4),    # Peak
        (195, 0.8),    # Dip
        (210, 1.4),    # Peak
        (225, 0.8),    # Dip
        (240, 1.4),    # Final peak
        
        # Phase 3: Gradual decrease
        (270, 1.0),
        (300, 0.6),
        (330, 0.4),
        (360, 0.2),
        
        # Phase 4: Low stress maintenance
        (390, 0.1),
        (480, 0.1)     # End
    ]
    
    simulator.set_multi_step_input(complex_schedule)
    
    print(f"Running complex pattern with {len(complex_schedule)} time points...")
    results = simulator.run()
    
    # Create enhanced plots
    plotter = Plotter(config)
    figures = plotter.create_all_plots(simulator, prefix="example3_complex")
    
    # Create a special plot showing the input pattern
    fig, ax = plt.subplots(figsize=(12, 6))
    times = [t for t, v in complex_schedule]
    values = [v for t, v in complex_schedule]
    
    ax.plot(np.array(times)/60, values, 'ro-', linewidth=2, markersize=6)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Shear Stress (Pa)', fontsize=12)
    ax.set_title('Complex Multi-Step Input Pattern', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add phase annotations
    ax.axvspan(0, 2, alpha=0.2, color='green', label='Phase 1: Gradual increase')
    ax.axvspan(2, 4, alpha=0.2, color='red', label='Phase 2: High oscillations')  
    ax.axvspan(4, 6, alpha=0.2, color='blue', label='Phase 3: Gradual decrease')
    ax.axvspan(6, 8, alpha=0.2, color='gray', label='Phase 4: Low maintenance')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{config.plot_directory}/example3_input_pattern.png", dpi=300, bbox_inches='tight')
    
    print(f"Example 3 completed. Complex pattern simulation finished.")
    return simulator


def example_4_debug_multi_step():
    """Example 4: Debug multi-step functionality with detailed output."""
    print("\n" + "=" * 60)
    print("Example 4: Multi-Step Debug")
    print("=" * 60)
    
    config = create_full_config()
    config.simulation_duration = 180  # 3 hours
    config.initial_cell_count = 30
    config.plot_interval = 5  # Record more frequently
    
    simulator = Simulator(config)
    simulator.initialize()
    
    # Simple schedule for debugging
    debug_schedule = [
        (0, 0.0),
        (30, 0.5),
        (60, 1.0),
        (90, 1.5),
        (120, 1.0),
        (150, 0.0)
    ]
    
    print("Debug schedule:")
    for time_point, value in debug_schedule:
        print(f"  {time_point:3.0f} min ({time_point/60:4.1f}h): {value:4.1f} Pa")
    
    simulator.set_multi_step_input(debug_schedule)
    
    # Check that the schedule was set correctly
    print(f"\nSchedule in simulator: {simulator.input_pattern}")
    
    # Run simulation with step-by-step monitoring
    print(f"\nRunning debug simulation...")
    results = simulator.run()
    
    # Check some time points manually
    print(f"\nChecking input values at different times:")
    test_times = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    
    for test_time in test_times:
        # Temporarily set simulator time and update input
        simulator.time = test_time
        current_input = simulator.update_input_value()
        step_info = simulator.get_current_step_info()
        
        print(f"  t={test_time:3.0f}min: input={current_input:.2f}Pa, step={step_info['step_number'] if step_info else 'N/A'}")
    
    # Create visualization
    plotter = Plotter(config)
    figures = plotter.create_all_plots(simulator, prefix="example4_debug")
    
    print(f"Example 4 completed. Debug information shown above.")
    return simulator


def main():
    """Run all examples."""
    print("Multi-Step Input Examples for Endothelial Cell Simulation")
    print("========================================================")
    
    try:
        # Run examples
        sim1 = example_1_custom_multi_step()
        sims2 = example_2_predefined_protocols()
        sim3 = example_3_complex_pattern()
        sim4 = example_4_debug_multi_step()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("Check the 'results' directory for all generated plots.")
        print("\nExamples run:")
        print("1. Custom multi-step schedule")
        print("2. Predefined protocols (5 variations)")
        print("3. Complex oscillating pattern")
        print("4. Debug multi-step functionality")
        
        # Optionally show plots
        show_plots = input("\nShow plots? (y/N): ").lower() == 'y'
        if show_plots:
            plt.show()
            
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
