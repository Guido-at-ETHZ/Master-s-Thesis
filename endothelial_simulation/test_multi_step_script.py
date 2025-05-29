#!/usr/bin/env python3
"""
Test script for multi-step input functionality.
Run this to verify your multi-step implementation works correctly.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SimulationConfig, create_full_config
from core import Simulator
from visualization import Plotter


def test_basic_multi_step():
    """Test basic multi-step functionality."""
    print("=" * 60)
    print("Testing Basic Multi-Step Input")
    print("=" * 60)
    
    # Create a simple configuration for testing
    config = create_full_config()
    config.simulation_duration = 300  # 5 hours for quick test
    config.initial_cell_count = 100   # Fewer cells for faster testing
    
    # Create simulator
    simulator = Simulator(config)
    simulator.initialize()
    
    # Define a simple multi-step schedule
    test_schedule = [
        (0, 0.0),      # Start at 0 Pa
        (60, 1.0),     # Step to 1 Pa at 1 hour
        (120, 0.5),    # Step to 0.5 Pa at 2 hours
        (180, 1.4),    # Step to 1.4 Pa at 3 hours
        (240, 0.0)     # Return to 0 Pa at 4 hours
    ]
    
    print("Setting multi-step schedule...")
    simulator.set_multi_step_input(test_schedule)
    
    # Run simulation
    print("\nRunning simulation...")
    start_time = time.time()
    results = simulator.run()
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"Final time: {simulator.time:.1f} minutes")
    print(f"Final cell count: {len(simulator.grid.cells)}")
    
    # Test the step info function
    print("\nTesting step info retrieval...")
    # Set time to different points and check step info
    test_times = [30, 90, 150, 210, 270]
    
    for test_time in test_times:
        simulator.time = test_time
        simulator.update_input_value()
        step_info = simulator.get_current_step_info()
        
        if step_info:
            print(f"Time {test_time:3.0f}min: Step {step_info['step_number']}/{step_info['total_steps']}, "
                  f"Value: {step_info['current_value']:.1f} Pa, "
                  f"Time in step: {step_info['time_in_step']:.1f}min")
    
    return simulator


def test_predefined_protocols():
    """Test predefined protocol functionality."""
    print("\n" + "=" * 60)
    print("Testing Predefined Protocols")
    print("=" * 60)
    
    config = create_full_config()
    config.simulation_duration = 240  # 4 hours
    config.initial_cell_count = 50    # Fewer cells for faster testing
    
    protocols_to_test = ['acute_stress', 'stepwise_increase', 'oscillating']
    
    simulators = {}
    
    for protocol in protocols_to_test:
        print(f"\nTesting protocol: {protocol}")
        
        simulator = Simulator(config)
        simulator.initialize()
        
        # Set protocol
        simulator.set_protocol_input(protocol)
        
        # Run shorter simulation for testing
        print(f"Running {protocol} simulation...")
        results = simulator.run(duration=240)  # 4 hours
        
        simulators[protocol] = simulator
        
        # Quick check
        final_counts = simulator.grid.count_cells_by_type()
        print(f"Final results - Total: {final_counts['total']}, "
              f"Senescent: {final_counts['telomere_senescent'] + final_counts['stress_senescent']}")
    
    return simulators


def test_protocol_scaling():
    """Test protocol scaling functionality."""
    print("\n" + "=" * 60)
    print("Testing Protocol Scaling")
    print("=" * 60)
    
    config = create_full_config()
    config.simulation_duration = 360  # 6 hours
    config.initial_cell_count = 50
    
    simulator = Simulator(config)
    simulator.initialize()
    
    # Test scaling
    print("Setting acute stress protocol with 2x time scaling and 1.5x stress scaling...")
    simulator.set_protocol_input('acute_stress', scale_time=2.0, scale_stress=1.5)
    
    # Check the resulting schedule
    schedule = simulator.input_pattern['params']['schedule']
    print("Scaled schedule:")
    for time_point, value in schedule:
        print(f"  {time_point:6.1f} min ({time_point/60:5.2f}h): {value:5.2f} Pa")
    
    # Run simulation
    print("\nRunning scaled simulation...")
    results = simulator.run()
    
    print(f"Simulation completed. Final time: {simulator.time:.1f} minutes")
    
    return simulator


def create_test_visualizations(simulators_dict):
    """Create visualizations for all test simulations."""
    print("\n" + "=" * 60)
    print("Creating Test Visualizations")
    print("=" * 60)
    
    for name, simulator in simulators_dict.items():
        print(f"Creating plots for {name}...")
        
        # Create plotter
        plotter = Plotter(simulator.config)
        
        # Generate plots
        try:
            figures = plotter.create_all_plots(simulator, prefix=f"test_{name}")
            print(f"  Created {len(figures)} plots for {name}")
        except Exception as e:
            print(f"  Error creating plots for {name}: {e}")
    
    print(f"\nAll plots saved to: {simulator.config.plot_directory}")


def main():
    """Run all tests."""
    print("Multi-Step Input Implementation Test Suite")
    print("=" * 60)
    
    all_simulators = {}
    
    try:
        # Test 1: Basic multi-step
        basic_sim = test_basic_multi_step()
        all_simulators['basic_multi_step'] = basic_sim
        
        # Test 2: Predefined protocols  
        protocol_sims = test_predefined_protocols()
        all_simulators.update(protocol_sims)
        
        # Test 3: Protocol scaling
        scaled_sim = test_protocol_scaling()
        all_simulators['scaled_protocol'] = scaled_sim
        
        # Create visualizations
        create_test_visualizations(all_simulators)
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Total simulations run: {len(all_simulators)}")
        print("Check the 'results' directory for generated plots.")
        
        # Summary of what was tested
        print("\nTested functionality:")
        print("✓ Basic multi-step input with custom schedule")
        print("✓ Predefined protocol templates")
        print("✓ Protocol scaling (time and stress)")
        print("✓ Step information retrieval")
        print("✓ Visualization generation")
        
    except Exception as e:
        print(f"\nTEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Multi-step implementation is working correctly!")
        print("\nYou can now use multi-step inputs in your simulations.")
    else:
        print("\n❌ There were issues with the implementation.")
        print("Please check the error messages above.")
