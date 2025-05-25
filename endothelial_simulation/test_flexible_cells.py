"""
Example script to test the flexible cell system.
"""
import numpy as np
import matplotlib.pyplot as plt
from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.core import Simulator
from endothelial_simulation.visualization.flexible_plotter import FlexibleCellPlotter


def test_flexible_cells():
    """Test the flexible cell system with adaptive sizing."""

    # Create configuration
    config = SimulationConfig()
    config.simulation_duration = 120  # 2 hours
    config.initial_cell_count = 50
    config.plot_interval = 5
    config.grid_size = (800, 600)  # Smaller grid to test crowding

    # Create simulator with flexible cells
    simulator = Simulator(config)

    # Initialize with cells
    print("Initializing cells...")
    simulator.initialize()

    # Print initial statistics
    density_stats = simulator.grid.get_density_statistics()
    print(f"Initial density statistics:")
    print(f"  Total cells: {density_stats['total_cells']}")
    print(f"  Packing density: {density_stats['packing_density']:.3f}")
    print(f"  Average crowding factor: {density_stats['average_crowding_factor']:.3f}")

    # Set constant shear stress
    simulator.set_constant_input(1.0)  # 1 Pa

    # Run simulation
    print("Running simulation...")
    results = simulator.run()

    # Print final statistics
    final_density_stats = simulator.grid.get_density_statistics()
    print(f"\nFinal density statistics:")
    print(f"  Total cells: {final_density_stats['total_cells']}")
    print(f"  Packing density: {final_density_stats['packing_density']:.3f}")
    print(f"  Average crowding factor: {final_density_stats['average_crowding_factor']:.3f}")

    # Create visualizations
    print("Creating visualizations...")
    plotter = FlexibleCellPlotter(config)
    figures = plotter.create_all_flexible_plots(simulator)

    print(f"Simulation completed! Generated {len(figures)} plots.")

    # Show plots
    plt.show()

    return simulator


def test_high_density_scenario():
    """Test high density scenario with many cells."""

    # Create configuration for high density
    config = SimulationConfig()
    config.simulation_duration = 60  # 1 hour
    config.initial_cell_count = 150  # Many cells in small space
    config.plot_interval = 5
    config.grid_size = (600, 600)  # Small grid to force crowding

    simulator = Simulator(config)

    print("Testing high density scenario...")
    simulator.initialize()

    # Check if cells were successfully placed
    density_stats = simulator.grid.get_density_statistics()
    print(f"High density test results:")
    print(f"  Attempted to place: {config.initial_cell_count} cells")
    print(f"  Successfully placed: {density_stats['total_cells']} cells")
    print(f"  Packing density: {density_stats['packing_density']:.3f}")
    print(f"  Average crowding factor: {density_stats['average_crowding_factor']:.3f}")
    print(f"  Area demand vs supply: {density_stats['area_demand_vs_supply']:.3f}")

    # Quick simulation
    simulator.set_constant_input(0.5)
    results = simulator.run()

    # Create visualization
    plotter = FlexibleCellPlotter(config)
    fig = plotter.plot_flexible_cell_visualization(
        simulator,
        show_crowding=True,
        show_territories=True,
        save_path="high_density_test.png"
    )

    plt.show()
    return simulator


if __name__ == "__main__":
    # Test basic flexible cells
    simulator1 = test_flexible_cells()

    # Test high density scenario
    simulator2 = test_high_density_scenario()

    print("All tests completed!")