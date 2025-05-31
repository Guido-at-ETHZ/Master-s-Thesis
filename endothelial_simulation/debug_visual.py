"""
Debug script to check if spatial properties are working
"""
from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.core import Simulator
import numpy as np


def debug_spatial_properties():
    """Debug the spatial properties connection."""

    print("DEBUGGING SPATIAL PROPERTIES")
    print("=" * 50)

    # Create simple test case
    config = SimulationConfig()
    config.simulation_duration = 60  # Just 1 hour for testing
    config.initial_cell_count = 10  # Fewer cells for easier debugging
    config.enable_spatial_properties = True
    config.enable_temporal_dynamics = True
    config.enable_population_dynamics = False  # Disable to prevent cell death

    simulator = Simulator(config)
    simulator.initialize()

    print(f"Initial cells: {len(simulator.grid.cells)}")

    # Check if cells have spatial properties set
    print("\nChecking initial spatial properties:")
    for i, (cell_id, cell) in enumerate(simulator.grid.cells.items()):
        if i >= 3:  # Only show first 3 cells
            break
        print(f"Cell {cell_id}:")
        print(f"  is_senescent: {cell.is_senescent}")
        print(f"  has target_orientation: {hasattr(cell, 'target_orientation')}")
        print(f"  has actual_orientation: {hasattr(cell, 'actual_orientation')}")
        if hasattr(cell, 'target_orientation'):
            print(f"  target_orientation: {getattr(cell, 'target_orientation', 'None')}")
        if hasattr(cell, 'actual_orientation'):
            print(f"  actual_orientation: {np.degrees(cell.actual_orientation):.1f}°")

    # Set high pressure
    simulator.set_constant_input(1.4)

    print(f"\nSet pressure to 1.4 Pa")
    print("Running 10 simulation steps...")

    # Run a few steps and monitor
    for step in range(10):
        step_info = simulator.step()

        if step % 5 == 0:  # Check every 5 steps
            print(f"\nStep {step}:")
            print(f"  Cells: {len(simulator.grid.cells)}")
            print(f"  Biological fitness: {simulator.grid.get_biological_fitness():.3f}")

            # Check if spatial model is setting targets
            if 'spatial' in simulator.models:
                spatial_model = simulator.models['spatial']
                print(f"  Spatial model active: True")

                # Check first cell's targets
                if simulator.grid.cells:
                    first_cell = list(simulator.grid.cells.values())[0]
                    print(f"  First cell targets:")
                    print(f"    target_orientation: {getattr(first_cell, 'target_orientation', 'None')}")
                    print(f"    target_aspect_ratio: {getattr(first_cell, 'target_aspect_ratio', 'None')}")
                    print(f"    actual_orientation: {np.degrees(first_cell.actual_orientation):.1f}°")
            else:
                print(f"  Spatial model active: False")

    return simulator


def check_spatial_model_directly():
    """Check if the spatial model is working independently."""

    print("\nCHECKING SPATIAL MODEL DIRECTLY")
    print("=" * 40)

    from endothelial_simulation.models.spatial_properties import SpatialPropertiesModel
    from endothelial_simulation.models.temporal_dynamics import TemporalDynamicsModel

    config = SimulationConfig()
    temporal_model = TemporalDynamicsModel(config)
    spatial_model = SpatialPropertiesModel(config, temporal_model)

    # Create a test cell
    from endothelial_simulation.core.cell import Cell
    test_cell = Cell(1, (100, 100), 0, False, None, 1000)

    print("Testing spatial model with 1.4 Pa pressure on healthy cell:")

    # Update cell properties
    result = spatial_model.update_cell_properties(test_cell, 1.4, 1.0, {1: test_cell})

    print(f"Target orientation: {getattr(test_cell, 'target_orientation', 'None')}")
    print(f"Target aspect ratio: {getattr(test_cell, 'target_aspect_ratio', 'None')}")
    print(f"Target area: {getattr(test_cell, 'target_area', 'None')}")

    if hasattr(test_cell, 'target_orientation'):
        print(f"Target orientation in degrees: {np.degrees(test_cell.target_orientation):.1f}°")

    return spatial_model, test_cell


if __name__ == "__main__":
    # Run debugging
    simulator = debug_spatial_properties()
    spatial_model, test_cell = check_spatial_model_directly()

    print("\n" + "=" * 50)
    print("DIAGNOSIS:")
    print("=" * 50)

    if len(simulator.grid.cells) < 5:
        print("❌ ISSUE: Too few cells - population dynamics may be killing cells")
        print("   FIX: Disable population dynamics for testing")

    fitness = simulator.grid.get_biological_fitness()
    if fitness < 0.5:
        print(f"❌ ISSUE: Low biological fitness ({fitness:.3f})")
        print("   FIX: Increase adaptation strength or frequency")

    if not hasattr(test_cell, 'target_orientation'):
        print("❌ ISSUE: Spatial model not setting target properties")
        print("   FIX: Check spatial model implementation")
    else:
        expected_orientation = 0.0  # Should be horizontal for 1.4 Pa
        actual_target = np.degrees(test_cell.target_orientation)
        if abs(actual_target) > 30:  # Should be close to 0°
            print(f"❌ ISSUE: Target orientation is {actual_target:.1f}° (should be ~0°)")
            print("   FIX: Check spatial model calculations")
        else:
            print(f"✅ Target orientation correct: {actual_target:.1f}°")