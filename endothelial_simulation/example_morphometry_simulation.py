"""
Example script demonstrating the updated endothelial cell morphometry simulation
with senescence and pressure effects.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from endothelial_simulation.config import SimulationConfig
from endothelial_simulation.core import Simulator
from endothelial_simulation.visualization import Plotter


def run_pressure_senescence_simulation(pressure, initial_senescence_fraction=0.0, duration=360):
    """
    Run a simulation with specified pressure and initial senescence fraction.
    
    Parameters:
        pressure: Applied pressure in Pa (0 to 1.4)
        initial_senescence_fraction: Initial fraction of senescent cells (0 to 1)
        duration: Simulation duration in minutes
    
    Returns:
        simulator: Completed simulator object
    """
    # Create configuration
    config = SimulationConfig()
    config.simulation_duration = duration
    config.initial_cell_count = 100
    config.plot_interval = 10
    
    # Create simulator
    simulator = Simulator(config)
    
    # Initialize cells with specified senescence fraction
    total_cells = config.initial_cell_count
    senescent_cells = int(total_cells * initial_senescence_fraction)
    normal_cells = total_cells - senescent_cells
    
    # Add normal cells
    for _ in range(normal_cells):
        simulator.grid.add_cell(
            position=None,  # Random position
            divisions=np.random.randint(0, 10),
            is_senescent=False
        )
    
    # Add senescent cells
    for _ in range(senescent_cells):
        simulator.grid.add_cell(
            position=None,  # Random position
            divisions=config.max_divisions,
            is_senescent=True,
            senescence_cause='telomere' if np.random.random() > 0.5 else 'stress'
        )
    
    # Set constant pressure input
    simulator.set_constant_input(pressure)
    
    # Run simulation
    print(f"Running simulation: Pressure={pressure} Pa, Initial senescence={initial_senescence_fraction*100:.0f}%")
    simulator.run(duration)
    
    return simulator


def compare_morphometry_scenarios():
    """
    Compare cell morphometry under different pressure and senescence scenarios.
    """
    # Define scenarios
    scenarios = [
        {"pressure": 0.0, "senescence": 0.0, "label": "0 Pa, 0% Senescent"},
        {"pressure": 0.0, "senescence": 0.5, "label": "0 Pa, 50% Senescent"},
        {"pressure": 0.0, "senescence": 1.0, "label": "0 Pa, 100% Senescent"},
        {"pressure": 1.4, "senescence": 0.0, "label": "1.4 Pa, 0% Senescent"},
        {"pressure": 1.4, "senescence": 0.5, "label": "1.4 Pa, 50% Senescent"},
        {"pressure": 1.4, "senescence": 1.0, "label": "1.4 Pa, 100% Senescent"},
    ]
    
    # Run simulations
    results = {}
    for scenario in scenarios:
        simulator = run_pressure_senescence_simulation(
            pressure=scenario["pressure"],
            initial_senescence_fraction=scenario["senescence"],
            duration=180  # 3 hours
        )
        results[scenario["label"]] = simulator
    
    # Create comparison plots
    create_morphometry_comparison_plots(results)
    
    return results


def create_morphometry_comparison_plots(results):
    """
    Create plots comparing morphometry across different scenarios.
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot cell visualizations for each scenario
    for idx, (label, simulator) in enumerate(results.items()):
        ax = axes[idx]
        
        # Set axis limits
        ax.set_xlim(0, simulator.grid.width)
        ax.set_ylim(0, simulator.grid.height)
        
        # Plot cells
        for cell in simulator.grid.cells.values():
            # Get cell properties
            x, y = cell.position
            orientation = cell.orientation
            aspect_ratio = cell.aspect_ratio
            area = cell.area
            
            # Calculate ellipse dimensions
            a = np.sqrt(area * aspect_ratio)
            b = area / a
            
            # Create ellipse
            from matplotlib.patches import Ellipse
            ellipse = Ellipse(
                xy=(x, y),
                width=2 * a,
                height=2 * b,
                angle=np.degrees(orientation),
                alpha=0.7
            )
            
            # Set color based on cell type
            if not cell.is_senescent:
                color = 'green'
            elif cell.senescence_cause == 'telomere':
                color = 'red'
            else:
                color = 'blue'
            
            ellipse.set_facecolor(color)
            ellipse.set_edgecolor('black')
            ellipse.set_linewidth(0.5)
            
            ax.add_patch(ellipse)
        
        ax.set_title(label, fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Cell Morphometry Under Different Pressure and Senescence Conditions', fontsize=16)
    plt.tight_layout()
    plt.savefig('morphometry_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create quantitative comparison plot
    create_quantitative_comparison(results)


def create_quantitative_comparison(results):
    """
    Create plots showing quantitative morphometry metrics.
    """
    # Extract metrics
    scenarios = []
    avg_areas = []
    avg_aspect_ratios = []
    avg_orientations = []
    
    for label, simulator in results.items():
        scenarios.append(label)
        
        # Calculate average metrics
        areas = [cell.area for cell in simulator.grid.cells.values()]
        aspect_ratios = [cell.aspect_ratio for cell in simulator.grid.cells.values()]
        orientations = [np.degrees(cell.orientation) for cell in simulator.grid.cells.values()]
        
        avg_areas.append(np.mean(areas))
        avg_aspect_ratios.append(np.mean(aspect_ratios))
        avg_orientations.append(np.mean(orientations))
    
    # Create bar plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    x = np.arange(len(scenarios))
    
    # Plot average area
    ax1.bar(x, avg_areas, color=['green', 'yellow', 'red', 'lightgreen', 'orange', 'darkred'])
    ax1.set_ylabel('Average Cell Area (pixels)', fontsize=12)
    ax1.set_title('Cell Area Across Scenarios', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add reference lines
    ax1.axhline(y=500, color='g', linestyle='--', alpha=0.5, label='Normal at 0 Pa')
    ax1.axhline(y=1000, color='b', linestyle='--', alpha=0.5, label='Normal at 1.4 Pa')
    ax1.axhline(y=2000, color='r', linestyle='--', alpha=0.5, label='Senescent')
    ax1.legend()
    
    # Plot average aspect ratio
    ax2.bar(x, avg_aspect_ratios, color=['green', 'yellow', 'red', 'lightgreen', 'orange', 'darkred'])
    ax2.set_ylabel('Average Aspect Ratio', fontsize=12)
    ax2.set_title('Cell Aspect Ratio Across Scenarios', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Plot average orientation
    ax3.bar(x, avg_orientations, color=['green', 'yellow', 'red', 'lightgreen', 'orange', 'darkred'])
    ax3.set_ylabel('Average Orientation (degrees)', fontsize=12)
    ax3.set_title('Cell Orientation Across Scenarios', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add reference line for 45 degrees (senescent orientation)
    ax3.axhline(y=45, color='r', linestyle='--', alpha=0.5, label='Senescent orientation')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('morphometry_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_pressure_response():
    """
    Demonstrate how cells respond to pressure changes over time.
    """
    # Create configuration
    config = SimulationConfig()
    config.simulation_duration = 480  # 8 hours
    config.initial_cell_count = 50
    
    # Create simulator with mixed population
    simulator = Simulator(config)
    
    # Initialize with 30% senescent cells
    for i in range(config.initial_cell_count):
        is_senescent = i < config.initial_cell_count * 0.3
        simulator.grid.add_cell(
            is_senescent=is_senescent,
            senescence_cause='telomere' if is_senescent else None
        )
    
    # Set step input: 0 Pa initially, then 1.4 Pa after 2 hours
    simulator.set_step_input(
        initial_value=0.0,
        final_value=1.4,
        step_time=120  # 2 hours
    )
    
    # Run simulation
    print("Running pressure step response simulation...")
    simulator.run()
    
    # Create plots
    plotter = Plotter(config)
    plotter.create_all_plots(simulator)
    
    # Create specific morphometry tracking plot
    create_morphometry_tracking_plot(simulator)
    
    return simulator


def create_morphometry_tracking_plot(simulator):
    """
    Create a plot tracking morphometry changes over time.
    """
    # This would require storing morphometry data during simulation
    # For now, we'll create a conceptual plot
    
    time = np.array([state['time'] / 60 for state in simulator.history])  # Convert to hours
    pressure = np.array([state['input_value'] for state in simulator.history])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot pressure
    ax1.plot(time, pressure, 'r-', linewidth=2)
    ax1.set_ylabel('Pressure (Pa)', fontsize=12)
    ax1.set_title('Pressure Input', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot cell count (as proxy for morphometry changes)
    if 'cells' in simulator.history[0]:
        cell_count = np.array([state['cells'] for state in simulator.history])
        ax2.plot(time, cell_count, 'b-', linewidth=2)
        ax2.set_ylabel('Total Cell Count', fontsize=12)
    
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_title('Cell Population Response', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Cell Response to Pressure Step Change', fontsize=16)
    plt.tight_layout()
    plt.savefig('pressure_response.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run demonstrations
    print("=== Endothelial Cell Morphometry Simulation ===")
    print("Demonstrating senescence and pressure effects on cell morphometry\n")
    
    # 1. Compare different scenarios
    print("1. Comparing morphometry across pressure and senescence scenarios...")
    results = compare_morphometry_scenarios()
    
    # 2. Demonstrate pressure response
    print("\n2. Demonstrating cell response to pressure changes...")
    simulator = demonstrate_pressure_response()
    
    print("\nSimulation complete! Check the generated plots for results.")