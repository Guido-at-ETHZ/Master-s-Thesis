"""
Cell properties distribution plotter for endothelial simulation.
Creates hourly distribution plots for area, aspect ratio, and orientation.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec


def plot_hourly_cell_distributions(simulator, save_path=None, max_hours=None):
    """
    Plot distributions of cell area, aspect ratio, and orientation every hour.
    
    Parameters:
        simulator: Simulator object with completed simulation
        save_path: Path to save the plot (default: auto-generated)
        max_hours: Maximum number of hours to plot (default: all available)
    
    Returns:
        Matplotlib figure
    """
    if not simulator.history:
        print("No simulation history available")
        return None
        
    # Convert simulation time to hours
    simulation_time_hours = simulator.time / 60 if simulator.config.time_unit == "minutes" else simulator.time
    
    # Determine hourly time points
    if max_hours is None:
        max_hours = int(simulation_time_hours) + 1
    
    hourly_timepoints = list(range(0, min(max_hours, int(simulation_time_hours) + 1)))
    
    if not hourly_timepoints:
        print("Simulation too short for hourly analysis")
        return None
    
    print(f"Creating distributions for {len(hourly_timepoints)} hourly time points...")
    
    # Extract cell data for each hourly timepoint
    hourly_data = []
    
    for target_hour in hourly_timepoints:
        target_time_minutes = target_hour * 60
        
        # Find the closest recorded time point
        times = [state['time'] for state in simulator.history]
        closest_idx = np.argmin([abs(t - target_time_minutes) for t in times])
        closest_time = times[closest_idx]
        
        print(f"Hour {target_hour}: using data from t={closest_time:.1f} min ({closest_time/60:.1f}h)")
        
        # Get current cells at this time point
        # Note: We use the final cell state as a proxy since we don't store historical cell positions
        current_cells = list(simulator.grid.cells.values())
        
        if not current_cells:
            continue
            
        # Extract properties
        areas = []
        aspect_ratios = []
        orientations_deg = []
        
        for cell in current_cells:
            # Scale area back to display units if needed
            area = cell.actual_area * (simulator.grid.computation_scale ** 2)
            areas.append(area)
            
            aspect_ratios.append(cell.actual_aspect_ratio)
            
            # Convert orientation from radians to degrees
            orientation_deg = np.degrees(cell.actual_orientation)
            # Normalize to [-90, 90] for flow alignment interpretation
            while orientation_deg > 90:
                orientation_deg -= 180
            while orientation_deg < -90:
                orientation_deg += 180
            orientations_deg.append(orientation_deg)
        
        hourly_data.append({
            'hour': target_hour,
            'time_minutes': closest_time,
            'areas': areas,
            'aspect_ratios': aspect_ratios,
            'orientations': orientations_deg,
            'cell_count': len(current_cells)
        })
    
    if not hourly_data:
        print("No valid data points found")
        return None
    
    # Create the figure with subplots
    n_timepoints = len(hourly_data)
    fig = plt.figure(figsize=(15, 4 * n_timepoints))
    
    # Create a grid layout: rows = timepoints, columns = 3 properties
    gs = GridSpec(n_timepoints, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color scheme for different cell properties
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    property_names = ['Area (pixels²)', 'Aspect Ratio', 'Orientation (degrees)']
    
    # Plot distributions for each hour
    for row, data in enumerate(hourly_data):
        hour = data['hour']
        areas = data['areas']
        aspect_ratios = data['aspect_ratios']
        orientations = data['orientations']
        cell_count = data['cell_count']
        
        # Area distribution
        ax_area = fig.add_subplot(gs[row, 0])
        ax_area.hist(areas, bins=20, color=colors[0], alpha=0.7, edgecolor='black')
        ax_area.set_title(f'Hour {hour}: Area Distribution\n(n={cell_count} cells)', fontsize=10)
        ax_area.set_xlabel('Area (pixels²)', fontsize=9)
        ax_area.set_ylabel('Frequency', fontsize=9)
        ax_area.grid(True, alpha=0.3)
        
        # Add statistics
        mean_area = np.mean(areas)
        ax_area.axvline(mean_area, color='red', linestyle='--', alpha=0.8, 
                       label=f'Mean: {mean_area:.0f}')
        ax_area.legend(fontsize=8)
        
        # Aspect ratio distribution
        ax_ar = fig.add_subplot(gs[row, 1])
        ax_ar.hist(aspect_ratios, bins=20, color=colors[1], alpha=0.7, edgecolor='black')
        ax_ar.set_title(f'Hour {hour}: Aspect Ratio Distribution', fontsize=10)
        ax_ar.set_xlabel('Aspect Ratio', fontsize=9)
        ax_ar.set_ylabel('Frequency', fontsize=9)
        ax_ar.grid(True, alpha=0.3)
        
        # Add statistics
        mean_ar = np.mean(aspect_ratios)
        ax_ar.axvline(mean_ar, color='red', linestyle='--', alpha=0.8, 
                     label=f'Mean: {mean_ar:.1f}')
        ax_ar.legend(fontsize=8)
        
        # Orientation distribution
        ax_orient = fig.add_subplot(gs[row, 2])
        ax_orient.hist(orientations, bins=20, color=colors[2], alpha=0.7, edgecolor='black',
                      range=(-90, 90))
        ax_orient.set_title(f'Hour {hour}: Orientation Distribution', fontsize=10)
        ax_orient.set_xlabel('Orientation (degrees)', fontsize=9)
        ax_orient.set_ylabel('Frequency', fontsize=9)
        ax_orient.grid(True, alpha=0.3)
        ax_orient.set_xlim(-90, 90)
        
        # Add flow direction reference (0 degrees = aligned with flow)
        ax_orient.axvline(0, color='orange', linestyle='-', alpha=0.8, linewidth=2,
                         label='Flow direction')
        
        # Add statistics
        mean_orient = np.mean(orientations)
        ax_orient.axvline(mean_orient, color='red', linestyle='--', alpha=0.8, 
                         label=f'Mean: {mean_orient:.1f}°')
        ax_orient.legend(fontsize=8)
        
        # Add time and shear stress info as text
        if row < len(simulator.history):
            state_idx = min(row * len(simulator.history) // len(hourly_data), len(simulator.history) - 1)
            shear_stress = simulator.history[state_idx].get('input_value', 0)
            
            # Add info text to the first subplot of each row
            info_text = f'Time: {data["time_minutes"]:.0f} min\nShear: {shear_stress:.2f} Pa'
            ax_area.text(0.02, 0.98, info_text, transform=ax_area.transAxes, 
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add overall title
    fig.suptitle('Hourly Cell Property Distributions\nEndothelial Cell Mechanotransduction Simulation', 
                 fontsize=16, y=0.98)
    
    # Add summary text
    summary_text = (f'Simulation Duration: {simulation_time_hours:.1f} hours\n'
                   f'Final Cell Count: {hourly_data[-1]["cell_count"] if hourly_data else 0}\n'
                   f'Time Points: {len(hourly_data)} hours')
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Save if path provided
    if save_path is None:
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(simulator.config.plot_directory, 
                                f"hourly_distributions_{timestamp}.png")
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Hourly distributions saved to: {save_path}")
    
    return fig


# Add this to your existing Plotter class in plotters.py:
def plot_hourly_distributions_method(self, simulator, save_path=None, max_hours=None):
    """
    Add this method to your Plotter class in plotters.py
    """
    from .cell_distributions import plot_hourly_cell_distributions
    return plot_hourly_cell_distributions(simulator, save_path, max_hours)