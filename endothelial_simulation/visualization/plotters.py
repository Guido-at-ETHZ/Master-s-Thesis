"""
Visualization module for plotting simulation results with mosaic cells.
Fixed to properly handle coordinate scaling and enhanced senescent cell growth.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os
from matplotlib.gridspec import GridSpec

from . import animations

class Plotter:
    """
    Class for creating visualizations of simulation results with mosaic cells.
    """

    def __init__(self, config):
        """
        Initialize the plotter.

        Parameters:
            config: SimulationConfig object with parameter settings
        """
        self.config = config

        # Create output directory
        os.makedirs(config.plot_directory, exist_ok=True)

        # Set default plot style
        plt.style.use('seaborn-v0_8-darkgrid')


    def plot_cell_visualization(self, simulator, save_path=None, show_boundaries=True, show_seeds=False):
        """
        Create a visualization of cells as mosaic territories.
        Enhanced to show senescent cell growth factors.

        Parameters:
            simulator: Simulator object with current state
            save_path: Path to save the plot (default: auto-generated)
            show_boundaries: Whether to show cell boundaries
            show_seeds: Whether to show seed points

        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))

        # Set axis limits based on grid size
        ax.set_xlim(0, simulator.grid.width)
        ax.set_ylim(0, simulator.grid.height)

        # Get display territories (properly scaled)
        display_territories = simulator.grid.get_display_territories()

        # Plot each cell's territory
        for cell_id, cell in simulator.grid.cells.items():
            if cell_id not in display_territories:
                continue

            display_pixels = display_territories[cell_id]
            if not display_pixels:
                continue

            # Enhanced color determination with size-based shading
            edge_width = 0.5  # Default edge width

            if not cell.is_senescent:
                color = 'green'
                alpha = 0.6
            else:
                # Senescent cells - color intensity based on size
                growth_factor = getattr(cell, 'senescent_growth_factor', 1.0)

                # Base color depends on senescence cause
                if cell.senescence_cause == 'telomere':
                    base_color = '#DC143C'  # Crimson
                else:
                    base_color = '#4169E1'  # Royal Blue

                # Darker and more opaque for larger senescent cells
                size_intensity = min(1.0, (growth_factor - 1.0) / 2.0)  # 0 to 1 scale
                alpha = 0.7 + 0.2 * size_intensity  # 0.7 to 0.9
                edge_width = 0.5 + 1.5 * size_intensity  # 0.5 to 2.0

                # Make larger cells darker
                if growth_factor > 2.0:
                    color = '#8B0000' if cell.senescence_cause == 'telomere' else '#191970'  # Very dark
                elif growth_factor > 1.5:
                    color = '#B22222' if cell.senescence_cause == 'telomere' else '#000080'  # Dark
                else:
                    color = base_color  # Normal senescent color

            # Create polygon from display territory pixels
            if len(display_pixels) > 10:  # Only for territories with reasonable size
                try:
                    # Sample pixels if territory is very large (for performance)
                    pixels_to_use = display_pixels
                    if len(pixels_to_use) > 500:
                        # Randomly sample pixels for boundary detection
                        indices = np.random.choice(len(pixels_to_use), 500, replace=False)
                        pixels_to_use = [pixels_to_use[i] for i in indices]

                    # Create convex hull for visualization
                    from scipy.spatial import ConvexHull
                    points = np.array(pixels_to_use)
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]

                    polygon = Polygon(hull_points, facecolor=color, alpha=alpha,
                                      edgecolor='black' if show_boundaries else color,
                                      linewidth=edge_width)
                    ax.add_patch(polygon)
                except Exception as e:
                    # Fallback: scatter plot of pixels
                    if len(display_pixels) > 100:
                        # Sample pixels for performance
                        sample_size = min(100, len(display_pixels))
                        indices = np.random.choice(len(display_pixels), sample_size, replace=False)
                        sampled_pixels = [display_pixels[i] for i in indices]
                        points = np.array(sampled_pixels)
                    else:
                        points = np.array(display_pixels)

                    ax.scatter(points[:, 0], points[:, 1], c=color, alpha=alpha, s=2, marker='s')

            # Show seed point if requested
            if show_seeds:
                seed_x, seed_y = cell.position
                ax.plot(seed_x, seed_y, 'ko', markersize=4)

            # Show orientation vector at centroid
            if cell.centroid is not None:
                # Convert centroid from computational to display coordinates
                display_centroid = simulator.grid._comp_to_display_coords(cell.centroid[0], cell.centroid[1])
                cx, cy = display_centroid

                # Draw orientation vector
                vector_length = np.sqrt(cell.actual_area * (simulator.grid.computation_scale ** 2)) * 0.15
                dx = vector_length * np.cos(cell.actual_orientation)
                dy = vector_length * np.sin(cell.actual_orientation)

                ax.arrow(cx, cy, dx, dy, head_width=vector_length * 0.2,
                         head_length=vector_length * 0.2, fc='white', ec='black',
                         alpha=0.9, width=vector_length * 0.05, zorder=10)

        # Enhanced legend showing size categories
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', alpha=0.6, label='Healthy'),
            Patch(facecolor='#DC143C', edgecolor='black', alpha=0.7, label='Senescent (Tel)'),
            Patch(facecolor='#4169E1', edgecolor='black', alpha=0.7, label='Senescent (Stress)'),
            Patch(facecolor='#B22222', edgecolor='black', alpha=0.8, label='Enlarged Sen. (Tel)'),
            Patch(facecolor='#000080', edgecolor='black', alpha=0.8, label='Enlarged Sen. (Stress)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        # Format plot
        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)
        ax.set_aspect('equal')

        # Add info text
        total_cells = len(simulator.grid.cells)
        grid_stats = simulator.grid.get_grid_statistics()

        # Count enlarged senescent cells
        enlarged_senescent = sum(1 for cell in simulator.grid.cells.values()
                               if cell.is_senescent and getattr(cell, 'senescent_growth_factor', 1.0) > 1.2)

        info_text = (
            f"Time: {simulator.time:.1f} {simulator.config.time_unit}\n"
            f"Shear Stress: {simulator.input_pattern['value']:.2f} Pa\n"
            f"Total Cells: {total_cells}\n"
            f"Enlarged Senescent: {enlarged_senescent}\n"
            f"Packing Efficiency: {grid_stats.get('packing_efficiency', 0):.2f}\n"
            f"Global Pressure: {grid_stats.get('global_pressure', 1.0):.2f}"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Add flow direction indicator
        arrow_length = simulator.grid.width * 0.08
        arrow_x = simulator.grid.width * 0.5
        arrow_y = simulator.grid.height * 0.05

        ax.arrow(arrow_x - arrow_length / 2, arrow_y, arrow_length, 0,
                 head_width=arrow_length * 0.3, head_length=arrow_length * 0.2,
                 fc='black', ec='black', width=arrow_length * 0.08, zorder=5)

        ax.text(arrow_x, arrow_y - arrow_length * 0.5, "Flow Direction",
                ha='center', va='top', fontsize=12, weight='bold')

        # Add title
        plt.title('Endothelial Cell Mosaic Visualization with Growth Factors', fontsize=16)

        # Save if requested
        if save_path is not None:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        return fig

    def plot_cell_population(self, history, save_path=None):
        """
        Plot cell population over time.

        Parameters:
            history: List of state dictionaries from simulation
            save_path: Path to save the plot (default: auto-generated)

        Returns:
            Matplotlib figure
        """
        # Extract data
        time = np.array([state['time'] for state in history])

        # Convert time to hours if needed
        if self.config.time_unit == "minutes":
            time_in_hours = time / 60
            time_label = "Time (hours)"
        else:
            time_in_hours = time
            time_label = f"Time ({self.config.time_unit})"

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Check if detailed population data exists
        if 'healthy_cells' in history[0]:
            healthy = np.array([state['healthy_cells'] for state in history])
            sen_tel = np.array([state['senescent_tel'] for state in history])
            sen_stress = np.array([state['senescent_stress'] for state in history])

            ax.plot(time_in_hours, healthy, 'g-', linewidth=2, label='Healthy Cells')
            ax.plot(time_in_hours, sen_tel, 'r-', linewidth=2, label='Telomere-Induced Senescent')
            ax.plot(time_in_hours, sen_stress, 'b-', linewidth=2, label='Stress-Induced Senescent')
            ax.plot(time_in_hours, healthy + sen_tel + sen_stress, 'k--', linewidth=1, label='Total Cells')
        else:
            # Just plot total cells
            total_cells = np.array([state['cells'] for state in history])
            ax.plot(time_in_hours, total_cells, 'k-', linewidth=2, label='Total Cells')

        # Format plot
        ax.set_xlabel(time_label, fontsize=12)
        ax.set_ylabel('Cell Count', fontsize=12)
        ax.set_title('Cell Population Dynamics', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True)

        # Save if requested
        if save_path is not None:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return fig

    def plot_senescent_growth_metrics(self, history, save_path=None):
        """
        Plot senescent cell growth metrics over time.

        Parameters:
            history: List of state dictionaries from simulation
            save_path: Path to save the plot (default: auto-generated)

        Returns:
            Matplotlib figure
        """
        if not history or 'senescent_count' not in history[0]:
            print("Senescent growth data not available")
            return None

        # Extract time data
        time = np.array([state['time'] for state in history])
        time_hours = time / 60 if self.config.time_unit == "minutes" else time
        time_label = "Time (hours)" if self.config.time_unit == "minutes" else f"Time ({self.config.time_unit})"

        # Extract data
        senescent_count = np.array([state.get('senescent_count', 0) for state in history])
        enlarged_count = np.array([state.get('enlarged_senescent_count', 0) for state in history])
        mean_size = np.array([state.get('mean_senescent_size', 1.0) for state in history])
        max_size = np.array([state.get('max_senescent_size', 1.0) for state in history])

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Population plot
        ax1.plot(time_hours, senescent_count, 'r-', linewidth=2, label='Total Senescent')
        ax1.plot(time_hours, enlarged_count, 'darkred', linewidth=2, label='Enlarged (>1.2x)')
        ax1.fill_between(time_hours, 0, enlarged_count, alpha=0.3, color='darkred')

        ax1.set_ylabel('Cell Count', fontsize=12)
        ax1.set_title('Senescent Cell Growth Over Time', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True)

        # Size metrics plot
        ax2.plot(time_hours, mean_size, 'b-', linewidth=2, label='Mean Size Factor')
        ax2.plot(time_hours, max_size, 'r--', linewidth=2, label='Max Size Factor')
        ax2.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='Normal Size')
        ax2.axhline(y=3.0, color='r', linestyle=':', alpha=0.5, label='Max Allowed')

        ax2.set_xlabel(time_label, fontsize=12)
        ax2.set_ylabel('Size Factor', fontsize=12)
        ax2.set_title('Senescent Cell Size Evolution', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        return fig

    def plot_mosaic_metrics(self, history, save_path=None):
        """
        Plot mosaic-specific metrics over time.

        Parameters:
            history: List of state dictionaries from simulation
            save_path: Path to save the plot (default: auto-generated)

        Returns:
            Matplotlib figure
        """
        # Extract data
        time = np.array([state['time'] for state in history])

        # Convert time to hours if needed
        if self.config.time_unit == "minutes":
            time_in_hours = time / 60
            time_label = "Time (hours)"
        else:
            time_in_hours = time
            time_label = f"Time ({self.config.time_unit})"

        # Check if mosaic metrics exist
        if 'packing_efficiency' not in history[0]:
            print("Mosaic metrics not available in history data")
            return None

        packing_efficiency = np.array([state['packing_efficiency'] for state in history])
        global_pressure = np.array([state.get('global_pressure', 1.0) for state in history])
        mean_compression = np.array([state.get('mean_compression_ratio', 1.0) for state in history])

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        # Plot metrics
        axes[0].plot(time_in_hours, packing_efficiency, 'g-', linewidth=2)
        axes[0].set_ylabel('Packing Efficiency', fontsize=12)
        axes[0].set_title('Cell Packing Quality', fontsize=12)
        axes[0].set_ylim(0, 1)

        axes[1].plot(time_in_hours, global_pressure, 'r-', linewidth=2)
        axes[1].set_ylabel('Global Pressure', fontsize=12)
        axes[1].set_title('Crowding Pressure', fontsize=12)
        axes[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No pressure')
        axes[1].legend()

        axes[2].plot(time_in_hours, mean_compression, 'b-', linewidth=2)
        axes[2].set_ylabel('Mean Compression Ratio', fontsize=12)
        axes[2].set_title('Average Cell Compression', fontsize=12)
        axes[2].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No compression')
        axes[2].legend()

        # Format plot
        axes[2].set_xlabel(time_label, fontsize=12)
        plt.suptitle('Mosaic Structure Metrics', fontsize=14)

        # Save if requested
        if save_path is not None:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        return fig

    def plot_spatial_metrics(self, history, save_path=None):
        """
        Plot spatial metrics over time.

        Parameters:
            history: List of state dictionaries from simulation
            save_path: Path to save the plot (default: auto-generated)

        Returns:
            Matplotlib figure
        """
        # Extract data
        time = np.array([state['time'] for state in history])

        # Check if spatial metrics exist
        if 'alignment_index' not in history[0]:
            print("Spatial metrics not available in history data")
            return None

        alignment = np.array([state['alignment_index'] for state in history])
        shape_index = np.array([state['shape_index'] for state in history])
        packing_quality = np.array([state.get('packing_quality', 1.0) for state in history])

        # Convert time to hours if needed
        if self.config.time_unit == "minutes":
            time_in_hours = time / 60
            time_label = "Time (hours)"
        else:
            time_in_hours = time
            time_label = f"Time ({self.config.time_unit})"

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        # Plot metrics
        axes[0].plot(time_in_hours, alignment, 'b-', linewidth=2)
        axes[0].set_ylabel('Alignment Index', fontsize=12)
        axes[0].set_title('Cell Alignment with Flow Direction', fontsize=12)
        axes[0].set_ylim(0, 1)

        axes[1].plot(time_in_hours, shape_index, 'g-', linewidth=2)
        axes[1].set_ylabel('Shape Index', fontsize=12)
        axes[1].set_title('Cell Shape Index', fontsize=12)

        axes[2].plot(time_in_hours, packing_quality, 'r-', linewidth=2)
        axes[2].set_ylabel('Packing Quality', fontsize=12)
        axes[2].set_title('Mosaic Packing Quality', fontsize=12)
        axes[2].set_ylim(0, 1)

        # Format plot
        axes[2].set_xlabel(time_label, fontsize=12)
        plt.suptitle('Spatial Metrics of Endothelial Mosaic', fontsize=14)

        # Save if requested
        if save_path is not None:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        return fig

    def plot_input_pattern(self, history, save_path=None):
        """
        Plot input pattern over time.

        Parameters:
            history: List of state dictionaries from simulation
            save_path: Path to save the plot (default: auto-generated)

        Returns:
            Matplotlib figure
        """
        # Extract data
        time = np.array([state['time'] for state in history])
        input_value = np.array([state['input_value'] for state in history])

        # Convert time to hours if needed
        if self.config.time_unit == "minutes":
            time_in_hours = time / 60
            time_label = "Time (hours)"
        else:
            time_in_hours = time
            time_label = f"Time ({self.config.time_unit})"

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot input pattern
        ax.plot(time_in_hours, input_value, 'r-', linewidth=2, drawstyle='steps-post')

        # Format plot
        ax.set_xlabel(time_label, fontsize=12)
        ax.set_ylabel('Shear Stress (Pa)', fontsize=12)
        ax.set_title('Input Shear Stress Pattern', fontsize=14)
        ax.grid(True)

        # Save if requested
        if save_path is not None:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        return fig

    def plot_cell_distributions(self, simulator, save_path=None, time_points='auto', show_evolution=True):
        """
        Plot cell property distributions at multiple time points or as evolution.
        FIXED: No double conversion of orientation angles.

        Parameters:
            simulator: Simulator object with completed simulation
            save_path: Path to save the plot (default: auto-generated)
            time_points: 'auto' for automatic selection, 'hourly', or list of specific times
            show_evolution: If True, show evolution plot; if False, show static snapshots

        Returns:
            Matplotlib figure
        """
        if not simulator.history:
            print("No simulation history available")
            return None

        print("Creating enhanced cell distributions with FIXED orientation handling...")

        # Extract time data
        times = [state['time'] for state in simulator.history]
        max_time = max(times)
        simulation_hours = max_time / 60 if self.config.time_unit == "minutes" else max_time

        # Determine time points to analyze
        if time_points == 'auto':
            # Automatic selection: start, 25%, 50%, 75%, end
            selected_indices = [
                0,
                len(simulator.history) // 4,
                len(simulator.history) // 2,
                3 * len(simulator.history) // 4,
                len(simulator.history) - 1
            ]
            selected_indices = sorted(list(set(selected_indices)))  # Remove duplicates
        elif time_points == 'hourly':
            # Hourly time points
            target_hours = list(range(0, int(simulation_hours) + 1))
            selected_indices = []
            for target_hour in target_hours:
                target_time = target_hour * 60
                closest_idx = np.argmin([abs(t - target_time) for t in times])
                selected_indices.append(closest_idx)
            selected_indices = sorted(list(set(selected_indices)))
        else:
            # Custom time points
            selected_indices = []
            for target_time in time_points:
                closest_idx = np.argmin([abs(t - target_time) for t in times])
                selected_indices.append(closest_idx)

        # Extract data for selected time points
        time_data = []
        for idx in selected_indices:
            state = simulator.history[idx]

            if 'cell_properties' not in state:
                print(f"Warning: No cell properties at time {state['time']:.1f}")
                continue

            cell_props = state['cell_properties']

            # FIXED: No double conversion - orientations are already 0-90° alignment angles
            areas = cell_props['areas']
            aspect_ratios = cell_props['aspect_ratios']
            orientations = cell_props['orientations']  # Already 0-90° flow alignment angles!

            time_data.append({
                'time_minutes': state['time'],
                'time_hours': state['time'] / 60,
                'shear_stress': state.get('input_value', 0),
                'areas': areas,
                'aspect_ratios': aspect_ratios,
                'orientations': orientations,  # No conversion needed!
                'cell_count': len(areas)
            })

            print(f"  Time {state['time']:.1f}min: {len(areas)} cells, "
                  f"orientation range {min(orientations):.1f}°-{max(orientations):.1f}°")

        if not time_data:
            print("No valid time points found")
            return None

        if show_evolution:
            return self._create_evolution_plot(time_data, save_path)
        else:
            return self._create_snapshot_plot(time_data, save_path)

    def _create_evolution_plot(self, time_data, save_path):
        """Create a plot showing how distributions evolve over time."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cell Property Distribution Evolution\n(FIXED: Correct Flow Alignment Angles)',
                     fontsize=16, y=0.95)

        # Calculate statistics over time
        time_stats = {
            'times': [data['time_hours'] for data in time_data],
            'mean_areas': [np.mean(data['areas']) for data in time_data],
            'std_areas': [np.std(data['areas']) for data in time_data],
            'mean_ars': [np.mean(data['aspect_ratios']) for data in time_data],
            'std_ars': [np.std(data['aspect_ratios']) for data in time_data],
            'mean_orients': [np.mean(data['orientations']) for data in time_data],
            'std_orients': [np.std(data['orientations']) for data in time_data],
            'cell_counts': [data['cell_count'] for data in time_data],
            'shear_stresses': [data['shear_stress'] for data in time_data]
        }

        # 1. Area evolution
        ax1 = axes[0, 0]
        ax1.errorbar(time_stats['times'], time_stats['mean_areas'],
                     yerr=time_stats['std_areas'],
                     marker='o', linewidth=2, capsize=5, label='Mean ± Std')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Cell Area (pixels²)')
        ax1.set_title('Area Distribution Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Aspect ratio evolution
        ax2 = axes[0, 1]
        ax2.errorbar(time_stats['times'], time_stats['mean_ars'],
                     yerr=time_stats['std_ars'],
                     marker='s', linewidth=2, capsize=5, label='Mean ± Std', color='red')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Aspect Ratio')
        ax2.set_title('Aspect Ratio Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. Flow alignment evolution (FIXED)
        ax3 = axes[1, 0]
        ax3.errorbar(time_stats['times'], time_stats['mean_orients'],
                     yerr=time_stats['std_orients'],
                     marker='^', linewidth=2, capsize=5, label='Mean ± Std', color='green')
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Flow Alignment Angle (degrees)')
        ax3.set_title('Flow Alignment Evolution\n(0° = perfect alignment, 90° = perpendicular)')
        ax3.set_ylim(0, 90)
        ax3.axhline(0, color='green', linestyle='--', alpha=0.5, label='Perfect alignment')
        ax3.axhline(45, color='orange', linestyle='--', alpha=0.5, label='45° intermediate')
        ax3.axhline(90, color='red', linestyle='--', alpha=0.5, label='Perpendicular')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Input stress and cell count
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()

        line1 = ax4.plot(time_stats['times'], time_stats['shear_stresses'],
                         'b-', linewidth=2, marker='o', label='Shear Stress')
        line2 = ax4_twin.plot(time_stats['times'], time_stats['cell_counts'],
                              'r-', linewidth=2, marker='s', label='Cell Count')

        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Shear Stress (Pa)', color='blue')
        ax4_twin.set_ylabel('Cell Count', color='red')
        ax4.set_title('Input Stress & Cell Count')
        ax4.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')

        plt.tight_layout()

        # Save plot
        if save_path is None:
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(self.config.plot_directory,
                                     f"cell_distributions_evolution_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Evolution plot saved to: {save_path}")
        print(f"   ✅ FIXED: Orientations are correctly displayed as 0-90° flow alignment")

        return fig

    def _create_snapshot_plot(self, time_data, save_path):
        """Create snapshot plots at different time points."""
        n_timepoints = len(time_data)
        fig = plt.figure(figsize=(15, 4 * n_timepoints))
        gs = GridSpec(n_timepoints, 3, figure=fig, hspace=0.3, wspace=0.3)

        colors = ['skyblue', 'lightcoral', 'lightgreen']

        for row, data in enumerate(time_data):
            time_hours = data['time_hours']
            areas = data['areas']
            aspect_ratios = data['aspect_ratios']
            orientations = data['orientations']  # Already 0-90° alignment angles
            cell_count = data['cell_count']

            # Area distribution
            ax_area = fig.add_subplot(gs[row, 0])
            ax_area.hist(areas, bins=20, color=colors[0], alpha=0.7, edgecolor='black')
            ax_area.set_title(f't={time_hours:.1f}h: Area Distribution\n(n={cell_count})', fontsize=10)
            ax_area.set_xlabel('Area (pixels²)', fontsize=9)
            ax_area.set_ylabel('Frequency', fontsize=9)
            ax_area.grid(True, alpha=0.3)

            mean_area = np.mean(areas)
            ax_area.axvline(mean_area, color='red', linestyle='--', alpha=0.8,
                            label=f'Mean: {mean_area:.0f}')
            ax_area.legend(fontsize=8)

            # Aspect ratio distribution
            ax_ar = fig.add_subplot(gs[row, 1])
            ax_ar.hist(aspect_ratios, bins=20, color=colors[1], alpha=0.7, edgecolor='black')
            ax_ar.set_title(f't={time_hours:.1f}h: Aspect Ratio Distribution', fontsize=10)
            ax_ar.set_xlabel('Aspect Ratio', fontsize=9)
            ax_ar.set_ylabel('Frequency', fontsize=9)
            ax_ar.grid(True, alpha=0.3)

            mean_ar = np.mean(aspect_ratios)
            ax_ar.axvline(mean_ar, color='red', linestyle='--', alpha=0.8,
                          label=f'Mean: {mean_ar:.1f}')
            ax_ar.legend(fontsize=8)

            # Flow alignment distribution (FIXED - no double conversion)
            ax_orient = fig.add_subplot(gs[row, 2])
            ax_orient.hist(orientations, bins=20, color=colors[2], alpha=0.7, edgecolor='black',
                           range=(0, 90))
            ax_orient.set_title(f't={time_hours:.1f}h: Flow Alignment (FIXED)', fontsize=10)
            ax_orient.set_xlabel('Alignment Angle (degrees)', fontsize=9)
            ax_orient.set_ylabel('Frequency', fontsize=9)
            ax_orient.grid(True, alpha=0.3)
            ax_orient.set_xlim(0, 90)

            # Add alignment references
            ax_orient.axvline(0, color='green', linestyle='-', alpha=0.8, linewidth=2)
            ax_orient.axvline(90, color='red', linestyle='-', alpha=0.8, linewidth=2)

            mean_orient = np.mean(orientations)
            ax_orient.axvline(mean_orient, color='red', linestyle='--', alpha=0.8,
                              label=f'Mean: {mean_orient:.1f}°')
            ax_orient.legend(fontsize=8)

            # Add info text
            info_text = f'Shear: {data["shear_stress"]:.2f} Pa'
            ax_area.text(0.02, 0.98, info_text, transform=ax_area.transAxes,
                         fontsize=8, verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        fig.suptitle('Cell Property Distributions at Multiple Time Points\n(FIXED: Correct Orientation Handling)',
                     fontsize=16, y=0.98)

        # Save plot
        if save_path is None:
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(self.config.plot_directory,
                                     f"cell_distributions_snapshots_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Snapshot plot saved to: {save_path}")

        return fig

    def create_distribution_animation(self, simulator, save_path=None, fps=2, max_timepoints=20):
        """
        Create animated cell distributions with FIXED orientation handling.
        """
        if not simulator.history:
            print("No simulation history available")
            return None

        print("Creating FIXED distribution animation...")

        # Select timepoints
        total_points = len(simulator.history)
        step = max(1, total_points // max_timepoints)
        selected_indices = list(range(0, total_points, step))
        if selected_indices[-1] != total_points - 1:
            selected_indices.append(total_points - 1)

        # Extract data
        animation_data = []
        for idx in selected_indices:
            state = simulator.history[idx]
            if 'cell_properties' not in state:
                continue

            cell_props = state['cell_properties']

            # FIXED: orientations are already 0-90° flow alignment angles
            animation_data.append({
                'time_hours': state['time'] / 60,
                'shear_stress': state.get('input_value', 0),
                'areas': cell_props['areas'],
                'aspect_ratios': cell_props['aspect_ratios'],
                'orientations': cell_props['orientations'],  # No conversion!
                'cell_count': len(cell_props['areas'])
            })

        if not animation_data:
            print("No valid data for animation")
            return None

        # Calculate ranges for consistent axes
        all_areas = [area for data in animation_data for area in data['areas']]
        all_ars = [ar for data in animation_data for ar in data['aspect_ratios']]

        area_range = (min(all_areas) * 0.9, max(all_areas) * 1.1) if all_areas else (0, 1000)
        ar_range = (min(all_ars) * 0.9, max(all_ars) * 1.1) if all_ars else (1, 3)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('FIXED Cell Property Distributions Over Time', fontsize=16)

        # Time text
        time_text = fig.text(0.02, 0.95, '', fontsize=12,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        def update_frame(frame_idx):
            for ax in axes:
                ax.clear()

            data = animation_data[frame_idx]

            # Area distribution
            axes[0].hist(data['areas'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
            axes[0].set_title('Area Distribution')
            axes[0].set_xlabel('Area (pixels²)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_xlim(area_range)
            axes[0].grid(True, alpha=0.3)

            if data['areas']:
                mean_area = np.mean(data['areas'])
                axes[0].axvline(mean_area, color='red', linestyle='--', alpha=0.8,
                                label=f'Mean: {mean_area:.0f}')
                axes[0].legend()

            # Aspect ratio distribution
            axes[1].hist(data['aspect_ratios'], bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
            axes[1].set_title('Aspect Ratio Distribution')
            axes[1].set_xlabel('Aspect Ratio')
            axes[1].set_ylabel('Frequency')
            axes[1].set_xlim(ar_range)
            axes[1].grid(True, alpha=0.3)

            if data['aspect_ratios']:
                mean_ar = np.mean(data['aspect_ratios'])
                axes[1].axvline(mean_ar, color='red', linestyle='--', alpha=0.8,
                                label=f'Mean: {mean_ar:.1f}')
                axes[1].legend()

            # FIXED Flow alignment distribution
            axes[2].hist(data['orientations'], bins=20, color='lightgreen', alpha=0.7,
                         edgecolor='black', range=(0, 90))
            axes[2].set_title('Flow Alignment (FIXED)')
            axes[2].set_xlabel('Alignment Angle (degrees)')
            axes[2].set_ylabel('Frequency')
            axes[2].set_xlim(0, 90)
            axes[2].grid(True, alpha=0.3)

            # Reference lines
            axes[2].axvline(0, color='green', linestyle='-', alpha=0.8, linewidth=2, label='Perfect')
            axes[2].axvline(90, color='red', linestyle='-', alpha=0.8, linewidth=2, label='Perpendicular')

            if data['orientations']:
                mean_orient = np.mean(data['orientations'])
                axes[2].axvline(mean_orient, color='red', linestyle='--', alpha=0.8,
                                label=f'Mean: {mean_orient:.1f}°')
                axes[2].legend()

            # Update time text
            time_str = (f"t = {data['time_hours']:.1f}h\n"
                        f"Shear: {data['shear_stress']:.2f} Pa\n"
                        f"Cells: {data['cell_count']}")
            time_text.set_text(time_str)

            return axes

        # Create animation
        ani = animation.FuncAnimation(fig, update_frame, frames=len(animation_data),
                                      interval=1000 / fps, repeat=True, blit=False)

        # Save animation
        if save_path is None:
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(self.config.plot_directory,
                                     f"FIXED_distribution_animation_{timestamp}.mp4")

        try:
            if 'ffmpeg' in animation.writers.list():
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='Fixed Endothelial Simulation'), bitrate=1800)
                ani.save(save_path, writer=writer, dpi=150)
                print(f"✅ FIXED animation saved to: {save_path}")
            else:
                print("❌ ffmpeg not available")
        except Exception as e:
            print(f"❌ Error saving animation: {e}")

        return ani

    def plot_polar_cell_distribution(self, simulator, save_path=None):
        """
        Polar plot showing cell orientations (which direction cells are pointing).
        """
        if not simulator.grid.cells:
            print("No cells to plot")
            return None

        # Extract cell orientations and colors
        orientations = []
        colors = []

        for cell in simulator.grid.cells.values():
            # Get cell orientation (direction cell is pointing)
            orientation = cell.actual_orientation
            orientations.append(orientation)

            # Set color based on cell type
            if not cell.is_senescent:
                colors.append('green')
            elif cell.senescence_cause == 'telomere':
                colors.append('red')
            else:
                colors.append('blue')

        # Create polar histogram of orientations
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Convert to degrees for histogram
        orientations_deg = np.degrees(orientations)

        # Create histogram bins (36 bins = 10 degrees each)
        bins = np.linspace(-180, 180, 37)
        hist, bin_edges = np.histogram(orientations_deg, bins=bins)

        # Convert bin centers back to radians for polar plot
        bin_centers = np.radians((bin_edges[:-1] + bin_edges[1:]) / 2)
        width = np.radians(10)  # 10 degree width

        # Create bar chart
        bars = ax.bar(bin_centers, hist, width=width, alpha=0.7,
                      color='skyblue', edgecolor='black')

        # Add flow direction reference (0° = aligned with flow)
        ax.axvline(0, color='red', linewidth=3, alpha=0.8, label='Flow Direction')

        # Customize plot
        ax.set_title(f'Cell Orientations\n{len(orientations)} cells',
                     fontsize=14, pad=20)
        ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1))
        ax.grid(True, alpha=0.3)

        # Save if requested
        if save_path is not None:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        return fig

    def create_polar_animation(self, simulator, save_path=None, fps=2, show_comparison=True):
        """
        Create MP4 animation of cell orientations evolving over time.
        FIXED: Shows 0-90° flow alignment angles only.

        Parameters:
            simulator: Simulator object with history
            save_path: Path to save MP4
            fps: Frames per second
            show_comparison: If True, shows both target (blue) and actual (red) orientations
        """
        import matplotlib.animation as animation

        if not simulator.history:
            print("No simulation history for animation")
            return None

        # Extract orientation data for each time point
        time_data = []
        history_indices = list(range(0, len(simulator.history), 10))
        if history_indices[-1] != len(simulator.history) - 1:
            history_indices.append(len(simulator.history) - 1)

        for hist_idx in history_indices:
            state = simulator.history[hist_idx]
            if 'cell_properties' not in state:
                continue

            cell_props = state['cell_properties']

            # Get BOTH target and actual orientations
            target_orientations = None
            actual_orientations = None

            if 'target_orientations_degrees' in cell_props:
                target_orientations = cell_props['target_orientations_degrees']
            elif 'target_orientations' in cell_props:
                target_orientations = [np.degrees(a) for a in cell_props['target_orientations']]

            if 'orientations' in cell_props:
                actual_orientations = cell_props['orientations']

            if not target_orientations and not actual_orientations:
                continue

            # FIXED: Convert to flow alignment angles (0-90°)
            if target_orientations and show_comparison:
                # Convert target orientations to flow alignment (0-90°)
                target_orientations = [min(abs(angle) % 180, 180 - abs(angle) % 180)
                                       for angle in target_orientations]

            if actual_orientations:
                # Convert actual orientations to flow alignment (0-90°)
                actual_orientations = [min(abs(angle) % 180, 180 - abs(angle) % 180)
                                       for angle in actual_orientations]

            # FIXED: Create histograms for 0-90° range only
            bins = np.linspace(0, 90, 20)  # 0-90° 20 bins
            data_entry = {
                'hour': state['time'] / 60,
                'shear_stress': state.get('input_value', 0),
                'time_minutes': state['time']
            }

            if target_orientations and show_comparison:
                target_hist, _ = np.histogram(target_orientations, bins=bins)
                target_centers = np.radians((bins[:-1] + bins[1:]) / 2)
                data_entry.update({
                    'target_hist': target_hist,
                    'target_centers': target_centers,
                    'target_mean': np.mean(target_orientations),
                    'has_target': True
                })
            else:
                data_entry['has_target'] = False

            if actual_orientations:
                actual_hist, _ = np.histogram(actual_orientations, bins=bins)
                actual_centers = np.radians((bins[:-1] + bins[1:]) / 2)
                data_entry.update({
                    'actual_hist': actual_hist,
                    'actual_centers': actual_centers,
                    'actual_mean': np.mean(actual_orientations),
                    'has_actual': True
                })
            else:
                data_entry['has_actual'] = False

            # Calculate deviation if both available
            if target_orientations and actual_orientations and len(target_orientations) == len(actual_orientations):
                deviation = np.mean([abs(t - a) for t, a in zip(target_orientations, actual_orientations)])
                data_entry['deviation'] = deviation
            else:
                data_entry['deviation'] = None

            time_data.append(data_entry)

        if not time_data:
            print("No valid orientation data found")
            return None

        print(f"Creating {'comparison ' if show_comparison else ''}animation with {len(time_data)} frames...")

        # Create animation
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Calculate max for consistent scale
        all_hists = []
        for data in time_data:
            if data.get('has_target'): all_hists.extend(data['target_hist'])
            if data.get('has_actual'): all_hists.extend(data['actual_hist'])
        max_hist = max(all_hists) * 1.1 if all_hists else 10

        def animate(frame):
            data = time_data[frame]
            ax.clear()

            # FIXED: Restrict to 0-90° quadrant
            ax.set_thetamax(90)  # Maximum angle: 90°
            ax.set_thetamin(0)  # Minimum angle: 0°
            ax.set_thetagrids([0, 15, 30, 45, 60, 75, 90],
                              ['0°\n(Flow)', '15°', '30°', '45°', '60°', '75°', '90°\n(Perp)'])

            ax.set_ylim(0, max_hist)
            ax.grid(True, alpha=0.3)
            ax.set_theta_zero_location('E')  # 0° at right
            ax.set_theta_direction(1)  # Counterclockwise

            width = np.radians(10)  # FIXED: Narrower bars for 0-90° range

            # Plot target orientations (blue, behind)
            if data.get('has_target') and show_comparison:
                ax.bar(data['target_centers'], data['target_hist'], width=width,
                       alpha=0.6, color='lightblue', edgecolor='blue',
                       label=f'Target (mean: {data["target_mean"]:.1f}°)')

            # Plot actual orientations (red, in front)
            if data.get('has_actual'):
                alpha = 0.8 if show_comparison else 0.7
                color = 'lightcoral' if show_comparison else 'skyblue'
                edge_color = 'darkred' if show_comparison else 'black'
                label = f'Actual (mean: {data["actual_mean"]:.1f}°)' if show_comparison else f'Orientations (mean: {data["actual_mean"]:.1f}°)'

                ax.bar(data['actual_centers'], data['actual_hist'], width=width,
                       alpha=alpha, color=color, edgecolor=edge_color, label=label)

            # Add flow direction reference
            ax.axvline(0, color='green', linewidth=3, alpha=0.8, label='Flow Direction')

            # Title with comparison info
            if show_comparison and data['deviation'] is not None:
                title = f'Hour {data["hour"]:.1f}: Target vs Actual Flow Alignment\nDeviation: {data["deviation"]:.1f}° | Shear: {data["shear_stress"]:.2f} Pa'
            else:
                title = f'Hour {data["hour"]:.1f}: Cell Flow Alignment\nShear: {data["shear_stress"]:.2f} Pa'

            ax.set_title(title, fontsize=12, pad=20)
            ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=9)

            return ax.patches + ax.lines

        ani = animation.FuncAnimation(fig, animate, frames=len(time_data),
                                      interval=1000 / fps, blit=False, repeat=True)

        # Save animation
        if save_path is None:
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            suffix = "_flow_alignment" if show_comparison else "_flow_alignment_single"
            save_path = os.path.join(self.config.plot_directory, f"orientation_animation{suffix}_{timestamp}.mp4")

        try:
            if 'ffmpeg' in animation.writers.list():
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='Endothelial Simulation'), bitrate=1800)
                ani.save(save_path, writer=writer, dpi=150)
                print(f"✅ Animation saved to: {save_path}")
                print(f"   Shows flow alignment angles (0-90°)")
                if show_comparison:
                    print(f"   Blue = Target alignment, Red = Actual alignment")
            else:
                print("❌ ffmpeg not available")
        except Exception as e:
            print(f"❌ Error saving animation: {e}")

        return ani

    def plot_configuration_mosaic(self, grid, configuration_data, save_path=None, title_suffix=""):
        """
        Plot a single configuration's mosaic using the existing cell visualization.

        Parameters:
            grid: Grid object
            configuration_data: Single configuration dict from generate_multiple_initial_configurations
            save_path: Path to save the plot
            title_suffix: Additional text for title

        Returns:
            Matplotlib figure
        """
        # Store original state
        original_cells = grid.cells.copy()
        original_seeds = grid.cell_seeds.copy()
        original_territories = grid.territory_map.copy()

        try:
            # Reconstruct this configuration
            grid._reconstruct_configuration(configuration_data['cell_data'])

            # Create a mock simulator object for plot_cell_visualization
            class MockSimulator:
                def __init__(self, grid, config_data):
                    self.grid = grid
                    self.time = 0.0
                    self.config = grid.config
                    self.input_pattern = {'value': 0.0}
                    self._config_energy = config_data['energy']
                    self._config_fitness = config_data['fitness']
                    self._config_idx = config_data['config_idx']

            mock_sim = MockSimulator(grid, configuration_data)

            # Use existing visualization function
            fig = self.plot_cell_visualization(mock_sim, save_path=None, show_boundaries=True)

            # Update title to include configuration info
            current_title = fig._suptitle.get_text() if fig._suptitle else "Cell Mosaic"
            new_title = (f"{current_title} - Config #{configuration_data['config_idx'] + 1}\n"
                         f"Energy: {configuration_data['energy']:.3f} | "
                         f"Fitness: {configuration_data['fitness']:.3f}{title_suffix}")
            fig.suptitle(new_title, fontsize=14)

            # Save if requested
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            return fig

        finally:
            # Always restore original state
            grid.cells = original_cells
            grid.cell_seeds = original_seeds
            grid.territory_map = original_territories
            grid._update_voronoi_tessellation()

    def plot_configuration_mosaics_grid(self, grid, configurations_data, save_path=None,
                                        show_top_n=6, cols=3):
        """
        Plot multiple configuration mosaics in a grid layout.

        Parameters:
            grid: Grid object
            configurations_data: Result from generate_multiple_initial_configurations
            save_path: Path to save the plot
            show_top_n: Number of configurations to show
            cols: Number of columns in grid

        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt

        configurations = configurations_data['all_configurations']
        best_idx = configurations_data['selected_idx']

        # Sort by energy and take top N
        sorted_configs = sorted(configurations, key=lambda x: x['energy'])[:show_top_n]

        # Calculate grid layout
        rows = (len(sorted_configs) + cols - 1) // cols

        # Store original state
        original_cells = grid.cells.copy()
        original_seeds = grid.cell_seeds.copy()
        original_territories = grid.territory_map.copy()

        try:
            # Create figure
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
            if len(sorted_configs) == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes] if cols == 1 else list(axes)
            else:
                axes = axes.flatten()

            for i, config in enumerate(sorted_configs):
                if i >= len(axes):
                    break

                ax = axes[i]

                # Reconstruct configuration
                grid._reconstruct_configuration(config['cell_data'])

                # Get display territories using existing grid method
                display_territories = grid.get_display_territories()

                # Plot using similar logic to plot_cell_visualization but simplified
                ax.set_xlim(0, grid.width)
                ax.set_ylim(0, grid.height)
                ax.set_aspect('equal')

                # Plot cell territories
                for cell_id, cell in grid.cells.items():
                    if cell_id not in display_territories:
                        continue

                    display_pixels = display_territories[cell_id]
                    if not display_pixels:
                        continue

                    # Color determination (same as plot_cell_visualization)
                    if not cell.is_senescent:
                        color = 'green'
                        alpha = 0.6
                    else:
                        growth_factor = getattr(cell, 'senescent_growth_factor', 1.0)
                        if cell.senescence_cause == 'telomere':
                            color = '#DC143C' if growth_factor <= 1.5 else '#B22222'
                        else:
                            color = '#4169E1' if growth_factor <= 1.5 else '#000080'
                        alpha = 0.7 + 0.1 * min(1.0, (growth_factor - 1.0) / 2.0)

                    # Plot territory (simplified)
                    if len(display_pixels) > 10:
                        try:
                            from scipy.spatial import ConvexHull
                            points = np.array(display_pixels)
                            if len(points) > 100:  # Sample for performance
                                indices = np.random.choice(len(points), 100, replace=False)
                                points = points[indices]
                            hull = ConvexHull(points)
                            hull_points = points[hull.vertices]

                            polygon = plt.Polygon(hull_points, facecolor=color, alpha=alpha,
                                                  edgecolor='black', linewidth=0.5)
                            ax.add_patch(polygon)
                        except:
                            # Fallback to scatter
                            points = np.array(display_pixels[:100])  # Limit for performance
                            ax.scatter(points[:, 0], points[:, 1], c=color, alpha=alpha, s=1, marker='s')

                # Title and formatting
                is_selected = config['config_idx'] == best_idx
                title_prefix = "- SELECTED -\n" if is_selected else ""
                title = (f"{title_prefix}Config #{config['config_idx'] + 1}\n"
                         f"Energy: {config['energy']:.3f}")

                ax.set_title(title, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

                # Highlight selected configuration
                if is_selected:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('gold')
                        spine.set_linewidth(3)

            # Hide unused subplots
            for j in range(len(sorted_configs), len(axes)):
                axes[j].set_visible(False)

            # Main title
            fig.suptitle(f'Configuration Mosaics (Top {len(sorted_configs)} by Energy)', fontsize=16)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Configuration mosaics saved to: {save_path}")
                plt.close()

            return fig

        finally:
            # Restore original state
            grid.cells = original_cells
            grid.cell_seeds = original_seeds
            grid.territory_map = original_territories
            grid._update_voronoi_tessellation()

    def create_mosaic_animation(self, simulator, save_path=None, fps=2, max_frames=30, show_info=True):
        """
        Create an animation showing REAL individual cells evolving over time.
        FIXED: Now shows actual cell lifecycles with proper senescence transitions.
        """
        import matplotlib.animation as animation
        import numpy as np
        from matplotlib.patches import Polygon

        # Check if we have frame data (individual cell tracking)
        if not simulator.frame_data:
            print("❌ No frame data available. Make sure recording is enabled!")
            print("   Set config.create_animations = True before running simulation")
            return None

        print(f"✅ Creating REAL mosaic animation from {len(simulator.frame_data)} frames...")

        # Select frames to animate (skip some if too many)
        total_frames = len(simulator.frame_data)
        if total_frames > max_frames:
            step = total_frames // max_frames
            selected_indices = list(range(0, total_frames, step))
            if selected_indices[-1] != total_frames - 1:
                selected_indices.append(total_frames - 1)
        else:
            selected_indices = list(range(total_frames))

        print(f"   Animating {len(selected_indices)} frames showing REAL cell transitions...")

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))

        # Info text overlay
        if show_info:
            info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        def animate_frame(frame_idx):
            """
            Animate one frame using REAL historical cell data.
            NO MORE FAKE PROPORTIONAL ASSIGNMENTS!
            """
            # Clear the plot
            ax.clear()

            # Get REAL historical data for this specific time point
            frame = simulator.frame_data[selected_indices[frame_idx]]

            print(f"   Frame {frame_idx}: t={frame['time']:.1f}, cells={len(frame['cells'])}")

            # Set up the plot area
            ax.set_xlim(0, simulator.grid.width)
            ax.set_ylim(0, simulator.grid.height)
            ax.set_aspect('equal')

            # Count cells by type (for display info)
            cell_counts = {'healthy': 0, 'senescent_tel': 0, 'senescent_stress': 0}

            # Plot each REAL individual cell from the historical data
            for cell_data in frame['cells']:
                cell_id = cell_data['cell_id']

                # Get this cell's REAL territory at this time point
                territory = cell_data.get('territory', [])
                if not territory or len(territory) < 3:
                    continue  # Skip cells with no territory

                # Use the cell's REAL senescence status at this time
                # (Not fake assignment based on population ratios!)
                is_senescent = cell_data['is_senescent']
                senescence_cause = cell_data.get('senescence_cause')

                # Determine color based on REAL cell state
                if not is_senescent:
                    color = 'lightgreen'
                    alpha = 0.7
                    edge_color = 'darkgreen'
                    cell_counts['healthy'] += 1

                elif senescence_cause == 'telomere':
                    color = '#DC143C'  # Crimson red
                    alpha = 0.8
                    edge_color = 'darkred'
                    cell_counts['senescent_tel'] += 1

                else:  # stress senescence
                    color = '#4169E1'  # Royal blue
                    alpha = 0.8
                    edge_color = 'darkblue'
                    cell_counts['senescent_stress'] += 1

                # Draw the cell's ACTUAL territory from this time point
                if len(territory) > 10:
                    try:
                        # Create a clean polygon from the territory
                        from scipy.spatial import ConvexHull
                        points = np.array(territory)

                        # If territory is huge, sample it for performance
                        if len(points) > 200:
                            indices = np.random.choice(len(points), 200, replace=False)
                            points = points[indices]

                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]

                        # Create and add the polygon
                        polygon = Polygon(hull_points,
                                          facecolor=color,
                                          alpha=alpha,
                                          edgecolor=edge_color,
                                          linewidth=0.5)
                        ax.add_patch(polygon)

                        # ADD THIS: Display cell ID at the centroid
                        centroid_x = np.mean([p[0] for p in territory])
                        centroid_y = np.mean([p[1] for p in territory])

                        ax.text(centroid_x, centroid_y, str(cell_id),
                                fontsize=8, ha='center', va='center',
                                color='white', fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

                    except Exception as e:
                        # Fallback: scatter plot if polygon creation fails
                        points = np.array(territory[:100])  # Limit for performance
                        ax.scatter(points[:, 0], points[:, 1],
                                   c=color, alpha=alpha, s=1, marker='s')

                # Mark the cell center
                pos = cell_data['position']
                ax.plot(pos[0], pos[1], 'ko', markersize=1.5)

            # Create legend
            healthy_patch = plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.7, label='Healthy')
            tel_patch = plt.Rectangle((0, 0), 1, 1, facecolor='#DC143C', alpha=0.8, label='Senescent (Telomere)')
            stress_patch = plt.Rectangle((0, 0), 1, 1, facecolor='#4169E1', alpha=0.8, label='Senescent (Stress)')
            ax.legend(handles=[healthy_patch, tel_patch, stress_patch], loc='upper right', fontsize=10)

            # Calculate time for display
            time_hours = frame['time'] / 60 if self.config.time_unit == "minutes" else frame['time']
            time_unit = "hours" if self.config.time_unit == "minutes" else self.config.time_unit

            # Set title
            ax.set_title(f'Real Endothelial Cell Mosaic Evolution\n'
                         f'Time: {time_hours:.1f} {time_unit} | Individual Cell Tracking',
                         fontsize=14)

            # Update info text
            if show_info:
                total_cells = sum(cell_counts.values())
                info_str = (f"Time: {time_hours:.1f} {time_unit}\n"
                            f"Shear Stress: {frame['input_value']:.2f} Pa\n"
                            f"Healthy: {cell_counts['healthy']}\n"
                            f"Sen (Telomere): {cell_counts['senescent_tel']}\n"
                            f"Sen (Stress): {cell_counts['senescent_stress']}\n"
                            f"Total: {total_cells}\n"
                            f"🎯 Real Individual Cells!")
                info_text.set_text(info_str)

            return []

        # Create the animation
        print("   Creating animation object...")
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(selected_indices),
                                       interval=1000 // fps, repeat=True, blit=False)

        # Generate save path if not provided
        if save_path is None:
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(self.config.plot_directory, f"REAL_mosaic_animation_{timestamp}.mp4")

        # Save the animation
        try:
            print(f"   Saving animation to: {save_path}")
            if 'ffmpeg' in animation.writers.list():
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='Real Mosaic Animation'), bitrate=1800)
                anim.save(save_path, writer=writer, dpi=150)
                print(f"✅ REAL mosaic animation saved successfully!")
                print(f"   📊 {len(selected_indices)} frames at {fps} fps")
                print(f"   🧬 Shows actual individual cell lifecycles")
                print(f"   🎯 No more instant senescence transitions!")
            else:
                print("⚠️  ffmpeg not available, saving as GIF...")
                anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=fps)
                print(f"✅ Animation saved as GIF")
        except Exception as e:
            print(f"❌ Error saving animation: {e}")
            print("   Try installing ffmpeg: pip install ffmpeg-python")

        return anim

    # BONUS: Function to verify your data is ready
    def check_animation_readiness(simulator):
        """
        Check if your simulator has the data needed for real animation.
        Call this before creating animation to make sure everything is ready.
        """
        print("🔍 Checking animation readiness...")

        # Check 1: Frame data exists
        if not hasattr(simulator, 'frame_data') or not simulator.frame_data:
            print("❌ No frame data found!")
            print("   Solution: Set config.create_animations = True before running simulation")
            return False

        # Check 2: Frame data has cells
        sample_frame = simulator.frame_data[0]
        if 'cells' not in sample_frame or not sample_frame['cells']:
            print("❌ Frame data exists but has no cell information!")
            return False

        # Check 3: Cells have territory data
        sample_cell = sample_frame['cells'][0]
        if 'territory' not in sample_cell:
            print("❌ Cell data exists but missing territory information!")
            print("   Solution: Add territory data to _record_frame() method")
            return False

        # Check 4: Territory data is not empty
        if not sample_cell['territory']:
            print("⚠️  Territory data exists but is empty - this might cause issues")

        print(f"✅ Animation ready! Found {len(simulator.frame_data)} frames with cell data")
        print(f"   📊 Each frame has ~{len(sample_frame['cells'])} cells")
        print(f"   🧬 Territory data: {'✅ Available' if sample_cell.get('territory') else '❌ Missing'}")

        return True

    def create_all_plots(self, simulator, prefix=None):
        """
        Create all available plots for the simulation.
        MODIFIED: Now automatically includes energy analysis.

        Parameters:
            simulator: Simulator object with current state
            history: List of state dictionaries (default: from simulator)
            prefix: Prefix for filenames (default: auto-generated timestamp)

        Returns:
            List of created figure objects
        """

        # Use simulator history
        history = simulator.history

        # Check if simulator has history
        if not history:
            print("⚠️  No simulation history available")
            return []

        # Use timestamp prefix if not provided
        if prefix is None:
            import time
            prefix = time.strftime("%Y%m%d-%H%M%S")

        # Create output directory
        os.makedirs(self.config.plot_directory, exist_ok=True)

        # Create figures
        figures = []

        print(f"📊 Creating all plots with prefix: {prefix}")
        # Use simulator history if not provided
        if history is None:
            history = simulator.history

        # Use timestamp prefix if not provided
        if prefix is None:
            import time
            prefix = time.strftime("%Y%m%d-%H%M%S")

        # Create output directory
        os.makedirs(self.config.plot_directory, exist_ok=True)

        # Create figures
        figures = []

        # Cell population plot
        pop_fig = self.plot_cell_population(
            history,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_population.png")
        )
        figures.append(pop_fig)

        # Input pattern plot
        input_fig = self.plot_input_pattern(
            history,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_input.png")
        )
        figures.append(input_fig)

        # Spatial metrics plot (if available)
        spatial_fig = self.plot_spatial_metrics(
            history,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_spatial.png")
        )
        if spatial_fig:
            figures.append(spatial_fig)

        # Mosaic metrics plot (if available)
        mosaic_fig = self.plot_mosaic_metrics(
            history,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_mosaic.png")
        )
        if mosaic_fig:
            figures.append(mosaic_fig)

        # Senescent growth metrics plot (if available)
        growth_fig = self.plot_senescent_growth_metrics(
            history,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_senescent_growth.png")
        )
        if growth_fig:
            figures.append(growth_fig)

        # Cell visualization
        cell_vis_fig = self.plot_cell_visualization(
            simulator,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_cells.png")
        )
        figures.append(cell_vis_fig)

        # Cell distribution plots
        evolution_fig = self.plot_cell_distributions(
            simulator,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_distributions_evolution.png"),
            show_evolution=True
        )
        if evolution_fig:
            figures.append(evolution_fig)

        snapshot_fig = self.plot_cell_distributions(
            simulator,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_distributions_snapshots.png"),
            show_evolution=False,
            time_points='auto'
        )
        if snapshot_fig:
            figures.append(snapshot_fig)

        # Polar distribution plot
        polar_fig = self.plot_polar_cell_distribution(
            simulator,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_polar.png")
        )
        if polar_fig:
            figures.append(polar_fig)

        if hasattr(simulator, '_config_results') and simulator._config_results:
            try:
                print("📊 Adding configuration comparison plots...")

                # Energy landscape plot
                energy_landscape_fig = self.create_energy_landscape_plot(
                    simulator.grid,
                    simulator._config_results,
                    save_path=os.path.join(self.config.plot_directory, f"{prefix}_energy_landscape.png")
                )
                if energy_landscape_fig:
                    figures.append(energy_landscape_fig)
                    print("✅ Energy landscape plot created")

                print("🎨 Creating configuration mosaic plots...")

                mosaic_grid_fig = self.plot_configuration_mosaics_grid(
                    simulator.grid,
                    simulator._config_results,
                    save_path=os.path.join(self.config.plot_directory, f"{prefix}_config_mosaics.png"),
                    show_top_n=6
                )
                if mosaic_grid_fig:
                    figures.append(mosaic_grid_fig)
                    print("✅ Configuration mosaics grid created")

                best_config = simulator._config_results['best_config']
                selected_mosaic_fig = self.plot_configuration_mosaic(
                    simulator.grid,
                    best_config,
                    save_path=os.path.join(self.config.plot_directory, f"{prefix}_selected_config_mosaic.png"),
                    title_suffix=" (SELECTED)"
                )
                if selected_mosaic_fig:
                    figures.append(selected_mosaic_fig)
                    print("✅ Selected configuration detailed mosaic created")


            except Exception as e:
                print(f"⚠️  Configuration visualization skipped: {e}")

        if self.config.create_animations and hasattr(simulator, 'frame_data') and simulator.frame_data:
            print("🎬 Creating animations...")
            try:
                animations.create_detailed_cell_animation(self, simulator.frame_data, simulator)
                animations.create_mosaic_animation(self, simulator)
                animations.create_polar_animation(self, simulator)
            except Exception as e:
                print(f"⚠️ Could not create animations: {e}")

        return figures

    def _plot_energy_analysis(self, simulator, save_path=None):
        """
        FIXED: Memory-safe energy analysis plot that handles large cell counts.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec

            print(f"Creating energy analysis for {len(simulator.grid.cells)} cells...")

            # Get basic energy information safely
            try:
                if hasattr(simulator.grid, 'get_energy_summary'):
                    summary = simulator.grid.get_energy_summary()
                else:
                    print("Energy summary not available")
                    return None

                if 'error' in summary:
                    print(f"Energy tracking not available: {summary['error']}")
                    return None
            except Exception as e:
                print(f"Could not get energy summary: {e}")
                return None

            # Create a simple, safe energy plot
            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

            # 1. Basic energy information (text-based to avoid large calculations)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.axis('off')

            # Safely get basic statistics
            total_cells = len(simulator.grid.cells)
            biological_energy = getattr(simulator.grid, 'calculate_biological_energy', lambda: 0)()

            # Limit precision to avoid overflow
            energy_per_cell = biological_energy / max(1, total_cells) if total_cells > 0 else 0
            energy_per_cell = min(energy_per_cell, 1e6)  # Cap to prevent overflow

            stats_text = f"""Energy Statistics

    Total Cells: {total_cells:,}
    Bio Energy: {biological_energy:.4f}
    Energy/Cell: {energy_per_cell:.6f}

    Grid Size: {simulator.grid.width}×{simulator.grid.height}
    Time: {simulator.time:.1f} min"""

            ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            ax1.set_title('Energy Overview', fontsize=12)

            # 2. Cell distribution (safe histogram)
            ax2 = fig.add_subplot(gs[0, 1])
            try:
                # Sample cells if there are too many to avoid memory issues
                cell_sample = list(simulator.grid.cells.values())
                if len(cell_sample) > 1000:  # Limit sample size
                    import random
                    cell_sample = random.sample(cell_sample, 1000)

                if cell_sample:
                    areas = [cell.area for cell in cell_sample if hasattr(cell, 'area') and cell.area < 1e6]
                    if areas:
                        ax2.hist(areas, bins=min(30, len(areas) // 10), alpha=0.7, color='skyblue', edgecolor='black')
                        ax2.set_xlabel('Cell Area (pixels²)')
                        ax2.set_ylabel('Frequency')
                        ax2.set_title(f'Area Distribution\n(Sample of {len(areas)} cells)')
                    else:
                        ax2.text(0.5, 0.5, 'No area data available', ha='center', va='center', transform=ax2.transAxes)
                        ax2.set_title('Area Distribution')
                else:
                    ax2.text(0.5, 0.5, 'No cell data', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Area Distribution')
            except Exception as e:
                ax2.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Area Distribution')

            # 3. Senescence statistics (safe counting)
            ax3 = fig.add_subplot(gs[1, 0])
            try:
                healthy_count = sum(
                    1 for cell in simulator.grid.cells.values() if not getattr(cell, 'is_senescent', False))
                senescent_count = total_cells - healthy_count

                if total_cells > 0:
                    categories = ['Healthy', 'Senescent']
                    counts = [healthy_count, senescent_count]
                    colors = ['green', 'red']

                    bars = ax3.bar(categories, counts, color=colors, alpha=0.7)
                    ax3.set_ylabel('Cell Count')
                    ax3.set_title('Cell Health Distribution')

                    # Add count labels on bars
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                                 f'{count:,}', ha='center', va='bottom', fontsize=10)
                else:
                    ax3.text(0.5, 0.5, 'No cells', ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Cell Health Distribution')
            except Exception as e:
                ax3.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Cell Health Distribution')

            # 4. Simulation progress
            ax4 = fig.add_subplot(gs[1, 1])
            try:
                if hasattr(simulator, 'history') and simulator.history:
                    # Safe history plotting - limit data points
                    history_sample = simulator.history[-100:] if len(simulator.history) > 100 else simulator.history
                    times = [state.get('time', 0) for state in history_sample]
                    cell_counts = [state.get('cells', 0) for state in history_sample]

                    if times and cell_counts:
                        ax4.plot(times, cell_counts, 'b-', linewidth=2, marker='o', markersize=3)
                        ax4.set_xlabel('Time (minutes)')
                        ax4.set_ylabel('Cell Count')
                        ax4.set_title('Population Evolution')
                        ax4.grid(True, alpha=0.3)
                    else:
                        ax4.text(0.5, 0.5, 'No history data', ha='center', va='center', transform=ax4.transAxes)
                        ax4.set_title('Population Evolution')
                else:
                    ax4.text(0.5, 0.5, 'No simulation history', ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Population Evolution')
            except Exception as e:
                ax4.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Population Evolution')

            # Set main title
            fig.suptitle('Biological Energy Analysis (Safe Mode)\nEndothelial Cell Mechanotransduction Simulation',
                         fontsize=14, y=0.95)

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')  # Lower DPI to save memory
                plt.close()
                print(f"Safe energy analysis saved to: {save_path}")


            return fig

        except Exception as e:
            print(f"Error creating safe energy analysis plot: {e}")
            return None

    def visualize_all_configurations(self, grid, configurations_data, save_path=None, show_top_n=None):
        """
        Create a visual comparison of all generated configurations.

        Parameters:
            grid: Grid object containing the configurations
            configurations_data: Result from generate_multiple_initial_configurations
            save_path: Path to save the visualization
            show_top_n: Show only the top N configurations (None = show all)

        Returns:
            matplotlib.Figure object
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import Polygon
        import numpy as np  # Add this import
        import time

        configurations = configurations_data['all_configurations']
        best_idx = configurations_data['selected_idx']

        # Sort configurations by energy for better visualization
        sorted_configs = sorted(configurations, key=lambda x: x['energy'])
        if show_top_n:
            sorted_configs = sorted_configs[:show_top_n]

        n_configs = len(sorted_configs)

        # Calculate grid layout
        cols = min(4, n_configs)  # Max 4 columns
        rows = (n_configs + cols - 1) // cols

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if n_configs == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else list(axes)
        else:
            axes = axes.flatten()

        # Color map for energy levels
        energies = [config['energy'] for config in sorted_configs]
        min_energy, max_energy = min(energies), max(energies)

        # Store original state
        original_cells = grid.cells.copy()
        original_seeds = grid.cell_seeds.copy()
        original_territories = grid.territory_map.copy()

        for i, config in enumerate(sorted_configs):
            if i >= len(axes):
                break

            ax = axes[i]

            # Reconstruct this configuration temporarily
            grid._reconstruct_configuration(config['cell_data'])

            # Clear axis
            ax.clear()
            ax.set_xlim(0, grid.width)
            ax.set_ylim(0, grid.height)
            ax.set_aspect('equal')

            # Get display territories
            display_territories = grid.get_display_territories()

            # Color mapping
            cell_colors = {'healthy': 'lightgreen', 'telomere': 'lightcoral', 'stress': 'lightblue'}

            # Plot each cell's territory
            for cell_id, territory_pixels in display_territories.items():
                if cell_id in grid.cells:
                    cell = grid.cells[cell_id]

                    # Determine cell color
                    if not cell.is_senescent:
                        color = cell_colors['healthy']
                    elif cell.senescence_cause == 'telomere':
                        color = cell_colors['telomere']
                    else:
                        color = cell_colors['stress']

                    # Plot territory as scatter (simplified for speed)
                    if len(territory_pixels) > 0:
                        pixels_array = np.array(territory_pixels)
                        ax.scatter(pixels_array[:, 0], pixels_array[:, 1],
                                   c=color, alpha=0.6, s=0.5, marker='s')

                    # Plot cell center
                    cx, cy = cell.position
                    ax.plot(cx, cy, 'ko', markersize=3, alpha=0.8)

            # Energy-based border color
            if max_energy > min_energy:
                energy_norm = (config['energy'] - min_energy) / (max_energy - min_energy)
                border_color = plt.cm.RdYlBu_r(energy_norm)  # Red = high energy, Blue = low energy
            else:
                border_color = 'blue'

            # Highlight best configuration
            if config['config_idx'] == best_idx:
                # Thick gold border for selected configuration
                for spine in ax.spines.values():
                    spine.set_edgecolor('gold')
                    spine.set_linewidth(4)
                title_prefix = "* SELECTED *\n"
            else:
                # Colored border based on energy
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2)
                title_prefix = ""

            # Title with key metrics
            title = (f"{title_prefix}Config #{config['config_idx'] + 1}\n"
                     f"Energy: {config['energy']:.3f}\n"
                     f"Fitness: {config['fitness']:.3f}\n"
                     f"Packing: {config['packing_efficiency']:.3f}")

            ax.set_title(title, fontsize=8, pad=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        # Add overall title
        energy_improvement = configurations_data['energy_improvement']
        improvement_pct = (energy_improvement / max_energy * 100) if max_energy > 0 else 0

        main_title = (f"Initial Configuration Comparison\n"
                      f"Tested {len(configurations)} configurations | "
                      f"Best energy: {min_energy:.3f} | "
                      f"Improvement: {improvement_pct:.1f}%")

        fig.suptitle(main_title, fontsize=14, y=0.98)

        # Add legend
        legend_elements = [
            patches.Patch(color='lightgreen', label='Healthy Cells'),
            patches.Patch(color='lightcoral', label='Telomere-Senescent'),
            patches.Patch(color='lightblue', label='Stress-Senescent'),
            patches.Patch(color='gold', label='Selected Configuration'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                       markersize=6, label='Cell Centers')
        ]

        fig.legend(handles=legend_elements, loc='lower center', ncol=5,
                   bbox_to_anchor=(0.5, 0.02), fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.08)

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Configuration comparison saved to: {save_path}")

        # Restore original state
        grid.cells = original_cells
        grid.cell_seeds = original_seeds
        grid.territory_map = original_territories
        grid._update_voronoi_tessellation()

        return fig

    def create_configuration_animation(self, grid, configurations_data, save_path=None,
                                       show_top_n=10, fps=2):
        """
        Create an animation showing the progression of configurations by energy rank.

        Parameters:
            grid: Grid object containing the configurations
            configurations_data: Result from generate_multiple_initial_configurations
            save_path: Path to save animation (should end in .mp4 or .gif)
            show_top_n: Number of top configurations to animate
            fps: Frames per second for animation
        """
        try:
            import matplotlib.animation as animation
            import matplotlib.pyplot as plt
            import numpy as np  # Add this import
            import time

            configurations = configurations_data['all_configurations']
            best_idx = configurations_data['selected_idx']

            # Sort by energy (best to worst)
            sorted_configs = sorted(configurations, key=lambda x: x['energy'])[:show_top_n]

            # Store original state
            original_cells = grid.cells.copy()
            original_seeds = grid.cell_seeds.copy()
            original_territories = grid.territory_map.copy()

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))

            def update_frame(frame_idx):
                ax.clear()
                ax.set_xlim(0, grid.width)
                ax.set_ylim(0, grid.height)
                ax.set_aspect('equal')

                if frame_idx < len(sorted_configs):
                    config = sorted_configs[frame_idx]

                    # Reconstruct configuration
                    grid._reconstruct_configuration(config['cell_data'])

                    # Get display territories
                    display_territories = grid.get_display_territories()

                    # Color mapping
                    cell_colors = {'healthy': 'lightgreen', 'telomere': 'lightcoral', 'stress': 'lightblue'}

                    # Plot each cell's territory
                    for cell_id, territory_pixels in display_territories.items():
                        if cell_id in grid.cells:
                            cell = grid.cells[cell_id]

                            # Determine cell color
                            if not cell.is_senescent:
                                color = cell_colors['healthy']
                            elif cell.senescence_cause == 'telomere':
                                color = cell_colors['telomere']
                            else:
                                color = cell_colors['stress']

                            # Plot territory
                            if len(territory_pixels) > 0:
                                pixels_array = np.array(territory_pixels)
                                ax.scatter(pixels_array[:, 0], pixels_array[:, 1],
                                           c=color, alpha=0.7, s=1, marker='s')

                            # Plot cell center
                            cx, cy = cell.position
                            ax.plot(cx, cy, 'ko', markersize=4, alpha=0.9)

                    # Title with ranking and metrics
                    rank = frame_idx + 1
                    is_selected = config['config_idx'] == best_idx
                    selected_text = "- SELECTED -" if is_selected else ""

                    title = (f"Configuration Ranking - #{rank}/{len(sorted_configs)}{selected_text}\n"
                             f"Config #{config['config_idx'] + 1} | "
                             f"Energy: {config['energy']:.3f} | "
                             f"Fitness: {config['fitness']:.3f} | "
                             f"Packing: {config['packing_efficiency']:.3f}")

                    ax.set_title(title, fontsize=14, pad=20)

                    # Color border based on selection
                    border_color = 'gold' if is_selected else 'black'
                    border_width = 4 if is_selected else 2
                    for spine in ax.spines.values():
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(border_width)

                ax.set_xticks([])
                ax.set_yticks([])

            # Create animation
            anim = animation.FuncAnimation(
                fig, update_frame, frames=len(sorted_configs),
                interval=1000 // fps, repeat=True, blit=False
            )

            # Save animation
            if save_path:
                if save_path.endswith('.gif'):
                    anim.save(save_path, writer='pillow', fps=fps)
                else:
                    if 'ffmpeg' in animation.writers.list():
                        Writer = animation.writers['ffmpeg']
                        writer = Writer(fps=fps, metadata=dict(artist='Configuration Comparison'), bitrate=1800)
                        anim.save(save_path, writer=writer)
                    else:
                        print("Warning: ffmpeg not available, saving as GIF instead")
                        gif_path = save_path.rsplit('.', 1)[0] + '.gif'
                        anim.save(gif_path, writer='pillow', fps=fps)

                print(f"🎬 Configuration animation saved to: {save_path}")

            # Restore original state
            grid.cells = original_cells
            grid.cell_seeds = original_seeds
            grid.territory_map = original_territories
            grid._update_voronoi_tessellation()

            return fig, anim

        except ImportError:
            print("Animation requires matplotlib.animation - creating static plot instead")
            return self.visualize_all_configurations(grid, configurations_data, save_path)
        except Exception as e:
            print(f"Error creating animation: {e}")
            return self.visualize_all_configurations(grid, configurations_data, save_path)

    def create_energy_landscape_plot(self, grid, configurations_data, save_path=None):
        """
        Create a detailed energy landscape visualization.

        Parameters:
            grid: Grid object (not used directly but for consistency)
            configurations_data: Result from generate_multiple_initial_configurations
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        import numpy as np

        configurations = configurations_data['all_configurations']
        best_idx = configurations_data['selected_idx']

        # Extract data
        config_indices = [c['config_idx'] for c in configurations]
        energies = [c['energy'] for c in configurations]
        fitnesses = [c['fitness'] for c in configurations]
        packing_effs = [c['packing_efficiency'] for c in configurations]

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Energy vs Configuration Index
        colors = ['gold' if i == best_idx else 'blue' for i in config_indices]
        ax1.scatter(config_indices, energies, c=colors, s=60, alpha=0.7)
        ax1.axhline(y=configurations_data['best_config']['energy'],
                    color='red', linestyle='--', alpha=0.7, label='Selected Energy')
        ax1.set_xlabel('Configuration Index')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy by Configuration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Energy Distribution
        ax2.hist(energies, bins=min(15, len(configurations) // 2), alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=configurations_data['best_config']['energy'],
                    color='red', linestyle='--', linewidth=2, label='Selected')
        ax2.set_xlabel('Energy')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Energy Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Energy vs Fitness
        scatter = ax3.scatter(energies, fitnesses, c=config_indices, cmap='viridis', s=60, alpha=0.7)
        # Highlight selected
        selected_config = configurations[best_idx]
        ax3.scatter([selected_config['energy']], [selected_config['fitness']],
                    c='red', s=120, marker='*', label='Selected', edgecolor='black', linewidth=1)
        ax3.set_xlabel('Energy')
        ax3.set_ylabel('Fitness')
        ax3.set_title('Energy vs Fitness')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Config Index')

        # 4. Multi-metric comparison
        sorted_configs = sorted(configurations, key=lambda x: x['energy'])
        ranks = list(range(1, len(sorted_configs) + 1))
        sorted_energies = [c['energy'] for c in sorted_configs]
        sorted_fitnesses = [c['fitness'] for c in sorted_configs]
        sorted_packing = [c['packing_efficiency'] for c in sorted_configs]

        ax4.plot(ranks, sorted_energies, 'b-o', label='Energy', alpha=0.7)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(ranks, sorted_fitnesses, 'g-s', label='Fitness', alpha=0.7)
        ax4_twin.plot(ranks, sorted_packing, 'r-^', label='Packing Eff.', alpha=0.7)

        # Mark selected configuration
        selected_rank = next(i for i, c in enumerate(sorted_configs) if c['config_idx'] == best_idx) + 1
        ax4.axvline(x=selected_rank, color='gold', linestyle='--', linewidth=3, alpha=0.8, label='Selected')

        ax4.set_xlabel('Configuration Rank (by Energy)')
        ax4.set_ylabel('Energy', color='blue')
        ax4_twin.set_ylabel('Fitness / Packing Efficiency', color='green')
        ax4.set_title('Metrics by Configuration Rank')

        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        ax4.grid(True, alpha=0.3)

        # Main title
        main_title = (f"Configuration Energy Landscape Analysis\n"
                      f"Configurations: {len(configurations)} | "
                      f"Best Energy: {min(energies):.3f} | "
                      f"Energy Range: {max(energies) - min(energies):.3f}")
        fig.suptitle(main_title, fontsize=16, y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Energy landscape plot saved to: {save_path}")

        return fig



