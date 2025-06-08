"""
Visualization module for plotting simulation results with mosaic cells.
Fixed to properly handle coordinate scaling and enhanced senescent cell growth.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os
from matplotlib.gridspec import GridSpec

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

        return fig

    def plot_hourly_cell_distributions(self, simulator, save_path=None, fps=5,
                                       time_step_interval=1, show_targets=True):
        """
        Create continuous animation of cell property distributions at every time step.
        """
        import matplotlib.animation as animation
        from matplotlib.gridspec import GridSpec

        if not simulator.history:
            print("No simulation history available")
            return None

        print(f"Creating CONTINUOUS tessellated distributions animation...")
        print(f"Total history points: {len(simulator.history)}")

        # Extract data for ALL time steps (or every Nth step)
        time_step_data = []

        for i in range(0, len(simulator.history), time_step_interval):
            state = simulator.history[i]

            # Check if tessellated cell properties exist
            if 'cell_properties' not in state:
                continue

            # Extract ACTUAL tessellated properties
            cell_props = state['cell_properties']

            actual_areas = cell_props.get('areas', [])
            actual_aspect_ratios = cell_props.get('aspect_ratios', [])
            actual_orientations_deg = cell_props.get('orientations_deg', [])

            # Target values for comparison
            target_areas = cell_props.get('target_areas', actual_areas)
            target_aspect_ratios = cell_props.get('target_aspect_ratios', actual_aspect_ratios)
            target_orientations_deg = cell_props.get('target_orientations_deg', actual_orientations_deg)

            if not actual_areas:  # Skip if no cells
                continue

            time_step_data.append({
                'step': i,
                'time_minutes': state['time'],
                'time_hours': state['time'] / 60 if self.config.time_unit == "minutes" else state['time'],
                'actual_areas': actual_areas,
                'actual_aspect_ratios': actual_aspect_ratios,
                'actual_orientations_deg': actual_orientations_deg,
                'target_areas': target_areas,
                'target_aspect_ratios': target_aspect_ratios,
                'target_orientations_deg': target_orientations_deg,
                'cell_count': len(actual_areas),
                'shear_stress': state.get('input_value', 0),
                'area_adaptation': state.get('area_adaptation_quality', 1.0),
                'orientation_adaptation': state.get('orientation_adaptation_quality', 1.0)
            })

        if not time_step_data:
            print("No valid tessellated data points found")
            return None

        print(f"Animation will show {len(time_step_data)} time steps")

        # Calculate global ranges for consistent axes
        all_areas = [area for data in time_step_data for area in data['actual_areas']]
        all_aspect_ratios = [ar for data in time_step_data for ar in data['actual_aspect_ratios']]

        area_range = (min(all_areas) * 0.9, max(all_areas) * 1.1) if all_areas else (0, 100)
        ar_range = (min(all_aspect_ratios) * 0.9, max(all_aspect_ratios) * 1.1) if all_aspect_ratios else (1, 5)

        # Create figure for animation
        fig = plt.figure(figsize=(18, 8))

        # Color scheme
        colors = {
            'actual': ['skyblue', 'lightcoral', 'lightgreen'],
            'target': ['darkblue', 'darkred', 'darkgreen']
        }

        def update_frame(frame_idx):
            # Clear all subplots
            fig.clear()

            # Recreate the grid layout
            gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

            # Get current data
            data = time_step_data[frame_idx]

            # === TOP ROW: DISTRIBUTIONS ===

            # Area Distribution
            ax_area = fig.add_subplot(gs[0, 0])
            ax_area.hist(data['actual_areas'], bins=15, color=colors['actual'][0],
                         alpha=0.7, edgecolor='black', label='Tessellated')

            if show_targets and data['target_areas']:
                ax_area.hist(data['target_areas'], bins=15, color=colors['target'][0],
                             alpha=0.3, histtype='step', linewidth=2, label='Target')

            ax_area.set_title(f'Area Distribution\n(n={data["cell_count"]} cells)', fontsize=12)
            ax_area.set_xlabel('Area (pixels²)', fontsize=10)
            ax_area.set_ylabel('Frequency', fontsize=10)
            ax_area.set_xlim(area_range)
            ax_area.grid(True, alpha=0.3)

            # Statistics
            mean_area = np.mean(data['actual_areas'])
            ax_area.axvline(mean_area, color='red', linestyle='--', alpha=0.8,
                            label=f'Mean: {mean_area:.0f}')
            ax_area.legend(fontsize=9)

            # Aspect Ratio Distribution
            ax_ar = fig.add_subplot(gs[0, 1])
            ax_ar.hist(data['actual_aspect_ratios'], bins=15, color=colors['actual'][1],
                       alpha=0.7, edgecolor='black', label='Tessellated')

            if show_targets and data['target_aspect_ratios']:
                ax_ar.hist(data['target_aspect_ratios'], bins=15, color=colors['target'][1],
                           alpha=0.3, histtype='step', linewidth=2, label='Target')

            ax_ar.set_title('Aspect Ratio Distribution', fontsize=12)
            ax_ar.set_xlabel('Aspect Ratio', fontsize=10)
            ax_ar.set_ylabel('Frequency', fontsize=10)
            ax_ar.set_xlim(ar_range)
            ax_ar.grid(True, alpha=0.3)

            mean_ar = np.mean(data['actual_aspect_ratios'])
            ax_ar.axvline(mean_ar, color='red', linestyle='--', alpha=0.8,
                          label=f'Mean: {mean_ar:.1f}')
            ax_ar.legend(fontsize=9)

            # Orientation Distribution (0-360°)
            ax_orient = fig.add_subplot(gs[0, 2])
            ax_orient.hist(data['actual_orientations_deg'], bins=20, color=colors['actual'][2],
                           alpha=0.7, edgecolor='black', range=(0, 360), label='Tessellated')

            if show_targets and data['target_orientations_deg']:
                ax_orient.hist(data['target_orientations_deg'], bins=20, color=colors['target'][2],
                               alpha=0.3, histtype='step', linewidth=2, range=(0, 360), label='Target')

            ax_orient.set_title('Tessellated Orientation Distribution', fontsize=12)
            ax_orient.set_xlabel('Orientation (degrees)', fontsize=10)
            ax_orient.set_ylabel('Frequency', fontsize=10)
            ax_orient.set_xlim(0, 360)
            ax_orient.grid(True, alpha=0.3)

            # FIXED: Flow direction markers (removed unicode symbol)
            ax_orient.axvline(0, color='green', linestyle='-', alpha=0.8, linewidth=2, label='Flow (0°)')
            ax_orient.axvline(90, color='orange', linestyle='--', alpha=0.6, linewidth=1, label='Perp (90°)')
            ax_orient.axvline(180, color='green', linestyle='-', alpha=0.8, linewidth=2, label='Flow (180°)')
            ax_orient.axvline(270, color='orange', linestyle='--', alpha=0.6, linewidth=1, label='Perp (270°)')

            mean_orient = np.mean(data['actual_orientations_deg'])
            ax_orient.axvline(mean_orient, color='red', linestyle='--', alpha=0.8,
                              label=f'Mean: {mean_orient:.1f}°')
            ax_orient.legend(fontsize=8)

            # === BOTTOM ROW: TIME SERIES AND POLAR PLOT ===

            # Time series of mean properties
            ax_timeseries = fig.add_subplot(gs[1, :2])

            # Extract time series data up to current frame
            times = [d['time_hours'] for d in time_step_data[:frame_idx + 1]]
            mean_areas_series = [np.mean(d['actual_areas']) for d in time_step_data[:frame_idx + 1]]
            mean_ars_series = [np.mean(d['actual_aspect_ratios']) for d in time_step_data[:frame_idx + 1]]

            # Plot with secondary y-axes
            line1 = ax_timeseries.plot(times, mean_areas_series, 'b-', linewidth=2, label='Mean Area')
            ax_timeseries.set_xlabel('Time (hours)', fontsize=10)
            ax_timeseries.set_ylabel('Mean Area (pixels²)', color='b', fontsize=10)
            ax_timeseries.tick_params(axis='y', labelcolor='b')

            # Second y-axis for aspect ratio
            ax2 = ax_timeseries.twinx()
            line2 = ax2.plot(times, mean_ars_series, 'r-', linewidth=2, label='Mean Aspect Ratio')
            ax2.set_ylabel('Mean Aspect Ratio', color='r', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='r')

            # Mark current time
            current_time = data['time_hours']
            ax_timeseries.axvline(current_time, color='black', linestyle='--', alpha=0.8, linewidth=2)

            ax_timeseries.set_title('Property Evolution Over Time', fontsize=12)
            ax_timeseries.grid(True, alpha=0.3)

            # Polar plot for orientation
            ax_polar = fig.add_subplot(gs[1, 2], projection='polar')

            # Create polar histogram of current orientations
            orientations_rad = np.radians(data['actual_orientations_deg'])
            theta_bins = np.linspace(0, 2 * np.pi, 25)
            hist, _ = np.histogram(orientations_rad, bins=theta_bins)

            # Plot as bars
            width = 2 * np.pi / 24
            bars = ax_polar.bar(theta_bins[:-1], hist, width=width, alpha=0.7, color='lightgreen')

            ax_polar.set_title('Current Orientations\n(Polar View)', fontsize=12, pad=20)
            ax_polar.set_theta_zero_location('E')  # 0° at right (flow direction)
            ax_polar.set_theta_direction(1)  # Counterclockwise

            # Add flow direction arrow
            if hist.max() > 0:
                ax_polar.arrow(0, 0, hist.max() * 0.8, 0, width=0.1, head_width=0.2,
                               head_length=0.1, fc='red', ec='red', alpha=0.8)
                ax_polar.text(0, hist.max() * 0.9, 'Flow', ha='center', va='center',
                              fontsize=10, weight='bold',
                              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

            # Update dynamic text
            time_str = (f"STEP {data['step']:4d} | "
                        f"Time: {data['time_minutes']:6.1f} min ({data['time_hours']:5.2f}h) | "
                        f"Shear: {data['shear_stress']:5.2f} Pa | "
                        f"Cells: {data['cell_count']:3d}")

            adaptation_str = (f"Adaptation Quality: "
                              f"Area: {data['area_adaptation']:.2f} | "
                              f"Orientation: {data['orientation_adaptation']:.2f}")

            # Add text to figure
            fig.text(0.5, 0.95, time_str, ha='center', fontsize=12, weight='bold',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))

            fig.text(0.5, 0.02, adaptation_str, ha='center', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

            # Overall title
            fig.suptitle('LIVE Tessellated Cell Property Evolution', fontsize=16, y=0.98)

        # Create animation
        ani = animation.FuncAnimation(
            fig, update_frame, frames=len(time_step_data),
            interval=1000 / fps, repeat=True, blit=False
        )

        # FIXED: Generate save path with .mp4 extension
        if save_path is None:
            import time as time_module
            timestamp = time_module.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(self.config.plot_directory,
                                     f"live_tessellated_evolution_{timestamp}.mp4")
        else:
            # FIXED: Ensure .mp4 extension
            if save_path.endswith('.png'):
                save_path = save_path.replace('.png', '.mp4')
            elif not save_path.endswith('.mp4'):
                save_path = save_path + '.mp4'

        # Save animation
        print(f"Saving LIVE tessellated evolution animation to {save_path}...")

        try:
            if 'ffmpeg' in animation.writers.list():
                Writer = animation.writers['ffmpeg']
                # FIXED: Reduced bitrate and added error handling
                writer = Writer(fps=fps, metadata=dict(artist='Tessellated Evolution'), bitrate=1800)
                ani.save(save_path, writer=writer, dpi=100)
                print(f"✅ LIVE animation saved to: {save_path}")
            else:
                print("❌ Error: ffmpeg writer not available.")
                # Try alternative format
                save_path_gif = save_path.replace('.mp4', '.gif')
                ani.save(save_path_gif, writer='pillow', fps=fps)
                print(f"✅ Animation saved as GIF to: {save_path_gif}")
        except Exception as e:
            print(f"❌ Error saving animation: {e}")
            print("Trying alternative GIF format...")
            try:
                save_path_gif = save_path.replace('.mp4', '.gif')
                ani.save(save_path_gif, writer='pillow', fps=fps)
                print(f"✅ Animation saved as GIF to: {save_path_gif}")
            except Exception as e2:
                print(f"❌ Failed to save as GIF too: {e2}")

        return ani

    def plot_animated_cell_distributions(self, simulator, save_path=None, max_hours=None, fps=2):
        """
        REPLACE your entire function with this FIXED version.
        Create an animated video showing how cell property distributions evolve over time.
        """
        import matplotlib.animation as animation

        if not simulator.history:
            print("No simulation history available")
            return None

        # Reuse the data extraction logic from the static method
        print("Extracting data for animation...")

        # Convert simulation time to hours
        simulation_time_hours = simulator.time / 60 if self.config.time_unit == "minutes" else simulator.time

        # Determine hourly time points
        if max_hours is None:
            max_hours = int(simulation_time_hours) + 1

        hourly_timepoints = list(range(0, min(max_hours, int(simulation_time_hours) + 1)))

        if not hourly_timepoints:
            print("Simulation too short for animation")
            return None

        # Extract HISTORICAL data for each hourly timepoint
        hourly_data = []

        for target_hour in hourly_timepoints:
            target_time_minutes = target_hour * 60

            # Find the closest recorded time point
            times = [state['time'] for state in simulator.history]
            closest_idx = np.argmin([abs(t - target_time_minutes) for t in times])
            closest_time = times[closest_idx]

            print(f"Hour {target_hour}: using data from t={closest_time:.1f} min ({closest_time / 60:.1f}h)")

            historical_state = simulator.history[closest_idx]

            # Check if historical cell properties exist
            if 'cell_properties' not in historical_state:
                print(f"Warning: No historical cell properties found for animation at time {closest_time:.1f}")
                continue

            # Extract historical properties
            cell_props = historical_state['cell_properties']
            areas = cell_props['areas']
            aspect_ratios = cell_props['aspect_ratios']
            orientations_deg = cell_props['orientations']

            if not areas:  # Skip if no cells
                continue

            # Get shear stress for this time point from historical data
            shear_stress = historical_state.get('input_value', 0)

            hourly_data.append({
                'hour': target_hour,
                'time_minutes': closest_time,
                'areas': areas,
                'aspect_ratios': aspect_ratios,
                'orientations': orientations_deg,
                'cell_count': len(areas),
                'shear_stress': shear_stress
            })

        if not hourly_data:
            print("No valid data points found")
            return None

        print(f"Creating animation with {len(hourly_data)} frames...")

        # Calculate global ranges for consistent axes
        all_areas = [area for data in hourly_data for area in data['areas']]
        all_aspect_ratios = [ar for data in hourly_data for ar in data['aspect_ratios']]

        area_range = (min(all_areas) * 0.9, max(all_areas) * 1.1) if all_areas else (0, 100)
        ar_range = (min(all_aspect_ratios) * 0.9, max(all_aspect_ratios) * 1.1) if all_aspect_ratios else (1, 5)

        # Create figure with fixed layout
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Cell Property Distributions Over Time', fontsize=16)

        # DEFINE COLORS FIRST (before using them)
        colors = ['skyblue', 'lightcoral', 'lightgreen']

        # Initialize empty plots (NO data plotting here, just setup)
        axes[0].set_title('Area Distribution')
        axes[0].set_xlabel('Area (pixels²)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_xlim(area_range)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title('Aspect Ratio Distribution')
        axes[1].set_xlabel('Aspect Ratio')
        axes[1].set_ylabel('Frequency')
        axes[1].set_xlim(ar_range)
        axes[1].grid(True, alpha=0.3)

        # FIXED: Just setup the third axis, don't plot data yet
        axes[2].set_title('Flow Alignment Distribution')
        axes[2].set_xlabel('Alignment Angle (degrees)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_xlim(0, 90)  # FIXED: Use 0-90 range
        axes[2].grid(True, alpha=0.3)

        # Create text objects for dynamic info
        time_text = fig.text(0.02, 0.95, '', fontsize=12,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # FIXED Animation update function
        def update_frame(frame_idx):
            # Clear all axes
            for ax in axes:
                ax.clear()

            # Get current data - THIS WAS THE MISSING LINE!
            data = hourly_data[frame_idx]

            # Plot Area distribution
            axes[0].hist(data['areas'], bins=20, color=colors[0], alpha=0.7, edgecolor='black')
            axes[0].set_title('Area Distribution')
            axes[0].set_xlabel('Area (pixels²)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_xlim(area_range)
            axes[0].grid(True, alpha=0.3)

            # Add mean line for area
            if data['areas']:
                mean_area = np.mean(data['areas'])
                axes[0].axvline(mean_area, color='red', linestyle='--', alpha=0.8,
                                label=f'Mean: {mean_area:.0f}')
                axes[0].legend(fontsize=9)

            # Plot Aspect Ratio distribution
            axes[1].hist(data['aspect_ratios'], bins=20, color=colors[1], alpha=0.7, edgecolor='black')
            axes[1].set_title('Aspect Ratio Distribution')
            axes[1].set_xlabel('Aspect Ratio')
            axes[1].set_ylabel('Frequency')
            axes[1].set_xlim(ar_range)
            axes[1].grid(True, alpha=0.3)

            # Add mean line for aspect ratio
            if data['aspect_ratios']:
                mean_ar = np.mean(data['aspect_ratios'])
                axes[1].axvline(mean_ar, color='red', linestyle='--', alpha=0.8,
                                label=f'Mean: {mean_ar:.1f}')
                axes[1].legend(fontsize=9)

            # FIXED: Plot Flow Alignment distribution (0-90° range)
            axes[2].hist(data['orientations'], bins=20, color=colors[2], alpha=0.7, edgecolor='black',
                         range=(0, 90))  # FIXED: Use 0-90 range
            axes[2].set_title('Flow Alignment Distribution')
            axes[2].set_xlabel('Alignment Angle (degrees)')
            axes[2].set_ylabel('Frequency')
            axes[2].set_xlim(0, 90)  # FIXED: Use 0-90 range
            axes[2].grid(True, alpha=0.3)

            # FIXED: Add alignment references (not flow direction)
            axes[2].axvline(0, color='green', linestyle='-', alpha=0.8, linewidth=2,
                            label='Perfect alignment')
            axes[2].axvline(45, color='orange', linestyle='--', alpha=0.6, linewidth=1,
                            label='45° (intermediate)')
            axes[2].axvline(90, color='red', linestyle='-', alpha=0.8, linewidth=2,
                            label='Perpendicular')

            # Add mean orientation
            if data['orientations']:
                mean_orient = np.mean(data['orientations'])
                axes[2].axvline(mean_orient, color='red', linestyle='--', alpha=0.8,
                                label=f'Mean: {mean_orient:.1f}°')
                axes[2].legend(fontsize=9)

            # Update time text
            time_str = (f"Hour {data['hour']} ({data['time_minutes']:.0f} min)\n"
                        f"Shear Stress: {data['shear_stress']:.2f} Pa\n"
                        f"Cells: {data['cell_count']}")
            time_text.set_text(time_str)

            return axes

        # Create animation
        ani = animation.FuncAnimation(
            fig, update_frame, frames=len(hourly_data),
            interval=1000 / fps, repeat=True, blit=False
        )

        # Generate save path if not provided
        if save_path is None:
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(self.config.plot_directory,
                                     f"historical_animated_distributions_{timestamp}.mp4")

        # Save animation
        print(f"Saving animation to {save_path}...")

        # Check if ffmpeg writer is available
        if 'ffmpeg' in animation.writers.list():
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Endothelial Simulation'), bitrate=1800)
            ani.save(save_path, writer=writer, dpi=150)
            print(f"✅ Animation saved to: {save_path}")
        else:
            print("❌ Error: ffmpeg writer not available. Please install ffmpeg to save animations.")
            print("You can install ffmpeg using: pip install ffmpeg-python")

        plt.tight_layout()
        return ani

    def create_all_plots(self, simulator, history=None, prefix=None):
        """
        Create all available plots for the simulation with fixed tessellated properties.

        Parameters:
            simulator: Simulator object with current state
            history: List of state dictionaries (default: from simulator)
            prefix: Prefix for filenames (default: auto-generated timestamp)

        Returns:
            List of created figure objects
        """
        # Use simulator history if not provided
        if history is None:
            history = simulator.history

        # Use timestamp prefix if not provided
        if prefix is None:
            import time
            prefix = time.strftime("%Y%m%d-%H%M%S")

        # Create output directory
        os.makedirs(self.config.plot_directory, exist_ok=True)

        print(f"Creating all plots with prefix: {prefix}")

        # Create figures list
        figures = []

        # ===== ESSENTIAL PLOTS (Always create these) =====

        # 1. Cell population plot
        try:
            pop_fig = self.plot_cell_population(
                history,
                save_path=os.path.join(self.config.plot_directory, f"{prefix}_population.png")
            )
            figures.append(pop_fig)
            print("✅ Cell population plot created")
        except Exception as e:
            print(f"❌ Error creating population plot: {e}")

        # 2. Input pattern plot
        try:
            input_fig = self.plot_input_pattern(
                history,
                save_path=os.path.join(self.config.plot_directory, f"{prefix}_input.png")
            )
            figures.append(input_fig)
            print("✅ Input pattern plot created")
        except Exception as e:
            print(f"❌ Error creating input plot: {e}")

        # 3. Cell visualization (mosaic)
        try:
            cell_vis_fig = self.plot_cell_visualization(
                simulator,
                save_path=os.path.join(self.config.plot_directory, f"{prefix}_cells.png")
            )
            figures.append(cell_vis_fig)
            print("✅ Cell visualization plot created")
        except Exception as e:
            print(f"❌ Error creating cell visualization: {e}")

        # ===== TESSELLATED PROPERTIES ANIMATION (New fixed version) =====

        # 4. LIVE tessellated distributions animation
        try:
            print("Creating LIVE tessellated distributions animation...")
            tessellated_ani = self.plot_hourly_cell_distributions(
                simulator,
                save_path=os.path.join(self.config.plot_directory, f"{prefix}_tessellated_evolution.mp4"),
                fps=8,  # Good balance of speed and smoothness
                time_step_interval=2,  # Every 2nd step for performance
                show_targets=True
            )
            if tessellated_ani:
                figures.append(tessellated_ani)
                print("✅ LIVE tessellated evolution animation created")
        except Exception as e:
            print(f"❌ Error creating tessellated animation: {e}")
            print("   Trying fallback static version...")
            try:
                # Fallback: create static hourly distributions
                static_fig = self.plot_hourly_cell_distributions_static(simulator, prefix)
                if static_fig:
                    figures.append(static_fig)
                    print("✅ Static tessellated distributions created as fallback")
            except:
                print("❌ Fallback static version also failed")

        # ===== CONDITIONAL PLOTS (Only if data is available) =====

        # 5. Spatial metrics plot (only if available)
        try:
            if history and 'alignment_index' in history[0]:
                spatial_fig = self.plot_spatial_metrics(
                    history,
                    save_path=os.path.join(self.config.plot_directory, f"{prefix}_spatial.png")
                )
                if spatial_fig:
                    figures.append(spatial_fig)
                    print("✅ Spatial metrics plot created")
            else:
                print("ℹ️  Spatial metrics not available in history data")
        except Exception as e:
            print(f"❌ Error creating spatial metrics plot: {e}")

        # 6. Mosaic metrics plot (only if available)
        try:
            if history and 'packing_efficiency' in history[0]:
                mosaic_fig = self.plot_mosaic_metrics(
                    history,
                    save_path=os.path.join(self.config.plot_directory, f"{prefix}_mosaic.png")
                )
                if mosaic_fig:
                    figures.append(mosaic_fig)
                    print("✅ Mosaic metrics plot created")
            else:
                print("ℹ️  Mosaic metrics not available in history data")
        except Exception as e:
            print(f"❌ Error creating mosaic metrics plot: {e}")

        # 7. Senescent growth metrics (only if available)
        try:
            if history and any('senescent_count' in state for state in history):
                growth_fig = self.plot_senescent_growth_metrics(
                    history,
                    save_path=os.path.join(self.config.plot_directory, f"{prefix}_senescent_growth.png")
                )
                if growth_fig:
                    figures.append(growth_fig)
                    print("✅ Senescent growth metrics plot created")
            else:
                print("ℹ️  Senescent growth data not available")
        except Exception as e:
            print(f"❌ Error creating senescent growth plot: {e}")

        # ===== VALIDATION AND ANALYSIS PLOTS =====

        # 8. Tessellation validation plot (if you added this method)
        try:
            if hasattr(self, 'plot_tessellation_validation'):
                validation_fig = self.plot_tessellation_validation(
                    simulator,
                    save_path=os.path.join(self.config.plot_directory, f"{prefix}_validation.png")
                )
                if validation_fig:
                    figures.append(validation_fig)
                    print("✅ Tessellation validation plot created")
        except Exception as e:
            print(f"❌ Error creating validation plot: {e}")

        # 9. Summary statistics plot
        try:
            summary_fig = self.create_summary_plot(simulator, history, prefix)
            if summary_fig:
                figures.append(summary_fig)
                print("✅ Summary statistics plot created")
        except Exception as e:
            print(f"❌ Error creating summary plot: {e}")

        # ===== FINAL REPORT =====

        print(f"\n{'=' * 60}")
        print(f"VISUALIZATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total figures created: {len(figures)}")
        print(f"Output directory: {self.config.plot_directory}")
        print(f"Prefix used: {prefix}")

        # List all created files
        try:
            import glob
            pattern = os.path.join(self.config.plot_directory, f"{prefix}_*")
            created_files = glob.glob(pattern)
            print(f"\nCreated files:")
            for file in sorted(created_files):
                file_size = os.path.getsize(file) / 1024  # KB
                print(f"  📄 {os.path.basename(file)} ({file_size:.1f} KB)")
        except Exception as e:
            print(f"Error listing files: {e}")

        print(f"{'=' * 60}\n")

        return figures

    def plot_hourly_cell_distributions_static(self, simulator, prefix):
        """
        Fallback method: Create static hourly distributions if animation fails.
        """
        try:
            from matplotlib.gridspec import GridSpec

            if not simulator.history:
                return None

            # Get final state for static plot
            final_state = simulator.history[-1]

            if 'cell_properties' not in final_state:
                return None

            cell_props = final_state['cell_properties']
            areas = cell_props.get('areas', [])
            aspect_ratios = cell_props.get('aspect_ratios', [])
            orientations_deg = cell_props.get('orientations_deg', [])

            if not areas:
                return None

            # Create static plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Area distribution
            axes[0].hist(areas, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
            axes[0].set_title('Final Area Distribution')
            axes[0].set_xlabel('Area (pixels²)')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)

            # Aspect ratio distribution
            axes[1].hist(aspect_ratios, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
            axes[1].set_title('Final Aspect Ratio Distribution')
            axes[1].set_xlabel('Aspect Ratio')
            axes[1].set_ylabel('Frequency')
            axes[1].grid(True, alpha=0.3)

            # Orientation distribution
            axes[2].hist(orientations_deg, bins=30, color='lightgreen', alpha=0.7,
                         edgecolor='black', range=(0, 360))
            axes[2].set_title('Final Tessellated Orientations')
            axes[2].set_xlabel('Orientation (degrees)')
            axes[2].set_ylabel('Frequency')
            axes[2].set_xlim(0, 360)
            axes[2].axvline(0, color='green', linestyle='-', alpha=0.8, linewidth=2, label='Flow (0°)')
            axes[2].axvline(180, color='green', linestyle='-', alpha=0.8, linewidth=2, label='Flow (180°)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.suptitle('Final Tessellated Cell Properties (Static Fallback)', fontsize=16)
            plt.tight_layout()

            # Save
            save_path = os.path.join(self.config.plot_directory, f"{prefix}_tessellated_static.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

            return fig

        except Exception as e:
            print(f"Error in static fallback: {e}")
            return None

    def create_summary_plot(self, simulator, history, prefix):
        """
        Create a summary plot with key metrics and final state info.
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

            # Extract time series data
            times = [state['time'] for state in history] if history else []
            time_hours = [t / 60 for t in times] if self.config.time_unit == "minutes" else times

            # 1. Cell count over time
            if history and 'cells' in history[0]:
                cell_counts = [state['cells'] for state in history]
                ax1.plot(time_hours, cell_counts, 'b-', linewidth=2)
                ax1.set_title('Cell Count Evolution')
                ax1.set_xlabel('Time (hours)')
                ax1.set_ylabel('Cell Count')
                ax1.grid(True, alpha=0.3)

            # 2. Input pattern
            if history and 'input_value' in history[0]:
                input_values = [state['input_value'] for state in history]
                ax2.plot(time_hours, input_values, 'r-', linewidth=2)
                ax2.set_title('Shear Stress Input')
                ax2.set_xlabel('Time (hours)')
                ax2.set_ylabel('Shear Stress (Pa)')
                ax2.grid(True, alpha=0.3)

            # 3. Final cell type distribution
            if simulator.grid and simulator.grid.cells:
                cell_types = {'Healthy': 0, 'Tel-Senescent': 0, 'Stress-Senescent': 0}
                for cell in simulator.grid.cells.values():
                    if not cell.is_senescent:
                        cell_types['Healthy'] += 1
                    elif cell.senescence_cause == 'telomere':
                        cell_types['Tel-Senescent'] += 1
                    else:
                        cell_types['Stress-Senescent'] += 1

                labels = list(cell_types.keys())
                sizes = list(cell_types.values())
                colors = ['green', 'red', 'blue']

                if sum(sizes) > 0:
                    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax3.set_title('Final Cell Type Distribution')

            # 4. Summary text
            ax4.axis('off')

            # Collect summary statistics
            final_time = times[-1] if times else 0
            final_time_hours = final_time / 60 if self.config.time_unit == "minutes" else final_time
            total_cells = len(simulator.grid.cells) if simulator.grid else 0

            summary_text = f"""
    SIMULATION SUMMARY
    {'=' * 25}

    Duration: {final_time_hours:.1f} hours
    Final Cell Count: {total_cells}

    Grid Size: {simulator.grid.width}×{simulator.grid.height} pixels
    Time Steps: {len(history)} recorded

    Configuration:
    - Population Dynamics: {'✓' if self.config.enable_population_dynamics else '✗'}
    - Spatial Properties: {'✓' if self.config.enable_spatial_properties else '✗'}
    - Temporal Dynamics: {'✓' if self.config.enable_temporal_dynamics else '✗'}
    - Senescence: {'✓' if self.config.enable_senescence else '✗'}

    Final State:
    - Healthy: {cell_types.get('Healthy', 0)}
    - Tel-Senescent: {cell_types.get('Tel-Senescent', 0)}
    - Stress-Senescent: {cell_types.get('Stress-Senescent', 0)}
            """

            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            plt.suptitle(f'Simulation Summary: {prefix}', fontsize=16)
            plt.tight_layout()

            # Save
            save_path = os.path.join(self.config.plot_directory, f"{prefix}_summary.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

            return fig

        except Exception as e:
            print(f"Error creating summary plot: {e}")
            return None