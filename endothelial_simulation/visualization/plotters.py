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

    def create_all_plots(self, simulator, history=None, prefix=None):
        """
        Create all available plots for the simulation.

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

        return figures