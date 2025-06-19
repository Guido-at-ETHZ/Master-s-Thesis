"""
Visualization module for plotting simulation results with mosaic cells.
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

    def plot_cell_visualization(self, simulator, save_path=None, show_boundaries=True, show_seeds=False):
        """
        Create a visualization of cells as mosaic territories.

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

        # Color mapping for cell types
        color_map = {
            'healthy': 'green',
            'telomere': 'red',
            'stress': 'blue'
        }

        # Plot each cell's territory
        for cell in simulator.grid.cells.values():
            if not cell.territory_pixels:
                continue

            # Determine cell color
            if not cell.is_senescent:
                color = color_map['healthy']
                alpha = 0.6
            elif cell.senescence_cause == 'telomere':
                color = color_map['telomere']
                alpha = 0.8
            else:  # stress-induced
                color = color_map['stress']
                alpha = 0.8

            # Create polygon from territory pixels
            if len(cell.territory_pixels) > 2:
                # Create a simplified boundary polygon
                boundary_coords = np.array(cell.boundary_points)
                if len(boundary_coords) > 2:
                    # Sort boundary points to form a proper polygon
                    # Simple convex hull approach
                    from scipy.spatial import ConvexHull
                    try:
                        hull = ConvexHull(boundary_coords)
                        polygon_coords = boundary_coords[hull.vertices]

                        polygon = Polygon(polygon_coords, facecolor=color, alpha=alpha,
                                        edgecolor='black' if show_boundaries else color,
                                        linewidth=1 if show_boundaries else 0)
                        ax.add_patch(polygon)
                    except:
                        # Fallback: just plot pixels as points
                        pixels_array = np.array(cell.territory_pixels)
                        ax.scatter(pixels_array[:, 0], pixels_array[:, 1],
                                 c=color, alpha=alpha, s=1, marker='s')

            # Show seed point if requested
            if show_seeds:
                seed_x, seed_y = cell.position
                ax.plot(seed_x, seed_y, 'ko', markersize=3)

            # Show orientation vector
            if cell.centroid is not None:
                cx, cy = cell.centroid
                # Draw orientation vector
                vector_length = np.sqrt(cell.actual_area) * 0.3
                dx = vector_length * np.cos(cell.actual_orientation)
                dy = vector_length * np.sin(cell.actual_orientation)

                ax.arrow(cx, cy, dx, dy, head_width=vector_length*0.1,
                        head_length=vector_length*0.1, fc='white', ec='black',
                        alpha=0.8, width=vector_length*0.02)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', alpha=0.6, label='Healthy'),
            Patch(facecolor='red', edgecolor='black', alpha=0.8, label='Telomere-Senescent'),
            Patch(facecolor='blue', edgecolor='black', alpha=0.8, label='Stress-Senescent')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Format plot
        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)
        ax.set_aspect('equal')

        # Add info text
        total_cells = len(simulator.grid.cells)
        grid_stats = simulator.grid.get_grid_statistics()

        info_text = (
            f"Time: {simulator.time:.1f} {simulator.config.time_unit}\n"
            f"Shear Stress: {simulator.input_pattern['value']:.2f} Pa\n"
            f"Total Cells: {total_cells}\n"
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
                 fc='black', ec='black', width=arrow_length * 0.08)

        ax.text(arrow_x, arrow_y - arrow_length * 0.5, "Flow Direction",
                ha='center', va='top', fontsize=12, weight='bold')

        # Add title
        plt.title('Endothelial Cell Mosaic Visualization', fontsize=16)

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
        ax.plot(time_in_hours, input_value, 'r-', linewidth=2)

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

        # Cell visualization
        cell_vis_fig = self.plot_cell_visualization(
            simulator,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_cells.png")
        )
        figures.append(cell_vis_fig)

        return figures