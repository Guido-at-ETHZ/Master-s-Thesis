"""
Visualization module for plotting simulation results.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import os


class Plotter:
    """
    Class for creating visualizations of simulation results.
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
        confluency = np.array([state['confluency'] for state in history])

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
        axes[1].set_title('Cell Shape Index (P/sqrt(4πA))', fontsize=12)

        axes[2].plot(time_in_hours, confluency, 'r-', linewidth=2)
        axes[2].set_ylabel('Confluency', fontsize=12)
        axes[2].set_title('Monolayer Confluency', fontsize=12)
        axes[2].set_ylim(0, 1)

        # Format plot
        axes[2].set_xlabel(time_label, fontsize=12)
        plt.suptitle('Spatial Metrics of Endothelial Monolayer', fontsize=14)

        # Save if requested
        if save_path is not None:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_population_metrics(self, history, save_path=None):
        """
        Plot population metrics over time.

        Parameters:
            history: List of state dictionaries from simulation
            save_path: Path to save the plot (default: auto-generated)

        Returns:
            Matplotlib figure
        """
        # Extract data
        time = np.array([state['time'] for state in history])

        # Check if population metrics exist
        if 'avg_division_age' not in history[0]:
            print("Population metrics not available in history data")
            return None

        avg_division = np.array([state.get('avg_division_age', np.nan) for state in history])
        telomere_length = np.array([state.get('telomere_length', np.nan) for state in history])

        # Filter out NaN values
        valid_indices = ~np.isnan(avg_division)

        # Convert time to hours if needed
        if self.config.time_unit == "minutes":
            time_in_hours = time / 60
            time_label = "Time (hours)"
        else:
            time_in_hours = time
            time_label = f"Time ({self.config.time_unit})"

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot metrics
        if np.any(valid_indices):
            axes[0].plot(time_in_hours[valid_indices], avg_division[valid_indices], 'b-', linewidth=2)
        axes[0].set_ylabel('Average Division Age', fontsize=12)
        axes[0].set_title('Average Division Age of Healthy Cells', fontsize=12)
        axes[0].set_ylim(0, self.config.max_divisions)

        if np.any(valid_indices):
            axes[1].plot(time_in_hours[valid_indices], telomere_length[valid_indices], 'g-', linewidth=2)
        axes[1].set_ylabel('Telomere Length', fontsize=12)
        axes[1].set_title('Average Telomere Length of Healthy Cells', fontsize=12)
        axes[1].set_ylim(10, 110)
        axes[1].axhline(y=20, color='r', linestyle='--', label='Critical Length')
        axes[1].legend()

        # Format plot
        axes[1].set_xlabel(time_label, fontsize=12)
        plt.suptitle('Population Metrics of Endothelial Cells', fontsize=14)

        # Save if requested
        if save_path is not None:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_senescence_metrics(self, history, save_path=None):
        """
        Plot senescence metrics over time.

        Parameters:
            history: List of state dictionaries from simulation
            save_path: Path to save the plot (default: auto-generated)

        Returns:
            Matplotlib figure
        """
        # Extract data
        time = np.array([state['time'] for state in history])

        # Check if detailed population data exists
        if 'healthy_cells' not in history[0]:
            print("Senescence metrics not available in history data")
            return None

        healthy = np.array([state['healthy_cells'] for state in history])
        sen_tel = np.array([state['senescent_tel'] for state in history])
        sen_stress = np.array([state['senescent_stress'] for state in history])

        # Calculate senescence fractions
        total_cells = healthy + sen_tel + sen_stress
        sen_fraction = np.zeros_like(time)
        tel_fraction_of_sen = np.zeros_like(time)

        for i in range(len(time)):
            if total_cells[i] > 0:
                sen_fraction[i] = (sen_tel[i] + sen_stress[i]) / total_cells[i]

                if sen_tel[i] + sen_stress[i] > 0:
                    tel_fraction_of_sen[i] = sen_tel[i] / (sen_tel[i] + sen_stress[i])

        # Convert time to hours if needed
        if self.config.time_unit == "minutes":
            time_in_hours = time / 60
            time_label = "Time (hours)"
        else:
            time_in_hours = time
            time_label = f"Time ({self.config.time_unit})"

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot metrics
        axes[0].plot(time_in_hours, sen_fraction, 'r-', linewidth=2)
        axes[0].set_ylabel('Fraction', fontsize=12)
        axes[0].set_title('Fraction of Senescent Cells', fontsize=12)
        axes[0].set_ylim(0, 1)

        axes[1].plot(time_in_hours, tel_fraction_of_sen, 'b-', linewidth=2)
        axes[1].set_ylabel('Fraction', fontsize=12)
        axes[1].set_title('Fraction of Senescent Cells Due to Telomere Shortening', fontsize=12)
        axes[1].set_ylim(0, 1)

        # Format plot
        axes[1].set_xlabel(time_label, fontsize=12)
        plt.suptitle('Senescence Metrics of Endothelial Cells', fontsize=14)

        # Save if requested
        if save_path is not None:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_cell_visualization(self, simulator, save_path=None):
        """
        Create a visualization of cells.

        Parameters:
            simulator: Simulator object with current state
            save_path: Path to save the plot (default: auto-generated)

        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Set axis limits based on grid size
        ax.set_xlim(0, simulator.grid.width)
        ax.set_ylim(0, simulator.grid.height)

        # Plot cells
        for cell in simulator.grid.cells.values():
            # Get cell properties
            x, y = cell.position
            orientation = cell.orientation
            aspect_ratio = cell.aspect_ratio
            area = cell.area
            is_senescent = cell.is_senescent
            senescence_cause = cell.senescence_cause

            # Calculate ellipse dimensions
            a = np.sqrt(area * aspect_ratio)  # Semi-major axis
            b = area / a  # Semi-minor axis

            # Create ellipse
            ellipse = Ellipse(
                xy=(x, y),
                width=2 * a,
                height=2 * b,
                angle=np.degrees(orientation),
                alpha=0.7
            )

            # Set color based on cell type
            if not is_senescent:
                color = 'green'
            elif senescence_cause == 'telomere':
                color = 'red'
            else:  # stress-induced
                color = 'blue'

            ellipse.set_facecolor(color)
            ellipse.set_edgecolor('black')
            ellipse.set_linewidth(0.5)

            # Add to plot
            ax.add_patch(ellipse)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', alpha=0.7, label='Healthy'),
            Patch(facecolor='red', edgecolor='black', alpha=0.7, label='Telomere-Senescent'),
            Patch(facecolor='blue', edgecolor='black', alpha=0.7, label='Stress-Senescent')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Format plot
        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)

        # Add info text
        info_text = (
            f"Time: {simulator.time:.1f} {simulator.config.time_unit}\n"
            f"Shear Stress: {simulator.input_pattern['value']:.2f} Pa\n"
            f"Cells: {len(simulator.grid.cells)}"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add flow direction indicator
        arrow_length = simulator.grid.width * 0.1
        arrow_x = simulator.grid.width * 0.5
        arrow_y = simulator.grid.height * 0.05

        ax.arrow(arrow_x - arrow_length / 2, arrow_y, arrow_length, 0,
                 head_width=arrow_length * 0.2, head_length=arrow_length * 0.2,
                 fc='black', ec='black', width=arrow_length * 0.05)

        ax.text(arrow_x, arrow_y - arrow_length * 0.3, "Flow Direction",
                ha='center', va='top', fontsize=10)

        # Add title
        plt.title('Endothelial Cell Visualization', fontsize=14)

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
        if 'alignment_index' in history[0]:
            spatial_fig = self.plot_spatial_metrics(
                history,
                save_path=os.path.join(self.config.plot_directory, f"{prefix}_spatial.png")
            )
            figures.append(spatial_fig)

        # Population metrics plot (if available)
        if 'avg_division_age' in history[0]:
            pop_metrics_fig = self.plot_population_metrics(
                history,
                save_path=os.path.join(self.config.plot_directory, f"{prefix}_pop_metrics.png")
            )
            figures.append(pop_metrics_fig)

        # Senescence metrics plot (if available)
        if 'healthy_cells' in history[0]:
            sen_fig = self.plot_senescence_metrics(
                history,
                save_path=os.path.join(self.config.plot_directory, f"{prefix}_senescence.png")
            )
            figures.append(sen_fig)

        # Cell visualization
        cell_vis_fig = self.plot_cell_visualization(
            simulator,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_cells.png")
        )
        figures.append(cell_vis_fig)

        return figures