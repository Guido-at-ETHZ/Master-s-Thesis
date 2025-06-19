"""
Enhanced visualization module for plotting simulation results with senescence effects.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
import os


class EnhancedPlotter:
    """
    Enhanced plotter class for creating visualizations of simulation results with senescence.
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
        
        # Create custom colormap for senescence levels
        colors = ['green', 'yellow', 'orange', 'red']
        self.senescence_cmap = LinearSegmentedColormap.from_list("senescence", colors)

    def plot_cell_visualization_enhanced(self, simulator, save_path=None, show_senescence_levels=True,
                                       show_orientation_vectors=True, show_stress_field=False):
        """
        Create an enhanced visualization of cells with senescence levels and morphometry.

        Parameters:
            simulator: Simulator object with current state
            save_path: Path to save the plot (default: auto-generated)
            show_senescence_levels: Whether to color cells by senescence level
            show_orientation_vectors: Whether to show cell orientation vectors
            show_stress_field: Whether to show background stress field

        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Set axis limits based on grid size
        ax.set_xlim(0, simulator.grid.width)
        ax.set_ylim(0, simulator.grid.height)

        # Show stress field background if requested
        if show_stress_field:
            self._add_stress_field_background(ax, simulator)

        # Plot cells
        senescence_levels = []
        
        for cell in simulator.grid.cells.values():
            # Get cell properties
            x, y = cell.position
            orientation = cell.orientation
            aspect_ratio = cell.aspect_ratio
            area = cell.area
            senescence_level = getattr(cell, 'senescence_level', 0.0)
            
            senescence_levels.append(senescence_level)

            # Calculate ellipse dimensions
            a = np.sqrt(area * aspect_ratio)  # Semi-major axis
            b = area / a  # Semi-minor axis

            # Create ellipse
            ellipse = Ellipse(
                xy=(x, y),
                width=2 * a,
                height=2 * b,
                angle=np.degrees(orientation),
                alpha=0.8
            )

            # Set color based on senescence level or cell type
            if show_senescence_levels:
                # Color by senescence level (0-100%)
                color_intensity = senescence_level / 100.0
                color = self.senescence_cmap(color_intensity)
                ellipse.set_facecolor(color)
            else:
                # Traditional coloring by cell type
                if not cell.is_senescent:
                    color = 'green'
                elif cell.senescence_cause == 'telomere':
                    color = 'red'
                else:  # stress-induced
                    color = 'blue'
                ellipse.set_facecolor(color)

            ellipse.set_edgecolor('black')
            ellipse.set_linewidth(0.5)

            # Add to plot
            ax.add_patch(ellipse)

            # Add orientation vector if requested
            if show_orientation_vectors:
                vector_length = np.sqrt(area) * 0.8
                dx = vector_length * np.cos(orientation)
                dy = vector_length * np.sin(orientation)
                
                ax.arrow(x, y, dx, dy, head_width=vector_length*0.1, 
                        head_length=vector_length*0.1, fc='black', ec='black', 
                        alpha=0.6, width=vector_length*0.02)

        # Add colorbar for senescence levels if shown
        if show_senescence_levels and senescence_levels:
            # Create colorbar
            sm = plt.cm.ScalarMappable(cmap=self.senescence_cmap, 
                                     norm=plt.Normalize(vmin=0, vmax=100))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Senescence Level (%)', fontsize=12)

        # Add traditional legend if not showing senescence levels
        if not show_senescence_levels:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', edgecolor='black', alpha=0.8, label='Healthy'),
                Patch(facecolor='red', edgecolor='black', alpha=0.8, label='Telomere-Senescent'),
                Patch(facecolor='blue', edgecolor='black', alpha=0.8, label='Stress-Senescent')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

        # Format plot
        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)

        # Add comprehensive info text
        total_cells = len(simulator.grid.cells)
        avg_senescence = np.mean(senescence_levels) if senescence_levels else 0
        
        # Count cells by senescence level ranges
        fully_senescent = sum(1 for s in senescence_levels if s >= 100)
        partially_senescent = sum(1 for s in senescence_levels if 0 < s < 100)
        healthy = sum(1 for s in senescence_levels if s == 0)
        
        info_text = (
            f"Time: {simulator.time:.1f} {simulator.config.time_unit}\n"
            f"Pressure: {simulator.input_pattern['value']:.2f} Pa\n"
            f"Total Cells: {total_cells}\n"
            f"Healthy: {healthy} ({100*healthy/max(1,total_cells):.1f}%)\n"
            f"Partially Senescent: {partially_senescent} ({100*partially_senescent/max(1,total_cells):.1f}%)\n"
            f"Fully Senescent: {fully_senescent} ({100*fully_senescent/max(1,total_cells):.1f}%)\n"
            f"Avg Senescence: {avg_senescence:.1f}%"
        )
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Add flow direction indicator
        self._add_flow_direction_indicator(ax, simulator.grid.width, simulator.grid.height)

        # Add title
        title = 'Enhanced Endothelial Cell Visualization'
        if show_senescence_levels:
            title += ' (Senescence Levels)'
        plt.title(title, fontsize=14)

        # Save if requested
        if save_path is not None:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_morphometry_metrics(self, history, save_path=None):
        """
        Plot morphometry-specific metrics over time.

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

        # Check if morphometry metrics exist
        has_morphometry = any('avg_senescence_level' in state for state in history)
        
        if not has_morphometry:
            print("Morphometry metrics not available in history data")
            return None

        # Extract morphometry data
        avg_senescence = np.array([state.get('avg_senescence_level', 0) for state in history])
        avg_area = np.array([state.get('avg_cell_area', 0) for state in history])
        avg_aspect_ratio = np.array([state.get('avg_aspect_ratio', 1) for state in history])
        avg_eccentricity = np.array([state.get('avg_eccentricity', 0) for state in history])
        avg_circularity = np.array([state.get('avg_circularity', 0) for state in history])

        # Create figure with subplots
        fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

        # Plot metrics
        axes[0].plot(time_in_hours, avg_senescence, 'r-', linewidth=2)
        axes[0].set_ylabel('Avg Senescence Level (%)', fontsize=12)
        axes[0].set_title('Average Senescence Level', fontsize=12)
        axes[0].set_ylim(0, 100)
        axes[0].grid(True)

        axes[1].plot(time_in_hours, avg_area, 'b-', linewidth=2)
        axes[1].set_ylabel('Avg Cell Area (pixels²)', fontsize=12)
        axes[1].set_title('Average Cell Area', fontsize=12)
        axes[1].grid(True)

        axes[2].plot(time_in_hours, avg_aspect_ratio, 'g-', linewidth=2)
        axes[2].set_ylabel('Avg Aspect Ratio', fontsize=12)
        axes[2].set_title('Average Cell Aspect Ratio', fontsize=12)
        axes[2].grid(True)

        axes[3].plot(time_in_hours, avg_eccentricity, 'm-', linewidth=2)
        axes[3].set_ylabel('Avg Eccentricity', fontsize=12)
        axes[3].set_title('Average Cell Eccentricity', fontsize=12)
        axes[3].set_ylim(0, 1)
        axes[3].grid(True)

        axes[4].plot(time_in_hours, avg_circularity, 'c-', linewidth=2)
        axes[4].set_ylabel('Avg Circularity', fontsize=12)
        axes[4].set_title('Average Cell Circularity', fontsize=12)
        axes[4].set_ylim(0, 1)
        axes[4].grid(True)

        # Format plot
        axes[4].set_xlabel(time_label, fontsize=12)
        plt.suptitle('Cell Morphometry Metrics Evolution', fontsize=16)

        # Save if requested
        if save_path is not None:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_senescence_distribution(self, simulator, save_path=None, bins=20):
        """
        Plot distribution of senescence levels across the cell population.

        Parameters:
            simulator: Simulator object with current state
            save_path: Path to save the plot
            bins: Number of histogram bins

        Returns:
            Matplotlib figure
        """
        # Extract senescence levels
        senescence_levels = []
        for cell in simulator.grid.cells.values():
            senescence_level = getattr(cell, 'senescence_level', 0.0)
            senescence_levels.append(senescence_level)

        if not senescence_levels:
            print("No cells found in simulator")
            return None

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram of senescence levels
        ax1.hist(senescence_levels, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Senescence Level (%)', fontsize=12)
        ax1.set_ylabel('Number of Cells', fontsize=12)
        ax1.set_title('Distribution of Senescence Levels', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Add statistics
        mean_senescence = np.mean(senescence_levels)
        median_senescence = np.median(senescence_levels)
        ax1.axvline(mean_senescence, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_senescence:.1f}%')
        ax1.axvline(median_senescence, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_senescence:.1f}%')
        ax1.legend()

        # Pie chart of senescence categories
        healthy = sum(1 for s in senescence_levels if s == 0)
        partially_senescent = sum(1 for s in senescence_levels if 0 < s < 100)
        fully_senescent = sum(1 for s in senescence_levels if s >= 100)

        labels = ['Healthy', 'Partially Senescent', 'Fully Senescent']
        sizes = [healthy, partially_senescent, fully_senescent]
        colors = ['green', 'yellow', 'red']
        
        # Only include non-zero categories
        non_zero_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors) if size > 0]
        if non_zero_data:
            labels, sizes, colors = zip(*non_zero_data)
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                             startangle=90, textprops={'fontsize': 11})
            ax2.set_title('Senescence Categories', fontsize=14)

        plt.tight_layout()

        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_pressure_response_curves(self, save_path=None):
        """
        Plot the theoretical morphometry response curves to pressure.

        Parameters:
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        # Create pressure range
        pressures = np.linspace(0, 1.4, 100)
        
        # Initialize spatial model to get parameter values
        from spatial_properties_updated import SpatialPropertiesModel
        spatial_model = SpatialPropertiesModel(self.config)
        
        # Calculate parameter values for different senescence levels
        senescence_levels = [0, 25, 50, 75, 100]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        properties = ['area', 'aspect_ratio', 'eccentricity', 'circularity']
        property_labels = ['Cell Area (pixels²)', 'Aspect Ratio', 'Eccentricity', 'Circularity']
        
        for prop_idx, (prop, label) in enumerate(zip(properties, property_labels)):
            ax = axes[prop_idx]
            
            for senescence in senescence_levels:
                values = []
                for pressure in pressures:
                    if senescence >= 100:
                        # Fully senescent - constant value
                        value = spatial_model.senescent_params[prop]
                    elif senescence == 0:
                        # Normal cells - pressure dependent
                        value = spatial_model.calculate_pressure_dependent_value(prop, pressure)
                    else:
                        # Partially senescent - linear combination
                        s_fraction = senescence / 100.0
                        normal_value = spatial_model.calculate_pressure_dependent_value(prop, pressure)
                        senescent_value = spatial_model.senescent_params[prop]
                        value = s_fraction * senescent_value + (1 - s_fraction) * normal_value
                    
                    values.append(value)
                
                # Plot curve
                color = self.senescence_cmap(senescence / 100.0)
                ax.plot(pressures, values, linewidth=2, color=color, 
                       label=f'{senescence}% Senescent')
            
            ax.set_xlabel('Pressure (Pa)', fontsize=12)
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(f'{label} vs Pressure', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        
        plt.suptitle('Theoretical Morphometry Response to Pressure', fontsize=16)
        plt.tight_layout()

        # Save if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _add_stress_field_background(self, ax, simulator):
        """Add background stress field visualization."""
        # Create a simple uniform stress field background
        current_stress = simulator.input_pattern['value']
        
        # Create a subtle background color based on stress level
        stress_color = plt.cm.Reds(min(current_stress / 5.0, 1.0))  # Normalize to 0-5 Pa range
        ax.add_patch(plt.Rectangle((0, 0), simulator.grid.width, simulator.grid.height, 
                                  facecolor=stress_color, alpha=0.1, zorder=0))

    def _add_flow_direction_indicator(self, ax, width, height):
        """Add flow direction indicator to the plot."""
        arrow_length = width * 0.1
        arrow_x = width * 0.5
        arrow_y = height * 0.05

        ax.arrow(arrow_x - arrow_length / 2, arrow_y, arrow_length, 0,
                 head_width=arrow_length * 0.2, head_length=arrow_length * 0.2,
                 fc='black', ec='black', width=arrow_length * 0.05, zorder=10)

        ax.text(arrow_x, arrow_y - arrow_length * 0.3, "Flow Direction",
                ha='center', va='top', fontsize=10, weight='bold')

    def create_all_enhanced_plots(self, simulator, history=None, prefix=None):
        """
        Create all enhanced plots for the simulation.

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

        # Enhanced cell visualization
        enhanced_vis_fig = self.plot_cell_visualization_enhanced(
            simulator,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_enhanced_cells.png"),
            show_senescence_levels=True,
            show_orientation_vectors=True
        )
        figures.append(enhanced_vis_fig)

        # Traditional cell visualization for comparison
        traditional_vis_fig = self.plot_cell_visualization_enhanced(
            simulator,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_traditional_cells.png"),
            show_senescence_levels=False,
            show_orientation_vectors=False
        )
        figures.append(traditional_vis_fig)

        # Morphometry metrics
        morphometry_fig = self.plot_morphometry_metrics(
            history,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_morphometry.png")
        )
        if morphometry_fig:
            figures.append(morphometry_fig)

        # Senescence distribution
        senescence_dist_fig = self.plot_senescence_distribution(
            simulator,
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_senescence_dist.png")
        )
        if senescence_dist_fig:
            figures.append(senescence_dist_fig)

        # Pressure response curves
        response_curves_fig = self.plot_pressure_response_curves(
            save_path=os.path.join(self.config.plot_directory, f"{prefix}_response_curves.png")
        )
        figures.append(response_curves_fig)

        return figures
