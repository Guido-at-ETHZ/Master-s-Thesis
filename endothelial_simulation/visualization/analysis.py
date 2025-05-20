"""
Analysis module for endothelial cell mechanotransduction simulation results.

This module provides functions for statistical analysis, metrics extraction,
and comparison of simulation results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


class SimulationAnalyzer:
    """
    Class for analyzing endothelial cell simulation results.
    """

    def __init__(self, config=None):
        """
        Initialize the analyzer with optional configuration.

        Parameters:
            config: SimulationConfig object with parameter settings
        """
        self.config = config

    def history_to_dataframe(self, history):
        """
        Convert simulation history to a pandas DataFrame.

        Parameters:
            history: List of state dictionaries from simulation

        Returns:
            pandas.DataFrame containing all metrics over time
        """
        # Convert history to DataFrame
        df = pd.DataFrame(history)

        # Convert time to hours if available and config specifies minutes
        if 'time' in df.columns and self.config and self.config.time_unit == "minutes":
            df['time_hours'] = df['time'] / 60

        # Calculate additional metrics if possible
        if all(col in df.columns for col in ['healthy_cells', 'senescent_tel', 'senescent_stress']):
            # Total cells
            df['total_cells'] = df['healthy_cells'] + df['senescent_tel'] + df['senescent_stress']

            # Senescent cell fraction
            df['senescent_fraction'] = (df['senescent_tel'] + df['senescent_stress']) / df['total_cells']

            # Type-specific senescent fractions
            df['tel_senescent_fraction'] = df['senescent_tel'] / df['total_cells']
            df['stress_senescent_fraction'] = df['senescent_stress'] / df['total_cells']

            # Ratio of senescence types
            sen_total = df['senescent_tel'] + df['senescent_stress']
            df['tel_to_stress_ratio'] = np.where(df['senescent_stress'] > 0,
                                                 df['senescent_tel'] / df['senescent_stress'],
                                                 np.inf)

        return df

    def extract_steady_state_metrics(self, df, steady_state_window=0.2):
        """
        Extract steady-state metrics from time series data.

        Parameters:
            df: DataFrame containing simulation metrics over time
            steady_state_window: Fraction of final simulation time to consider as steady state

        Returns:
            Dictionary of steady-state metrics
        """
        # Determine steady-state region (last x% of simulation)
        if 'time' in df.columns:
            max_time = df['time'].max()
            steady_state_start = max_time * (1 - steady_state_window)
            steady_df = df[df['time'] >= steady_state_start]
        else:
            # If no time column, use last 20% of rows
            n_rows = len(df)
            steady_df = df.iloc[int(n_rows * (1 - steady_state_window)):]

        # Calculate means and standard deviations for all numeric columns
        metrics = {}

        for col in steady_df.select_dtypes(include=[np.number]).columns:
            metrics[f'{col}_mean'] = steady_df[col].mean()
            metrics[f'{col}_std'] = steady_df[col].std()
            metrics[f'{col}_min'] = steady_df[col].min()
            metrics[f'{col}_max'] = steady_df[col].max()

        return metrics

    def calculate_response_times(self, df, response_threshold=0.95):
        """
        Calculate response times to reach specific thresholds.

        Parameters:
            df: DataFrame containing simulation metrics over time
            response_threshold: Fraction of final value to consider as "responded"

        Returns:
            Dictionary of response times for different metrics
        """
        response_times = {}

        # Metrics to analyze (customize based on your needs)
        metrics_to_analyze = [
            'alignment_index', 'shape_index', 'confluency',
            'healthy_cells', 'senescent_fraction'
        ]

        for metric in metrics_to_analyze:
            if metric in df.columns:
                # Get final value
                final_value = df[metric].iloc[-1]

                # Calculate threshold value
                if metric in ['alignment_index', 'shape_index', 'confluency', 'senescent_fraction']:
                    # For metrics where higher is typically the response
                    threshold_value = final_value * response_threshold
                    # Find first time point where value exceeds threshold
                    if df[metric].iloc[0] < threshold_value:
                        response_row = df[df[metric] >= threshold_value].iloc[0] if not df[
                            df[metric] >= threshold_value].empty else None
                    else:
                        # Skip if already above threshold
                        continue
                else:
                    # For metrics like cell counts where the interpretation depends on trend
                    if final_value > df[metric].iloc[0]:  # Increasing trend
                        threshold_value = df[metric].iloc[0] + (final_value - df[metric].iloc[0]) * response_threshold
                        response_row = df[df[metric] >= threshold_value].iloc[0] if not df[
                            df[metric] >= threshold_value].empty else None
                    else:  # Decreasing trend
                        threshold_value = df[metric].iloc[0] - (df[metric].iloc[0] - final_value) * response_threshold
                        response_row = df[df[metric] <= threshold_value].iloc[0] if not df[
                            df[metric] <= threshold_value].empty else None

                # Store response time
                if response_row is not None:
                    time_col = 'time_hours' if 'time_hours' in df.columns else 'time'
                    response_times[metric] = response_row[time_col] if time_col in df.columns else None

        return response_times

    def compute_cell_dynamics_rates(self, df):
        """
        Compute growth, senescence, and death rates from cell count data.

        Parameters:
            df: DataFrame containing cell count metrics over time

        Returns:
            Dictionary with computed rates
        """
        if not all(col in df.columns for col in ['time', 'healthy_cells', 'senescent_tel', 'senescent_stress']):
            return None

        # Prepare results dictionary
        rates = {}

        # Time differences in hours
        if 'time_hours' in df.columns:
            df['dt'] = df['time_hours'].diff()
        else:
            df['dt'] = df['time'].diff() / 60  # Convert minutes to hours

        # Skip first row which has NaN diff
        df_rates = df.iloc[1:].copy()

        # Calculate cell count differences
        df_rates['d_healthy'] = df_rates['healthy_cells'].diff()
        df_rates['d_sen_tel'] = df_rates['senescent_tel'].diff()
        df_rates['d_sen_stress'] = df_rates['senescent_stress'].diff()
        df_rates['d_total'] = df_rates['d_healthy'] + df_rates['d_sen_tel'] + df_rates['d_sen_stress']

        # Calculate instantaneous rates (per hour)
        df_rates['healthy_growth_rate'] = df_rates['d_healthy'] / df_rates['dt'] / df_rates['healthy_cells'].shift(1)
        df_rates['tel_senescence_rate'] = df_rates['d_sen_tel'] / df_rates['dt'] / df_rates['healthy_cells'].shift(1)
        df_rates['stress_senescence_rate'] = df_rates['d_sen_stress'] / df_rates['dt'] / df_rates[
            'healthy_cells'].shift(1)

        # Apply smoothing to reduce noise
        for col in ['healthy_growth_rate', 'tel_senescence_rate', 'stress_senescence_rate']:
            df_rates[f'{col}_smooth'] = gaussian_filter1d(df_rates[col].fillna(0), sigma=3)

        # Average rates over different phases
        early_phase = df_rates.iloc[:len(df_rates) // 3]
        mid_phase = df_rates.iloc[len(df_rates) // 3:2 * len(df_rates) // 3]
        late_phase = df_rates.iloc[2 * len(df_rates) // 3:]

        # Store phase-specific rates
        for phase, phase_df in [('early', early_phase), ('mid', mid_phase), ('late', late_phase)]:
            for col in ['healthy_growth_rate', 'tel_senescence_rate', 'stress_senescence_rate']:
                rates[f'{phase}_{col}'] = phase_df[col].mean()

        # Overall rates
        for col in ['healthy_growth_rate', 'tel_senescence_rate', 'stress_senescence_rate']:
            rates[f'overall_{col}'] = df_rates[col].mean()
            rates[f'peak_{col}'] = df_rates[col].max()

        return rates, df_rates

    def compare_simulations(self, simulators, names=None, save_dir=None):
        """
        Compare results from multiple simulations.

        Parameters:
            simulators: List of Simulator objects or histories
            names: Optional list of simulation names
            save_dir: Directory to save comparison plots

        Returns:
            DataFrame with comparison metrics and dictionary of figures
        """
        # Setup names if not provided
        if names is None:
            names = [f"Simulation {i + 1}" for i in range(len(simulators))]

        # Create output directory if needed
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # Extract histories
        histories = []
        for sim in simulators:
            if hasattr(sim, 'history'):
                histories.append(sim.history)
            else:
                histories.append(sim)  # Assume it's already a history

        # Convert to DataFrames
        dataframes = []
        for i, history in enumerate(histories):
            df = self.history_to_dataframe(history)
            df['simulation'] = names[i]
            dataframes.append(df)

        # Combine all data
        combined_df = pd.concat(dataframes)

        # Extract comparison metrics
        comparison_metrics = []

        for name, df in zip(names, dataframes):
            # Get steady state metrics
            steady_metrics = self.extract_steady_state_metrics(df)

            # Get response times
            response_times = self.calculate_response_times(df)

            # Combine metrics
            metrics = {'simulation': name}
            metrics.update(steady_metrics)
            metrics.update(response_times)

            comparison_metrics.append(metrics)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_metrics)

        # Create comparison plots
        figures = {}

        # 1. Cell population comparison
        if 'healthy_cells' in combined_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))

            for name in names:
                sim_df = combined_df[combined_df['simulation'] == name]
                time_col = 'time_hours' if 'time_hours' in sim_df.columns else 'time'
                ax.plot(sim_df[time_col], sim_df['healthy_cells'], '-', linewidth=2, label=f"{name} (Healthy)")
                ax.plot(sim_df[time_col], sim_df['senescent_tel'] + sim_df['senescent_stress'],
                        '--', linewidth=2, label=f"{name} (Senescent)")

            ax.set_xlabel('Time (hours)', fontsize=12)
            ax.set_ylabel('Cell Count', fontsize=12)
            ax.set_title('Cell Population Comparison', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True)

            figures['population'] = fig

            if save_dir:
                plt.savefig(os.path.join(save_dir, 'comparison_population.png'), dpi=300, bbox_inches='tight')

        # 2. Senescence fraction comparison
        if 'senescent_fraction' in combined_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))

            for name in names:
                sim_df = combined_df[combined_df['simulation'] == name]
                time_col = 'time_hours' if 'time_hours' in sim_df.columns else 'time'
                ax.plot(sim_df[time_col], sim_df['senescent_fraction'], '-', linewidth=2, label=name)

            ax.set_xlabel('Time (hours)', fontsize=12)
            ax.set_ylabel('Senescent Fraction', fontsize=12)
            ax.set_title('Senescence Fraction Comparison', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True)

            figures['senescence'] = fig

            if save_dir:
                plt.savefig(os.path.join(save_dir, 'comparison_senescence.png'), dpi=300, bbox_inches='tight')

        # 3. Spatial metrics comparison (if available)
        if 'alignment_index' in combined_df.columns:
            fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

            for name in names:
                sim_df = combined_df[combined_df['simulation'] == name]
                time_col = 'time_hours' if 'time_hours' in sim_df.columns else 'time'

                axes[0].plot(sim_df[time_col], sim_df['alignment_index'], '-', linewidth=2, label=name)
                axes[1].plot(sim_df[time_col], sim_df['shape_index'], '-', linewidth=2, label=name)
                axes[2].plot(sim_df[time_col], sim_df['confluency'], '-', linewidth=2, label=name)

            axes[0].set_ylabel('Alignment Index', fontsize=12)
            axes[0].set_title('Cell Alignment Comparison', fontsize=12)
            axes[0].legend(fontsize=10)
            axes[0].grid(True)

            axes[1].set_ylabel('Shape Index', fontsize=12)
            axes[1].set_title('Cell Shape Comparison', fontsize=12)
            axes[1].grid(True)

            axes[2].set_ylabel('Confluency', fontsize=12)
            axes[2].set_title('Monolayer Confluency Comparison', fontsize=12)
            axes[2].set_xlabel('Time (hours)', fontsize=12)
            axes[2].grid(True)

            figures['spatial'] = fig

            if save_dir:
                plt.savefig(os.path.join(save_dir, 'comparison_spatial.png'), dpi=300, bbox_inches='tight')

        # Return comparison results
        return comparison_df, combined_df, figures

    def sensitivity_analysis(self, parameter_data, output_metrics, save_dir=None):
        """
        Perform sensitivity analysis to quantify parameter effects on outputs.

        Parameters:
            parameter_data: DataFrame with parameter values and results
            output_metrics: List of output metrics to analyze
            save_dir: Directory to save sensitivity plots

        Returns:
            DataFrame with sensitivity coefficients and dictionary of figures
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Identify parameter columns (those not in output_metrics)
        parameter_cols = [col for col in parameter_data.columns
                          if col not in output_metrics and col != 'simulation']

        # Initialize results
        sensitivity_results = []
        figures = {}

        # For each parameter-output combination
        for param in parameter_cols:
            param_sensitivities = {'parameter': param}

            for output in output_metrics:
                if output in parameter_data.columns:
                    # Calculate correlation coefficient
                    corr, p_value = stats.pearsonr(parameter_data[param], parameter_data[output])

                    # Calculate normalized sensitivity coefficient
                    # S = (dY/Y)/(dX/X) ≈ (X/Y)*(dY/dX) ≈ (X_mean/Y_mean)*slope
                    X_mean = parameter_data[param].mean()
                    Y_mean = parameter_data[output].mean()

                    if Y_mean != 0 and X_mean != 0:
                        # Linear regression to get slope
                        slope, intercept = np.polyfit(parameter_data[param], parameter_data[output], 1)
                        sens_coef = (X_mean / Y_mean) * slope
                    else:
                        sens_coef = np.nan

                    # Store results
                    param_sensitivities[f'{output}_correlation'] = corr
                    param_sensitivities[f'{output}_p_value'] = p_value
                    param_sensitivities[f'{output}_sensitivity'] = sens_coef

            sensitivity_results.append(param_sensitivities)

        # Create sensitivity DataFrame
        sensitivity_df = pd.DataFrame(sensitivity_results)

        # Create sensitivity plots
        for output in output_metrics:
            if output in parameter_data.columns:
                fig, axes = plt.subplots(len(parameter_cols), 1, figsize=(10, 4 * len(parameter_cols)))

                if len(parameter_cols) == 1:
                    axes = [axes]

                for i, param in enumerate(parameter_cols):
                    ax = axes[i]

                    # Scatter plot
                    ax.scatter(parameter_data[param], parameter_data[output], alpha=0.7)

                    # Add regression line
                    if not parameter_data[param].empty and not parameter_data[output].empty:
                        try:
                            m, b = np.polyfit(parameter_data[param], parameter_data[output], 1)
                            x_vals = np.array([parameter_data[param].min(), parameter_data[param].max()])
                            ax.plot(x_vals, m * x_vals + b, 'r-', linewidth=2)

                            # Add correlation and sensitivity info
                            corr = \
                            sensitivity_df.loc[sensitivity_df['parameter'] == param, f'{output}_correlation'].values[0]
                            sens = \
                            sensitivity_df.loc[sensitivity_df['parameter'] == param, f'{output}_sensitivity'].values[0]
                            ax.text(0.05, 0.95, f"Correlation: {corr:.3f}\nSensitivity: {sens:.3f}",
                                    transform=ax.transAxes, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                        except:
                            # Skip regression if it fails
                            pass

                    ax.set_xlabel(param, fontsize=12)
                    ax.set_ylabel(output, fontsize=12)
                    ax.grid(True)

                plt.tight_layout()
                figures[output] = fig

                if save_dir:
                    plt.savefig(os.path.join(save_dir, f'sensitivity_{output}.png'), dpi=300, bbox_inches='tight')

        # Create sensitivity heatmap
        if len(output_metrics) > 0 and len(parameter_cols) > 0:
            # Prepare heatmap data
            heatmap_data = []

            for param in parameter_cols:
                row = []
                for output in output_metrics:
                    if output in parameter_data.columns:
                        sens = sensitivity_df.loc[sensitivity_df['parameter'] == param, f'{output}_sensitivity'].values
                        row.append(sens[0] if len(sens) > 0 else np.nan)
                    else:
                        row.append(np.nan)
                heatmap_data.append(row)

            heatmap_data = np.array(heatmap_data)

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Sensitivity Coefficient', rotation=-90, va="bottom")

            # Set ticks and labels
            ax.set_xticks(np.arange(len(output_metrics)))
            ax.set_yticks(np.arange(len(parameter_cols)))
            ax.set_xticklabels(output_metrics)
            ax.set_yticklabels(parameter_cols)

            # Rotate x labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Add values in cells
            for i in range(len(parameter_cols)):
                for j in range(len(output_metrics)):
                    if not np.isnan(heatmap_data[i, j]):
                        text_color = 'black' if abs(heatmap_data[i, j]) < 0.5 else 'white'
                        ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                                ha="center", va="center", color=text_color)

            ax.set_title("Parameter Sensitivity Heatmap")
            fig.tight_layout()

            figures['heatmap'] = fig

            if save_dir:
                plt.savefig(os.path.join(save_dir, 'sensitivity_heatmap.png'), dpi=300, bbox_inches='tight')

        return sensitivity_df, figures

    def spatial_pattern_analysis(self, simulator, save_path=None):
        """
        Analyze spatial patterns and cell distribution.

        Parameters:
            simulator: Simulator object with current state
            save_path: Directory to save spatial analysis plots

        Returns:
            Dictionary with spatial metrics and figures
        """
        # Initialize results
        results = {}
        figures = {}

        # Get cells from simulator
        cells = simulator.grid.cells

        if not cells:
            return {"error": "No cells in simulator"}

        # Extract cell positions and properties
        positions = []
        orientations = []
        aspect_ratios = []
        areas = []
        cell_types = []

        for cell_id, cell in cells.items():
            positions.append(cell.position)
            orientations.append(cell.orientation)
            aspect_ratios.append(cell.aspect_ratio)
            areas.append(cell.area)

            if cell.is_senescent:
                if cell.senescence_cause == 'telomere':
                    cell_types.append('telomere_senescent')
                else:
                    cell_types.append('stress_senescent')
            else:
                cell_types.append('healthy')

        # Convert to numpy arrays
        positions = np.array(positions)
        orientations = np.array(orientations)
        aspect_ratios = np.array(aspect_ratios)
        areas = np.array(areas)

        # Basic spatial stats
        results['cell_count'] = len(cells)
        results['mean_aspect_ratio'] = np.mean(aspect_ratios)
        results['std_aspect_ratio'] = np.std(aspect_ratios)
        results['mean_area'] = np.mean(areas)
        results['cell_density'] = len(cells) / (simulator.grid.width * simulator.grid.height)

        # Orientation analysis
        mean_orientation = np.mean(np.cos(2 * orientations))
        results['orientation_order_parameter'] = mean_orientation

        # Cell type counts
        type_counts = {cell_type: cell_types.count(cell_type) for cell_type in set(cell_types)}
        for cell_type, count in type_counts.items():
            results[f'{cell_type}_count'] = count
            results[f'{cell_type}_fraction'] = count / len(cells)

        # Radial distribution analysis
        if len(positions) > 1:
            # Calculate center of mass
            center = np.mean(positions, axis=0)

            # Calculate radial distances
            distances = np.sqrt(np.sum((positions - center) ** 2, axis=1))
            results['mean_distance_from_center'] = np.mean(distances)

            # Calculate nearest neighbor distances
            nn_distances = []

            for i, pos in enumerate(positions):
                dists = np.sqrt(np.sum((positions - pos) ** 2, axis=1))
                dists[i] = np.inf  # Exclude self
                nn_distances.append(np.min(dists))

            nn_distances = np.array(nn_distances)
            results['mean_nearest_neighbor_distance'] = np.mean(nn_distances)
            results['std_nearest_neighbor_distance'] = np.std(nn_distances)

            # Spatial distribution plots
            # 1. Cell position scatter plot
            fig1, ax1 = plt.subplots(figsize=(10, 10))

            # Create color map for cell types
            colors = {'healthy': 'green', 'telomere_senescent': 'red', 'stress_senescent': 'blue'}
            scatter_colors = [colors[ct] for ct in cell_types]

            # Plot cells
            scatter = ax1.scatter(positions[:, 0], positions[:, 1], c=scatter_colors, alpha=0.7)

            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=cell_type)
                for cell_type, color in colors.items()]
            ax1.legend(handles=legend_elements, loc='upper right')

            ax1.set_xlabel('X Position (pixels)', fontsize=12)
            ax1.set_ylabel('Y Position (pixels)', fontsize=12)
            ax1.set_title('Cell Spatial Distribution', fontsize=14)
            ax1.set_xlim(0, simulator.grid.width)
            ax1.set_ylim(0, simulator.grid.height)
            ax1.grid(True)

            figures['spatial_distribution'] = fig1

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                fig1.savefig(os.path.join(save_path, 'spatial_distribution.png'), dpi=300, bbox_inches='tight')

            # 2. Orientation rose plot
            fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

            # Create orientation histogram
            bins = np.linspace(-np.pi, np.pi, 37)  # 36 bins of 10 degrees each
            hist, _ = np.histogram(orientations, bins=bins)

            # Plot rose plot
            width = 2 * np.pi / 36
            bars = ax2.bar(bins[:-1], hist, width=width, alpha=0.7)

            # Color bars based on orientation
            cm = plt.cm.viridis
            for i, bar in enumerate(bars):
                bar.set_facecolor(cm(i / 36))

            ax2.set_title('Cell Orientation Distribution', fontsize=14)
            ax2.grid(True)

            figures['orientation_distribution'] = fig2

            if save_path:
                fig2.savefig(os.path.join(save_path, 'orientation_distribution.png'), dpi=300, bbox_inches='tight')

            # 3. Nearest neighbor distance distribution
            fig3, ax3 = plt.subplots(figsize=(10, 6))

            # Plot histogram
            ax3.hist(nn_distances, bins=20, alpha=0.7, color='blue')
            ax3.axvline(x=np.mean(nn_distances), color='r', linestyle='--',
                        label=f'Mean: {np.mean(nn_distances):.2f}')

            ax3.set_xlabel('Nearest Neighbor Distance (pixels)', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_title('Nearest Neighbor Distance Distribution', fontsize=14)
            ax3.legend()
            ax3.grid(True)

            figures['nn_distance_distribution'] = fig3

            if save_path:
                fig3.savefig(os.path.join(save_path, 'nn_distance_distribution.png'), dpi=300, bbox_inches='tight')

        return results, figures

    def time_series_analysis(self, df, metrics=None, smoothing_sigma=2, save_dir=None):
        """
        Perform time series analysis on simulation metrics.

        Parameters:
            df: DataFrame with simulation metrics over time
            metrics: List of metrics to analyze (default: all numeric columns)
            smoothing_sigma: Sigma value for Gaussian smoothing
            save_dir: Directory to save time series plots

        Returns:
            Dictionary with time series analysis results and figures
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Default to all numeric columns if metrics not specified
        if metrics is None:
            metrics = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove certain columns
            for col in ['time', 'time_hours', 'step_count', 'dt']:
                if col in metrics:
                    metrics.remove(col)

        # Check if time column exists
        time_col = 'time_hours' if 'time_hours' in df.columns else 'time'
        if time_col not in df.columns:
            return {"error": "No time column in DataFrame"}

        # Initialize results and figures
        results = {}
        figures = {}

        # Smooth the metrics
        df_smooth = df.copy()
        for metric in metrics:
            if metric in df.columns:
                df_smooth[f'{metric}_smooth'] = gaussian_filter1d(df[metric].fillna(0), sigma=smoothing_sigma)

        # Calculate derivatives (rates of change)
        df_deriv = df_smooth.copy()
        for metric in metrics:
            smooth_col = f'{metric}_smooth'
            if smooth_col in df_smooth.columns:
                # Calculate derivative
                df_deriv[f'{metric}_rate'] = df_smooth[smooth_col].diff() / df_smooth[time_col].diff()

                # Calculate second derivative
                df_deriv[f'{metric}_accel'] = df_deriv[f'{metric}_rate'].diff() / df_smooth[time_col].diff()

                # Skip first two rows which have NaN values
                valid_rows = df_deriv.iloc[2:].copy()

                # Find inflection points (where second derivative crosses zero)
                sign_changes = np.sign(valid_rows[f'{metric}_accel']).diff().fillna(0)
                inflection_indices = valid_rows.index[sign_changes != 0].tolist()

                inflection_points = []
                for idx in inflection_indices:
                    inflection_points.append((df.loc[idx, time_col], df.loc[idx, metric]))

                # Store results
                results[f'{metric}_avg_rate'] = df_deriv[f'{metric}_rate'].mean()
                results[f'{metric}_max_rate'] = df_deriv[f'{metric}_rate'].max()
                results[f'{metric}_min_rate'] = df_deriv[f'{metric}_rate'].min()
                results[f'{metric}_inflection_points'] = inflection_points
                results[f'{metric}_inflection_count'] = len(inflection_points)

                # Create time series plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

                # Plot original and smoothed data
                ax1.plot(df[time_col], df[metric], 'b-', alpha=0.3, label='Original')
                ax1.plot(df_smooth[time_col], df_smooth[smooth_col], 'r-', linewidth=2, label='Smoothed')

                # Mark inflection points
                for time_val, metric_val in inflection_points:
                    ax1.plot(time_val, metric_val, 'go', markersize=8)

                ax1.set_ylabel(metric, fontsize=12)
                ax1.set_title(f'Time Series Analysis: {metric}', fontsize=14)
                ax1.legend(fontsize=10)
                ax1.grid(True)

                # Plot rate of change
                ax2.plot(df_deriv[time_col], df_deriv[f'{metric}_rate'], 'g-', linewidth=2)
                ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

                ax2.set_xlabel(f'Time ({time_col})', fontsize=12)
                ax2.set_ylabel(f'Rate of Change of {metric}', fontsize=12)
                ax2.grid(True)

                plt.tight_layout()
                figures[metric] = fig

                if save_dir:
                    plt.savefig(os.path.join(save_dir, f'time_series_{metric}.png'), dpi=300, bbox_inches='tight')

        # Correlation analysis between metrics
        if len(metrics) > 1:
            # Calculate correlation matrix
            corr_matrix = df[metrics].corr()
            results['correlation_matrix'] = corr_matrix.to_dict()

            # Create correlation heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', ax=ax, vmin=-1, vmax=1)
            ax.set_title('Correlation Between Metrics', fontsize=14)

            figures['correlation'] = fig

            if save_dir:
                plt.savefig(os.path.join(save_dir, 'metric_correlations.png'), dpi=300, bbox_inches='tight')

            # Principal Component Analysis (if enough metrics)
            if len(metrics) >= 3:
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[metrics])

                # Apply PCA
                pca = PCA()
                pca_result = pca.fit_transform(scaled_data)

                # Get explained variance
                explained_var = pca.explained_variance_ratio_
                cumulative_var = np.cumsum(explained_var)

                # Store PCA results
                results['pca_explained_variance'] = explained_var.tolist()
                results['pca_cumulative_variance'] = cumulative_var.tolist()
                results['pca_components'] = pca.components_.tolist()

                # Create PCA plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # Explained variance plot
                ax1.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7)
                ax1.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-', linewidth=2)
                ax1.axhline(y=0.95, color='k', linestyle='--', alpha=0.5)
                ax1.set_xlabel('Principal Component', fontsize=12)
                ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
                ax1.set_title('PCA Explained Variance', fontsize=14)
                ax1.set_xticks(range(1, len(explained_var) + 1))
                ax1.grid(True)

                # First two components scatter plot
                time_points = df[time_col].values
                sc = ax2.scatter(pca_result[:, 0], pca_result[:, 1], c=time_points, cmap='viridis')
                ax2.set_xlabel('PC1', fontsize=12)
                ax2.set_ylabel('PC2', fontsize=12)
                ax2.set_title('First Two Principal Components', fontsize=14)
                fig.colorbar(sc, ax=ax2, label=time_col)
                ax2.grid(True)

                plt.tight_layout()
                figures['pca'] = fig

                if save_dir:
                    plt.savefig(os.path.join(save_dir, 'pca_analysis.png'), dpi=300, bbox_inches='tight')

                # Biplot of PCA loadings
                fig, ax = plt.subplots(figsize=(10, 8))

                # Plot observations
                x_pc = pca_result[:, 0]
                y_pc = pca_result[:, 1]

                # Scale for visibility
                x_scale = 1.0 / (x_pc.max() - x_pc.min())
                y_scale = 1.0 / (y_pc.max() - y_pc.min())

                # We want to time-color the observations
                sc = ax.scatter(x_pc * x_scale, y_pc * y_scale, c=time_points, cmap='viridis', alpha=0.7)
                fig.colorbar(sc, ax=ax, label=time_col)

                # Plot feature arrows
                for i, metric in enumerate(metrics):
                    ax.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
                             color='r', alpha=0.8, head_width=0.05)
                    ax.text(pca.components_[0, i] * 1.1, pca.components_[1, i] * 1.1,
                            metric, color='r', fontsize=10)

                ax.set_xlabel(f'PC1 ({explained_var[0]:.2%} explained var.)', fontsize=12)
                ax.set_ylabel(f'PC2 ({explained_var[1]:.2%} explained var.)', fontsize=12)
                ax.set_title('PCA Biplot: Features and Time Evolution', fontsize=14)
                ax.grid(True)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

                figures['pca_biplot'] = fig

                if save_dir:
                    plt.savefig(os.path.join(save_dir, 'pca_biplot.png'), dpi=300, bbox_inches='tight')

        return results, figures


def analyze_experiment(simulator, analysis_type='basic', save_dir=None):
    """
    Perform comprehensive analysis of an experiment.

    Parameters:
        simulator: Simulator object with results
        analysis_type: Type of analysis to perform ('basic', 'detailed', 'spatial', 'temporal', 'all')
        save_dir: Directory to save analysis results

    Returns:
        Dictionary with analysis results
    """
    # Create analyzer
    analyzer = SimulationAnalyzer(simulator.config)

    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Convert history to DataFrame
    if not simulator.history:
        return {"error": "No history data in simulator"}

    df = analyzer.history_to_dataframe(simulator.history)

    # Initialize results
    results = {
        'analysis_type': analysis_type,
        'simulation_time': simulator.time,
        'cell_count': len(simulator.grid.cells) if hasattr(simulator.grid, 'cells') else 0
    }

    # Perform analysis based on type
    if analysis_type in ['basic', 'all']:
        # Basic metrics
        steady_state = analyzer.extract_steady_state_metrics(df)
        response_times = analyzer.calculate_response_times(df)

        results.update({
            'steady_state': steady_state,
            'response_times': response_times
        })

    if analysis_type in ['detailed', 'all'] and 'healthy_cells' in df.columns:
        # Population dynamics rates
        if 'healthy_cells' in df.columns:
            rates, rates_df = analyzer.compute_cell_dynamics_rates(df)
            results['population_rates'] = rates

            # Save rates data if requested
            if save_dir:
                rates_df.to_csv(os.path.join(save_dir, 'cell_dynamics_rates.csv'), index=False)

    if analysis_type in ['spatial', 'all']:
        # Spatial pattern analysis
        spatial_results, spatial_figures = analyzer.spatial_pattern_analysis(
            simulator,
            save_path=save_dir if save_dir else None
        )
        results['spatial_analysis'] = spatial_results

    if analysis_type in ['temporal', 'all']:
        # Time series analysis
        ts_results, ts_figures = analyzer.time_series_analysis(
            df,
            save_dir=save_dir if save_dir else None
        )
        results['time_series_analysis'] = ts_results

    # Save complete DataFrame if requested
    if save_dir:
        df.to_csv(os.path.join(save_dir, 'simulation_data.csv'), index=False)

        # Save summary results
        with open(os.path.join(save_dir, 'analysis_summary.txt'), 'w') as f:
            f.write("Endothelial Cell Simulation Analysis\n")
            f.write("====================================\n\n")

            f.write(f"Analysis Type: {analysis_type}\n")
            f.write(f"Simulation Time: {simulator.time} {simulator.config.time_unit}\n")
            f.write(f"Cell Count: {len(simulator.grid.cells) if hasattr(simulator.grid, 'cells') else 0}\n\n")

            if 'steady_state' in results:
                f.write("Steady-State Metrics:\n")
                for k, v in results['steady_state'].items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

            if 'response_times' in results:
                f.write("Response Times:\n")
                for k, v in results['response_times'].items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

            if 'population_rates' in results:
                f.write("Population Dynamics Rates:\n")
                for k, v in results['population_rates'].items():
                    if not isinstance(v, list):
                        f.write(f"  {k}: {v}\n")
                f.write("\n")

    return results


def compare_input_patterns(simulator_functions, labels, shear_stress_values, input_types,
                           duration=None, save_dir=None):
    """
    Compare different input patterns (constant, step, ramp, oscillatory) at various shear stress levels.

    Parameters:
        simulator_functions: Dict mapping input type to simulator creation function
        labels: Labels for different input types
        shear_stress_values: List of shear stress values to test
        input_types: List of input pattern types to test
        duration: Simulation duration (if None, use config default)
        save_dir: Directory to save comparison results

    Returns:
        Dictionary with comparison results
    """
    # Create results directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Run simulations for all combinations
    simulators = {}

    for input_type in input_types:
        for stress in shear_stress_values:
            sim_key = f"{input_type}_{stress}"
            sim_label = f"{labels[input_type]} (τ={stress})"

            # Create and run simulator
            simulator = simulator_functions[input_type](stress, duration)
            simulators[sim_key] = {
                'simulator': simulator,
                'label': sim_label,
                'input_type': input_type,
                'shear_stress': stress
            }

    # Create analyzer
    analyzer = SimulationAnalyzer(simulator.config)

    # Compare simulations
    simulator_list = [sim_info['simulator'] for sim_info in simulators.values()]
    label_list = [sim_info['label'] for sim_info in simulators.values()]

    comparison_df, combined_df, figures = analyzer.compare_simulations(
        simulator_list, label_list, save_dir
    )

    # Organize results by input type
    input_results = {}

    for input_type in input_types:
        input_sims = [
            sim_info for key, sim_info in simulators.items()
            if sim_info['input_type'] == input_type
        ]

        input_simulator_list = [sim_info['simulator'] for sim_info in input_sims]
        input_label_list = [sim_info['label'] for sim_info in input_sims]

        # Compare simulations for this input type
        input_comparison_dir = os.path.join(save_dir, input_type) if save_dir else None
        if input_comparison_dir:
            os.makedirs(input_comparison_dir, exist_ok=True)

        input_comparison_df, input_combined_df, input_figures = analyzer.compare_simulations(
            input_simulator_list, input_label_list, input_comparison_dir
        )

        input_results[input_type] = {
            'comparison_df': input_comparison_df,
            'combined_df': input_combined_df,
            'figures': input_figures
        }

    # Create summary plots
    # 1. Response time comparison by input type
    if 'response_times' in comparison_df.columns.values:
        fig, ax = plt.subplots(figsize=(12, 8))

        metrics = [col for col in comparison_df.columns if col.endswith('alignment_index')]

        # Group by input type
        grouped_df = pd.DataFrame()
        grouped_df['simulation'] = label_list

        for metric in metrics:
            grouped_df[metric] = comparison_df[metric]

        # Add input type
        grouped_df['input_type'] = [sim_info['input_type'] for sim_info in simulators.values()]
        grouped_df['shear_stress'] = [sim_info['shear_stress'] for sim_info in simulators.values()]

        # Plot
        input_colors = {'constant': 'blue', 'step': 'green', 'ramp': 'red', 'oscillatory': 'purple'}

        for input_type in input_types:
            input_data = grouped_df[grouped_df['input_type'] == input_type]

            if not input_data.empty and metrics:
                ax.scatter(input_data['shear_stress'], input_data[metrics[0]],
                           label=labels[input_type], color=input_colors.get(input_type, 'black'),
                           s=100, alpha=0.7)

                # Add trend line
                if len(input_data) > 1:
                    try:
                        m, b = np.polyfit(input_data['shear_stress'], input_data[metrics[0]], 1)
                        x_vals = np.array([min(input_data['shear_stress']), max(input_data['shear_stress'])])
                        ax.plot(x_vals, m * x_vals + b, '--', color=input_colors.get(input_type, 'black'))
                    except:
                        pass

        ax.set_xlabel('Shear Stress (Pa)', fontsize=12)
        ax.set_ylabel('Response Time (hours)', fontsize=12)
        ax.set_title('Alignment Response Time by Input Type', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True)

        if save_dir:
            plt.savefig(os.path.join(save_dir, 'response_time_comparison.png'), dpi=300, bbox_inches='tight')

    # Return all results
    return {
        'simulators': simulators,
        'comparison_df': comparison_df,
        'combined_df': combined_df,
        'figures': figures,
        'input_results': input_results
    }