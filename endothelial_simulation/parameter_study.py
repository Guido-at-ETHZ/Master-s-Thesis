"""
Parameter study module for endothelial cell mechanotransduction simulations.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import multiprocessing
import time

from config import SimulationConfig, create_full_config
from core import Simulator
from visualization import Plotter


def run_parameter_combination(params):
    """
    Run a simulation with a specific parameter combination.

    Parameters:
        params: Dictionary with parameter values and indices

    Returns:
        Dictionary with results
    """
    # Extract parameters
    param_idx = params['idx']
    shear_stress = params['shear_stress']
    senolytic_conc = params.get('senolytic_conc', 0)
    stem_cell_rate = params.get('stem_cell_rate', 10)
    duration = params.get('duration', 1080)

    # Create configuration
    config = create_full_config()
    config.simulation_duration = duration
    config.senolytic_concentration = senolytic_conc

    # Create simulator
    simulator = Simulator(config)

    # Initialize with cells
    simulator.initialize()

    # Set constant input
    simulator.set_constant_input(shear_stress)

    # Run simulation
    print(f"Running combination {param_idx}: τ={shear_stress}, seno={senolytic_conc}, stem={stem_cell_rate}")
    start_time = time.time()
    results = simulator.run()
    run_time = time.time() - start_time
    print(f"Completed combination {param_idx} in {run_time:.1f} seconds")

    # Extract key metrics from final state
    if simulator.history:
        final_state = simulator.history[-1]

        # Get metrics
        metrics = {
            'idx': param_idx,
            'shear_stress': shear_stress,
            'senolytic_conc': senolytic_conc,
            'stem_cell_rate': stem_cell_rate,
            'duration': duration,
            'run_time': run_time,
            'total_cells': final_state.get('cells', 0)
        }

        # Add detailed metrics if available
        if 'healthy_cells' in final_state:
            metrics.update({
                'healthy_cells': final_state['healthy_cells'],
                'senescent_tel': final_state['senescent_tel'],
                'senescent_stress': final_state['senescent_stress'],
                'senescent_fraction': (final_state['senescent_tel'] + final_state['senescent_stress']) /
                                      max(1, final_state['cells']),
                'telomere_length': final_state.get('telomere_length', np.nan)
            })

        if 'alignment_index' in final_state:
            metrics.update({
                'alignment_index': final_state['alignment_index'],
                'shape_index': final_state['shape_index'],
                'confluency': final_state['confluency']
            })
    else:
        # No history data
        metrics = {
            'idx': param_idx,
            'shear_stress': shear_stress,
            'senolytic_conc': senolytic_conc,
            'stem_cell_rate': stem_cell_rate,
            'duration': duration,
            'run_time': run_time,
            'total_cells': 0
        }

    # Generate a plot for this combination
    plotter = Plotter(config)
    prefix = f"param_study_idx{param_idx}_tau{shear_stress}_seno{senolytic_conc}_stem{stem_cell_rate}"
    plotter.create_all_plots(simulator, prefix=prefix)

    # Save results file
    results_file = os.path.join(config.plot_directory, f"{prefix}_results.npz")
    simulator.save_results(filename=f"{prefix}_results.npz")

    metrics['results_file'] = results_file

    return metrics


def run_parameter_study(shear_stress_values, senolytic_conc_values=None, stem_cell_rate_values=None,
                        duration=None, parallel=True, n_processes=None):
    """
    Run a parameter study by simulating all combinations of parameter values.

    Parameters:
        shear_stress_values: List of shear stress values to test
        senolytic_conc_values: List of senolytic concentration values (default: [0])
        stem_cell_rate_values: List of stem cell rate values (default: [10])
        duration: Simulation duration (default: from config)
        parallel: Whether to run simulations in parallel
        n_processes: Number of parallel processes (default: CPU count)

    Returns:
        DataFrame with results
    """
    # Set default parameter values if not provided
    if senolytic_conc_values is None:
        senolytic_conc_values = [0]

    if stem_cell_rate_values is None:
        stem_cell_rate_values = [10]

    if duration is None:
        config = create_full_config()
        duration = config.simulation_duration

    # Create output directory
    config = create_full_config()
    os.makedirs(config.plot_directory, exist_ok=True)

    # Generate all parameter combinations
    param_combinations = list(product(shear_stress_values, senolytic_conc_values, stem_cell_rate_values))
    n_combinations = len(param_combinations)

    print(f"Running parameter study with {n_combinations} combinations:")
    print(f"  Shear stress values: {shear_stress_values}")
    print(f"  Senolytic concentration values: {senolytic_conc_values}")
    print(f"  Stem cell rate values: {stem_cell_rate_values}")
    print(f"  Duration: {duration} minutes")

    # Prepare parameter dictionaries
    params_list = []
    for idx, (tau, seno, stem) in enumerate(param_combinations):
        params_list.append({
            'idx': idx,
            'shear_stress': tau,
            'senolytic_conc': seno,
            'stem_cell_rate': stem,
            'duration': duration
        })

    # Run simulations
    results = []

    if parallel and n_combinations > 1:
        # Use multiple processes
        if n_processes is None:
            n_processes = min(multiprocessing.cpu_count(), n_combinations)

        print(f"Running in parallel with {n_processes} processes")

        with multiprocessing.Pool(n_processes) as pool:
            results = pool.map(run_parameter_combination, params_list)
    else:
        # Run sequentially
        print("Running sequentially")
        for params in params_list:
            result = run_parameter_combination(params)
            results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(config.plot_directory, f"parameter_study_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    return results_df


def plot_parameter_study_results(results_df, output_dir=None):
    """
    Create visualizations of parameter study results.

    Parameters:
        results_df: DataFrame with parameter study results
        output_dir: Directory to save plots (default: from config)

    Returns:
        List of created figure objects
    """
    if output_dir is None:
        config = create_full_config()
        output_dir = config.plot_directory

    os.makedirs(output_dir, exist_ok=True)

    # Create timestamp for filenames
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Extract parameter values
    shear_stress_values = sorted(results_df['shear_stress'].unique())
    senolytic_conc_values = sorted(results_df['senolytic_conc'].unique())
    stem_cell_rate_values = sorted(results_df['stem_cell_rate'].unique())

    # Check which metrics are available
    has_detailed_metrics = 'healthy_cells' in results_df.columns
    has_spatial_metrics = 'alignment_index' in results_df.columns

    figures = []

    # Plot 1: Cell counts vs shear stress
    if len(senolytic_conc_values) == 1 and len(stem_cell_rate_values) == 1:
        # Simple case: just vary shear stress
        fig, ax = plt.subplots(figsize=(10, 6))

        if has_detailed_metrics:
            # Plot detailed cell counts
            healthy = results_df['healthy_cells'].values
            sen_tel = results_df['senescent_tel'].values
            sen_stress = results_df['senescent_stress'].values

            ax.plot(shear_stress_values, healthy, 'g-o', linewidth=2, label='Healthy Cells')
            ax.plot(shear_stress_values, sen_tel, 'r-s', linewidth=2, label='Telomere-Induced Senescent')
            ax.plot(shear_stress_values, sen_stress, 'b-^', linewidth=2, label='Stress-Induced Senescent')
            ax.plot(shear_stress_values, healthy + sen_tel + sen_stress, 'k--', linewidth=1, label='Total Cells')
        else:
            # Just plot total cells
            total_cells = results_df['total_cells'].values
            ax.plot(shear_stress_values, total_cells, 'k-o', linewidth=2, label='Total Cells')

        ax.set_xlabel('Shear Stress (Pa)', fontsize=12)
        ax.set_ylabel('Cell Count', fontsize=12)
        ax.set_title('Cell Population vs Shear Stress', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"param_study_cell_counts_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        figures.append(fig)
    else:
        # More complex case: multiple parameters
        # Create grid of plots for cell counts
        fig_cell_counts = plt.figure(figsize=(15, 10))

        for i, seno in enumerate(senolytic_conc_values):
            for j, stem in enumerate(stem_cell_rate_values):
                ax = fig_cell_counts.add_subplot(len(senolytic_conc_values), len(stem_cell_rate_values),
                                                 i * len(stem_cell_rate_values) + j + 1)

                # Filter data for this parameter combination
                mask = (results_df['senolytic_conc'] == seno) & (results_df['stem_cell_rate'] == stem)
                subset = results_df[mask].sort_values('shear_stress')

                if has_detailed_metrics:
                    # Plot detailed cell counts
                    ax.plot(subset['shear_stress'], subset['healthy_cells'], 'g-o', linewidth=2, label='Healthy')
                    ax.plot(subset['shear_stress'], subset['senescent_tel'], 'r-s', linewidth=2, label='Tel-Sen')
                    ax.plot(subset['shear_stress'], subset['senescent_stress'], 'b-^', linewidth=2, label='Stress-Sen')
                else:
                    # Just plot total cells
                    ax.plot(subset['shear_stress'], subset['total_cells'], 'k-o', linewidth=2, label='Total')

                ax.set_title(f'Seno={seno}, Stem={stem}', fontsize=10)
                ax.set_xlabel('Shear Stress (Pa)', fontsize=8)
                ax.set_ylabel('Cell Count', fontsize=8)
                ax.tick_params(labelsize=8)
                ax.grid(True)

                if i == 0 and j == 0:
                    ax.legend(fontsize=8)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"param_study_cell_counts_grid_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        figures.append(fig_cell_counts)

    # Plot 2: Senescence fraction vs shear stress (if available)
    if has_detailed_metrics:
        if len(senolytic_conc_values) == 1 and len(stem_cell_rate_values) == 1:
            # Simple case: just vary shear stress
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(shear_stress_values, results_df['senescent_fraction'].values, 'r-o', linewidth=2)

            ax.set_xlabel('Shear Stress (Pa)', fontsize=12)
            ax.set_ylabel('Senescent Fraction', fontsize=12)
            ax.set_title('Senescent Cell Fraction vs Shear Stress', fontsize=14)
            ax.set_ylim(0, 1)
            ax.grid(True)

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"param_study_senescence_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            figures.append(fig)
        else:
            # More complex case: multiple parameters
            # Create grid of plots for senescence fraction
            fig_sen = plt.figure(figsize=(15, 10))

            for i, seno in enumerate(senolytic_conc_values):
                for j, stem in enumerate(stem_cell_rate_values):
                    ax = fig_sen.add_subplot(len(senolytic_conc_values), len(stem_cell_rate_values),
                                             i * len(stem_cell_rate_values) + j + 1)

                    # Filter data for this parameter combination
                    mask = (results_df['senolytic_conc'] == seno) & (results_df['stem_cell_rate'] == stem)
                    subset = results_df[mask].sort_values('shear_stress')

                    ax.plot(subset['shear_stress'], subset['senescent_fraction'], 'r-o', linewidth=2)

                    ax.set_title(f'Seno={seno}, Stem={stem}', fontsize=10)
                    ax.set_xlabel('Shear Stress (Pa)', fontsize=8)
                    ax.set_ylabel('Senescent Fraction', fontsize=8)
                    ax.set_ylim(0, 1)
                    ax.tick_params(labelsize=8)
                    ax.grid(True)

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"param_study_senescence_grid_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            figures.append(fig_sen)

    # Plot 3: Spatial metrics vs shear stress (if available)
    if has_spatial_metrics:
        if len(senolytic_conc_values) == 1 and len(stem_cell_rate_values) == 1:
            # Simple case: just vary shear stress
            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

            axes[0].plot(shear_stress_values, results_df['alignment_index'].values, 'b-o', linewidth=2)
            axes[0].set_ylabel('Alignment Index', fontsize=12)
            axes[0].set_title('Cell Alignment with Flow Direction', fontsize=12)
            axes[0].set_ylim(0, 1)
            axes[0].grid(True)

            axes[1].plot(shear_stress_values, results_df['shape_index'].values, 'g-o', linewidth=2)
            axes[1].set_ylabel('Shape Index', fontsize=12)
            axes[1].set_title('Cell Shape Index', fontsize=12)
            axes[1].grid(True)

            axes[2].plot(shear_stress_values, results_df['confluency'].values, 'r-o', linewidth=2)
            axes[2].set_ylabel('Confluency', fontsize=12)
            axes[2].set_title('Monolayer Confluency', fontsize=12)
            axes[2].set_ylim(0, 1)
            axes[2].set_xlabel('Shear Stress (Pa)', fontsize=12)
            axes[2].grid(True)

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"param_study_spatial_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            figures.append(fig)

    # Plot 4: 3D surface plot of healthy cells vs shear stress and senolytic conc
    if has_detailed_metrics and len(stem_cell_rate_values) == 1 and len(senolytic_conc_values) > 1:
        from mpl_toolkits.mplot3d import Axes3D

        # Create regular grid of shear stress and senolytic values
        shear_points = len(shear_stress_values)
        seno_points = len(senolytic_conc_values)

        X, Y = np.meshgrid(shear_stress_values, senolytic_conc_values)
        Z = np.zeros((seno_points, shear_points))

        # Fill in healthy cell counts
        for i, seno in enumerate(senolytic_conc_values):
            for j, tau in enumerate(shear_stress_values):
                mask = (results_df['senolytic_conc'] == seno) & (results_df['shear_stress'] == tau)
                if mask.any():
                    Z[i, j] = results_df[mask]['healthy_cells'].values[0]

        # Create surface plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='k', linewidth=0.5)

        ax.set_xlabel('Shear Stress (Pa)', fontsize=12)
        ax.set_ylabel('Senolytic Concentration', fontsize=12)
        ax.set_zlabel('Healthy Cell Count', fontsize=12)
        ax.set_title('Healthy Cell Count vs Shear Stress and Senolytic Concentration', fontsize=14)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"param_study_3d_surface_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        figures.append(fig)

    return figures


def find_optimal_senolytic_concentration(shear_stress_values, senolytic_conc_values, output_dir=None):
    """
    Find the optimal senolytic concentration for each shear stress value.

    Parameters:
        shear_stress_values: List of shear stress values to test
        senolytic_conc_values: List of senolytic concentration values to test
        output_dir: Directory to save plots (default: from config)

    Returns:
        DataFrame with optimal concentrations
    """
    # Run parameter study
    results_df = run_parameter_study(shear_stress_values, senolytic_conc_values)

    # Set output directory
    if output_dir is None:
        config = create_full_config()
        output_dir = config.plot_directory

    # Create timestamp for filenames
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Find optimal concentration for each shear stress value
    optimal_results = []

    for tau in shear_stress_values:
        # Filter data for this shear stress
        tau_data = results_df[results_df['shear_stress'] == tau]

        # Find concentration with maximum healthy cells
        optimal_idx = tau_data['healthy_cells'].idxmax()
        optimal_row = tau_data.loc[optimal_idx]

        optimal_results.append({
            'shear_stress': tau,
            'optimal_senolytic_conc': optimal_row['senolytic_conc'],
            'healthy_cells': optimal_row['healthy_cells'],
            'senescent_tel': optimal_row['senescent_tel'],
            'senescent_stress': optimal_row['senescent_stress'],
            'total_cells': optimal_row['healthy_cells'] + optimal_row['senescent_tel'] + optimal_row[
                'senescent_stress'],
            'senescent_fraction': optimal_row['senescent_fraction']
        })

    # Create DataFrame
    optimal_df = pd.DataFrame(optimal_results)

    # Save results to CSV
    csv_path = os.path.join(output_dir, f"optimal_senolytic_{timestamp}.csv")
    optimal_df.to_csv(csv_path, index=False)
    print(f"Optimal senolytic concentrations saved to {csv_path}")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(optimal_df['shear_stress'], optimal_df['optimal_senolytic_conc'], 'bo-', linewidth=2)

    ax.set_xlabel('Shear Stress (Pa)', fontsize=12)
    ax.set_ylabel('Optimal Senolytic Concentration', fontsize=12)
    ax.set_title('Optimal Senolytic Concentration vs Shear Stress', fontsize=14)
    ax.grid(True)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"optimal_senolytic_plot_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return optimal_df


def main():
    """
    Main function for running the parameter study.
    """
    # Define parameter ranges
    shear_stress_values = [0, 5, 10, 15, 20, 30, 45]
    senolytic_conc_values = [0, 5, 10, 20]
    stem_cell_rate_values = [0, 5, 10, 20]

    # Reduced parameter set for testing
    # shear_stress_values = [5, 15]
    # senolytic_conc_values = [0, 10]
    # stem_cell_rate_values = [10]

    # Run parameter study
    print("Running parameter study...")
    results_df = run_parameter_study(shear_stress_values, senolytic_conc_values, stem_cell_rate_values)

    # Plot results
    print("Plotting results...")
    figures = plot_parameter_study_results(results_df)

    # Find optimal senolytic concentration
    print("Finding optimal senolytic concentrations...")
    optimal_df = find_optimal_senolytic_concentration(shear_stress_values, senolytic_conc_values)

    print("Parameter study completed successfully.")


if __name__ == "__main__":
    main()