"""
Energy Analysis System for Endothelial Cell Simulation
Provides detailed energy tracking and visualization for biological tessellation optimization.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import os
from collections import defaultdict


class EnergyAnalyzer:
    """
    Comprehensive energy analysis for biological tessellation optimization.
    """
    
    def __init__(self, config):
        """Initialize the energy analyzer."""
        self.config = config
        self.energy_history = []
        self.detailed_energy_history = []
        self.optimization_step_energy = []
        self.per_cell_energy_history = []
        
    def calculate_detailed_energy_breakdown(self, grid):
        """
        Calculate detailed energy breakdown by component and cell.
        
        Parameters:
            grid: Grid object with cells
            
        Returns:
            Dictionary with detailed energy information
        """
        energy_breakdown = {
            'total_energy': 0.0,
            'area_energy': 0.0,
            'aspect_ratio_energy': 0.0,
            'orientation_energy': 0.0,
            'per_cell_energies': [],
            'component_stats': {},
            'cell_count': len(grid.cells)
        }
        
        if not grid.cells:
            return energy_breakdown
            
        area_energies = []
        ar_energies = []
        orientation_energies = []
        
        for cell_id, cell in grid.cells.items():
            cell_energy = {
                'cell_id': cell_id,
                'area_energy': 0.0,
                'ar_energy': 0.0,
                'orientation_energy': 0.0,
                'total_energy': 0.0,
                'is_senescent': cell.is_senescent
            }
            
            # Area energy
            if hasattr(cell, 'target_area') and cell.target_area > 0 and cell.actual_area > 0:
                area_ratio = cell.actual_area / cell.target_area
                area_energy = (area_ratio - 1.0) ** 2
                cell_energy['area_energy'] = area_energy * grid.energy_weights['area']
                area_energies.append(cell_energy['area_energy'])
            
            # Aspect ratio energy
            if hasattr(cell, 'target_aspect_ratio') and cell.target_aspect_ratio > 0:
                ar_ratio = cell.actual_aspect_ratio / cell.target_aspect_ratio
                ar_energy = (ar_ratio - 1.0) ** 2
                cell_energy['ar_energy'] = ar_energy * grid.energy_weights['aspect_ratio']
                ar_energies.append(cell_energy['ar_energy'])
            
            # Orientation energy
            if hasattr(cell, 'target_orientation'):
                target_align = grid.to_alignment_angle(cell.target_orientation)
                actual_align = grid.to_alignment_angle(cell.actual_orientation)
                angle_diff = abs(target_align - actual_align)
                orientation_energy = (angle_diff / (np.pi/2)) ** 2
                cell_energy['orientation_energy'] = orientation_energy * grid.energy_weights['orientation']
                orientation_energies.append(cell_energy['orientation_energy'])
            
            # Total cell energy
            cell_energy['total_energy'] = (cell_energy['area_energy'] + 
                                          cell_energy['ar_energy'] + 
                                          cell_energy['orientation_energy'])
            
            energy_breakdown['per_cell_energies'].append(cell_energy)
        
        # Aggregate energies
        energy_breakdown['area_energy'] = sum(area_energies)
        energy_breakdown['aspect_ratio_energy'] = sum(ar_energies)
        energy_breakdown['orientation_energy'] = sum(orientation_energies)
        energy_breakdown['total_energy'] = (energy_breakdown['area_energy'] + 
                                           energy_breakdown['aspect_ratio_energy'] + 
                                           energy_breakdown['orientation_energy'])
        
        # Component statistics
        energy_breakdown['component_stats'] = {
            'area': {
                'mean': np.mean(area_energies) if area_energies else 0,
                'std': np.std(area_energies) if area_energies else 0,
                'max': np.max(area_energies) if area_energies else 0,
                'count': len(area_energies)
            },
            'aspect_ratio': {
                'mean': np.mean(ar_energies) if ar_energies else 0,
                'std': np.std(ar_energies) if ar_energies else 0,
                'max': np.max(ar_energies) if ar_energies else 0,
                'count': len(ar_energies)
            },
            'orientation': {
                'mean': np.mean(orientation_energies) if orientation_energies else 0,
                'std': np.std(orientation_energies) if orientation_energies else 0,
                'max': np.max(orientation_energies) if orientation_energies else 0,
                'count': len(orientation_energies)
            }
        }
        
        return energy_breakdown
    
    def record_energy_state(self, grid, time_step, optimization_step=None):
        """
        Record energy state at a given time step.
        
        Parameters:
            grid: Grid object
            time_step: Current simulation time step
            optimization_step: Optional optimization step number
        """
        energy_data = self.calculate_detailed_energy_breakdown(grid)
        energy_data['time_step'] = time_step
        energy_data['simulation_time'] = time_step * self.config.time_step
        
        if optimization_step is not None:
            energy_data['optimization_step'] = optimization_step
            self.optimization_step_energy.append(energy_data)
        else:
            self.detailed_energy_history.append(energy_data)
    
    def record_optimization_iteration(self, grid, iteration, step_type="unknown"):
        """
        Record energy state during optimization iterations.
        
        Parameters:
            grid: Grid object
            iteration: Iteration number within optimization
            step_type: Type of optimization step
        """
        energy_data = self.calculate_detailed_energy_breakdown(grid)
        energy_data['iteration'] = iteration
        energy_data['step_type'] = step_type
        energy_data['timestamp'] = len(self.optimization_step_energy)
        
        self.optimization_step_energy.append(energy_data)
    
    def get_energy_evolution_dataframe(self):
        """
        Convert energy history to pandas DataFrame for analysis.
        
        Returns:
            pandas.DataFrame with energy evolution data
        """
        if not self.detailed_energy_history:
            print("No detailed energy history available")
            return None
            
        records = []
        for entry in self.detailed_energy_history:
            record = {
                'time_step': entry['time_step'],
                'simulation_time': entry['simulation_time'],
                'total_energy': entry['total_energy'],
                'area_energy': entry['area_energy'],
                'aspect_ratio_energy': entry['aspect_ratio_energy'],
                'orientation_energy': entry['orientation_energy'],
                'cell_count': entry['cell_count']
            }
            
            # Add component statistics
            for component in ['area', 'aspect_ratio', 'orientation']:
                stats = entry['component_stats'][component]
                record[f'{component}_mean'] = stats['mean']
                record[f'{component}_std'] = stats['std']
                record[f'{component}_max'] = stats['max']
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def plot_energy_evolution(self, save_path=None):
        """
        Create comprehensive energy evolution plots.
        
        Parameters:
            save_path: Path to save the plot
            
        Returns:
            matplotlib.Figure
        """
        df = self.get_energy_evolution_dataframe()
        if df is None or len(df) == 0:
            print("No energy data available for plotting")
            return None
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Convert time to hours if needed
        time_col = 'simulation_time'
        if self.config.time_unit == "minutes":
            df['time_hours'] = df['simulation_time'] / 60
            time_col = 'time_hours'
            time_label = 'Time (hours)'
        else:
            time_label = f'Time ({self.config.time_unit})'
        
        # 1. Total energy evolution
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df[time_col], df['total_energy'], 'k-', linewidth=2, label='Total Energy')
        ax1.set_ylabel('Total Energy', fontsize=12)
        ax1.set_title('Total Biological Energy Evolution', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Component energies
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(df[time_col], df['area_energy'], 'b-', linewidth=2, label='Area')
        ax2.plot(df[time_col], df['aspect_ratio_energy'], 'r-', linewidth=2, label='Aspect Ratio')
        ax2.plot(df[time_col], df['orientation_energy'], 'g-', linewidth=2, label='Orientation')
        ax2.set_ylabel('Component Energy', fontsize=12)
        ax2.set_xlabel(time_label, fontsize=12)
        ax2.set_title('Energy by Component', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Energy per cell (normalized)
        ax3 = fig.add_subplot(gs[1, 1])
        df['energy_per_cell'] = df['total_energy'] / df['cell_count']
        ax3.plot(df[time_col], df['energy_per_cell'], 'purple', linewidth=2)
        ax3.set_ylabel('Energy per Cell', fontsize=12)
        ax3.set_xlabel(time_label, fontsize=12)
        ax3.set_title('Normalized Energy per Cell', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 4. Component mean energies
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(df[time_col], df['area_mean'], 'b--', linewidth=2, label='Area (mean)')
        ax4.plot(df[time_col], df['aspect_ratio_mean'], 'r--', linewidth=2, label='AR (mean)')
        ax4.plot(df[time_col], df['orientation_mean'], 'g--', linewidth=2, label='Orient (mean)')
        ax4.set_ylabel('Mean Component Energy', fontsize=12)
        ax4.set_xlabel(time_label, fontsize=12)
        ax4.set_title('Mean Energy per Component', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Component max energies (outliers)
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(df[time_col], df['area_max'], 'b:', linewidth=2, label='Area (max)')
        ax5.plot(df[time_col], df['aspect_ratio_max'], 'r:', linewidth=2, label='AR (max)')
        ax5.plot(df[time_col], df['orientation_max'], 'g:', linewidth=2, label='Orient (max)')
        ax5.set_ylabel('Max Component Energy', fontsize=12)
        ax5.set_xlabel(time_label, fontsize=12)
        ax5.set_title('Maximum Energy per Component (Outliers)', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Biological Energy Analysis', fontsize=16)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Energy evolution plot saved to: {save_path}")
        
        return fig
    
    def plot_optimization_energy(self, save_path=None, max_iterations=50):
        """
        Plot energy evolution during optimization iterations.
        
        Parameters:
            save_path: Path to save the plot
            max_iterations: Maximum number of iterations to show
            
        Returns:
            matplotlib.Figure
        """
        if not self.optimization_step_energy:
            print("No optimization energy data available")
            return None
        
        # Convert to DataFrame
        opt_records = []
        for entry in self.optimization_step_energy[-max_iterations:]:
            record = {
                'iteration': entry.get('iteration', entry.get('timestamp', 0)),
                'total_energy': entry['total_energy'],
                'area_energy': entry['area_energy'],
                'aspect_ratio_energy': entry['aspect_ratio_energy'],
                'orientation_energy': entry['orientation_energy'],
                'step_type': entry.get('step_type', 'unknown')
            }
            opt_records.append(record)
        
        if not opt_records:
            print("No optimization records to plot")
            return None
            
        opt_df = pd.DataFrame(opt_records)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Total energy
        ax1.plot(opt_df['iteration'], opt_df['total_energy'], 'ko-', linewidth=2, markersize=4)
        ax1.set_ylabel('Total Energy', fontsize=12)
        ax1.set_title('Energy Evolution During Optimization', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Component energies
        ax2.plot(opt_df['iteration'], opt_df['area_energy'], 'b.-', linewidth=2, label='Area')
        ax2.plot(opt_df['iteration'], opt_df['aspect_ratio_energy'], 'r.-', linewidth=2, label='Aspect Ratio')
        ax2.plot(opt_df['iteration'], opt_df['orientation_energy'], 'g.-', linewidth=2, label='Orientation')
        ax2.set_ylabel('Component Energy', fontsize=12)
        ax2.set_xlabel('Optimization Iteration', fontsize=12)
        ax2.set_title('Component Energy Evolution', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add energy reduction annotations
        if len(opt_df) > 1:
            initial_energy = opt_df['total_energy'].iloc[0]
            final_energy = opt_df['total_energy'].iloc[-1]
            reduction = initial_energy - final_energy
            reduction_pct = (reduction / initial_energy) * 100 if initial_energy > 0 else 0
            
            ax1.text(0.02, 0.98, f'Energy Reduction: {reduction:.3f} ({reduction_pct:.1f}%)',
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Optimization energy plot saved to: {save_path}")
        
        return fig
    
    def plot_per_cell_energy_distribution(self, time_step=None, save_path=None):
        """
        Plot energy distribution across individual cells.
        
        Parameters:
            time_step: Specific time step to analyze (default: latest)
            save_path: Path to save the plot
            
        Returns:
            matplotlib.Figure
        """
        if not self.detailed_energy_history:
            print("No detailed energy history available")
            return None
        
        # Get data for specified time step
        if time_step is None:
            energy_data = self.detailed_energy_history[-1]  # Latest
        else:
            # Find closest time step
            target_data = None
            min_diff = float('inf')
            for data in self.detailed_energy_history:
                diff = abs(data['time_step'] - time_step)
                if diff < min_diff:
                    min_diff = diff
                    target_data = data
            energy_data = target_data
        
        if not energy_data or not energy_data['per_cell_energies']:
            print("No per-cell energy data available")
            return None
        
        # Extract per-cell data
        cell_data = energy_data['per_cell_energies']
        total_energies = [cell['total_energy'] for cell in cell_data]
        area_energies = [cell['area_energy'] for cell in cell_data]
        ar_energies = [cell['ar_energy'] for cell in cell_data]
        orient_energies = [cell['orientation_energy'] for cell in cell_data]
        
        # Separate by cell type
        healthy_total = [cell['total_energy'] for cell in cell_data if not cell['is_senescent']]
        senescent_total = [cell['total_energy'] for cell in cell_data if cell['is_senescent']]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Per-Cell Energy Distribution (t={energy_data["simulation_time"]:.1f})', fontsize=16)
        
        # 1. Total energy histogram
        ax1 = axes[0, 0]
        ax1.hist(total_energies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Total Energy per Cell')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Total Energy Distribution')
        ax1.axvline(np.mean(total_energies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(total_energies):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Component energies
        ax2 = axes[0, 1]
        width = 0.25
        x = np.arange(3)
        means = [np.mean(area_energies), np.mean(ar_energies), np.mean(orient_energies)]
        stds = [np.std(area_energies), np.std(ar_energies), np.std(orient_energies)]
        
        bars = ax2.bar(x, means, width, yerr=stds, capsize=5, 
                      color=['blue', 'red', 'green'], alpha=0.7)
        ax2.set_xlabel('Energy Component')
        ax2.set_ylabel('Mean Energy ± Std')
        ax2.set_title('Mean Component Energies')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Area', 'Aspect Ratio', 'Orientation'])
        ax2.grid(True, alpha=0.3)
        
        # 3. Energy by cell type
        ax3 = axes[1, 0]
        if healthy_total and senescent_total:
            ax3.hist([healthy_total, senescent_total], bins=15, alpha=0.7, 
                    color=['green', 'red'], label=['Healthy', 'Senescent'])
            ax3.set_xlabel('Total Energy per Cell')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Energy Distribution by Cell Type')
            ax3.legend()
        elif healthy_total:
            ax3.hist(healthy_total, bins=20, alpha=0.7, color='green', label='Healthy')
            ax3.set_xlabel('Total Energy per Cell')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Healthy Cell Energy Distribution')
            ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Energy scatter plot (area vs orientation)
        ax4 = axes[1, 1]
        colors = ['green' if not cell['is_senescent'] else 'red' for cell in cell_data]
        scatter = ax4.scatter(area_energies, orient_energies, c=colors, alpha=0.6)
        ax4.set_xlabel('Area Energy')
        ax4.set_ylabel('Orientation Energy')
        ax4.set_title('Area vs Orientation Energy')
        
        # Add legend for cell types
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                                 markersize=8, label='Healthy'),
                          Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                 markersize=8, label='Senescent')]
        ax4.legend(handles=legend_elements)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-cell energy distribution plot saved to: {save_path}")
        
        return fig
    
    def get_energy_summary(self):
        """
        Get a summary of energy statistics.
        
        Returns:
            Dictionary with energy summary
        """
        if not self.detailed_energy_history:
            return {"error": "No energy history available"}
        
        # Get latest energy data
        latest = self.detailed_energy_history[-1]
        
        # Calculate trends if we have enough data
        trends = {}
        if len(self.detailed_energy_history) > 5:
            recent_energies = [entry['total_energy'] for entry in self.detailed_energy_history[-5:]]
            trends['recent_trend'] = 'decreasing' if recent_energies[-1] < recent_energies[0] else 'increasing'
            trends['trend_magnitude'] = abs(recent_energies[-1] - recent_energies[0])
        
        summary = {
            'latest_total_energy': latest['total_energy'],
            'latest_area_energy': latest['area_energy'],
            'latest_ar_energy': latest['aspect_ratio_energy'],
            'latest_orientation_energy': latest['orientation_energy'],
            'cell_count': latest['cell_count'],
            'energy_per_cell': latest['total_energy'] / max(1, latest['cell_count']),
            'component_breakdown': {
                'area_fraction': latest['area_energy'] / max(1e-10, latest['total_energy']),
                'ar_fraction': latest['aspect_ratio_energy'] / max(1e-10, latest['total_energy']),
                'orientation_fraction': latest['orientation_energy'] / max(1e-10, latest['total_energy'])
            },
            'component_stats': latest['component_stats'],
            'trends': trends,
            'time_points': len(self.detailed_energy_history)
        }
        
        return summary
    
    def save_energy_data(self, save_path):
        """
        Save energy data to file for external analysis.
        
        Parameters:
            save_path: Path to save the data
        """
        df = self.get_energy_evolution_dataframe()
        if df is not None:
            df.to_csv(save_path, index=False)
            print(f"Energy data saved to: {save_path}")
        else:
            print("No energy data to save")


def integrate_energy_analyzer_with_grid(grid, analyzer):
    """
    Modify grid optimization methods to use energy analyzer.
    
    Parameters:
        grid: Grid object to modify
        analyzer: EnergyAnalyzer instance
    """
    # Store original methods
    original_optimization = grid.optimize_biological_tessellation
    original_adaptive_opt = grid._run_adaptive_optimization
    
    def enhanced_optimization():
        """Enhanced optimization with energy tracking."""
        # Record energy before optimization
        analyzer.record_optimization_iteration(grid, 0, "before_optimization")
        
        # Run original optimization
        result = original_optimization()
        
        # Record energy after optimization  
        analyzer.record_optimization_iteration(grid, 1, "after_optimization")
        
        return result
    
    def enhanced_adaptive_optimization(intensity, initial_energy):
        """Enhanced adaptive optimization with detailed energy tracking."""
        # Record initial state
        analyzer.record_optimization_iteration(grid, 0, f"start_{intensity}")
        
        # Run original optimization with iteration tracking
        params = grid.optimization_params[intensity]
        
        for step in range(params['max_steps']):
            # Record before this step
            analyzer.record_optimization_iteration(grid, step + 1, f"{intensity}_step_{step}")
            
            # Your existing optimization step code here
            position_adjustments = grid._calculate_local_position_adjustments()
            
            movements_applied = 0
            for cell_id, adjustment in position_adjustments.items():
                scaled_adjustment = adjustment * params['displacement_scale']
                
                if np.linalg.norm(scaled_adjustment) > params['movement_threshold']:
                    current_pos = np.array(grid.cell_seeds[cell_id])
                    new_pos = current_pos + scaled_adjustment
                    
                    new_pos[0] = max(20, min(grid.width - 20, new_pos[0]))
                    new_pos[1] = max(20, min(grid.height - 20, new_pos[1]))
                    
                    grid.cell_seeds[cell_id] = tuple(new_pos)
                    grid.cells[cell_id].update_position(tuple(new_pos))
                    movements_applied += 1
            
            if movements_applied == 0:
                break
                
            grid._update_voronoi_tessellation()
            
            # Record after tessellation update
            analyzer.record_optimization_iteration(grid, step + 2, f"{intensity}_after_tessellation_{step}")
            
            current_energy = grid.calculate_biological_energy()
            improvement = initial_energy - current_energy
            
            if improvement < params['convergence_threshold']:
                break
                
            initial_energy = current_energy
    
    # Replace methods
    grid.optimize_biological_tessellation = enhanced_optimization
    grid._run_adaptive_optimization = enhanced_adaptive_optimization
    
    return grid


# Example usage function
def analyze_simulation_energy(simulator, create_plots=True, save_directory=None):
    """
    Comprehensive energy analysis of a simulation.
    
    Parameters:
        simulator: Simulator object with completed simulation
        create_plots: Whether to create visualization plots
        save_directory: Directory to save analysis results
        
    Returns:
        EnergyAnalyzer instance with results
    """
    print("Starting comprehensive energy analysis...")
    
    # Create analyzer
    analyzer = EnergyAnalyzer(simulator.config)
    
    # Analyze energy at each recorded time step
    if simulator.history:
        print(f"Analyzing energy for {len(simulator.history)} time steps...")
        for i, state in enumerate(simulator.history):
            if i % 10 == 0:  # Every 10th step to avoid too much data
                analyzer.record_energy_state(simulator.grid, state['step_count'])
    
    # Record current state
    analyzer.record_energy_state(simulator.grid, simulator.step_count)
    
    # Create visualizations
    figures = []
    if create_plots:
        print("Creating energy visualization plots...")
        
        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
            
            # Energy evolution plot
            evolution_path = os.path.join(save_directory, "energy_evolution.png")
            fig1 = analyzer.plot_energy_evolution(evolution_path)
            if fig1: figures.append(fig1)
            
            # Per-cell energy distribution
            distribution_path = os.path.join(save_directory, "energy_distribution.png")
            fig2 = analyzer.plot_per_cell_energy_distribution(save_path=distribution_path)
            if fig2: figures.append(fig2)
            
            # Optimization energy (if available)
            if analyzer.optimization_step_energy:
                opt_path = os.path.join(save_directory, "optimization_energy.png")
                fig3 = analyzer.plot_optimization_energy(opt_path)
                if fig3: figures.append(fig3)
            
            # Save energy data
            data_path = os.path.join(save_directory, "energy_data.csv")
            analyzer.save_energy_data(data_path)
        else:
            # Just create plots without saving
            fig1 = analyzer.plot_energy_evolution()
            if fig1: figures.append(fig1)
            
            fig2 = analyzer.plot_per_cell_energy_distribution()
            if fig2: figures.append(fig2)
    
    # Print summary
    summary = analyzer.get_energy_summary()
    print("\nENERGY ANALYSIS SUMMARY")
    print("=" * 40)
    print(f"Total Energy: {summary['latest_total_energy']:.4f}")
    print(f"Energy per Cell: {summary['energy_per_cell']:.4f}")
    print(f"Component Breakdown:")
    print(f"  Area: {summary['latest_area_energy']:.4f} ({summary['component_breakdown']['area_fraction']*100:.1f}%)")
    print(f"  Aspect Ratio: {summary['latest_ar_energy']:.4f} ({summary['component_breakdown']['ar_fraction']*100:.1f}%)")
    print(f"  Orientation: {summary['latest_orientation_energy']:.4f} ({summary['component_breakdown']['orientation_fraction']*100:.1f}%)")
    
    if 'recent_trend' in summary['trends']:
        print(f"Recent Trend: {summary['trends']['recent_trend']} (Δ={summary['trends']['trend_magnitude']:.4f})")
    
    print(f"Analysis based on {summary['time_points']} time points")
    print("=" * 40)
    
    return analyzer