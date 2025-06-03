"""
Realistic Tessellation Optimization with Experimental Parameters
Incorporates area, aspect ratio, orientation mean/std, and cell type variations.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ExperimentalParameters:
    """Store experimental parameters with proper variability."""

    def __init__(self):
        # Control cell parameters
        self.control_params = {
            'area': {
                0.0: 11712,  # Static control
                1.4: 11712  # Flow control (assume same area)
            },
            'aspect_ratio': {
                0.0: 1.9,  # Static control
                1.4: 2.3  # Flow control (corrected from 200.3)
            },
            'orientation_mean': {
                0.0: 49.0,  # Random orientation (degrees)
                1.4: 20.0  # Aligned with flow (degrees)
            },
            'orientation_std': {
                0.0: 25.0,  # Standard deviation for static
                1.4: 14.0  # Standard deviation for flow (more aligned)
            }
        }

        # Senescent cell parameters
        self.senescent_params = {
            'area_small': 11995,  # Small senescent cells
            'area_large': 46869,  # Large senescent cells
            'area_threshold': 27174,  # Threshold for small vs large
            'aspect_ratio': {
                0.0: 1.9,  # Static senescent
                1.4: 2.0  # Flow senescent (minimal change)
            },
            'orientation_mean': {
                0.0: 42.0,  # Random orientation static
                1.4: 45.0  # Random orientation flow (no alignment)
            },
            'orientation_std': {
                0.0: 26.0,  # Standard deviation static
                1.4: 27.0  # Standard deviation flow (still random)
            }
        }


class RealisticCell:
    """Cell with realistic experimental parameters and variability."""

    def __init__(self, cell_id, x, y, pressure=1.4, senescent_probability=0.15):
        self.cell_id = cell_id
        self.position = np.array([x, y])
        self.pressure = pressure  # 0.0 (static) or 1.4 (flow)

        # Determine cell type
        self.is_senescent = random.random() < senescent_probability
        self.senescence_cause = 'telomere' if random.random() < 0.5 else 'stress'

        # Get experimental parameters
        self.exp_params = ExperimentalParameters()

        # Set target properties based on experimental data
        self._set_target_properties()

        # Current measured properties (will be updated during optimization)
        self.current_aspect_ratio = 1.0
        self.current_orientation = 0.0
        self.current_area = self.target_area

        # Territory information
        self.territory_vertices = []
        self.territory_area = 0.0

    def _set_target_properties(self):
        """Set target properties based on experimental parameters."""
        if self.is_senescent:
            # Senescent cell targets
            params = self.exp_params.senescent_params

            # Area: randomly choose small or large senescent
            if random.random() < 0.7:  # 70% are small senescent
                self.target_area = params['area_small']
            else:
                self.target_area = params['area_large']

            # Aspect ratio
            self.target_aspect_ratio = params['aspect_ratio'][self.pressure]

            # Orientation (sample from distribution)
            mean_orient = params['orientation_mean'][self.pressure]
            std_orient = params['orientation_std'][self.pressure]
            self.target_orientation = np.random.normal(mean_orient, std_orient)

        else:
            # Control cell targets
            params = self.exp_params.control_params

            # Area
            self.target_area = params['area'][self.pressure]

            # Aspect ratio (add some variability)
            base_ar = params['aspect_ratio'][self.pressure]
            self.target_aspect_ratio = np.random.normal(base_ar, base_ar * 0.1)  # 10% CV

            # Orientation (sample from distribution)
            mean_orient = params['orientation_mean'][self.pressure]
            std_orient = params['orientation_std'][self.pressure]
            self.target_orientation = np.random.normal(mean_orient, std_orient)

        # Convert orientation to radians and clamp aspect ratio
        self.target_orientation = np.radians(self.target_orientation)
        self.target_aspect_ratio = max(1.0, min(4.0, self.target_aspect_ratio))


class PopulationMetrics:
    """Calculate and track population-level metrics."""

    def __init__(self, cells):
        self.cells = cells
        self.control_cells = [c for c in cells if not c.is_senescent]
        self.senescent_cells = [c for c in cells if c.is_senescent]

    def calculate_population_stats(self):
        """Calculate current population statistics."""
        stats = {}

        for cell_type, cell_list in [('control', self.control_cells), ('senescent', self.senescent_cells)]:
            if not cell_list:
                continue

            # Aspect ratios
            ars = [c.current_aspect_ratio for c in cell_list if hasattr(c, 'current_aspect_ratio')]
            if ars:
                stats[f'{cell_type}_ar_mean'] = np.mean(ars)
                stats[f'{cell_type}_ar_std'] = np.std(ars)

            # Orientations (convert to degrees)
            orients = [np.degrees(c.current_orientation) for c in cell_list if hasattr(c, 'current_orientation')]
            if orients:
                # Handle circular statistics for orientation
                orients_rad = np.radians(orients)
                mean_orient = np.degrees(np.arctan2(np.mean(np.sin(orients_rad)), np.mean(np.cos(orients_rad))))
                stats[f'{cell_type}_orient_mean'] = mean_orient
                stats[f'{cell_type}_orient_std'] = np.std(orients)

            # Areas
            areas = [c.territory_area for c in cell_list if hasattr(c, 'territory_area') and c.territory_area > 0]
            if areas:
                stats[f'{cell_type}_area_mean'] = np.mean(areas)
                stats[f'{cell_type}_area_std'] = np.std(areas)

        return stats

    def calculate_target_stats(self):
        """Calculate target population statistics."""
        stats = {}

        for cell_type, cell_list in [('control', self.control_cells), ('senescent', self.senescent_cells)]:
            if not cell_list:
                continue

            # Target aspect ratios
            target_ars = [c.target_aspect_ratio for c in cell_list]
            if target_ars:
                stats[f'{cell_type}_ar_target_mean'] = np.mean(target_ars)
                stats[f'{cell_type}_ar_target_std'] = np.std(target_ars)

            # Target orientations
            target_orients = [np.degrees(c.target_orientation) for c in cell_list]
            if target_orients:
                target_orients_rad = np.radians(target_orients)
                mean_orient = np.degrees(
                    np.arctan2(np.mean(np.sin(target_orients_rad)), np.mean(np.cos(target_orients_rad))))
                stats[f'{cell_type}_orient_target_mean'] = mean_orient
                stats[f'{cell_type}_orient_target_std'] = np.std(target_orients)

            # Target areas
            target_areas = [c.target_area for c in cell_list]
            if target_areas:
                stats[f'{cell_type}_area_target_mean'] = np.mean(target_areas)
                stats[f'{cell_type}_area_target_std'] = np.std(target_areas)

        return stats


def robust_voronoi_construction(positions, width=800, height=600):
    """Construct Voronoi diagram with robust error handling."""
    positions_array = np.array(positions)

    if positions_array.shape[1] != 2:
        positions_array = positions_array[:, :2]

    # Ensure minimum distance between points
    unique_positions = []
    min_distance = min(width, height) * 0.025

    for pos in positions_array:
        for existing_pos in unique_positions:
            if np.linalg.norm(pos - existing_pos) < min_distance:
                angle = np.random.uniform(0, 2 * np.pi)
                perturbation = min_distance * 1.5 * np.array([np.cos(angle), np.sin(angle)])
                pos = pos + perturbation
                break

        pos[0] = np.clip(pos[0], 100, width - 100)
        pos[1] = np.clip(pos[1], 100, height - 100)
        unique_positions.append(pos)

    try:
        vor = Voronoi(unique_positions)
        return vor, [pos.tolist() for pos in unique_positions]
    except Exception as e:
        print(f"    Voronoi failed: {e}, using grid fallback")
        # Grid fallback
        n_points = len(positions)
        grid_size = int(np.ceil(np.sqrt(n_points)))
        fallback_positions = []

        for i in range(n_points):
            grid_x = i % grid_size
            grid_y = i // grid_size
            x = 100 + (grid_x + 1) * (width - 200) / (grid_size + 1)
            y = 100 + (grid_y + 1) * (height - 200) / (grid_size + 1)
            x += np.random.uniform(-30, 30)
            y += np.random.uniform(-30, 30)
            x = np.clip(x, 100, width - 100)
            y = np.clip(y, 100, height - 100)
            fallback_positions.append([x, y])

        vor = Voronoi(fallback_positions)
        return vor, fallback_positions


def update_cell_properties(cells, vor, positions):
    """Update cell properties based on current Voronoi territories."""
    for i, cell in enumerate(cells):
        if i >= len(vor.point_region):
            continue

        region_index = vor.point_region[i]
        if region_index >= len(vor.regions):
            continue

        region = vor.regions[region_index]
        if -1 in region or len(region) < 3:
            continue

        vertices = [vor.vertices[j] for j in region]
        if len(vertices) < 3:
            continue

        # Validate vertices
        valid_vertices = [v for v in vertices if np.isfinite(v).all()]
        if len(valid_vertices) < 3:
            continue

        cell.territory_vertices = valid_vertices

        # Calculate territory area using shoelace formula
        vertices_array = np.array(valid_vertices)
        n = len(vertices_array)
        area = 0.0
        for j in range(n):
            k = (j + 1) % n
            area += vertices_array[j][0] * vertices_array[k][1]
            area -= vertices_array[k][0] * vertices_array[j][1]
        cell.territory_area = abs(area) / 2.0

        # Calculate aspect ratio and orientation
        centroid = np.mean(vertices_array, axis=0)
        centered = vertices_array - centroid

        if len(centered) > 2:
            cov_matrix = np.cov(centered, rowvar=False)
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            if eigenvals[1] > 1e-10:
                cell.current_aspect_ratio = np.sqrt(eigenvals[0] / eigenvals[1])
                cell.current_orientation = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
                cell.current_aspect_ratio = np.clip(cell.current_aspect_ratio, 1.0, 5.0)
            else:
                cell.current_aspect_ratio = 1.0
                cell.current_orientation = 0.0


def calculate_population_fitness(cells):
    """Calculate fitness based on population-level experimental targets."""
    metrics = PopulationMetrics(cells)
    current_stats = metrics.calculate_population_stats()
    target_stats = metrics.calculate_target_stats()

    fitness_components = []

    # Control cell fitness (if present)
    if metrics.control_cells:
        exp_params = ExperimentalParameters()
        pressure = cells[0].pressure

        # Aspect ratio fitness
        if 'control_ar_mean' in current_stats and 'control_ar_target_mean' in target_stats:
            target_ar_mean = exp_params.control_params['aspect_ratio'][pressure]
            current_ar_mean = current_stats['control_ar_mean']
            ar_error = abs(current_ar_mean - target_ar_mean) / target_ar_mean
            ar_fitness = max(0, 1.0 - ar_error)
            fitness_components.append(('control_ar', ar_fitness, 0.4))  # High weight

        # Orientation std fitness (alignment measure)
        if 'control_orient_std' in current_stats:
            target_orient_std = exp_params.control_params['orientation_std'][pressure]
            current_orient_std = current_stats['control_orient_std']
            orient_std_error = abs(current_orient_std - target_orient_std) / target_orient_std
            orient_std_fitness = max(0, 1.0 - orient_std_error)
            fitness_components.append(('control_orient_std', orient_std_fitness, 0.3))

        # Orientation mean fitness
        if 'control_orient_mean' in current_stats:
            target_orient_mean = exp_params.control_params['orientation_mean'][pressure]
            current_orient_mean = current_stats['control_orient_mean']
            # Handle circular difference
            orient_diff = abs(current_orient_mean - target_orient_mean)
            orient_diff = min(orient_diff, 360 - orient_diff)
            orient_mean_error = orient_diff / 90  # Normalize by 90 degrees
            orient_mean_fitness = max(0, 1.0 - orient_mean_error)
            fitness_components.append(('control_orient_mean', orient_mean_fitness, 0.2))

        # Area fitness (lower weight since it's harder to control with Voronoi)
        if 'control_area_mean' in current_stats:
            target_area_mean = exp_params.control_params['area'][pressure]
            current_area_mean = current_stats['control_area_mean']
            area_error = abs(current_area_mean - target_area_mean) / target_area_mean
            area_fitness = max(0, 1.0 - area_error)
            fitness_components.append(('control_area', area_fitness, 0.1))

    # Senescent cell fitness (if present) - lower weight since they're less responsive
    if metrics.senescent_cells:
        exp_params = ExperimentalParameters()
        pressure = cells[0].pressure

        if 'senescent_ar_mean' in current_stats:
            target_ar_mean = exp_params.senescent_params['aspect_ratio'][pressure]
            current_ar_mean = current_stats['senescent_ar_mean']
            ar_error = abs(current_ar_mean - target_ar_mean) / target_ar_mean
            ar_fitness = max(0, 1.0 - ar_error)
            fitness_components.append(('senescent_ar', ar_fitness, 0.1))  # Lower weight

    # Calculate weighted fitness
    if fitness_components:
        total_weight = sum(weight for _, _, weight in fitness_components)
        weighted_fitness = sum(fitness * weight for _, fitness, weight in fitness_components) / total_weight

        # Store component details for debugging
        component_details = {name: fitness for name, fitness, _ in fitness_components}

        return weighted_fitness, component_details
    else:
        return 0.0, {}


def apply_population_based_transformation(cells, positions, width, height, iteration):
    """Apply transformation based on population-level targets."""
    metrics = PopulationMetrics(cells)
    current_stats = metrics.calculate_population_stats()
    exp_params = ExperimentalParameters()
    pressure = cells[0].pressure

    positions_array = np.array(positions)
    center = np.mean(positions_array, axis=0)
    centered = positions_array - center

    # Conservative damping
    damping = max(0.1, 1.0 - iteration * 0.03)

    # Control cell transformations (primary focus)
    control_cells = [c for c in cells if not c.is_senescent]
    if control_cells and 'control_ar_mean' in current_stats:

        # 1. Aspect ratio transformation
        target_ar = exp_params.control_params['aspect_ratio'][pressure]
        current_ar = current_stats['control_ar_mean']
        ar_error = target_ar - current_ar

        if abs(ar_error) > 0.05:
            # Conservative scaling
            max_ar_change = 0.03 * damping
            ar_change = np.clip(ar_error / current_ar, -max_ar_change, max_ar_change)
            scale_factor = 1.0 + ar_change

            # Prevent overshooting
            if current_ar >= target_ar * 0.9:
                scale_factor = np.clip(scale_factor, 0.995, 1.005)
            elif current_ar >= target_ar:
                scale_factor = np.clip(scale_factor, 0.97, 0.995)
            else:
                scale_factor = np.clip(scale_factor, 0.98, 1.03)

            # Apply scaling along flow direction (x-axis)
            flow_dir = np.array([1, 0])
            perp_dir = np.array([0, 1])

            projections = centered @ flow_dir.reshape(-1, 1)
            perp_projections = centered @ perp_dir.reshape(-1, 1)

            centered = (projections * scale_factor * flow_dir.reshape(1, -1) +
                        perp_projections * perp_dir.reshape(1, -1))

        # 2. Orientation alignment transformation
        target_orient_std = exp_params.control_params['orientation_std'][pressure]
        if 'control_orient_std' in current_stats:
            current_orient_std = current_stats['control_orient_std']

            # If orientation std is too high, apply alignment force
            if current_orient_std > target_orient_std * 1.2:
                target_orient_mean = exp_params.control_params['orientation_mean'][pressure]
                alignment_angle = np.radians(target_orient_mean)

                # Small rotation toward target orientation
                rotation_strength = min((current_orient_std - target_orient_std) / target_orient_std, 0.1)
                rotation_angle = rotation_strength * damping * 0.05  # Very small rotation

                cos_r, sin_r = np.cos(rotation_angle), np.sin(rotation_angle)
                rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
                centered = centered @ rotation_matrix.T

    # Return to center and enforce boundaries
    final_positions = centered + center

    margin = 120
    for i, pos in enumerate(final_positions):
        pos[0] = np.clip(pos[0], margin, width - margin)
        pos[1] = np.clip(pos[1], margin, height - margin)

    return final_positions.tolist()


def run_realistic_optimization(cells, width=800, height=600, max_iterations=30, pressure=1.4):
    """Run optimization with realistic experimental parameters."""
    print(f"🔬 Running REALISTIC optimization (Pressure: {pressure} Pa)")
    print("Strategy: Population-based experimental parameter matching")

    positions = [cell.position.tolist() for cell in cells]

    # Print initial targets
    metrics = PopulationMetrics(cells)
    target_stats = metrics.calculate_target_stats()
    exp_params = ExperimentalParameters()

    print(f"  Targets for control cells:")
    print(f"    AR: {exp_params.control_params['aspect_ratio'][pressure]:.2f}")
    print(
        f"    Orientation: {exp_params.control_params['orientation_mean'][pressure]:.1f}° ± {exp_params.control_params['orientation_std'][pressure]:.1f}°")
    print(f"    Area: {exp_params.control_params['area'][pressure]:.0f}")

    best_fitness = 0
    stagnant_count = 0

    for iteration in range(max_iterations):
        try:
            # Create Voronoi and update cell properties
            vor, positions = robust_voronoi_construction(positions, width, height)
            update_cell_properties(cells, vor, positions)

            # Update cell positions
            for i, pos in enumerate(positions):
                if i < len(cells):
                    cells[i].position = np.array(pos)

            # Calculate population fitness
            fitness, components = calculate_population_fitness(cells)

            # Get current population stats
            metrics = PopulationMetrics(cells)
            current_stats = metrics.calculate_population_stats()

            # Print progress
            control_ar = current_stats.get('control_ar_mean', 0)
            control_orient_std = current_stats.get('control_orient_std', 0)
            control_orient_mean = current_stats.get('control_orient_mean', 0)

            print(f"  Iteration {iteration + 1}: Fitness = {fitness:.3f}")
            print(f"    Control AR: {control_ar:.2f}, Orient: {control_orient_mean:.1f}° ± {control_orient_std:.1f}°")

            # Track improvement
            if fitness > best_fitness + 0.01:
                best_fitness = fitness
                stagnant_count = 0
            else:
                stagnant_count += 1

            # Check convergence
            if fitness > 0.85:
                print(f"  ✅ Excellent fitness achieved!")
                break

            if stagnant_count >= 10:
                print(f"  ✅ Converged (stagnant for {stagnant_count} iterations)")
                break

            # Apply transformation
            if fitness < 0.8:
                new_positions = apply_population_based_transformation(
                    cells, positions, width, height, iteration)
                positions = new_positions

            # Light Lloyd's relaxation
            lloyd_positions = []
            for i, pos in enumerate(positions):
                if i >= len(vor.point_region):
                    lloyd_positions.append(pos)
                    continue

                region_index = vor.point_region[i]
                if region_index >= len(vor.regions):
                    lloyd_positions.append(pos)
                    continue

                region = vor.regions[region_index]
                if -1 in region or len(region) == 0:
                    lloyd_positions.append(pos)
                    continue

                vertices = [vor.vertices[j] for j in region]
                if len(vertices) >= 3:
                    valid_vertices = [v for v in vertices if np.isfinite(v).all()]
                    if len(valid_vertices) >= 3:
                        centroid = np.mean(valid_vertices, axis=0)
                        current_pos = np.array(pos)
                        movement = (centroid - current_pos) * 0.02  # Very gentle
                        new_pos = current_pos + movement
                        new_pos[0] = np.clip(new_pos[0], 120, width - 120)
                        new_pos[1] = np.clip(new_pos[1], 120, height - 120)
                        lloyd_positions.append(new_pos.tolist())
                    else:
                        lloyd_positions.append(pos)
                else:
                    lloyd_positions.append(pos)

            positions = lloyd_positions

        except Exception as e:
            print(f"  Error in iteration {iteration + 1}: {e}")
            break

    # Final assessment
    try:
        vor, positions = robust_voronoi_construction(positions, width, height)
        update_cell_properties(cells, vor, positions)
        final_fitness, final_components = calculate_population_fitness(cells)

        metrics = PopulationMetrics(cells)
        final_stats = metrics.calculate_population_stats()

        print(f"\n  FINAL RESULTS:")
        print(f"  Overall Fitness: {final_fitness:.3f}")
        print(f"  Control cells:")
        if 'control_ar_mean' in final_stats:
            print(
                f"    AR: {final_stats['control_ar_mean']:.2f} (target: {exp_params.control_params['aspect_ratio'][pressure]:.2f})")
        if 'control_orient_mean' in final_stats and 'control_orient_std' in final_stats:
            print(
                f"    Orientation: {final_stats['control_orient_mean']:.1f}° ± {final_stats['control_orient_std']:.1f}°")
            print(
                f"    (target: {exp_params.control_params['orientation_mean'][pressure]:.1f}° ± {exp_params.control_params['orientation_std'][pressure]:.1f}°)")

    except Exception as e:
        print(f"  Final assessment failed: {e}")
        final_fitness = 0

    return cells, positions, vor, final_fitness, iteration + 1


def initialize_realistic_cells(n_cells, pressure=1.4, width=800, height=600):
    """Initialize cells with realistic experimental parameters."""
    cells = []
    positions = []

    # Grid-based initialization
    grid_size = int(np.ceil(np.sqrt(n_cells)))
    x_spacing = (width - 300) / grid_size
    y_spacing = (height - 300) / grid_size

    for i in range(n_cells):
        grid_x = i % grid_size
        grid_y = i // grid_size

        x = 150 + grid_x * x_spacing + np.random.uniform(-x_spacing * 0.15, x_spacing * 0.15)
        y = 150 + grid_y * y_spacing + np.random.uniform(-y_spacing * 0.15, y_spacing * 0.15)

        x = np.clip(x, 150, width - 150)
        y = np.clip(y, 150, height - 150)

        cell = RealisticCell(i, x, y, pressure=pressure)
        cells.append(cell)
        positions.append([x, y])

    return cells, positions


def plot_realistic_results(cells, vor, positions, pressure, title_suffix=""):
    """Plot results with realistic experimental context."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Main tessellation plot
    ax_main = axes[0, 0]
    voronoi_plot_2d(vor, ax=ax_main, show_vertices=False, line_colors='black',
                    line_width=1, point_size=0)

    # Color territories
    for i, cell in enumerate(cells):
        if i >= len(vor.point_region):
            continue

        region_index = vor.point_region[i]
        if region_index >= len(vor.regions):
            continue

        region = vor.regions[region_index]
        if -1 in region or len(region) == 0:
            continue

        vertices = [vor.vertices[j] for j in region]
        if len(vertices) < 3:
            continue

        polygon = Polygon(vertices, alpha=0.7, linewidth=1.2, edgecolor='black')

        # Color by cell type
        if not cell.is_senescent:
            color = 'lightgreen'
        elif cell.senescence_cause == 'telomere':
            color = 'lightcoral'
        else:
            color = 'lightblue'

        polygon.set_facecolor(color)
        ax_main.add_patch(polygon)

    ax_main.set_xlim(0, 800)
    ax_main.set_ylim(0, 600)
    ax_main.set_aspect('equal')
    ax_main.set_title(f'Realistic Tessellation (Pressure: {pressure} Pa){title_suffix}')

    # Add flow direction arrow
    if pressure > 0:
        arrow_start = [50, 550]
        arrow_end = [150, 550]
        ax_main.annotate('', xy=arrow_end, xytext=arrow_start,
                         arrowprops=dict(arrowstyle='->', lw=3, color='red'))
        ax_main.text(100, 570, 'Flow Direction', ha='center', fontsize=10, color='red', weight='bold')

    # Statistics plots
    metrics = PopulationMetrics(cells)
    current_stats = metrics.calculate_population_stats()
    target_stats = metrics.calculate_target_stats()
    exp_params = ExperimentalParameters()

    # Aspect ratio distribution
    ax_ar = axes[0, 1]
    control_cells = [c for c in cells if not c.is_senescent]
    senescent_cells = [c for c in cells if c.is_senescent]

    if control_cells:
        control_ars = [c.current_aspect_ratio for c in control_cells if hasattr(c, 'current_aspect_ratio')]
        ax_ar.hist(control_ars, bins=15, alpha=0.7, label='Control', color='green')
        target_ar = exp_params.control_params['aspect_ratio'][pressure]
        ax_ar.axvline(target_ar, color='green', linestyle='--', linewidth=2, label=f'Target Control: {target_ar:.1f}')

    if senescent_cells:
        senescent_ars = [c.current_aspect_ratio for c in senescent_cells if hasattr(c, 'current_aspect_ratio')]
        ax_ar.hist(senescent_ars, bins=15, alpha=0.7, label='Senescent', color='red')
        target_ar_sen = exp_params.senescent_params['aspect_ratio'][pressure]
        ax_ar.axvline(target_ar_sen, color='red', linestyle='--', linewidth=2,
                      label=f'Target Senescent: {target_ar_sen:.1f}')

    ax_ar.set_xlabel('Aspect Ratio')
    ax_ar.set_ylabel('Count')
    ax_ar.set_title('Aspect Ratio Distribution')
    ax_ar.legend()

    # Orientation distribution
    ax_orient = axes[1, 0]
    if control_cells:
        control_orients = [np.degrees(c.current_orientation) for c in control_cells if
                           hasattr(c, 'current_orientation')]
        ax_orient.hist(control_orients, bins=15, alpha=0.7, label='Control', color='green')
        target_orient = exp_params.control_params['orientation_mean'][pressure]
        target_std = exp_params.control_params['orientation_std'][pressure]
        ax_orient.axvline(target_orient, color='green', linestyle='--', linewidth=2,
                          label=f'Target: {target_orient:.0f}°±{target_std:.0f}°')

    ax_orient.set_xlabel('Orientation (degrees)')
    ax_orient.set_ylabel('Count')
    ax_orient.set_title('Orientation Distribution')
    ax_orient.legend()

    # Summary statistics
    ax_summary = axes[1, 1]
    ax_summary.axis('off')

    summary_text = f"EXPERIMENTAL PARAMETERS SUMMARY\n"
    summary_text += f"Pressure: {pressure} Pa\n\n"

    if 'control_ar_mean' in current_stats:
        target_ar = exp_params.control_params['aspect_ratio'][pressure]
        current_ar = current_stats['control_ar_mean']
        ar_achievement = (current_ar / target_ar) * 100
        summary_text += f"CONTROL CELLS:\n"
        summary_text += f"  AR: {current_ar:.2f} (target: {target_ar:.2f}) - {ar_achievement:.1f}%\n"

        if 'control_orient_mean' in current_stats and 'control_orient_std' in current_stats:
            target_orient_mean = exp_params.control_params['orientation_mean'][pressure]
            target_orient_std = exp_params.control_params['orientation_std'][pressure]
            current_orient_mean = current_stats['control_orient_mean']
            current_orient_std = current_stats['control_orient_std']

            summary_text += f"  Orientation: {current_orient_mean:.1f}°±{current_orient_std:.1f}°\n"
            summary_text += f"  (target: {target_orient_mean:.1f}°±{target_orient_std:.1f}°)\n"

        summary_text += "\n"

    if len(senescent_cells) > 0:
        summary_text += f"SENESCENT CELLS: {len(senescent_cells)}/{len(cells)} ({len(senescent_cells) / len(cells) * 100:.1f}%)\n"

    # Add fitness information
    fitness, components = calculate_population_fitness(cells)
    summary_text += f"\nOVERALL FITNESS: {fitness:.3f}\n"

    if fitness > 0.8:
        summary_text += "✅ EXCELLENT MATCH\n"
    elif fitness > 0.6:
        summary_text += "✅ GOOD MATCH\n"
    else:
        summary_text += "⚠️ NEEDS IMPROVEMENT\n"

    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')

    # Add legend for cell types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', edgecolor='black', label='Control'),
        Patch(facecolor='lightcoral', edgecolor='black', label='Telomere-Senescent'),
        Patch(facecolor='lightblue', edgecolor='black', label='Stress-Senescent')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig


def run_pressure_comparison(n_cells=25):
    """Compare static vs flow conditions."""
    print("🔬 REALISTIC PRESSURE COMPARISON")
    print("=" * 60)

    results = {}

    # Test static condition (0.0 Pa)
    print("Testing STATIC condition (0.0 Pa)...")
    np.random.seed(42)
    random.seed(42)

    cells_static, positions_static = initialize_realistic_cells(n_cells, pressure=0.0)
    cells_static, positions_static, vor_static, fitness_static, iterations_static = run_realistic_optimization(
        cells_static, pressure=0.0, max_iterations=25)

    results['static'] = {
        'cells': cells_static,
        'positions': positions_static,
        'vor': vor_static,
        'fitness': fitness_static,
        'iterations': iterations_static
    }

    # Test flow condition (1.4 Pa)
    print(f"\n{'=' * 60}")
    print("Testing FLOW condition (1.4 Pa)...")
    np.random.seed(42)
    random.seed(42)

    cells_flow, positions_flow = initialize_realistic_cells(n_cells, pressure=1.4)
    cells_flow, positions_flow, vor_flow, fitness_flow, iterations_flow = run_realistic_optimization(
        cells_flow, pressure=1.4, max_iterations=25)

    results['flow'] = {
        'cells': cells_flow,
        'positions': positions_flow,
        'vor': vor_flow,
        'fitness': fitness_flow,
        'iterations': iterations_flow
    }

    # Create comparison plots
    print(f"\n{'=' * 60}")
    print("Creating comparison visualization...")

    fig1 = plot_realistic_results(cells_static, vor_static, positions_static, 0.0, " - Static")
    plt.savefig('realistic_tessellation_static.png', dpi=300, bbox_inches='tight')

    fig2 = plot_realistic_results(cells_flow, vor_flow, positions_flow, 1.4, " - Flow")
    plt.savefig('realistic_tessellation_flow.png', dpi=300, bbox_inches='tight')

    # Summary comparison
    print(f"\n{'=' * 60}")
    print("📊 FINAL COMPARISON")
    print("=" * 60)

    print(f"STATIC (0.0 Pa):")
    print(f"  Fitness: {fitness_static:.3f}")
    print(f"  Iterations: {iterations_static}")

    metrics_static = PopulationMetrics(cells_static)
    stats_static = metrics_static.calculate_population_stats()
    if 'control_ar_mean' in stats_static:
        print(f"  Control AR: {stats_static['control_ar_mean']:.2f}")
        print(f"  Control Orientation Std: {stats_static.get('control_orient_std', 0):.1f}°")

    print(f"\nFLOW (1.4 Pa):")
    print(f"  Fitness: {fitness_flow:.3f}")
    print(f"  Iterations: {iterations_flow}")

    metrics_flow = PopulationMetrics(cells_flow)
    stats_flow = metrics_flow.calculate_population_stats()
    if 'control_ar_mean' in stats_flow:
        print(f"  Control AR: {stats_flow['control_ar_mean']:.2f}")
        print(f"  Control Orientation Std: {stats_flow.get('control_orient_std', 0):.1f}°")

    print(f"\n🎯 KEY FINDINGS:")
    exp_params = ExperimentalParameters()
    static_target_ar = exp_params.control_params['aspect_ratio'][0.0]
    flow_target_ar = exp_params.control_params['aspect_ratio'][1.4]

    print(f"  • Static target AR: {static_target_ar}")
    print(f"  • Flow target AR: {flow_target_ar}")
    print(f"  • Flow should show higher alignment (lower orientation std)")
    print(f"  • Senescent cells should be less responsive to flow")

    try:
        plt.show()
    except:
        print("   Note: Check saved PNG files for visualizations")

    return results


if __name__ == "__main__":
    try:
        results = run_pressure_comparison(n_cells=25)

        print("\n" + "=" * 60)
        print("✅ REALISTIC OPTIMIZATION COMPLETE!")
        print("=" * 60)
        print("Key features implemented:")
        print("• Experimental parameter distributions")
        print("• Population-level optimization")
        print("• Control vs senescent cell types")
        print("• Static vs flow conditions")
        print("• Orientation variability (std) optimization")
        print("• Multi-objective fitness function")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()