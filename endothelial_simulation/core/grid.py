"""
Enhanced Grid module with biological energy minimization that integrates with existing temporal dynamics.
"""
import os
import time

import numpy as np
from scipy.spatial.distance import cdist
from .cell import Cell
from .holes import Hole, HoleManager
import random
from typing import List, Tuple, Dict, Optional


class Grid:
    """
    Enhanced Grid class that couples tessellation with biological target properties
    while preserving existing temporal dynamics.
    """

    def __init__(self, width, height, config):
        """Initialize the biological grid with gradient-based energy optimization."""
        self.width = width
        self.height = height
        self.config = config

        # Computational grid (reduced resolution for performance)
        self.computation_scale = 4
        self.comp_width = width // self.computation_scale
        self.comp_height = height // self.computation_scale

        print(f"Grid initialized: Display {width}x{height}, Computation {self.comp_width}x{self.comp_height}")

        # Core data structures
        self.cells = {}
        self.pixel_ownership = np.full((self.comp_height, self.comp_width), -1, dtype=int)
        self.pixel_coords = self._create_pixel_coordinate_grid()
        self.next_cell_id = 0
        self.cell_seeds = {}
        self.territory_map = {}

        # Biological optimization parameters
        self.energy_weights = {
            'area': 1.0,  # Weight for area deviation
            'aspect_ratio': 5.5,  # Weight for aspect ratio deviation
            'orientation': 5.0,  # Weight for orientation deviation
            'overlap': 2.0,  # Weight for preventing overlap
            'boundary': 0.1  # Weight for staying within bounds
        }

        # NEW: Gradient-based optimization parameters
        self.energy_history = []
        self.energy_window = 5  # Track last 5 energy values
        self.stuck_threshold = 3  # Steps with minimal improvement = "stuck"
        self.stuck_counter = 0

        # NEW: Intensity-based optimization parameters
        self.optimization_params = {
            'gentle': {
                'max_steps': 1,
                'displacement_scale': 0.3,
                'convergence_threshold': 0.001,
                'movement_threshold': 1.5
            },
            'moderate': {
                'max_steps': 2,
                'displacement_scale': 0.6,
                'convergence_threshold': 0.01,
                'movement_threshold': 1.0
            },
            'active': {
                'max_steps': 4,
                'displacement_scale': 0.9,
                'convergence_threshold': 0.05,
                'movement_threshold': 0.5
            },
            'crisis': {
                'max_steps': 6,
                'displacement_scale': 1.2,
                'convergence_threshold': 0.1,
                'movement_threshold': 0.3
            },
            'escape': {  # Special mode for escaping local minima
                'max_steps': 8,
                'displacement_scale': 1.5,
                'convergence_threshold': 0.2,
                'movement_threshold': 0.1
            }
        }

        # Adaptation parameters
        self.adaptation_strength = 0.25  # How strongly cells adapt toward targets

        # Keep original grid parameters for compatibility
        self.global_pressure = 1.0
        self.adaptation_rate = 0.1

        # Initialize step counter
        self._adaptation_step_counter = 0

        # Hole management system
        self.hole_manager = HoleManager(self) if getattr(config, 'enable_holes', True) else None
        self.holes_enabled = getattr(config, 'enable_holes', True)

        if self.holes_enabled:
            print(f"🕳️  Hole management enabled (max {getattr(config, 'max_holes', 5)} holes)")


    def add_cell(self, position=None, divisions=0, is_senescent=False, senescence_cause=None, target_area=100.0):
        """Add a new cell with biological properties."""
        if position is None:
            margin = 20
            position = (
                np.random.uniform(margin, self.width - margin),
                np.random.uniform(margin, self.height - margin)
            )

        cell_id = self.next_cell_id
        self.next_cell_id += 1

        comp_target_area = target_area / (self.computation_scale ** 2)
        cell = Cell(cell_id, position, divisions, is_senescent, senescence_cause, comp_target_area)

        # Add to data structures
        self.cells[cell_id] = cell
        self.cell_seeds[cell_id] = position

        return cell

    def calculate_biological_energy(self):
        """
        Calculate the total biological energy of the system.
        Lower energy means better fit to target properties.
        """
        total_energy = 0.0

        for cell in self.cells.values():
            if not hasattr(cell, 'target_area') or cell.target_area is None:
                continue

            # Area deviation energy
            if cell.actual_area > 0 and cell.target_area > 0:
                area_ratio = cell.actual_area / cell.target_area
                area_energy = (area_ratio - 1.0) ** 2
                total_energy += self.energy_weights['area'] * area_energy

            # Aspect ratio deviation energy
            if hasattr(cell, 'target_aspect_ratio') and cell.target_aspect_ratio > 0:
                ar_ratio = cell.actual_aspect_ratio / cell.target_aspect_ratio
                ar_energy = (ar_ratio - 1.0) ** 2
                total_energy += self.energy_weights['aspect_ratio'] * ar_energy

            # Orientation deviation energy
            if hasattr(cell, 'target_orientation'):
                target_align = self.to_alignment_angle(cell.target_orientation)
                actual_align = self.to_alignment_angle(cell.actual_orientation)
                angle_diff = abs(target_align - actual_align)
                orientation_energy = (angle_diff / (np.pi/2)) ** 2
                total_energy += self.energy_weights['orientation'] * orientation_energy

        return total_energy

    def update_biological_adaptation(self):
        """
        Update biological adaptation with gradient-based optimization.
        """
        self._adaptation_step_counter += 1

        # Always use gradient-based adaptive optimization
        self.optimize_biological_tessellation()

    def optimize_biological_tessellation(self):
        """
        Gradient-based adaptive optimization - responds to energy trajectory.
        """
        if len(self.cells) < 2:
            return

        current_energy = self.calculate_biological_energy()
        self.energy_history.append(current_energy)

        # Keep only recent history
        if len(self.energy_history) > self.energy_window:
            self.energy_history.pop(0)

        # Calculate optimization intensity based on energy trajectory
        intensity = self._calculate_optimization_intensity(current_energy)

        # Apply optimization with calculated intensity
        self._run_adaptive_optimization(intensity, current_energy)

    def _calculate_optimization_intensity(self, current_energy):
        """Calculate optimization intensity based on energy dynamics."""

        if len(self.energy_history) < 3:
            return 'moderate'  # Default for early steps

        # Calculate energy gradient (rate of change)
        recent_energies = self.energy_history[-3:]
        energy_gradient = (recent_energies[-1] - recent_energies[0]) / 2
        energy_variance = np.var(recent_energies)

        # Debug output if enabled
        if hasattr(self.config, 'debug_adaptive') and self.config.debug_adaptive:
            print(f"  Energy gradient: {energy_gradient:.4f}, variance: {energy_variance:.4f}")

        # Detect different scenarios
        if energy_gradient > 0.1:  # Energy increasing - crisis!
            self.stuck_counter = 0
            return 'crisis'

        elif abs(energy_gradient) < 0.01 and energy_variance < 0.001:  # Stuck/oscillating
            self.stuck_counter += 1
            if self.stuck_counter > self.stuck_threshold:
                return 'escape'  # Special mode to escape local minima
            else:
                return 'active'

        elif energy_gradient < -0.05:  # Rapidly improving
            self.stuck_counter = 0
            return 'gentle'  # Don't interfere with good progress

        elif current_energy < 1.0:  # Already quite good
            self.stuck_counter = 0
            return 'gentle'

        else:  # Moderate improvement needed
            self.stuck_counter = 0
            return 'moderate'

    def _run_adaptive_optimization(self, intensity, initial_energy):
        """Run optimization with specified intensity parameters."""

        if intensity not in self.optimization_params:
            intensity = 'moderate'  # Fallback

        params = self.optimization_params[intensity]

        if hasattr(self.config, 'debug_adaptive') and self.config.debug_adaptive:
            print(f"Step {self._adaptation_step_counter}: Energy={initial_energy:.2f} → {intensity} mode")

        # Run optimization with adaptive parameters
        for step in range(params['max_steps']):
            position_adjustments = self._calculate_local_position_adjustments()

            # Apply adaptive scaling
            movements_applied = 0
            for cell_id, adjustment in position_adjustments.items():
                scaled_adjustment = adjustment * params['displacement_scale']

                # Apply movement threshold from adaptive parameters
                if np.linalg.norm(scaled_adjustment) > params['movement_threshold']:
                    current_pos = np.array(self.cell_seeds[cell_id])
                    new_pos = current_pos + scaled_adjustment

                    # Constrain to grid bounds
                    new_pos[0] = max(20, min(self.width - 20, new_pos[0]))
                    new_pos[1] = max(20, min(self.height - 20, new_pos[1]))

                    self.cell_seeds[cell_id] = tuple(new_pos)
                    self.cells[cell_id].update_position(tuple(new_pos))
                    movements_applied += 1

            if movements_applied == 0:
                if hasattr(self.config, 'debug_adaptive') and self.config.debug_adaptive:
                    print(f"    Converged at step {step} (no movements)")
                break

            # Update tessellation after movements
            self._update_voronoi_tessellation()

            # Check convergence
            current_energy = self.calculate_biological_energy()
            improvement = initial_energy - current_energy

            if hasattr(self.config, 'debug_adaptive') and self.config.debug_adaptive:
                print(f"    Step {step}: energy {current_energy:.2f} (Δ{improvement:+.3f})")

            if improvement < params['convergence_threshold']:
                if hasattr(self.config, 'debug_adaptive') and self.config.debug_adaptive:
                    print(f"    Converged at step {step} (energy threshold)")
                break

            initial_energy = current_energy

    def _calculate_local_position_adjustments(self):
        """
        Calculate position adjustments for each cell based on biological targets.
        These targets are evolving via your temporal dynamics system.
        """
        adjustments = {}

        for cell_id, cell in self.cells.items():
            if not hasattr(cell, 'target_area') or cell.target_area is None:
                adjustments[cell_id] = np.array([0.0, 0.0])
                continue

            total_force = np.array([0.0, 0.0])

            # Force from area mismatch
            if cell.actual_area > 0 and cell.target_area > 0:
                area_ratio = cell.actual_area / cell.target_area

                if area_ratio < 0.9:  # Cell needs to grow
                    # Move away from neighbors to claim more space
                    repulsion_force = self._calculate_repulsion_force(cell_id)
                    total_force += self.adaptation_strength * repulsion_force

                elif area_ratio > 1.1:  # Cell needs to shrink
                    # Move toward centroid to reduce territory
                    if cell.centroid is not None:
                        current_pos = np.array(self.cell_seeds[cell_id])
                        display_centroid = self._comp_to_display_coords(cell.centroid[0], cell.centroid[1])
                        centroid_force = np.array(display_centroid) - current_pos
                        total_force += self.adaptation_strength * centroid_force * 0.5

            # Force from aspect ratio mismatch
            if (hasattr(cell, 'target_aspect_ratio') and hasattr(cell, 'target_orientation') and
                cell.target_aspect_ratio > 0):

                ar_error = cell.target_aspect_ratio - cell.actual_aspect_ratio

                if abs(ar_error) > 0.2:
                    # Move in direction that promotes desired aspect ratio
                    stretch_direction = np.array([
                        np.cos(cell.target_orientation),
                        np.sin(cell.target_orientation)
                    ])

                    # Scale force for extreme aspect ratios (like your 200.3 value)
                    ar_force_magnitude = np.tanh(ar_error * 0.05) * 6  # Reduced for extreme values
                    ar_force = ar_force_magnitude * stretch_direction
                    total_force += ar_force

            # Force from orientation mismatch (ALIGNMENT-AWARE)
            if hasattr(cell, 'target_orientation') and cell.territory_pixels:
                # Convert to alignment angles for comparison
                target_align = np.abs(cell.target_orientation) % (np.pi / 2)
                actual_align = np.abs(cell.actual_orientation) % (np.pi / 2)

                alignment_error = target_align - actual_align

                if abs(alignment_error) > np.radians(5):  # ~5 degrees threshold
                    # Move perpendicular to current orientation to change it
                    perp_direction = np.array([
                        -np.sin(cell.actual_orientation),
                        np.cos(cell.actual_orientation)
                    ])

                    if alignment_error < 0:
                        perp_direction = -perp_direction

                    orientation_force_magnitude = abs(alignment_error) * 4
                    orientation_force = orientation_force_magnitude * perp_direction
                    total_force += orientation_force

            # Limit maximum displacement per step
            force_magnitude = np.linalg.norm(total_force)
            max_displacement = 12.0  # Reasonable default
            if force_magnitude > max_displacement:
                total_force = total_force / force_magnitude * max_displacement

            adjustments[cell_id] = total_force

        return adjustments

    def _calculate_repulsion_force(self, cell_id):
        """Calculate repulsion force from neighboring cells."""
        current_pos = np.array(self.cell_seeds[cell_id])
        repulsion_force = np.array([0.0, 0.0])

        for other_id, other_pos in self.cell_seeds.items():
            if other_id == cell_id:
                continue

            other_pos_array = np.array(other_pos)
            distance_vector = current_pos - other_pos_array
            distance = np.linalg.norm(distance_vector)

            if distance > 0:
                # Repulsion inversely proportional to distance
                repulsion_magnitude = min(40.0 / distance, 15.0)
                repulsion_direction = distance_vector / distance
                repulsion_force += repulsion_magnitude * repulsion_direction

        return repulsion_force

    def to_alignment_angle(self, angle_rad):
        """Convert any angle to [0, π/2] flow alignment angle"""
        return np.abs(angle_rad) % (np.pi / 2)

    def _update_voronoi_tessellation(self):
        """
        Update Voronoi tessellation with area-based weighting.
        """
        if not self.cells:
            self.pixel_ownership.fill(-1)
            self.territory_map.clear()
            return

        seed_points = []
        cell_ids = []
        cell_weights = []

        # UPDATE PIXEL COORDINATES TO EXCLUDE HOLES
        if self.holes_enabled and self.hole_manager:
            self.pixel_coords = self._create_pixel_coordinate_grid_with_holes()

            # Mark hole pixels in ownership map
            hole_pixels = self.hole_manager.get_hole_pixels()
            for x, y in hole_pixels:
                if 0 <= x < self.comp_width and 0 <= y < self.comp_height:
                    self.pixel_ownership[y, x] = -999  # Special value for holes

        for cell_id, display_pos in self.cell_seeds.items():
            comp_pos = self._display_to_comp_coords(display_pos[0], display_pos[1])
            seed_points.append(comp_pos)
            cell_ids.append(cell_id)

            # Weight based on target area (evolving via temporal dynamics)
            cell = self.cells[cell_id]
            if hasattr(cell, 'target_area') and cell.target_area > 0:
                weight = np.sqrt(cell.target_area)
            else:
                weight = 10.0
            cell_weights.append(weight)

        if len(seed_points) < 2:
            if len(seed_points) == 1:
                cell_id = cell_ids[0]
                all_pixels = [(x, y) for x in range(self.comp_width) for y in range(self.comp_height)]
                self._assign_territory_to_cell(cell_id, all_pixels)
            return

        seed_array = np.array(seed_points)
        weights_array = np.array(cell_weights)

        # Weighted Voronoi: d_weighted = distance - weight_factor
        distances = cdist(self.pixel_coords, seed_array)

        # Apply weights
        weight_factor = 0.08  # Adjusted for better balance
        for i, weight in enumerate(weights_array):
            distances[:, i] -= weight_factor * weight

        nearest_seed_indices = np.argmin(distances, axis=1)

        self.territory_map.clear()
        self.pixel_ownership.fill(-1)

        for i, cell_id in enumerate(cell_ids):
            pixel_indices = np.where(nearest_seed_indices == i)[0]
            pixels = [(int(self.pixel_coords[idx][0]), int(self.pixel_coords[idx][1]))
                     for idx in pixel_indices]
            self._assign_territory_to_cell(cell_id, pixels)

    # Include all the other necessary methods from the previous response
    def _angle_difference(self, target, actual):
        """Calculate angle difference considering 0°/180° equivalence for flow direction."""
        target_norm = target % (2 * np.pi)
        actual_norm = actual % (2 * np.pi)
        diff1 = target_norm - actual_norm
        diff2 = (target_norm + np.pi) % (2 * np.pi) - actual_norm
        diff1 = ((diff1 + np.pi) % (2 * np.pi)) - np.pi
        diff2 = ((diff2 + np.pi) % (2 * np.pi)) - np.pi
        if abs(diff1) <= abs(diff2):
            return diff1
        else:
            return diff2

    def _display_to_comp_coords(self, x, y):
        """Convert display coordinates to computational coordinates."""
        return (x / self.computation_scale, y / self.computation_scale)

    def _comp_to_display_coords(self, x, y):
        """Convert computational coordinates to display coordinates."""
        return (x * self.computation_scale, y * self.computation_scale)

    def _create_pixel_coordinate_grid(self):
        """Create a grid of pixel coordinates for computation."""
        y_coords, x_coords = np.mgrid[0:self.comp_height, 0:self.comp_width]
        return np.column_stack([x_coords.ravel(), y_coords.ravel()])

    def _assign_territory_to_cell(self, cell_id, pixels):
        """Assign a territory to a cell."""
        self.territory_map[cell_id] = pixels

        for x, y in pixels:
            if 0 <= x < self.comp_width and 0 <= y < self.comp_height:
                self.pixel_ownership[y, x] = cell_id

        if cell_id in self.cells:
            self.cells[cell_id].assign_territory(pixels)

    # Add all the missing methods from your original Grid class
    def adapt_cell_properties(self):
        """Adapt cell properties based on space constraints and targets."""
        if not self.cells:
            return

        total_target_area = sum(cell.target_area for cell in self.cells.values())
        total_available_area = self.comp_width * self.comp_height
        self.global_pressure = total_target_area / total_available_area if total_available_area > 0 else 1.0

        for cell in self.cells.values():
            local_space_factor = min(1.0, 1.0 / self.global_pressure)
            cell.adapt_to_constraints(local_space_factor)

    def remove_cell(self, cell_id):
        """Remove a cell."""
        if cell_id not in self.cells:
            return False

        del self.cells[cell_id]
        if cell_id in self.cell_seeds:
            del self.cell_seeds[cell_id]
        if cell_id in self.territory_map:
            del self.territory_map[cell_id]

        return True

    def apply_shear_stress_field(self, shear_stress_function, duration):
        """Apply a shear stress field to all cells."""
        for cell in self.cells.values():
            if cell.centroid is not None:
                display_centroid = self._comp_to_display_coords(cell.centroid[0], cell.centroid[1])
                x, y = display_centroid
            else:
                x, y = cell.position
            shear_stress = shear_stress_function(x, y)
            cell.apply_shear_stress(shear_stress, duration)

    def calculate_confluency(self):
        """Calculate the confluency."""
        return 1.0

    def calculate_packing_efficiency(self):
        """Calculate packing efficiency."""
        if not self.cells:
            return 1.0

        total_deviation = 0
        for cell in self.cells.values():
            if cell.target_area > 0:
                deviation = abs(cell.actual_area - cell.target_area) / cell.target_area
                total_deviation += deviation

        average_deviation = total_deviation / len(self.cells)
        return max(0, 1.0 - average_deviation)

    def optimize_cell_positions(self, iterations=2):
        """
        Enhanced version that combines original Lloyd algorithm with biological optimization.
        """
        for iteration in range(iterations):
            position_updates = {}

            for cell_id, cell in self.cells.items():
                if cell.territory_pixels and cell.centroid is not None:
                    # Original Lloyd algorithm: move toward centroid
                    display_centroid = self._comp_to_display_coords(cell.centroid[0], cell.centroid[1])
                    current_pos = self.cell_seeds[cell_id]

                    # Calculate movement toward centroid
                    centroid_movement = np.array(display_centroid) - np.array(current_pos)

                    # Add biological adaptation force
                    adjustments = self._calculate_local_position_adjustments()
                    biological_force = adjustments.get(cell_id, np.array([0.0, 0.0]))

                    # Combine movements (70% geometric, 30% biological)
                    total_movement = centroid_movement * 0.7 + biological_force * 0.3

                    # Apply smoothing
                    smoothing_factor = 0.3
                    new_pos = (
                        current_pos[0] * (1 - smoothing_factor) + (current_pos[0] + total_movement[0]) * smoothing_factor,
                        current_pos[1] * (1 - smoothing_factor) + (current_pos[1] + total_movement[1]) * smoothing_factor
                    )

                    # Constrain to grid bounds
                    new_pos = (
                        max(0, min(self.width - 1, new_pos[0])),
                        max(0, min(self.height - 1, new_pos[1]))
                    )

                    position_updates[cell_id] = new_pos

            # Apply position updates
            for cell_id, new_pos in position_updates.items():
                self.cell_seeds[cell_id] = new_pos
                self.cells[cell_id].update_position(new_pos)

            # Update tessellation
            self._update_voronoi_tessellation()

    def get_biological_fitness(self):
        """Get current biological fitness (0-1, where 1 is perfect)."""
        if not self.cells:
            return 1.0

        total_fitness = 0.0
        cell_count = 0

        for cell in self.cells.values():
            if not hasattr(cell, 'target_orientation') or cell.is_senescent:
                continue

            cell_fitness = 0.0
            component_count = 0

            # Orientation fitness
            if hasattr(cell, 'actual_orientation'):
                orientation_error = abs(self._angle_difference(cell.target_orientation, cell.actual_orientation))
                orientation_fitness = max(0, 1.0 - orientation_error / np.pi)
                cell_fitness += orientation_fitness
                component_count += 1

            # Aspect ratio fitness
            if hasattr(cell, 'target_aspect_ratio') and hasattr(cell, 'actual_aspect_ratio'):
                if cell.target_aspect_ratio > 0:
                    ar_ratio = min(cell.actual_aspect_ratio / cell.target_aspect_ratio,
                                  cell.target_aspect_ratio / cell.actual_aspect_ratio)
                    cell_fitness += ar_ratio
                    component_count += 1

            # Area fitness
            if hasattr(cell, 'target_area') and cell.actual_area > 0:
                if cell.target_area > 0:
                    area_ratio = min(cell.actual_area / cell.target_area,
                                   cell.target_area / cell.actual_area)
                    cell_fitness += area_ratio
                    component_count += 1

            if component_count > 0:
                total_fitness += cell_fitness / component_count
                cell_count += 1

        return total_fitness / cell_count if cell_count > 0 else 1.0

    def count_cells_by_type(self):
        """Count cells by type."""
        normal_count = 0
        tel_sen_count = 0
        stress_sen_count = 0

        for cell in self.cells.values():
            if not cell.is_senescent:
                normal_count += 1
            elif cell.senescence_cause == 'telomere':
                tel_sen_count += 1
            elif cell.senescence_cause == 'stress':
                stress_sen_count += 1

        return {
            'normal': normal_count,
            'telomere_senescent': tel_sen_count,
            'stress_senescent': stress_sen_count,
            'total': len(self.cells)
        }

    def get_grid_statistics(self):
        """Get comprehensive statistics about the grid state."""
        if not self.cells:
            return {}

        cell_counts = self.count_cells_by_type()
        biological_energy = self.calculate_biological_energy()
        biological_fitness = self.get_biological_fitness()

        target_areas = [cell.target_area * (self.computation_scale ** 2) for cell in self.cells.values()]
        actual_areas = [cell.actual_area * (self.computation_scale ** 2) for cell in self.cells.values()]
        compression_ratios = [cell.compression_ratio for cell in self.cells.values()]

        aspect_ratios = [cell.actual_aspect_ratio for cell in self.cells.values()]
        orientations = [cell.actual_orientation for cell in self.cells.values()]

        # Add hole statistics
        hole_stats = self.get_hole_statistics()

        return {
            'cell_counts': cell_counts,
            'global_pressure': self.global_pressure,
            'packing_efficiency': self.calculate_packing_efficiency(),
            'biological_fitness': biological_fitness,
            'biological_energy': biological_energy,
            'hole_statistics': hole_stats,
            'area_stats': {
                'mean_target_area': np.mean(target_areas),
                'mean_actual_area': np.mean(actual_areas),
                'mean_compression_ratio': np.mean(compression_ratios),
                'std_compression_ratio': np.std(compression_ratios)
            },
            'shape_stats': {
                'mean_aspect_ratio': np.mean(aspect_ratios),
                'std_aspect_ratio': np.std(aspect_ratios),
                'mean_orientation': np.mean(orientations),
                'std_orientation': np.std(orientations)
            }
        }

    def get_display_territories(self):
        """Get cell territories scaled up to display resolution."""
        display_territories = {}

        for cell_id, comp_pixels in self.territory_map.items():
            display_pixels = []
            for comp_x, comp_y in comp_pixels:
                for dx in range(self.computation_scale):
                    for dy in range(self.computation_scale):
                        display_x = comp_x * self.computation_scale + dx
                        display_y = comp_y * self.computation_scale + dy
                        if 0 <= display_x < self.width and 0 <= display_y < self.height:
                            display_pixels.append((display_x, display_y))
            display_territories[cell_id] = display_pixels

        return display_territories

    def populate_grid(self, count, division_distribution=None, area_distribution=None):
        """Populate the grid with initial cells."""
        created_cells = []

        if division_distribution is None:
            max_div = self.config.max_divisions
            def division_distribution():
                r = np.random.random()
                return int(max_div * (1 - np.sqrt(r)))

        if area_distribution is None:
            def area_distribution():
                base_area = (self.width * self.height) / count
                return np.random.uniform(base_area * 0.7, base_area * 1.3)

        positions = self._generate_poisson_disk_samples(count)

        print(f"Creating {count} cells...")

        for i, position in enumerate(positions):
            if i % 50 == 0:
                print(f"Created {i}/{count} cells")

            divisions = division_distribution()
            target_area = area_distribution()

            is_senescent = False
            senescence_cause = None

            if divisions >= self.config.max_divisions:
                is_senescent = True
                senescence_cause = 'telomere'
            elif np.random.random() < 0.05:
                is_senescent = True
                senescence_cause = 'stress'

            cell = self.add_cell(position, divisions, is_senescent, senescence_cause, target_area)
            created_cells.append(cell)

        print("Computing initial tessellation...")
        self._update_voronoi_tessellation()

        print("Optimizing initial positions...")
        self.optimize_cell_positions(iterations=2)

        print(f"Grid populated with {len(created_cells)} cells")
        return created_cells

    def _generate_poisson_disk_samples(self, count):
        """Generate positions using Poisson disk sampling."""
        area = self.width * self.height
        min_dist = np.sqrt(area / count) * 0.5

        positions = []
        attempts = 0
        max_attempts = count * 50

        while len(positions) < count and attempts < max_attempts:
            x = np.random.uniform(min_dist, self.width - min_dist)
            y = np.random.uniform(min_dist, self.height - min_dist)
            pos = (x, y)

            valid = True
            for existing_pos in positions:
                dist = np.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                if dist < min_dist:
                    valid = False
                    break

            if valid:
                positions.append(pos)

            attempts += 1

        while len(positions) < count:
            x = np.random.uniform(20, self.width - 20)
            y = np.random.uniform(20, self.height - 20)
            positions.append((x, y))

        return positions

    def add_controlled_variability(self):
        """Add controlled variability to cell orientations and positions."""
        for cell in self.cells.values():
            if cell.territory_pixels:
                max_displacement = 8
                current_pos = self.cell_seeds[cell.cell_id]

                displacement = (
                    np.random.uniform(-max_displacement, max_displacement),
                    np.random.uniform(-max_displacement, max_displacement)
                )

                new_pos = (
                    max(0, min(self.width - 1, current_pos[0] + displacement[0])),
                    max(0, min(self.height - 1, current_pos[1] + displacement[1]))
                )

                if cell.centroid is not None:
                    display_centroid = self._comp_to_display_coords(cell.centroid[0], cell.centroid[1])
                    centroid_dist = np.sqrt((new_pos[0] - display_centroid[0])**2 + (new_pos[1] - display_centroid[1])**2)
                    if centroid_dist < max_displacement * 2:
                        self.cell_seeds[cell.cell_id] = new_pos
                        cell.update_position(new_pos)

    def enable_energy_tracking(self):
        """
        Enable comprehensive energy tracking for your existing Grid.
        Call this method after creating your grid to start tracking energy.
        """
        import numpy as np

        # Initialize energy tracking
        self.energy_tracker = {
            'history': [],
            'detailed_history': [],
            'optimization_iterations': [],
            'weights': self.energy_weights.copy()
        }

        # Store original methods
        self._original_update_biological_adaptation = self.update_biological_adaptation
        self._original_run_adaptive_optimization = self._run_adaptive_optimization

        # Replace with tracking versions
        self.update_biological_adaptation = self._tracked_update_biological_adaptation
        self._run_adaptive_optimization = self._tracked_run_adaptive_optimization

        print("✅ Energy tracking enabled for Grid")

    def get_detailed_energy_breakdown(self):
        """
        Get detailed energy breakdown by component and cell.

        Returns:
            Dictionary with comprehensive energy information
        """
        import numpy as np

        breakdown = {
            'total_energy': 0.0,
            'area_energy': 0.0,
            'aspect_ratio_energy': 0.0,
            'orientation_energy': 0.0,
            'per_cell_energies': [],
            'energy_stats': {}
        }

        if not self.cells:
            return breakdown

        area_energies = []
        ar_energies = []
        orientation_energies = []

        for cell_id, cell in self.cells.items():
            cell_energy = {'cell_id': cell_id, 'is_senescent': cell.is_senescent}

            # Area energy
            if hasattr(cell, 'target_area') and cell.target_area > 0 and cell.actual_area > 0:
                area_ratio = cell.actual_area / cell.target_area
                area_energy = (area_ratio - 1.0) ** 2 * self.energy_weights['area']
                cell_energy['area_energy'] = area_energy
                area_energies.append(area_energy)
            else:
                cell_energy['area_energy'] = 0.0

            # Aspect ratio energy
            if hasattr(cell, 'target_aspect_ratio') and cell.target_aspect_ratio > 0:
                ar_ratio = cell.actual_aspect_ratio / cell.target_aspect_ratio
                ar_energy = (ar_ratio - 1.0) ** 2 * self.energy_weights['aspect_ratio']
                cell_energy['ar_energy'] = ar_energy
                ar_energies.append(ar_energy)
            else:
                cell_energy['ar_energy'] = 0.0

            # Orientation energy
            if hasattr(cell, 'target_orientation'):
                target_align = self.to_alignment_angle(cell.target_orientation)
                actual_align = self.to_alignment_angle(cell.actual_orientation)
                angle_diff = abs(target_align - actual_align)
                orientation_energy = (angle_diff / (np.pi / 2)) ** 2 * self.energy_weights['orientation']
                cell_energy['orientation_energy'] = orientation_energy
                orientation_energies.append(orientation_energy)
            else:
                cell_energy['orientation_energy'] = 0.0

            # Total cell energy
            cell_energy['total_energy'] = (cell_energy['area_energy'] +
                                           cell_energy['ar_energy'] +
                                           cell_energy['orientation_energy'])

            breakdown['per_cell_energies'].append(cell_energy)

        # Aggregate energies
        breakdown['area_energy'] = sum(area_energies)
        breakdown['aspect_ratio_energy'] = sum(ar_energies)
        breakdown['orientation_energy'] = sum(orientation_energies)
        breakdown['total_energy'] = (breakdown['area_energy'] +
                                     breakdown['aspect_ratio_energy'] +
                                     breakdown['orientation_energy'])

        # Statistics
        breakdown['energy_stats'] = {
            'mean_area_energy': np.mean(area_energies) if area_energies else 0,
            'mean_ar_energy': np.mean(ar_energies) if ar_energies else 0,
            'mean_orientation_energy': np.mean(orientation_energies) if orientation_energies else 0,
            'max_area_energy': np.max(area_energies) if area_energies else 0,
            'max_ar_energy': np.max(ar_energies) if ar_energies else 0,
            'max_orientation_energy': np.max(orientation_energies) if orientation_energies else 0,
            'cell_count': len(self.cells)
        }

        return breakdown

    def record_energy_state(self, time_step=None, label=""):
        """
        Record current energy state.

        Parameters:
            time_step: Current time step
            label: Optional label for this energy recording
        """
        if not hasattr(self, 'energy_tracker'):
            return

        breakdown = self.get_detailed_energy_breakdown()
        breakdown['time_step'] = time_step
        breakdown['label'] = label
        breakdown['timestamp'] = len(self.energy_tracker['history'])

        self.energy_tracker['history'].append(breakdown)

    def _tracked_update_biological_adaptation(self):
        """Enhanced biological adaptation with energy tracking."""
        # Record energy before adaptation
        if hasattr(self, 'energy_tracker'):
            self.record_energy_state(label="before_adaptation")

        # Run original adaptation
        result = self._original_update_biological_adaptation()

        # Record energy after adaptation
        if hasattr(self, 'energy_tracker'):
            self.record_energy_state(label="after_adaptation")

        return result

    def _tracked_run_adaptive_optimization(self, intensity, initial_energy):
        """Enhanced adaptive optimization with detailed energy tracking."""
        if hasattr(self, 'energy_tracker'):
            # Record optimization start
            breakdown = self.get_detailed_energy_breakdown()
            breakdown['optimization_intensity'] = intensity
            breakdown['iteration'] = 0
            breakdown['phase'] = 'start'
            self.energy_tracker['optimization_iterations'].append(breakdown)

        # Run original optimization
        result = self._original_run_adaptive_optimization(intensity, initial_energy)

        if hasattr(self, 'energy_tracker'):
            # Record optimization end
            breakdown = self.get_detailed_energy_breakdown()
            breakdown['optimization_intensity'] = intensity
            breakdown['iteration'] = 1
            breakdown['phase'] = 'end'
            self.energy_tracker['optimization_iterations'].append(breakdown)

        return result

    def get_energy_summary(self):
        """
        Get a summary of current energy state.

        Returns:
            Dictionary with energy summary
        """
        if not hasattr(self, 'energy_tracker'):
            return {"error": "Energy tracking not enabled. Call enable_energy_tracking() first."}

        breakdown = self.get_detailed_energy_breakdown()

        summary = {
            'current_total_energy': breakdown['total_energy'],
            'current_area_energy': breakdown['area_energy'],
            'current_ar_energy': breakdown['aspect_ratio_energy'],
            'current_orientation_energy': breakdown['orientation_energy'],
            'energy_per_cell': breakdown['total_energy'] / max(1, len(self.cells)),
            'cell_count': len(self.cells),
            'recordings_count': len(self.energy_tracker['history']) if hasattr(self, 'energy_tracker') else 0,
            'optimization_iterations': len(self.energy_tracker['optimization_iterations']) if hasattr(self,
                                                                                                      'energy_tracker') else 0
        }

        # Add component breakdown percentages
        total = breakdown['total_energy']
        if total > 0:
            summary['component_percentages'] = {
                'area_percent': (breakdown['area_energy'] / total) * 100,
                'ar_percent': (breakdown['aspect_ratio_energy'] / total) * 100,
                'orientation_percent': (breakdown['orientation_energy'] / total) * 100
            }
        else:
            summary['component_percentages'] = {'area_percent': 0, 'ar_percent': 0, 'orientation_percent': 0}

        # Energy trends
        if hasattr(self, 'energy_tracker') and len(self.energy_tracker['history']) > 1:
            recent_energies = [h['total_energy'] for h in self.energy_tracker['history'][-5:]]
            summary['recent_trend'] = 'decreasing' if recent_energies[-1] < recent_energies[0] else 'increasing'
            summary['energy_change'] = recent_energies[-1] - recent_energies[0]

        return summary

    def plot_energy_evolution(self, save_path=None):
        """
        Create a simple energy evolution plot.

        Parameters:
            save_path: Path to save the plot

        Returns:
            matplotlib.Figure
        """
        if not hasattr(self, 'energy_tracker') or not self.energy_tracker['history']:
            print("No energy history available. Enable tracking and run simulation first.")
            return None

        import matplotlib.pyplot as plt

        # Extract data
        history = self.energy_tracker['history']
        timestamps = [h['timestamp'] for h in history]
        total_energies = [h['total_energy'] for h in history]
        area_energies = [h['area_energy'] for h in history]
        ar_energies = [h['aspect_ratio_energy'] for h in history]
        orientation_energies = [h['orientation_energy'] for h in history]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Total energy
        ax1.plot(timestamps, total_energies, 'k-', linewidth=2, label='Total Energy')
        ax1.set_ylabel('Total Energy')
        ax1.set_title('Energy Evolution Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Component energies
        ax2.plot(timestamps, area_energies, 'b-', linewidth=2, label='Area')
        ax2.plot(timestamps, ar_energies, 'r-', linewidth=2, label='Aspect Ratio')
        ax2.plot(timestamps, orientation_energies, 'g-', linewidth=2, label='Orientation')
        ax2.set_xlabel('Recording Number')
        ax2.set_ylabel('Component Energy')
        ax2.set_title('Energy by Component')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Energy evolution plot saved to: {save_path}")

        return fig

    def print_energy_report(self):
        """Print a comprehensive energy report."""
        import numpy as np

        if not hasattr(self, 'energy_tracker'):
            print("❌ Energy tracking not enabled. Call grid.enable_energy_tracking() first.")
            return

        summary = self.get_energy_summary()

        print("\n" + "=" * 50)
        print("ENERGY ANALYSIS REPORT")
        print("=" * 50)
        print(f"Total Energy: {summary['current_total_energy']:.4f}")
        print(f"Energy per Cell: {summary['energy_per_cell']:.4f}")
        print(f"Cell Count: {summary['cell_count']}")
        print(f"Recordings: {summary['recordings_count']}")
        print(f"Optimization Iterations: {summary['optimization_iterations']}")

        print(f"\nComponent Breakdown:")
        print(
            f"  Area Energy: {summary['current_area_energy']:.4f} ({summary['component_percentages']['area_percent']:.1f}%)")
        print(
            f"  Aspect Ratio Energy: {summary['current_ar_energy']:.4f} ({summary['component_percentages']['ar_percent']:.1f}%)")
        print(
            f"  Orientation Energy: {summary['current_orientation_energy']:.4f} ({summary['component_percentages']['orientation_percent']:.1f}%)")

        if 'recent_trend' in summary:
            print(f"\nRecent Trend: {summary['recent_trend']} (Δ{summary['energy_change']:+.4f})")

        # Find highest energy cells
        breakdown = self.get_detailed_energy_breakdown()
        if breakdown['per_cell_energies']:
            sorted_cells = sorted(breakdown['per_cell_energies'],
                                  key=lambda x: x['total_energy'], reverse=True)
            print(f"\nTop 3 Highest Energy Cells:")
            for i, cell in enumerate(sorted_cells[:3]):
                print(f"  {i + 1}. Cell {cell['cell_id']}: {cell['total_energy']:.4f} "
                      f"(A:{cell['area_energy']:.3f}, AR:{cell['ar_energy']:.3f}, O:{cell['orientation_energy']:.3f}) "
                      f"{'[SEN]' if cell['is_senescent'] else '[HEALTHY]'}")

        print("=" * 50)
        return

    def update_holes(self, dt):
            """Update hole system for one timestep."""
            if self.holes_enabled and self.hole_manager:
                self.hole_manager.update(dt)

    def get_hole_statistics(self):
            """Get current hole statistics."""
            if self.holes_enabled and self.hole_manager:
                return self.hole_manager.get_hole_statistics()
            return {
                'hole_count': 0,
                'total_hole_area': 0,
                'average_hole_size': 0,
                'hole_area_fraction': 0,
                'holes': []
            }

    def _create_pixel_coordinate_grid_with_holes(self):
        """Create pixel coordinate grid excluding hole areas."""
        # Get all pixel coordinates
        y_coords, x_coords = np.mgrid[0:self.comp_height, 0:self.comp_width]
        all_coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])

        if not self.holes_enabled or not self.hole_manager or not self.hole_manager.holes:
            return all_coords

        # Filter out hole pixels
        valid_coords = []
        for coord in all_coords:
            x, y = coord
            if not self.hole_manager.is_point_in_hole(x, y):
                valid_coords.append(coord)

        return np.array(valid_coords) if valid_coords else all_coords

    def generate_multiple_initial_configurations(self, cell_count, num_configurations=10,
                                                 division_distribution=None, area_distribution=None,
                                                 optimization_iterations=3, verbose=True):
        """
        Generate multiple initial configurations and select the one with lowest energy.

        Parameters:
            cell_count: Number of cells to create
            num_configurations: Number of different initial configurations to try
            division_distribution: Function to generate division counts
            area_distribution: Function to generate target areas
            optimization_iterations: Number of optimization steps for each configuration
            verbose: Whether to print progress information

        Returns:
            Dictionary with best configuration info and all results
        """
        if verbose:
            print(f"🔬 Generating {num_configurations} initial configurations...")

        # Store original state
        original_cells = self.cells.copy()
        original_seeds = self.cell_seeds.copy()
        original_territories = self.territory_map.copy()

        configurations = []

        for config_idx in range(num_configurations):
            if verbose and config_idx % max(1, num_configurations // 5) == 0:
                print(f"   Testing configuration {config_idx + 1}/{num_configurations}...")

            # Clear current state
            self.cells.clear()
            self.cell_seeds.clear()
            self.territory_map.clear()
            self.pixel_ownership.fill(-1)
            self.next_cell_id = 0

            # Generate new configuration
            created_cells = self.populate_grid(
                cell_count,
                division_distribution=division_distribution,
                area_distribution=area_distribution
            )

            # Run optimization to settle the configuration
            for _ in range(optimization_iterations):
                self.update_biological_adaptation()
                self.optimize_cell_positions(iterations=1)

            # Calculate energy
            energy = self.calculate_biological_energy()
            fitness = self.get_biological_fitness()

            # Get detailed statistics
            grid_stats = self.get_grid_statistics()

            # Store configuration data
            config_data = {
                'config_idx': config_idx,
                'energy': energy,
                'fitness': fitness,
                'cell_count': len(self.cells),
                'packing_efficiency': grid_stats.get('packing_efficiency', 0),
                'biological_fitness': grid_stats.get('biological_fitness', 0),
                'global_pressure': grid_stats.get('global_pressure', 1.0),
                # Store cell positions and properties for reconstruction
                'cell_data': {
                    cell_id: {
                        'position': cell.position,
                        'divisions': cell.divisions,
                        'is_senescent': cell.is_senescent,
                        'senescence_cause': cell.senescence_cause,
                        'target_area': cell.target_area,
                        'target_orientation': getattr(cell, 'target_orientation', 0.0),
                        'target_aspect_ratio': getattr(cell, 'target_aspect_ratio', 1.0)
                    }
                    for cell_id, cell in self.cells.items()
                },
                'grid_stats': grid_stats
            }

            configurations.append(config_data)

        # Find best configuration (lowest energy)
        best_config = min(configurations, key=lambda x: x['energy'])
        best_idx = best_config['config_idx']

        if verbose:
            print(f"\n✅ Best configuration found:")
            print(f"   Configuration #{best_idx + 1}")
            print(f"   Energy: {best_config['energy']:.4f}")
            print(f"   Fitness: {best_config['fitness']:.4f}")
            print(f"   Packing efficiency: {best_config['packing_efficiency']:.3f}")

            # Show energy distribution
            energies = [c['energy'] for c in configurations]
            print(f"\n📊 Energy distribution across {num_configurations} configurations:")
            print(f"   Best: {min(energies):.4f}")
            print(f"   Worst: {max(energies):.4f}")
            print(f"   Mean: {np.mean(energies):.4f}")
            print(f"   Std: {np.std(energies):.4f}")
            print(f"   Improvement: {((max(energies) - min(energies)) / max(energies) * 100):.1f}%")

        # Reconstruct best configuration
        self._reconstruct_configuration(best_config['cell_data'])

        return {
            'best_config': best_config,
            'all_configurations': configurations,
            'energy_improvement': (max(c['energy'] for c in configurations) - best_config['energy']),
            'selected_idx': best_idx
        }

    def _reconstruct_configuration(self, cell_data):
        """
        Reconstruct a specific configuration from stored cell data.

        Parameters:
            cell_data: Dictionary with cell information
        """
        # Clear current state
        self.cells.clear()
        self.cell_seeds.clear()
        self.territory_map.clear()
        self.pixel_ownership.fill(-1)
        self.next_cell_id = 0

        # Recreate cells
        for cell_id, data in cell_data.items():
            # Create cell
            cell = self.add_cell(
                position=data['position'],
                divisions=data['divisions'],
                is_senescent=data['is_senescent'],
                senescence_cause=data['senescence_cause'],
                target_area=data['target_area']
            )

            # Set target properties
            cell.target_orientation = data['target_orientation']
            cell.target_aspect_ratio = data['target_aspect_ratio']
            cell.target_area = data['target_area']

        # Update tessellation
        self._update_voronoi_tessellation()

        # Run brief optimization to settle
        self.optimize_cell_positions(iterations=2)

    def save_configuration_analysis(self, configurations_data, filename=None):
        """
        Save detailed analysis of configuration comparison.

        Parameters:
            configurations_data: Result from generate_multiple_initial_configurations
            filename: Optional filename for saving
        """
        import pandas as pd
        import matplotlib.pyplot as plt

        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"configuration_analysis_{timestamp}"

        configurations = configurations_data['all_configurations']

        # Create DataFrame for analysis
        df_data = []
        for config in configurations:
            row = {
                'config_idx': config['config_idx'],
                'energy': config['energy'],
                'fitness': config['fitness'],
                'packing_efficiency': config['packing_efficiency'],
                'biological_fitness': config['biological_fitness'],
                'global_pressure': config['global_pressure'],
                'cell_count': config['cell_count']
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Save CSV
        csv_path = os.path.join(self.config.plot_directory, f"{filename}.csv")
        df.to_csv(csv_path, index=False)

        # Create analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Energy distribution
        axes[0, 0].hist(df['energy'], bins=min(10, len(configurations) // 2), alpha=0.7, color='blue')
        axes[0, 0].axvline(configurations_data['best_config']['energy'], color='red', linestyle='--',
                           label=f"Best: {configurations_data['best_config']['energy']:.4f}")
        axes[0, 0].set_xlabel('Energy')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Energy Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Energy vs Fitness
        axes[0, 1].scatter(df['energy'], df['fitness'], alpha=0.7)
        best_idx = configurations_data['selected_idx']
        axes[0, 1].scatter(df.loc[best_idx, 'energy'], df.loc[best_idx, 'fitness'],
                           color='red', s=100, label='Selected')
        axes[0, 1].set_xlabel('Energy')
        axes[0, 1].set_ylabel('Fitness')
        axes[0, 1].set_title('Energy vs Fitness')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Energy vs Packing Efficiency
        axes[1, 0].scatter(df['energy'], df['packing_efficiency'], alpha=0.7)
        axes[1, 0].scatter(df.loc[best_idx, 'energy'], df.loc[best_idx, 'packing_efficiency'],
                           color='red', s=100, label='Selected')
        axes[1, 0].set_xlabel('Energy')
        axes[1, 0].set_ylabel('Packing Efficiency')
        axes[1, 0].set_title('Energy vs Packing Efficiency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Configuration ranking
        df_sorted = df.sort_values('energy')
        axes[1, 1].plot(range(len(df_sorted)), df_sorted['energy'], 'b-o', alpha=0.7)
        axes[1, 1].set_xlabel('Configuration Rank')
        axes[1, 1].set_ylabel('Energy')
        axes[1, 1].set_title('Configuration Ranking by Energy')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.config.plot_directory, f"{filename}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Configuration analysis saved:")
        print(f"   Data: {csv_path}")
        print(f"   Plot: {plot_path}")

        return csv_path, plot_path