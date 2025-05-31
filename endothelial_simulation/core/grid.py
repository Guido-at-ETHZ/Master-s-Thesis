"""
Enhanced Grid module - DROP-IN REPLACEMENT for the original grid.py
Simply replace your existing grid.py with this file.
"""
import numpy as np
from scipy.spatial.distance import cdist
from .cell import Cell
import random


class Grid:
    """
    Enhanced Grid class that automatically links spatial properties to visual representation.
    This is a drop-in replacement for the original Grid class.
    """

    def __init__(self, width, height, config):
        """Initialize the enhanced grid (same interface as original)."""
        self.width = width
        self.height = height
        self.config = config

        # Use reduced resolution for computation, scale up for visualization
        self.computation_scale = 4
        self.comp_width = width // self.computation_scale
        self.comp_height = height // self.computation_scale

        print(f"Grid initialized: Display {width}x{height}, Computation {self.comp_width}x{self.comp_height}")

        # Core data structures (same as original)
        self.cells = {}
        self.pixel_ownership = np.full((self.comp_height, self.comp_width), -1, dtype=int)
        self.pixel_coords = self._create_pixel_coordinate_grid()
        self.next_cell_id = 0
        self.cell_seeds = {}
        self.territory_map = {}

        # Original Grid parameters
        self.global_pressure = 1.0
        self.adaptation_rate = 0.1

        # NEW: Biological constraint parameters (only added, nothing removed)
        self.enable_biological_adaptation = True  # Can be turned off
        self.biological_adaptation_strength = 0.15  # How much biological targets influence positioning
        self.adaptation_forces = {}  # Maps cell_id to force vectors
        self.biological_fitness_history = []

    def add_cell(self, position=None, divisions=0, is_senescent=False, senescence_cause=None, target_area=100.0):
        """Add a new cell (same interface as original)."""
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
        self.adaptation_forces[cell_id] = np.array([0.0, 0.0])

        return cell

    def remove_cell(self, cell_id):
        """Remove a cell (same interface as original)."""
        if cell_id not in self.cells:
            return False

        del self.cells[cell_id]
        if cell_id in self.cell_seeds:
            del self.cell_seeds[cell_id]
        if cell_id in self.territory_map:
            del self.territory_map[cell_id]
        if cell_id in self.adaptation_forces:
            del self.adaptation_forces[cell_id]

        return True

    def optimize_cell_positions(self, iterations=3):
        """Enhanced version of original method that includes biological constraints."""
        for iteration in range(iterations):
            position_updates = {}

            for cell_id, cell in self.cells.items():
                if cell.territory_pixels and cell.centroid is not None:
                    # Original Lloyd algorithm: move toward centroid
                    display_centroid = self._comp_to_display_coords(cell.centroid[0], cell.centroid[1])
                    current_pos = self.cell_seeds[cell_id]

                    # Calculate movement toward centroid
                    centroid_movement = np.array(display_centroid) - np.array(current_pos)

                    # NEW: Add biological adaptation force
                    biological_force = np.array([0.0, 0.0])
                    if self.enable_biological_adaptation and cell_id in self.adaptation_forces:
                        biological_force = self.adaptation_forces[cell_id]

                    # Combine movements (70% geometric, 30% biological)
                    total_movement = centroid_movement * 0.7 + biological_force * 0.3

                    # Apply smoothing (same as original)
                    smoothing_factor = 0.3
                    new_pos = (
                        current_pos[0] * (1 - smoothing_factor) + (current_pos[0] + total_movement[0]) * smoothing_factor,
                        current_pos[1] * (1 - smoothing_factor) + (current_pos[1] + total_movement[1]) * smoothing_factor
                    )

                    # Constrain to grid bounds (same as original)
                    new_pos = (
                        max(0, min(self.width - 1, new_pos[0])),
                        max(0, min(self.height - 1, new_pos[1]))
                    )

                    position_updates[cell_id] = new_pos

            # Apply position updates (same as original)
            for cell_id, new_pos in position_updates.items():
                self.cell_seeds[cell_id] = new_pos
                self.cells[cell_id].update_position(new_pos)

            # Update tessellation (same as original)
            self._update_voronoi_tessellation()

    def update_biological_adaptation(self):
        """
        Use EXACT aspect ratio targets from spatial model (no capping).
        """
        if not self.enable_biological_adaptation or not self.cells:
            return

        print(f"Updating biological adaptation with {len(self.cells)} cells...")

        fitness_before = self.get_biological_fitness()

        # Parameters - adjust aspect ratio strength for extreme values
        orientation_strength = 0.6
        aspect_ratio_strength = 0.05  # MUCH lower for extreme values like 200.3
        max_displacement = 20.0

        total_movements = 0
        for cell_id, cell in self.cells.items():

            if cell.is_senescent:
                self.adaptation_forces[cell_id] = np.array([0.0, 0.0])
                continue

            if not hasattr(cell, 'target_orientation') or cell.target_orientation is None:
                self.adaptation_forces[cell_id] = np.array([0.0, 0.0])
                continue

            total_force = np.array([0.0, 0.0])

            # Orientation force (same as before)
            if hasattr(cell, 'actual_orientation') and cell.territory_pixels:
                orientation_error = self._angle_difference(cell.target_orientation, cell.actual_orientation)

                target_deg = np.degrees(cell.target_orientation)
                actual_deg = np.degrees(cell.actual_orientation)
                error_deg = np.degrees(orientation_error)

                print(f"  Cell {cell_id}: target={target_deg:.1f}°, actual={actual_deg:.1f}°, error={error_deg:.1f}°")

                if abs(orientation_error) > 0.2:
                    force_magnitude = abs(orientation_error) * orientation_strength * 20

                    if orientation_error > 0:
                        force_angle = cell.actual_orientation + np.pi / 2
                    else:
                        force_angle = cell.actual_orientation - np.pi / 2

                    orientation_force = force_magnitude * np.array([np.cos(force_angle), np.sin(force_angle)])
                    total_force += orientation_force

                    print(f"    Applied orientation force: magnitude={force_magnitude:.1f}")

            # Aspect ratio force - USE EXACT TARGETS (including 200.3!)
            if (hasattr(cell, 'target_aspect_ratio') and hasattr(cell, 'actual_aspect_ratio') and
                    hasattr(cell, 'target_orientation')):

                ar_error = cell.target_aspect_ratio - cell.actual_aspect_ratio

                # Print the actual values to verify
                print(
                    f"    AR: target={cell.target_aspect_ratio:.1f}, actual={cell.actual_aspect_ratio:.1f}, error={ar_error:.1f}")

                if abs(ar_error) > 0.5:  # Lower threshold for extreme values
                    # Scale force based on error magnitude (for extreme values like 200.3)
                    force_magnitude = np.tanh(ar_error * aspect_ratio_strength) * 15  # Use tanh to prevent explosion
                    stretch_direction = np.array([np.cos(cell.target_orientation), np.sin(cell.target_orientation)])
                    ar_force = force_magnitude * stretch_direction
                    total_force += ar_force

                    print(f"    Applied aspect ratio force: magnitude={force_magnitude:.1f}")

            # Limit total force
            force_magnitude = np.linalg.norm(total_force)
            if force_magnitude > max_displacement:
                total_force = total_force / force_magnitude * max_displacement

            self.adaptation_forces[cell_id] = total_force

            if force_magnitude > 1.0:
                total_movements += 1

        print(f"  Cells with significant forces: {total_movements}")

        self._apply_strong_adaptation_forces()

        fitness_after = self.get_biological_fitness()
        improvement = fitness_after - fitness_before

        print(f"  Biological fitness: {fitness_before:.3f} -> {fitness_after:.3f} (Δ={improvement:+.3f})")

    def _apply_strong_adaptation_forces(self):
        """Apply adaptation forces with stronger movement."""
        movements_applied = 0

        for cell_id, force in self.adaptation_forces.items():
            force_magnitude = np.linalg.norm(force)

            if force_magnitude > 0.5:  # Lower threshold for movement
                current_pos = np.array(self.cell_seeds[cell_id])

                # Apply force with STRONGER adaptation rate
                displacement = force * 0.2  # Increased from 0.05
                new_pos = current_pos + displacement

                # Constrain to grid bounds
                new_pos[0] = max(20, min(self.width - 20, new_pos[0]))
                new_pos[1] = max(20, min(self.height - 20, new_pos[1]))

                # Update position
                self.cell_seeds[cell_id] = tuple(new_pos)
                self.cells[cell_id].update_position(tuple(new_pos))
                movements_applied += 1

        if movements_applied > 0:
            print(f"    Moved {movements_applied} cells")
            # Update tessellation after movements
            self._update_voronoi_tessellation()

    def get_biological_fitness(self):
        """NEW METHOD: Get current biological fitness (0-1, where 1 is perfect)."""
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

            if component_count > 0:
                total_fitness += cell_fitness / component_count
                cell_count += 1

        return total_fitness / cell_count if cell_count > 0 else 1.0

    def _angle_difference(self, target, actual):
        """
        Calculate angle difference considering 0°/180° equivalence for flow direction.
        """
        # Normalize angles to [0, 2π]
        target_norm = target % (2 * np.pi)
        actual_norm = actual % (2 * np.pi)

        # For flow alignment, 0° and 180° (π) are equivalent
        diff1 = target_norm - actual_norm
        diff2 = (target_norm + np.pi) % (2 * np.pi) - actual_norm

        # Normalize differences to [-π, π]
        diff1 = ((diff1 + np.pi) % (2 * np.pi)) - np.pi
        diff2 = ((diff2 + np.pi) % (2 * np.pi)) - np.pi

        # Choose the smaller rotation
        if abs(diff1) <= abs(diff2):
            return diff1
        else:
            return diff2

    def _create_pixel_coordinate_grid(self):
        """Create a grid of pixel coordinates for computation."""
        y_coords, x_coords = np.mgrid[0:self.comp_height, 0:self.comp_width]
        return np.column_stack([x_coords.ravel(), y_coords.ravel()])

    def _display_to_comp_coords(self, x, y):
        """Convert display coordinates to computational coordinates."""
        return (x / self.computation_scale, y / self.computation_scale)

    def _comp_to_display_coords(self, x, y):
        """Convert computational coordinates to display coordinates."""
        return (x * self.computation_scale, y * self.computation_scale)

    def _update_voronoi_tessellation(self):
        """Update the Voronoi tessellation and assign territories to cells."""
        if not self.cells:
            self.pixel_ownership.fill(-1)
            self.territory_map.clear()
            return

        seed_points = []
        cell_ids = []

        for cell_id, display_pos in self.cell_seeds.items():
            comp_pos = self._display_to_comp_coords(display_pos[0], display_pos[1])
            seed_points.append(comp_pos)
            cell_ids.append(cell_id)

        if len(seed_points) < 2:
            if len(seed_points) == 1:
                cell_id = cell_ids[0]
                all_pixels = [(x, y) for x in range(self.comp_width) for y in range(self.comp_height)]
                self._assign_territory_to_cell(cell_id, all_pixels)
            return

        seed_array = np.array(seed_points)
        distances = cdist(self.pixel_coords, seed_array)
        nearest_seed_indices = np.argmin(distances, axis=1)

        self.territory_map.clear()
        self.pixel_ownership.fill(-1)

        for i, cell_id in enumerate(cell_ids):
            pixel_indices = np.where(nearest_seed_indices == i)[0]
            pixels = [(int(self.pixel_coords[idx][0]), int(self.pixel_coords[idx][1]))
                     for idx in pixel_indices]
            self._assign_territory_to_cell(cell_id, pixels)

    def _assign_territory_to_cell(self, cell_id, pixels):
        """Assign a territory to a cell."""
        self.territory_map[cell_id] = pixels

        for x, y in pixels:
            if 0 <= x < self.comp_width and 0 <= y < self.comp_height:
                self.pixel_ownership[y, x] = cell_id

        if cell_id in self.cells:
            self.cells[cell_id].assign_territory(pixels)

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

    def add_controlled_variability(self):
        """Add controlled variability to cell orientations and positions."""
        for cell in self.cells.values():
            if cell.territory_pixels:
                max_displacement = 10
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

        # Add biological fitness (NEW)
        biological_fitness = self.get_biological_fitness()

        target_areas = [cell.target_area * (self.computation_scale ** 2) for cell in self.cells.values()]
        actual_areas = [cell.actual_area * (self.computation_scale ** 2) for cell in self.cells.values()]
        compression_ratios = [cell.compression_ratio for cell in self.cells.values()]

        aspect_ratios = [cell.actual_aspect_ratio for cell in self.cells.values()]
        orientations = [cell.actual_orientation for cell in self.cells.values()]

        return {
            'cell_counts': cell_counts,
            'global_pressure': self.global_pressure,
            'packing_efficiency': self.calculate_packing_efficiency(),
            'biological_fitness': biological_fitness,  # NEW
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