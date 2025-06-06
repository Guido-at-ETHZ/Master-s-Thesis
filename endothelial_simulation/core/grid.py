"""
Enhanced Grid module with biological energy minimization that integrates with existing temporal dynamics.
"""
import numpy as np
from scipy.spatial.distance import cdist
from .cell import Cell
import random


class Grid:
    """
    Enhanced Grid class that couples tessellation with biological target properties
    while preserving existing temporal dynamics.
    """

    def __init__(self, width, height, config):
        """Initialize the biological grid with energy-based optimization."""
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
            'area': 1.0,        # Weight for area deviation
            'aspect_ratio': 0.5, # Weight for aspect ratio deviation
            'orientation': 2.0,  # Weight for orientation deviation
            'overlap': 2.0,      # Weight for preventing overlap
            'boundary': 0.1      # Weight for staying within bounds
        }

        # Optimization settings
        self.max_displacement_per_step = 12.0  # Maximum cell movement per optimization step
        self.convergence_threshold = 0.001     # Energy convergence threshold
        self.max_optimization_steps = 8       # Maximum optimization iterations per time step

        # Adaptation parameters
        self.adaptation_strength = 0.25        # How strongly cells adapt toward targets
        self.global_adaptation_interval = 3   # Steps between global optimizations

        # Keep original grid parameters for compatibility
        self.global_pressure = 1.0
        self.adaptation_rate = 0.1

        # Initialize step counter
        self._adaptation_step_counter = 0

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
        Update biological adaptation every 2 steps (same as your original).
        This integrates with your existing temporal dynamics.
        """
        self._adaptation_step_counter += 1

        # Perform global optimization periodically
        if self._adaptation_step_counter % self.global_adaptation_interval == 0:
            self.optimize_biological_tessellation()
        else:
            # Perform lighter local optimization more frequently
            self._light_local_optimization()

    def optimize_biological_tessellation(self):
        """
        Optimize cell positions and tessellation to minimize biological energy.
        This works with targets that are evolving via your temporal dynamics.
        """
        if len(self.cells) < 2:
            return

        initial_energy = self.calculate_biological_energy()

        for step in range(self.max_optimization_steps):
            # Local optimization: small adjustments to each cell
            position_adjustments = self._calculate_local_position_adjustments()

            # Apply adjustments with constraints
            movements_applied = 0
            for cell_id, adjustment in position_adjustments.items():
                if np.linalg.norm(adjustment) > 1.0:  # Only apply significant adjustments
                    current_pos = np.array(self.cell_seeds[cell_id])
                    new_pos = current_pos + adjustment

                    # Constrain to grid bounds
                    new_pos[0] = max(20, min(self.width - 20, new_pos[0]))
                    new_pos[1] = max(20, min(self.height - 20, new_pos[1]))

                    self.cell_seeds[cell_id] = tuple(new_pos)
                    self.cells[cell_id].update_position(tuple(new_pos))
                    movements_applied += 1

            if movements_applied > 0:
                # Update tessellation after movements
                self._update_voronoi_tessellation()

                # Check convergence
                current_energy = self.calculate_biological_energy()
                energy_improvement = initial_energy - current_energy

                if abs(energy_improvement) < self.convergence_threshold:
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
            if force_magnitude > self.max_displacement_per_step:
                total_force = total_force / force_magnitude * self.max_displacement_per_step

            adjustments[cell_id] = total_force

        return adjustments

    def _light_local_optimization(self):
        """Perform lighter local optimization between global optimization steps."""
        if len(self.cells) < 2:
            return

        # Small position adjustments only
        position_adjustments = self._calculate_local_position_adjustments()

        movements_applied = 0
        for cell_id, adjustment in position_adjustments.items():
            # Apply smaller adjustments for frequent updates
            adjustment = adjustment * 0.4

            if np.linalg.norm(adjustment) > 0.5:
                current_pos = np.array(self.cell_seeds[cell_id])
                new_pos = current_pos + adjustment

                # Constrain to grid bounds
                new_pos[0] = max(20, min(self.width - 20, new_pos[0]))
                new_pos[1] = max(20, min(self.height - 20, new_pos[1]))

                self.cell_seeds[cell_id] = tuple(new_pos)
                self.cells[cell_id].update_position(tuple(new_pos))
                movements_applied += 1

        if movements_applied > 0:
            self._update_voronoi_tessellation()

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

        return {
            'cell_counts': cell_counts,
            'global_pressure': self.global_pressure,
            'packing_efficiency': self.calculate_packing_efficiency(),
            'biological_fitness': biological_fitness,
            'biological_energy': biological_energy,
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