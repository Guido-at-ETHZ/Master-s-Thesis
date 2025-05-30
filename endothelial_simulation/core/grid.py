"""
Grid module for managing the spatial arrangement of cells with Voronoi tessellation.
Optimized version for better performance.
"""
import numpy as np
from scipy.spatial.distance import cdist
from .cell import Cell
import random


class Grid:
    """
    Class representing the spatial grid for cell placement using Voronoi tessellation.
    Each pixel belongs to exactly one cell, creating a mosaic structure.
    Optimized for performance with reduced grid resolution.
    """

    def __init__(self, width, height, config):
        """
        Initialize the grid with dimensions and configuration.

        Parameters:
            width: Width of the grid in pixels
            height: Height of the grid in pixels
            config: SimulationConfig object
        """
        self.width = width
        self.height = height
        self.config = config

        # Use reduced resolution for computation, scale up for visualization
        self.computation_scale = 4  # Use 1/X resolution for computation X = 4
        self.comp_width = width // self.computation_scale
        self.comp_height = height // self.computation_scale

        print(f"Grid initialized: Display {width}x{height}, Computation {self.comp_width}x{self.comp_height}")

        # Dictionary to store cells by ID
        self.cells = {}

        # Pixel ownership grid - use reduced resolution
        self.pixel_ownership = np.full((self.comp_height, self.comp_width), -1, dtype=int)

        # Grid of coordinates for each computational pixel
        self.pixel_coords = self._create_pixel_coordinate_grid()

        # Tracking the next available cell ID
        self.next_cell_id = 0

        # For Voronoi tessellation
        self.cell_seeds = {}  # Maps cell_id to (x, y) seed position (in display coordinates)

        # Territory management
        self.territory_map = {}  # Maps cell_id to list of pixels (in computational coordinates)

        # Adaptation parameters
        self.global_pressure = 1.0  # Overall crowding pressure
        self.adaptation_rate = 0.1  # How fast cells adapt to space constraints

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

    def add_cell(self, position=None, divisions=0, is_senescent=False, senescence_cause=None, target_area=100.0):
        """
        Add a new cell to the grid and perform Voronoi tessellation.

        Parameters:
            position: (x, y) coordinates in display space, random if None
            divisions: Number of divisions the cell has undergone
            is_senescent: Boolean indicating if the cell is senescent
            senescence_cause: 'telomere' or 'stress' indicating the cause of senescence
            target_area: Target area for the cell

        Returns:
            The newly created Cell object
        """
        # Generate a random position if none provided, avoiding edges
        if position is None:
            margin = 20
            position = (
                np.random.uniform(margin, self.width - margin),
                np.random.uniform(margin, self.height - margin)
            )

        # Create a new cell
        cell_id = self.next_cell_id
        self.next_cell_id += 1

        # Scale target area to computational resolution
        comp_target_area = target_area / (self.computation_scale ** 2)

        cell = Cell(cell_id, position, divisions, is_senescent, senescence_cause, comp_target_area)

        # Add cell to the dictionary and store seed position
        self.cells[cell_id] = cell
        self.cell_seeds[cell_id] = position

        # Don't update tessellation immediately - batch it
        return cell

    def remove_cell(self, cell_id):
        """
        Remove a cell from the grid and update tessellation.

        Parameters:
            cell_id: ID of the cell to remove

        Returns:
            Boolean indicating if removal was successful
        """
        if cell_id not in self.cells:
            return False

        # Remove from all data structures
        del self.cells[cell_id]
        if cell_id in self.cell_seeds:
            del self.cell_seeds[cell_id]
        if cell_id in self.territory_map:
            del self.territory_map[cell_id]

        return True

    def _update_voronoi_tessellation(self):
        """Update the Voronoi tessellation and assign territories to cells."""
        if not self.cells:
            self.pixel_ownership.fill(-1)
            self.territory_map.clear()
            return

        # Get seed points in computational coordinates
        seed_points = []
        cell_ids = []

        for cell_id, display_pos in self.cell_seeds.items():
            comp_pos = self._display_to_comp_coords(display_pos[0], display_pos[1])
            seed_points.append(comp_pos)
            cell_ids.append(cell_id)

        if len(seed_points) < 2:
            # Special case: only one cell gets everything
            if len(seed_points) == 1:
                cell_id = cell_ids[0]
                all_pixels = [(x, y) for x in range(self.comp_width) for y in range(self.comp_height)]
                self._assign_territory_to_cell(cell_id, all_pixels)
            return

        # Calculate distances from each computational pixel to each seed
        seed_array = np.array(seed_points)
        distances = cdist(self.pixel_coords, seed_array)

        # Assign each pixel to the nearest seed
        nearest_seed_indices = np.argmin(distances, axis=1)

        # Clear previous territories
        self.territory_map.clear()
        self.pixel_ownership.fill(-1)

        # Build territory map
        for i, cell_id in enumerate(cell_ids):
            pixel_indices = np.where(nearest_seed_indices == i)[0]
            pixels = [(int(self.pixel_coords[idx][0]), int(self.pixel_coords[idx][1]))
                     for idx in pixel_indices]
            self._assign_territory_to_cell(cell_id, pixels)

    def _assign_territory_to_cell(self, cell_id, pixels):
        """
        Assign a territory (list of computational pixels) to a cell.

        Parameters:
            cell_id: ID of the cell
            pixels: List of (x, y) pixel coordinates in computational space
        """
        # Store territory in computational coordinates
        self.territory_map[cell_id] = pixels

        # Update pixel ownership grid
        for x, y in pixels:
            if 0 <= x < self.comp_width and 0 <= y < self.comp_height:
                self.pixel_ownership[y, x] = cell_id

        # Update cell with its territory
        if cell_id in self.cells:
            self.cells[cell_id].assign_territory(pixels)

    def optimize_cell_positions(self, iterations=3):
        """
        Optimize cell positions to reduce shape deviation using Lloyd's algorithm.

        Parameters:
            iterations: Number of optimization iterations
        """
        for _ in range(iterations):
            # Move each cell towards the centroid of its territory
            position_updates = {}

            for cell_id, cell in self.cells.items():
                if cell.territory_pixels and cell.centroid is not None:
                    # Centroid is in computational coordinates, convert to display
                    display_centroid = self._comp_to_display_coords(cell.centroid[0], cell.centroid[1])

                    # Apply some smoothing to avoid oscillations
                    current_pos = self.cell_seeds[cell_id]
                    smoothing_factor = 0.3
                    new_pos = (
                        current_pos[0] * (1 - smoothing_factor) + display_centroid[0] * smoothing_factor,
                        current_pos[1] * (1 - smoothing_factor) + display_centroid[1] * smoothing_factor
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

    def adapt_cell_properties(self):
        """Adapt cell properties based on space constraints and targets."""
        if not self.cells:
            return

        # Calculate global pressure based on total desired vs actual area
        total_target_area = sum(cell.target_area for cell in self.cells.values())
        total_available_area = self.comp_width * self.comp_height
        self.global_pressure = total_target_area / total_available_area if total_available_area > 0 else 1.0

        # Adapt each cell
        for cell in self.cells.values():
            # Calculate local space factor
            local_space_factor = min(1.0, 1.0 / self.global_pressure)

            # Apply constraints
            cell.adapt_to_constraints(local_space_factor)

    def add_controlled_variability(self):
        """Add controlled variability to cell orientations and positions."""
        for cell in self.cells.values():
            # Add small random displacement to seed position (within reason)
            if cell.territory_pixels:
                max_displacement = 10  # pixels in display space
                current_pos = self.cell_seeds[cell.cell_id]

                displacement = (
                    np.random.uniform(-max_displacement, max_displacement),
                    np.random.uniform(-max_displacement, max_displacement)
                )

                new_pos = (
                    max(0, min(self.width - 1, current_pos[0] + displacement[0])),
                    max(0, min(self.height - 1, current_pos[1] + displacement[1]))
                )

                # Only apply if it doesn't move too far from centroid
                if cell.centroid is not None:
                    display_centroid = self._comp_to_display_coords(cell.centroid[0], cell.centroid[1])
                    centroid_dist = np.sqrt((new_pos[0] - display_centroid[0])**2 + (new_pos[1] - display_centroid[1])**2)
                    if centroid_dist < max_displacement * 2:
                        self.cell_seeds[cell.cell_id] = new_pos
                        cell.update_position(new_pos)

    def get_cell(self, cell_id):
        """Get a cell by its ID."""
        return self.cells.get(cell_id)

    def populate_grid(self, count, division_distribution=None, area_distribution=None):
        """
        Populate the grid with initial cells using improved spatial distribution.

        Parameters:
            count: Number of cells to create
            division_distribution: Optional function that returns division counts
            area_distribution: Optional function that returns target areas

        Returns:
            List of created Cell objects
        """
        created_cells = []

        # Default distributions
        if division_distribution is None:
            max_div = self.config.max_divisions
            def division_distribution():
                r = np.random.random()
                return int(max_div * (1 - np.sqrt(r)))

        if area_distribution is None:
            def area_distribution():
                # Vary area around base value (in display coordinates)
                base_area = (self.width * self.height) / count  # Average area per cell
                return np.random.uniform(base_area * 0.7, base_area * 1.3)

        # Use Poisson disk sampling for better initial distribution
        positions = self._generate_poisson_disk_samples(count)

        print(f"Creating {count} cells...")

        # Create cells without updating tessellation each time
        for i, position in enumerate(positions):
            if i % 50 == 0:
                print(f"Created {i}/{count} cells")

            divisions = division_distribution()
            target_area = area_distribution()

            # Determine if cell is senescent
            is_senescent = False
            senescence_cause = None

            if divisions >= self.config.max_divisions:
                is_senescent = True
                senescence_cause = 'telomere'
            elif np.random.random() < 0.05:
                is_senescent = True
                senescence_cause = 'stress'

            # Create the cell
            cell = self.add_cell(position, divisions, is_senescent, senescence_cause, target_area)
            created_cells.append(cell)

        # Now update tessellation once for all cells
        print("Computing initial tessellation...")
        self._update_voronoi_tessellation()

        # Optimize initial positions
        print("Optimizing initial positions...")
        self.optimize_cell_positions(iterations=2)

        print(f"Grid populated with {len(created_cells)} cells")
        return created_cells

    def _generate_poisson_disk_samples(self, count):
        """
        Generate positions using Poisson disk sampling for better spatial distribution.

        Parameters:
            count: Target number of positions

        Returns:
            List of (x, y) positions in display coordinates
        """
        # Estimate minimum distance between points
        area = self.width * self.height
        min_dist = np.sqrt(area / count) * 0.5  # Allow some overlap

        positions = []
        attempts = 0
        max_attempts = count * 50  # Reduced attempts for speed

        while len(positions) < count and attempts < max_attempts:
            # Generate random position
            x = np.random.uniform(min_dist, self.width - min_dist)
            y = np.random.uniform(min_dist, self.height - min_dist)
            pos = (x, y)

            # Check distance to existing positions
            valid = True
            for existing_pos in positions:
                dist = np.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                if dist < min_dist:
                    valid = False
                    break

            if valid:
                positions.append(pos)

            attempts += 1

        # If we couldn't get enough positions with Poisson disk, fill with random
        while len(positions) < count:
            x = np.random.uniform(20, self.width - 20)
            y = np.random.uniform(20, self.height - 20)
            positions.append((x, y))

        return positions

    def apply_shear_stress_field(self, shear_stress_function, duration):
        """Apply a shear stress field to all cells."""
        for cell in self.cells.values():
            # Use display coordinates for shear stress calculation
            if cell.centroid is not None:
                display_centroid = self._comp_to_display_coords(cell.centroid[0], cell.centroid[1])
                x, y = display_centroid
            else:
                x, y = cell.position
            shear_stress = shear_stress_function(x, y)
            cell.apply_shear_stress(shear_stress, duration)

    def calculate_confluency(self):
        """Calculate the confluency (always 1.0 since all pixels are owned)."""
        return 1.0

    def calculate_packing_efficiency(self):
        """
        Calculate how efficiently cells are packed (how close actual areas are to targets).

        Returns:
            Packing efficiency (0-1, where 1 is perfect)
        """
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
        """Count cells by type (normal, telomere-senescent, stress-senescent)."""
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
        """Get comprehensive statistics about the grid state - real parameters only."""
        if not self.cells:
            return {}

        # Basic counts
        cell_counts = self.count_cells_by_type()

        # Area statistics (scale up for display)
        target_areas = [cell.target_area * (self.computation_scale ** 2) for cell in self.cells.values()]
        actual_areas = [cell.actual_area * (self.computation_scale ** 2) for cell in self.cells.values()]
        compression_ratios = [cell.compression_ratio for cell in self.cells.values()]

        # Shape statistics - real parameters only
        aspect_ratios = [cell.actual_aspect_ratio for cell in self.cells.values()]
        orientations = [cell.actual_orientation for cell in self.cells.values()]

        return {
            'cell_counts': cell_counts,
            'global_pressure': self.global_pressure,
            'packing_efficiency': self.calculate_packing_efficiency(),
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
                # Removed: mean_compactness (fake parameter)
            }
        }

    def get_display_territories(self):
        """
        Get cell territories scaled up to display resolution for visualization.

        Returns:
            Dictionary mapping cell_id to list of display pixels
        """
        display_territories = {}

        for cell_id, comp_pixels in self.territory_map.items():
            display_pixels = []
            for comp_x, comp_y in comp_pixels:
                # Scale up computational pixel to display pixels
                for dx in range(self.computation_scale):
                    for dy in range(self.computation_scale):
                        display_x = comp_x * self.computation_scale + dx
                        display_y = comp_y * self.computation_scale + dy
                        if 0 <= display_x < self.width and 0 <= display_y < self.height:
                            display_pixels.append((display_x, display_y))
            display_territories[cell_id] = display_pixels

        return display_territories