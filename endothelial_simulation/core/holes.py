"""
CREATE A NEW FILE: endothelial_simulation/core/holes.py
====================================================

This file contains the hole management system for the tessellation.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional


class Hole:
    """
    Represents a hole in the tessellation - an area that cells cannot occupy.
    (Same as original Hole class)
    """

    def __init__(self, hole_id: int, center: Tuple[float, float], radius: float):
        self.hole_id = hole_id
        self.center = center
        self.radius = radius
        self.original_radius = radius
        self.age = 0
        self.compression_factor = 1.0

    def get_area(self) -> float:
        """Get the current area of the hole."""
        return np.pi * (self.radius ** 2)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this hole."""
        distance = np.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2)
        return distance <= self.radius

    def update_compression(self, compression_factor: float):
        """Update hole size based on compression factor."""
        self.compression_factor = max(0.1, compression_factor)
        self.radius = self.original_radius * self.compression_factor

    def get_pixels(self, grid_width: int, grid_height: int) -> List[Tuple[int, int]]:
        """Get all pixel coordinates that belong to this hole."""
        pixels = []

        min_x = max(0, int(self.center[0] - self.radius))
        max_x = min(grid_width, int(self.center[0] + self.radius) + 1)
        min_y = max(0, int(self.center[1] - self.radius))
        max_y = min(grid_height, int(self.center[1] + self.radius) + 1)

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if self.contains_point(x, y):
                    pixels.append((x, y))

        return pixels


class HoleManager:
    """
    Biological HoleManager - Complete drop-in replacement for original HoleManager.
    Uses spatial constraints instead of arbitrary thresholds.
    """

    def __init__(self, grid):
        """Initialize hole manager with biological decision logic."""
        self.grid = grid
        self.holes: Dict[int, Hole] = {}
        self.next_hole_id = 0

        # Configuration parameters (same as original)
        self.max_holes = getattr(grid.config, 'max_holes', 5)
        self.hole_creation_probability_base = getattr(grid.config, 'hole_creation_probability_base', 0.02)
        self.hole_creation_threshold_senescence = getattr(grid.config, 'hole_creation_threshold_senescence', 0.30)
        self.compression_reference_density = getattr(grid.config, 'hole_compression_reference_density', 15)

        # Hole size parameters
        self.min_hole_radius = None
        self.max_hole_radius = None

        # Biological parameters
        self.max_expansion_healthy = 1.2  # Healthy cells: 120% of target area
        self.max_expansion_senescent = 1.0  # Senescent cells: 100% of target area (no growth)
        self.senescence_threshold = 0.30  # 30% senescence threshold

        print(f"🧬 Biological HoleManager initialized (max {self.max_holes} holes)")

    # =============================================================================
    # BIOLOGICAL DECISION LOGIC (REPLACES ARBITRARY THRESHOLDS)
    # =============================================================================

    def should_create_hole(self) -> bool:
        """
        Biological decision for hole creation based on spatial constraints.

        Returns:
            True if hole should be created, False otherwise
        """
        # Check hole limit
        if len(self.holes) >= self.max_holes:
            return False

        # Calculate spatial constraints
        unfillable_area = self._calculate_unfillable_area()
        senescence_fraction = self._get_senescence_fraction()

        # DETERMINISTIC: Space cannot be filled
        if unfillable_area > 0:
            return True

        # PROBABILISTIC: High senescence but space can be filled
        if senescence_fraction >= self.senescence_threshold:
            probability = self.hole_creation_probability_base * (1.0 + 2.0 * senescence_fraction)
            return random.random() < min(0.8, probability)

        # No hole formation
        return False

    def should_fill_holes(self) -> bool:
        """
        Biological decision for hole filling based on spatial capacity.

        Returns:
            True if holes should be filled, False otherwise
        """
        if not self.holes:
            return False

        unfillable_area = self._calculate_unfillable_area()
        senescence_fraction = self._get_senescence_fraction()

        # Fill when space can be filled AND senescence is low
        return unfillable_area <= 0 and senescence_fraction < self.senescence_threshold

    def _calculate_unfillable_area(self) -> float:
        """Calculate area that cells cannot fill."""
        # Total available area
        total_area = self.grid.comp_width * self.grid.comp_height

        # Subtract existing holes
        hole_area = sum(hole.get_area() for hole in self.holes.values())
        available_area = total_area - hole_area

        # Calculate total expansion capacity
        expansion_capacity = 0
        for cell in self.grid.cells.values():
            base_area = getattr(cell, 'target_area', 30)
            if cell.is_senescent:
                expansion_capacity += base_area * self.max_expansion_senescent
            else:
                expansion_capacity += base_area * self.max_expansion_healthy

        return max(0, available_area - expansion_capacity)

    def _get_senescence_fraction(self) -> float:
        """Calculate fraction of senescent cells."""
        cell_counts = self.grid.count_cells_by_type()
        total_cells = cell_counts['total']

        if total_cells == 0:
            return 0.0

        senescent_cells = cell_counts['telomere_senescent'] + cell_counts['stress_senescent']
        return senescent_cells / total_cells

    # =============================================================================
    # ALL ORIGINAL HOLEMANAGER METHODS (PRESERVED)
    # =============================================================================

    def calculate_hole_size_range(self):
        """Calculate hole size range based on current cell sizes."""
        if not self.grid.cells:
            self.min_hole_radius = 20 / self.grid.computation_scale
            self.max_hole_radius = 100 / self.grid.computation_scale
            return

        cell_areas = [cell.actual_area for cell in self.grid.cells.values() if cell.actual_area > 0]
        if cell_areas:
            avg_cell_area = np.mean(cell_areas)
            avg_cell_radius = np.sqrt(avg_cell_area / np.pi)
            self.min_hole_radius = avg_cell_radius * 0.2
            self.max_hole_radius = avg_cell_radius * 1.0
        else:
            self.min_hole_radius = 10 / self.grid.computation_scale
            self.max_hole_radius = 50 / self.grid.computation_scale

    def create_hole(self) -> Optional[Hole]:
        """Create a new hole at a random location."""
        if len(self.holes) >= self.max_holes:
            return None

        self.calculate_hole_size_range()

        # Generate random position (avoid edges)
        margin = self.max_hole_radius * 2
        center_x = random.uniform(margin, self.grid.comp_width - margin)
        center_y = random.uniform(margin, self.grid.comp_height - margin)
        center = (center_x, center_y)

        # Generate random size, influenced by senescence level
        senescence_fraction = self._get_senescence_fraction()
        size_bias = 1.0 + senescence_fraction * 0.5  # Up to 50% larger with high senescence
        base_radius = random.uniform(self.min_hole_radius, self.max_hole_radius)
        radius = min(self.max_hole_radius, base_radius * size_bias)

        # Create hole
        hole_id = self.next_hole_id
        self.next_hole_id += 1

        hole = Hole(hole_id, center, radius)
        self.holes[hole_id] = hole

        print(
            f"🕳️  Created biological hole {hole_id} at {center} with radius {radius:.1f} (senescence: {senescence_fraction:.1%})")

        return hole

    def fill_hole(self, hole_id: int) -> bool:
        """Fill (remove) a specific hole."""
        if hole_id in self.holes:
            hole = self.holes[hole_id]
            print(f"🔌 Filled hole {hole_id} at {hole.center} (age: {hole.age:.1f})")
            del self.holes[hole_id]
            return True
        return False

    def fill_random_hole(self) -> bool:
        """Fill a randomly selected hole."""
        if not self.holes:
            return False

        # Prefer to fill older holes
        hole_ids = list(self.holes.keys())
        hole_ages = [self.holes[hid].age for hid in hole_ids]

        if hole_ages:
            weights = np.array(hole_ages) + 1
            weights = weights / np.sum(weights)
            selected_id = np.random.choice(hole_ids, p=weights)
        else:
            selected_id = random.choice(hole_ids)

        return self.fill_hole(selected_id)

    def update_hole_compression(self):
        """Update hole compression based on current cell density."""
        if not self.holes:
            return

        total_cells = len(self.grid.cells)

        # Calculate compression factor
        if total_cells <= 5:
            compression_factor = 1.0
        elif total_cells <= self.compression_reference_density:
            compression_factor = 1.0 - 0.5 * (total_cells - 5) / (self.compression_reference_density - 5)
        else:
            excess_cells = total_cells - self.compression_reference_density
            additional_compression = min(0.4, excess_cells * 0.02)
            compression_factor = 0.5 - additional_compression

        compression_factor = max(0.1, compression_factor)

        # Apply compression to all holes
        for hole in self.holes.values():
            hole.update_compression(compression_factor)

    def update(self, dt: float):
        """Update hole system for one timestep."""
        # Age existing holes
        for hole in self.holes.values():
            hole.age += dt

        # Update hole compression
        self.update_hole_compression()

        # Check if we should fill holes
        if self.should_fill_holes():
            if random.random() < 0.2:  # 20% probability per timestep
                self.fill_random_hole()

        # Check if we should create new holes
        elif self.should_create_hole():
            self.create_hole()

    def get_hole_pixels(self) -> List[Tuple[int, int]]:
        """Get all pixels occupied by holes."""
        all_hole_pixels = []
        for hole in self.holes.values():
            hole_pixels = hole.get_pixels(self.grid.comp_width, self.grid.comp_height)
            all_hole_pixels.extend(hole_pixels)
        return all_hole_pixels

    def is_point_in_hole(self, x: float, y: float) -> bool:
        """Check if a point is inside any hole."""
        for hole in self.holes.values():
            if hole.contains_point(x, y):
                return True
        return False

    def get_hole_statistics(self) -> Dict:
        """Get statistics about current holes."""
        if not self.holes:
            return {
                'hole_count': 0,
                'total_hole_area': 0,
                'average_hole_size': 0,
                'hole_area_fraction': 0,
                'holes': []
            }

        hole_areas = [hole.get_area() for hole in self.holes.values()]
        total_area = self.grid.comp_width * self.grid.comp_height
        total_hole_area = sum(hole_areas)

        return {
            'hole_count': len(self.holes),
            'total_hole_area': total_hole_area,
            'average_hole_size': np.mean(hole_areas),
            'hole_area_fraction': total_hole_area / total_area,
            'holes': [
                {
                    'id': hole.hole_id,
                    'center': hole.center,
                    'radius': hole.radius,
                    'area': hole.get_area(),
                    'age': hole.age,
                    'compression': hole.compression_factor
                }
                for hole in self.holes.values()
            ]
        }

    def print_status(self):
        """Print current hole system status for debugging."""
        cell_counts = self.grid.count_cells_by_type()
        total_cells = cell_counts['total']
        senescent_cells = cell_counts['telomere_senescent'] + cell_counts['stress_senescent']
        senescence_fraction = senescent_cells / total_cells if total_cells > 0 else 0

        # Biological status
        unfillable_area = self._calculate_unfillable_area()

        print(f"🧬 Biological Hole System Status:")
        print(f"   Cells: {total_cells}")
        print(f"   Senescence: {senescence_fraction:.1%}")
        print(f"   Unfillable area: {unfillable_area:.0f} pixels")
        print(f"   Can fill space: {unfillable_area <= 0}")
        print(f"   Holes: {len(self.holes)}/{self.max_holes}")

        if self.holes:
            for hole in self.holes.values():
                print(f"   Hole {hole.hole_id}: radius={hole.radius:.1f}, "
                      f"compression={hole.compression_factor:.2f}, age={hole.age:.1f}")

    # =============================================================================
    # ADDITIONAL BIOLOGICAL STATUS METHODS
    # =============================================================================

    def get_biological_status(self) -> Dict:
        """Get detailed biological status for monitoring."""
        unfillable_area = self._calculate_unfillable_area()
        senescence_fraction = self._get_senescence_fraction()

        return {
            'unfillable_area': unfillable_area,
            'senescence_fraction': senescence_fraction,
            'can_fill_space': unfillable_area <= 0,
            'high_senescence': senescence_fraction >= self.senescence_threshold,
            'deterministic_hole_formation': unfillable_area > 0,
            'probabilistic_region': (unfillable_area <= 0 and
                                     senescence_fraction >= self.senescence_threshold),
            'decision_mode': self._get_decision_mode()
        }

    def _get_decision_mode(self) -> str:
        """Get current decision mode for debugging."""
        unfillable_area = self._calculate_unfillable_area()
        senescence_fraction = self._get_senescence_fraction()

        if unfillable_area > 0:
            return "DETERMINISTIC_CREATE"
        elif senescence_fraction >= self.senescence_threshold:
            return "PROBABILISTIC_CREATE"
        elif unfillable_area <= 0 and senescence_fraction < self.senescence_threshold:
            return "FILL_HOLES"
        else:
            return "NO_ACTION"
