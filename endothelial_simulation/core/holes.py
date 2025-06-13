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
    """
    
    def __init__(self, hole_id: int, center: Tuple[float, float], radius: float):
        """
        Initialize a hole.
        
        Parameters:
            hole_id: Unique identifier for the hole
            center: (x, y) center coordinates of the hole
            radius: Radius of the hole in pixels
        """
        self.hole_id = hole_id
        self.center = center
        self.radius = radius
        self.original_radius = radius  # Store original size for compression calculations
        self.age = 0  # How long the hole has existed
        self.compression_factor = 1.0  # Current compression (1.0 = no compression)
        
    def get_area(self) -> float:
        """Get the current area of the hole."""
        return np.pi * (self.radius ** 2)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this hole."""
        distance = np.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2)
        return distance <= self.radius
    
    def update_compression(self, compression_factor: float):
        """
        Update hole size based on compression factor.
        
        Parameters:
            compression_factor: Factor to compress the hole (0-1, where 1 = no compression)
        """
        self.compression_factor = max(0.1, compression_factor)  # Minimum 10% of original size
        self.radius = self.original_radius * self.compression_factor
    
    def get_pixels(self, grid_width: int, grid_height: int) -> List[Tuple[int, int]]:
        """
        Get all pixel coordinates that belong to this hole.
        
        Parameters:
            grid_width: Width of the grid
            grid_height: Height of the grid
            
        Returns:
            List of (x, y) pixel coordinates inside the hole
        """
        pixels = []
        
        # Bounding box around the hole
        min_x = max(0, int(self.center[0] - self.radius))
        max_x = min(grid_width, int(self.center[0] + self.radius) + 1)
        min_y = max(0, int(self.center[1] - self.radius))
        max_y = min(grid_height, int(self.center[1] + self.radius) + 1)
        
        # Check each pixel in the bounding box
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if self.contains_point(x, y):
                    pixels.append((x, y))
        
        return pixels


class HoleManager:
    """
    Manages holes in the tessellation system.
    """
    
    def __init__(self, grid):
        """
        Initialize hole manager.
        
        Parameters:
            grid: Grid instance to attach hole management to
        """
        self.grid = grid
        self.holes: Dict[int, Hole] = {}
        self.next_hole_id = 0
        
        # Get configuration parameters with defaults
        self.max_holes = getattr(grid.config, 'max_holes', 5)
        self.hole_creation_probability_base = getattr(grid.config, 'hole_creation_probability_base', 0.02)
        self.hole_creation_threshold_cells = getattr(grid.config, 'hole_creation_threshold_cells', 10)
        self.hole_creation_threshold_senescence = getattr(grid.config, 'hole_creation_threshold_senescence', 0.30)
        self.compression_reference_density = getattr(grid.config, 'hole_compression_reference_density', 15)
        
        # Hole size parameters (in computational grid units)
        self.min_hole_radius = None  # Will be calculated based on cell size
        self.max_hole_radius = None  # Will be calculated based on cell size
        
    def calculate_hole_size_range(self):
        """Calculate hole size range based on current cell sizes."""
        if not self.grid.cells:
            # Default values if no cells exist
            self.min_hole_radius = 20 / self.grid.computation_scale
            self.max_hole_radius = 100 / self.grid.computation_scale
            return
        
        # Calculate average cell area
        cell_areas = [cell.actual_area for cell in self.grid.cells.values() if cell.actual_area > 0]
        if cell_areas:
            avg_cell_area = np.mean(cell_areas)
            # Convert area to approximate radius
            avg_cell_radius = np.sqrt(avg_cell_area / np.pi)
            
            # Hole size range: 1/5 to 1x of average cell radius
            self.min_hole_radius = avg_cell_radius * 0.2  # 1/5 of cell size
            self.max_hole_radius = avg_cell_radius * 1.0   # Same as cell size
        else:
            # Fallback values
            self.min_hole_radius = 10 / self.grid.computation_scale
            self.max_hole_radius = 50 / self.grid.computation_scale
    
    def should_create_hole(self) -> bool:
        """
        Determine if a hole should be created based on current conditions.
        
        Returns:
            True if a hole should be created, False otherwise
        """
        # Check if we've reached the maximum number of holes
        if len(self.holes) >= self.max_holes:
            return False
        
        # Get current cell statistics
        cell_counts = self.grid.count_cells_by_type()
        total_cells = cell_counts['total']
        senescent_cells = cell_counts['telomere_senescent'] + cell_counts['stress_senescent']
        
        # Check conditions for hole creation
        if total_cells >= self.hole_creation_threshold_cells:
            return False
        
        if total_cells == 0:
            senescence_fraction = 0
        else:
            senescence_fraction = senescent_cells / total_cells
        
        if senescence_fraction < self.hole_creation_threshold_senescence:
            return False
        
        # Calculate probability based on severity of conditions
        cell_factor = max(0, (self.hole_creation_threshold_cells - total_cells) / self.hole_creation_threshold_cells)
        senescence_factor = min(1.0, senescence_fraction / self.hole_creation_threshold_senescence)
        
        # Combined probability (higher when conditions are more severe)
        probability = self.hole_creation_probability_base * cell_factor * senescence_factor
        
        # Increase probability if we have very few holes and conditions are met
        if len(self.holes) < 2:
            probability *= 1.5
        
        return random.random() < probability
    
    def should_fill_holes(self) -> bool:
        """
        Determine if holes should be filled based on current conditions.
        
        Returns:
            True if holes should be filled, False otherwise
        """
        if not self.holes:
            return False
        
        # Get current cell statistics
        cell_counts = self.grid.count_cells_by_type()
        total_cells = cell_counts['total']
        senescent_cells = cell_counts['telomere_senescent'] + cell_counts['stress_senescent']
        
        # Fill holes when cell count > 10 AND senescence < 30%
        senescence_fraction = senescent_cells / total_cells if total_cells > 0 else 0
        
        return (total_cells > self.hole_creation_threshold_cells and 
                senescence_fraction < self.hole_creation_threshold_senescence)
    
    def create_hole(self) -> Optional[Hole]:
        """
        Create a new hole at a random location.
        
        Returns:
            Created Hole object or None if creation failed
        """
        if len(self.holes) >= self.max_holes:
            return None
        
        # Update hole size range based on current cells
        self.calculate_hole_size_range()
        
        # Generate random position (avoid edges)
        margin = self.max_hole_radius * 2
        center_x = random.uniform(margin, self.grid.comp_width - margin)
        center_y = random.uniform(margin, self.grid.comp_height - margin)
        center = (center_x, center_y)
        
        # Generate random size, potentially influenced by senescence level
        cell_counts = self.grid.count_cells_by_type()
        total_cells = cell_counts['total']
        senescent_cells = cell_counts['telomere_senescent'] + cell_counts['stress_senescent']
        senescence_fraction = senescent_cells / total_cells if total_cells > 0 else 0
        
        # Larger holes with higher senescence
        size_bias = 1.0 + senescence_fraction * 0.5  # Up to 50% larger with high senescence
        base_radius = random.uniform(self.min_hole_radius, self.max_hole_radius)
        radius = min(self.max_hole_radius, base_radius * size_bias)
        
        # Create hole
        hole_id = self.next_hole_id
        self.next_hole_id += 1
        
        hole = Hole(hole_id, center, radius)
        self.holes[hole_id] = hole
        
        print(f"🕳️  Created hole {hole_id} at {center} with radius {radius:.1f} (senescence: {senescence_fraction:.1%})")
        
        return hole
    
    def fill_hole(self, hole_id: int) -> bool:
        """
        Fill (remove) a specific hole.
        
        Parameters:
            hole_id: ID of the hole to fill
            
        Returns:
            True if hole was filled, False if not found
        """
        if hole_id in self.holes:
            hole = self.holes[hole_id]
            print(f"🔌 Filled hole {hole_id} at {hole.center} (age: {hole.age:.1f})")
            del self.holes[hole_id]
            return True
        return False
    
    def fill_random_hole(self) -> bool:
        """
        Fill a randomly selected hole.
        
        Returns:
            True if a hole was filled, False if no holes exist
        """
        if not self.holes:
            return False
        
        # Prefer to fill older holes
        hole_ids = list(self.holes.keys())
        hole_ages = [self.holes[hid].age for hid in hole_ids]
        
        # Weight selection by age (older holes more likely to be filled)
        if hole_ages:
            weights = np.array(hole_ages) + 1  # Add 1 to avoid zero weights
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
        
        # Calculate compression factor based on cell density
        if total_cells <= 5:
            compression_factor = 1.0  # No compression for very low density
        elif total_cells <= self.compression_reference_density:
            # Linear compression from 1.0 to 0.5
            compression_factor = 1.0 - 0.5 * (total_cells - 5) / (self.compression_reference_density - 5)
        else:
            # Further compression for high density
            excess_cells = total_cells - self.compression_reference_density
            additional_compression = min(0.4, excess_cells * 0.02)  # Up to 40% additional compression
            compression_factor = 0.5 - additional_compression
        
        compression_factor = max(0.1, compression_factor)  # Minimum 10% of original size
        
        # Apply compression to all holes
        for hole in self.holes.values():
            hole.update_compression(compression_factor)
    
    def update(self, dt: float):
        """
        Update hole system for one timestep.
        
        Parameters:
            dt: Time step in minutes
        """
        # Age existing holes
        for hole in self.holes.values():
            hole.age += dt
        
        # Update hole compression based on current density
        self.update_hole_compression()
        
        # Check if we should fill holes
        if self.should_fill_holes():
            # Fill one hole with 20% probability per timestep when conditions are met
            if random.random() < 0.2:
                self.fill_random_hole()
        
        # Check if we should create new holes
        elif self.should_create_hole():
            self.create_hole()
    
    def get_hole_pixels(self) -> List[Tuple[int, int]]:
        """
        Get all pixels occupied by holes.
        
        Returns:
            List of (x, y) pixel coordinates that are holes
        """
        all_hole_pixels = []
        for hole in self.holes.values():
            hole_pixels = hole.get_pixels(self.grid.comp_width, self.grid.comp_height)
            all_hole_pixels.extend(hole_pixels)
        return all_hole_pixels
    
    def is_point_in_hole(self, x: float, y: float) -> bool:
        """
        Check if a point is inside any hole.
        
        Parameters:
            x, y: Coordinates to check
            
        Returns:
            True if point is in a hole, False otherwise
        """
        for hole in self.holes.values():
            if hole.contains_point(x, y):
                return True
        return False
    
    def get_hole_statistics(self) -> Dict:
        """
        Get statistics about current holes.
        
        Returns:
            Dictionary with hole statistics
        """
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
        
        print(f"🕳️  Hole System Status:")
        print(f"   Cells: {total_cells} (threshold: {self.hole_creation_threshold_cells})")
        print(f"   Senescence: {senescence_fraction:.1%} (threshold: {self.hole_creation_threshold_senescence:.1%})")
        print(f"   Holes: {len(self.holes)}/{self.max_holes}")
        
        if self.holes:
            for hole in self.holes.values():
                print(f"   Hole {hole.hole_id}: radius={hole.radius:.1f}, "
                      f"compression={hole.compression_factor:.2f}, age={hole.age:.1f}")
