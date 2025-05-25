"""
Core module for the Cell class that represents a single endothelial cell with territory management.
Optimized version for better performance.
"""
import numpy as np


class Cell:
    """
    Class representing a single endothelial cell in the simulation with territory-based properties.
    Optimized for performance.
    """

    def __init__(self, cell_id, position=(0, 0), divisions=0, is_senescent=False, senescence_cause=None, target_area=100.0):
        """
        Initialize a cell with its properties.

        Parameters:
            cell_id: Unique identifier for the cell
            position: (x, y) coordinates of the cell center (seed point)
            divisions: Number of divisions the cell has undergone
            is_senescent: Boolean indicating if the cell is senescent
            senescence_cause: 'telomere' or 'stress' indicating the cause of senescence
            target_area: Target area the cell wants to achieve
        """
        # Basic cell properties
        self.cell_id = cell_id
        self.position = position
        self.divisions = divisions
        self.is_senescent = is_senescent
        self.senescence_cause = senescence_cause

        # Territory and morphology properties
        self.target_area = target_area  # Desired area from biological parameters
        self.actual_area = 0.0  # Actual area assigned in the mosaic
        self.territory_pixels = []  # List of (x, y) pixel coordinates owned by this cell
        self.boundary_points = []  # Boundary of the cell territory
        self.centroid = position  # Actual centroid of the territory (may differ from seed)

        # Orientation properties
        self.target_orientation = 0.0  # Target orientation from flow/senescence
        self.actual_orientation = 0.0  # Actual orientation of the cell territory
        self.orientation_variability = 0.1  # How much orientation can vary (in radians)

        # Shape adaptation properties
        self.target_aspect_ratio = 1.0  # Target aspect ratio from biological parameters
        self.actual_aspect_ratio = 1.0  # Actual aspect ratio from territory shape
        self.shape_flexibility = 0.3  # How much the cell can deviate from target shape (0-1)

        # Computed properties from territory
        self.perimeter = 0.0
        self.eccentricity = 0.8
        self.circularity = 0.5
        self.compactness = 1.0  # Area/(perimeter^2), measure of how circular the territory is

        # Cell state properties
        self.age = 0.0
        self.adhesion_strength = 1.0
        self.response = 1.0

        # Mechanical properties
        self.local_shear_stress = 0.0
        self.stress_exposure_time = 0.0

        # Growth and adaptation properties
        self.growth_pressure = 0.0  # Pressure to expand beyond current territory
        self.compression_ratio = 1.0  # How compressed the cell is compared to target size

        # Senescent growth
        # Probabilistic senescent growth properties
        self.senescent_growth_factor = 1.0  # Current size multiplier (starts at 1.0)
        self.max_senescent_growth = 3.0  # Maximum size (3x normal)
        self.growth_probability_base = 0.15  # 15% chance per hour to grow
        self.growth_increment = 0.05  # 5% size increase when growth occurs

    def assign_territory(self, pixel_list):
        """
        Assign a list of pixels to this cell's territory.
        Optimized version with sampling for large territories.

        Parameters:
            pixel_list: List of (x, y) tuples representing pixels owned by this cell
        """
        self.territory_pixels = pixel_list
        self.actual_area = len(pixel_list)

        if pixel_list:
            # Calculate centroid efficiently
            pixels_array = np.array(pixel_list)
            self.centroid = np.mean(pixels_array, axis=0)

            # Calculate boundary points (optimized)
            self._calculate_boundary_fast()

            # Calculate geometric properties (optimized)
            self._calculate_geometry_fast()

            # Update compression ratio
            self.compression_ratio = self.actual_area / max(1, self.target_area)

            # Calculate growth pressure (higher when compressed)
            if self.compression_ratio < 1.0:
                self.growth_pressure = (1.0 - self.compression_ratio) * 2.0
            else:
                self.growth_pressure = 0.0

    def update_senescent_growth(self, dt_hours):
        """
        Probabilistic growth for senescent cells.

        Parameters:
            dt_hours: Time step in hours

        Returns:
            Boolean indicating if growth occurred
        """
        if not self.is_senescent:
            return False

        # Can't grow beyond maximum
        if self.senescent_growth_factor >= self.max_senescent_growth:
            return False

        # Calculate growth probability for this time step
        growth_prob = self.growth_probability_base * dt_hours

        # Factors that influence growth probability:
        # 1. Mechanical stress increases growth probability
        stress_factor = 1.0 + 0.1 * self.local_shear_stress

        # 2. Compression increases growth probability (crowded cells try to expand)
        compression_factor = max(1.0, 2.0 - self.compression_ratio)

        # 3. Growth becomes less likely as cell approaches maximum size
        size_factor = (self.max_senescent_growth - self.senescent_growth_factor) / \
                      (self.max_senescent_growth - 1.0)

        # Combined probability
        final_prob = growth_prob * stress_factor * compression_factor * size_factor
        final_prob = min(0.3 * dt_hours, final_prob)  # Cap at 30% per hour

        # Random growth check
        if np.random.random() < final_prob:
            # Grow by the increment
            growth_increase = self.growth_increment * np.random.uniform(0.8, 1.2)  # Add variability
            self.senescent_growth_factor = min(self.max_senescent_growth,
                                               self.senescent_growth_factor + growth_increase)

            # Update target area
            base_area = self.target_area / (self.senescent_growth_factor - growth_increase + 1.0)  # Back-calculate base
            self.target_area = base_area * self.senescent_growth_factor

            return True

        return False

    def _calculate_boundary_fast(self):
        """Calculate the boundary points of the cell territory - optimized version."""
        if not self.territory_pixels:
            self.boundary_points = []
            self.perimeter = 0
            return

        # For large territories, sample boundary points
        if len(self.territory_pixels) > 1000:
            # Use convex hull for large territories
            try:
                from scipy.spatial import ConvexHull
                points = np.array(self.territory_pixels)
                hull = ConvexHull(points)
                self.boundary_points = points[hull.vertices].tolist()
                self.perimeter = len(self.boundary_points)
                return
            except:
                pass

        # For smaller territories, find actual boundary
        pixels_set = set(self.territory_pixels)
        boundary = []

        # Sample pixels for boundary detection if too many
        pixels_to_check = self.territory_pixels
        if len(pixels_to_check) > 500:
            # Randomly sample pixels for boundary detection
            sample_size = min(500, len(pixels_to_check))
            pixels_to_check = np.random.choice(len(pixels_to_check), sample_size, replace=False)
            pixels_to_check = [self.territory_pixels[i] for i in pixels_to_check]

        # Check each pixel for boundary
        for x, y in pixels_to_check:
            # Check 4-connected neighbors (faster than 8-connected)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (x + dx, y + dy)
                if neighbor not in pixels_set:
                    boundary.append((x, y))
                    break

        self.boundary_points = boundary
        self.perimeter = len(boundary)

    def _calculate_geometry_fast(self):
        """Calculate geometric properties from the territory - optimized version."""
        if not self.territory_pixels:
            return

        pixels_array = np.array(self.territory_pixels)

        # For very large territories, sample points for PCA
        if len(pixels_array) > 1000:
            # Sample points for PCA calculation
            sample_size = min(1000, len(pixels_array))
            indices = np.random.choice(len(pixels_array), sample_size, replace=False)
            sample_pixels = pixels_array[indices]
        else:
            sample_pixels = pixels_array

        # Calculate actual orientation using principal component analysis
        if len(sample_pixels) > 1:
            # Center the points
            centered = sample_pixels - self.centroid

            # Calculate covariance matrix efficiently
            if len(centered) > 1:
                try:
                    cov_matrix = np.cov(centered, rowvar=False)

                    # Get eigenvalues and eigenvectors
                    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

                    # Sort by eigenvalue (largest first)
                    idx = np.argsort(eigenvals)[::-1]
                    eigenvals = eigenvals[idx]
                    eigenvecs = eigenvecs[:, idx]

                    # Principal axis gives orientation
                    principal_axis = eigenvecs[:, 0]
                    self.actual_orientation = np.arctan2(principal_axis[1], principal_axis[0])

                    # Aspect ratio from eigenvalues
                    if eigenvals[1] > 0:
                        self.actual_aspect_ratio = np.sqrt(max(eigenvals[0] / eigenvals[1], 1.0))
                    else:
                        self.actual_aspect_ratio = 1.0

                    # Eccentricity
                    if eigenvals[0] > 0:
                        self.eccentricity = np.sqrt(1 - min(eigenvals[1] / eigenvals[0], 1.0))
                    else:
                        self.eccentricity = 0.0
                except:
                    # Fallback to simple calculations
                    self._calculate_geometry_simple()
            else:
                self._calculate_geometry_simple()
        else:
            self._calculate_geometry_simple()

        # Circularity (4π*Area/Perimeter²)
        if self.perimeter > 0:
            self.circularity = 4 * np.pi * self.actual_area / (self.perimeter ** 2)
        else:
            self.circularity = 1.0

        # Compactness
        if self.perimeter > 0:
            self.compactness = self.actual_area / (self.perimeter ** 2)
        else:
            self.compactness = 1.0

    def _calculate_geometry_simple(self):
        """Simple fallback geometry calculation."""
        self.actual_orientation = self.target_orientation
        self.actual_aspect_ratio = max(1.0, self.target_aspect_ratio)
        self.eccentricity = 0.5

    def adapt_to_constraints(self, available_space_factor=1.0):
        """
        Adapt cell properties based on space constraints.

        Parameters:
            available_space_factor: Factor indicating how much space is available (0-1)
        """
        # Adjust target area based on available space
        adjusted_target_area = self.target_area * available_space_factor

        # If compressed, try to maintain shape as much as possible
        if self.actual_area < adjusted_target_area:
            # Cell is compressed - increase growth pressure
            self.growth_pressure = (adjusted_target_area - self.actual_area) / adjusted_target_area
        else:
            # Cell has enough or more space
            self.growth_pressure = 0.0

        # Adapt orientation gradually towards target
        orientation_diff = self.target_orientation - self.actual_orientation

        # Handle angle wrapping
        while orientation_diff > np.pi:
            orientation_diff -= 2 * np.pi
        while orientation_diff < -np.pi:
            orientation_diff += 2 * np.pi

        # Add some variability around target orientation
        variability = np.random.normal(0, self.orientation_variability)

        # Gradually adjust actual orientation
        adaptation_rate = 0.1
        self.actual_orientation += adaptation_rate * orientation_diff

    def update_target_properties(self, target_orientation, target_aspect_ratio, target_area):
        """
        Update target properties based on biological parameters.

        Parameters:
            target_orientation: Target orientation in radians
            target_aspect_ratio: Target aspect ratio
            target_area: Target area in pixels
        """
        self.target_orientation = target_orientation
        self.target_aspect_ratio = target_aspect_ratio
        self.target_area = target_area

    def get_shape_deviation(self):
        """
        Calculate how much the cell deviates from its target shape.

        Returns:
            Dictionary with deviation metrics
        """
        orientation_deviation = abs(self.actual_orientation - self.target_orientation)
        if orientation_deviation > np.pi:
            orientation_deviation = 2 * np.pi - orientation_deviation

        # Protect against division by zero
        if self.target_aspect_ratio > 0:
            aspect_ratio_deviation = abs(self.actual_aspect_ratio - self.target_aspect_ratio) / self.target_aspect_ratio
        else:
            aspect_ratio_deviation = 0

        if self.target_area > 0:
            area_deviation = abs(self.actual_area - self.target_area) / self.target_area
        else:
            area_deviation = 0

        return {
            'orientation_deviation': orientation_deviation,
            'aspect_ratio_deviation': aspect_ratio_deviation,
            'area_deviation': area_deviation,
            'total_deviation': (orientation_deviation + aspect_ratio_deviation + area_deviation) / 3
        }

    def get_territory_info(self):
        """
        Get information about the cell's territory.

        Returns:
            Dictionary with territory information
        """
        return {
            'territory_size': len(self.territory_pixels),
            'actual_area': self.actual_area,
            'target_area': self.target_area,
            'compression_ratio': self.compression_ratio,
            'growth_pressure': self.growth_pressure,
            'centroid': self.centroid,
            'boundary_length': len(self.boundary_points),
            'compactness': self.compactness
        }

    # Keep existing methods for compatibility
    def update_shape(self, orientation, aspect_ratio, area, eccentricity=None, circularity=None):
        """Update target shape properties (for compatibility)."""
        self.target_orientation = orientation
        self.target_aspect_ratio = aspect_ratio
        self.target_area = area
        if eccentricity is not None:
            self.eccentricity = eccentricity
        if circularity is not None:
            self.circularity = circularity

    def update_response(self, new_response):
        """Update the cell's temporal response value."""
        self.response = new_response

    def update_position(self, new_position):
        """Update the cell's seed position."""
        self.position = new_position

    def increment_age(self, time_step):
        """Increment the cell's age by the given time step."""
        self.age += time_step

    def divide(self):
        """Perform cell division, increasing the division count."""
        if self.is_senescent:
            return False
        self.divisions += 1
        self.age = 0.0
        return True

    def induce_senescence(self, cause):
        """Induce senescence in the cell."""
        if self.is_senescent:
            return False
        self.is_senescent = True
        self.senescence_cause = cause
        return True

    def apply_shear_stress(self, shear_stress, duration):
        """Apply shear stress to the cell for the given duration."""
        self.local_shear_stress = shear_stress
        self.stress_exposure_time += duration

    def calculate_senescence_probability(self, config):
        """Calculate probability of senescence based on cell state and conditions."""
        if self.is_senescent:
            return {'telomere': 0.0, 'stress': 0.0}

        # Telomere-induced senescence probability
        tel_prob = 0.0
        if self.divisions >= config.max_divisions:
            tel_prob = 1.0
        elif self.divisions > 0.7 * config.max_divisions:
            tel_prob = ((self.divisions - 0.7 * config.max_divisions) /
                        (0.3 * config.max_divisions)) * 0.5

        # Stress-induced senescence probability (affected by compression)
        stress_factor = self._calculate_stress_factor(config)
        # More compressed cells are more likely to become senescent
        compression_stress = max(0, 2.0 - self.compression_ratio)
        stress_prob = min((stress_factor + compression_stress) * self.stress_exposure_time * config.time_step, 0.95)

        return {'telomere': tel_prob, 'stress': stress_prob}

    def _calculate_stress_factor(self, config):
        """Calculate stress factor based on shear stress magnitude."""
        tau = self.local_shear_stress
        if tau <= 10:
            return 0.002 + tau * 0.0005
        elif tau <= 20:
            return 0.007 + (tau - 10) * 0.001
        else:
            return 0.017 + (tau - 20) * 0.005

    def get_state_dict(self):
        """Get a dictionary representation of the cell state."""
        return {
            'cell_id': self.cell_id,
            'position': self.position,
            'centroid': self.centroid,
            'divisions': self.divisions,
            'is_senescent': self.is_senescent,
            'senescence_cause': self.senescence_cause,
            'target_orientation': self.target_orientation,
            'actual_orientation': self.actual_orientation,
            'target_aspect_ratio': self.target_aspect_ratio,
            'actual_aspect_ratio': self.actual_aspect_ratio,
            'target_area': self.target_area,
            'actual_area': self.actual_area,
            'territory_size': len(self.territory_pixels),
            'perimeter': self.perimeter,
            'eccentricity': self.eccentricity,
            'circularity': self.circularity,
            'compactness': self.compactness,
            'compression_ratio': self.compression_ratio,
            'growth_pressure': self.growth_pressure,
            'age': self.age,
            'adhesion_strength': self.adhesion_strength,
            'response': self.response,
            'local_shear_stress': self.local_shear_stress,
            'stress_exposure_time': self.stress_exposure_time,
            # ADD these two lines:
            'senescent_growth_factor': self.senescent_growth_factor,
            'is_enlarged_senescent': self.senescent_growth_factor > 1.5
        }