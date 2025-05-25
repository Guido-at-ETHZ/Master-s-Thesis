"""
Spatial properties model for endothelial cell mechanotransduction with mosaic structure.
Updated to work with territory-based cells that adapt to available space.
Fixed division by zero errors.
"""
import numpy as np


class SpatialPropertiesModel:
    """
    Model for spatial arrangement and morphological adaptations of endothelial cells in a mosaic structure.

    This model calculates target properties that cells try to achieve while respecting
    the constraints of the Voronoi tessellation.
    """

    def __init__(self, config):
        """
        Initialize the spatial properties model with configuration parameters.

        Parameters:
            config: SimulationConfig object with parameter settings
        """
        self.config = config

        # Define pressure points for interpolation
        self.pressure_points = [0.0, 1.4]  # Pa

        # Normal cell parameters at different pressures
        self.normal_params = {
            'area': {0.0: 500, 1.4: 1000},  # pixels
            'aspect_ratio': {0.0: 1.803642, 1.4: 1.669303},
            'eccentricity': {0.0: 0.8322278133, 1.4: 0.8007098915},
            'circularity': {0.0: 0.4634693156, 1.4: 0.4579688549}
        }

        # Senescent cell parameters (constant, no pressure dependence)
        self.senescent_params = {
            'area': 2000,  # pixels
            'orientation': 45,  # degrees
            'aspect_ratio': 1.4,  # Less elongated than normal cells
            'eccentricity': 0.7,  # Less eccentric (more circular)
            'circularity': 0.6   # More circular
        }

        # Adaptation parameters
        self.orientation_adaptation_rate = 0.05  # How fast orientation adapts
        self.size_adaptation_rate = 0.1  # How fast size adapts
        self.shape_flexibility = 0.3  # How much cells can deviate from target shape

    def get_population_senescence_level(self, cells):
        """
        Calculate the senescence level of the population.

        Parameters:
            cells: Dictionary of Cell objects

        Returns:
            Senescence level as a percentage (0-100)
        """
        if not cells:
            return 0.0

        total_cells = len(cells)
        senescent_cells = sum(1 for cell in cells.values() if cell.is_senescent)

        return (senescent_cells / total_cells) * 100

    def interpolate_pressure_effect(self, param_name, pressure):
        """
        Linearly interpolate parameter value based on pressure.

        Parameters:
            param_name: Name of the parameter to interpolate
            pressure: Applied pressure in Pa

        Returns:
            Interpolated parameter value
        """
        if param_name not in self.normal_params:
            raise ValueError(f"Unknown parameter: {param_name}")

        # Get values at known pressure points
        values = self.normal_params[param_name]
        p0, p1 = self.pressure_points
        v0, v1 = values[p0], values[p1]

        # Linear interpolation
        if pressure <= p0:
            return v0
        elif pressure >= p1:
            return v1
        else:
            return v0 + (v1 - v0) * (pressure - p0) / (p1 - p0)

    def calculate_target_orientation(self, senescence_level, pressure, is_senescent):
        """
        Calculate target cell orientation based on senescence level and pressure.

        Parameters:
            senescence_level: Population senescence level (0-100%)
            pressure: Applied pressure in Pa
            is_senescent: Whether the individual cell is senescent

        Returns:
            Target orientation in radians with some variability
        """
        # Baseline orientation formula (in degrees)
        baseline_orientation_deg = 0.2250 * senescence_level + 22.5000

        if is_senescent:
            # Senescent cells have fixed orientation (no mechanoadaptation)
            target_orientation_deg = 45.0
        else:
            # Normal cells: orientation depends on pressure
            if pressure <= 0:
                target_orientation_deg = baseline_orientation_deg
            else:
                # Normal cells adapt under pressure
                aligned_orientation = baseline_orientation_deg * 0.7  # 30% more aligned
                factor = min(pressure / 1.4, 1.0)
                target_orientation_deg = baseline_orientation_deg * (1 - factor) + aligned_orientation * factor

        # Add variability - cells don't all have exactly the same orientation
        variability_deg = 15.0 if is_senescent else 10.0  # Senescent cells have more variability
        actual_orientation_deg = target_orientation_deg + np.random.normal(0, variability_deg)

        # Convert to radians
        return np.radians(actual_orientation_deg)

    def calculate_target_area(self, pressure, is_senescent, senescence_level=None, base_area=None):
        """
        Calculate target cell area based on pressure and senescence state.

        Parameters:
            pressure: Applied pressure in Pa
            is_senescent: Whether the individual cell is senescent
            senescence_level: Population senescence level (0-100%) for partial senescence
            base_area: Base area to use (if None, uses pressure-dependent value)

        Returns:
            Target cell area in pixels
        """
        if base_area is None:
            if is_senescent:
                base_area = self.senescent_params['area']
            else:
                base_area = self.interpolate_pressure_effect('area', pressure)

        # Add some biological variability
        variability_factor = np.random.uniform(0.8, 1.2)
        return max(1.0, base_area * variability_factor)  # Ensure minimum area

    def calculate_target_aspect_ratio(self, pressure, is_senescent):
        """
        Calculate target cell aspect ratio based on pressure and senescence state.

        Parameters:
            pressure: Applied pressure in Pa
            is_senescent: Whether the individual cell is senescent

        Returns:
            Target aspect ratio (major axis / minor axis)
        """
        if is_senescent:
            base_ratio = self.senescent_params['aspect_ratio']
        else:
            base_ratio = self.interpolate_pressure_effect('aspect_ratio', pressure)

        # Add variability
        variability_factor = np.random.uniform(0.9, 1.1)
        return max(1.0, base_ratio * variability_factor)

    def update_cell_properties(self, cell, pressure, dt, cells_dict=None):
        """
        Update a cell's target properties and allow adaptation toward them.

        Parameters:
            cell: Cell object to update
            pressure: Applied pressure in Pa
            dt: Time step
            cells_dict: Dictionary of all cells for population-level calculations

        Returns:
            Dictionary of updated property values
        """
        # Get population senescence level
        senescence_level = self.get_population_senescence_level(cells_dict) if cells_dict else 0

        # Calculate target properties
        target_orientation = self.calculate_target_orientation(senescence_level, pressure, cell.is_senescent)
        target_area = self.calculate_target_area(pressure, cell.is_senescent, senescence_level)
        target_aspect_ratio = self.calculate_target_aspect_ratio(pressure, cell.is_senescent)

        # Update cell's target properties
        cell.update_target_properties(target_orientation, target_aspect_ratio, target_area)

        # The actual adaptation happens in the cell's adapt_to_constraints method
        # which is called by the grid during tessellation updates

        return {
            'target_orientation': target_orientation,
            'target_area': target_area,
            'target_aspect_ratio': target_aspect_ratio,
            'actual_orientation': cell.actual_orientation,
            'actual_area': cell.actual_area,
            'actual_aspect_ratio': cell.actual_aspect_ratio
        }

    def calculate_collective_properties(self, cells_dict, pressure):
        """
        Calculate collective properties of the cell population.

        Parameters:
            cells_dict: Dictionary of Cell objects
            pressure: Applied pressure in Pa

        Returns:
            Dictionary with collective properties
        """
        if not cells_dict:
            return {
                'mean_actual_orientation': 0,
                'std_actual_orientation': 0,
                'mean_target_orientation': 0,
                'orientation_adaptation': 1.0,
                'mean_actual_area': 0,
                'mean_target_area': 0,
                'area_adaptation': 1.0,
                'mean_actual_aspect_ratio': 1.0,
                'mean_target_aspect_ratio': 1.0,
                'mean_compression_ratio': 1.0,
                'std_compression_ratio': 0,
                'mean_shape_deviation': 0,
                'adaptation_quality': 1.0
            }

        # Collect properties
        actual_orientations = []
        target_orientations = []
        actual_areas = []
        target_areas = []
        actual_aspect_ratios = []
        target_aspect_ratios = []
        compression_ratios = []
        shape_deviations = []

        for cell in cells_dict.values():
            actual_orientations.append(cell.actual_orientation)
            target_orientations.append(cell.target_orientation)
            actual_areas.append(max(0.1, cell.actual_area))  # Ensure minimum area
            target_areas.append(max(0.1, cell.target_area))  # Ensure minimum area
            actual_aspect_ratios.append(cell.actual_aspect_ratio)
            target_aspect_ratios.append(cell.target_aspect_ratio)
            compression_ratios.append(cell.compression_ratio)

            # Calculate shape deviation
            deviation = cell.get_shape_deviation()
            shape_deviations.append(deviation['total_deviation'])

        # Calculate area adaptation safely
        area_adaptations = []
        for a, t in zip(actual_areas, target_areas):
            if a > 0 and t > 0:
                adaptation = min(a/t, t/a)
                area_adaptations.append(adaptation)

        area_adaptation = np.mean(area_adaptations) if area_adaptations else 1.0

        # Calculate orientation adaptation safely
        orientation_diffs = []
        for a, t in zip(actual_orientations, target_orientations):
            diff = abs(a - t)
            # Handle angle wrapping
            if diff > np.pi:
                diff = 2 * np.pi - diff
            orientation_diffs.append(diff)

        mean_orientation_diff = np.mean(orientation_diffs) if orientation_diffs else 0
        orientation_adaptation = max(0, 1.0 - mean_orientation_diff / np.pi)

        # Calculate statistics
        return {
            'mean_actual_orientation': np.mean(actual_orientations) if actual_orientations else 0,
            'std_actual_orientation': np.std(actual_orientations) if actual_orientations else 0,
            'mean_target_orientation': np.mean(target_orientations) if target_orientations else 0,
            'orientation_adaptation': orientation_adaptation,

            'mean_actual_area': np.mean(actual_areas) if actual_areas else 0,
            'mean_target_area': np.mean(target_areas) if target_areas else 0,
            'area_adaptation': area_adaptation,

            'mean_actual_aspect_ratio': np.mean(actual_aspect_ratios) if actual_aspect_ratios else 1.0,
            'mean_target_aspect_ratio': np.mean(target_aspect_ratios) if target_aspect_ratios else 1.0,

            'mean_compression_ratio': np.mean(compression_ratios) if compression_ratios else 1.0,
            'std_compression_ratio': np.std(compression_ratios) if compression_ratios else 0,

            'mean_shape_deviation': np.mean(shape_deviations) if shape_deviations else 0,
            'adaptation_quality': max(0, 1.0 - np.mean(shape_deviations)) if shape_deviations else 1.0
        }

    def calculate_alignment_index(self, cells, flow_direction=0):
        """
        Calculate the alignment index for a collection of cells.

        Parameters:
            cells: Dictionary or list of Cell objects
            flow_direction: Direction of flow in radians (default: 0)

        Returns:
            Alignment index (0-1)
        """
        if not cells:
            return 0

        # Convert cells input to a list of cell objects
        if isinstance(cells, dict):
            cell_list = list(cells.values())
        else:
            cell_list = cells

        # Calculate alignment using actual orientations
        alignment_sum = 0
        cell_count = 0

        for cell in cell_list:
            # Use actual orientation from territory shape
            angle_diff = cell.actual_orientation - flow_direction
            # Normalize to [-pi, pi]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

            # Cosine of the angle difference gives alignment (-1 to 1)
            alignment = abs(np.cos(angle_diff))  # Use absolute value for bidirectional alignment

            alignment_sum += alignment
            cell_count += 1

        # Average alignment (0-1)
        if cell_count > 0:
            return alignment_sum / cell_count
        else:
            return 0

    def calculate_shape_index(self, cells):
        """
        Calculate the shape index for a collection of cells.

        Shape index = P/(sqrt(4πA)) where P is perimeter and A is area.

        Parameters:
            cells: Dictionary or list of Cell objects

        Returns:
            Average shape index
        """
        if not cells:
            return 0

        # Convert cells input to a list of cell objects
        if isinstance(cells, dict):
            cell_list = list(cells.values())
        else:
            cell_list = cells

        # Calculate shape index for each cell
        shape_sum = 0
        cell_count = 0

        for cell in cell_list:
            if cell.actual_area > 0 and cell.perimeter > 0:
                # Shape index = P/sqrt(4πA)
                shape_index = cell.perimeter / np.sqrt(4 * np.pi * cell.actual_area)
                shape_sum += shape_index
                cell_count += 1

        # Average shape index
        if cell_count > 0:
            return shape_sum / cell_count
        else:
            return 1.0  # Default value

    def calculate_packing_quality(self, cells):
        """
        Calculate how well cells are packed (how close they are to their targets).

        Parameters:
            cells: Dictionary or list of Cell objects

        Returns:
            Packing quality index (0-1, where 1 is perfect)
        """
        if not cells:
            return 1.0

        # Convert cells input to a list of cell objects
        if isinstance(cells, dict):
            cell_list = list(cells.values())
        else:
            cell_list = cells

        total_quality = 0
        cell_count = 0

        for cell in cell_list:
            # Quality based on how close actual properties are to targets
            deviation = cell.get_shape_deviation()
            quality = max(0, 1.0 - deviation['total_deviation'])
            total_quality += quality
            cell_count += 1

        return total_quality / cell_count if cell_count > 0 else 1.0

    def calculate_stress_adaptation_index(self, cells, target_pressure):
        """
        Calculate how well cells have adapted to the applied stress.

        Parameters:
            cells: Dictionary or list of Cell objects
            target_pressure: The pressure being applied

        Returns:
            Adaptation index (0-1)
        """
        if not cells:
            return 1.0

        # Convert cells input to a list of cell objects
        if isinstance(cells, dict):
            cell_list = list(cells.values())
        else:
            cell_list = cells

        total_adaptation = 0
        cell_count = 0

        for cell in cell_list:
            # For normal cells, check how well they've adapted to pressure
            if not cell.is_senescent:
                try:
                    # Compare actual orientation to expected orientation
                    expected_orientation = self.calculate_target_orientation(0, target_pressure, False)
                    orientation_diff = abs(cell.actual_orientation - expected_orientation)
                    if orientation_diff > np.pi:
                        orientation_diff = 2 * np.pi - orientation_diff

                    orientation_adaptation = max(0, 1.0 - orientation_diff / (np.pi / 4))  # Normalize by 45 degrees

                    # Compare actual aspect ratio to expected
                    expected_aspect_ratio = self.calculate_target_aspect_ratio(target_pressure, False)
                    if expected_aspect_ratio > 0:
                        aspect_ratio_diff = abs(cell.actual_aspect_ratio - expected_aspect_ratio) / expected_aspect_ratio
                        aspect_ratio_adaptation = max(0, 1.0 - aspect_ratio_diff)
                    else:
                        aspect_ratio_adaptation = 1.0

                    # Combined adaptation
                    adaptation = (orientation_adaptation + aspect_ratio_adaptation) / 2
                except:
                    adaptation = 1.0  # Default if calculation fails
            else:
                # Senescent cells don't adapt, so their "adaptation" is based on maintaining their state
                adaptation = 1.0  # They're perfectly adapted to being senescent

            total_adaptation += adaptation
            cell_count += 1

        return total_adaptation / cell_count if cell_count > 0 else 1.0

    def update_population_dynamics(self, cells_dict, pressure, dt):
        """
        Update the spatial properties of all cells in the population.

        Parameters:
            cells_dict: Dictionary of Cell objects
            pressure: Applied pressure in Pa
            dt: Time step

        Returns:
            Dictionary with population-level statistics
        """
        # Update each cell's target properties
        for cell in cells_dict.values():
            self.update_cell_properties(cell, pressure, dt, cells_dict)

        # Calculate collective properties
        collective_props = self.calculate_collective_properties(cells_dict, pressure)

        # Add additional metrics
        collective_props.update({
            'alignment_index': self.calculate_alignment_index(cells_dict),
            'shape_index': self.calculate_shape_index(cells_dict),
            'packing_quality': self.calculate_packing_quality(cells_dict),
            'stress_adaptation_index': self.calculate_stress_adaptation_index(cells_dict, pressure)
        })

        return collective_props