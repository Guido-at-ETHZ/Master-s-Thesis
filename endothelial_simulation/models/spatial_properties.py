"""
Spatial properties model for endothelial cell mechanotransduction.
Updated to include senescence and pressure-dependent morphometry.
"""
import numpy as np


class SpatialPropertiesModel:
    """
    Model for spatial arrangement and morphological adaptations of endothelial cells.

    This model calculates how mechanical forces and senescence influence cell orientation,
    aspect ratio, area, and other geometric properties.
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
            # For senescent cells, we need to infer reasonable values for these
            # Based on the pattern, senescent cells are larger and more circular
            'aspect_ratio': 1.4,  # Less elongated than normal cells
            'eccentricity': 0.7,  # Less eccentric (more circular)
            'circularity': 0.6   # More circular
        }

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
            # Linear interpolation formula
            return v0 + (v1 - v0) * (pressure - p0) / (p1 - p0)

    def calculate_cell_orientation(self, senescence_level, pressure, is_senescent):
        """
        Calculate cell orientation based on senescence level and pressure.

        Parameters:
            senescence_level: Population senescence level (0-100%)
            pressure: Applied pressure in Pa
            is_senescent: Whether the individual cell is senescent

        Returns:
            Orientation in radians
        """
        # Baseline orientation formula (in degrees)
        baseline_orientation_deg = 0.2250 * senescence_level + 22.5000

        if is_senescent:
            # Senescent cells have fixed orientation (no mechanoadaptation)
            orientation_deg = 45.0  # As per specification
        else:
            # Normal cells: orientation depends on pressure
            # At 0 Pa: use baseline formula
            # At 1.4 Pa: cells align more with flow (lower angle)

            if pressure <= 0:
                orientation_deg = baseline_orientation_deg
            else:
                # Normal cells adapt under pressure
                # Interpolate between baseline and more aligned state
                # At 1.4 Pa, normal cells become more aligned
                aligned_orientation = baseline_orientation_deg * 0.7  # 30% more aligned

                # Linear interpolation based on pressure
                factor = min(pressure / 1.4, 1.0)
                orientation_deg = baseline_orientation_deg * (1 - factor) + aligned_orientation * factor

        # Convert to radians
        return np.radians(orientation_deg)

    def calculate_cell_area(self, pressure, is_senescent, senescence_level=None):
        """
        Calculate cell area based on pressure and senescence state.

        Parameters:
            pressure: Applied pressure in Pa
            is_senescent: Whether the individual cell is senescent
            senescence_level: Population senescence level (0-100%) for partial senescence

        Returns:
            Cell area in pixels
        """
        if is_senescent:
            # Senescent cells have fixed area (no mechanoadaptation)
            return self.senescent_params['area']
        else:
            # Normal cells: area depends on pressure
            return self.interpolate_pressure_effect('area', pressure)

    def calculate_cell_aspect_ratio(self, pressure, is_senescent):
        """
        Calculate cell aspect ratio based on pressure and senescence state.

        Parameters:
            pressure: Applied pressure in Pa
            is_senescent: Whether the individual cell is senescent

        Returns:
            Aspect ratio (major axis / minor axis)
        """
        if is_senescent:
            # Senescent cells have fixed aspect ratio
            return self.senescent_params['aspect_ratio']
        else:
            # Normal cells: aspect ratio depends on pressure
            return self.interpolate_pressure_effect('aspect_ratio', pressure)

    def calculate_cell_eccentricity(self, pressure, is_senescent):
        """
        Calculate cell eccentricity based on pressure and senescence state.

        Parameters:
            pressure: Applied pressure in Pa
            is_senescent: Whether the individual cell is senescent

        Returns:
            Eccentricity value
        """
        if is_senescent:
            return self.senescent_params['eccentricity']
        else:
            return self.interpolate_pressure_effect('eccentricity', pressure)

    def calculate_cell_circularity(self, pressure, is_senescent):
        """
        Calculate cell circularity based on pressure and senescence state.

        Parameters:
            pressure: Applied pressure in Pa
            is_senescent: Whether the individual cell is senescent

        Returns:
            Circularity value
        """
        if is_senescent:
            return self.senescent_params['circularity']
        else:
            return self.interpolate_pressure_effect('circularity', pressure)

    def calculate_cell_perimeter(self, area, aspect_ratio):
        """
        Calculate cell perimeter based on area and aspect ratio.

        Parameters:
            area: Cell area in pixels
            aspect_ratio: Cell aspect ratio

        Returns:
            Cell perimeter in pixels
        """
        # Model cell as an ellipse
        # Area = π * a * b, where a is semi-major axis, b is semi-minor axis
        # aspect_ratio = a / b

        # From area and aspect ratio, calculate semi-axes
        b = np.sqrt(area / (np.pi * aspect_ratio))
        a = aspect_ratio * b

        # Use Ramanujan's approximation for ellipse perimeter
        h = ((a - b) / (a + b)) ** 2
        perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))

        return perimeter

    def calculate_optimal_properties(self, pressure, is_senescent, cells=None):
        """
        Calculate all optimal cell properties under given conditions.

        Parameters:
            pressure: Applied pressure in Pa
            is_senescent: Whether the cell is senescent
            cells: Dictionary of all cells (for population senescence level)

        Returns:
            Dictionary of optimal property values
        """
        # Get population senescence level if cells provided
        senescence_level = self.get_population_senescence_level(cells) if cells else 0

        # Calculate orientation
        orientation = self.calculate_cell_orientation(senescence_level, pressure, is_senescent)

        # Calculate other properties
        area = self.calculate_cell_area(pressure, is_senescent, senescence_level)
        aspect_ratio = self.calculate_cell_aspect_ratio(pressure, is_senescent)
        eccentricity = self.calculate_cell_eccentricity(pressure, is_senescent)
        circularity = self.calculate_cell_circularity(pressure, is_senescent)

        # Calculate perimeter based on area and aspect ratio
        perimeter = self.calculate_cell_perimeter(area, aspect_ratio)

        # Add some random variation for biological realism
        if not is_senescent:
            # Normal cells have more variation
            area *= np.random.uniform(0.9, 1.1)
            aspect_ratio *= np.random.uniform(0.95, 1.05)
        else:
            # Senescent cells have less variation
            area *= np.random.uniform(0.95, 1.05)
            aspect_ratio *= np.random.uniform(0.98, 1.02)

        return {
            'orientation': orientation,
            'aspect_ratio': aspect_ratio,
            'area': area,
            'perimeter': perimeter,
            'eccentricity': eccentricity,
            'circularity': circularity
        }

    def update_cell_properties(self, cell, pressure, dt, adaptation_rate=0.2):
        """
        Update a cell's spatial properties based on pressure.

        Parameters:
            cell: Cell object to update
            pressure: Applied pressure in Pa
            dt: Time step
            adaptation_rate: Rate at which properties adapt to optimal values

        Returns:
            Dictionary of updated property values
        """
        # Get all cells from the grid (assuming cell has access to grid through some mechanism)
        # For now, we'll calculate properties without population context
        # In practice, you'd pass the cells dictionary here

        # Calculate optimal properties
        optimal = self.calculate_optimal_properties(pressure, cell.is_senescent)

        # For senescent cells with mechanoadaptation inhibited,
        # properties change more slowly or not at all
        if cell.is_senescent:
            adaptation_rate *= 0.1  # 90% reduction in adaptation rate

        # Current properties
        current_properties = {
            'orientation': cell.orientation,
            'aspect_ratio': cell.aspect_ratio,
            'area': cell.area
        }

        updated_properties = {}

        # Update each property with adaptation
        for prop, current in current_properties.items():
            optimal_value = optimal[prop]

            # For orientation, handle circular nature
            if prop == 'orientation':
                # Calculate difference accounting for circular nature of angles
                diff = optimal_value - current
                # Normalize to [-pi, pi]
                diff = (diff + np.pi) % (2 * np.pi) - np.pi

                # Update with partial adaptation
                updated_properties[prop] = current + adaptation_rate * diff * dt

                # Normalize result to [-pi, pi]
                updated_properties[prop] = (updated_properties[prop] + np.pi) % (2 * np.pi) - np.pi
            else:
                # Regular properties use simple exponential approach
                diff = optimal_value - current
                updated_properties[prop] = current + adaptation_rate * diff * dt

        # Ensure positive values for size-related properties
        updated_properties['area'] = max(100, updated_properties['area'])
        updated_properties['aspect_ratio'] = max(1.0, updated_properties['aspect_ratio'])

        # Recalculate perimeter based on updated area and aspect ratio
        updated_properties['perimeter'] = self.calculate_cell_perimeter(
            updated_properties['area'],
            updated_properties['aspect_ratio']
        )

        # Update eccentricity and circularity
        updated_properties['eccentricity'] = optimal['eccentricity']
        updated_properties['circularity'] = optimal['circularity']

        # Update cell properties
        cell.update_shape(
            updated_properties['orientation'],
            updated_properties['aspect_ratio'],
            updated_properties['area']
        )

        return updated_properties

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

        # Calculate the average cosine of the angle between cell orientation and flow
        alignment_sum = 0
        cell_count = 0

        for cell in cell_list:
            # Calculate angle between cell orientation and flow direction
            angle_diff = cell.orientation - flow_direction
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
            # Shape index = P/sqrt(4πA)
            shape_index = cell.perimeter / np.sqrt(4 * np.pi * cell.area)

            shape_sum += shape_index
            cell_count += 1

        # Average shape index
        if cell_count > 0:
            return shape_sum / cell_count
        else:
            return 0

    def calculate_population_averaged_property(self, cells, pressure, property_name):
        """
        Calculate population-averaged property considering partial senescence.

        This implements the linear combination formula for the population:
        Value_population = (S/100) * Value_senescent + (1 - S/100) * Value_normal(P)

        Parameters:
            cells: Dictionary of Cell objects
            pressure: Applied pressure in Pa
            property_name: Name of the property to calculate

        Returns:
            Population-averaged property value
        """
        if not cells:
            return 0

        senescence_level = self.get_population_senescence_level(cells)
        S = senescence_level / 100  # Convert to fraction

        # Get values for normal and senescent cells
        if property_name == 'area':
            value_normal = self.interpolate_pressure_effect('area', pressure)
            value_senescent = self.senescent_params['area']
        elif property_name == 'aspect_ratio':
            value_normal = self.interpolate_pressure_effect('aspect_ratio', pressure)
            value_senescent = self.senescent_params['aspect_ratio']
        elif property_name == 'eccentricity':
            value_normal = self.interpolate_pressure_effect('eccentricity', pressure)
            value_senescent = self.senescent_params['eccentricity']
        elif property_name == 'circularity':
            value_normal = self.interpolate_pressure_effect('circularity', pressure)
            value_senescent = self.senescent_params['circularity']
        else:
            raise ValueError(f"Unknown property: {property_name}")

        # Linear combination
        return S * value_senescent + (1 - S) * value_normal