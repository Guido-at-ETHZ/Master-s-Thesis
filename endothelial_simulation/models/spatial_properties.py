"""
Spatial properties model for endothelial cell mechanotransduction.
"""
import numpy as np


class SpatialPropertiesModel:
    """
    Model for spatial arrangement and morphological adaptations of endothelial cells.

    This model calculates how mechanical forces influence cell orientation, aspect ratio,
    and other geometric properties.
    """

    def __init__(self, config):
        """
        Initialize the spatial properties model with configuration parameters.

        Parameters:
            config: SimulationConfig object with parameter settings
        """
        self.config = config

    def calculate_optimal_orientation(self, shear_stress, is_senescent=False):
        """
        Calculate the optimal cell orientation under given shear stress.

        Parameters:
            shear_stress: Magnitude of wall shear stress (Pa)
            is_senescent: Whether the cell is senescent

        Returns:
            Optimal orientation angle in radians (0 = aligned with flow)
        """
        # Orientation is primarily determined by flow direction (0 radians = aligned with flow)
        # Orientation approaches 0 (aligned with flow) as shear stress increases

        # Baseline random orientation if no flow
        if shear_stress < 0.2:
            return np.random.uniform(-np.pi / 2, np.pi / 2)

        # Senescent cells have impaired ability to align with flow
        if is_senescent:
            # More random orientation even under flow
            max_deviation = max(0.2, np.exp(-0.2 * shear_stress)) * np.pi / 2
            return np.random.uniform(-max_deviation, max_deviation)

        # For healthy cells, orientation becomes more aligned with flow as shear stress increases
        # Small random deviation that decreases with increasing shear stress
        max_deviation = np.exp(-0.5 * shear_stress) * np.pi / 4
        return np.random.uniform(-max_deviation, max_deviation)

    def calculate_optimal_aspect_ratio(self, shear_stress, orientation, is_senescent=False):
        """
        Calculate the optimal cell aspect ratio under given conditions.

        Parameters:
            shear_stress: Magnitude of wall shear stress (Pa)
            orientation: Cell orientation in radians
            is_senescent: Whether the cell is senescent

        Returns:
            Optimal aspect ratio (ratio of major axis to minor axis)
        """
        # Base aspect ratio (no shear stress)
        base_aspect_ratio = 1.2

        # Senescent cells have impaired ability to elongate
        if is_senescent:
            max_aspect_ratio = 1.5
            shear_effect = 0.3 * (1 - np.exp(-0.1 * shear_stress))
            return base_aspect_ratio + shear_effect

        # Maximum aspect ratio under high shear stress
        max_aspect_ratio = 4.0

        # Effect of shear stress on aspect ratio
        # Saturating function: approaches max_aspect_ratio as shear stress increases
        shear_effect = (max_aspect_ratio - base_aspect_ratio) * (1 - np.exp(-0.2 * shear_stress))

        # Effect of orientation on aspect ratio
        # Cells tend to be more elongated when aligned with flow (orientation ≈ 0)
        # or perpendicular to flow (orientation ≈ π/2)
        orientation_factor = np.cos(2 * orientation) ** 2

        # Combine effects
        aspect_ratio = base_aspect_ratio + shear_effect * orientation_factor

        return aspect_ratio

    def calculate_optimal_area(self, shear_stress, aspect_ratio, is_senescent=False):
        """
        Calculate the optimal cell area under given conditions.

        Parameters:
            shear_stress: Magnitude of wall shear stress (Pa)
            aspect_ratio: Cell aspect ratio
            is_senescent: Whether the cell is senescent

        Returns:
            Optimal cell area in square pixels
        """
        # Base area for a healthy cell
        base_area = 100.0  # square pixels

        # Senescent cells are significantly larger
        if is_senescent:
            # Senescent cells are 2-3x larger than healthy cells
            return base_area * np.random.uniform(2.0, 3.0)

        # Effect of shear stress on area
        # Cells tend to become slightly smaller under high shear stress
        shear_factor = max(0.8, 1.0 - 0.01 * shear_stress)

        # Effect of aspect ratio on area
        # More elongated cells tend to have slightly larger area
        elongation_factor = 1.0 + 0.05 * (aspect_ratio - 1.0)

        # Combine effects with some random variation
        area = base_area * shear_factor * elongation_factor * np.random.uniform(0.9, 1.1)

        return area

    def calculate_optimal_properties(self, shear_stress, is_senescent=False):
        """
        Calculate all optimal cell properties under given conditions.

        Parameters:
            shear_stress: Magnitude of wall shear stress (Pa)
            is_senescent: Whether the cell is senescent

        Returns:
            Dictionary of optimal property values
        """
        # Calculate orientation first
        orientation = self.calculate_optimal_orientation(shear_stress, is_senescent)

        # Calculate aspect ratio based on orientation
        aspect_ratio = self.calculate_optimal_aspect_ratio(shear_stress, orientation, is_senescent)

        # Calculate area based on aspect ratio
        area = self.calculate_optimal_area(shear_stress, aspect_ratio, is_senescent)

        # Calculate perimeter based on area and aspect ratio
        # Approximation for ellipse perimeter
        a = np.sqrt(area * aspect_ratio)  # Semi-major axis
        b = area / a  # Semi-minor axis

        # Ramanujan's approximation for ellipse perimeter
        h = ((a - b) / (a + b)) ** 2
        perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))

        return {
            'orientation': orientation,
            'aspect_ratio': aspect_ratio,
            'area': area,
            'perimeter': perimeter
        }

    def update_cell_properties(self, cell, shear_stress, dt, adaptation_rate=0.2):
        """
        Update a cell's spatial properties based on shear stress.

        Parameters:
            cell: Cell object to update
            shear_stress: Magnitude of wall shear stress (Pa)
            dt: Time step
            adaptation_rate: Rate at which properties adapt to optimal values

        Returns:
            Dictionary of updated property values
        """
        # Calculate optimal properties
        optimal = self.calculate_optimal_properties(shear_stress, cell.is_senescent)

        # For each property, move current value toward optimal value
        # Use exponential approach: change is proportional to distance from optimal
        current_properties = {
            'orientation': cell.orientation,
            'aspect_ratio': cell.aspect_ratio,
            'area': cell.area
        }

        updated_properties = {}

        for prop, current in current_properties.items():
            optimal_value = optimal[prop]

            # For orientation, handle circular/angular nature
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

        # Calculate updated perimeter based on area and aspect ratio
        a = np.sqrt(updated_properties['area'] * updated_properties['aspect_ratio'])
        b = updated_properties['area'] / a
        h = ((a - b) / (a + b)) ** 2
        updated_properties['perimeter'] = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))

        # Update cell properties
        cell.update_shape(
            updated_properties['orientation'],
            updated_properties['aspect_ratio'],
            updated_properties['area']
        )

        return updated_properties

    def calculate_elongation_index(self, aspect_ratio):
        """
        Calculate the elongation index from aspect ratio.

        Elongation index = (L-W)/(L+W) where L is length and W is width.

        Parameters:
            aspect_ratio: Ratio of major axis to minor axis

        Returns:
            Elongation index (0-1)
        """
        return (aspect_ratio - 1) / (aspect_ratio + 1)

    def calculate_alignment_index(self, cells, flow_direction=0):
        """
        Calculate the alignment index for a collection of cells.

        Alignment index is a measure of how well cells are aligned with the flow direction.

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