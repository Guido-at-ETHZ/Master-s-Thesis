"""
Updated spatial properties model using real experimental data.
Only includes measurable parameters: area, aspect ratio, and orientation.
"""
import numpy as np


class SpatialPropertiesModel:
    """
    Model for spatial arrangement and morphological adaptations of endothelial cells.
    Uses real experimental data for area, aspect ratio, and orientation.
    """

    def __init__(self, config, temporal_model=None):
        """
        Initialize the spatial properties model with real experimental parameters.

        Parameters:
            config: SimulationConfig object with parameter settings
            temporal_model: TemporalDynamicsModel instance for shared time constant calculation
        """
        self.config = config
        self.temporal_model = temporal_model

        # Real experimental data converted to pixels
        # Pixel spacing: 0.429 μm/pixel, so 1 pixel² = 0.184041 μm²

        # Control cell parameters at different pressures (in pixels)
        self.control_params = {
            'area': {
                0.0: 11712,    # 2155 μm² static
                1.4: 11712     # Assume same area under flow (not measured separately)
            },
            'aspect_ratio': {
                0.0: 1.9,      # Static control
                1.4: 200.3       # Flow control (increased elongation) 2.3 in reality
            },
            'orientation_mean': {
                0.0: 49.0,     # Random orientation (degrees)
                1.4: 20.0      # Aligned with flow (degrees)
            },
            'orientation_std': {
                0.0: 25.0,     # Standard deviation for static
                1.4: 14.0      # Standard deviation for flow (more aligned)
            }
        }

        # Senescent cell parameters (in pixels)
        self.senescent_params = {
            'area_small': 11995,     # 2207 μm² (< 5000 μm²)
            'area_large': 46869,     # 8626 μm² (≥ 5000 μm²)
            'area_threshold': 27174, # 5000 μm² threshold in pixels
            'aspect_ratio': {
                0.0: 1.9,            # Static senescent
                1.4: 2.0             # Flow senescent (no significant change)
            },
            'orientation_mean': {
                0.0: 42.0,           # Random orientation static
                1.4: 45.0            # Random orientation flow (no alignment)
            },
            'orientation_std': {
                0.0: 26.0,           # Standard deviation static
                1.4: 27.0            # Standard deviation flow (still random)
            }
        }

        # Probability that a senescent cell will be large (adjustable parameter)
        self.large_senescent_probability = 0.3  # 30% of senescent cells become large

    def interpolate_pressure_effect(self, param_dict, pressure):
        """
        Linearly interpolate parameter value based on pressure.

        Parameters:
            param_dict: Dictionary with pressure-dependent values
            pressure: Applied pressure in Pa

        Returns:
            Interpolated parameter value
        """
        # Get values at known pressure points (0.0 and 1.4 Pa)
        p0, p1 = 0.0, 1.4
        v0, v1 = param_dict[p0], param_dict[p1]

        # Linear interpolation
        if pressure <= p0:
            return v0
        elif pressure >= p1:
            return v1
        else:
            return v0 + (v1 - v0) * (pressure - p0) / (p1 - p0)

    def calculate_target_area(self, pressure, is_senescent, senescence_cause=None):
        """
        Calculate target cell area based on pressure and senescence state.

        Parameters:
            pressure: Applied pressure in Pa
            is_senescent: Whether the cell is senescent
            senescence_cause: Type of senescence (not used for area calculation)

        Returns:
            Target cell area in pixels²
        """
        if is_senescent:
            # Randomly assign small or large senescent area
            if np.random.random() < self.large_senescent_probability:
                return self.senescent_params['area_large']
            else:
                return self.senescent_params['area_small']
        else:
            # Control cells (area doesn't change with pressure based on data)
            base_area = self.interpolate_pressure_effect(self.control_params['area'], pressure)
            # Add biological variability (±5% based on experimental std)
            variability = np.random.normal(1.0, 0.05)
            return max(1000, base_area * variability)  # Minimum 1000 pixels²

    def calculate_target_aspect_ratio(self, pressure, is_senescent):
        """
        Calculate target cell aspect ratio using YOUR experimental values.
        """
        if is_senescent:
            # Senescent cells: no significant response to flow
            base_ratio = self.interpolate_pressure_effect(self.senescent_params['aspect_ratio'], pressure)
            std_dev = 0.1  # Small variation
            variability = np.random.normal(1.0, std_dev)
            return max(1.0, base_ratio * variability)
        else:
            # Control cells: USE YOUR EXACT VALUES
            base_ratio = self.interpolate_pressure_effect(self.control_params['aspect_ratio'], pressure)
            std_dev = 0.05  # Small variation to see your exact values
            variability = np.random.normal(1.0, std_dev)
            return max(1.0, base_ratio * variability)  # NO CAPPING - use your 200.3!

    def calculate_target_orientation(self, pressure, is_senescent):
        """
        Calculate target cell orientation based on pressure and senescence state.
        """
        if is_senescent:
            # Senescent cells: remain randomly oriented regardless of flow
            return np.random.uniform(-np.pi, np.pi)
        else:
            # Normal cells: align with flow direction (0°) under shear stress
            if pressure <= 0.0:
                # No flow: random orientation
                return np.random.uniform(-np.pi, np.pi)
            else:
                # Flow present: align horizontally (0° ± small variation)
                flow_direction = 0.0  # Horizontal flow
                variability = 0.2 * np.exp(-pressure * 0.5)  # Less variation at higher pressure
                return np.random.normal(flow_direction, variability)

    def update_cell_properties(self, cell, pressure, dt, cells_dict=None):
        """
        Update a cell's target properties using temporal dynamics with real parameters.

        Parameters:
            cell: Cell object to update
            pressure: Applied pressure in Pa
            dt: Time step
            cells_dict: Dictionary of all cells (not used in this simplified version)

        Returns:
            Dictionary of updated property values and dynamics info
        """
        # Calculate instantaneous target values
        instant_target_area = self.calculate_target_area(pressure, cell.is_senescent, cell.senescence_cause)
        instant_target_aspect_ratio = self.calculate_target_aspect_ratio(pressure, cell.is_senescent)
        instant_target_orientation = self.calculate_target_orientation(pressure, cell.is_senescent)

        # Initialize cell target properties if not already set
        if not hasattr(cell, 'target_orientation') or cell.target_orientation is None:
            cell.target_orientation = instant_target_orientation
        if not hasattr(cell, 'target_area') or cell.target_area is None:
            cell.target_area = instant_target_area
        if not hasattr(cell, 'target_aspect_ratio') or cell.target_aspect_ratio is None:
            cell.target_aspect_ratio = instant_target_aspect_ratio

        # For senescent cells, targets don't adapt (they're mechanically impaired)
        if cell.is_senescent:
            # Senescent cells maintain fixed properties - no temporal adaptation
            cell.target_orientation = instant_target_orientation
            cell.target_area = instant_target_area
            cell.target_aspect_ratio = instant_target_aspect_ratio

            dynamics_info = {'adaptation_disabled': True}
        else:
            # Normal cells adapt with temporal dynamics
            dynamics_info = {}

            # Get time constants from temporal model if available
            if self.temporal_model:
                tau_area, _ = self.temporal_model.get_scaled_tau_and_amax(pressure, 'area')
                tau_orient, _ = self.temporal_model.get_scaled_tau_and_amax(pressure, 'orientation')
                tau_ar, _ = self.temporal_model.get_scaled_tau_and_amax(pressure, 'aspect_ratio')
            else:
                # Fallback time constants (in minutes)
                tau_area = 30.0
                tau_orient = 45.0  # Orientation changes slightly slower
                tau_ar = 60.0     # Aspect ratio changes slowest

            # 1. Area evolution
            area_diff = instant_target_area - cell.target_area
            cell.target_area += dt * area_diff / tau_area
            dynamics_info['tau_area'] = tau_area

            # 2. Orientation evolution (handle angle wrapping)
            orientation_diff = instant_target_orientation - cell.target_orientation
            while orientation_diff > np.pi:
                orientation_diff -= 2 * np.pi
            while orientation_diff < -np.pi:
                orientation_diff += 2 * np.pi

            cell.target_orientation += dt * orientation_diff / tau_orient
            dynamics_info['tau_orientation'] = tau_orient

            # 3. Aspect ratio evolution
            ar_diff = instant_target_aspect_ratio - cell.target_aspect_ratio
            cell.target_aspect_ratio += dt * ar_diff / tau_ar
            dynamics_info['tau_aspect_ratio'] = tau_ar

        # Update the cell's properties
        cell.update_target_properties(
            cell.target_orientation,
            cell.target_aspect_ratio,
            cell.target_area
        )

        return {
            'target_orientation': cell.target_orientation,
            'target_area': cell.target_area,
            'target_aspect_ratio': cell.target_aspect_ratio,
            'actual_orientation': cell.actual_orientation,
            'actual_area': cell.actual_area,
            'actual_aspect_ratio': cell.actual_aspect_ratio,
            'dynamics_info': dynamics_info
        }

    def calculate_collective_properties(self, cells_dict, pressure):
        """
        Calculate collective properties focusing on real measured parameters.

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
                'mean_actual_area': 0,
                'mean_target_area': 0,
                'mean_actual_aspect_ratio': 1.0,
                'mean_target_aspect_ratio': 1.0,
                'orientation_alignment': 0,
                'area_adaptation': 1.0,
                'aspect_ratio_adaptation': 1.0
            }

        # Collect real measured properties
        actual_orientations = []
        target_orientations = []
        actual_areas = []
        target_areas = []
        actual_aspect_ratios = []
        target_aspect_ratios = []

        for cell in cells_dict.values():
            actual_orientations.append(cell.actual_orientation)
            target_orientations.append(getattr(cell, 'target_orientation', cell.actual_orientation))
            actual_areas.append(cell.actual_area)
            target_areas.append(getattr(cell, 'target_area', cell.actual_area))
            actual_aspect_ratios.append(cell.actual_aspect_ratio)
            target_aspect_ratios.append(getattr(cell, 'target_aspect_ratio', cell.actual_aspect_ratio))

        # Calculate alignment index (how well oriented toward flow direction)
        # Flow direction is 0 degrees (horizontal)
        flow_direction = 0.0
        alignment_scores = []
        for orientation in actual_orientations:
            # Calculate how close the orientation is to flow direction
            angle_diff = abs(orientation - flow_direction)
            # Handle angle wrapping
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            # Convert to alignment score (1 = perfectly aligned, 0 = perpendicular)
            alignment = np.cos(angle_diff)
            alignment_scores.append(alignment)

        # Calculate adaptation quality
        orientation_adaptation = np.mean([
            1.0 - min(1.0, abs(a - t) / np.pi)
            for a, t in zip(actual_orientations, target_orientations)
        ])

        area_adaptation = np.mean([
            min(a/t, t/a) if t > 0 else 1.0
            for a, t in zip(actual_areas, target_areas)
        ])

        aspect_ratio_adaptation = np.mean([
            min(a/t, t/a) if t > 0 else 1.0
            for a, t in zip(actual_aspect_ratios, target_aspect_ratios)
        ])

        return {
            'mean_actual_orientation': np.degrees(np.mean(actual_orientations)),  # Convert to degrees for display
            'std_actual_orientation': np.degrees(np.std(actual_orientations)),
            'mean_target_orientation': np.degrees(np.mean(target_orientations)),

            'mean_actual_area': np.mean(actual_areas),
            'mean_target_area': np.mean(target_areas),

            'mean_actual_aspect_ratio': np.mean(actual_aspect_ratios),
            'mean_target_aspect_ratio': np.mean(target_aspect_ratios),

            'orientation_alignment': np.mean(alignment_scores),  # How aligned with flow
            'area_adaptation': area_adaptation,
            'aspect_ratio_adaptation': aspect_ratio_adaptation,

            # Additional metrics
            'large_senescent_fraction': len([c for c in cells_dict.values()
                                           if c.is_senescent and getattr(c, 'target_area', 0) > 27174]) / len(cells_dict),
            'pressure': pressure
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

        if isinstance(cells, dict):
            cell_list = list(cells.values())
        else:
            cell_list = cells

        alignment_sum = 0
        cell_count = 0

        for cell in cell_list:
            # Convert to alignment angle (0-90°)
            orientation_rad = cell.actual_orientation
            alignment_angle = np.abs(orientation_rad) % (np.pi / 2)  # 0 to π/2 radians

            # Convert to alignment score (1 = perfectly aligned, 0 = perpendicular)
            alignment_score = np.cos(alignment_angle)  # cos(0) = 1, cos(π/2) = 0

            alignment_sum += alignment_score
            cell_count += 1

        return alignment_sum / cell_count if cell_count > 0 else 0

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
            orientation_quality = 1.0
            area_quality = 1.0
            aspect_ratio_quality = 1.0

            # Orientation quality
            if hasattr(cell, 'target_orientation'):
                orientation_diff = abs(cell.actual_orientation - cell.target_orientation)
                if orientation_diff > np.pi:
                    orientation_diff = 2 * np.pi - orientation_diff
                orientation_quality = max(0, 1.0 - orientation_diff / np.pi)

            # Area quality
            if hasattr(cell, 'target_area') and cell.target_area > 0:
                area_ratio = min(cell.actual_area / cell.target_area,
                                 cell.target_area / cell.actual_area)
                area_quality = area_ratio

            # Aspect ratio quality
            if hasattr(cell, 'target_aspect_ratio') and cell.target_aspect_ratio > 0:
                ar_ratio = min(cell.actual_aspect_ratio / cell.target_aspect_ratio,
                               cell.target_aspect_ratio / cell.actual_aspect_ratio)
                aspect_ratio_quality = ar_ratio

            # Combined quality (average of the three components)
            cell_quality = (orientation_quality + area_quality + aspect_ratio_quality) / 3
            total_quality += cell_quality
            cell_count += 1

        return total_quality / cell_count if cell_count > 0 else 1.0

    def calculate_stress_adaptation_index(self, cells, target_pressure):
        """
        Calculate how well cells have adapted to the applied stress based on real parameters.

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
            if not cell.is_senescent:
                # For normal cells, check adaptation to expected experimental values
                expected = self.get_expected_values(target_pressure, 'control')

                # Orientation adaptation (how close to expected mean)
                expected_orientation_rad = np.radians(expected['orientation_mean'])
                orientation_diff = abs(cell.actual_orientation - expected_orientation_rad)
                if orientation_diff > np.pi:
                    orientation_diff = 2 * np.pi - orientation_diff
                orientation_adaptation = max(0, 1.0 - orientation_diff / (np.pi / 2))  # Normalize by 90 degrees

                # Aspect ratio adaptation
                expected_ar = expected['aspect_ratio']
                if expected_ar > 0:
                    ar_ratio = min(cell.actual_aspect_ratio / expected_ar,
                                   expected_ar / cell.actual_aspect_ratio)
                    aspect_ratio_adaptation = ar_ratio
                else:
                    aspect_ratio_adaptation = 1.0

                # Area adaptation
                expected_area = expected['area']
                if expected_area > 0:
                    area_ratio = min(cell.actual_area / expected_area,
                                     expected_area / cell.actual_area)
                    area_adaptation = area_ratio
                else:
                    area_adaptation = 1.0

                # Combined adaptation
                adaptation = (orientation_adaptation + aspect_ratio_adaptation + area_adaptation) / 3
            else:
                # Senescent cells don't adapt, so their "adaptation" is maintaining senescent state
                adaptation = 1.0

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
        # Update each cell's target properties with temporal dynamics
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

    def get_expected_values(self, pressure, cell_type='control'):
        """
        Get expected experimental values for comparison.

        Parameters:
            pressure: Applied pressure in Pa
            cell_type: 'control' or 'senescent'

        Returns:
            Dictionary with expected values
        """
        if cell_type == 'control':
            return {
                'area': self.interpolate_pressure_effect(self.control_params['area'], pressure),
                'aspect_ratio': self.interpolate_pressure_effect(self.control_params['aspect_ratio'], pressure),
                'orientation_mean': self.interpolate_pressure_effect(self.control_params['orientation_mean'], pressure),
                'orientation_std': self.interpolate_pressure_effect(self.control_params['orientation_std'], pressure)
            }
        else:  # senescent
            return {
                'area_small': self.senescent_params['area_small'],
                'area_large': self.senescent_params['area_large'],
                'aspect_ratio': self.interpolate_pressure_effect(self.senescent_params['aspect_ratio'], pressure),
                'orientation_mean': self.interpolate_pressure_effect(self.senescent_params['orientation_mean'], pressure),
                'orientation_std': self.interpolate_pressure_effect(self.senescent_params['orientation_std'], pressure)
            }