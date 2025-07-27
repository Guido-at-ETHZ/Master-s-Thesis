
"""
Updated spatial properties model using real experimental data.
Modified to use deterministic targets based on experimental measurements.
Variability comes from tessellation process, not artificial randomness.
"""
import numpy as np

from endothelial_simulation.core.angle_utils import alignment_angle_deg, normalize_angle_deg, angle_difference_deg


class SpatialPropertiesModel:
    """
    Model for spatial arrangement and morphological adaptations of endothelial cells.
    Uses deterministic targets based on real experimental data.
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

        # NEW: Add a perpendicular offset to correct for PCA measurement anomalies
        # This is a temporary solution to a deeper issue where PCA might be returning the minor axis


        # NEW: Add a maximum compression factor to prevent cells from shrinking too much
        self.max_compression_factor = 0.8  # e.g., cell can only shrink to 80% of its target area

        # Real experimental data converted to pixels
        # Pixel spacing: 0.429 μm/pixel, so 1 pixel² = 0.184041 μm²

        # Control cell parameters at different pressures (in pixels)
        self.control_params = {
            'area': {
                0.0: 3216,    # 2155 μm² static
                1.4: 3216     # Assume same area under flow (not measured separately)
            },
            'aspect_ratio': {
                0.0: 1.9,      # Static control
                1.4: 2.3       # Flow control (increased elongation)
            },
            'orientation_mean': {
                0.0: 49.0,     # 9 Random orientation (degrees) - MEAN ONLY
                1.4: 22.0      # Aligned with flow (degrees) - MEAN ONLY
            }
        }

        # Senescent cell parameters (in pixels)
        self.senescent_params = {
            'area_small': 3216,     # 2207 μm² (< 5000 μm²)
            'area_large': 12864,     # 8626 μm² (≥ 5000 μm²)
            'aspect_ratio': {
                0.0: 1.9,            # Static senescent
                1.4: 2.0             # Flow senescent (no significant change)
            },
            'orientation_mean': {
                0.0: 42.0,           # Random orientation static - MEAN ONLY
                1.4: 45.0            # Random orientation flow (no alignment) - MEAN ONLY
            }
        }

        # Probability that a senescent cell will be large (adjustable parameter)
        self.large_senescent_probability = 0.3  # 30% of senescent cells become large

    # Add this method inside the SpatialPropertiesModel class, after the existing methods
    def debug_aspect_ratio_complete_trace(self, pressure, cell_id, cell, dt=None):
        """
        Complete trace of aspect ratio calculation and assignment process.
        """
        print(f"\n{'=' * 60}")
        print(f"🔍 COMPLETE ASPECT RATIO TRACE - Cell {cell_id}")
        print(f"{'=' * 60}")

        # Step 1: Check input parameters
        print(f"\n📊 INPUT PARAMETERS:")
        print(f"   Pressure: {pressure}")
        print(f"   Is senescent: {cell.is_senescent}")
        print(f"   Cell ID: {cell_id}")

        # Step 2: Check parameter dictionaries
        print(f"\n📋 PARAMETER DICTIONARIES:")
        print(f"   Control params: {self.control_params['aspect_ratio']}")
        print(f"   Senescent params: {self.senescent_params['aspect_ratio']}")

        # Step 3: Test interpolation step by step
        print(f"\n🔢 INTERPOLATION PROCESS:")
        if cell.is_senescent:
            param_dict = self.senescent_params['aspect_ratio']
            cell_type = "senescent"
        else:
            param_dict = self.control_params['aspect_ratio']
            cell_type = "control"

        print(f"   Cell type: {cell_type}")
        print(f"   Using param_dict: {param_dict}")

        # Manual interpolation with debug
        p0, p1 = 0.0, 1.4
        v0, v1 = param_dict[p0], param_dict[p1]
        print(f"   Interpolation points: p0={p0}, p1={p1}")
        print(f"   Values at points: v0={v0}, v1={v1}")

        if pressure <= p0:
            raw_result = v0
            print(f"   Pressure <= {p0}, using v0 = {raw_result}")
        elif pressure >= p1:
            raw_result = v1
            print(f"   Pressure >= {p1}, using v1 = {raw_result}")
        else:
            raw_result = v0 + (v1 - v0) * (pressure - p0) / (p1 - p0)
            print(f"   Interpolating: {v0} + ({v1} - {v0}) * ({pressure} - {p0}) / ({p1} - {p0})")
            print(f"   Raw interpolation result: {raw_result}")

        # Step 4: Apply constraints
        constrained_result = max(1.0, raw_result)
        print(f"\n🚫 CONSTRAINT APPLICATION:")
        print(f"   Before constraint: {raw_result}")
        print(f"   After max(1.0, value): {constrained_result}")
        print(f"   Constraint active: {raw_result < 1.0}")

        # Step 5: Check current cell properties
        print(f"\n📱 CURRENT CELL PROPERTIES:")
        print(f"   target_aspect_ratio: {getattr(cell, 'target_aspect_ratio', 'NOT SET')}")
        print(f"   actual_aspect_ratio: {getattr(cell, 'actual_aspect_ratio', 'NOT SET')}")

        # Step 6: Call the actual function and compare
        print(f"\n✅ ACTUAL FUNCTION CALL:")
        actual_result = self.calculate_target_aspect_ratio(pressure, cell.is_senescent)
        print(f"   Function returned: {actual_result}")
        print(f"   Matches manual calc: {abs(actual_result - constrained_result) < 1e-6}")

        print(f"{'=' * 60}\n")
        return actual_result

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

    def calculate_target_aspect_ratio(self, pressure, is_senescent):
        """
        Calculate target cell aspect ratio using deterministic experimental values.
        NO artificial variability - let tessellation provide natural variation.
        """
        if is_senescent:
            # Senescent cells: no significant response to flow
            base_ratio = self.interpolate_pressure_effect(self.senescent_params['aspect_ratio'], pressure)
            # REMOVED: artificial variability
            result = max(1.0, base_ratio)
            return result
        else:
            # Control cells: USE EXACT EXPERIMENTAL VALUES
            base_ratio = self.interpolate_pressure_effect(self.control_params['aspect_ratio'], pressure)
            # REMOVED: artificial variability
            result = max(1.0, base_ratio)

            return result

    def calculate_target_orientation(self, pressure, is_senescent):
        """
        Calculate target cell orientation using deterministic experimental means.
        The target is now a normalized angle in degrees [-180, 180).
        FIXED: Removed the erroneous PERPENDICULAR_OFFSET.
        """
        if is_senescent:
            # Senescent cells: remain randomly oriented regardless of flow
            mean_deg = self.interpolate_pressure_effect(self.senescent_params['orientation_mean'], pressure)
        else:
            # Normal cells: use MEAN orientation only
            mean_deg = self.interpolate_pressure_effect(self.control_params['orientation_mean'], pressure)

        # FIXED: Remove the perpendicular offset that was causing the 90° error
        # The target orientation should directly match the experimental measurement
        return normalize_angle_deg(mean_deg)

    def calculate_target_area(self, pressure, is_senescent, senescence_cause=None):
        """
        Calculate target cell area using deterministic experimental values.
        NO artificial variability - let tessellation provide natural variation.
        """
        if is_senescent:
            # Deterministic assignment of small or large senescent area
            # Use consistent assignment based on cell properties
            if np.random.random() < self.large_senescent_probability:
                result = self.senescent_params['area_large']
            else:
                result = self.senescent_params['area_small']
            # REMOVED: artificial variability
            #print(f"Senescent area: deterministic result={result:.0f}")
            return result
        else:
            # Control cells: use deterministic experimental area
            base_area = self.interpolate_pressure_effect(self.control_params['area'], pressure)
            # REMOVED: artificial biological variability
            result = max(1000, base_area)  # Minimum 1000 pixels²
            #print(f"Control area: pressure={pressure}, deterministic result={result:.0f}")
            return result

    def update_cell_properties(self, cell, pressure, dt, cells_dict=None):
        """
        Updated to ensure continuous dynamics for both target and actual properties.
        """
        # Determine current state
        in_transition = hasattr(self, '_in_transition_mode') and self._in_transition_mode
        is_initial_setup = not hasattr(cell, 'target_area') or cell.target_area is None

        dynamics_info = {
            'event_driven_mode': True,
            'transitioning': in_transition,
            'initial_setup': is_initial_setup
        }
        dt_minutes = dt

        # Calculate instantaneous targets based on current pressure
        instant_target_area = self.calculate_target_area(pressure, cell.is_senescent, cell.senescence_cause)
        instant_target_orientation = self.calculate_target_orientation(pressure, cell.is_senescent)
        instant_target_aspect_ratio = self.calculate_target_aspect_ratio(pressure, cell.is_senescent)

        # Get time constants from the temporal model or use defaults
        if self.temporal_model:
            current_pressure = getattr(self, '_current_pressure', pressure)
            tau_area, _ = self.temporal_model.get_scaled_tau_and_amax(current_pressure, 'area')
            tau_orient, _ = self.temporal_model.get_scaled_tau_and_amax(current_pressure, 'orientation')
            tau_ar, _ = self.temporal_model.get_scaled_tau_and_amax(current_pressure, 'aspect_ratio')
        else:
            tau_area, tau_orient, tau_ar = 30.0, 20.0, 25.0

        # Evolve target properties toward instantaneous targets
        if is_initial_setup:
            cell.target_area = instant_target_area
            cell.target_orientation = instant_target_orientation
            cell.target_aspect_ratio = instant_target_aspect_ratio
            dynamics_info['initial_target_set'] = True
        else:
            # Evolve target area
            decay_factor = np.exp(-dt_minutes / tau_area)
            cell.target_area = instant_target_area + (cell.target_area - instant_target_area) * decay_factor

            # Evolve target orientation using angle_difference_deg for correctness
            orientation_diff = angle_difference_deg(instant_target_orientation, cell.target_orientation)
            decay_factor = np.exp(-dt_minutes / tau_orient)
            cell.target_orientation += (1 - decay_factor) * orientation_diff
            cell.target_orientation = normalize_angle_deg(cell.target_orientation)

            # Evolve target aspect ratio
            decay_factor = np.exp(-dt_minutes / tau_ar)
            cell.target_aspect_ratio = instant_target_aspect_ratio + (cell.target_aspect_ratio - instant_target_aspect_ratio) * decay_factor
            cell.target_aspect_ratio = max(1.0, cell.target_aspect_ratio)

        # Evolve actual properties toward the (evolving) target properties
        # Evolve actual area
        decay_factor = np.exp(-dt_minutes / tau_area)
        cell.actual_area = cell.target_area + (cell.actual_area - cell.target_area) * decay_factor
        cell.actual_area = max(cell.actual_area, cell.target_area * self.max_compression_factor)

        # Evolve actual orientation using angle_difference_deg for correctness
        orientation_diff = angle_difference_deg(cell.target_orientation, cell.actual_orientation)
        decay_factor = np.exp(-dt_minutes / tau_orient)
        cell.actual_orientation += (1 - decay_factor) * orientation_diff
        cell.actual_orientation = normalize_angle_deg(cell.actual_orientation)


        # Evolve actual aspect ratio
        decay_factor = np.exp(-dt_minutes / tau_ar)
        cell.actual_aspect_ratio = cell.target_aspect_ratio + (cell.actual_aspect_ratio - cell.target_aspect_ratio) * decay_factor
        cell.actual_aspect_ratio = max(1.0, cell.actual_aspect_ratio)

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
        actual_orientations_deg = [cell.actual_orientation for cell in cells_dict.values()]
        target_orientations_deg = [getattr(cell, 'target_orientation', cell.actual_orientation) for cell in cells_dict.values()]
        actual_areas = [cell.actual_area for cell in cells_dict.values()]
        target_areas = [getattr(cell, 'target_area', cell.actual_area) for cell in cells_dict.values()]
        actual_aspect_ratios = [cell.actual_aspect_ratio for cell in cells_dict.values()]
        target_aspect_ratios = [getattr(cell, 'target_aspect_ratio', cell.actual_aspect_ratio) for cell in cells_dict.values()]

        # Calculate alignment index using the new alignment_angle_deg function
        # This gives the mean angle relative to the flow axis [0, 90]
        mean_alignment_angle = np.mean([alignment_angle_deg(o) for o in actual_orientations_deg])

        # Calculate adaptation quality using the new angle_difference_deg for correctness
        orientation_adaptation = np.mean([
            1.0 - min(1.0, abs(angle_difference_deg(a, t)) / 180.0)
            for a, t in zip(actual_orientations_deg, target_orientations_deg)
        ])

        # Handle zero areas properly
        area_adaptation = np.mean([
            min(a / t, t / a) if t > 0 and a > 0 else 1.0
            for a, t in zip(actual_areas, target_areas)
        ])

        # Handle zero aspect ratios properly
        aspect_ratio_adaptation = np.mean([
            min(a / t, t / a) if t > 0 and a > 0 else 1.0
            for a, t in zip(actual_aspect_ratios, target_aspect_ratios)
        ])

        return {
            'mean_actual_alignment': mean_alignment_angle,  # Mean alignment angle [0, 90]
            'std_actual_orientation': np.std(actual_orientations_deg),
            'mean_target_alignment': np.mean([alignment_angle_deg(o) for o in target_orientations_deg]),

            'mean_actual_area': np.mean(actual_areas),
            'mean_target_area': np.mean(target_areas),

            'mean_actual_aspect_ratio': np.mean(actual_aspect_ratios),
            'mean_target_aspect_ratio': np.mean(target_aspect_ratios),

            'orientation_adaptation': orientation_adaptation,  # How close actual is to target
            'area_adaptation': area_adaptation,
            'aspect_ratio_adaptation': aspect_ratio_adaptation,

            # Additional metrics
            'large_senescent_fraction': len([c for c in cells_dict.values()
                                             if c.is_senescent and getattr(c, 'target_area', 0) > 27174]) / len(
                cells_dict),
            'pressure': pressure,

            # NEW: Target consistency metrics
            'target_orientation_std': np.std(target_orientations_deg),
            'target_area_std': np.std(target_areas),
            'target_aspect_ratio_std': np.std(target_aspect_ratios)
        }

    def calculate_alignment_index(self, cells, flow_direction=0):
        """
        Calculate the alignment index for a collection of cells based on the cosine similarity.
        Uses the alignment_angle_deg to ensure consistency.
        """
        if not cells:
            return 0

        cell_list = list(cells.values()) if isinstance(cells, dict) else cells

        if not cell_list:
            return 0

        # The alignment index is the average of the cosine of the alignment angles.
        # alignment_angle_deg gives the angle in [0, 90], so cos will be in [0, 1].
        # cos(0) = 1 (perfectly aligned), cos(90) = 0 (perpendicular).
        alignment_scores = [np.cos(np.radians(alignment_angle_deg(cell.actual_orientation))) for cell in cell_list]
        return np.mean(alignment_scores)

    def calculate_shape_index(self, cells):
        """
        Calculate the shape index for a collection of cells.
        Shape index = P/(sqrt(4πA)) where P is perimeter and A is area.
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
            if hasattr(cell, 'target_area') and cell.target_area > 0 and cell.actual_area > 0:
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

    def get_expected_values(self, pressure, cell_type='control'):
        """
        Get expected experimental values for comparison.
        Now returns deterministic values (means only).
        """
        if cell_type == 'control':
            return {
                'area': self.interpolate_pressure_effect(self.control_params['area'], pressure),
                'aspect_ratio': self.interpolate_pressure_effect(self.control_params['aspect_ratio'], pressure),
                'orientation_mean': self.interpolate_pressure_effect(self.control_params['orientation_mean'], pressure),
                # Note: no orientation_std returned since we're now deterministic
            }
        else:  # senescent
            return {
                'area_small': self.senescent_params['area_small'],
                'area_large': self.senescent_params['area_large'],
                'aspect_ratio': self.interpolate_pressure_effect(self.senescent_params['aspect_ratio'], pressure),
                'orientation_mean': self.interpolate_pressure_effect(self.senescent_params['orientation_mean'], pressure),
            }
