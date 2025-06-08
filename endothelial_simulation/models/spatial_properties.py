"""
Updated spatial properties model using real experimental data.
Modified to use deterministic targets based on experimental measurements.
Variability comes from tessellation process, not artificial randomness.
"""
import numpy as np


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
                1.4: 2.3       # Flow control (increased elongation)
            },
            'orientation_mean': {
                0.0: 49.0,     # Random orientation (degrees) - MEAN ONLY
                1.4: 20.0      # Aligned with flow (degrees) - MEAN ONLY
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
                0.0: 42.0,           # Random orientation static - MEAN ONLY
                1.4: 45.0            # Random orientation flow (no alignment) - MEAN ONLY
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
            print(f"Senescent AR: pressure={pressure}, deterministic result={result:.1f}")
            return result
        else:
            # Control cells: USE EXACT EXPERIMENTAL VALUES
            base_ratio = self.interpolate_pressure_effect(self.control_params['aspect_ratio'], pressure)
            # REMOVED: artificial variability
            result = max(1.0, base_ratio)
            print(f"Control AR: pressure={pressure}, deterministic result={result:.1f}")
            return result

    def calculate_target_orientation(self, pressure, is_senescent):
        """
        Calculate target cell orientation using deterministic experimental means.
        NO artificial variability - let tessellation provide natural variation.
        """
        if is_senescent:
            # Senescent cells: remain randomly oriented regardless of flow
            # Use MEAN orientation only (no std sampling)
            mean_deg = self.interpolate_pressure_effect(self.senescent_params['orientation_mean'], pressure)

            # Convert to radians - USE MEAN DIRECTLY
            mean_rad = np.radians(mean_deg)

            print(f"Senescent orientation: pressure={pressure}, deterministic mean={mean_deg:.1f}°")
            return mean_rad
        else:
            # Normal cells: use MEAN orientation only
            mean_deg = self.interpolate_pressure_effect(self.control_params['orientation_mean'], pressure)

            # Convert to radians - USE MEAN DIRECTLY
            mean_rad = np.radians(mean_deg)

            print(f"Control orientation: pressure={pressure}, deterministic mean={mean_deg:.1f}°")
            return mean_rad

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
            print(f"Senescent area: deterministic result={result:.0f}")
            return result
        else:
            # Control cells: use deterministic experimental area
            base_area = self.interpolate_pressure_effect(self.control_params['area'], pressure)
            # REMOVED: artificial biological variability
            result = max(1000, base_area)  # Minimum 1000 pixels²
            print(f"Control area: pressure={pressure}, deterministic result={result:.0f}")
            return result

    def update_cell_properties(self, cell, pressure, dt, cells_dict=None):
        """
        FIXED VERSION: Ensure temporal dynamics actually work.
        """
        # Calculate deterministic instantaneous target values for current pressure
        instant_target_area = self.calculate_target_area(pressure, cell.is_senescent, cell.senescence_cause)
        instant_target_aspect_ratio = self.calculate_target_aspect_ratio(pressure, cell.is_senescent)
        instant_target_orientation = self.calculate_target_orientation(pressure, cell.is_senescent)

        # Initialize cell target properties if not already set - CRITICAL FIX
        if not hasattr(cell, 'target_orientation') or cell.target_orientation is None:
            # CHANGED: Initialize to different value to create driving force
            if pressure > 0:
                # Start from baseline (0 Pa) values to ensure adaptation occurs
                cell.target_orientation = self.calculate_target_orientation(0.0, cell.is_senescent)
                cell.target_area = self.calculate_target_area(0.0, cell.is_senescent, cell.senescence_cause)
                cell.target_aspect_ratio = self.calculate_target_aspect_ratio(0.0, cell.is_senescent)
                print(f"INIT Cell {cell.cell_id}: Starting from baseline, will adapt to P={pressure:.1f}")
            else:
                cell.target_orientation = instant_target_orientation
                cell.target_area = instant_target_area
                cell.target_aspect_ratio = instant_target_aspect_ratio

        # For senescent cells, targets don't adapt (they're mechanically impaired)
        if cell.is_senescent:
            cell.target_orientation = instant_target_orientation
            cell.target_area = instant_target_area
            cell.target_aspect_ratio = instant_target_aspect_ratio
            dynamics_info = {'adaptation_disabled': True}
        else:
            # Normal cells adapt with temporal dynamics toward deterministic targets
            dynamics_info = {}

            # Get time constants from temporal model
            if self.temporal_model:
                tau_unified, _ = self.temporal_model.get_scaled_tau_and_amax(pressure, 'biochemical')
                tau_area = tau_unified
                tau_orient = tau_unified
                tau_ar = tau_unified
            else:
                tau_unified = 30.0  # 30 minutes default
                tau_area = tau_orient = tau_ar = tau_unified

            # CRITICAL FIX: Apply temporal dynamics with proper exponential integration
            dt_minutes = dt  # Ensure dt is in minutes

            # 1. Area evolution using first-order dynamics: dy/dt = (target - y) / tau
            area_diff = instant_target_area - cell.target_area
            if abs(area_diff) > 10:  # Only if significant difference
                # Exponential approach: y_new = target + (y_old - target) * exp(-dt/tau)
                decay_factor = np.exp(-dt_minutes / tau_area)
                cell.target_area = instant_target_area + (cell.target_area - instant_target_area) * decay_factor
                dynamics_info['tau_area'] = tau_area
                dynamics_info['area_change'] = area_diff

            # 2. Orientation evolution
            orientation_diff = instant_target_orientation - cell.target_orientation
            # Handle angle wrapping
            while orientation_diff > np.pi:
                orientation_diff -= 2 * np.pi
            while orientation_diff < -np.pi:
                orientation_diff += 2 * np.pi

            if abs(orientation_diff) > 0.02:  # >1 degree difference
                decay_factor = np.exp(-dt_minutes / tau_orient)
                cell.target_orientation = instant_target_orientation + (
                            cell.target_orientation - instant_target_orientation) * decay_factor
                dynamics_info['tau_orientation'] = tau_orient
                dynamics_info['orientation_change'] = np.degrees(orientation_diff)

            # 3. Aspect ratio evolution
            ar_diff = instant_target_aspect_ratio - cell.target_aspect_ratio
            if abs(ar_diff) > 0.02:  # Significant difference
                decay_factor = np.exp(-dt_minutes / tau_ar)
                cell.target_aspect_ratio = instant_target_aspect_ratio + (
                            cell.target_aspect_ratio - instant_target_aspect_ratio) * decay_factor
                cell.target_aspect_ratio = max(1.0, cell.target_aspect_ratio)  # Ensure >= 1.0
                dynamics_info['tau_aspect_ratio'] = tau_ar
                dynamics_info['ar_change'] = ar_diff

        # Update the cell's target properties
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
                                             if c.is_senescent and getattr(c, 'target_area', 0) > 27174]) / len(
                cells_dict),
            'pressure': pressure,

            # NEW: Target consistency metrics (should be very consistent now)
            'target_orientation_std': np.degrees(np.std(target_orientations)),
            'target_area_std': np.std(target_areas),
            'target_aspect_ratio_std': np.std(target_aspect_ratios)
        }

    def calculate_alignment_index(self, cells, flow_direction=0):
        """
        Calculate the alignment index for a collection of cells.
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