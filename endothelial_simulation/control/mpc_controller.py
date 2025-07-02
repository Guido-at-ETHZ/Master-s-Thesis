import numpy as np
from typing import Dict, Tuple, Optional


class EndothelialMPCController:
    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config
        self.weights = {'tracking': 1.0, 'holes': 1000.0, 'senescence': 3.0}
        self.shear_stress_limits = (0.0, 4.0)
        self.max_rate_change = 0.7
        self.baseline_shear = 1.4
        self.targets = {}

    def set_targets(self, targets: Dict):
        self.targets = targets

    def get_current_state(self) -> Dict:
        cells = self.simulator.grid.cells
        if not cells:
            return {}

        responses = [getattr(cell, 'response', 1.0) for cell in cells.values()]
        senescent_count = sum(1 for cell in cells.values() if cell.is_senescent)
        senescence_fraction = senescent_count / len(cells)

        hole_manager = getattr(self.simulator.grid, 'hole_manager', None)
        hole_count = len(hole_manager.holes) if hole_manager else 0

        return {
            'responses': np.array(responses),
            'senescence_fraction': senescence_fraction,
            'hole_count': hole_count,
            'current_shear': self.simulator.input_pattern.get('value', 0.0)
        }

    def control_step(self, targets: Optional[Dict] = None) -> Tuple[float, Dict]:
        if targets is not None:
            self.set_targets(targets)

        current_state = self.get_current_state()
        if not current_state:
            return self.baseline_shear, {'error': 'No state available'}

        # Emergency check
        if current_state['hole_count'] > 0:
            return 0.0, {'emergency': True, 'reason': 'holes_exist'}

        # Simple control law
        current_shear = current_state.get('current_shear', 0.0)
        target_response = self.targets.get('response', 2.0)

        if len(current_state['responses']) > 0:
            avg_response = np.mean(current_state['responses'])
            error = target_response - avg_response
            control_adjustment = error * 0.3  # Proportional gain
            optimal_shear = current_shear + control_adjustment
        else:
            optimal_shear = self.baseline_shear

        # Apply constraints
        optimal_shear = np.clip(optimal_shear, self.shear_stress_limits[0], self.shear_stress_limits[1])

        # Rate limiting
        rate_change = abs(optimal_shear - current_shear)
        if rate_change > self.max_rate_change:
            if optimal_shear > current_shear:
                optimal_shear = current_shear + self.max_rate_change
            else:
                optimal_shear = current_shear - self.max_rate_change

        return optimal_shear, {
            'optimal_shear': optimal_shear,
            'current_state': current_state,
            'targets': self.targets.copy()
        }