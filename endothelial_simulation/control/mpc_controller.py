import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional


class EndothelialMPCController:
    """
    Model Predictive Controller for endothelial cell mechanotransduction system.

    Features:
    - Short prediction horizon (20 min) for biological accuracy
    - High update frequency (1 min) for fast response
    - Hole prevention as NO-GO constraint
    - Initial ramp handling (0→1.4 Pa without rate constraint)
    """

    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config

        # MPC Parameters - optimized for biology
        self.prediction_horizon = 20  # minutes
        self.control_horizon = 15  # minutes
        self.sampling_time = 1  # minutes
        self.N_pred = 20  # prediction steps
        self.N_ctrl = 15  # control steps

        # Weights - hole prevention is NO-GO
        self.weights = {
            'tracking': 1.0,
            'holes': 1000.0,  # Barrier function
            'senescence': 3.0
        }

        # Control constraints
        self.shear_stress_limits = (0.0, 4.0)
        self.max_rate_change = 0.7
        self.baseline_shear = 1.4

        # Initial ramp handling
        self.initial_ramp_completed = False
        self.initial_target = 1.4

        # Simplified model parameters
        self.tau_fast = 30.0
        self.targets = {}

    def set_targets(self, targets: Dict):
        """Set target values for optimization."""
        self.targets = targets

    def get_current_state(self) -> Dict:
        """Extract current system state."""
        cells = self.simulator.grid.cells
        if not cells:
            return {}

        # Extract key metrics
        responses = [getattr(cell, 'response', 1.0) for cell in cells.values()]
        senescent_count = sum(1 for cell in cells.values() if cell.is_senescent)
        senescence_fraction = senescent_count / len(cells)

        # Hole status
        hole_manager = getattr(self.simulator.grid, 'hole_manager', None)
        if hole_manager:
            hole_count = len(hole_manager.holes)
            unfillable_area = hole_manager._calculate_unfillable_area()
        else:
            hole_count = 0
            unfillable_area = 0.0

        return {
            'responses': np.array(responses),
            'senescence_fraction': senescence_fraction,
            'hole_count': hole_count,
            'unfillable_area': unfillable_area,
            'current_shear': self.simulator.input_pattern.get('value', 0.0)
        }

    def predict_system(self, current_state: Dict, control_sequence: np.ndarray) -> list:
        """Fast prediction using simplified biological model."""
        predictions = []

        # Initial conditions
        response = np.mean(current_state['responses']) if len(current_state['responses']) > 0 else 1.0
        senescence = current_state['senescence_fraction']

        for k in range(self.N_pred):
            shear = control_sequence[k] if k < len(control_sequence) else control_sequence[-1]

            # Response dynamics: exponential approach to target
            target_response = 1.0 + shear * 1.0  # Simplified gain
            response += self.sampling_time / self.tau_fast * (target_response - response)

            # Senescence progression
            if shear > 2.0:  # High stress
                senescence += 0.0001 * self.sampling_time
            senescence = min(1.0, senescence)

            # Hole risk
            hole_count = current_state['hole_count']
            if senescence > 0.30 and hole_count == 0:
                hole_count = 1

            unfillable_area = max(0, 100 * (senescence - 0.30))

            predictions.append({
                'responses': np.array([response]),
                'senescence_fraction': senescence,
                'hole_count': hole_count,
                'unfillable_area': unfillable_area
            })

        return predictions

    def calculate_objective(self, control_sequence: np.ndarray, current_state: Dict) -> float:
        """Calculate MPC objective function."""
        predictions = self.predict_system(current_state, control_sequence)
        total_cost = 0.0

        for pred_state in predictions:
            # Hole NO-GO cost
            hole_cost = self._hole_barrier_cost(pred_state)

            # Target tracking cost
            tracking_cost = 0.0
            if 'response' in self.targets and len(pred_state['responses']) > 0:
                error = (np.mean(pred_state['responses']) - self.targets['response']) ** 2
                tracking_cost = error

            # Senescence cost
            senescence_cost = max(0, pred_state['senescence_fraction'] - 0.8) ** 2

            total_cost += (self.weights['tracking'] * tracking_cost +
                           self.weights['holes'] * hole_cost +
                           self.weights['senescence'] * senescence_cost)

        return total_cost

    def _hole_barrier_cost(self, state: Dict) -> float:
        """Barrier function for hole prevention."""
        if state['hole_count'] > 0:
            return 1e6  # Massive penalty

        if state['unfillable_area'] > 0:
            return 1e4 * (state['unfillable_area'] / 100.0) ** 2

        # Approaching 30% threshold
        if state['senescence_fraction'] > 0.25:
            distance = 0.30 - state['senescence_fraction']
            if distance > 0:
                return min(1000.0 / distance, 1e4)
            else:
                return 1e4

        return 0.0

    def optimize_control(self, current_state: Dict) -> Tuple[np.ndarray, Dict]:
        """Optimize control sequence."""
        current_shear = current_state.get('current_shear', 0.0)

        # Initial guess with ramp handling
        if not self.initial_ramp_completed and current_shear < self.initial_target * 0.95:
            initial_guess = np.full(self.N_ctrl, self.initial_target)
        else:
            self.initial_ramp_completed = True
            initial_guess = np.full(self.N_ctrl, current_shear)

        # Bounds
        bounds = [(self.shear_stress_limits[0], self.shear_stress_limits[1])
                  for _ in range(self.N_ctrl)]

        # Rate constraints (after initial ramp)
        constraints = []
        if self.initial_ramp_completed:
            for i in range(1, self.N_ctrl):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda u, i=i: self.max_rate_change - abs(u[i] - u[i - 1])
                })
            constraints.append({
                'type': 'ineq',
                'fun': lambda u: self.max_rate_change - abs(u[0] - current_shear)
            })

        # Emergency constraint
        def emergency_constraint(u):
            if current_state['senescence_fraction'] > 0.28 and np.max(u) > 1.0:
                return -1.0
            return 1.0

        constraints.append({'type': 'ineq', 'fun': emergency_constraint})

        # Optimize
        try:
            result = minimize(
                fun=lambda u: self.calculate_objective(u, current_state),
                x0=initial_guess,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': 50, 'ftol': 1e-4}
            )

            if not result.success:
                result.x = np.full(self.N_ctrl, max(0.0, current_shear - 0.1))

        except Exception as e:
            result = type('obj', (object,), {
                'x': np.full(self.N_ctrl, max(0.0, current_shear - 0.2)),
                'success': False,
                'message': f"Emergency fallback: {e}"
            })

        return result.x, {
            'success': result.success,
            'cost': getattr(result, 'fun', 0),
            'message': getattr(result, 'message', 'OK')
        }

    def control_step(self, targets: Optional[Dict] = None) -> Tuple[float, Dict]:
        """Execute one MPC control step."""
        if targets is not None:
            self.set_targets(targets)

        current_state = self.get_current_state()
        if not current_state:
            return self.baseline_shear, {'error': 'No state available'}

        # Emergency check
        if current_state['hole_count'] > 0:
            return 0.0, {
                'emergency': True,
                'reason': 'holes_exist',
                'hole_count': current_state['hole_count']
            }

        # Optimize
        optimal_sequence, opt_info = self.optimize_control(current_state)
        optimal_shear = optimal_sequence[0]

        return optimal_shear, {
            'optimal_shear': optimal_shear,
            'current_state': current_state,
            'optimization': opt_info,
            'targets': self.targets.copy()
        }


# Usage example
def run_endothelial_mpc_experiment(simulator):
    """Run 6-hour experiment with MPC controller."""

    # Create controller
    mpc = EndothelialMPCController(simulator, simulator.config)

    # Set targets
    mpc.set_targets({
        'response': 2.0,
        'area': 11712,
        'aspect_ratio': 2.3,
        'orientation': 0.35
    })

    # Run experiment (6 hours = 360 minutes)
    results = []
    for minute in range(360):
        optimal_shear, control_info = mpc.control_step()

        # Check emergency
        if control_info.get('emergency', False):
            print(f"🛑 EXPERIMENT TERMINATED: {control_info['reason']}")
            break

        # Apply control
        simulator.set_constant_input(optimal_shear)
        simulator.step(dt=1.0)

        # Record every 5 minutes
        if minute % 5 == 0:
            state = control_info['current_state']
            results.append({
                'time': minute,
                'shear_stress': optimal_shear,
                'cell_count': len(simulator.grid.cells),
                'hole_count': state['hole_count'],
                'senescence_fraction': state['senescence_fraction']
            })

        # Progress every 30 minutes
        if minute % 30 == 0:
            state = control_info['current_state']
            print(f"T={minute}min: Shear={optimal_shear:.2f}Pa, "
                  f"Sen={state['senescence_fraction']:.1%}")

    return results, mpc