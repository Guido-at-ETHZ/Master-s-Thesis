import numpy as np
from typing import Dict, Tuple, Optional

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.optimize import minimize
import warnings
from ..models.temporal_dynamics import TemporalDynamicsModel
from ..models.population_dynamics import PopulationDynamicsModel

warnings.filterwarnings('ignore')


class EndothelialMPCController:
    """
    Enhanced MPC Controller with soft constraints and predictive capabilities.

    Features:
    - Soft constraint for senescent fraction ≤ 0.30
    - Soft constraint for hole area ≤ 5% of total area
    - Predictive hole prevention
    - Cell density constraints (min/max cells)
    - Rate limitation (0.7 Pa)
    - Spatial boundary constraints
    """

    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config

        # Control parameters
        self.prediction_horizon = 30  # steps
        self.control_horizon = 10  # steps
        self.dt = 1.0  # minute

        # Constraint parameters
        self.senescence_threshold = 0.30  # 30% senescent fraction limit
        self.hole_area_threshold = 0.05  # 5% hole area limit
        self.rate_limit = 0.7  # Pa/min
        self.shear_stress_limits = (0.0, 4.0)  # Pa

        # Soft constraint weights (penalty scaling)
        self.weights = {
            'tracking': 1.0,  # Response tracking
            'senescence': 20.0,  # Soft senescence penalty (was 100.0)
            'holes': 80.0,  # Soft hole penalty (was 1000.0)
            'cell_density': 15.0,  # Cell density penalty (was 50.0)
            'rate_limit': 25.0,  # Rate limit penalty (was 200.0)
            'control_effort': 0.1,  # Control effort penalty
            'hole_prediction': 40.0,  # Predictive hole prevention (was 500.0)
            'flow_alignment': 8.0,  # Flow alignment penalty (was 25.0)
        }

        # Spatial parameters
        self.average_cell_area = 30.0  # pixels^2
        self.average_expansion_factor = 1.1

        # Control targets
        self.targets = {}
        self.baseline_shear = 1.4

        # State history for prediction
        self.state_history = []
        self.max_history_length = 10

        print("🎯 Enhanced MPC Controller initialized with soft constraints")
        print(f"   Senescence threshold: {self.senescence_threshold:.1%}")
        print(f"   Hole area threshold: {self.hole_area_threshold:.1%}")
        print(f"   Rate limit: {self.rate_limit} Pa/min")
        print(f"   Prediction horizon: {self.prediction_horizon} steps")

    def set_targets(self, targets: Dict):
        """Set control targets."""
        self.targets = targets

    def get_current_state(self) -> Dict:
        """Get comprehensive current state including orientations."""
        cells = self.simulator.grid.cells
        if not cells:
            return {}

        # Basic cell properties
        responses = [getattr(cell, 'response', 1.0) for cell in cells.values()]
        senescent_count = sum(1 for cell in cells.values() if cell.is_senescent)
        senescence_fraction = senescent_count / len(cells)

        # NEW: Add orientation data
        orientations = [cell.actual_orientation for cell in cells.values()]
        target_orientation = self.targets.get('orientation', 0.0)

        # Calculate alignment metrics
        alignment_errors = []
        for orientation in orientations:
            aligned_angle = self.simulator.grid.to_alignment_angle(orientation)
            target_aligned = self.simulator.grid.to_alignment_angle(target_orientation)
            alignment_errors.append(abs(aligned_angle - target_aligned))

        # Hole information (keep existing)
        hole_manager = getattr(self.simulator.grid, 'hole_manager', None)
        hole_count = len(hole_manager.holes) if hole_manager else 0

        # Calculate hole area fraction (keep existing)
        total_area = self.simulator.grid.comp_width * self.simulator.grid.comp_height
        hole_area = sum(hole.get_area() for hole in hole_manager.holes.values()) if hole_manager else 0
        hole_area_fraction = hole_area / total_area if total_area > 0 else 0

        # Cell density constraints (keep existing)
        available_area = total_area - hole_area
        minimum_cells = available_area / (self.average_cell_area * self.average_expansion_factor)
        maximum_cells = 1.5 * minimum_cells

        # Current shear stress (keep existing)
        current_shear = self.simulator.input_pattern.get('value', 0.0)

        # Calculate biological status for hole prediction (keep existing)
        unfillable_area = 0
        if hole_manager:
            biological_status = hole_manager.get_biological_status()
            unfillable_area = biological_status.get('unfillable_area', 0)

        state = {
            'responses': np.array(responses),
            'senescence_fraction': senescence_fraction,
            'hole_count': hole_count,
            'hole_area_fraction': hole_area_fraction,
            'current_shear': current_shear,
            'cell_count': len(cells),
            'minimum_cells': minimum_cells,
            'maximum_cells': maximum_cells,
            'total_area': total_area,
            'available_area': available_area,
            'unfillable_area': unfillable_area,
            'time': getattr(self.simulator, 'time', 0.0),
            # NEW: Add orientation data
            'orientations': np.array(orientations),
            'mean_alignment_error': np.mean(alignment_errors) if alignment_errors else 0.0,
            'alignment_variance': np.var(alignment_errors) if alignment_errors else 0.0,
        }

        # Update state history (keep existing)
        self.state_history.append(state)
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)

        return state

    def _extract_senescence_rate(self, current_state: Dict, shear_stress: float) -> float:
        """Extract actual senescence rate from PopulationDynamicsModel."""
        try:
            # Initialize population model
            pop_model = PopulationDynamicsModel(self.config)

            # Set current state from cells
            current_cells = self.simulator.grid.cells
            if not current_cells:
                return 0.0

            # Get current population state
            pop_model.update_from_cells(current_cells, dt=0, tau=shear_stress)
            initial_state = pop_model.state.copy()

            # Predict one time step forward
            dt_hours = self.dt / 60.0  # Convert minutes to hours
            predicted_state = pop_model.update(dt_hours, tau=shear_stress)

            # Calculate senescence rate
            initial_senescent = initial_state['S_tel'] + initial_state['S_stress']
            initial_total = sum(initial_state['E']) + initial_senescent

            predicted_senescent = predicted_state['S_tel'] + predicted_state['S_stress']
            predicted_total = sum(predicted_state['E']) + predicted_senescent

            if initial_total > 0 and predicted_total > 0:
                initial_fraction = initial_senescent / initial_total
                predicted_fraction = predicted_senescent / predicted_total
                senescence_rate = (predicted_fraction - initial_fraction) / self.dt
            else:
                senescence_rate = 0.0

            return max(0.0, senescence_rate)  # Ensure non-negative

        except Exception as e:
            print(f"⚠️ Senescence rate extraction failed: {e}")
            # Fallback to simple model
            return 0.001 * max(0, shear_stress - 2.0)

    def _extract_hole_dynamics(self, current_state: Dict, senescence_fraction: float) -> Dict:
        """Extract hole formation probability from BiologicalHoleManager."""
        try:
            hole_manager = self.simulator.grid.hole_manager
            if not hole_manager:
                return {'creation_prob': 0.0, 'filling_prob': 0.0}

            # Use actual biological decision logic
            unfillable_area = hole_manager._calculate_unfillable_area()

            # Deterministic hole creation
            if unfillable_area > 0:
                creation_probability = 1.0
            elif senescence_fraction >= hole_manager.senescence_threshold:
                # Use actual probabilistic model
                base_prob = hole_manager.hole_creation_probability_base
                creation_probability = base_prob * (1.0 + 2.0 * senescence_fraction)
                creation_probability = min(0.8, creation_probability)
            else:
                creation_probability = 0.0

            # Hole filling probability
            if unfillable_area <= 0 and senescence_fraction < hole_manager.senescence_threshold:
                filling_probability = 0.3
            else:
                filling_probability = 0.0

            return {
                'creation_prob': creation_probability,
                'filling_prob': filling_probability,
                'unfillable_area': unfillable_area
            }
        except Exception as e:
            print(f"⚠️ Hole dynamics extraction failed: {e}")
            # Fallback to simple model
            hole_risk = max(0, senescence_fraction - self.senescence_threshold) * 0.1
            return {'creation_prob': hole_risk, 'filling_prob': 0.0, 'unfillable_area': 0}

    def _extract_orientation_dynamics(self, current_state: Dict, shear_stress: float) -> np.array:
        """Extract orientation dynamics from TemporalDynamicsModel."""
        try:
            current_orientations = current_state.get('orientations', [])
            if len(current_orientations) == 0:
                return np.array([])

            # Initialize temporal model
            temporal_model = TemporalDynamicsModel(self.config)

            # Get time constant for orientation
            tau_orient, A_max = temporal_model.get_scaled_tau_and_amax(shear_stress, 'orientation')

            # Target orientation
            target_orientation = self.targets.get('orientation', 0.0)

            # Apply first-order dynamics
            predicted_orientations = []
            for current_orientation in current_orientations:
                # Calculate orientation difference (handle angle wrapping)
                orientation_diff = target_orientation - current_orientation
                while orientation_diff > np.pi:
                    orientation_diff -= 2 * np.pi
                while orientation_diff < -np.pi:
                    orientation_diff += 2 * np.pi

                # Apply dy/dt = (target - current) / tau
                decay_factor = np.exp(-self.dt / tau_orient)
                new_orientation = target_orientation - orientation_diff * decay_factor
                predicted_orientations.append(new_orientation)

            return np.array(predicted_orientations)

        except Exception as e:
            print(f"⚠️ Orientation dynamics extraction failed: {e}")
            # Fallback: no change
            return current_state.get('orientations', np.array([]))

    def predict_future_state(self, current_state: Dict, control_sequence: List[float]) -> List[Dict]:
        """Enhanced prediction using extracted dynamics from existing models."""
        predictions = []
        state = current_state.copy()

        for i, u in enumerate(control_sequence):
            # 1. EXTRACT ACTUAL SENESCENCE DYNAMICS
            senescence_rate = self._extract_senescence_rate(state, u)
            new_senescence = min(1.0, state['senescence_fraction'] + senescence_rate * self.dt)

            # 2. EXTRACT ACTUAL HOLE FORMATION DYNAMICS
            hole_dynamics = self._extract_hole_dynamics(state, new_senescence)

            # Predict hole changes
            current_hole_count = state['hole_count']
            expected_new_holes = hole_dynamics['creation_prob'] * self.dt
            expected_filled_holes = hole_dynamics['filling_prob'] * current_hole_count * self.dt

            new_hole_count = max(0, current_hole_count + expected_new_holes - expected_filled_holes)
            new_hole_count = min(new_hole_count, getattr(self.simulator.grid.hole_manager, 'max_holes', 5))

            # Estimate hole area
            avg_hole_area = state['hole_area_fraction'] / current_hole_count if current_hole_count > 0 else 0.01
            new_hole_area = new_hole_count * avg_hole_area

            # 3. EXTRACT ACTUAL FLOW ALIGNMENT DYNAMICS
            predicted_orientations = self._extract_orientation_dynamics(state, u)

            # 4. EXTRACT RESPONSE DYNAMICS (existing temporal model)
            temporal_model = TemporalDynamicsModel(self.config)
            current_responses = state.get('responses', [])

            if len(current_responses) > 0:
                A_max = temporal_model.calculate_A_max(u)
                tau = temporal_model.calculate_tau(A_max)

                # Apply dy/dt = (A_max - y) / tau for each cell
                new_responses = []
                for response in current_responses:
                    # Analytical solution
                    decay_factor = np.exp(-self.dt / tau)
                    new_response = A_max - (A_max - response) * decay_factor
                    new_responses.append(new_response)
            else:
                new_responses = []

            # Calculate alignment metrics for predicted orientations
            target_orientation = self.targets.get('orientation', 0.0)
            alignment_errors = []
            if len(predicted_orientations) > 0:
                for orientation in predicted_orientations:
                    aligned_angle = self.simulator.grid.to_alignment_angle(orientation)
                    target_aligned = self.simulator.grid.to_alignment_angle(target_orientation)
                    alignment_errors.append(abs(aligned_angle - target_aligned))

            # 5. UPDATE PREDICTED STATE
            predicted_state = state.copy()
            predicted_state.update({
                'senescence_fraction': new_senescence,
                'hole_count': new_hole_count,
                'hole_area_fraction': new_hole_area,
                'orientations': predicted_orientations,
                'mean_alignment_error': np.mean(alignment_errors) if alignment_errors else 0.0,
                'responses': np.array(new_responses),
                'current_shear': u,
                'time': state['time'] + (i + 1) * self.dt,
            })

            predictions.append(predicted_state)
            state = predicted_state

        return predictions

    def calculate_cost(self, control_sequence: List[float], current_state: Dict) -> float:
        """Calculate total cost function with soft constraints."""
        total_cost = 0.0

        # Predict future states
        predictions = self.predict_future_state(current_state, control_sequence)

        for i, (u, predicted_state) in enumerate(zip(control_sequence, predictions)):
            step_cost = 0.0

            # 1. TRACKING COST
            target_response = self.targets.get('response', 2.0)
            if len(predicted_state['responses']) > 0:
                response_error = target_response - np.mean(predicted_state['responses'])
                step_cost += self.weights['tracking'] * response_error ** 2

            # FLOW ALIGNMENT COST
            target_orientation = self.targets.get('orientation', 0.0)
            if 'mean_alignment_error' in predicted_state:
                alignment_error = predicted_state['mean_alignment_error']
                step_cost += self.weights['flow_alignment'] * alignment_error ** 2

            # 2. SENESCENCE SOFT CONSTRAINT
            senescence_violation = max(0, predicted_state['senescence_fraction'] - self.senescence_threshold)
            if senescence_violation > 0:
                # Exponential penalty for severe violations
                penalty = np.exp(senescence_violation * 10) - 1
                step_cost += self.weights['senescence'] * penalty

            # 3. HOLE AREA SOFT CONSTRAINT (5% threshold)
            hole_violation = max(0, predicted_state['hole_area_fraction'] - self.hole_area_threshold)
            if hole_violation > 0:
                # Quadratic penalty with scaling
                penalty = (hole_violation / self.hole_area_threshold) ** 2
                step_cost += self.weights['holes'] * penalty

            # 4. PREDICTIVE HOLE PREVENTION
            if predicted_state['unfillable_area'] > 0:
                # Penalize conditions that lead to unfillable area
                penalty = predicted_state['unfillable_area'] / predicted_state['total_area']
                step_cost += self.weights['hole_prediction'] * penalty ** 2

            # 5. CELL DENSITY CONSTRAINTS
            cell_count = predicted_state['cell_count']
            min_cells = predicted_state['minimum_cells']
            max_cells = predicted_state['maximum_cells']

            if cell_count < min_cells:
                violation = (min_cells - cell_count) / min_cells
                step_cost += self.weights['cell_density'] * violation ** 2
            elif cell_count > max_cells:
                violation = (cell_count - max_cells) / max_cells
                step_cost += self.weights['cell_density'] * violation ** 2

            # 6. CONTROL EFFORT
            step_cost += self.weights['control_effort'] * u ** 2

            # 7. RATE LIMIT CONSTRAINT
            if i == 0:  # Only check first control action
                rate_change = abs(u - current_state['current_shear'])
                if rate_change > self.rate_limit:
                    violation = rate_change - self.rate_limit
                    step_cost += self.weights['rate_limit'] * violation ** 2

            # Weight by prediction horizon (recent predictions more important)
            time_weight = 0.9 ** i
            total_cost += time_weight * step_cost

        return total_cost

    def optimize_control(self, current_state: Dict) -> Tuple[float, Dict]:
        """Optimize control action using pure MPC approach."""

        # REMOVED: All emergency-related code

        # Set up optimization problem
        def objective(control_sequence):
            return self.calculate_cost(control_sequence, current_state)

        # Initial guess (maintain current control)
        x0 = [current_state['current_shear']] * self.control_horizon

        # Bounds for control variables
        bounds = [(self.shear_stress_limits[0], self.shear_stress_limits[1])] * self.control_horizon

        # Rate limit constraints
        constraints = []
        for i in range(self.control_horizon):
            if i == 0:
                # First control action rate limit
                def rate_constraint(x, i=i):
                    return self.rate_limit - abs(x[i] - current_state['current_shear'])

                constraints.append({'type': 'ineq', 'fun': rate_constraint})
            else:
                # Subsequent control actions rate limit
                def rate_constraint(x, i=i):
                    return self.rate_limit - abs(x[i] - x[i - 1])

                constraints.append({'type': 'ineq', 'fun': rate_constraint})

        # Solve optimization
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-6}
            )

            if result.success:
                optimal_control = result.x[0]
                cost = result.fun
            else:
                optimal_control = self._fallback_control(current_state)
                cost = float('inf')

        except Exception as e:
            print(f"⚠️ Optimization failed: {e}")
            optimal_control = self._fallback_control(current_state)
            cost = float('inf')

        # CLEAN RETURN - No emergency fields
        return optimal_control, {
            'optimal_shear': optimal_control,
            'cost': cost,
            'current_state': current_state
        }

    def _fallback_control(self, current_state: Dict) -> float:
        """Fallback control law when optimization fails."""
        current_shear = current_state['current_shear']
        target_response = self.targets.get('response', 2.0)

        if len(current_state['responses']) > 0:
            avg_response = np.mean(current_state['responses'])
            error = target_response - avg_response

            # Simple proportional control with constraint awareness
            kp = 0.3
            control_adjustment = kp * error

            # Reduce control if approaching soft constraints
            if current_state['senescence_fraction'] > 0.25:  # 25% - warning level
                control_adjustment *= 0.5

            if current_state['hole_area_fraction'] > 0.03:  # 3% - warning level
                control_adjustment *= 0.3

            optimal_shear = current_shear + control_adjustment
        else:
            optimal_shear = self.baseline_shear

        # Apply hard constraints
        optimal_shear = np.clip(optimal_shear, self.shear_stress_limits[0], self.shear_stress_limits[1])

        # Rate limiting
        rate_change = abs(optimal_shear - current_shear)
        if rate_change > self.rate_limit:
            if optimal_shear > current_shear:
                optimal_shear = current_shear + self.rate_limit
            else:
                optimal_shear = current_shear - self.rate_limit

        return optimal_shear

    def control_step(self, targets: Optional[Dict] = None) -> Tuple[float, Dict]:
        """Main control step function."""
        if targets is not None:
            self.set_targets(targets)

        # Get current state
        current_state = self.get_current_state()
        if not current_state:
            return self.baseline_shear, {'error': 'No state available'}

        # Optimize control
        optimal_shear, control_info = self.optimize_control(current_state)

        # Add constraint status to info
        control_info.update({
            'constraints': {
                'senescence_fraction': current_state['senescence_fraction'],
                'senescence_violation': max(0, current_state['senescence_fraction'] - self.senescence_threshold),
                'hole_area_fraction': current_state['hole_area_fraction'],
                'hole_violation': max(0, current_state['hole_area_fraction'] - self.hole_area_threshold),
                'cell_count': current_state['cell_count'],
                'cell_density_ok': (current_state['minimum_cells'] <= current_state['cell_count'] <= current_state[
                    'maximum_cells']),
                'rate_limit_ok': abs(optimal_shear - current_state['current_shear']) <= self.rate_limit
            },
            'targets': self.targets.copy()
        })

        return optimal_shear, control_info

    def get_constraint_status(self) -> Dict:
        """Get detailed constraint status for monitoring."""
        if not self.state_history:
            return {}

        current_state = self.state_history[-1]

        return {
            'senescence': {
                'current': current_state['senescence_fraction'],
                'threshold': self.senescence_threshold,
                'violation': max(0, current_state['senescence_fraction'] - self.senescence_threshold),
                'status': 'OK' if current_state['senescence_fraction'] <= self.senescence_threshold else 'VIOLATED'
            },
            'hole_area': {
                'current': current_state['hole_area_fraction'],
                'threshold': self.hole_area_threshold,
                'violation': max(0, current_state['hole_area_fraction'] - self.hole_area_threshold),
                'status': 'OK' if current_state['hole_area_fraction'] <= self.hole_area_threshold else 'VIOLATED'
            },
            'cell_density': {
                'current': current_state['cell_count'],
                'minimum': current_state['minimum_cells'],
                'maximum': current_state['maximum_cells'],
                'status': 'OK' if current_state['minimum_cells'] <= current_state['cell_count'] <= current_state[
                    'maximum_cells'] else 'VIOLATED'
            },
            'rate_limit': {
                'limit': self.rate_limit,
                'status': 'OK'  # Checked dynamically during control
            }
        }

