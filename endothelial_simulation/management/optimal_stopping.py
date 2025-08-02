import numpy as np
from collections import deque

class OptimalStopping:
    """
    Determines the optimal time to stop the simulation based on cell quality metrics.
    """
    def __init__(self, config, simulator, mpc_controller):
        self.config = config.stopping_criteria
        self.simulator = simulator
        self.mpc_controller = mpc_controller
        self.enabled = config.enable_optimal_stopping

        # State tracking
        self.response_history = deque(maxlen=self.config['response_stability_window'])
        self.quality_history = deque(maxlen=self.config['response_stability_window'])
        self.senescence_history = []

        if self.enabled:
            print("✅ Optimal Stopping System enabled.")

    def _calculate_cell_quality(self, state):
        """
        Calculates a composite cell quality score.
        """
        # Factors influencing quality (all normalized between 0 and 1)
        senescence_penalty = state.get('senescence_fraction', 0)
        hole_penalty = state.get('hole_area_fraction', 0)
        
        # Ideal response is the target, deviation is bad
        target_response = self.mpc_controller.targets.get('response', 2.0)
        response_error = abs(np.mean(state.get('responses', 0)) - target_response) / target_response

        # Quality is 1 minus penalties
        quality = 1.0 - (senescence_penalty + hole_penalty + response_error)
        return max(0, quality)

    def check_criteria(self, current_time, current_state):
        """
        Check if the simulation should be stopped based on the defined criteria.
        """
        if not self.enabled or current_time < self.config['min_simulation_time']:
            return None

        # --- Update State ---
        self.response_history.append(np.mean(current_state.get('responses', 0)))
        self.senescence_history.append(current_state.get('senescence_fraction', 0))
        
        quality = self._calculate_cell_quality(current_state)
        self.quality_history.append(quality)

        # --- Evaluate Criteria ---
        # 1. Senescence limit
        if current_state['senescence_fraction'] > self.config['max_senescence']:
            return f"Senescence limit exceeded: {current_state['senescence_fraction']:.2%} > {self.config['max_senescence']:.2%}"

        # 2. Response stability
        if len(self.response_history) == self.config['response_stability_window']:
            response_std_dev = np.std(self.response_history)
            if response_std_dev < self.config['response_stability_threshold']:
                return f"Response has stabilized with std dev: {response_std_dev:.4f} < {self.config['response_stability_threshold']:.4f}"

        # 3. Peak quality detection
        if len(self.quality_history) == self.config['response_stability_window']:
            # Check if quality has started to decline
            current_quality = np.mean(list(self.quality_history)[-5:])
            past_quality = np.mean(list(self.quality_history)[:5])
            if current_quality < past_quality:
                return f"Cell quality is declining. Peak quality likely passed."

        return None