"""
Temporal dynamics model for endothelial cell adaptation to mechanical stimuli.
"""
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


class TemporalDynamicsModel:
    """
    Model for time-dependent adaptation of endothelial cells to mechanical stimuli.

    This model captures how cellular responses evolve following pressure changes,
    implementing the first-order differential equation described in the thesis.
    """

    def __init__(self, config):
        """
        Initialize the temporal dynamics model with configuration parameters.

        Parameters:
            config: SimulationConfig object with parameter settings
        """
        self.config = config

        # Known pressure values and their corresponding Amax values
        self.P_values = np.array(config.known_pressures)
        self.A_max_map = config.known_A_max

        # Initial response value
        self.y0 = config.initial_response

        # Time constant parameters
        self.tau_base = config.tau_base
        self.lambda_scale = config.lambda_scale

        # Calculate linear model parameters for A_max
        P_known = np.array(list(self.A_max_map.keys()))
        A_max_known = np.array(list(self.A_max_map.values()))
        self.slope, self.intercept = np.polyfit(P_known, A_max_known, 1)

    def calculate_A_max(self, P):
        """
        Calculate the maximum response (steady-state value) for a given pressure.

        Uses a hybrid approach with direct lookup for known pressures and linear
        interpolation for unknown pressures.

        Parameters:
            P: Pressure value (Pa)

        Returns:
            Maximum attainable response at the given pressure
        """
        # For known pressure values, use the original A_max from the map
        if P in self.A_max_map:
            return self.A_max_map[P]
        else:
            # For unknown P values, use linear interpolation/extrapolation
            # Ensure A_max is non-negative (minimum value of 1)
            return max(1, self.slope * P + self.intercept)

    def calculate_tau(self, A_max):
        """
        Calculate the time constant based on A_max.

        Time constant scales with A_max following a power law relationship.

        Parameters:
            A_max: Maximum attainable response

        Returns:
            Time constant (tau) value
        """
        # Reference value is 1.0
        return self.tau_base * (A_max ** self.lambda_scale)

    def model(self, y, t, P):
        """
        First-order differential equation model for cellular response.

        dy/dt = (A_max - y) / tau

        Parameters:
            y: Current response value
            t: Time
            P: Pressure value (Pa)

        Returns:
            Rate of change of response (dy/dt)
        """
        # Calculate A_max for this pressure
        A_max = self.calculate_A_max(P)

        # Calculate tau based on A_max
        tau = self.calculate_tau(A_max)

        # Differential equation: dy/dt = (A_max - y) / tau
        dydt = (A_max - y) / tau

        return dydt

    def simulate(self, P, y0=None, t_span=(0, 8), t_points=100):
        """
        Simulate the cellular response to a constant pressure over time.

        Parameters:
            P: Pressure value (Pa)
            y0: Initial response value, uses default if None
            t_span: Time range (start, end) in arbitrary units
            t_points: Number of time points to evaluate

        Returns:
            t: Time points
            y: Response values at each time point
        """
        # Use default initial value if none provided
        if y0 is None:
            y0 = self.y0

        # Create time points for evaluation
        t = np.linspace(t_span[0], t_span[1], t_points)

        # Solve the ODE
        solution = odeint(self.model, y0, t, args=(P,))

        # Flatten the solution array
        y = solution.flatten()

        return t, y

    def simulate_step_response(self, P_initial, P_final, t_step, y0=None, t_span=(0, 12), t_points=100):
        """
        Simulate the cellular response to a step change in pressure.

        Parameters:
            P_initial: Initial pressure value (Pa)
            P_final: Final pressure value after step (Pa)
            t_step: Time at which the step occurs
            y0: Initial response value, uses default if None
            t_span: Time range (start, end) in arbitrary units
            t_points: Number of time points to evaluate

        Returns:
            t: Time points
            y: Response values at each time point
            P: Pressure at each time point
        """
        # Use default initial value if none provided
        if y0 is None:
            y0 = self.y0

        # Create time points for evaluation
        t = np.linspace(t_span[0], t_span[1], t_points)

        # Initialize arrays for solution and pressure
        y = np.zeros_like(t)
        P = np.zeros_like(t)

        # First phase with P_initial
        t1_indices = t <= t_step
        t1 = t[t1_indices]
        P[t1_indices] = P_initial

        if len(t1) > 0:
            sol1 = odeint(self.model, y0, t1, args=(P_initial,))
            y[t1_indices] = sol1.flatten()

        # Second phase with P_final
        t2_indices = t > t_step
        t2 = t[t2_indices]
        P[t2_indices] = P_final

        if len(t2) > 0:
            # Start second phase from end of first phase
            y0_2 = y[t1_indices][-1] if len(t1) > 0 else y0

            # Adjust time to start from 0 for ODE solver
            t2_adjusted = t2 - t_step

            sol2 = odeint(self.model, y0_2, t2_adjusted, args=(P_final,))
            y[t2_indices] = sol2.flatten()

        return t, y, P

    def simulate_ramp_response(self, P_initial, P_final, t_ramp_start, t_ramp_end,
                               y0=None, t_span=(0, 16), t_points=100):
        """
        Simulate the cellular response to a ramp change in pressure.

        Parameters:
            P_initial: Initial pressure value (Pa)
            P_final: Final pressure value after ramp (Pa)
            t_ramp_start: Time at which the ramp begins
            t_ramp_end: Time at which the ramp ends
            y0: Initial response value, uses default if None
            t_span: Time range (start, end) in arbitrary units
            t_points: Number of time points to evaluate

        Returns:
            t: Time points
            y: Response values at each time point
            P: Pressure at each time point
        """
        # Use default initial value if none provided
        if y0 is None:
            y0 = self.y0

        # Create time points for evaluation
        t = np.linspace(t_span[0], t_span[1], t_points)

        # Initialize array for solution
        y = np.zeros_like(t)

        # Calculate pressure at each time point
        P = np.zeros_like(t)

        # Phase 1: Initial pressure
        mask1 = t <= t_ramp_start
        P[mask1] = P_initial

        # Phase 2: Ramp
        mask2 = (t > t_ramp_start) & (t <= t_ramp_end)
        ramp_duration = t_ramp_end - t_ramp_start
        P[mask2] = P_initial + (P_final - P_initial) * (t[mask2] - t_ramp_start) / ramp_duration

        # Phase 3: Final pressure
        mask3 = t > t_ramp_end
        P[mask3] = P_final

        # Solve the ODE numerically for each small time step
        y[0] = y0

        for i in range(1, len(t)):
            # Use the previous value as initial condition
            y0_i = y[i - 1]

            # Time interval for this step
            t_i = np.array([t[i - 1], t[i]])

            # Average pressure during this interval
            P_avg = (P[i - 1] + P[i]) / 2

            # Solve for this small interval
            sol_i = odeint(self.model, y0_i, t_i, args=(P_avg,))

            # Store the result
            y[i] = sol_i[1]

        return t, y, P

    def update_cell_responses(self, cells, P, dt):
        """
        Update the response values of all cells based on the current pressure.

        Parameters:
            cells: Dictionary of Cell objects
            P: Current pressure value (Pa)
            dt: Time step

        Returns:
            Dictionary mapping cell_id to new response value
        """
        updated_responses = {}

        for cell_id, cell in cells.items():
            # Current response
            y0 = cell.response

            # Calculate parameters
            A_max = self.calculate_A_max(P)
            tau = self.calculate_tau(A_max)

            # Analytical solution for one time step
            y_new = A_max - (A_max - y0) * np.exp(-dt / tau)

            # Store new response
            updated_responses[cell_id] = y_new

            # Update cell response
            cell.update_response(y_new)

        return updated_responses

    def fit_parameters(self, experimental_data, initial_params=None, bounds=None):
        """
        Fit model parameters to experimental data.

        Parameters:
            experimental_data: Dictionary mapping pressure values to time and response data
            initial_params: Dictionary of initial parameter values
            bounds: Dictionary of parameter bounds (min, max)

        Returns:
            optimized_params: Dictionary of optimized parameter values
        """
        if initial_params is None:
            initial_params = {'tau_base': self.tau_base, 'lambda_scale': self.lambda_scale}

        if bounds is None:
            bounds = {'tau_base': (0.1, 5.0), 'lambda_scale': (0.1, 2.0)}

        # Extract parameters and bounds for optimizer
        param_names = ['tau_base', 'lambda_scale']
        initial_values = [initial_params[name] for name in param_names]
        param_bounds = [(bounds[name][0], bounds[name][1]) for name in param_names]

        # Define objective function for minimization
        def objective_function(params):
            # Set model parameters
            tau_base, lambda_scale = params
            tau_base_original = self.tau_base
            lambda_scale_original = self.lambda_scale

            self.tau_base = tau_base
            self.lambda_scale = lambda_scale

            # Calculate error for each pressure condition
            total_error = 0

            for P, data in experimental_data.items():
                # Get experimental data for this pressure
                t_exp = data['t']
                y_exp = data['y']

                # Simulate model response
                _, y_model = self.simulate(P, t_span=(t_exp[0], t_exp[-1]), t_points=len(t_exp))

                # Calculate sum of squared errors
                error = np.sum((y_model - y_exp) ** 2)
                total_error += error

            # Restore original parameters
            self.tau_base = tau_base_original
            self.lambda_scale = lambda_scale_original

            return total_error

        # Run optimization
        result = minimize(
            objective_function,
            initial_values,
            method='L-BFGS-B',
            bounds=param_bounds
        )

        # Extract optimized parameters
        optimized_values = result.x
        optimized_params = {name: value for name, value in zip(param_names, optimized_values)}

        # Update model parameters
        self.tau_base = optimized_params['tau_base']
        self.lambda_scale = optimized_params['lambda_scale']

        # Update config parameters
        self.config.tau_base = self.tau_base
        self.config.lambda_scale = self.lambda_scale

        return optimized_params

    def get_parameters(self):
        """
        Get current model parameters.

        Returns:
            Dictionary of parameter values
        """
        return {
            'tau_base': self.tau_base,
            'lambda_scale': self.lambda_scale,
            'slope': self.slope,
            'intercept': self.intercept
        }