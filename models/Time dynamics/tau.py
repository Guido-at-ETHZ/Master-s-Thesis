import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- 1. Experimental Data ---
# Shear stress values
S_values = np.array([15, 25, 45])
# Corresponding experimental max amplitudes (steady state targets)
A_max_experimental = {15: 1.4, 25: 3.4, 45: 5.7}

# Extracted time-course data points (approximated from image)
# List of tuples: (S, time_points, y_values)
experimental_data = [
    (15, np.array([1, 2, 6]), np.array([1.4, 1.0, 1.1])),
    (25, np.array([1, 3, 6]), np.array([3.6, 3.0, 2.9])),
    (45, np.array([1, 2, 6]), np.array([3.0, 3.5, 5.3]))
]


# --- 2. Model Definition ---

def tau_model(S, y, k, p, b):
    """Calculates time constant tau based on S, y, and parameters."""
    # Prevent division by zero or negative y if model assumes y>=0
    y = np.maximum(y, 0)
    # Prevent S=0 issues if applicable, though our S values are positive
    S = np.maximum(S, 1e-6)
    # Ensure g(y) term doesn't go to zero or negative if b is large negative
    g_y = 1 + b * y
    # Add safeguards if g_y could become <= 0 depending on range of b and y
    g_y = np.maximum(g_y, 1e-6)  # Prevent tau becoming zero or negative

    # Calculate f(S) = k / S^p
    f_S = k / (S ** p)

    return f_S * g_y


def ode_system(t, y, S, A_max_target, k, p, b):
    """Defines the differential equation dy/dt."""
    # Ensure y doesn't exceed A_max (numerical stability)
    y = np.minimum(y, A_max_target - 1e-9)
    y = np.maximum(y, 0)  # Ensure y >= 0

    current_tau = tau_model(S, y, k, p, b)

    # Check for very small tau which could cause instability
    if current_tau < 1e-8:
        # Handle potential singularity, e.g., return a large slope
        # or adjust logic. For now, just cap the rate.
        print(f"Warning: Very small tau ({current_tau}) at S={S}, y={y}")
        # return large number or handle appropriately
        return (A_max_target - y) / 1e-8

    dydt = (A_max_target - y) / current_tau
    return dydt


# --- 3. Simulation and Optimization ---

def simulate_response(S, A_max_target, t_eval, k, p, b):
    """Solves the ODE for a given S and parameters."""
    y0 = [1]  # Initial condition: y(0) = 1
    t_span = [0, max(t_eval)]

    sol = solve_ivp(
        ode_system,
        t_span,
        y0,
        args=(S, A_max_target, k, p, b),
        t_eval=t_eval,  # Evaluate solution at these specific times
        method='RK45'  # Common ODE solver
    )

    # Return interpolated results at t_eval points
    # solve_ivp might not return exactly t_eval if integration steps differ
    # We need results AT the experimental time points.
    # Re-check if sol.t matches t_eval. If using dense_output=True might be better.
    # For simplicity assuming sol.y[0] corresponds to t_eval after t=0

    # Need to return values corresponding to t_eval requested
    # If solve_ivp guarantees output at t_eval, sol.y[0] is fine.
    # Otherwise, need interpolation: sol.sol(t_eval)[0] if dense_output=True used

    # Assuming solve_ivp hit the t_eval points:
    # Need to handle the case where t_eval includes t=0
    if 0 in t_eval:
        y_simulated = np.interp(t_eval, sol.t, sol.y[0])
        # Or if dense output: y_simulated = sol.sol(t_eval)[0]
    else:
        # Insert t=0 for interpolation if needed
        t_eval_with_zero = np.insert(t_eval, 0, 0)
        y_simulated_with_zero = np.interp(t_eval_with_zero, sol.t, sol.y[0])
        y_simulated = y_simulated_with_zero[1:]  # Remove the y(0) value

    # Ensure simulated y doesn't exceed A_max physically
    y_simulated = np.minimum(y_simulated, A_max_target)

    return y_simulated


def objective_function(params, data, A_max_map):
    """Calculates the error between model and experimental data."""
    k, p, b = params
    total_error = 0

    # Basic parameter bounds check
    if k <= 0 or p < 0:  # k should be > 0, p usually >= 0
        return np.inf  # Return large error if parameters are invalid

    for S, t_exp, y_exp in data:
        A_max_target = A_max_map[S]

        # We need to evaluate the model at the *experimental* time points
        t_eval = t_exp  # Time points where we have experimental y values

        try:
            y_model = simulate_response(S, A_max_target, t_eval, k, p, b)

            # Calculate squared error for this condition
            error = np.sum((y_model - y_exp) ** 2)
            total_error += error
        except Exception as e:
            # Penalize if simulation fails for these parameters
            print(f"Simulation failed for params {params}, S={S}: {e}")
            return np.inf

    # print(f"Params: {params}, Error: {total_error}") # Debugging print
    return total_error


# --- 4. Run Optimization ---

# Initial guess for parameters [k, p, b]
# Finding good initial guesses can be crucial and might require trial/error
# Guess: k relates to overall time scale, p to S-dependence, b to y-dependence
initial_guess = [10.0, 1.0, 0.1]  # Example: Adjust based on expected scales

# Parameter bounds (optional but recommended)
# k > 0, p >= 0, b might be positive or negative? User said tau increases with y -> b > 0?
# Bounds: [(k_min, k_max), (p_min, p_max), (b_min, b_max)]
bounds = [(1e-3, None), (0, None), (0, None)]  # Assuming b >= 0

# Perform optimization
result = minimize(
    objective_function,
    initial_guess,
    args=(experimental_data, A_max_experimental),
    method='L-BFGS-B',  # Method that handles bounds
    bounds=bounds
)

if result.success:
    optimal_params = result.x
    print(f"Optimization Successful!")
    print(f"Optimal Parameters (k, p, b): {optimal_params}")
    print(f"Minimum Error (Sum of Squares): {result.fun}")

    # --- 5. Plot Results ---
    plt.figure(figsize=(10, 6))

    t_plot = np.linspace(0, 6, 100)  # Time points for smooth curve plotting

    for S, t_exp, y_exp in experimental_data:
        A_max_target = A_max_experimental[S]

        # Plot experimental data points
        plt.scatter(t_exp, y_exp, label=f'S={S} (Experimental)')

        # Plot optimized model simulation
        y_model_plot = simulate_response(S, A_max_target, t_plot, *optimal_params)
        plt.plot(t_plot, y_model_plot, label=f'S={S} (Model Fit)')

    plt.xlabel("Duration of Shear Stress (Hours)")
    plt.ylabel("Normalized Level of B-Actin mRNA")
    plt.title("Model Fit vs Experimental Data")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()

else:
    print(f"Optimization failed: {result.message}")