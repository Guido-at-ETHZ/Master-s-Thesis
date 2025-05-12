import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def fit_parameters_unweighted():
    """Perform parameter fitting based on experimental data"""
    # Experimental data points for maximum β-actin mRNA levels
    experimental_tau = np.array([15, 25, 45])
    experimental_A_max = np.array([1.5, 3.7, 5.3])

    # Fixed parameters
    tau_threshold = 10  # dyn/cm^2

    # Define the A_max function (as per the paper)
    def A_max(tau, a, b):
        """Calculate maximum actin expression based on parameters a and b"""
        if np.isscalar(tau):
            if tau <= tau_threshold:
                return 1.0
            else:
                return 1.0 + a * (tau - tau_threshold) ** b
        else:  # Handle array input
            result = np.ones_like(tau, dtype=float)
            mask = tau > tau_threshold
            result[mask] = 1.0 + a * (tau[mask] - tau_threshold) ** b
            return result

    # Error function for optimization
    def error_function(params):
        """Calculate sum of squared errors between model and experimental data"""
        a, b = params
        predicted = np.array([A_max(tau, a, b) for tau in experimental_tau])
        return np.sum((predicted - experimental_A_max) ** 2)

    # Grid search to find initial parameters
    print("Performing grid search to find initial parameters...")
    a_values = np.linspace(0.01, 1.0, 50)
    b_values = np.linspace(0.01, 2.0, 50)

    best_a, best_b = 0, 0
    min_error = float('inf')

    for a in a_values:
        for b in b_values:
            error = error_function([a, b])
            if error < min_error:
                min_error = error
                best_a, best_b = a, b

    print(f"Grid search results: a = {best_a:.4f}, b = {best_b:.4f}, Error = {min_error:.6f}")

    # Refine with optimization
    print("\nRefining parameters with optimization...")
    initial_guess = [best_a, best_b]
    result = minimize(error_function, initial_guess, method='Nelder-Mead')

    opt_a, opt_b = result.x
    print(f"Optimized parameters: a = {opt_a:.4f}, b = {opt_b:.4f}, Error = {result.fun:.6f}")

    # Compare model predictions with experimental data
    print("\nComparison of model predictions with experimental data:")
    print("τ (dyn/cm²) | Experimental | Optimized  | Error (%)")
    print("------------|--------------|------------|-----------")

    for tau, exp_val in zip(experimental_tau, experimental_A_max):
        opt_pred = A_max(tau, opt_a, opt_b)
        opt_error = abs(opt_pred - exp_val) / exp_val * 100
        print(f"{tau:12.1f} | {exp_val:12.2f} | {opt_pred:10.2f} | {opt_error:9.2f}")

    # Plot the fitted curve
    plot_fitted_curve(experimental_tau, experimental_A_max, opt_a, opt_b, tau_threshold, A_max)

    # Return the optimized parameters
    return {
        'a': opt_a,
        'b': opt_b,
        'tau_threshold': tau_threshold
    }


def fit_parameters():
    """Perform parameter fitting based on experimental data with weighted optimization"""
    # Experimental data points for maximum β-actin mRNA levels
    experimental_tau = np.array([15, 25, 45])
    experimental_A_max = np.array([1.5, 3.7, 5.3])

    # Define weights (higher values give more importance)
    # Give more weight to values at ~15 and 25 dyn/cm²
    weights = np.array([3.0, 3.0, 1.0])  # Higher weights for first two data points

    # Fixed parameters
    tau_threshold = 10  # dyn/cm^2

    # Define the A_max function (as per the paper)
    def A_max(tau, a, b):
        """Calculate maximum actin expression based on parameters a and b"""
        if np.isscalar(tau):
            if tau <= tau_threshold:
                return 1.0
            else:
                return 1.0 + a * (tau - tau_threshold) ** b
        else:  # Handle array input
            result = np.ones_like(tau, dtype=float)
            mask = tau > tau_threshold
            result[mask] = 1.0 + a * (tau[mask] - tau_threshold) ** b
            return result

    # Weighted error function for optimization
    def weighted_error_function(params):
        """Calculate weighted sum of squared errors between model and experimental data"""
        a, b = params
        predicted = np.array([A_max(tau, a, b) for tau in experimental_tau])
        weighted_errors = weights * ((predicted - experimental_A_max) ** 2)
        return np.sum(weighted_errors)

    # Grid search to find initial parameters
    print("Performing grid search to find initial parameters with weighted optimization...")
    a_values = np.linspace(0.01, 1.0, 50)
    b_values = np.linspace(0.01, 2.0, 50)

    best_a, best_b = 0, 0
    min_error = float('inf')

    for a in a_values:
        for b in b_values:
            error = weighted_error_function([a, b])
            if error < min_error:
                min_error = error
                best_a, best_b = a, b

    print(f"Grid search results: a = {best_a:.4f}, b = {best_b:.4f}, Weighted Error = {min_error:.6f}")

    # Refine with optimization
    print("\nRefining parameters with weighted optimization...")
    initial_guess = [best_a, best_b]
    result = minimize(weighted_error_function, initial_guess, method='Nelder-Mead')

    opt_a, opt_b = result.x
    print(f"Optimized parameters: a = {opt_a:.4f}, b = {opt_b:.4f}, Weighted Error = {result.fun:.6f}")

    # Compare model predictions with experimental data
    print("\nComparison of model predictions with experimental data:")
    print("τ (dyn/cm²) | Experimental | Optimized  | Error (%) | Weight")
    print("------------|--------------|------------|-----------|-------")

    for tau, exp_val, w in zip(experimental_tau, experimental_A_max, weights):
        opt_pred = A_max(tau, opt_a, opt_b)
        opt_error = abs(opt_pred - exp_val) / exp_val * 100
        print(f"{tau:12.1f} | {exp_val:12.2f} | {opt_pred:10.2f} | {opt_error:9.2f} | {w:5.1f}")

    # Plot the fitted curve with weights visualization
    plot_weighted_fitted_curve(experimental_tau, experimental_A_max, weights, opt_a, opt_b, tau_threshold, A_max)

    # Return the optimized parameters
    return {
        'a': opt_a,
        'b': opt_b,
        'tau_threshold': tau_threshold
    }


def plot_weighted_fitted_curve(exp_tau, exp_A_max, weights, a, b, tau_threshold, A_max_func):
    """Plot the fitted curve along with experimental data, visualizing weights"""
    plt.figure(figsize=(12, 8))

    # Generate curve
    tau_range = np.linspace(tau_threshold, 50, 100)
    A_max_values = [A_max_func(tau, a, b) for tau in tau_range]

    # Plot curve
    plt.plot(tau_range, A_max_values, 'b-', label=f'Weighted Fitted Model (a={a:.4f}, b={b:.4f})')

    # Plot experimental data with size proportional to weight
    for tau, A_max_val, weight in zip(exp_tau, exp_A_max, weights):
        size = 100 * weight  # Scale marker size by weight
        plt.scatter(tau, A_max_val, color='red', marker='o', s=size,
                    label=f'Data point τ={tau} (weight={weight})')

    # Plot settings
    plt.axvline(x=tau_threshold, color='r', linestyle='--',
                label=f'Threshold = {tau_threshold} dyn/cm²')
    plt.xlabel('Shear Stress (dyn/cm²)', fontsize=12)
    plt.ylabel('Maximum β-actin mRNA Level', fontsize=12)
    plt.title('Weighted Fitted Model: A_max(τ) = 1 + a·(τ-τ_threshold)^b', fontsize=14)

    # Create custom legend without duplicate entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid(True, alpha=0.3)
    plt.savefig('weighted_fitted_model.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_fitted_curve(exp_tau, exp_A_max, a, b, tau_threshold, A_max_func):
    """Plot the fitted curve along with experimental data"""
    plt.figure(figsize=(12, 8))

    # Generate curve
    tau_range = np.linspace(tau_threshold, 50, 100)
    A_max_values = [A_max_func(tau, a, b) for tau in tau_range]

    # Plot curve and experimental data
    plt.plot(tau_range, A_max_values, 'b-', label=f'Fitted Model (a={a:.4f}, b={b:.4f})')
    plt.scatter(exp_tau, exp_A_max, color='red', marker='o', s=100, label='Experimental Data')

    # Plot settings
    plt.axvline(x=tau_threshold, color='r', linestyle='--',
                label=f'Threshold = {tau_threshold} dyn/cm²')
    plt.xlabel('Shear Stress (dyn/cm²)', fontsize=12)
    plt.ylabel('Maximum β-actin mRNA Level', fontsize=12)
    plt.title('Fitted Model: A_max(τ) = 1 + a·(τ-τ_threshold)^b', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fitted_model.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # First, perform parameter fitting
    print("Step 1: Performing parameter fitting...")
    opt_params = fit_parameters()

    # Set up model parameters using the optimized values
    params = {
        'k0': 0.5,  # h^-1
        'alpha': 0.4,  # dimensionless
        'beta': 0.5,  # dimensionless
        'gamma': 0.3,  # dimensionless
        'tau_threshold': opt_params['tau_threshold'],  # dyn/cm^2
        'tau_ref': 25,  # dyn/cm^2
        'a': opt_params['a'],  # (dyn/cm^2)^-b
        'b': opt_params['b'],  # dimensionless
    }

    print("\nStep 2: Using optimized parameters for the full model...")
    print(f"params = {params}")

    # Generate all plots
    plot_constant_shear_response(params)
    plot_changing_shear_response(params)
    plot_max_actin_vs_shear(params)
    plot_rate_constant(params)
    plot_initial_conditions(params)
    plot_experimental_comparison(params)
    plot_phase_portrait(params)

    print("\nAll plots have been generated:")
    print("1. fitted_model.png - Shows fitted A_max curve with experimental data")
    print("2. constant_shear_response.png - Shows response to different constant shear stress levels")
    print("3. changing_shear_response.png - Shows response to changes in shear stress")
    print("4. max_actin_vs_shear.png - Shows maximum actin level as a function of shear stress")
    print("5. rate_constant.png - Shows rate constant as a function of shear stress and current actin level")
    print("6. initial_conditions.png - Shows time course for different initial conditions")
    print("7. experimental_comparison.png - Compares model predictions with experimental data")
    print("8. phase_portrait.png - Shows phase portrait with rate of change vs. actin level")


def heaviside(x):
    """Heaviside step function"""
    return np.where(x >= 0, 1, 0)


def A_max(tau, params):
    """Calculate the maximum actin expression level for a given shear stress"""
    tau_threshold = params['tau_threshold']
    a = params['a']
    b = params['b']

    # Apply heaviside first to avoid negative power issues
    delta_tau = np.maximum(0, tau - tau_threshold)
    return 1 + a * (delta_tau) ** b * heaviside(tau - tau_threshold)


def rate_constant(tau, A, A_max_val, params):
    """Calculate the rate constant based on shear stress and current actin level"""
    k0 = params['k0']
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    tau_threshold = params['tau_threshold']
    tau_ref = params['tau_ref']

    # Avoid division by zero
    if A_max_val == 0:
        normalized_diff = 0
    else:
        normalized_diff = abs(A_max_val - A) / A_max_val

    # Use np.maximum to avoid negative values before raising to power
    delta_tau = np.maximum(0, tau - tau_threshold)
    shear_factor = 1 + alpha * (delta_tau / tau_ref) ** beta
    actin_factor = gamma + (1 - gamma) * normalized_diff

    return k0 * shear_factor * actin_factor


def actin_derivative(t, A, tau, params):
    """Differential equation for actin mRNA dynamics"""
    A_max_val = A_max(tau, params)
    k = rate_constant(tau, A, A_max_val, params)
    return k * (A_max_val - A)


def simulate_constant_shear(tau, t_max=10, A0=1, params=None):
    """Simulate actin response to constant shear stress"""
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 100)

    def derivative(t, A):
        return actin_derivative(t, A, tau, params)

    # Use float for initial condition to avoid complex number issues
    sol = solve_ivp(derivative, t_span, [float(A0)], t_eval=t_eval, method='RK45')

    return sol.t, sol.y[0]


def simulate_changing_shear(tau1, tau2, t_switch, t_max=20, A0=1, params=None):
    """Simulate actin response to a change in shear stress at t_switch"""
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 200)

    def derivative(t, A):
        tau = tau1 if t < t_switch else tau2
        return actin_derivative(t, A, tau, params)

    # Use float for initial condition
    sol = solve_ivp(derivative, t_span, [float(A0)], t_eval=t_eval, method='RK45')

    return sol.t, sol.y[0]


def plot_constant_shear_response(params):
    """Plot response to different constant shear stress levels"""
    plt.figure(figsize=(12, 8))

    shear_stresses = [0, 5, 10, 15, 25, 45]
    colors = ['gray', 'blue', 'green', 'red', 'purple', 'orange']

    for tau, color in zip(shear_stresses, colors):
        t, A = simulate_constant_shear(tau, t_max=10, params=params)
        plt.plot(t, A, color=color, label=f'τ = {tau} dyn/cm²')

    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Normalized β-actin mRNA Level', fontsize=12)
    plt.title('β-actin mRNA Response to Different Constant Shear Stress Levels', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add annotation showing the threshold
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    plt.annotate(f'Threshold: {params["tau_threshold"]} dyn/cm²',
                 xy=(5, 1.1), xytext=(6, 1.5),
                 arrowprops=dict(arrowstyle='->'))

    plt.savefig('constant_shear_response.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_changing_shear_response(params):
    """Plot response to changes in shear stress"""
    plt.figure(figsize=(12, 8))

    # Case 1: Increase in shear stress
    t1, A1 = simulate_changing_shear(0, 25, 5, t_max=20, params=params)
    plt.plot(t1, A1, 'b-', label='0 → 25 dyn/cm²')

    # Case 2: Decrease in shear stress
    t2, A2 = simulate_changing_shear(25, 0, 5, t_max=20, params=params)
    plt.plot(t2, A2, 'r-', label='25 → 0 dyn/cm²')

    # Case 3: Partial reduction
    t3, A3 = simulate_changing_shear(25, 12.5, 5, t_max=20, params=params)
    plt.plot(t3, A3, 'g-', label='25 → 12.5 dyn/cm²')

    plt.axvline(x=5, color='k', linestyle='--', alpha=0.5)
    plt.annotate('Stress Change', xy=(5, 1), xytext=(5.5, 1),
                 arrowprops=dict(arrowstyle='->'))

    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Normalized β-actin mRNA Level', fontsize=12)
    plt.title('β-actin mRNA Response to Changes in Shear Stress', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('changing_shear_response.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_max_actin_vs_shear(params):
    """Plot maximum actin level as a function of shear stress"""
    tau_values = np.linspace(0, 50, 100)
    A_max_values = [A_max(tau, params) for tau in tau_values]

    plt.figure(figsize=(10, 6))
    plt.plot(tau_values, A_max_values, 'b-')
    plt.axvline(x=params['tau_threshold'], color='r', linestyle='--',
                label=f'Threshold = {params["tau_threshold"]} dyn/cm²')

    # Add experimental data points
    exp_tau = [15, 25, 45]
    exp_A_max = [1.5, 3.7, 5.3]
    plt.scatter(exp_tau, exp_A_max, color='red', marker='o', s=100,
                label='Experimental Data')

    plt.xlabel('Shear Stress (dyn/cm²)', fontsize=12)
    plt.ylabel('Maximum β-actin mRNA Level', fontsize=12)
    plt.title('Steady-State Maximum β-actin mRNA Level vs. Shear Stress', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('max_actin_vs_shear.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_rate_constant(params):
    """Plot rate constant as a function of shear stress and current actin level"""
    tau_values = np.linspace(0, 50, 10)
    A_values = np.linspace(0, 6, 10)

    tau_grid, A_grid = np.meshgrid(tau_values, A_values)
    k_grid = np.zeros_like(tau_grid)

    for i in range(len(tau_values)):
        for j in range(len(A_values)):
            tau = tau_grid[j, i]
            A = A_grid[j, i]
            A_max_val = A_max(tau, params)
            k_grid[j, i] = rate_constant(tau, A, A_max_val, params)

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(tau_grid, A_grid, k_grid, 20, cmap='viridis')
    plt.colorbar(contour, label='Rate Constant k (h⁻¹)')

    plt.xlabel('Shear Stress (dyn/cm²)', fontsize=12)
    plt.ylabel('Current β-actin mRNA Level', fontsize=12)
    plt.title('Rate Constant as a Function of Shear Stress and Actin Level', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('rate_constant.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_initial_conditions(params):
    """Plot time course for different initial conditions"""
    plt.figure(figsize=(12, 8))

    tau = 25  # Fixed shear stress
    initial_conditions = [0.5, 1.0, 2.0, 3.0]
    colors = ['blue', 'green', 'red', 'purple']

    for A0, color in zip(initial_conditions, colors):
        t, A = simulate_constant_shear(tau, t_max=10, A0=A0, params=params)
        plt.plot(t, A, color=color, label=f'A₀ = {A0}')

    plt.axhline(y=A_max(tau, params), color='k', linestyle='--',
                label=f'Aₘₐₓ for τ = {tau} dyn/cm²')

    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Normalized β-actin mRNA Level', fontsize=12)
    plt.title(f'β-actin mRNA Response with Different Initial Conditions (τ = {tau} dyn/cm²)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('initial_conditions.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_experimental_comparison(params):
    """Plot comparison with experimental data"""
    plt.figure(figsize=(12, 8))

    # Experimental times from Image 6
    exp_times = [0, 1, 3, 6]

    # Experimental data points (approximate values from Image 6)
    exp_15 = [1.0, 1.4, 0.8, 1.0]  # 15 dyn/cm²
    exp_25 = [1.0, 3.7, 3.0, 2.8]  # 25 dyn/cm²
    exp_45 = [1.0, 3.0, 3.4, None]  # 45 dyn/cm² (missing data at 6h)

    # Plot experimental data
    plt.scatter(exp_times[:len(exp_15)], exp_15, marker='s', color='gray', s=100, label='Exp: 15 dyn/cm²')
    plt.scatter(exp_times[:len(exp_25)], exp_25, marker='o', color='blue', s=100, label='Exp: 25 dyn/cm²')
    plt.scatter(exp_times[:3], exp_45[:3], marker='^', color='orange', s=100, label='Exp: 45 dyn/cm²')

    # Model predictions
    for tau, color, style in zip([15, 25, 45], ['gray', 'blue', 'orange'], ['-', '-', '-']):
        t, A = simulate_constant_shear(tau, t_max=6, params=params)
        plt.plot(t, A, color=color, linestyle=style, label=f'Model: {tau} dyn/cm²')

    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Normalized β-actin mRNA Level', fontsize=12)
    plt.title('Comparison of Model Predictions with Experimental Data', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('experimental_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_phase_portrait(params):
    """Plot phase portrait showing rate of change vs. actin level"""
    plt.figure(figsize=(12, 8))

    # Define a range of actin levels and shear stresses
    A_values = np.linspace(0, 6, 20)
    tau_values = [0, 10, 15, 25, 45]
    colors = ['gray', 'green', 'red', 'purple', 'orange']

    for tau, color in zip(tau_values, colors):
        derivatives = []
        for A in A_values:
            dA = actin_derivative(0, A, tau, params)
            derivatives.append(dA)

        # Plot the phase portrait
        plt.plot(A_values, derivatives, color=color, label=f'τ = {tau} dyn/cm²')

        # Mark the equilibrium point where dA/dt = 0
        A_eq = A_max(tau, params)
        plt.plot(A_eq, 0, 'o', color=color, markersize=8)

    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('β-actin mRNA Level (A)', fontsize=12)
    plt.ylabel('Rate of Change (dA/dt)', fontsize=12)
    plt.title('Phase Portrait: Rate of Change vs. Actin Level for Different Shear Stresses', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('phase_portrait.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()