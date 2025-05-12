import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def main():
    # Experimental data points for maximum β-actin mRNA levels
    # Converting from Pa to dyn/cm²: 1 Pa = 10 dyn/cm²
    experimental_tau = np.array([15, 25, 45])  # dyn/cm²
    experimental_A_max = np.array([1.5, 3.7, 5.3])

    # Fixed parameters with revised threshold
    tau_threshold = 5  # dyn/cm²
    threshold_value = 1  # Revised initial value before response

    # Define weights for optimization (prioritize lower shear stress values)
    # Higher weight means more importance during optimization
    weights = np.array([5.0, 3.0, 1.0])  # More weight on 15 and 25 dyn/cm²

    print("Using weighted optimization with weights:")
    print(f"τ = 15 dyn/cm²: weight = {weights[0]}")
    print(f"τ = 25 dyn/cm²: weight = {weights[1]}")
    print(f"τ = 45 dyn/cm²: weight = {weights[2]}")

    # Define the A_max function (as per the paper, but with revised threshold)
    def A_max(tau, a, b):
        """Calculate maximum actin expression based on parameters a and b"""
        if np.isscalar(tau):
            if tau <= tau_threshold:
                return threshold_value
            else:
                return threshold_value + a * (tau - tau_threshold) ** b
        else:  # Handle array input
            result = np.ones_like(tau, dtype=float) * threshold_value
            mask = tau > tau_threshold
            result[mask] = threshold_value + a * (tau[mask] - tau_threshold) ** b
            return result

    # Error function for standard optimization
    def error_function_standard(params):
        """Calculate sum of squared errors between model and experimental data"""
        a, b = params
        predicted = np.array([A_max(tau, a, b) for tau in experimental_tau])
        return np.sum((predicted - experimental_A_max) ** 2)

    # Error function for weighted optimization
    def error_function_weighted(params):
        """Calculate weighted sum of squared errors"""
        a, b = params
        predicted = np.array([A_max(tau, a, b) for tau in experimental_tau])
        weighted_errors = weights * ((predicted - experimental_A_max) ** 2)
        return np.sum(weighted_errors)

    # Grid search with weighted errors
    print("\nPerforming grid search with weighted errors...")
    a_values = np.linspace(0.01, 1.0, 50)
    b_values = np.linspace(0.01, 2.0, 50)

    best_a, best_b = 0, 0
    min_error = float('inf')

    for a in a_values:
        for b in b_values:
            error = error_function_weighted([a, b])
            if error < min_error:
                min_error = error
                best_a, best_b = a, b

    print(f"Grid search results: a = {best_a:.4f}, b = {best_b:.4f}, Weighted Error = {min_error:.6f}")

    # Refine with optimization using weighted errors
    print("\nRefining parameters with weighted optimization...")
    initial_guess = [best_a, best_b]
    result = minimize(error_function_weighted, initial_guess, method='Nelder-Mead')

    opt_a, opt_b = result.x
    print(f"Optimized parameters: a = {opt_a:.4f}, b = {opt_b:.4f}, Weighted Error = {result.fun:.6f}")

    # Compare model predictions with experimental data
    print("\nComparison of model predictions with experimental data:")
    print("τ (dyn/cm²) | Experimental | Weighted Opt | Error (%) | Weight")
    print("------------|--------------|--------------|-----------|-------")

    for i, (tau, exp_val) in enumerate(zip(experimental_tau, experimental_A_max)):
        opt_pred = A_max(tau, opt_a, opt_b)
        opt_error = abs(opt_pred - exp_val) / exp_val * 100
        print(f"{tau:12.1f} | {exp_val:12.2f} | {opt_pred:12.2f} | {opt_error:9.2f} | {weights[i]:6.1f}")

    # For comparison, also run standard (unweighted) optimization
    print("\nRunning standard (unweighted) optimization for comparison...")
    result_std = minimize(error_function_standard, initial_guess, method='Nelder-Mead')
    std_a, std_b = result_std.x

    print(f"Standard optimization parameters: a = {std_a:.4f}, b = {std_b:.4f}, Error = {result_std.fun:.6f}")

    print("\nComparison of weighted vs. standard optimization:")
    print("τ (dyn/cm²) | Experimental | Weighted Opt | Standard Opt | Weighted Error (%) | Standard Error (%)")
    print("------------|--------------|--------------|--------------|-------------------|------------------")

    for tau, exp_val in zip(experimental_tau, experimental_A_max):
        weighted_pred = A_max(tau, opt_a, opt_b)
        standard_pred = A_max(tau, std_a, std_b)
        weighted_error = abs(weighted_pred - exp_val) / exp_val * 100
        standard_error = abs(standard_pred - exp_val) / exp_val * 100
        print(f"{tau:12.1f} | {exp_val:12.2f} | {weighted_pred:12.2f} | {standard_pred:12.2f} | {weighted_error:19.2f} | {standard_error:18.2f}")

    # Plot the fitted curves
    plot_comparison(experimental_tau, experimental_A_max, opt_a, opt_b, std_a, std_b, tau_threshold, threshold_value, weights)

    # Return the optimized parameters for use in the main simulation
    return {
        'weighted': {
            'a': opt_a,
            'b': opt_b,
            'tau_threshold': tau_threshold,
            'threshold_value': threshold_value
        },
        'standard': {
            'a': std_a,
            'b': std_b,
            'tau_threshold': tau_threshold,
            'threshold_value': threshold_value
        }
    }

def plot_comparison(exp_tau, exp_A_max, w_a, w_b, s_a, s_b, tau_threshold, threshold_value, weights):
    """Plot comparison of weighted vs. standard optimization"""
    plt.figure(figsize=(12, 8))

    # Define A_max functions
    def weighted_A_max(tau):
        if np.isscalar(tau):
            if tau <= tau_threshold:
                return threshold_value
            else:
                return threshold_value + w_a * (tau - tau_threshold) ** w_b
        else:
            result = np.ones_like(tau, dtype=float) * threshold_value
            mask = tau > tau_threshold
            result[mask] = threshold_value + w_a * (tau[mask] - tau_threshold) ** w_b
            return result

    def standard_A_max(tau):
        if np.isscalar(tau):
            if tau <= tau_threshold:
                return threshold_value
            else:
                return threshold_value + s_a * (tau - tau_threshold) ** s_b
        else:
            result = np.ones_like(tau, dtype=float) * threshold_value
            mask = tau > tau_threshold
            result[mask] = threshold_value + s_a * (tau[mask] - tau_threshold) ** s_b
            return result

    # Generate curves
    tau_range = np.linspace(0, 50, 100)
    weighted_values = [weighted_A_max(tau) for tau in tau_range]
    standard_values = [standard_A_max(tau) for tau in tau_range]

    # Plot curves
    plt.plot(tau_range, weighted_values, 'b-',
             label=f'Weighted Optimization (a={w_a:.4f}, b={w_b:.4f})')
    plt.plot(tau_range, standard_values, 'r--',
             label=f'Standard Optimization (a={s_a:.4f}, b={s_b:.4f})')

    # Plot experimental data with size proportional to weight
    sizes = weights * 50  # Scale weights for marker size
    plt.scatter(exp_tau, exp_A_max, c='black', s=sizes, alpha=0.7,
                label='Experimental Data (size ∝ weight)')

    # Add value annotations
    for i, (tau, exp) in enumerate(zip(exp_tau, exp_A_max)):
        plt.annotate(f"{exp:.1f}", xy=(tau, exp), xytext=(tau+0.5, exp+0.2),
                    fontsize=10, color='black')

        # Add weight annotation
        plt.annotate(f"w={weights[i]:.1f}", xy=(tau, exp), xytext=(tau+0.5, exp-0.3),
                    fontsize=8, color='blue')

    # Add model prediction markers and vertical error lines
    weighted_preds = [weighted_A_max(tau) for tau in exp_tau]
    standard_preds = [standard_A_max(tau) for tau in exp_tau]

    plt.scatter(exp_tau, weighted_preds, color='blue', marker='+', s=100,
                label='Weighted Model Predictions')
    plt.scatter(exp_tau, standard_preds, color='red', marker='x', s=100,
                label='Standard Model Predictions')

    for i, tau in enumerate(exp_tau):
        # Vertical lines showing error for weighted model
        plt.plot([tau, tau], [exp_A_max[i], weighted_preds[i]], 'b:', alpha=0.7)
        # Vertical lines showing error for standard model
        plt.plot([tau, tau], [exp_A_max[i], standard_preds[i]], 'r:', alpha=0.7)

    # Plot settings
    plt.axvline(x=tau_threshold, color='k', linestyle='--', alpha=0.5,
                label=f'Threshold = {tau_threshold} dyn/cm²')
    plt.axhline(y=threshold_value, color='g', linestyle='--', alpha=0.5,
                label=f'Baseline = {threshold_value}')
    plt.xlabel('Shear Stress (dyn/cm²)', fontsize=12)
    plt.ylabel('Maximum β-actin mRNA Level', fontsize=12)
    plt.title('Comparison of Weighted vs. Standard Optimization', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('weighted_vs_standard.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a bar chart comparing the errors
    plt.figure(figsize=(10, 6))
    labels = [f'τ = {tau}' for tau in exp_tau]
    x = np.arange(len(labels))
    width = 0.35

    weighted_errors = [abs(weighted_preds[i] - exp_A_max[i]) / exp_A_max[i] * 100
                      for i in range(len(exp_tau))]
    standard_errors = [abs(standard_preds[i] - exp_A_max[i]) / exp_A_max[i] * 100
                      for i in range(len(exp_tau))]

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, weighted_errors, width, label='Weighted Optimization')
    rects2 = ax.bar(x + width/2, standard_errors, width, label='Standard Optimization')

    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title('Error Comparison: Weighted vs. Standard Optimization', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add weight annotations
    for i, weight in enumerate(weights):
        ax.annotate(f"Weight: {weight:.1f}", xy=(i, 0), xytext=(i-0.4, -5),
                   textcoords="offset points", ha='center', fontsize=8)

    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.grid(True, alpha=0.3)
    plt.savefig('weighted_vs_standard_errors.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    optimized_params = main()

    # Print final parameters to use in the main model
    print("\nParameters for use in the simulation (weighted optimization):")
    params = optimized_params['weighted']
    print(f"params = {{")
    print(f"    'k0': 0.5,        # h^-1")
    print(f"    'alpha': 0.4,     # dimensionless")
    print(f"    'beta': 0.5,      # dimensionless")
    print(f"    'gamma': 0.3,     # dimensionless")
    print(f"    'tau_threshold': {params['tau_threshold']}, # dyn/cm^2")
    print(f"    'tau_ref': 25,    # dyn/cm^2")
    print(f"    'a': {params['a']:.6f},        # (dyn/cm^2)^-b")
    print(f"    'b': {params['b']:.6f},        # dimensionless")
    print(f"    'threshold_value': {params['threshold_value']},  # baseline value")
    print(f"}}")