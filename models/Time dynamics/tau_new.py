import numpy as np
import matplotlib.pyplot as plt

# --- Parameters for the Model ---
P_values = np.array([15, 25, 45])
# Use A_max_experimental as the target steady state (yf)
A_max_map = {15: 1.5, 25: 3.7, 45: 5.3}
# Use Tau values derived from user's 5*tau estimates
tau_map = {15: 0.34, 25: 0.50, 45: 1.60}
# Initial value
y0 = 1.0

# --- Estimated Experimental Data Points from Original Image ---
# (for visual comparison)
exp_data_orig = {
    15: {'t': np.array([0, 1, 3, 6]), 'y': np.array([1.0, 1.4, 1.0, 1.1])}, # Approx squares
    25: {'t': np.array([0, 1, 3, 6]), 'y': np.array([1.0, 3.6, 3.0, 2.9])}, # Approx circles
    45: {'t': np.array([0, 1, 2, 6]), 'y': np.array([1.0, 3.0, 3.5, 5.3])}  # Approx triangles
}
colors = {15: 'blue', 25: 'green', 45: 'red'}
markers = {15: 's', 25: 'o', 45: '^'}


# --- Model Definition ---
# First-order step response: y(t) = y0 + (yf - y0) * (1 - exp(-t / tau))
def first_order_model(t, y0, yf, tau):
    """
    Calculates first-order step response.
    t: time
    y0: Initial value
    yf: Final value (A_max)
    tau: Time constant
    """
    # Prevent division by zero during calculation if tau is somehow zero
    tau = max(tau, 1e-9)
    return y0 + (yf - y0) * (1 - np.exp(-t / tau))

# --- Plotting Setup ---
plt.figure(figsize=(10, 7)) # Increased height slightly for potentially longer legend
# Time vector for plotting smooth curves
t_plot = np.linspace(0, 7, 200) # Plot up to 7 hours

# --- Dictionary to store fit metrics ---
fit_metrics = {}

# --- Plotting and Calculation Loop ---
print("--- Model Parameters and Goodness-of-Fit ---")
for P in P_values:
    yf = A_max_map[P]
    tau = tau_map[P]

    # Calculate the model dynamics for smooth plotting
    y_model_plot = first_order_model(t_plot, y0, yf, tau)

    # Plot the model curve
    plt.plot(t_plot, y_model_plot,
             label=f'S={P} Model (Amax={yf:.1f}, τ={tau:.2f})',
             color=colors[P], linestyle='--')

    # --- Goodness-of-Fit Calculation ---
    if P in exp_data_orig:
        t_exp = exp_data_orig[P]['t']
        y_exp = exp_data_orig[P]['y']
        n_points = len(y_exp)

        # Calculate model predictions ONLY at experimental time points
        y_model_at_exp_t = first_order_model(t_exp, y0, yf, tau)

        # Calculate residuals
        residuals = y_exp - y_model_at_exp_t

        # Calculate Sum of Squared Residuals (SSR)
        ssr = np.sum(residuals**2)

        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(ssr / n_points)

        # Calculate R-squared (Coefficient of Determination)
        mean_y_exp = np.mean(y_exp)
        sst = np.sum((y_exp - mean_y_exp)**2) # Total Sum of Squares
        if sst == 0: # Handle cases with zero variance in data (unlikely here)
            r_squared = 1.0 if ssr == 0 else 0.0
        else:
            r_squared = 1 - (ssr / sst)

        # Store metrics
        fit_metrics[P] = {'SSR': ssr, 'RMSE': rmse, 'R2': r_squared}

        # Print results for this P value
        print(f"\nS = {P}:")
        print(f"  Parameters: A_max (yf) = {yf:.1f}, Tau = {tau:.2f}")
        print(f"  Fit Metrics:")
        print(f"    SSR  = {ssr:.4f}")
        print(f"    RMSE = {rmse:.4f}")
        print(f"    R²   = {r_squared:.4f}")

        # Plot estimated experimental data points from original image
        plt.scatter(t_exp, y_exp,
                    label=f'S={P} Exp. Data (RMSE={rmse:.2f}, R²={r_squared:.2f})', # Add metrics to label
                    color=colors[P], marker=markers[P], s=60, zorder=5) # Slightly larger markers
    else:
        print(f"\nS = {P}: Model plotted, but no experimental data provided for fit calculation.")


# --- Final Plot Configuration ---
plt.xlabel("Duration of Shear Stress (Hours)")
plt.ylabel("Normalized Level of β-Actin mRNA")
plt.title("First-Order Model vs. Estimated Experimental Data with Fit Metrics")
plt.legend(loc='best') # 'best' tries to find an optimal location
plt.grid(True, linestyle=':', alpha=0.7)
plt.ylim(bottom=0)
plt.xlim(left=0, right=7)
plt.tight_layout() # Adjust plot to prevent labels overlapping
plt.show()

# You can also access the stored metrics later if needed:
print("\nStored Fit Metrics Dictionary:")
print(fit_metrics)