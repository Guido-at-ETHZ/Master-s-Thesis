import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint

# --- Original Parameters ---
P_values = np.array([15, 25, 45])
A_max_map = {15: 1.5, 25: 3.7, 45: 5.3}
y0 = 1.0

# Experimental data
exp_data_orig = {
    15: {'t': np.array([0, 1, 3, 6]), 'y': np.array([1.0, 1.4, 1.0, 1.1])},
    25: {'t': np.array([0, 1, 3, 6]), 'y': np.array([1.0, 3.6, 3.0, 2.9])},
    45: {'t': np.array([0, 1, 2, 6]), 'y': np.array([1.0, 3.0, 3.5, 5.3])}
}
colors = {15: 'blue', 25: 'green', 45: 'red'}
markers = {15: 's', 25: 'o', 45: '^'}


# --- Define a model where the derivative (dy/dt) follows Michaelis-Menten kinetics ---
def mm_derivative_model(t, S, params):
    """
    Model where the derivative (dy/dt) follows Michaelis-Menten kinetics,
    with an additional feedback term to capture the adaptation seen in data.

    Parameters:
    t: time points
    S: shear stress
    params: model parameters
    """

    # Set up ODE function
    def mm_dynamics(y, t):
        # Calculate driving force - how far from target we are
        # This captures the feedback mechanism
        driving_force = params['target_mult'] * S - y

        # Michaelis-Menten production rate with driving force
        # Rate is high when far from target, slows down as we approach
        if driving_force > 0:
            mm_rate = params['Vmax'] * abs(driving_force) / (params['Km'] + abs(driving_force))
        else:
            # When above target, return to baseline with a different (usually faster) rate
            mm_rate = -params['Vdown'] * abs(driving_force) / (params['Km_down'] + abs(driving_force))

        # Add a time-dependent adaptation term that causes some initial overshoot
        # and then return to steady state (important for S=15 and S=25)
        adaptation = params['adapt_amp'] * S * np.exp(-params['adapt_rate'] * t)

        return mm_rate + adaptation

    # Solve ODE
    y_solution = odeint(mm_dynamics, y0, t)

    return y_solution.flatten()


# --- Function to fit model parameters ---
def fit_mm_derivative_model(exp_data, P_values):
    """
    Fit the Michaelis-Menten derivative model to experimental data
    """
    # Collect all data points for fitting
    all_t = []
    all_y = []
    all_S = []

    for S in P_values:
        if S in exp_data:
            t_points = exp_data[S]['t']
            y_points = exp_data[S]['y']
            all_t.extend(t_points)
            all_y.extend(y_points)
            all_S.extend([S] * len(t_points))

    all_t = np.array(all_t)
    all_y = np.array(all_y)
    all_S = np.array(all_S)

    # Define wrapper function for curve_fit
    def model_wrapper(t_S, Vmax, Km, Vdown, Km_down, target_mult, adapt_amp, adapt_rate):
        """Wrapper for curve_fit that unpacks parameters"""
        # Extract time and shear stress from combined array
        times = t_S[:len(t_S) // 2]
        stresses = t_S[len(t_S) // 2:]

        params = {
            'Vmax': Vmax,  # Maximum production rate
            'Km': Km,  # MM constant for production
            'Vdown': Vdown,  # Maximum decay/return rate
            'Km_down': Km_down,  # MM constant for decay
            'target_mult': target_mult,  # Multiplier for target level based on S
            'adapt_amp': adapt_amp,  # Amplitude of adaptation term
            'adapt_rate': adapt_rate  # Rate of adaptation
        }

        predictions = []
        for i in range(len(times)):
            t = np.array([0, times[i]])  # Include t=0 for ODE solver
            S = stresses[i]
            y_pred = mm_derivative_model(t, S, params)
            predictions.append(y_pred[-1])  # Take last value (t=times[i])

        return np.array(predictions)

    # Pack time and stress into single array for curve_fit
    t_S = np.concatenate((all_t, all_S))

    # Initial parameter guesses
    # These should be good starting values for this system
    initial_params = [
        1.0,  # Vmax
        2.0,  # Km
        0.5,  # Vdown
        1.0,  # Km_down
        0.1,  # target_mult
        0.5,  # adapt_amp
        0.3  # adapt_rate
    ]

    try:
        # Bounds to ensure physically meaningful parameters
        lower_bounds = [0.01, 0.01, 0.01, 0.01, 0.01, -2.0, 0.01]
        upper_bounds = [10.0, 10.0, 10.0, 10.0, 0.2, 2.0, 5.0]

        # Perform the fit
        popt, pcov = curve_fit(
            model_wrapper,
            t_S,
            all_y,
            p0=initial_params,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000,  # Increase max function evaluations
            method='trf'  # Use trust region reflective method for better handling of bounds
        )

        # Extract fitted parameters
        Vmax, Km, Vdown, Km_down, target_mult, adapt_amp, adapt_rate = popt

        # Calculate parameter errors
        perr = np.sqrt(np.diag(pcov))

        # Create parameter dictionary
        fitted_params = {
            'Vmax': Vmax,
            'Km': Km,
            'Vdown': Vdown,
            'Km_down': Km_down,
            'target_mult': target_mult,
            'adapt_amp': adapt_amp,
            'adapt_rate': adapt_rate
        }

        # Calculate goodness of fit
        fit_metrics = {}

        # Calculate model predictions for all data points
        all_predictions = []
        for i, (t, S) in enumerate(zip(all_t, all_S)):
            t_eval = np.array([0, t])  # Include t=0 for solver
            pred = mm_derivative_model(t_eval, S, fitted_params)
            all_predictions.append(pred[-1])
        all_predictions = np.array(all_predictions)

        # Calculate overall metrics
        residuals = all_y - all_predictions
        ssr = np.sum(residuals ** 2)
        n_points = len(all_y)
        rmse = np.sqrt(ssr / n_points)

        ss_tot = np.sum((all_y - np.mean(all_y)) ** 2)
        r_squared = 1 - (ssr / ss_tot) if ss_tot > 0 else 0.0

        # Calculate metrics for each shear stress value
        for S in P_values:
            if S in exp_data:
                t_exp = exp_data[S]['t']
                y_exp = exp_data[S]['y']

                # Calculate model predictions
                y_pred = []
                for t in t_exp:
                    t_eval = np.array([0, t])
                    pred = mm_derivative_model(t_eval, S, fitted_params)
                    y_pred.append(pred[-1])
                y_pred = np.array(y_pred)

                # Calculate metrics
                residuals = y_exp - y_pred
                ssr = np.sum(residuals ** 2)
                n_points = len(y_exp)
                rmse = np.sqrt(ssr / n_points)

                mean_y_exp = np.mean(y_exp)
                ss_tot = np.sum((y_exp - mean_y_exp) ** 2)
                r_squared = 1 - (ssr / ss_tot) if ss_tot > 0 else 0.0

                fit_metrics[S] = {
                    'SSR': ssr,
                    'RMSE': rmse,
                    'R2': r_squared
                }

        # Calculate steady state values (by running model for a long time)
        steady_states = {}
        for S in P_values:
            t_long = np.linspace(0, 50, 100)  # Run for a long time
            y_long = mm_derivative_model(t_long, S, fitted_params)
            steady_states[S] = y_long[-1]  # Take the last value

        return fitted_params, perr, fit_metrics, steady_states

    except Exception as e:
        print(f"Fitting error: {e}")
        # Return default parameters in case of error
        return {
            'Vmax': 1.0,
            'Km': 2.0,
            'Vdown': 0.5,
            'Km_down': 1.0,
            'target_mult': 0.1,
            'adapt_amp': 0.5,
            'adapt_rate': 0.3
        }, np.zeros(7), {}, {}


# --- Main code execution ---
print("Fitting Michaelis-Menten derivative model to experimental data...")
fitted_params, param_errors, fit_metrics, steady_states = fit_mm_derivative_model(exp_data_orig, P_values)

# Print fitted parameters
print("\n--- Fitted Parameters ---")
print(f"Vmax = {fitted_params['Vmax']:.4f} ± {param_errors[0]:.4f}")
print(f"Km = {fitted_params['Km']:.4f} ± {param_errors[1]:.4f}")
print(f"Vdown = {fitted_params['Vdown']:.4f} ± {param_errors[2]:.4f}")
print(f"Km_down = {fitted_params['Km_down']:.4f} ± {param_errors[3]:.4f}")
print(f"target_mult = {fitted_params['target_mult']:.4f} ± {param_errors[4]:.4f}")
print(f"adapt_amp = {fitted_params['adapt_amp']:.4f} ± {param_errors[5]:.4f}")
print(f"adapt_rate = {fitted_params['adapt_rate']:.4f} ± {param_errors[6]:.4f}")

# Print steady states
print("\n--- Predicted Steady States vs. Target ---")
for S in P_values:
    print(f"S = {S}: Predicted steady state = {steady_states[S]:.4f}, Target = {A_max_map[S]}")

# Print fit metrics by shear stress
print("\n--- Fit Metrics by Shear Stress ---")
for S in P_values:
    if S in fit_metrics:
        print(f"S = {S}:")
        print(f"  RMSE = {fit_metrics[S]['RMSE']:.4f}")
        print(f"  R² = {fit_metrics[S]['R2']:.4f}")

# --- Plot model predictions vs experimental data ---
plt.figure(figsize=(12, 8))
t_plot = np.linspace(0, 7, 200)  # Plot up to 7 hours

# Plot model and data for each shear stress
for S in P_values:
    # Calculate model prediction
    y_model = mm_derivative_model(t_plot, S, fitted_params)

    # Plot model prediction
    plt.plot(t_plot, y_model, color=colors[S], linestyle='-', linewidth=2,
             label=f'S={S} Model (SS={steady_states[S]:.2f})')

    # Plot experimental data if available
    if S in exp_data_orig:
        t_exp = exp_data_orig[S]['t']
        y_exp = exp_data_orig[S]['y']

        # Plot data points
        plt.scatter(t_exp, y_exp, color=colors[S], marker=markers[S], s=100,
                    label=f'S={S} Exp. Data')

# Plot configuration
plt.xlabel("Duration of Shear Stress (Hours)", fontsize=12)
plt.ylabel("Normalized Level of β-Actin mRNA", fontsize=12)
plt.title("Michaelis-Menten Rate Model vs. Experimental Data", fontsize=14)
plt.legend(loc='best', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.ylim(bottom=0)
plt.xlim(left=0, right=7)
plt.tight_layout()

# --- Calculate and plot derivative (dy/dt) to show Michaelis-Menten behavior ---
plt.figure(figsize=(12, 6))

# Plot derivatives for each shear stress
for S in P_values:
    # Calculate model prediction with high time resolution
    t_dense = np.linspace(0, 7, 500)
    y_dense = mm_derivative_model(t_dense, S, fitted_params)

    # Numerically calculate derivative
    dy_dt = np.diff(y_dense) / np.diff(t_dense)
    t_mid = (t_dense[:-1] + t_dense[1:]) / 2  # Midpoints for derivative plot

    # Plot derivative
    plt.plot(t_mid, dy_dt, color=colors[S], linestyle='-',
             label=f'S={S} dy/dt')

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel("Time (Hours)")
plt.ylabel("Rate of Change (dy/dt)")
plt.title("Rate of Change Following Michaelis-Menten Kinetics")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- Plot Michaelis-Menten saturation curve for production and decay ---
plt.figure(figsize=(10, 8))

# Subplot 1: Production rate vs driving force (positive)
plt.subplot(2, 1, 1)
driving_force = np.linspace(0, 10, 100)
mm_rate = fitted_params['Vmax'] * driving_force / (fitted_params['Km'] + driving_force)

plt.plot(driving_force, mm_rate, 'b-', linewidth=2)
plt.axhline(y=fitted_params['Vmax'], color='r', linestyle='--', label=f'Vmax={fitted_params["Vmax"]:.2f}')
plt.axvline(x=fitted_params['Km'], color='g', linestyle='--', label=f'Km={fitted_params["Km"]:.2f}')
plt.xlabel("Driving Force (Target - Current)")
plt.ylabel("Production Rate")
plt.title("Michaelis-Menten Kinetics of Production Rate")
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 2: Decay rate vs driving force (negative)
plt.subplot(2, 1, 2)
driving_force = np.linspace(0, 10, 100)
mm_rate = fitted_params['Vdown'] * driving_force / (fitted_params['Km_down'] + driving_force)

plt.plot(driving_force, mm_rate, 'r-', linewidth=2)
plt.axhline(y=fitted_params['Vdown'], color='b', linestyle='--', label=f'Vdown={fitted_params["Vdown"]:.2f}')
plt.axvline(x=fitted_params['Km_down'], color='g', linestyle='--', label=f'Km_down={fitted_params["Km_down"]:.2f}')
plt.xlabel("Driving Force (Current - Target)")
plt.ylabel("Decay Rate")
plt.title("Michaelis-Menten Kinetics of Decay Rate")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()


# --- Analyze time to reach various target levels ---
# Define a function to find time to reach target level
def time_to_reach(S, target_level, params):
    """Find time required to reach a target level under shear stress S"""
    # Create dense time points for accurate interpolation
    t_dense = np.linspace(0, 20, 1000)
    y_dense = mm_derivative_model(t_dense, S, params)

    # Find first time point where y exceeds target
    indices = np.where(y_dense >= target_level)[0]
    if len(indices) > 0:
        return t_dense[indices[0]]
    else:
        return np.nan  # Can't reach this level


# Calculate times to reach various target levels
print("\n--- Time to Reach Various Target Levels ---")
target_levels = [1.5, 2.0, 3.0, 4.0, 5.0]

print("\nTarget Level | S=15 (hrs) | S=25 (hrs) | S=45 (hrs) | Ratio S45/S25")
print("---------------------------------------------------------------")

for level in target_levels:
    times = {}
    for S in P_values:
        times[S] = time_to_reach(S, level, fitted_params)

    # Calculate ratio of times between S=45 and S=25 (if both values exist)
    ratio = times[45] / times[25] if not np.isnan(times[45]) and not np.isnan(times[25]) else np.nan

    print(f"{level:11.1f} | {times[15]:10.2f} | {times[25]:10.2f} | {times[45]:10.2f} | {ratio:14.2f}")

# --- Plot time vs. target level curve ---
plt.figure(figsize=(10, 6))

# Set of target levels to analyze
level_range = np.linspace(1.1, 5.5, 50)

for S in P_values:
    times = []
    valid_levels = []

    for level in level_range:
        t = time_to_reach(S, level, fitted_params)
        if not np.isnan(t):
            times.append(t)
            valid_levels.append(level)

    plt.plot(valid_levels, times, '-', color=colors[S], linewidth=2,
             label=f'S={S} (SS={steady_states[S]:.2f})')

plt.xlabel('Target Level')
plt.ylabel('Time to Reach (Hours)')
plt.title('Time Required to Reach Different Target Levels')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Show all plots
plt.show()