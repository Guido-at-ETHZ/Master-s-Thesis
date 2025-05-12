import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.gridspec import GridSpec

# Optimized parameters from the original model
V_max = 2.000
K_m = 10.000
tau_base = 0.160
scaling_factor = 1.436

# Your provided data
P_values = np.array([15, 25, 45])
A_max_map = {15: 1.5, 25: 3.7, 45: 5.3}

# Colors for visualization
colors = {15: 'blue', 25: 'green', 45: 'red'}
markers = {15: 's', 25: 'o', 45: '^'}


# Function to calculate steady-state for a given P
def calculate_steady_state(P):
    v = V_max * P / (K_m + P)
    K = A_max_map[P] / v
    return K * v  # Should equal A_max_map[P]


# Function to calculate base time constant (tau) for a given steady-state
def calculate_base_tau(steady_state):
    return tau_base * (steady_state / 1.0) ** scaling_factor


# Calculate pressure-dependent time multiplier for falling
def calculate_fall_multiplier(P):
    if P < 10:
        return 1.5  # Minimum multiplier
    else:
        return 1.5 + 0.25 * ((P - 10) / 10)


# Function to directly solve the ODE using analytical solutions
def solve_exponential(t, y0, steady_state, tau):
    """Solve exponential approach to steady state with given tau"""
    return steady_state - (steady_state - y0) * np.exp(-t / tau)


# Simulation for a transition with correct direction handling
def simulate_transition(P_start, P_end, t_switch, t_max, use_linear_fall=False):
    # Calculate steady states
    ss_start = calculate_steady_state(P_start)
    ss_end = calculate_steady_state(P_end)

    # Determine if this is a rising or falling transition
    is_rising = ss_end > ss_start

    # Set up time points for simulation
    t = np.linspace(0, t_max, 500)
    y = np.zeros_like(t)

    # Initialize with steady state of P_start
    y[0] = ss_start

    # Time points before the switch - system is at steady state
    pre_switch_indices = t < t_switch
    y[pre_switch_indices] = ss_start

    # Get indices for the transition period
    post_switch_indices = t >= t_switch
    t_post = t[post_switch_indices] - t_switch  # Time since switch

    # Calculate appropriate tau based on direction
    if is_rising:
        # Rising: use base tau based on target state
        tau = calculate_base_tau(ss_end)

        # Solve using exponential approach
        y[post_switch_indices] = solve_exponential(t_post, ss_start, ss_end, tau)
    else:
        # Falling: apply multiplier to tau
        base_tau = calculate_base_tau(ss_start)
        multiplier = calculate_fall_multiplier(P_start)
        tau_fall = base_tau * multiplier

        if use_linear_fall:
            # Calculate time to reach 95% with modified tau for reference
            t_95_fall = -tau_fall * np.log(0.05)

            # Linear rate to achieve equivalent time scale
            rate = (ss_end - ss_start) / t_95_fall

            # Apply linear fall equation
            y_linear = ss_start + rate * t_post

            # Ensure we don't overshoot steady state
            for i, val in enumerate(y_linear):
                if rate < 0 and val < ss_end:  # Falling and below target
                    y_linear[i] = ss_end
                elif rate > 0 and val > ss_end:  # Rising and above target
                    y_linear[i] = ss_end

            y[post_switch_indices] = y_linear
        else:
            # Solve using exponential approach with modified tau
            y[post_switch_indices] = solve_exponential(t_post, ss_start, ss_end, tau_fall)

    return t, y


# Create figure with custom layout
fig = plt.figure(figsize=(15, 12))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

# Suptitle for entire figure
plt.suptitle('Corrected Asymmetric Model: Rising vs Falling Dynamics', fontsize=16, y=0.98)

# 1. Normalized step responses - comparing all transitions
ax1 = fig.add_subplot(gs[0, :])

# Create a list of transitions to compare
transitions = [
    (15, 25, "15→25 (Up)"),
    (25, 15, "25→15 (Down)"),
    (15, 45, "15→45 (Up)"),
    (45, 15, "45→15 (Down)"),
    (25, 45, "25→45 (Up)"),
    (45, 25, "45→25 (Down)")
]

# Common parameters for all simulations
t_switch = 2
t_max = 15

# Plot normalized exponential responses
for P_start, P_end, label in transitions:
    # Simulate the transition
    t, y = simulate_transition(P_start, P_end, t_switch, t_max)

    # Calculate steady states for normalization
    ss_start = calculate_steady_state(P_start)
    ss_end = calculate_steady_state(P_end)

    # Normalize the response
    y_norm = (y - ss_start) / (ss_end - ss_start)

    # Determine line style and color based on direction
    is_rising = ss_end > ss_start
    color = 'red' if is_rising else 'blue'

    # Plot normalized response
    ax1.plot(t - t_switch, y_norm,
             label=label,
             color=color,
             linestyle='-',
             linewidth=2)

# Add reference lines for percentages
percentages = [50, 63, 90, 95]
for pct in percentages:
    ax1.axhline(pct / 100, color='gray', linestyle=':', alpha=0.5)
    ax1.text(0.1, pct / 100 - 0.03, f"{pct}%", color='gray')

# Customize plot
ax1.set_xlim(-0.5, 12)
ax1.set_ylim(-0.1, 1.1)
ax1.set_xlabel('Time After Transition (time units)', fontsize=12)
ax1.set_ylabel('Normalized Response', fontsize=12)
ax1.set_title('Normalized Step Responses: Rising vs Falling (Exponential Model)', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right')

# 2. Compare specific transitions (exponential and linear models)
ax2 = fig.add_subplot(gs[1, 0])

# Choose specific transitions to showcase
transitions_to_show = [
    (15, 45, "15→45 (Rise)"),
    (45, 15, "45→15 (Fall)")
]

# Plot both exponential and linear models
for P_start, P_end, label in transitions_to_show:
    # Exponential model
    t, y_exp = simulate_transition(P_start, P_end, t_switch, t_max, use_linear_fall=False)

    # Linear fall model (only applies to falling transitions)
    t, y_lin = simulate_transition(P_start, P_end, t_switch, t_max, use_linear_fall=True)

    # Calculate steady states for normalization
    ss_start = calculate_steady_state(P_start)
    ss_end = calculate_steady_state(P_end)

    # Normalize responses
    y_exp_norm = (y_exp - ss_start) / (ss_end - ss_start)
    y_lin_norm = (y_lin - ss_start) / (ss_end - ss_start)

    # Determine style based on direction
    is_rising = ss_end > ss_start
    color = 'red' if is_rising else 'blue'

    # Plot exponential model
    ax2.plot(t - t_switch, y_exp_norm,
             label=f"{label} (Exp)",
             color=color,
             linestyle='-',
             linewidth=2)

    # Plot linear model
    if not is_rising:  # Only use linear for falling
        ax2.plot(t - t_switch, y_lin_norm,
                 label=f"{label} (Lin)",
                 color=color,
                 linestyle='--',
                 linewidth=2)

# Mark when responses reach specific percentages
for P_start, P_end, label in transitions_to_show:
    t, y = simulate_transition(P_start, P_end, t_switch, t_max)
    ss_start = calculate_steady_state(P_start)
    ss_end = calculate_steady_state(P_end)
    y_norm = (y - ss_start) / (ss_end - ss_start)

    is_rising = ss_end > ss_start
    color = 'red' if is_rising else 'blue'

    # Mark times to reach percentages
    for pct in [50, 90]:
        # Find first time point where response crosses percentage
        try:
            if is_rising:
                idx = np.where(y_norm >= pct / 100)[0][0]
            else:
                idx = np.where(y_norm <= pct / 100)[0][0]

            t_pct = t[idx] - t_switch
            ax2.plot(t_pct, pct / 100, 'o', color=color, markersize=8)
            ax2.text(t_pct + 0.2, pct / 100 + 0.03, f"{t_pct:.2f}", color=color)
        except IndexError:
            # Percentage not reached within simulation time
            pass

# Add reference lines
for pct in percentages:
    ax2.axhline(pct / 100, color='gray', linestyle=':', alpha=0.5)

# Customize plot
ax2.set_xlim(-0.5, 12)
ax2.set_ylim(-0.1, 1.1)
ax2.set_xlabel('Time After Transition (time units)', fontsize=12)
ax2.set_ylabel('Normalized Response', fontsize=12)
ax2.set_title('Comparing Specific Transitions: Exponential vs Linear Fall', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right')

# 3. Actual response curves with modified tau
ax3 = fig.add_subplot(gs[1, 1])

# Choose specific transitions
P_rise_start, P_rise_end = 15, 45
P_fall_start, P_fall_end = 45, 15

# Simulate transitions
t, y_rise = simulate_transition(P_rise_start, P_rise_end, t_switch, t_max)
t, y_fall = simulate_transition(P_fall_start, P_fall_end, t_switch, t_max)

# Calculate time constants
ss_rise_start = calculate_steady_state(P_rise_start)
ss_rise_end = calculate_steady_state(P_rise_end)
ss_fall_start = calculate_steady_state(P_fall_start)
ss_fall_end = calculate_steady_state(P_fall_end)

tau_rise = calculate_base_tau(ss_rise_end)
base_tau_fall = calculate_base_tau(ss_fall_start)
multiplier_fall = calculate_fall_multiplier(P_fall_start)
tau_fall = base_tau_fall * multiplier_fall

# Plot actual response values
ax3.plot(t, y_rise, color='red', linestyle='-', linewidth=2,
         label=f'Rise {P_rise_start}→{P_rise_end} (τ={tau_rise:.2f})')
ax3.plot(t, y_fall, color='blue', linestyle='-', linewidth=2,
         label=f'Fall {P_fall_start}→{P_fall_end} (τ={tau_fall:.2f})')

# Add steady state reference lines
ax3.axhline(ss_rise_start, color='gray', linestyle=':', alpha=0.5)
ax3.axhline(ss_rise_end, color='gray', linestyle=':', alpha=0.5)
ax3.axhline(ss_fall_start, color='gray', linestyle=':', alpha=0.5)
ax3.axhline(ss_fall_end, color='gray', linestyle=':', alpha=0.5)

# Mark transition time
ax3.axvline(t_switch, color='black', linestyle='--', alpha=0.5)
ax3.text(t_switch + 0.2, (ss_rise_start + ss_fall_start) / 2, "P changes", color='black')


# Calculate and mark time to reach 90% of change
def mark_time_point(ax, t, y, t_switch, y_start, y_end, percentage, color):
    # Calculate target value
    target = y_start + (y_end - y_start) * percentage / 100

    # Find first time point after switch where value crosses target
    post_switch = t >= t_switch
    t_post = t[post_switch]
    y_post = y[post_switch]

    is_rising = y_end > y_start
    if is_rising:
        idx = np.where(y_post >= target)[0]
    else:
        idx = np.where(y_post <= target)[0]

    if len(idx) > 0:
        idx = idx[0]
        t_target = t_post[idx]
        t_elapsed = t_target - t_switch

        # Mark on plot
        ax.plot(t_target, target, 'o', color=color, markersize=8)
        ax.annotate(f"{percentage}%: {t_elapsed:.2f}",
                    xy=(t_target, target),
                    xytext=(t_target + 0.5, target + 0.2 * (-1 if is_rising else 1)),
                    color=color,
                    arrowprops=dict(arrowstyle="->", color=color))

        return t_elapsed
    return None


# Mark time points for both transitions
mark_time_point(ax3, t, y_rise, t_switch, ss_rise_start, ss_rise_end, 90, 'red')
mark_time_point(ax3, t, y_fall, t_switch, ss_fall_start, ss_fall_end, 90, 'blue')

# Customize plot
ax3.set_xlabel('Time (time units)', fontsize=12)
ax3.set_ylabel('Response Value', fontsize=12)
ax3.set_title('Actual Response Values with Modified Time Constants', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='center right')

# Print transition summary table
print("Transition Summary with Corrected Implementation:")
print("-" * 90)
print(f"{'P Start':^8} {'P End':^8} {'Direction':^10} {'Δ Magnitude':^12}", end='')
print(f" {'Base τ':^10} {'Multiplier':^10} {'Modified τ':^12} {'Time to 95%':^12}")
print("-" * 90)

# Calculate and print summary for all transitions
for P_start in P_values:
    for P_end in P_values:
        if P_start != P_end:
            ss_start = calculate_steady_state(P_start)
            ss_end = calculate_steady_state(P_end)

            is_rising = ss_end > ss_start
            direction = "Rising" if is_rising else "Falling"

            if is_rising:
                # For rising: tau based on target steady state
                base_tau = calculate_base_tau(ss_end)
                multiplier = 1.0  # No multiplier for rising
                modified_tau = base_tau  # No change
            else:
                # For falling: tau based on initial steady state
                base_tau = calculate_base_tau(ss_start)
                multiplier = calculate_fall_multiplier(P_start)
                modified_tau = base_tau * multiplier  # Apply multiplier directly to tau

            # Calculate times with correct tau
            delta = abs(ss_end - ss_start)
            time_95 = -modified_tau * np.log(0.05)  # Time to reach 95% with modified tau

            print(f"{P_start:^8} {P_end:^8} {direction:^10} {delta:^12.2f}", end='')
            print(f" {base_tau:^10.3f} {multiplier:^10.2f} {modified_tau:^12.3f} {time_95:^12.3f}")

# Add a text box with key findings
key_findings = """Model Implementation:
1. Rising: Time constant τ = 0.160 * (steady_state)^1.436
2. Falling: τ_fall = τ_base * multiplier, where multiplier = 1.5 + 0.25*((P-10)/10)
3. This makes falling responses 1.5× to 2.4× slower than rising at equivalent magnitudes
4. The steeper the gradient (larger Δ), the longer it takes to adapt"""

fig.tight_layout()
fig.subplots_adjust(bottom=0.15)
fig.text(0.5, 0.04, key_findings, ha='center', va='center',
         bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

plt.show()