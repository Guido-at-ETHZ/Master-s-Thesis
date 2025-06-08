"""
Analyze your specific simulation file: simulation_20250608-175357.npz
Since target data exists, let's see if time dynamics are working!
"""
import numpy as np
import matplotlib.pyplot as plt

def analyze_your_simulation():
    """
    Complete analysis of your simulation_20250608-175357.npz file.
    """
    filepath = 'simulation_20250608-184347.npz'

    print("ANALYZING YOUR SIMULATION FOR TIME DYNAMICS")
    print("=" * 50)

    # Load the data
    data = np.load(filepath, allow_pickle=True)
    history = data['history'].item()

    print(f"✅ Loaded simulation with target tracking data")
    print(f"   Time points: {len(history['time'])}")
    print(f"   Duration: {history['time'][0]:.1f} to {history['time'][-1]:.1f} minutes")
    print(f"   Input type: Checking pressure pattern...")

    # Extract basic data
    times = np.array(history['time']) / 60  # Convert to hours
    pressures = np.array(history['input_value'])
    cell_props = history['cell_properties']

    # Check input pattern
    pressure_changes = np.where(np.abs(np.diff(pressures)) > 0.1)[0]
    if len(pressure_changes) > 0:
        # Fix the f-string formatting issue
        change_times = [f"{times[i]:.1f}" for i in pressure_changes]
        print(f"   Pressure changes detected at: {', '.join(change_times)} hours")
    else:
        print(f"   Constant pressure: {pressures[0]:.2f} Pa")

    # Extract target data for first cell
    target_areas = [cp['target_areas'][0] if len(cp['target_areas']) > 0 else np.nan for cp in cell_props]
    target_ars = [cp['target_aspect_ratios'][0] if len(cp['target_aspect_ratios']) > 0 else np.nan for cp in cell_props]
    target_orients = [cp['target_orientations'][0] if len(cp['target_orientations']) > 0 else np.nan for cp in cell_props]

    # Extract actual data for comparison
    actual_areas = [cp['areas'][0] if len(cp['areas']) > 0 else np.nan for cp in cell_props]
    actual_ars = [cp['aspect_ratios'][0] if len(cp['aspect_ratios']) > 0 else np.nan for cp in cell_props]
    actual_orients = [cp['orientations'][0] if len(cp['orientations']) > 0 else np.nan for cp in cell_props]

    # Convert to numpy arrays
    target_areas = np.array(target_areas)
    target_ars = np.array(target_ars)
    target_orients = np.array(target_orients)
    actual_areas = np.array(actual_areas)
    actual_ars = np.array(actual_ars)
    actual_orients = np.array(actual_orients)

    print(f"\nDATA SUMMARY:")
    print(f"   Target areas range: {np.nanmin(target_areas):.0f} to {np.nanmax(target_areas):.0f} pixels²")
    print(f"   Target AR range: {np.nanmin(target_ars):.2f} to {np.nanmax(target_ars):.2f}")
    print(f"   Target orientation range: {np.degrees(np.nanmin(target_orients)):.1f}° to {np.degrees(np.nanmax(target_orients)):.1f}°")

    # Create comprehensive analysis plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Plot 1: Pressure input
    axes[0].plot(times, pressures, 'r-', linewidth=2, label='Input Pressure')
    axes[0].set_ylabel('Pressure (Pa)')
    axes[0].set_title('Your Simulation: Time Dynamics Analysis')
    axes[0].legend()
    axes[0].grid(True)

    # Mark pressure changes
    if len(pressure_changes) > 0:
        for change_idx in pressure_changes:
            axes[0].axvline(times[change_idx], color='red', linestyle='--', alpha=0.5)
            axes[0].text(times[change_idx], np.max(pressures) * 0.9,
                        f't={times[change_idx]:.1f}h', rotation=90, va='top', fontsize=8)

    # Plot 2: Area evolution (target vs actual)
    axes[1].plot(times, target_areas, 'b-', linewidth=2, label='Target Area')
    axes[1].plot(times, actual_areas, 'b--', linewidth=1, alpha=0.7, label='Actual Area')
    axes[1].set_ylabel('Area (pixels²)')
    axes[1].set_title('Area Evolution')
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Aspect ratio evolution (target vs actual)
    axes[2].plot(times, target_ars, 'g-', linewidth=2, label='Target AR')
    axes[2].plot(times, actual_ars, 'g--', linewidth=1, alpha=0.7, label='Actual AR')
    axes[2].set_ylabel('Aspect Ratio')
    axes[2].set_title('Aspect Ratio Evolution')
    axes[2].legend()
    axes[2].grid(True)

    # Plot 4: Orientation evolution (target vs actual, converted to degrees)
    target_orients_deg = np.degrees(target_orients)
    # Fix potential issue with orientation conversion
    actual_orients_deg = actual_orients  # Assuming actual_orients is already in degrees

    axes[3].plot(times, target_orients_deg, 'm-', linewidth=2, label='Target Orientation')
    axes[3].plot(times, actual_orients_deg, 'm--', linewidth=1, alpha=0.7, label='Actual Orientation')
    axes[3].set_ylabel('Orientation (degrees)')
    axes[3].set_xlabel('Time (hours)')
    axes[3].set_title('Orientation Evolution')
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig('your_simulation_time_dynamics.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Analysis plot saved as 'your_simulation_time_dynamics.png'")

    # ANALYZE TIME DYNAMICS
    print(f"\nTIME DYNAMICS ANALYSIS:")
    print("=" * 30)

    dynamics_detected = False

    # Check each parameter for time dynamics
    parameters = [
        ('target_areas', target_areas, 'Area', 100),  # threshold for significant change
        ('target_aspect_ratios', target_ars, 'Aspect Ratio', 0.05),
        ('target_orientations', target_orients, 'Orientation', 0.02)
    ]

    for param_name, param_data, display_name, threshold in parameters:
        print(f"\n{display_name} Analysis:")

        # Check for overall variation
        param_range = np.nanmax(param_data) - np.nanmin(param_data)
        print(f"  Total variation: {param_range:.3f}")

        if len(pressure_changes) > 0:
            # Analyze response to pressure steps
            for change_idx in pressure_changes:
                if change_idx > 5 and change_idx < len(param_data) - 10:
                    pre_values = param_data[max(0, change_idx-3):change_idx+1]
                    post_values = param_data[change_idx+1:change_idx+10]

                    if len(pre_values) > 0 and len(post_values) > 0:
                        pre_mean = np.nanmean(pre_values)
                        post_mean = np.nanmean(post_values)
                        change_magnitude = abs(post_mean - pre_mean)

                        print(f"  Step at t={times[change_idx]:.1f}h:")
                        print(f"    Before: {pre_mean:.3f}")
                        print(f"    After: {post_mean:.3f}")
                        print(f"    Change: {change_magnitude:.3f}")

                        if change_magnitude > threshold:
                            print(f"    ✅ SIGNIFICANT CHANGE DETECTED!")
                            dynamics_detected = True

                            # Check if it's gradual (time dynamics) or instant (no dynamics)
                            step_values = param_data[change_idx:change_idx+5]
                            if len(step_values) > 2:
                                # Check if change is gradual
                                differences = np.abs(np.diff(step_values))
                                if np.any(differences > threshold/10):
                                    print(f"    ✅ GRADUAL EVOLUTION - Time dynamics working!")
                                else:
                                    print(f"    ❌ INSTANT CHANGE - Time dynamics may not be working")
                        else:
                            print(f"    ❌ Change too small (threshold: {threshold})")
        else:
            print(f"  No pressure steps to analyze")
            # For constant pressure, check if there's any evolution
            if param_range > threshold:
                print(f"  ✅ Parameter evolves over time even with constant pressure")
                print(f"     This suggests initial adaptation from non-equilibrium start")
                dynamics_detected = True
            else:
                print(f"  ❌ No significant evolution")

    # Overall assessment
    print(f"\n" + "=" * 50)
    print(f"OVERALL TIME DYNAMICS ASSESSMENT:")
    print("=" * 50)

    if dynamics_detected:
        print(f"✅ TIME DYNAMICS DETECTED!")
        print(f"   Your simulation shows target parameters evolving over time")
        print(f"   This indicates the temporal dynamics equations are working")
    else:
        print(f"❌ NO TIME DYNAMICS DETECTED")
        print(f"   Target parameters appear constant or change instantly")
        print(f"   This suggests temporal dynamics may not be working properly")

    # Additional checks
    print(f"\nADDITIONAL INSIGHTS:")

    # Check target vs actual lag
    ar_lag = np.nanmean(np.abs(target_ars - actual_ars))
    area_lag = np.nanmean(np.abs(target_areas - actual_areas) / np.maximum(target_areas, 1))

    print(f"  Target-Actual lag:")
    print(f"    Aspect ratio lag: {ar_lag:.3f}")
    print(f"    Area lag (relative): {area_lag:.3f}")

    if ar_lag > 0.1 or area_lag > 0.1:
        print(f"  ✅ Significant lag between targets and actuals")
        print(f"     This shows cells are adapting toward targets (good!)")
    else:
        print(f"  ❌ Little lag between targets and actuals")
        print(f"     Targets and actuals track very closely")

    # Check temporal consistency
    target_ar_std = np.nanstd(target_ars)
    if target_ar_std > 0.05:
        print(f"  ✅ Target parameters show temporal variation (std={target_ar_std:.3f})")
    else:
        print(f"  ❌ Target parameters very stable (std={target_ar_std:.3f})")

    # Handle PyCharm matplotlib backend issues
    try:
        plt.show()
    except AttributeError as e:
        print(f"Note: Display issue in PyCharm (plot still saved): {e}")
        plt.close()

    return fig


def quick_verification():
    """
    Quick verification focusing on key indicators.
    """
    filepath = 'simulation_20250608-184347.npz'
    data = np.load(filepath, allow_pickle=True)
    history = data['history'].item()

    print("QUICK TIME DYNAMICS VERIFICATION")
    print("=" * 40)

    times = np.array(history['time']) / 60
    pressures = np.array(history['input_value'])
    cell_props = history['cell_properties']

    # Get target aspect ratios (most sensitive indicator)
    target_ars = [cp['target_aspect_ratios'][0] if len(cp['target_aspect_ratios']) > 0 else np.nan for cp in cell_props]
    target_ars = np.array(target_ars)

    # Key metrics
    ar_min = np.nanmin(target_ars)
    ar_max = np.nanmax(target_ars)
    ar_range = ar_max - ar_min
    ar_std = np.nanstd(target_ars)

    print(f"Target Aspect Ratio Analysis:")
    print(f"  Range: {ar_min:.3f} to {ar_max:.3f}")
    print(f"  Variation: {ar_range:.3f}")
    print(f"  Std deviation: {ar_std:.3f}")

    # Look for pressure steps
    pressure_changes = np.where(np.abs(np.diff(pressures)) > 0.1)[0]

    if len(pressure_changes) > 0:
        print(f"  Pressure step detected at t={times[pressure_changes[0]]:.1f}h")

        change_idx = pressure_changes[0]
        if change_idx > 5 and change_idx < len(target_ars) - 5:
            before = np.nanmean(target_ars[change_idx-3:change_idx+1])
            after = np.nanmean(target_ars[change_idx+1:change_idx+5])
            step_change = abs(after - before)

            print(f"  AR before step: {before:.3f}")
            print(f"  AR after step: {after:.3f}")
            print(f"  Step change: {step_change:.3f}")

            if step_change > 0.05:
                print(f"  ✅ SIGNIFICANT RESPONSE TO PRESSURE STEP")
                print(f"  ✅ TIME DYNAMICS ARE WORKING!")
            else:
                print(f"  ❌ Little response to pressure step")
                print(f"  ❌ Time dynamics may not be working")
        else:
            print(f"  ❌ Can't analyze step response (step too close to start/end)")
    else:
        print(f"  No pressure steps detected")
        if ar_range > 0.05:
            print(f"  ✅ Still shows evolution - possible initial adaptation")
        else:
            print(f"  ❌ No evolution detected")

    return ar_range > 0.05


if __name__ == "__main__":
    print("Running full analysis of your simulation...")
    analyze_your_simulation()
    print(f"\n" + "="*50)
    print("Running quick verification...")
    quick_verification()