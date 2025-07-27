import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def analyze_simulation_orientations(npz_file_path):
    """
    Comprehensive analysis of cell orientations in simulation data.
    Determines if orientations change despite constant visual appearance.
    """

    print("=" * 60)
    print("CELL ORIENTATION ANALYSIS")
    print("=" * 60)

    # Load the NPZ file
    try:
        data = np.load(npz_file_path, allow_pickle=True)
        print(f"✅ Successfully loaded: {npz_file_path}")
        print(f"📁 File size: {os.path.getsize(npz_file_path) / 1024:.1f} KB")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # Examine the file structure
    print(f"\n📊 Data structure:")
    for key in data.keys():
        item = data[key]
        if hasattr(item, 'shape'):
            print(f"  {key}: shape {item.shape}, dtype {item.dtype}")
        else:
            print(f"  {key}: {type(item)} (length: {len(item) if hasattr(item, '__len__') else 'N/A'})")

    # Extract history data
    if 'history' in data:
        history = data['history']
        print(f"\n⏱️  Simulation history: {len(history)} time points")
    else:
        print("❌ No 'history' key found in data")
        print("Available keys:", list(data.keys()))
        return

    # Also check frame_data if history doesn't have what we need
    frame_data = None
    if 'frame_data' in data:
        frame_data = data['frame_data']
        print(f"📺 Frame data available: {len(frame_data)} frames")

        # Look at frame data structure
        if len(frame_data) > 0:
            print(f"   Frame 0 type: {type(frame_data[0])}")
            if hasattr(frame_data[0], 'keys'):
                print(f"   Frame 0 keys: {list(frame_data[0].keys())}")

    # Check time_points data
    if 'time_points' in data:
        time_points = data['time_points']
        print(f"⏰ Time points available: {len(time_points)} points")
        print(f"   Time range: {time_points[0]:.2f} - {time_points[-1]:.2f}")

    # Look at final_stats for additional insights
    if 'final_stats' in data:
        final_stats = data['final_stats'].item() if hasattr(data['final_stats'], 'item') else data['final_stats']
        print(f"📊 Final stats keys: {list(final_stats.keys()) if hasattr(final_stats, 'keys') else 'Not a dict'}")

    # Look at config_params for simulation setup
    if 'config_params' in data:
        config_params = data['config_params'].item() if hasattr(data['config_params'], 'item') else data[
            'config_params']
        print(
            f"⚙️  Config params keys: {list(config_params.keys()) if hasattr(config_params, 'keys') else 'Not a dict'}")
        if hasattr(config_params, 'keys') and 'simulation_duration' in config_params:
            print(f"   Simulation duration: {config_params['simulation_duration']} minutes")

    # Analyze orientation changes over time
    print(f"\n🔍 ORIENTATION ANALYSIS:")
    print("-" * 40)

    orientation_data = []
    pressure_data = []
    time_data = []

    # Debug: Look at first few entries to understand structure
    print(f"\n🔍 DEBUGGING DATA STRUCTURE:")
    for i in range(min(3, len(history))):
        state = history[i]
        print(f"  Time point {i}: type = {type(state)}")
        if hasattr(state, 'item'):
            try:
                content = state.item()
                print(f"    After .item(): {type(content)}")
                print(f"    Keys: {list(content.keys()) if hasattr(content, 'keys') else 'No keys'}")
            except:
                print(f"    .item() failed, treating as direct dict")
                content = state
        else:
            content = state
            print(f"    Direct access: {type(content)}")
            print(f"    Keys: {list(content.keys()) if hasattr(content, 'keys') else 'No keys'}")

        # Look for input values with different possible names
        if hasattr(content, 'keys'):
            pressure_keys = [k for k in content.keys() if
                             'pressure' in k.lower() or 'input' in k.lower() or 'shear' in k.lower()]
            print(f"    Potential pressure keys: {pressure_keys}")

    # Extract data from each time point
    for i, state in enumerate(history):
        try:
            # Handle different data formats
            if hasattr(state, 'item'):
                try:
                    time_point = state.item()  # Extract from numpy array
                except:
                    time_point = state  # Already a dictionary
            else:
                time_point = state  # Already a dictionary

            # Get time
            sim_time = time_point.get('time', i)
            time_data.append(sim_time)

            # Get pressure/input value - try multiple possible keys
            pressure = 0
            for key in ['input_value', 'pressure', 'shear_stress', 'applied_pressure', 'flow_pressure']:
                if key in time_point:
                    pressure = time_point[key]
                    break
            pressure_data.append(pressure)

            # Debug first few entries
            if i < 3:
                print(f"  Entry {i}: time={sim_time}, pressure={pressure}")

            # Get cell orientations
            if 'cell_properties' in time_point:
                cell_props = time_point['cell_properties']

                # Debug cell properties structure on first entry
                if i == 0:
                    print(
                        f"    Cell properties keys: {list(cell_props.keys()) if hasattr(cell_props, 'keys') else 'Not a dict'}")

                # Try multiple possible orientation keys
                orientations = None
                for key in ['orientations', 'actual_orientations', 'cell_orientations', 'angles']:
                    if key in cell_props:
                        orientations = cell_props[key]
                        if i == 0:
                            print(
                                f"    Found orientations under key '{key}': {len(orientations) if orientations else 0} cells")
                        break

                if orientations is not None:
                    orientation_data.append(orientations)
                    if i == 0:
                        print(f"  📐 Found {len(orientations)} cell orientations")
                        print(f"  📊 Sample orientations: {orientations[:5] if len(orientations) > 0 else 'None'}")
                else:
                    if i == 0:
                        print(f"  ⚠️  No orientations found in cell_properties")
                    orientation_data.append([])
            else:
                if i == 0:
                    print(f"  ⚠️  No cell_properties at time point {i}")
                orientation_data.append([])

        except Exception as e:
            if i < 5:  # Only show first few errors
                print(f"  ❌ Error processing time point {i}: {e}")
            elif i == 5:
                print(f"  ... (suppressing further error messages)")
            orientation_data.append([])
            pressure_data.append(0)
            time_data.append(i)

    # Convert to numpy arrays for analysis
    time_data = np.array(time_data)
    pressure_data = np.array(pressure_data)

    print(f"\n📈 SIMULATION SUMMARY:")
    print(f"  ⏱️  Time range: {time_data[0]:.1f} - {time_data[-1]:.1f} minutes")
    print(f"  🔘 Pressure range: {min(pressure_data):.2f} - {max(pressure_data):.2f} Pa")
    print(f"  📊 Data points: {len(orientation_data)}")

    # Analyze orientation changes
    if len(orientation_data) > 0 and len(orientation_data[0]) > 0:

        print(f"\n🔄 ORIENTATION CHANGE ANALYSIS:")
        print("-" * 40)

        # Calculate statistics for each time point
        mean_orientations = []
        std_orientations = []

        for orientations in orientation_data:
            if len(orientations) > 0:
                mean_orientations.append(np.mean(orientations))
                std_orientations.append(np.std(orientations))
            else:
                mean_orientations.append(np.nan)
                std_orientations.append(np.nan)

        mean_orientations = np.array(mean_orientations)
        std_orientations = np.array(std_orientations)

        # Remove NaN values for analysis
        valid_indices = ~np.isnan(mean_orientations)
        valid_means = mean_orientations[valid_indices]
        valid_times = time_data[valid_indices]
        valid_pressures = pressure_data[valid_indices]

        if len(valid_means) > 1:
            # Calculate changes
            orientation_change = np.abs(valid_means[-1] - valid_means[0])
            max_orientation_change = np.max(np.abs(np.diff(valid_means)))

            print(f"  📐 Initial mean orientation: {valid_means[0]:.2f}°")
            print(f"  📐 Final mean orientation: {valid_means[-1]:.2f}°")
            print(f"  🔄 Total orientation change: {orientation_change:.2f}°")
            print(f"  ⚡ Maximum step change: {max_orientation_change:.2f}°")

            # Analyze individual cell changes
            if len(orientation_data[0]) > 0 and len(orientation_data[-1]) > 0:
                initial_orientations = np.array(orientation_data[0])
                final_orientations = np.array(orientation_data[-1])

                if len(initial_orientations) == len(final_orientations):
                    individual_changes = np.abs(final_orientations - initial_orientations)
                    print(f"  🏠 Individual cell changes:")
                    print(f"     Mean change per cell: {np.mean(individual_changes):.2f}°")
                    print(f"     Max change per cell: {np.max(individual_changes):.2f}°")
                    print(f"     Cells with >5° change: {np.sum(individual_changes > 5)}/{len(individual_changes)}")
                    print(f"     Cells with >10° change: {np.sum(individual_changes > 10)}/{len(individual_changes)}")

            # DIAGNOSIS
            print(f"\n🩺 DIAGNOSIS:")
            print("-" * 20)

            if max_orientation_change < 0.5:
                print("❌ SYSTEM ISSUE: Orientations are NOT changing despite pressure change")
                print("   → This suggests a problem with your simulation mechanics")
                print("   → Check event-driven system, transition controllers, or pressure thresholds")
            elif max_orientation_change < 2.0:
                print("⚠️  MINIMAL CHANGE: Very small orientation changes detected")
                print("   → Changes might be too small to visualize clearly")
                print("   → Consider checking transition time constants or response sensitivity")
            else:
                print("✅ ORIENTATIONS ARE CHANGING: This is likely a visualization issue")
                print("   → Your simulation mechanics are working")
                print("   → Check visualization angle conversion or plotting ranges")

            # Create diagnostic plots
            create_diagnostic_plots(valid_times, valid_pressures, valid_means,
                                    orientation_data, npz_file_path)

        else:
            print("❌ Insufficient valid data for analysis")

    else:
        print("❌ No orientation data found in simulation")

    # FALLBACK: Try analyzing frame_data if main analysis failed
    if (len(orientation_data) == 0 or all(
            len(orients) == 0 for orients in orientation_data)) and frame_data is not None:
        print(f"\n🔄 FALLBACK: Analyzing frame_data...")
        analyze_frame_data(frame_data, npz_file_path)

    print(f"\n" + "=" * 60)


def analyze_frame_data(frame_data, npz_file_path):
    """Analyze frame_data as fallback"""
    print("📺 Frame Data Analysis:")

    for i, frame in enumerate(frame_data[:3]):  # Look at first 3 frames
        if hasattr(frame, 'keys'):
            print(f"  Frame {i} keys: {list(frame.keys())}")

            # Look for orientation-related data
            for key in frame.keys():
                if 'orient' in key.lower() or 'angle' in key.lower() or 'cell' in key.lower():
                    data_sample = frame[key]
                    print(f"    {key}: {type(data_sample)}, shape: {getattr(data_sample, 'shape', 'N/A')}")
                    if hasattr(data_sample, '__len__') and len(data_sample) > 0:
                        print(
                            f"      Sample: {data_sample[:3] if hasattr(data_sample, '__getitem__') else 'Not indexable'}")
        else:
            print(f"  Frame {i}: {type(frame)}")

    # Try to create a time series from frame_data
    if len(frame_data) > 1:
        print(f"  📈 Frame data spans {len(frame_data)} time points")
        # Additional analysis could be added here if frame_data contains the orientation info


def create_diagnostic_plots(times, pressures, mean_orientations, all_orientations, npz_file_path):
    """Create diagnostic plots for orientation analysis"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Pressure over time
    axes[0, 0].plot(times, pressures, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (minutes)')
    axes[0, 0].set_ylabel('Pressure (Pa)')
    axes[0, 0].set_title('Pressure Evolution')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Mean orientation over time
    axes[0, 1].plot(times, mean_orientations, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (minutes)')
    axes[0, 1].set_ylabel('Mean Orientation (degrees)')
    axes[0, 1].set_title('Mean Cell Orientation Evolution')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Pressure vs Mean Orientation
    axes[1, 0].scatter(pressures, mean_orientations, c=times, cmap='viridis', s=50)
    axes[1, 0].set_xlabel('Pressure (Pa)')
    axes[1, 0].set_ylabel('Mean Orientation (degrees)')
    axes[1, 0].set_title('Pressure vs Orientation')
    axes[1, 0].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('Time (minutes)')

    # Plot 4: Orientation distribution at first and last time points
    if len(all_orientations) > 0:
        first_orients = all_orientations[0] if len(all_orientations[0]) > 0 else []
        last_orients = all_orientations[-1] if len(all_orientations[-1]) > 0 else []

        if len(first_orients) > 0:
            axes[1, 1].hist(first_orients, bins=20, alpha=0.6, label=f'Initial (t={times[0]:.1f}min)', color='blue')
        if len(last_orients) > 0:
            axes[1, 1].hist(last_orients, bins=20, alpha=0.6, label=f'Final (t={times[-1]:.1f}min)', color='red')

        axes[1, 1].set_xlabel('Cell Orientation (degrees)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Orientation Distributions')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = Path(npz_file_path).parent / f"orientation_analysis_{Path(npz_file_path).stem}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Diagnostic plots saved to: {output_path}")
    plt.show()


# Main execution
if __name__ == "__main__":
    # Replace with your actual file path
    npz_file_path = "/Users/guidoputignano/PycharmProjects/Master/endothelial_simulation/results_event_driven_20250723-134315/simulation_20250723-134428.npz"

    analyze_simulation_orientations(npz_file_path)