"""
Extract and verify cell orientations from saved simulation data or frame data.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def load_and_analyze_simulation_file(filepath):
    """
    Load a saved simulation file and extract orientation data.
    """
    print(f"Loading simulation data from: {filepath}")

    try:
        # Try loading as NPZ file first
        data = np.load(filepath, allow_pickle=True)
        history = data['history'].item()

        print(f"✅ Loaded NPZ file with {len(history['time'])} time points")
        return analyze_orientation_from_history(history)

    except Exception as e:
        print(f"❌ Error loading NPZ file: {e}")
        return None


def analyze_orientation_from_history(history):
    """
    Extract orientation data from simulation history.
    """
    print("\nANALYZING ORIENTATIONS FROM HISTORY DATA")
    print("=" * 50)

    # Check if we have cell properties data
    if 'cell_properties' not in history:
        print("❌ No cell_properties found in history")
        return None

    # Get the last time point for analysis
    cell_props = history['cell_properties']
    final_state = cell_props[-1]  # Last time point

    print(f"Final time point data:")
    print(f"  Time points available: {len(cell_props)}")
    print(f"  Final time: {history['time'][-1]:.1f} minutes")

    # Extract orientation data
    orientations_data = {}

    # Check what orientation data is available
    available_keys = final_state.keys()
    print(f"  Available keys: {list(available_keys)}")

    # Extract different types of orientation data
    if 'orientations' in final_state:
        raw_orientations = final_state['orientations']
        orientations_data['raw'] = raw_orientations
        print(f"  Raw orientations: {len(raw_orientations)} cells")
        print(f"    Mean: {np.mean(raw_orientations):.1f}°")
        print(f"    Range: {np.min(raw_orientations):.1f}° to {np.max(raw_orientations):.1f}°")

    if 'target_orientations_degrees' in final_state:
        target_orientations = final_state['target_orientations_degrees']
        orientations_data['target'] = target_orientations
        print(f"  Target orientations: {len(target_orientations)} cells")
        print(f"    Mean: {np.mean(target_orientations):.1f}°")

    if 'target_orientations' in final_state:
        target_orientations_rad = final_state['target_orientations']
        target_orientations_deg = [np.degrees(angle) for angle in target_orientations_rad]
        orientations_data['target_from_rad'] = target_orientations_deg
        print(f"  Target orientations (from rad): {len(target_orientations_deg)} cells")
        print(f"    Mean: {np.mean(target_orientations_deg):.1f}°")

    return orientations_data


def create_comprehensive_orientation_analysis(orientations_data):
    """
    Create comprehensive plots to analyze all orientation data.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    colors = ['blue', 'red', 'green', 'orange']

    row = 0
    col = 0

    for data_type, orientations in orientations_data.items():
        if row >= 2:  # Only plot first 6 datasets
            break

        ax = axes[row, col]

        # 1. Raw histogram
        ax.hist(orientations, bins=30, alpha=0.7, color=colors[len(orientations_data) % len(colors)],
                edgecolor='black', label=data_type)
        ax.set_xlabel('Orientation (degrees)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{data_type.title()} Orientations Distribution')
        ax.axvline(np.mean(orientations), color='red', linestyle='--',
                   label=f'Mean: {np.mean(orientations):.1f}°')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Move to next subplot
        col += 1
        if col >= 3:
            col = 0
            row += 1

    # If we have both raw and target, create comparison plots
    if 'raw' in orientations_data and ('target' in orientations_data or 'target_from_rad' in orientations_data):
        target_key = 'target' if 'target' in orientations_data else 'target_from_rad'

        # Flow alignment comparison
        if row < 2 and col < 3:
            ax = axes[row, col]

            # Convert both to flow alignment (0-90°)
            raw_flow = convert_to_flow_alignment(orientations_data['raw'])
            target_flow = convert_to_flow_alignment(orientations_data[target_key])

            ax.hist(target_flow, bins=20, alpha=0.6, color='lightblue',
                    label=f'Target (mean: {np.mean(target_flow):.1f}°)', range=(0, 90))
            ax.hist(raw_flow, bins=20, alpha=0.6, color='lightcoral',
                    label=f'Actual (mean: {np.mean(raw_flow):.1f}°)', range=(0, 90))

            ax.set_xlabel('Flow Alignment Angle (degrees)')
            ax.set_ylabel('Frequency')
            ax.set_title('Target vs Actual Flow Alignment')
            ax.set_xlim(0, 90)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add reference lines
            ax.axvline(0, color='green', linestyle='-', alpha=0.8, linewidth=2, label='Perfect alignment')
            ax.axvline(45, color='orange', linestyle='--', alpha=0.6)
            ax.axvline(90, color='red', linestyle='-', alpha=0.8, linewidth=2, label='Perpendicular')

    plt.tight_layout()
    return fig


def convert_to_flow_alignment(orientations):
    """
    Convert orientations to flow alignment angles (0-90°).
    """
    flow_alignment = []
    for angle in orientations:
        angle_180 = angle % 180  # Normalize to 0-180°
        alignment_angle = min(angle_180, 180 - angle_180)  # Take acute angle
        flow_alignment.append(alignment_angle)
    return flow_alignment


def analyze_time_evolution_of_orientations(history):
    """
    Analyze how orientations change over time.
    """
    print("\nTIME EVOLUTION OF ORIENTATIONS")
    print("=" * 40)

    cell_props = history['cell_properties']
    times = history['time']

    # Extract orientations at different time points
    time_points_to_check = [0, len(times) // 4, len(times) // 2, 3 * len(times) // 4, -1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, time_idx in enumerate(time_points_to_check):
        if i >= len(axes):
            break

        props = cell_props[time_idx]
        time_val = times[time_idx]

        if 'orientations' in props:
            orientations = props['orientations']
            flow_alignment = convert_to_flow_alignment(orientations)

            ax = axes[i]
            ax.hist(flow_alignment, bins=15, alpha=0.7, range=(0, 90),
                    color='skyblue', edgecolor='black')
            ax.set_title(f't = {time_val:.0f} min ({time_val / 60:.1f}h)\nMean: {np.mean(flow_alignment):.1f}°')
            ax.set_xlabel('Flow Alignment (degrees)')
            ax.set_ylabel('Frequency')
            ax.set_xlim(0, 90)
            ax.grid(True, alpha=0.3)

            # Add mean line
            ax.axvline(np.mean(flow_alignment), color='red', linestyle='--', alpha=0.8)

    # Use last subplot for time series of mean orientation
    if len(time_points_to_check) < len(axes):
        ax = axes[-1]

        mean_orientations = []
        for props in cell_props:
            if 'orientations' in props:
                orientations = props['orientations']
                flow_alignment = convert_to_flow_alignment(orientations)
                mean_orientations.append(np.mean(flow_alignment))
            else:
                mean_orientations.append(np.nan)

        time_hours = np.array(times) / 60
        ax.plot(time_hours, mean_orientations, 'b-', linewidth=2)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Mean Flow Alignment (degrees)')
        ax.set_title('Mean Orientation Evolution')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 90)

    plt.tight_layout()
    return fig


def main():
    """
    Main function to analyze orientation data.
    """
    print("ORIENTATION DATA EXTRACTION AND VERIFICATION")
    print("=" * 60)

    # Example usage - replace with your actual file path
    filepath = "simulation_20250611-143910.npz"  # Replace with your file

    if os.path.exists(filepath):
        orientations_data = load_and_analyze_simulation_file(filepath)

        if orientations_data:
            # Create analysis plots
            fig1 = create_comprehensive_orientation_analysis(orientations_data)
            fig1.suptitle('Comprehensive Orientation Analysis', fontsize=16)

            # Save the analysis
            plt.figure(fig1.number)
            plt.savefig('orientation_verification_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Analysis saved as 'orientation_verification_analysis.png'")

            # Show the plot
            plt.show()
        else:
            print("❌ Could not extract orientation data")
    else:
        print(f"❌ File not found: {filepath}")
        print("Please update the filepath to point to your simulation data file")


# Standalone function to quickly check a specific file
def quick_orientation_check(filepath):
    """
    Quick check of orientation data in a file.
    """
    try:
        data = np.load(filepath, allow_pickle=True)
        history = data['history'].item()

        # Get final orientations
        final_props = history['cell_properties'][-1]

        if 'orientations' in final_props:
            orientations = final_props['orientations']
            flow_alignment = convert_to_flow_alignment(orientations)

            print(f"QUICK ORIENTATION CHECK")
            print(f"File: {filepath}")
            print(f"Final time: {history['time'][-1]:.1f} minutes")
            print(f"Number of cells: {len(orientations)}")
            print(f"Raw orientations - Mean: {np.mean(orientations):.1f}°, Std: {np.std(orientations):.1f}°")
            print(f"Flow alignment - Mean: {np.mean(flow_alignment):.1f}°, Std: {np.std(flow_alignment):.1f}°")
            print(f"Flow alignment range: {np.min(flow_alignment):.1f}° to {np.max(flow_alignment):.1f}°")

            return orientations, flow_alignment
        else:
            print(f"❌ No orientation data found in {filepath}")
            return None, None

    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return None, None


if __name__ == "__main__":
    main()