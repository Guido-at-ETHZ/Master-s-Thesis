"""
Script to check hole data in simulation results.
This will examine your simulation_20250613-113514.npz file and show hole-related data.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def examine_simulation_file(filepath):
    """
    Examine the contents of a simulation .npz file and check for hole data.

    Parameters:
        filepath: Path to the .npz file
    """
    print("🔍 Examining simulation file:", filepath)
    print("=" * 60)

    # Load the .npz file
    try:
        data = np.load(filepath, allow_pickle=True)
        print("✅ File loaded successfully!")
        print(f"📁 File contains {len(data.files)} arrays/objects")
        print(f"📋 Available keys: {list(data.files)}")
        print()

        # Check for history data
        if 'history' in data.files:
            history = data['history'].item()  # Convert back to dict
            print("📊 History data found!")
            print(f"   Number of time points: {len(history.get(list(history.keys())[0], []))}")
            print(f"   Available metrics: {len(history.keys())} total")
            print()

            # Check specifically for hole-related data
            hole_keys = [key for key in history.keys() if 'hole' in key.lower()]
            print("🕳️  HOLE-RELATED DATA:")
            print("-" * 30)

            if hole_keys:
                for key in hole_keys:
                    values = history[key]
                    if isinstance(values, np.ndarray) and len(values) > 0:
                        print(f"✅ {key}:")
                        print(f"   Shape: {values.shape}")
                        print(f"   Type: {type(values[0])}")
                        if np.issubdtype(values.dtype, np.number):
                            print(f"   Range: {np.min(values):.3f} to {np.max(values):.3f}")
                            print(f"   Final value: {values[-1]:.3f}")
                        else:
                            print(f"   Sample values: {values[:3]}")
                        print()
                    else:
                        print(f"❌ {key}: Empty or invalid data")
                        print()
            else:
                print("❌ No hole-related keys found in history!")
                print("   This might mean:")
                print("   1. Holes were not enabled in the simulation")
                print("   2. Hole data wasn't saved properly")
                print("   3. Different key names were used")
                print()

                # Let's check all keys to see if there might be hole data under different names
                print("🔍 All available keys (checking for potential hole data):")
                for key in sorted(history.keys()):
                    print(f"   📝 {key}")
                print()

            # If we found hole data, let's analyze it
            if hole_keys:
                print("📈 HOLE DATA ANALYSIS:")
                print("-" * 30)
                analyze_hole_data(history, hole_keys)

        else:
            print("❌ No 'history' key found in file!")
            print("   Available keys:", list(data.files))

        # Check config and other data
        print("\n📋 OTHER DATA IN FILE:")
        print("-" * 30)
        for key in data.files:
            if key != 'history':
                item = data[key]
                if hasattr(item, 'item'):
                    item = item.item()
                print(f"📝 {key}: {type(item)}")
                if isinstance(item, dict):
                    print(f"   Subkeys: {list(item.keys())}")
                print()

        data.close()

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def analyze_hole_data(history, hole_keys):
    """Analyze hole data if found."""

    # Get time data
    if 'time' in history:
        time_data = history['time']
        print(f"⏰ Time range: {time_data[0]:.1f} to {time_data[-1]:.1f} minutes")
        print(f"   ({time_data[-1] / 60:.1f} hours)")
        print()

    # Analyze each hole metric
    for key in hole_keys:
        values = history[key]
        if isinstance(values, np.ndarray) and len(values) > 0:

            if key == 'hole_count':
                print(f"🕳️  HOLE COUNT ANALYSIS:")
                print(f"   Maximum holes: {np.max(values)}")
                print(f"   Final holes: {values[-1]}")
                print(f"   Average holes: {np.mean(values):.2f}")

                # Count time with holes
                time_with_holes = np.sum(values > 0)
                print(
                    f"   Time points with holes: {time_with_holes}/{len(values)} ({100 * time_with_holes / len(values):.1f}%)")
                print()

            elif key == 'hole_area_fraction':
                print(f"📐 HOLE AREA FRACTION ANALYSIS:")
                print(f"   Maximum area fraction: {np.max(values):.4f} ({100 * np.max(values):.2f}%)")
                print(f"   Final area fraction: {values[-1]:.4f} ({100 * values[-1]:.2f}%)")
                print(f"   Average area fraction: {np.mean(values):.4f} ({100 * np.mean(values):.2f}%)")
                print()

            elif key == 'holes':
                print(f"🔍 INDIVIDUAL HOLES DATA:")
                print(f"   Data type: {type(values[0]) if len(values) > 0 else 'None'}")

                # Check if we have detailed hole information
                non_empty_holes = [h for h in values if (isinstance(h, list) and len(h) > 0)]
                if non_empty_holes:
                    print(f"   Time points with detailed hole data: {len(non_empty_holes)}")
                    print(f"   Sample hole data structure: {type(non_empty_holes[0][0])}")
                else:
                    print(f"   No detailed hole data found (mostly empty lists)")
                print()


def plot_hole_evolution(filepath, save_plot=True):
    """
    Create plots showing hole evolution over time.

    Parameters:
        filepath: Path to the .npz file
        save_plot: Whether to save the plot
    """
    try:
        data = np.load(filepath, allow_pickle=True)
        history = data['history'].item()

        # Check if we have hole data and time data
        if 'hole_count' in history and 'time' in history:
            time_data = history['time']
            hole_count = history['hole_count']

            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Plot hole count
            axes[0].plot(time_data, hole_count, 'b-', linewidth=2, marker='o', markersize=4)
            axes[0].set_ylabel('Number of Holes')
            axes[0].set_title('Hole Evolution Over Time')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(bottom=0)

            # Plot hole area fraction if available
            if 'hole_area_fraction' in history:
                hole_area_fraction = history['hole_area_fraction']
                axes[1].plot(time_data, hole_area_fraction * 100, 'r-', linewidth=2, marker='s', markersize=4)
                axes[1].set_ylabel('Hole Area Fraction (%)')
                axes[1].grid(True, alpha=0.3)
                axes[1].set_ylim(bottom=0)
            else:
                axes[1].text(0.5, 0.5, 'No hole area fraction data available',
                             ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_ylabel('Hole Area Fraction (%)')

            axes[1].set_xlabel('Time (minutes)')

            # Add text summary
            max_holes = np.max(hole_count)
            final_holes = hole_count[-1]
            avg_holes = np.mean(hole_count)

            summary_text = f"Max holes: {max_holes}\nFinal holes: {final_holes}\nAverage: {avg_holes:.1f}"
            axes[0].text(0.02, 0.98, summary_text, transform=axes[0].transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            if save_plot:
                plot_filename = filepath.replace('.npz', '_hole_evolution.png')
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"📊 Plot saved as: {plot_filename}")

            plt.show()

        else:
            print("❌ Cannot create plot: Missing hole_count or time data")

        data.close()

    except Exception as e:
        print(f"❌ Error creating plot: {e}")


def check_hole_configuration(filepath):
    """Check what hole configuration was used in the simulation."""
    try:
        data = np.load(filepath, allow_pickle=True)

        if 'config_params' in data.files:
            config = data['config_params'].item()
            print("⚙️  HOLE CONFIGURATION:")
            print("-" * 30)

            # Look for hole-related config parameters
            hole_config_keys = [key for key in config.keys() if 'hole' in key.lower()]

            if hole_config_keys:
                for key in hole_config_keys:
                    print(f"   {key}: {config[key]}")
            else:
                print("   No hole configuration parameters found in saved config")
                print("   This might mean holes were not enabled or config wasn't fully saved")

            print(f"\n   Other config parameters: {list(config.keys())}")

        else:
            print("❌ No config_params found in file")

        data.close()

    except Exception as e:
        print(f"❌ Error checking configuration: {e}")


def create_hole_summary_table(filepath):
    """Create a summary table of hole statistics."""
    try:
        data = np.load(filepath, allow_pickle=True)
        history = data['history'].item()

        if 'hole_count' in history and 'time' in history:
            # Create DataFrame with hole data
            df_data = {'Time (min)': history['time']}

            # Add hole metrics
            hole_metrics = ['hole_count', 'hole_area_fraction']
            for metric in hole_metrics:
                if metric in history:
                    df_data[metric.replace('_', ' ').title()] = history[metric]

            # Add additional metrics if available
            if 'cell_counts' in history:
                # This might be a more complex structure
                cell_counts = history['cell_counts']
                if len(cell_counts) > 0 and isinstance(cell_counts[0], dict):
                    df_data['Total Cells'] = [cc.get('total', 0) for cc in cell_counts]

            df = pd.DataFrame(df_data)

            # Show summary statistics
            print("📊 HOLE STATISTICS SUMMARY:")
            print("-" * 40)
            print(df.describe())

            # Show time points where holes exist
            if 'Hole Count' in df.columns:
                holes_present = df[df['Hole Count'] > 0]
                if not holes_present.empty:
                    print(f"\n🕳️  HOLES WERE PRESENT:")
                    print(f"   First appearance: {holes_present.iloc[0]['Time (min)']:.1f} min")
                    print(f"   Last appearance: {holes_present.iloc[-1]['Time (min)']:.1f} min")
                    print(f"   Duration with holes: {len(holes_present)} time points")
                else:
                    print("\n❌ No holes were present during the simulation")

        data.close()
        return df if 'df' in locals() else None

    except Exception as e:
        print(f"❌ Error creating summary table: {e}")
        return None


# Main execution
if __name__ == "__main__":
    # Replace this with the actual path to your file
    filepath = "simulation_20250613-113514.npz"

    print("🔍 COMPREHENSIVE HOLE DATA ANALYSIS")
    print("=" * 60)

    # 1. Examine file contents
    examine_simulation_file(filepath)

    # 2. Check configuration
    print("\n")
    check_hole_configuration(filepath)

    # 3. Create summary table
    print("\n")
    df = create_hole_summary_table(filepath)

    # 4. Create plots
    print("\n")
    plot_hole_evolution(filepath)

    print("\n✅ Analysis complete!")
    print("\nTo run this analysis on your file:")
    print("1. Make sure the file 'simulation_20250613-113514.npz' is in your current directory")
    print("2. Run this script")
    print("3. Check the generated plots and summary statistics")