"""
Toy code to test histogram binning for flow alignment analysis.
Tests what happens with angles 10°, 170°, 190°, 350° using 0-90° bins.
"""
import numpy as np
import matplotlib.pyplot as plt


def test_current_binning():
    """Test the current binning approach with problematic angles."""

    print("🧪 TESTING CURRENT BINNING APPROACH")
    print("=" * 50)

    # Create test data: 10 values each of 10°, 170°, 190°, 350°
    orientations_deg = (
            [10] * 10 +  # Should be 10° alignment
            [170] * 10 +  # Should be 10° alignment (170° = 180° - 10°)
            [190] * 10 +  # Should be 10° alignment (190° = 180° + 10°)
            [350] * 10  # Should be 10° alignment (350° = 360° - 10°)
    )

    print(f"Input angles: {set(orientations_deg)}")
    print(f"Total angles: {len(orientations_deg)}")
    print(f"Expected: All should represent 10° flow alignment")

    # Use the CURRENT binning approach (0-90° only)
    bins = np.linspace(0, 90, 19)  # 0-90 degrees
    hist, bin_edges = np.histogram(orientations_deg, bins=bins)

    print(f"\n📊 CURRENT BINNING RESULTS:")
    print(f"Histogram bins: {len(bins) - 1} bins from 0° to 90°")
    print(f"Total values captured: {np.sum(hist)} / {len(orientations_deg)}")
    print(f"Values lost: {len(orientations_deg) - np.sum(hist)}")

    # Show where each angle lands
    print(f"\n🎯 WHERE EACH ANGLE LANDS:")
    for angle in [10, 170, 190, 350]:
        # Find which bin this angle falls into
        bin_idx = np.digitize([angle], bins) - 1
        if 0 <= bin_idx[0] < len(hist):
            bin_center = (bins[bin_idx[0]] + bins[bin_idx[0] + 1]) / 2
            print(f"  {angle}° → Bin {bin_idx[0]} (center: {bin_center:.1f}°)")
        else:
            print(f"  {angle}° → OUTSIDE BINS (lost!)")

    # Show histogram values
    print(f"\n📈 HISTOGRAM VALUES:")
    for i, count in enumerate(hist):
        if count > 0:
            bin_center = (bins[i] + bins[i + 1]) / 2
            print(f"  Bin {i} ({bin_center:.1f}°): {count} values")

    return hist, bins, orientations_deg


def test_corrected_binning():
    """Test the corrected binning approach that converts to flow alignment."""

    print(f"\n" + "=" * 50)
    print("🔧 TESTING CORRECTED BINNING APPROACH")
    print("=" * 50)

    # Same test data
    orientations_deg = (
            [10] * 10 + [170] * 10 + [190] * 10 + [350] * 10
    )

    print(f"Input angles: {set(orientations_deg)}")

    # CORRECTED: Convert to flow alignment angles (0-90°)
    alignment_angles = []
    for angle in orientations_deg:
        # Convert any angle to alignment with flow (0° = aligned, 90° = perpendicular)
        angle_180 = angle % 180  # Normalize to 0-180° range
        alignment_angle = min(angle_180, 180 - angle_180)  # Take acute angle
        alignment_angles.append(alignment_angle)

    print(f"After conversion: {set(alignment_angles)}")
    print(f"All converted to: {alignment_angles[0]}° (flow alignment)")

    # Use same binning
    bins = np.linspace(0, 90, 19)
    hist, bin_edges = np.histogram(alignment_angles, bins=bins)

    print(f"\n📊 CORRECTED BINNING RESULTS:")
    print(f"Total values captured: {np.sum(hist)} / {len(alignment_angles)}")
    print(f"Values lost: {len(alignment_angles) - np.sum(hist)}")

    # Show where angles land after conversion
    print(f"\n🎯 WHERE EACH ANGLE LANDS AFTER CONVERSION:")
    for angle in [10, 170, 190, 350]:
        converted = abs(angle) % 90
        bin_idx = np.digitize([converted], bins) - 1
        if 0 <= bin_idx[0] < len(hist):
            bin_center = (bins[bin_idx[0]] + bins[bin_idx[0] + 1]) / 2
            print(f"  {angle}° → {converted}° → Bin {bin_idx[0]} (center: {bin_center:.1f}°)")

    # Show histogram values
    print(f"\n📈 HISTOGRAM VALUES:")
    for i, count in enumerate(hist):
        if count > 0:
            bin_center = (bins[i] + bins[i + 1]) / 2
            print(f"  Bin {i} ({bin_center:.1f}°): {count} values")

    return hist, bins, alignment_angles


def visualize_both_approaches():
    """Create visual comparison of both approaches."""

    # Test both approaches
    hist1, bins1, data1 = test_current_binning()
    hist2, bins2, data2 = test_corrected_binning()

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Current approach
    bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
    bars1 = ax1.bar(bin_centers1, hist1, width=4, alpha=0.7, color='red', edgecolor='black')
    ax1.set_title('❌ Current Approach (BROKEN)\nValues Outside 0-90° Are Lost', fontsize=12)
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Count')
    ax1.set_xlim(0, 90)
    ax1.grid(True, alpha=0.3)

    # Add text showing total captured
    ax1.text(0.7, 0.9, f'Captured: {np.sum(hist1)}/40\nLost: {40 - np.sum(hist1)}',
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Corrected approach
    bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
    bars2 = ax2.bar(bin_centers2, hist2, width=4, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('✅ Corrected Approach (WORKS)\nAll Angles Converted to Flow Alignment', fontsize=12)
    ax2.set_xlabel('Flow Alignment Angle (degrees)')
    ax2.set_ylabel('Count')
    ax2.set_xlim(0, 90)
    ax2.grid(True, alpha=0.3)

    # Add text showing total captured
    ax2.text(0.7, 0.9, f'Captured: {np.sum(hist2)}/40\nLost: {40 - np.sum(hist2)}',
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add reference lines
    for ax in [ax1, ax2]:
        ax.axvline(0, color='blue', linestyle='--', alpha=0.5, label='Perfect alignment')
        ax.axvline(90, color='orange', linestyle='--', alpha=0.5, label='Perpendicular')
        ax.legend()

    plt.tight_layout()
    plt.savefig('histogram_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'histogram_comparison.png'")

    return fig


def main():
    """Run all tests and create visualization."""

    print("🧪 TESTING HISTOGRAM BINNING FOR FLOW ALIGNMENT")
    print("Testing angles: 10°, 170°, 190°, 350°")
    print("(All should represent 10° flow alignment)")

    # Run tests
    test_current_binning()
    test_corrected_binning()

    # Create visualization
    fig = visualize_both_approaches()

    print(f"\n" + "=" * 60)
    print("🎯 SUMMARY:")
    print("=" * 60)
    print("❌ Current approach: Only captures angles already in 0-90° range")
    print("✅ Corrected approach: Converts all angles to flow alignment first")
    print("📊 Recommendation: Use the corrected approach for flow studies")

    # Optional: Show plot (comment out if running headless)
    # plt.show()


if __name__ == "__main__":
    main()