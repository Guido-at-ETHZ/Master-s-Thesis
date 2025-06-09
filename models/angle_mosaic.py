#!/usr/bin/env python3
"""
Standalone script to debug angle measurements and visualize the difference between
raw orientations vs flow alignment angles.

This helps understand what's happening in the endothelial simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import random


class DebugCell:
    """Simple cell class for testing angle measurements."""

    def __init__(self, x, y, orientation_degrees):
        self.x = x
        self.y = y
        self.raw_orientation_deg = orientation_degrees
        self.raw_orientation_rad = np.radians(orientation_degrees)

        # Calculate flow alignment angle (like in your simulation)
        self.flow_alignment_deg = self.calculate_flow_alignment()

    def calculate_flow_alignment(self):
        """Calculate flow alignment angle (0-90°) from raw orientation."""
        # Method 1: Using abs() % 90 (like in your _record_state)
        alignment_1 = abs(self.raw_orientation_deg) % 90

        # Method 2: Using min(angle_180, 180-angle_180) (like in histogram)
        angle_180 = abs(self.raw_orientation_deg) % 180
        alignment_2 = min(angle_180, 180 - angle_180)

        # Method 3: Actual angle from flow direction (0°)
        angle_from_flow = abs(self.raw_orientation_deg % 180)
        if angle_from_flow > 90:
            angle_from_flow = 180 - angle_from_flow
        alignment_3 = angle_from_flow

        print(
            f"Raw: {self.raw_orientation_deg:6.1f}° → Method1: {alignment_1:5.1f}°, Method2: {alignment_2:5.1f}°, Method3: {alignment_3:5.1f}°")

        return alignment_2  # Use method 2 (like your histogram)


def create_test_cells():
    """Create cells with known orientations for testing."""
    # Create a range of orientations to test
    test_orientations = [
        0,  # Perfect flow alignment
        15,  # Small angle
        30,  # Moderate angle
        45,  # 45 degrees
        60,  # Larger angle
        75,  # Near perpendicular
        90,  # Perpendicular
        105,  # Past perpendicular
        120,  # Backward diagonal
        135,  # Backward diagonal
        150,  # Almost opposite
        180,  # Opposite direction
        -30,  # Negative angles
        -45,
        -60,
        210,  # Past 180
        270,  # Straight down
        300,  # Upper right quadrant
    ]

    cells = []
    for i, orientation in enumerate(test_orientations):
        x = (i % 6) * 100 + 50  # Arrange in grid
        y = (i // 6) * 100 + 50
        cells.append(DebugCell(x, y, orientation))

    return cells


def visualize_angle_comparison(cells):
    """Create visualization comparing raw vs flow alignment angles."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Raw orientations (what you see visually)
    ax1.set_xlim(0, 600)
    ax1.set_ylim(0, 400)
    ax1.set_aspect('equal')
    ax1.set_title('Raw Orientations (What You See Visually)', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add flow direction reference
    ax1.arrow(50, 350, 100, 0, head_width=10, head_length=15,
              fc='green', ec='green', linewidth=3, alpha=0.8)
    ax1.text(100, 370, 'Flow Direction (0°)', ha='center', fontsize=12, weight='bold')

    for cell in cells:
        # Draw cell center
        circle = plt.Circle((cell.x, cell.y), 15, color='lightblue', alpha=0.7)
        ax1.add_patch(circle)

        # Draw orientation vector (raw angle)
        vector_length = 40
        dx = vector_length * np.cos(cell.raw_orientation_rad)
        dy = vector_length * np.sin(cell.raw_orientation_rad)

        arrow = FancyArrowPatch((cell.x, cell.y),
                                (cell.x + dx, cell.y + dy),
                                arrowstyle='->', mutation_scale=15,
                                color='red', linewidth=2)
        ax1.add_patch(arrow)

        # Label with raw angle
        ax1.text(cell.x, cell.y - 25, f'{cell.raw_orientation_deg:.0f}°',
                 ha='center', va='top', fontsize=9, weight='bold')

    # Plot 2: Flow alignment angles (what gets measured)
    ax2.set_xlim(0, 600)
    ax2.set_ylim(0, 400)
    ax2.set_aspect('equal')
    ax2.set_title('Flow Alignment Angles (What Gets Measured)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Add flow direction reference
    ax2.arrow(50, 350, 100, 0, head_width=10, head_length=15,
              fc='green', ec='green', linewidth=3, alpha=0.8)
    ax2.text(100, 370, 'Flow Direction (0°)', ha='center', fontsize=12, weight='bold')

    for cell in cells:
        # Draw cell center
        circle = plt.Circle((cell.x, cell.y), 15, color='lightblue', alpha=0.7)
        ax2.add_patch(circle)

        # Draw flow alignment vector (0-90° from flow direction)
        vector_length = 40
        flow_alignment_rad = np.radians(cell.flow_alignment_deg)
        dx = vector_length * np.cos(flow_alignment_rad)
        dy = vector_length * np.sin(flow_alignment_rad)

        arrow = FancyArrowPatch((cell.x, cell.y),
                                (cell.x + dx, cell.y + dy),
                                arrowstyle='->', mutation_scale=15,
                                color='blue', linewidth=2)
        ax2.add_patch(arrow)

        # Label with flow alignment angle
        ax2.text(cell.x, cell.y - 25, f'{cell.flow_alignment_deg:.0f}°',
                 ha='center', va='top', fontsize=9, weight='bold', color='blue')

    # Plot 3: Raw orientation histogram
    raw_angles = [cell.raw_orientation_deg for cell in cells]
    ax3.hist(raw_angles, bins=np.linspace(-180, 180, 25), alpha=0.7, color='red', edgecolor='black')
    ax3.set_title('Raw Orientations Distribution', fontsize=14)
    ax3.set_xlabel('Raw Orientation (degrees)')
    ax3.set_ylabel('Frequency')
    ax3.axvline(0, color='green', linewidth=3, alpha=0.8, label='Flow Direction')
    ax3.axvline(np.mean(raw_angles), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(raw_angles):.1f}°')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Flow alignment histogram
    flow_angles = [cell.flow_alignment_deg for cell in cells]
    ax4.hist(flow_angles, bins=np.linspace(0, 90, 19), alpha=0.7, color='blue', edgecolor='black')
    ax4.set_title('Flow Alignment Distribution', fontsize=14)
    ax4.set_xlabel('Flow Alignment Angle (degrees)')
    ax4.set_ylabel('Frequency')
    ax4.axvline(0, color='green', linewidth=3, alpha=0.8, label='Perfect Alignment')
    ax4.axvline(90, color='red', linewidth=3, alpha=0.8, label='Perpendicular')
    ax4.axvline(np.mean(flow_angles), color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(flow_angles):.1f}°')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 90)

    plt.tight_layout()
    plt.savefig('angle_debug_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    return raw_angles, flow_angles


def create_polar_comparison(raw_angles, flow_angles):
    """Create polar plots comparing raw vs flow alignment."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   subplot_kw=dict(projection='polar'))

    # Polar plot 1: Raw angles (full circle)
    raw_hist, raw_bins = np.histogram(raw_angles, bins=np.linspace(-180, 180, 25))
    raw_centers = np.radians((raw_bins[:-1] + raw_bins[1:]) / 2)
    width = np.radians(360 / 24)

    ax1.bar(raw_centers, raw_hist, width=width, alpha=0.7, color='red', edgecolor='darkred')
    ax1.set_title('Raw Orientations\n(Full Circle)', fontsize=14, pad=20)
    ax1.axvline(0, color='green', linewidth=3, alpha=0.8, label='Flow Direction')
    ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))

    # Polar plot 2: Flow alignment (quarter circle)
    flow_hist, flow_bins = np.histogram(flow_angles, bins=np.linspace(0, 90, 10))
    flow_centers = np.radians((flow_bins[:-1] + flow_bins[1:]) / 2)
    width_flow = np.radians(10)

    ax2.bar(flow_centers, flow_hist, width=width_flow, alpha=0.7, color='blue', edgecolor='darkblue')
    ax2.set_title('Flow Alignment\n(Quarter Circle)', fontsize=14, pad=20)
    ax2.set_thetamax(90)
    ax2.set_thetamin(0)
    ax2.set_thetagrids([0, 30, 60, 90], ['0°\n(Flow)', '30°', '60°', '90°\n(Perp)'])
    ax2.axvline(0, color='green', linewidth=3, alpha=0.8, label='Flow Direction')
    ax2.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))

    plt.tight_layout()
    plt.savefig('polar_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_angle_analysis(cells):
    """Print detailed angle analysis."""
    print("\n" + "=" * 80)
    print("ANGLE MEASUREMENT ANALYSIS")
    print("=" * 80)

    raw_angles = [cell.raw_orientation_deg for cell in cells]
    flow_angles = [cell.flow_alignment_deg for cell in cells]

    print(f"\nRaw Orientations:")
    print(f"  Range: {min(raw_angles):.1f}° to {max(raw_angles):.1f}°")
    print(f"  Mean: {np.mean(raw_angles):.1f}°")
    print(f"  Std: {np.std(raw_angles):.1f}°")

    print(f"\nFlow Alignment Angles:")
    print(f"  Range: {min(flow_angles):.1f}° to {max(flow_angles):.1f}°")
    print(f"  Mean: {np.mean(flow_angles):.1f}°")
    print(f"  Std: {np.std(flow_angles):.1f}°")

    print(f"\nKey Insight:")
    print(f"  Many raw angles appear < 45° visually")
    print(f"  But flow alignment mean is {np.mean(flow_angles):.1f}°")
    print(f"  This explains the discrepancy you observed!")

    # Show specific examples
    print(f"\nSpecific Examples:")
    interesting_cells = [c for c in cells if abs(c.raw_orientation_deg) < 45][:5]
    for cell in interesting_cells:
        print(f"  Raw: {cell.raw_orientation_deg:6.1f}° → Flow Alignment: {cell.flow_alignment_deg:5.1f}°")


def main():
    """Main function to run the angle debugging."""
    print("Creating test cells with known orientations...")
    print("Raw → Method1, Method2, Method3 (flow alignment methods):")
    print("-" * 60)

    cells = create_test_cells()

    print("\nCreating visualizations...")
    raw_angles, flow_angles = visualize_angle_comparison(cells)

    print("Creating polar comparison...")
    create_polar_comparison(raw_angles, flow_angles)

    print_angle_analysis(cells)

    print(f"\n✅ Debug complete! Check the generated images:")
    print(f"   - angle_debug_visualization.png")
    print(f"   - polar_comparison.png")


if __name__ == "__main__":
    main()