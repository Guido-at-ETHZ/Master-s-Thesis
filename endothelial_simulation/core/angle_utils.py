"""
Angle Utilities for Endothelial Simulation

This module provides a centralized and consistent mathematical framework for handling all angle-related
calculations in the endothelial cell simulation. It addresses the critical need for a unified coordinate
system to resolve inconsistencies in how angles are represented and manipulated.

The core of this framework is built on three key principles:
1.  **Standardized Internal Representation:** All angles are normalized to a standard range of [-180°, 180°).
    This ensures that calculations are performed in a consistent mathematical space, avoiding ambiguities
    from multiple representations (e.g., 0-360°, radians vs. degrees).

2.  **Biologically-Relevant Alignment:** A dedicated function, `alignment_angle_deg`, converts the standard
    internal angle to the biologically meaningful alignment angle in the range [0°, 90°]. This directly
    corresponds to the physical reality of cell orientation relative to flow, where a cell at 30° and
    150° are both considered "30° from the flow axis." This is the canonical representation for setting
    targets and interpreting results.

3.  **Correct Shortest-Path Calculations:** The `angle_difference_deg` function correctly calculates the
    shortest angular distance between two angles, handling the "wrap-around" nature of circular quantities.
    This is essential for the dynamics model, ensuring that a cell at 170° moves towards a target of -170°
    by taking the short 20° path, not the long 340° path.

By using these utilities, the simulation can achieve predictable, reliable, and mathematically sound behavior
for all orientation-related tasks.
"""
import numpy as np

def normalize_angle_deg(angle):
    """
    Normalizes an angle to the range [-180, 180) degrees.

    This function is the cornerstone of our consistent angle representation. It takes any angle
    (positive, negative, or outside the standard 360-degree range) and maps it into a
    canonical [-180, 180) interval. This ensures that all subsequent calculations are
    performed on a predictable and standardized input.

    Args:
        angle (float): The angle in degrees.

    Returns:
        float: The normalized angle in the range [-180, 180).
    """
    return (angle + 180) % 360 - 180

def alignment_angle_deg(angle):
    """
    Calculates the alignment of an angle relative to the 0-degree flow axis.

    This function implements the key biological constraint of cell symmetry. It converts a
    normalized angle into its equivalent "alignment angle" in the range [0, 90] degrees.
    For example, an angle of -30°, 30°, 150°, or -150° all correspond to a 30° alignment
    relative to the flow direction. This is the primary metric for specifying targets and
    evaluating cell orientation.

    Args:
        angle (float): The normalized angle in degrees, expected to be in [-180, 180).

    Returns:
        float: The alignment angle in the range [0, 90].
    """
    return np.abs(normalize_angle_deg(angle)) if np.abs(normalize_angle_deg(angle)) <= 90 else 180 - np.abs(normalize_angle_deg(angle))

def angle_difference_deg(angle1, angle2):
    """
    Calculates the shortest difference between two angles in degrees.

    This function correctly handles the circular nature of angles. For example, the difference
    between 350 degrees and 10 degrees is -20 degrees, not 340. This is crucial for any
    dynamic system where you need to calculate an error or a rotational adjustment, ensuring
    the system always takes the shortest path.

    Args:
        angle1 (float): The first angle in degrees.
        angle2 (float): The second angle in degrees.

    Returns:
        float: The shortest difference between the two angles, in the range [-180, 180).
    """
    diff = normalize_angle_deg(angle1 - angle2)
    return diff
