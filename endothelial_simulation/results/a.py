import numpy as np

# Test arctan2 output
angle_rad = np.arctan2(1, 1)  # 45-degree angle
angle_deg = np.degrees(angle_rad)

print(f"arctan2(1, 1) = {angle_rad:.4f} radians")  # Should be ~0.7854
print(f"In degrees: {angle_deg:.1f}°")              # Should be 45.0°
print(f"π/4 = {np.pi/4:.4f}")                       # Should match arctan2 result