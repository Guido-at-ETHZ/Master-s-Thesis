import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Create the plot
plt.figure(figsize=(12, 6), dpi=300)

# First distribution: Beta(2,2) mapped to [2,4]
x1 = np.linspace(0, 1, 1000)
y1 = beta.pdf(x1, 2, 2)
# Affine transformation: [0,1] -> [2,4]
x1_transformed = 2 + 2*x1  # Scale by 2 and shift by 2
# Scale y values to preserve area
y1_transformed = y1/2  # Divide by the scaling factor

# Second distribution: Beta(5,5) mapped to [5,6]
x2 = np.linspace(0, 1, 1000)
y2 = beta.pdf(x2, 5, 5)
# Affine transformation: [0,1] -> [5,6]
x2_transformed = 5 + x2    # Scale by 1 and shift by 5
# Scale y values to preserve area
y2_transformed = y2       # No scaling needed since interval length is 1

# Plot both distributions
plt.plot(x1_transformed, y1_transformed, label='Beta(2,2) on [2,4]', linewidth=2)
plt.plot(x2_transformed, y2_transformed, label='Beta(5,5) on [5,6]', linewidth=2)

# Customize the plot
plt.title('Transformed Beta Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot in high resolution
plt.savefig('transformed_beta_distributions.png', dpi=300, bbox_inches='tight')

plt.show()