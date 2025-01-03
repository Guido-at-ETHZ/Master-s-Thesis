import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Create points for x-axis
x = np.linspace(0, 1, 1000)

# Define different parameter sets for beta distributions
params = [
    (5, 5, 'Beta(5,5)'),
    (5, 2, 'Beta(5,2)'),
    (0.5, 0.5, 'Beta(0.5,0.5)'),
    (1, 1, 'Beta(1,1)')
]

# Create the plot with higher resolution
plt.figure(figsize=(10, 6), dpi=300)  # Increased DPI for higher resolution

# Plot each beta distribution
for alpha, beta_param, label in params:
    plt.plot(x, beta.pdf(x, alpha, beta_param), label=label)

# Customize the plot
plt.title('Beta Distributions with Different Parameters')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot in high resolution
plt.savefig('beta_distributions.png', dpi=300, bbox_inches='tight')  # Saves as PNG
# plt.savefig('beta_distributions.pdf', bbox_inches='tight')  # Uncomment to save as PDF

plt.show()