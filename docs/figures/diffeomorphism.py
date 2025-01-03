import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import matplotlib.animation as animation

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Create points for x-axis
x = np.linspace(0, 1, 1000)


# Function to animate
def animate(frame):
    plt.cla()  # Clear the current axis

    # Calculate current alpha and beta using the diffeomorphism
    t = frame / 100  # This will give us 100 frames from 0 to 1
    alpha = beta_param = 1 + 4 * t  # Diffeomorphism: φ(t) = (1 + 4t, 1 + 4t)

    # Plot the current beta distribution
    y = beta.pdf(x, alpha, beta_param)
    plt.plot(x, y, 'b-', lw=2, label=f'Beta({alpha:.2f}, {beta_param:.2f})')

    # Add text showing the diffeomorphism
    plt.text(0.02, plt.ylim()[1] * 0.95, f'φ(t) = (1 + 4t, 1 + 4t)\nt = {t:.2f}',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Customize the plot
    plt.title('Beta Distribution Progression')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Set consistent y-axis limits
    plt.ylim(0, 2.5)


# Create animation
anim = animation.FuncAnimation(fig, animate, frames=101, interval=50, blit=False)

# Save animation (optional)
anim.save('beta_progression.gif', writer='pillow', fps=20)

plt.show()

# Also create a static plot showing multiple stages
plt.figure(figsize=(12, 8))

# Plot several stages of the progression
t_values = [0, 0.25, 0.5, 0.75, 1]
for t in t_values:
    alpha = beta_param = 1 + 4 * t
    plt.plot(x, beta.pdf(x, alpha, beta_param),
             label=f'Beta({alpha:.2f}, {beta_param:.2f})')

plt.title('Beta Distribution Progression Stages')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(0.02, 2.3, 'φ(t) = (1 + 4t, 1 + 4t)\nt ∈ [0,1]',
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Save the static plot (optional)
plt.savefig('beta_stages.png', dpi=300, bbox_inches='tight')

plt.show()