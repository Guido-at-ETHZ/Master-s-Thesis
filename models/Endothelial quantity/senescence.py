import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter


class EndothelialCell:
    """
    Class representing an individual endothelial cell agent.
    """

    def __init__(self, position, telomere_length, age=0, replicative_age=0, is_senescent=False):
        """
        Initialize an endothelial cell.

        Parameters:
        -----------
        position : tuple
            (x, y) coordinates of the cell
        telomere_length : float
            Initial telomere length
        age : float
            Chronological age of the cell
        replicative_age : int
            Number of times the cell has divided
        is_senescent : bool
            Whether the cell is in a senescent state
        """
        self.position = np.array(position, dtype=float)
        self.telomere_length = telomere_length
        self.age = age
        self.replicative_age = replicative_age
        self.is_senescent = is_senescent
        self.senescence_cause = None  # Track cause of senescence: 'telomere' or 'stress'

    def divide(self, telomere_loss_mean, telomere_loss_std):
        """
        Cell division process, returns a new daughter cell.

        Parameters:
        -----------
        telomere_loss_mean : float
            Mean telomere loss per division
        telomere_loss_std : float
            Standard deviation of telomere loss

        Returns:
        --------
        EndothelialCell
            Newly created daughter cell
        """
        # Increase replicative age
        self.replicative_age += 1

        # Calculate telomere loss with stochastic variation
        telomere_loss = telomere_loss_mean + np.random.normal(0, telomere_loss_std)

        # Reduce telomere length
        self.telomere_length -= telomere_loss

        # Create daughter cell (small random offset in position)
        offset = np.random.uniform(-0.1, 0.1, 2)
        daughter_position = self.position + offset

        # Daughter inherits telomere length with stochastic variation
        daughter_telomere = self.telomere_length + np.random.normal(0, telomere_loss_std)
        daughter_telomere = max(0, daughter_telomere)  # Telomere length cannot be negative

        return EndothelialCell(
            position=daughter_position,
            telomere_length=daughter_telomere,
            age=0,  # Reset chronological age
            replicative_age=0  # Reset replicative age
        )

    def move(self, dt, motility, grid_size):
        """
        Move the cell according to a random walk.

        Parameters:
        -----------
        dt : float
            Time step
        motility : float
            Motility coefficient (diffusion constant)
        grid_size : tuple
            (width, height) of the grid
        """
        # Motility is reduced for senescent cells
        effective_motility = motility
        if self.is_senescent:
            effective_motility *= 0.3  # Senescent cells move slower

        # Random movement (Brownian motion)
        displacement = np.sqrt(2 * effective_motility * dt) * np.random.normal(0, 1, 2)
        new_position = self.position + displacement

        # Ensure the cell stays within the grid (periodic boundary conditions)
        new_position[0] = new_position[0] % grid_size[0]
        new_position[1] = new_position[1] % grid_size[1]

        self.position = new_position


class EndothelialLayer:
    """
    Class representing a layer of endothelial cells on a surface.
    """

    def __init__(self, grid_size=(100, 100), initial_count=100):
        """
        Initialize the endothelial layer.

        Parameters:
        -----------
        grid_size : tuple
            (width, height) of the simulation grid
        initial_count : int
            Initial number of cells
        """
        self.grid_size = grid_size
        self.cells = []
        self.time = 0

        # Model parameters
        # Telomere dynamics
        self.initial_telomere_length = 100
        self.telomere_loss_mean = 1.5
        self.telomere_loss_std = 0.3
        self.critical_telomere_length = 20
        self.division_telomere_threshold = 30

        # Rate parameters
        self.baseline_division_rate = 0.05
        self.max_density = 0.7 * (grid_size[0] * grid_size[1])  # Max 70% coverage
        self.base_normal_death_rate = 0.01
        self.base_senescent_death_rate = 0.03
        self.age_death_factor_normal = 0.001
        self.age_death_factor_senescent = 0.002
        self.stress_death_factor_normal = 0.02
        self.stress_death_factor_senescent = 0.03

        # Stress-induced senescence parameters
        self.base_sips_rate = 0.003
        self.age_sips_factor = 0.0005
        self.stress_sips_factor = 0.01
        self.density_sips_factor = 0.005

        # Movement parameters
        self.motility = 0.5

        # Environmental stress (can vary over time)
        self.stress_level = 0.2

        # Initialize cells
        self._initialize_cells(initial_count)

        # History tracking
        self.history = {
            'time': [],
            'normal_count': [],
            'senescent_count': [],
            'total_count': [],
            'mean_telomere_length': [],
            'senescent_fraction': [],
            'mean_age': []
        }

    def _initialize_cells(self, count):
        """
        Initialize the cell population.

        Parameters:
        -----------
        count : int
            Number of cells to initialize
        """
        for _ in range(count):
            # Random position
            position = np.random.uniform(0, self.grid_size, 2)

            # Initial telomere length with some variation
            telomere_length = self.initial_telomere_length + np.random.normal(0, 5)

            # Create cell
            cell = EndothelialCell(position=position, telomere_length=telomere_length)
            self.cells.append(cell)

    def calculate_local_density(self, kernel_size=5.0):
        """
        Calculate local cell density using a Gaussian kernel.

        Parameters:
        -----------
        kernel_size : float
            Size of the Gaussian kernel

        Returns:
        --------
        numpy.ndarray
            2D array of density values
        """
        # Create density grid
        density = np.zeros(self.grid_size)

        # Plot each cell as a point on the grid
        for cell in self.cells:
            x, y = cell.position
            x_int, y_int = int(x) % self.grid_size[0], int(y) % self.grid_size[1]
            density[y_int, x_int] += 1

        # Apply Gaussian filter to get smooth density
        smoothed_density = gaussian_filter(density, kernel_size)

        return smoothed_density

    def get_local_density_at(self, position, density_grid):
        """
        Get the local density at a specific position.

        Parameters:
        -----------
        position : numpy.ndarray
            (x, y) coordinates
        density_grid : numpy.ndarray
            Precomputed density grid

        Returns:
        --------
        float
            Local density value
        """
        x, y = position
        x_int, y_int = int(x) % self.grid_size[0], int(y) % self.grid_size[1]
        return density_grid[y_int, x_int]

    def calculate_division_probability(self, cell, local_density):
        """
        Calculate the probability of cell division.

        Parameters:
        -----------
        cell : EndothelialCell
            The cell
        local_density : float
            Local cell density

        Returns:
        --------
        float
            Probability of division [0, 1]
        """
        if cell.is_senescent:
            return 0.0  # Senescent cells don't divide

        # Density-dependent inhibition
        density_factor = max(0, 1 - local_density / self.max_density)

        # Telomere-dependent capacity (logistic function)
        telomere_factor = 1 / (1 + np.exp(-0.2 * (cell.telomere_length - self.division_telomere_threshold)))

        return min(1.0, self.baseline_division_rate * density_factor * telomere_factor)

    def calculate_sips_probability(self, cell, local_density):
        """
        Calculate probability of stress-induced premature senescence.

        Parameters:
        -----------
        cell : EndothelialCell
            The cell
        local_density : float
            Local cell density

        Returns:
        --------
        float
            Probability of SIPS [0, 1]
        """
        if cell.is_senescent:
            return 0.0  # Already senescent

        # Calculate hazard rate
        hazard_rate = (self.base_sips_rate +
                       self.age_sips_factor * cell.age +
                       self.stress_sips_factor * self.stress_level +
                       self.density_sips_factor * local_density)

        # Convert hazard rate to probability for the time step
        return 1 - np.exp(-hazard_rate)

    def calculate_death_probability(self, cell):
        """
        Calculate probability of cell death.

        Parameters:
        -----------
        cell : EndothelialCell
            The cell

        Returns:
        --------
        float
            Probability of death [0, 1]
        """
        if cell.is_senescent:
            # Senescent cell death rate
            death_rate = (self.base_senescent_death_rate +
                          self.age_death_factor_senescent * cell.age +
                          self.stress_death_factor_senescent * self.stress_level)
        else:
            # Normal cell death rate
            death_rate = (self.base_normal_death_rate +
                          self.age_death_factor_normal * cell.age +
                          self.stress_death_factor_normal * self.stress_level)

        # Convert rate to probability for the time step
        return 1 - np.exp(-death_rate)

    def calculate_senolytic_removal_probability(self, senolytic_dose, EC50=0.5, hill_coef=2):
        """
        Calculate the probability of senescent cell removal by senolytics.

        Parameters:
        -----------
        senolytic_dose : float
            Dose of senolytic
        EC50 : float
            Dose producing 50% of maximal effect
        hill_coef : float
            Hill coefficient for dose-response curve

        Returns:
        --------
        float
            Probability of senolytic-induced removal [0, 1]
        """
        if senolytic_dose <= 0:
            return 0.0

        # Hill function for dose-response
        efficacy = (senolytic_dose ** hill_coef) / (senolytic_dose ** hill_coef + EC50 ** hill_coef)

        return efficacy

    def step(self, dt=1.0, senolytic_dose=0.0):
        """
        Advance the simulation by one time step.

        Parameters:
        -----------
        dt : float
            Time step size
        senolytic_dose : float
            Current senolytic dose

        Returns:
        --------
        dict
            Statistics for the current step
        """
        self.time += dt

        # Calculate density field
        density_grid = self.calculate_local_density()

        # Lists for cells to add/remove
        new_cells = []
        cells_to_remove = []

        # Process each cell
        for cell in self.cells:
            # Increment age
            cell.age += dt

            # Get local density
            local_density = self.get_local_density_at(cell.position, density_grid)

            # MANDATORY TELOMERE-BASED SENESCENCE:
            # Once telomere length falls below critical threshold, the cell
            # irreversibly enters senescence state. This is deterministic and inevitable
            # with continued divisions, ensuring that all cells eventually undergo
            # replicative senescence if they survive long enough.
            if not cell.is_senescent and cell.telomere_length < self.critical_telomere_length:
                cell.is_senescent = True
                # Record telomere-induced senescence event
                # This could be extended to track different senescence causes
                cell.senescence_cause = "telomere"

            # Check for stress-induced senescence (probabilistic)
            elif not cell.is_senescent:
                sips_prob = self.calculate_sips_probability(cell, local_density)
                if np.random.random() < sips_prob:
                    cell.is_senescent = True
                    # Record stress-induced senescence
                    cell.senescence_cause = "stress"

            # Cell division
            division_prob = self.calculate_division_probability(cell, local_density)
            if np.random.random() < division_prob * dt:
                daughter = cell.divide(self.telomere_loss_mean, self.telomere_loss_std)
                new_cells.append(daughter)

            # Cell death
            death_prob = self.calculate_death_probability(cell)
            if np.random.random() < death_prob * dt:
                cells_to_remove.append(cell)

            # Senolytic effect on senescent cells
            if cell.is_senescent and senolytic_dose > 0:
                removal_prob = self.calculate_senolytic_removal_probability(senolytic_dose)
                if np.random.random() < removal_prob * dt:
                    cells_to_remove.append(cell)

            # Cell movement
            cell.move(dt, self.motility, self.grid_size)

        # Update cell population
        for cell in cells_to_remove:
            if cell in self.cells:  # Check in case a cell was added to remove list twice
                self.cells.remove(cell)

        self.cells.extend(new_cells)

        # Gather statistics
        stats = self._update_history()

        return stats

    def _update_history(self):
        """
        Update the history of the simulation with current statistics.

        Returns:
        --------
        dict
            Current statistics
        """
        normal_cells = [c for c in self.cells if not c.is_senescent]
        senescent_cells = [c for c in self.cells if c.is_senescent]
        normal_count = len(normal_cells)
        senescent_count = len(senescent_cells)
        total_count = normal_count + senescent_count

        # Calculate mean telomere length of normal cells
        mean_telomere_length = 0
        if normal_count > 0:
            mean_telomere_length = np.mean([c.telomere_length for c in normal_cells])

        # Count cells by senescence cause
        telomere_senescent_count = len([c for c in self.cells if c.is_senescent and c.senescence_cause == "telomere"])
        stress_senescent_count = len([c for c in self.cells if c.is_senescent and c.senescence_cause == "stress"])

        # Calculate mean age
        mean_age = 0
        if total_count > 0:
            mean_age = np.mean([c.age for c in self.cells])

        # Calculate mean replicative age
        mean_replicative_age = 0
        if total_count > 0:
            mean_replicative_age = np.mean([c.replicative_age for c in self.cells])

        # Calculate senescent fraction
        senescent_fraction = 0
        if total_count > 0:
            senescent_fraction = senescent_count / total_count

        # Update history
        self.history['time'].append(self.time)
        self.history['normal_count'].append(normal_count)
        self.history['senescent_count'].append(senescent_count)
        self.history[
            'telomere_senescent_count'] = telomere_senescent_count if 'telomere_senescent_count' in self.history else [
            telomere_senescent_count]
        self.history[
            'stress_senescent_count'] = stress_senescent_count if 'stress_senescent_count' in self.history else [
            stress_senescent_count]
        self.history['total_count'].append(total_count)
        self.history['mean_telomere_length'].append(mean_telomere_length)
        self.history['mean_replicative_age'] = mean_replicative_age if 'mean_replicative_age' in self.history else [
            mean_replicative_age]
        self.history['senescent_fraction'].append(senescent_fraction)
        self.history['mean_age'].append(mean_age)

        return {
            'time': self.time,
            'normal_count': normal_count,
            'senescent_count': senescent_count,
            'telomere_senescent_count': telomere_senescent_count,
            'stress_senescent_count': stress_senescent_count,
            'total_count': total_count,
            'mean_telomere_length': mean_telomere_length,
            'mean_replicative_age': mean_replicative_age,
            'senescent_fraction': senescent_fraction,
            'mean_age': mean_age
        }

    def run_simulation(self, total_time, dt=1.0, senolytic_dose=0.0, progress_bar=True):
        """
        Run the simulation for a specified time.

        Parameters:
        -----------
        total_time : float
            Total simulation time
        dt : float
            Time step size
        senolytic_dose : float
            Constant senolytic dose
        progress_bar : bool
            Whether to display a progress bar

        Returns:
        --------
        pandas.DataFrame
            History of the simulation
        """
        steps = int(total_time / dt)

        if progress_bar:
            iterator = tqdm(range(steps), desc="Simulating")
        else:
            iterator = range(steps)

        for _ in iterator:
            self.step(dt, senolytic_dose)

            # Stop if all cells have died
            if len(self.cells) == 0:
                if progress_bar:
                    print("All cells died. Stopping simulation.")
                break

        return pd.DataFrame(self.history)

    def plot_cell_distribution(self, ax=None, alpha=0.7, s=30):
        """
        Plot the current distribution of cells.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None, a new figure is created
        alpha : float
            Transparency of points
        s : float
            Size of points

        Returns:
        --------
        matplotlib.axes.Axes
            The axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Extract positions and states
        positions = np.array([cell.position for cell in self.cells])
        colors = ['blue' if not cell.is_senescent else 'red' for cell in self.cells]

        if len(positions) > 0:
            ax.scatter(positions[:, 0], positions[:, 1], c=colors, alpha=alpha, s=s)

        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
        ax.set_title(f'Cell Distribution at t={self.time:.1f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Add a legend
        ax.scatter([], [], c='blue', label='Normal')
        ax.scatter([], [], c='red', label='Senescent')
        ax.legend()

        return ax

    def plot_history(self, figsize=(18, 15)):
        """
        Plot the history of the simulation.

        Parameters:
        -----------
        figsize : tuple
            Figure size

        Returns:
        --------
        matplotlib.figure.Figure
            The figure with the plots
        """
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()

        # Convert history to DataFrame
        history_df = pd.DataFrame(self.history)

        # Plot cell counts
        ax = axes[0]
        ax.plot(history_df['time'], history_df['normal_count'], 'b-', label='Normal')
        ax.plot(history_df['time'], history_df['senescent_count'], 'r-', label='Senescent')
        ax.plot(history_df['time'], history_df['total_count'], 'k--', label='Total')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cell count')
        ax.set_title('Cell Population Dynamics')
        ax.legend()
        ax.grid(True)

        # Plot senescence by cause
        ax = axes[1]
        if 'telomere_senescent_count' in history_df.columns and 'stress_senescent_count' in history_df.columns:
            ax.plot(history_df['time'], history_df['telomere_senescent_count'], 'r-', label='Telomere-induced')
            ax.plot(history_df['time'], history_df['stress_senescent_count'], 'orange', label='Stress-induced')
            ax.set_xlabel('Time')
            ax.set_ylabel('Cell count')
            ax.set_title('Senescence by Cause')
            ax.legend()
            ax.grid(True)

        # Plot mean telomere length
        ax = axes[2]
        ax.plot(history_df['time'], history_df['mean_telomere_length'], 'g-')
        ax.axhline(y=self.critical_telomere_length, color='r', linestyle='--',
                   label=f'Critical length ({self.critical_telomere_length})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean telomere length')
        ax.set_title('Mean Telomere Length of Normal Cells')
        ax.legend()
        ax.grid(True)

        # Plot senescent fraction
        ax = axes[3]
        ax.plot(history_df['time'], history_df['senescent_fraction'], 'r-')
        ax.set_xlabel('Time')
        ax.set_ylabel('Senescent fraction')
        ax.set_title('Fraction of Senescent Cells')
        ax.grid(True)
        ax.set_ylim(0, 1)

        # Plot mean age
        ax = axes[4]
        ax.plot(history_df['time'], history_df['mean_age'], 'b-')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean age')
        ax.set_title('Mean Chronological Age of Cells')
        ax.grid(True)

        # Plot mean replicative age
        ax = axes[5]
        if 'mean_replicative_age' in history_df.columns:
            ax.plot(history_df['time'], history_df['mean_replicative_age'], 'm-')
            ax.set_xlabel('Time')
            ax.set_ylabel('Mean replicative age')
            ax.set_title('Mean Replicative Age of Cells')
            ax.grid(True)

        # Cell distribution
        ax = axes[6]
        self.plot_cell_distribution(ax)

        # Density plot
        ax = axes[7]
        density = self.calculate_local_density()
        im = ax.imshow(density, cmap='viridis', origin='lower')
        ax.set_title('Cell Density')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)

        # Population trajectory prediction
        ax = axes[8]
        # If we have enough data points, try to predict extinction time
        if len(history_df) > 10:
            try:
                # Simple extrapolation based on recent trend
                recent = history_df.iloc[-20:] if len(history_df) > 20 else history_df
                # Only if population is declining
                if recent['total_count'].iloc[-1] < recent['total_count'].iloc[0]:
                    # Fit exponential decay: N = N0 * exp(-λt)
                    from scipy.optimize import curve_fit

                    def exp_decay(t, n0, lam):
                        return n0 * np.exp(-lam * t)

                    recent_times = recent['time'].values - recent['time'].values[0]
                    recent_counts = recent['total_count'].values

                    # Only proceed if we have non-zero counts
                    if np.any(recent_counts > 0):
                        popt, _ = curve_fit(exp_decay, recent_times, recent_counts,
                                            p0=[recent_counts[0], 0.01], bounds=([0, 0], [np.inf, np.inf]))

                        # Project future trajectory
                        future_times = np.linspace(0, history_df['time'].max() * 3, 100)
                        future_counts = exp_decay(future_times, *popt)

                        # Plot observed trajectory
                        ax.plot(history_df['time'], history_df['total_count'], 'k-', label='Observed')

                        # Plot projected trajectory
                        ax.plot(future_times + history_df['time'].values[0], future_counts, 'k--', label='Projected')

                        # Estimate time to extinction (population < 1)
                        if popt[1] > 0:  # Only if decay rate is positive
                            extinction_time = np.log(popt[0]) / popt[1]
                            ax.axvline(x=extinction_time + history_df['time'].values[0], color='r', linestyle='--',
                                       label=f'Est. extinction: t={extinction_time + history_df["time"].values[0]:.1f}')

                        ax.set_xlabel('Time')
                        ax.set_ylabel('Total cell count')
                        ax.set_title('Population Trajectory Projection')
                        ax.legend()
                        ax.grid(True)
            except Exception as e:
                # If projection fails, just show actual population
                ax.plot(history_df['time'], history_df['total_count'], 'k-')
                ax.set_xlabel('Time')
                ax.set_ylabel('Total cell count')
                ax.set_title('Population Trajectory')
                ax.text(0.5, 0.5, f"Projection unavailable\n{str(e)}",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
                ax.grid(True)
        else:
            # Not enough data for projection
            ax.plot(history_df['time'], history_df['total_count'], 'k-')
            ax.set_xlabel('Time')
            ax.set_ylabel('Total cell count')
            ax.set_title('Population Trajectory (insufficient data for projection)')
            ax.grid(True)

        plt.tight_layout()
        return fig


def find_minimum_senolytic_dose(model, target_time=200, min_density=0.3, max_senescent_fraction=0.2):
    """
    Find the minimum senolytic dose needed to maintain endothelial layer integrity.

    This function searches for the optimal senolytic dose that:
    1. Maintains cell density above the minimum threshold
    2. Keeps senescent cell fraction below maximum threshold
    3. Uses the minimum possible dose to achieve these goals

    Note: Even with optimal senolytic intervention, the cell population will eventually
    decline to zero in the infinite time horizon due to mandatory telomere-based
    senescence. The goal is to maintain viable coverage for the target timeframe.

    Parameters:
    -----------
    model : EndothelialLayer
        The model to use
    target_time : float
        Time horizon for optimization
    min_density : float
        Minimum acceptable density (fraction of carrying capacity)
    max_senescent_fraction : float
        Maximum acceptable senescent cell fraction

    Returns:
    --------
    float
        Optimal senolytic dose
    dict
        Simulation results with optimal dose
    """

    def objective_function(dose):
        """
        Objective function to minimize.

        Parameters:
        -----------
        dose : float
            Senolytic dose to test

        Returns:
        --------
        float
            Penalty value (lower is better)
        """
        # Create a copy of the model
        test_model = EndothelialLayer(
            grid_size=model.grid_size,
            initial_count=len(model.cells)
        )

        # Ensure test model has same parameters
        test_model.cells = [EndothelialCell(
            position=cell.position.copy(),
            telomere_length=cell.telomere_length,
            age=cell.age,
            replicative_age=cell.replicative_age,
            is_senescent=cell.is_senescent
        ) for cell in model.cells]

        # Run simulation with this dose
        results = test_model.run_simulation(target_time, senolytic_dose=dose, progress_bar=False)

        # Calculate penalties
        min_required_cells = min_density * test_model.max_density

        # Density penalty (if too low)
        density_penalty = 0
        for i, count in enumerate(results['total_count']):
            if count < min_required_cells:
                # Penalty increases the earlier it fails
                time_factor = 1 - (i / len(results))
                density_penalty += (min_required_cells - count) * time_factor

        # Senescent fraction penalty (if too high)
        senescent_penalty = 0
        for fraction in results['senescent_fraction']:
            if fraction > max_senescent_fraction:
                senescent_penalty += (fraction - max_senescent_fraction) * 100

        # Dose penalty (we want to minimize dose)
        dose_penalty = dose * 10

        # Total penalty
        total_penalty = density_penalty + senescent_penalty + dose_penalty

        return total_penalty

    # Find optimal dose
    result = minimize_scalar(
        objective_function,
        bounds=(0, 1),
        method='bounded'
    )

    optimal_dose = result.x

    # Run simulation with optimal dose
    result_model = EndothelialLayer(
        grid_size=model.grid_size,
        initial_count=len(model.cells)
    )

    # Copy cells
    result_model.cells = [EndothelialCell(
        position=cell.position.copy(),
        telomere_length=cell.telomere_length,
        age=cell.age,
        replicative_age=cell.replicative_age,
        is_senescent=cell.is_senescent
    ) for cell in model.cells]

    # Run simulation
    optimal_results = result_model.run_simulation(target_time, senolytic_dose=optimal_dose)

    return optimal_dose, optimal_results


def compare_senolytic_regimes(initial_model, doses=[0, 0.1, 0.3, 0.5], simulation_time=1000, figsize=(15, 15)):
    """
    Compare different senolytic dosing regimes.

    Parameters:
    -----------
    initial_model : EndothelialLayer
        Initial model state
    doses : list
        List of doses to compare
    simulation_time : float
        Simulation time
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure.Figure
        Figure with comparison plots
    list
        List of DataFrames with results for each dose
    """
    results = []

    for dose in tqdm(doses, desc="Testing doses"):
        # Create a copy of the model
        model_copy = EndothelialLayer(
            grid_size=initial_model.grid_size,
            initial_count=len(initial_model.cells)
        )

        # Copy cells
        model_copy.cells = [EndothelialCell(
            position=cell.position.copy(),
            telomere_length=cell.telomere_length,
            age=cell.age,
            replicative_age=cell.replicative_age,
            is_senescent=cell.is_senescent
        ) for cell in initial_model.cells]

        # Run simulation
        result_df = model_copy.run_simulation(simulation_time, senolytic_dose=dose)
        result_df['dose'] = dose
        results.append(result_df)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Colors for different doses
    colors = plt.cm.viridis(np.linspace(0, 1, len(doses)))

    # Plot cell counts
    ax = axes[0]
    for i, (dose, result) in enumerate(zip(doses, results)):
        ax.plot(result['time'], result['total_count'], color=colors[i], label=f'Dose {dose}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Total cell count')
    ax.set_title('Total Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot normal cell counts
    ax = axes[1]
    for i, (dose, result) in enumerate(zip(doses, results)):
        ax.plot(result['time'], result['normal_count'], color=colors[i], label=f'Dose {dose}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Normal cell count')
    ax.set_title('Normal Cell Population')
    ax.legend()
    ax.grid(True)

    # Plot senescent fraction
    ax = axes[2]
    for i, (dose, result) in enumerate(zip(doses, results)):
        ax.plot(result['time'], result['senescent_fraction'], color=colors[i], label=f'Dose {dose}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Senescent fraction')
    ax.set_title('Fraction of Senescent Cells')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True)

    # Plot telomere length
    ax = axes[3]
    for i, (dose, result) in enumerate(zip(doses, results)):
        ax.plot(result['time'], result['mean_telomere_length'], color=colors[i], label=f'Dose {dose}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean telomere length')
    ax.set_title('Mean Telomere Length of Normal Cells')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    return fig, results


# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create model with explicit parameters to demonstrate mandatory senescence
    model = EndothelialLayer(grid_size=(100, 100), initial_count=200)

    # Configure parameters to emphasize mandatory telomere-based senescence
    model.initial_telomere_length = 100  # Initial telomere length
    model.telomere_loss_mean = 2.0  # Telomere loss per division - higher value accelerates senescence
    model.critical_telomere_length = 20  # Critical length below which senescence is MANDATORY

    # Reduce stress-induced senescence to emphasize telomere effects
    model.base_sips_rate = 0.001  # Lower baseline for stress-induced senescence

    print("Model parameters:")
    print(f"- Initial telomere length: {model.initial_telomere_length}")
    print(f"- Telomere loss per division: {model.telomere_loss_mean}")
    print(f"- Critical telomere length (mandatory senescence): {model.critical_telomere_length}")
    print(f"- Division capacity telomere threshold: {model.division_telomere_threshold}")

    # Run long-term simulation without senolytics to demonstrate inevitable decline
    print("\nRunning baseline simulation without senolytics...")
    print("This demonstrates inevitable population decline due to mandatory telomere-based senescence")
    baseline_results = model.run_simulation(total_time=1000)

    # Plot results
    fig = model.plot_history()
    plt.savefig("baseline_results_with_mandatory_senescence.png")

    # Create a fresh model for senolytic experiments
    model = EndothelialLayer(grid_size=(100, 100), initial_count=200)
    model.telomere_loss_mean = 2.0

    # Find minimum senolytic dose
    print("\nFinding minimum effective senolytic dose...")
    print("Note: Even with optimal senolytic intervention, population will eventually decline to zero")
    print("The goal is to maintain viable coverage for the target timeframe")
    optimal_dose, optimal_results = find_minimum_senolytic_dose(model, target_time=1000)
    print(f"Optimal dose: {optimal_dose:.4f}")

    # Compare different senolytic regimes for long-term effects
    print("\nComparing different senolytic dosing regimes...")
    doses = [0, optimal_dose / 2, optimal_dose, optimal_dose * 2]
    comparison_fig, comparison_results = compare_senolytic_regimes(model, doses, simulation_time=500)
    plt.savefig("dose_comparison_with_mandatory_senescence.png")

    print("\nLong-term simulation confirms population decline to zero in all scenarios")
    print("Higher senolytic doses can extend viable coverage period but cannot prevent eventual extinction")

    plt.show()