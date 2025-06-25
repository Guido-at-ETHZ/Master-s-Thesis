"""
Population dynamics model for endothelial cell mechanotransduction.
"""
import numpy as np


class PopulationDynamicsModel:
    """
    Model for endothelial cell population dynamics under mechanical stimuli.

    This model tracks the evolution of different cell populations (healthy cells at different
    division stages, telomere-induced senescent cells, and stress-induced senescent cells)
    in response to mechanical stimuli and other factors.
    """

    def __init__(self, config):
        """
        Initialize the population dynamics model with configuration parameters.

        Parameters:
            config: SimulationConfig object with parameter settings
        """
        self.config = config

        # Model parameters
        self.max_divisions = config.max_divisions
        self.r = config.proliferation_rate
        self.K = config.carrying_capacity
        self.d_E = config.death_rate_healthy
        self.d_S_tel = config.death_rate_senescent_tel
        self.d_S_stress = config.death_rate_senescent_stress
        self.gamma_S = config.senescence_induction_factor

        # Senolytic parameters
        self.senolytic_conc = config.senolytic_concentration
        self.sen_efficacy_tel = config.senolytic_efficacy_tel
        self.sen_efficacy_stress = config.senolytic_efficacy_stress

        # Initialize population state
        self.initialize_state()

    def initialize_state(self, initial_distribution=None):
        """
        Initialize the population state.

        Parameters:
            initial_distribution: Optional dictionary with initial population counts
        """
        # State variables:
        # E[i] for i=0..max_divisions: Healthy cells at division stage i
        # S_tel: Telomere-induced senescent cells
        # S_stress: Stress-induced senescent cells

        # Create state dictionary with zeros
        self.state = {
            'E': [0] * (self.max_divisions + 1),
            'S_tel': 0,
            'S_stress': 0
        }

        # Set initial values if provided
        if initial_distribution is not None:
            if 'E' in initial_distribution:
                # Handle array or list
                if isinstance(initial_distribution['E'], (list, np.ndarray)):
                    for i in range(min(len(initial_distribution['E']), self.max_divisions + 1)):
                        self.state['E'][i] = initial_distribution['E'][i]
                # Handle dictionary with indices
                elif isinstance(initial_distribution['E'], dict):
                    for i, count in initial_distribution['E'].items():
                        if 0 <= i <= self.max_divisions:
                            self.state['E'][i] = count

            if 'S_tel' in initial_distribution:
                self.state['S_tel'] = initial_distribution['S_tel']

            if 'S_stress' in initial_distribution:
                self.state['S_stress'] = initial_distribution['S_stress']

    def calculate_total_cells(self):
        """
        Calculate the total number of cells.

        Returns:
            Dictionary with totals for different cell categories
        """
        E_total = sum(self.state['E'])
        S_tel = self.state['S_tel']
        S_stress = self.state['S_stress']
        S_total = S_tel + S_stress
        total_cells = E_total + S_total

        return {
            'E_total': E_total,
            'S_tel': S_tel,
            'S_stress': S_stress,
            'S_total': S_total,
            'total': total_cells
        }

    def calculate_average_division_age(self):
        """
        Calculate the average division age of healthy cells.

        Returns:
            Average division age
        """
        E_total = sum(self.state['E'])

        if E_total > 0:
            weighted_sum = sum(i * self.state['E'][i] for i in range(self.max_divisions + 1))
            return weighted_sum / E_total
        else:
            return np.nan

    def calculate_telomere_length(self):
        """
        Calculate the average telomere length based on division age.

        Returns:
            Average telomere length
        """
        avg_division_age = self.calculate_average_division_age()

        if np.isnan(avg_division_age):
            return np.nan

        max_telomere = 100
        min_telomere = 20

        # Normalize division age to 0-1 range
        normalized_age = min(max(0, avg_division_age / self.max_divisions), 1)

        # Calculate telomere length (decreases with division age)
        telomere_length = max_telomere - (max_telomere - min_telomere) * normalized_age

        return telomere_length

    def calculate_shear_stress_effect(self, tau):
        """
        Calculate the effect of shear stress on senescence induction.

        Parameters:
            tau: Shear stress value (Pa)

        Returns:
            Senescence induction rate
        """
        # Implementation from thesis model
        if tau <= 1:  # Low shear stress
            return 0.002 + tau * 0.05  # Very minor effect - changed to try
        elif tau <= 2:  # Moderate shear stress
            return 0.007 + (tau - 1) * 0.1  # Slight increase
        else:  # High shear stress
            return 0.017 + (tau - 2) * 0.5  # More rapid increase

    def calculate_senolytic_effect(self, concentration, efficacy_factor=1.0):
        """
        Calculate the effect of senolytics on senescent cell death rate.

        Parameters:
            concentration: Senolytic concentration
            efficacy_factor: Efficacy multiplier

        Returns:
            Death rate increase due to senolytics
        """
        if concentration <= 0:
            return 0

        # Sigmoid response curve
        max_effect = 0.15 * efficacy_factor
        ec50 = 5
        hill = 3

        effect = max_effect * (concentration ** hill) / (ec50 ** hill + concentration ** hill)

        return effect

    def calculate_senolytic_toxicity(self, concentration):
        """
        Calculate the toxic effect of senolytics on healthy cells.

        Parameters:
            concentration: Senolytic concentration

        Returns:
            Base toxicity to healthy cells
        """
        if concentration <= 0:
            return 0

        # Base toxicity with linear component
        base_toxicity = 0.0004 * concentration

        # Non-linear component with therapeutic window
        max_toxicity = 0.05
        ec50_toxicity = 20
        hill_toxicity = 5

        non_linear_toxicity = max_toxicity * (concentration ** hill_toxicity) / (
                ec50_toxicity ** hill_toxicity + concentration ** hill_toxicity)

        # Combine components
        total_toxicity = base_toxicity + non_linear_toxicity

        return total_toxicity

    def calculate_stem_cell_distribution(self):
        """
        Calculate the distribution of stem cells across division stages.

        Returns:
            Array of distribution values (sums to 1)
        """
        distribution = np.zeros(self.max_divisions + 1)

        # Exponential distribution - more stem cells enter at earlier stages
        for i in range(self.max_divisions + 1):
            distribution[i] = np.exp(-0.7 * i)

        # Normalize to sum to 1
        distribution = distribution / np.sum(distribution)

        return distribution

    def calculate_density_factor(self, total_cells):
        """
        Calculate density-dependent inhibition factor for cell division.

        Parameters:
            total_cells: Total number of cells

        Returns:
            Density factor (0-1)
        """
        # Sigmoid function allowing growth until approaching capacity
        return 1 / (1 + np.exp(10 * (total_cells / self.K - 0.7)))

    def calculate_division_capacity(self, division_stage):
        """
        Calculate division capacity based on division stage.

        Parameters:
            division_stage: Number of divisions completed

        Returns:
            Division capacity factor (0-1)
        """
        if division_stage <= 0.7 * self.max_divisions:
            return 1.0
        else:
            # Linear reduction in division capacity as cells approach max divisions
            return 1.0 - 0.5 * ((division_stage - 0.7 * self.max_divisions) /
                                (0.3 * self.max_divisions))

    def update(self, dt, tau, stem_cell_rate=None):
        """
        Update the population state for one time step.

        Parameters:
            dt: Time step
            tau: Shear stress value (Pa)
            stem_cell_rate: Optional stem cell input rate, uses 0 if None

        Returns:
            Updated state
        """
        # Use default stem cell rate of 0 if none provided
        if stem_cell_rate is None:
            stem_cell_rate = 0

        # Get current state variables
        E = self.state['E'].copy()
        S_tel = self.state['S_tel']
        S_stress = self.state['S_stress']

        # Calculate total cells for density factor
        totals = self.calculate_total_cells()
        total_cells = totals['total']
        S_total = totals['S_total']

        # Calculate regulatory factors
        density_factor = self.calculate_density_factor(total_cells)
        gamma_tau = self.calculate_shear_stress_effect(tau)
        senolytic_effect_tel = self.calculate_senolytic_effect(self.senolytic_conc, self.sen_efficacy_tel)
        senolytic_effect_stress = self.calculate_senolytic_effect(self.senolytic_conc, self.sen_efficacy_stress)
        senolytic_toxicity = self.calculate_senolytic_toxicity(self.senolytic_conc)
        stem_cell_distribution = self.calculate_stem_cell_distribution()

        # Initialize arrays for new state
        E_new = [0] * (self.max_divisions + 1)
        S_tel_new = 0
        S_stress_new = 0

        # Update each division stage
        for i in range(self.max_divisions + 1):
            # Age-dependent death rate + senolytic toxicity
            age_sensitivity_factor = 1.0 + 0.08 * i
            death_rate = self.d_E * (1 + 0.03 * i) + senolytic_toxicity * age_sensitivity_factor

            # Stress-induced senescence rate
            stress_senescence_rate = gamma_tau * (1 + 0.05 * i)

            # Senescence induction by SASP
            sasp_senescence_rate = self.gamma_S * S_total

            # Division capacity
            division_capacity = self.calculate_division_capacity(i)

            # Calculate terms for this division stage
            if i == 0:
                # First group (E_0) - undivided cells
                division_out = self.r * E[i] * density_factor * division_capacity
                division_in = 0
            elif i < self.max_divisions:
                # Middle groups
                division_capacity_prev = self.calculate_division_capacity(i - 1)
                division_in = 2 * self.r * E[i - 1] * density_factor * division_capacity_prev
                division_out = self.r * E[i] * density_factor * division_capacity
            else:
                # Last group - at maximum division count
                division_capacity_prev = self.calculate_division_capacity(i - 1)
                division_in = 2 * self.r * E[i - 1] * density_factor * division_capacity_prev
                division_out = self.r * E[i] * density_factor * division_capacity

            # Cell loss terms
            death_term = death_rate * E[i]
            stress_senescence_term = stress_senescence_rate * E[i]
            sasp_senescence_term = sasp_senescence_rate * E[i]

            # Stem cell input
            stem_cell_input = stem_cell_rate * stem_cell_distribution[i]

            # Combine all terms
            if i == self.max_divisions:
                # Last division stage - all divisions lead to telomere-induced senescence
                E_new[i] = E[i] + dt * (division_in - death_term - stress_senescence_term -
                                        sasp_senescence_term - division_out + stem_cell_input)

                # Add contribution to telomere-induced senescence
                S_tel_new += dt * division_out
            else:
                E_new[i] = E[i] + dt * (division_in - division_out - death_term -
                                        stress_senescence_term - sasp_senescence_term +
                                        stem_cell_input)

            # Add contributions to stress-induced senescence
            S_stress_new += dt * (stress_senescence_term + sasp_senescence_term)

        # Update senescent cell populations
        S_tel_new += S_tel + dt * (-(self.d_S_tel + senolytic_effect_tel) * S_tel)
        S_stress_new += S_stress + dt * (-(self.d_S_stress + senolytic_effect_stress) * S_stress)

        # Ensure non-negative values
        E_new = [max(0, e) for e in E_new]
        S_tel_new = max(0, S_tel_new)
        S_stress_new = max(0, S_stress_new)

        # Update state
        self.state['E'] = E_new
        self.state['S_tel'] = S_tel_new
        self.state['S_stress'] = S_stress_new

        return self.state

    def update_from_cells(self, cells, dt, tau, stem_cell_rate=None):
        """
        Update the population state based on a collection of individual cells.

        Parameters:
            cells: Dictionary of Cell objects
            dt: Time step
            tau: Shear stress value (Pa)
            stem_cell_rate: Optional stem cell input rate, uses 0 if None

        Returns:
            Updated state
        """
        # Create divisions counter
        divisions_count = [0] * (self.max_divisions + 1)
        S_tel_count = 0
        S_stress_count = 0

        # Count cells by division stage and senescence status
        for cell in cells.values():
            if cell.is_senescent:
                if cell.senescence_cause == 'telomere':
                    S_tel_count += 1
                elif cell.senescence_cause == 'stress':
                    S_stress_count += 1
            else:
                div = min(cell.divisions, self.max_divisions)
                divisions_count[div] += 1

        # Set current state to cell counts
        self.state['E'] = divisions_count
        self.state['S_tel'] = S_tel_count
        self.state['S_stress'] = S_stress_count

        # Update state
        return self.update(dt, tau, stem_cell_rate)

    def synchronize_cells(self, cells):
        """
        Synchronize individual cells with the population state.

        Parameters:
            cells: Dictionary of Cell objects

        Returns:
            Actions dictionary with births, deaths, and senescence events
        """
        # Get current state
        E = self.state['E'].copy()
        S_tel = self.state['S_tel']
        S_stress = self.state['S_stress']

        # Count cells by division stage and senescence status
        divisions_count = [0] * (self.max_divisions + 1)
        S_tel_count = 0
        S_stress_count = 0

        for cell in cells.values():
            if cell.is_senescent:
                if cell.senescence_cause == 'telomere':
                    S_tel_count += 1
                elif cell.senescence_cause == 'stress':
                    S_stress_count += 1
            else:
                div = min(cell.divisions, self.max_divisions)
                divisions_count[div] += 1

        # Calculate differences
        E_diff = [int(round(E[i] - divisions_count[i])) for i in range(self.max_divisions + 1)]
        S_tel_diff = int(round(S_tel - S_tel_count))
        S_stress_diff = int(round(S_stress - S_stress_count))

        # Initialize actions
        actions = {
            'births': [],
            'deaths': [],
            'senescence': []
        }

        # Handle differences
        # Birth actions (positive E_diff)
        for i in range(self.max_divisions + 1):
            if E_diff[i] > 0:
                for _ in range(E_diff[i]):
                    actions['births'].append({
                        'type': 'healthy',
                        'divisions': i
                    })

        # Handle telomere-induced senescence
        if S_tel_diff > 0:
            for _ in range(S_tel_diff):
                actions['births'].append({
                    'type': 'senescent',
                    'cause': 'telomere'
                })

        # Handle stress-induced senescence
        if S_stress_diff > 0:
            for _ in range(S_stress_diff):
                actions['births'].append({
                    'type': 'senescent',
                    'cause': 'stress'
                })

        # Death actions (negative diffs)
        for i in range(self.max_divisions + 1):
            if E_diff[i] < 0:
                actions['deaths'].append({
                    'type': 'healthy',
                    'divisions': i,
                    'count': -E_diff[i]
                })

        if S_tel_diff < 0:
            actions['deaths'].append({
                'type': 'senescent',
                'cause': 'telomere',
                'count': -S_tel_diff
            })

        if S_stress_diff < 0:
            actions['deaths'].append({
                'type': 'senescent',
                'cause': 'stress',
                'count': -S_stress_diff
            })

        return actions