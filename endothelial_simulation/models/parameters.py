"""
Model parameters for the endothelial cell mechanotransduction simulation.

This module contains all parameters related to cell behavior, shear stress response,
temporal dynamics, and population characteristics.
"""

import numpy as np


class ModelParameters:
    def __init__(self):
        # ------------------------------
        # Shear stress parameters
        # ------------------------------
        # Wall shear stress in Pa (Pascal)
        self.shear_stress = 1.4

        # Critical values for different shear stress regimes
        self.low_shear_threshold = 0.7  # Below this is considered low shear
        self.normal_shear_range = (0.7, 4.0)  # Physiological range
        self.high_shear_threshold = 4.0  # Above this is considered high shear

        # ------------------------------
        # Temporal dynamics parameters
        # ------------------------------
        # Base time constant (in minutes)
        self.time_constant_base = 30.0

        # Scaling factor for time constant's dependence on response magnitude
        self.scaling_factor = 0.8


        # Parameters for Michaelis-Menten model of mechanotransduction
        self.v_max = 2.0  # Maximum production rate
        self.k_m = 10.0  # Michaelis constant

        # Experimental data from your thesis (pressure-response mapping)
        self.a_max_map = {15: 1.5, 25: 3.7, 45: 5.3}

        # Linear model coefficients for a_max prediction
        self.slope = 0.108  # From data fit
        self.intercept = 0.12

        # ------------------------------
        # Spatial parameters
        # ------------------------------
        # Base aspect ratio for cells
        self.optimal_aspect_ratio_base = 2.0

        # How much aspect ratio changes per unit of shear stress
        self.aspect_ratio_sensitivity = 0.2

        # Cell size parameters (in pixels)
        self.cell_size_min = 20
        self.cell_size_max = 40
        self.cell_size_mean = 30

        # ------------------------------
        # Population dynamics parameters
        # ------------------------------
        # Maximum number of divisions a cell can undergo
        self.max_divisions = 15

        # Base cell division rate (per minute)
        # 0.0006 would be about once per 28 hours on average
        self.division_rate = 0 #0.0006

        # Base cell death rate (per minute)
        self.death_rate = 0 #0.0001

        # Carrying capacity (maximum sustainable cell count)
        self.carrying_capacity = 200

        # ------------------------------
        # Senescence parameters
        # ------------------------------
        # Death rate of telomere-induced senescent cells (per minute)
        self.tel_death_rate = 0 #0.00033  # About 1/50 of normal turnover rate

        # Death rate of stress-induced senescent cells (per minute)
        self.stress_death_rate = 0 #0.00042  # Slightly higher than telomere-induced

        # Senescence induction by SASP (Senescence-Associated Secretory Phenotype)
        self.sasp_factor = 0 #0.0000008

        # ------------------------------
        # Senolytic parameters
        # ------------------------------
        # Concentration of senolytic drug
        self.senolytic_concentration = 5.0

        # Efficacy of senolytics on telomere-induced senescent cells
        self.sen_efficacy_tel = 1.0

        # Efficacy of senolytics on stress-induced senescent cells
        self.sen_efficacy_stress = 1.2  # Slightly more effective

        # ------------------------------
        # Stem cell parameters
        # ------------------------------
        # Rate of stem cell input (cells per minute)
        self.stem_cell_rate = 0 #0.17  # About 10 per hour

        # Distribution of stem cells across division stages
        self._stem_cell_distribution = None

    @property
    def stem_cell_distribution(self):
        """Calculate stem cell distribution on-demand"""
        if self._stem_cell_distribution is None:
            # Create an exponential distribution across division stages
            distribution = np.zeros(self.max_divisions + 1)
            for i in range(self.max_divisions + 1):
                distribution[i] = np.exp(-0.7 * i)
            self._stem_cell_distribution = distribution / np.sum(distribution)
        return self._stem_cell_distribution

    def calculate_optimal_aspect_ratio(self, tau):
        """
        Calculate optimal cell aspect ratio based on shear stress.

        Higher shear stress leads to more elongated cells.

        Args:
            tau: Shear stress in Pa

        Returns:
            Optimal aspect ratio for the given shear stress
        """
        return self.optimal_aspect_ratio_base + self.aspect_ratio_sensitivity * tau

    def calculate_shear_stress_effect(self, tau):
        """
        Model how shear stress affects senescence rate.

        Implements a piecewise function with different sensitivity in different
        shear stress regimes.

        Args:
            tau: Shear stress in Pa

        Returns:
            Rate of stress-induced senescence
        """
        if tau <= 10:  # Low shear stress
            return 0.002 + tau * 0.0005
        elif tau <= 20:  # Moderate shear stress
            return 0.007 + (tau - 10) * 0.001
        else:  # High shear stress
            return 0.017 + (tau - 20) * 0.005

    def calculate_max_response(self, tau):
        """
        Calculate maximum response based on pressure/shear stress.

        Uses experimental data where available, falls back to linear model
        for other values.

        Args:
            tau: Shear stress in Pa

        Returns:
            Maximum response value
        """
        # Use known value if available
        if tau in self.a_max_map:
            return self.a_max_map[tau]

        # Otherwise use linear model with minimum of 1.0
        return max(1.0, self.slope * tau + self.intercept)

    def calculate_time_constant(self, tau):
        """
        Calculate time constant based on shear stress.

        The time constant scales with the maximum response.

        Args:
            tau: Shear stress in Pa

        Returns:
            Time constant in minutes
        """
        a_max = self.calculate_max_response(tau)
        return self.time_constant_base * (a_max ** self.scaling_factor)

    def calculate_division_rate_modifier(self, shear_stress):
        """
        Calculate how shear stress modifies division rate.

        The division rate is highest at normal physiological shear levels
        and decreases at very low or very high values.

        Args:
            shear_stress: Wall shear stress in Pa

        Returns:
            Modifier to base division rate (0-1)
        """
        # Optimal shear is in the middle of the normal range
        optimal_shear = np.mean(self.normal_shear_range)

        # Gaussian-like response centered at optimal shear
        return np.exp(-((shear_stress - optimal_shear) ** 2) / (2 * 5.0 ** 2))


# Create default parameter set
default_params = ModelParameters()

# Low shear stress parameters
low_shear_params = ModelParameters()
low_shear_params.shear_stress = 1.0

# High shear stress parameters
high_shear_params = ModelParameters()
high_shear_params.shear_stress = 20.0

# Physiological parameters
physiological_params = ModelParameters()
physiological_params.shear_stress = 5.0