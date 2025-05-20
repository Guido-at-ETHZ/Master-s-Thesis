"""
Core module for the Cell class that represents a single endothelial cell.
"""
import numpy as np


class Cell:
    """
    Class representing a single endothelial cell in the simulation.
    """

    def __init__(self, cell_id, position=(0, 0), divisions=0, is_senescent=False, senescence_cause=None):
        """
        Initialize a cell with its properties.

        Parameters:
            cell_id: Unique identifier for the cell
            position: (x, y) coordinates of the cell center
            divisions: Number of divisions the cell has undergone
            is_senescent: Boolean indicating if the cell is senescent
            senescence_cause: 'telomere' or 'stress' indicating the cause of senescence
        """
        # Basic cell properties
        self.cell_id = cell_id
        self.position = position
        self.divisions = divisions
        self.is_senescent = is_senescent
        self.senescence_cause = senescence_cause

        # Cell shape and orientation properties
        self.orientation = 0.0  # Angle in radians (0 = aligned with flow)
        self.aspect_ratio = 1.0  # Ratio of major axis to minor axis
        self.area = 100.0  # Cell area in square pixels
        self.perimeter = 0.0

        # Cell state properties
        self.age = 0.0  # Time since creation or last division
        self.adhesion_strength = 1.0  # Relative adhesion strength
        self.response = 1.0  # Current response level for temporal dynamics

        # Cell territory
        self.boundary_points = []  # List of (x, y) coordinates defining the cell boundary

        # Cell neighbors
        self.neighbors = []  # List of cell_ids of neighboring cells

        # Mechanical properties
        self.local_shear_stress = 0.0  # Local wall shear stress experienced by this cell
        self.stress_exposure_time = 0.0  # Cumulative time exposed to high stress

    def update_shape(self, orientation, aspect_ratio, area):
        """
        Update the cell shape properties.

        Parameters:
            orientation: Angle in radians
            aspect_ratio: Ratio of major axis to minor axis
            area: Cell area in square pixels
        """
        self.orientation = orientation
        self.aspect_ratio = aspect_ratio
        self.area = area

        # Calculate perimeter approximation based on ellipse formula
        a = np.sqrt(self.area * self.aspect_ratio)  # Semi-major axis
        b = self.area / a  # Semi-minor axis

        # Ramanujan's approximation for ellipse perimeter
        h = ((a - b) / (a + b)) ** 2
        self.perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))

    def update_response(self, new_response):
        """
        Update the cell's temporal response value.

        Parameters:
            new_response: New response value
        """
        self.response = new_response

    def update_position(self, new_position):
        """
        Update the cell's position.

        Parameters:
            new_position: New (x, y) coordinates
        """
        self.position = new_position

    def increment_age(self, time_step):
        """
        Increment the cell's age by the given time step.

        Parameters:
            time_step: Time increment in simulation units
        """
        self.age += time_step

    def divide(self):
        """
        Perform cell division, increasing the division count.

        Returns:
            Boolean indicating if division was successful
        """
        if self.is_senescent:
            return False

        self.divisions += 1
        self.age = 0.0
        return True

    def induce_senescence(self, cause):
        """
        Induce senescence in the cell.

        Parameters:
            cause: 'telomere' or 'stress' indicating the cause of senescence

        Returns:
            Boolean indicating if senescence induction was successful
        """
        if self.is_senescent:
            return False

        self.is_senescent = True
        self.senescence_cause = cause
        return True

    def apply_shear_stress(self, shear_stress, duration):
        """
        Apply shear stress to the cell for the given duration.

        Parameters:
            shear_stress: Magnitude of wall shear stress (Pa)
            duration: Duration of exposure in simulation time units
        """
        self.local_shear_stress = shear_stress
        self.stress_exposure_time += duration

    def calculate_senescence_probability(self, config):
        """
        Calculate probability of senescence based on cell state and conditions.

        Parameters:
            config: SimulationConfig object with parameter settings

        Returns:
            Dictionary with probabilities for different senescence causes
        """
        if self.is_senescent:
            return {'telomere': 0.0, 'stress': 0.0}

        # Telomere-induced senescence probability
        tel_prob = 0.0
        if self.divisions >= config.max_divisions:
            tel_prob = 1.0
        elif self.divisions > 0.7 * config.max_divisions:
            # Increasing probability as divisions approach max
            tel_prob = ((self.divisions - 0.7 * config.max_divisions) /
                        (0.3 * config.max_divisions)) * 0.5

        # Stress-induced senescence probability
        stress_factor = self._calculate_stress_factor(config)
        stress_prob = min(stress_factor * self.stress_exposure_time * config.time_step, 0.95)

        return {'telomere': tel_prob, 'stress': stress_prob}

    def _calculate_stress_factor(self, config):
        """
        Calculate stress factor based on shear stress magnitude.

        Parameters:
            config: SimulationConfig object with parameter settings

        Returns:
            Stress factor for senescence induction
        """
        tau = self.local_shear_stress

        # Implementation of stress factor from thesis model
        if tau <= 10:  # Low shear stress
            return 0.002 + tau * 0.0005  # Very minor effect
        elif tau <= 20:  # Moderate shear stress
            return 0.007 + (tau - 10) * 0.001  # Slight increase
        else:  # High shear stress
            return 0.017 + (tau - 20) * 0.005  # More rapid increase

    def get_state_dict(self):
        """
        Get a dictionary representation of the cell state.

        Returns:
            Dictionary with all cell properties
        """
        return {
            'cell_id': self.cell_id,
            'position': self.position,
            'divisions': self.divisions,
            'is_senescent': self.is_senescent,
            'senescence_cause': self.senescence_cause,
            'orientation': self.orientation,
            'aspect_ratio': self.aspect_ratio,
            'area': self.area,
            'perimeter': self.perimeter,
            'age': self.age,
            'adhesion_strength': self.adhesion_strength,
            'response': self.response,
            'local_shear_stress': self.local_shear_stress,
            'stress_exposure_time': self.stress_exposure_time
        }