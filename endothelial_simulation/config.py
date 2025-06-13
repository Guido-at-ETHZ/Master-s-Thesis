"""
Configuration settings for the endothelial cell mechanotransduction simulation.

This module defines the configuration parameters and feature toggles
for the simulation, allowing customization of which model components
are active and their associated parameters.
"""


class SimulationConfig:
    """Configuration settings for the endothelial simulation."""

    def __init__(self):
        """Initialize configuration with default values."""
        # ---------------
        # temporary fix
        # ----------------
        # ----------------
        # Temporal dynamics parameters
        # ----------------
        self.known_pressures = [15, 25, 45]
        self.known_A_max = {15: 1.5, 25: 3.7, 45: 5.3}
        self.initial_response = 1.0
        self.tau_base = 60.0
        self.lambda_scale = 0.8

        # ----------------
        # Population dynamics parameters
        # ----------------
        self.max_divisions = 15
        self.proliferation_rate = 0.0006
        self.carrying_capacity = 3000
        self.death_rate_healthy = 0.0001
        self.death_rate_senescent_tel = 0.00033
        self.death_rate_senescent_stress = 0.00042
        self.senescence_induction_factor = 0.0000008
        self.senolytic_concentration = 5.0
        self.senolytic_efficacy_tel = 1.0
        self.senolytic_efficacy_stress = 1.2
        # ----------------
        # Simulation settings
        # ----------------
        # Duration of simulation in minutes (18 hours = 1080 minutes)
        self.simulation_duration = 1080

        # Time step for numerical integration (in minutes)
        self.time_step = 1.0

        # Size of the 2D grid for cell placement (width, height in pixels)
        self.grid_size = (1024, 1024)

        # Number of cells at the start of simulation
        self.initial_cell_count = 50

        # ----------------
        # Component toggles
        # ----------------
        # Enable/disable population dynamics (cell division, death)
        self.enable_population_dynamics = True

        # Enable/disable spatial properties (cell orientation, aspect ratio)
        self.enable_spatial_properties = True

        # Enable/disable temporal dynamics (time-dependent adaptation)
        self.enable_temporal_dynamics = True

        # Enable/disable senescence mechanisms
        self.enable_senescence = True

        # Enable/disable senolytic effects (drugs that target senescent cells)
        self.enable_senolytics = False

        # Enable/disable stem cell input
        self.enable_stem_cells = False

        # ----------------
        # Visualization settings
        # ----------------
        # Visualization interval (in simulation steps) - every 10 minutes
        self.plot_interval = 10

        # Whether to save plots to files
        self.save_plots = True

        # Directory for visualization output
        self.plot_directory = "results"

        # Whether to create animations
        self.create_animations = True

        # Whether to save detailed metrics to CSV
        self.save_metrics = True

        # Time units label for plot titles and axes
        self.time_unit = "minutes"

        # NEW: Biological optimization parameters
        self.biological_optimization_enabled = True
        self.adaptation_strength = 0.25  # How strongly cells adapt toward targets
        self.max_displacement_per_step = 12.0  # Maximum cell movement per step
        self.global_adaptation_interval = 3  # Steps between global optimizations
        self.convergence_threshold = 0.001  # Energy convergence threshold

        # Energy weights for different properties
        self.energy_weight_area = 1.0
        self.energy_weight_aspect_ratio = 0.5
        self.energy_weight_orientation = 0.5

        # ----------------
        # Hole system parameters
        # ----------------
        self.enable_holes = True
        self.max_holes = 5
        self.hole_creation_probability_base = 0.02  # 2% base probability per timestep
        self.hole_creation_threshold_cells = 10
        self.hole_creation_threshold_senescence = 0.30  # 30%

        # Hole size parameters
        self.hole_size_min_factor = 0.2  # 1/5 of cell size
        self.hole_size_max_factor = 1.0  # Same as cell size

        # Hole compression parameters
        self.hole_compression_reference_density = 15

    def disable_all_but_temporal(self):
        """Configure to focus only on temporal dynamics."""
        self.enable_population_dynamics = False
        self.enable_spatial_properties = False
        self.enable_senescence = False
        self.enable_senolytics = False
        self.enable_stem_cells = False
        self.enable_temporal_dynamics = True

    def disable_all_but_spatial(self):
        """Configure to focus only on spatial properties."""
        self.enable_population_dynamics = False
        self.enable_temporal_dynamics = False
        self.enable_senescence = False
        self.enable_senolytics = False
        self.enable_stem_cells = False
        self.enable_spatial_properties = True

    def enable_minimal_population(self):
        """Configure for basic population dynamics without senescence."""
        self.enable_population_dynamics = True
        self.enable_spatial_properties = False
        self.enable_temporal_dynamics = False
        self.enable_senescence = False
        self.enable_senolytics = False
        self.enable_stem_cells = False

    def enable_all(self):
        """Enable all simulation components."""
        self.enable_population_dynamics = True
        self.enable_spatial_properties = True
        self.enable_temporal_dynamics = True
        self.enable_senescence = True
        self.enable_senolytics = True
        self.enable_stem_cells = True

    def describe(self):
        """Generate a text description of the current configuration."""
        status = []
        status.append("Endothelial Simulation Configuration:")
        status.append("--------------------------------")
        status.append(
            f"Simulation duration: {self.simulation_duration} {self.time_unit} ({self.simulation_duration / 60:.1f} hours)")
        status.append(f"Time step: {self.time_step} {self.time_unit}")
        status.append(f"Grid size: {self.grid_size[0]} x {self.grid_size[1]} pixels")
        status.append(f"Initial cells: {self.initial_cell_count}")
        status.append(f"Visualization interval: Every {self.plot_interval} {self.time_unit}")
        status.append("\nActive Components:")
        status.append(f"- Population dynamics: {'✓' if self.enable_population_dynamics else '✗'}")
        status.append(f"- Spatial properties: {'✓' if self.enable_spatial_properties else '✗'}")
        status.append(f"- Temporal dynamics: {'✓' if self.enable_temporal_dynamics else '✗'}")
        status.append(f"- Senescence: {'✓' if self.enable_senescence else '✗'}")
        status.append(f"- Senolytics: {'✓' if self.enable_senolytics else '✗'}")
        status.append(f"- Stem cells: {'✓' if self.enable_stem_cells else '✗'}")
        status.append(f"- Holes: {'✓' if self.enable_holes else '✗'}")

        return "\n".join(status)

    def get_time_in_hours(self, minutes):
        """Convert simulation time from minutes to hours for display."""
        return minutes / 60.0


# Create a default configuration instance
default_config = SimulationConfig()


# Example configurations for different simulation types
def create_temporal_only_config():
    """Create a configuration focused only on temporal dynamics."""
    config = SimulationConfig()
    config.disable_all_but_temporal()
    config.plot_directory = "results_temporal"
    return config


def create_spatial_only_config():
    """Create a configuration focused only on spatial properties."""
    config = SimulationConfig()
    config.disable_all_but_spatial()
    config.plot_directory = "results_spatial"
    return config


def create_full_config():
    """Create a configuration with all components enabled."""
    config = SimulationConfig()
    #config.enable_all()
    return config

