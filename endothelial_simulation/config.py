"""
Simplified configuration for event-driven endothelial cell simulation.
"""
import os
import time


class SimulationConfig:
    """Clean, simplified configuration for event-driven system."""

    def __init__(self):
        # === CORE SIMULATION PARAMETERS ===
        self.simulation_duration = 360  # minutes (6 hours)
        self.time_step = 1.0  # minutes
        self.time_unit = "minutes"
        self.initial_cell_count = 20
        self.grid_size = (1024, 1024)  # pixels

        # === EVENT-DRIVEN SYSTEM (MAIN FEATURE) ===
        self.use_event_driven_system = True  # Always enabled
        self.biological_optimization_enabled = False  # Always disabled

        # Event detection sensitivity
        self.pressure_change_threshold = 0.1  # Pa
        self.cell_count_change_threshold = 5  # cells
        self.senescence_threshold_change = 0.05  # 5%

        # Reconfiguration timing
        self.min_reconfiguration_interval = 20.0  # minutes

        # Transition parameters
        self.max_compression_ratio = 0.7  # cells can compress to 70%
        self.transition_completion_threshold = 0.95
        self.trajectory_checkpoint_interval = 20.0  # minutes

        # Multi-configuration system
        self.multi_config_count = 10
        self.multi_config_optimization_steps = 3

        # === SIMULATION COMPONENTS ===
        self.enable_temporal_dynamics = True
        self.enable_spatial_properties = True
        self.enable_population_dynamics = True

        # Population features
        self.enable_senescence = True
        self.enable_senolytics = False
        self.enable_stem_cells = False

        # Hole system
        self.enable_holes = True
        self.max_holes = 5
        self.hole_creation_probability_base = 0.02
        self.hole_creation_threshold_cells = 10
        self.hole_creation_threshold_senescence = 0.30

        # === VISUALIZATION ===
        self.plot_interval = 10  # steps
        self.save_plots = True
        self.create_animations = True
        self.plot_directory = self._generate_plot_directory()

        # === DEBUG OPTIONS ===
        self.debug_events = False
        self.debug_transitions = False

        # === TEMPORAL DYNAMICS ===
        self.tau_area_minutes = 45.0
        self.tau_orientation_minutes = 60.0
        self.tau_aspect_ratio_minutes = 50.0

        # === TELOMERE SENESCENCE PARAMETERS ===
        self.max_divisions = 7  # Target number of divisions before senescence
        self.initial_telomere_mean = 100  # Average starting telomere length
        self.initial_telomere_std = 20  # Variability in starting length

        # Calculated automatically to match max_divisions
        self.telomere_loss_per_division = self.initial_telomere_mean / self.max_divisions

        # These are based on experimental data - adjust values as needed
        self.known_pressures = [0.0, 1.4]  # Pressure values in Pa
        self.known_A_max = {
            0.0: 1.0,  # A_max at 0 Pa (baseline)
            1.4: 2.5  # A_max at 1.4 Pa (example value - adjust based on your data)
        }

        # Initial response value
        self.initial_response = 1.0

        # Time constant parameters
        self.tau_base = 1.0  # Base time constant (minutes)
        self.lambda_scale = 0.5  # Lambda scaling parameter

        # === TEMPORAL DYNAMICS PARAMETERS ===
        self.known_pressures = [0.0, 1.4]
        self.known_A_max = {
            0.0: 1.0,  # A_max at 0 Pa (baseline)
            1.4: 2.5  # A_max at 1.4 Pa
        }
        self.initial_response = 1.0
        self.tau_base = 60.0  # Base time constant (minutes)
        self.lambda_scale = 0.3  # Lambda scaling parameter

        # === POPULATION DYNAMICS PARAMETERS ===
        self.proliferation_rate = 0 #0.0006
        self.carrying_capacity = 200
        self.death_rate_healthy = 0 #0.0001
        self.death_rate_senescent_tel = 0 #0.00033
        self.death_rate_senescent_stress = 0 #0.00042
        self.senescence_induction_factor = 0 #0.0000008
        self.senolytic_concentration = 0.0
        self.senolytic_efficacy_tel = 1.0
        self.senolytic_efficacy_stress = 1.2

        # Deterministic senescence parameters
        self.base_cellular_resistance = 0.5  # Base resistance threshold (adjusted for stress_factor scale)
        self.resistance_variability = 0.2  # Cell-to-cell variability


    def _generate_plot_directory(self):
        """Generate timestamped plot directory."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return f"results_event_driven_{timestamp}"

    # === SIMPLE CONFIGURATION METHODS ===

    def set_temporal_only(self):
        """Focus only on temporal dynamics."""
        self.enable_temporal_dynamics = True
        self.enable_spatial_properties = False
        self.enable_population_dynamics = False
        self.enable_senescence = False
        return self

    def set_spatial_only(self):
        """Focus only on spatial properties."""
        self.enable_temporal_dynamics = False
        self.enable_spatial_properties = True
        self.enable_population_dynamics = False
        self.enable_senescence = False
        return self

    def set_population_only(self):
        """Focus only on population dynamics."""
        self.enable_temporal_dynamics = False
        self.enable_spatial_properties = False
        self.enable_population_dynamics = True
        self.enable_senescence = True
        return self


    def set_full_simulation(self):
        """Enable all simulation components."""
        self.enable_temporal_dynamics = True
        self.enable_spatial_properties = True
        self.enable_population_dynamics = True
        self.enable_senescence = True
        return self

    def set_minimal(self):
        """Minimal simulation - basic tessellation only."""
        self.enable_temporal_dynamics = False
        self.enable_spatial_properties = False
        self.enable_population_dynamics = False
        self.enable_senescence = False
        self.enable_holes = False
        return self

    # === EVENT-DRIVEN CONFIGURATION ===

    def set_event_sensitivity(self, pressure=0.1, cells=5, senescence=0.05):
        """Set event detection sensitivity."""
        self.pressure_change_threshold = pressure
        self.cell_count_change_threshold = cells
        self.senescence_threshold_change = senescence
        return self

    def set_transition_params(self, compression=0.7, completion=0.95, interval=20.0):
        """Set transition parameters."""
        self.max_compression_ratio = compression
        self.transition_completion_threshold = completion
        self.trajectory_checkpoint_interval = interval
        return self

    def enable_debug(self, events=True, transitions=True):
        """Enable debug output."""
        self.debug_events = events
        self.debug_transitions = transitions
        return self

    # === QUICK SETUPS ===

    def quick_test(self):
        """Quick test configuration - small, fast."""
        self.initial_cell_count = 20
        self.simulation_duration = 120  # 2 hours
        self.multi_config_count = 5
        self.enable_holes = False
        self.create_animations = True
        return self

    def research_quality(self):
        """High-quality research configuration."""
        self.initial_cell_count = 100
        self.simulation_duration = 720  # 12 hours
        self.multi_config_count = 20
        self.multi_config_optimization_steps = 5
        self.create_animations = True
        return self

    def get_summary(self):
        """Get configuration summary."""
        return {
            'mode': 'event-driven',
            'components': [
                name for name, enabled in [
                    ('temporal', self.enable_temporal_dynamics),
                    ('spatial', self.enable_spatial_properties),
                    ('population', self.enable_population_dynamics),
                    ('senescence', self.enable_senescence),
                    ('holes', self.enable_holes)
                ] if enabled
            ],
            'duration': f"{self.simulation_duration} min ({self.simulation_duration/60:.1f}h)",
            'cells': self.initial_cell_count,
            'pressure_threshold': self.pressure_change_threshold,
            'multi_configs': self.multi_config_count,
            'debug': self.debug_events or self.debug_transitions
        }

    def describe(self):
        """Generate description of configuration."""
        summary = self.get_summary()

        desc = f"""
🧬 Event-Driven Endothelial Cell Simulation Configuration

📊 Simulation Settings:
   Duration: {summary['duration']}
   Initial cells: {summary['cells']}
   Grid size: {self.grid_size[0]}×{self.grid_size[1]}
   Time step: {self.time_step} min

🔄 Event-Driven System:
   Pressure threshold: {self.pressure_change_threshold} Pa
   Min reconfig interval: {self.min_reconfiguration_interval} min
   Max compression: {self.max_compression_ratio}
   Multi-configurations: {self.multi_config_count}

🧪 Enabled Components: {', '.join(summary['components'])}

🐛 Debug: {'ON' if summary['debug'] else 'OFF'}
"""
        return desc.strip()


# === SIMPLE CONFIGURATION CREATION FUNCTIONS ===

def create_full_config():
    """Create full event-driven configuration."""
    return SimulationConfig().set_full_simulation()

def create_temporal_only_config():
    """Create temporal-only configuration."""
    return SimulationConfig().set_temporal_only()

def create_spatial_only_config():
    """Create spatial-only configuration."""
    return SimulationConfig().set_spatial_only()

def create_population_only_config():
    """Create population-only configuration."""
    return SimulationConfig().set_population_only()

def create_minimal_config():
    """Create minimal configuration."""
    return SimulationConfig().set_minimal()

def enable_all(self):
    """Enable all simulation components (backward compatibility method)."""
    return self.set_full_simulation()

def create_test_config():
    """Create quick test configuration."""
    return SimulationConfig().set_full_simulation().quick_test()

def create_research_config():
    """Create research-quality configuration."""
    return SimulationConfig().set_full_simulation().research_quality()
