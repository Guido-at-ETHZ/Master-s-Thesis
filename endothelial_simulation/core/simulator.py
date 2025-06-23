"""
Complete event-driven simulator implementation with all missing functionality restored.
This version includes all the original features while using the event-driven system.
"""
import numpy as np
import time
import os
import random
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from endothelial_simulation.core.cell import Cell
from endothelial_simulation.core.grid import Grid
from endothelial_simulation.models.temporal_dynamics import TemporalDynamicsModel
from endothelial_simulation.models.population_dynamics import PopulationDynamicsModel
from endothelial_simulation.models.spatial_properties import SpatialPropertiesModel
from endothelial_simulation.visualization import Plotter
from endothelial_simulation.visualization.animations import create_detailed_cell_animation, create_metrics_animation


class Simulator:
    """
    Complete event-driven simulator with all original functionality restored.
    """

    def __init__(self, config):
        """Initialize the simulator with configuration parameters."""
        # Set random seed for reproducibility
        import time, random, numpy as np
        seed = int(time.time_ns() % (2 ** 32))
        random.seed(seed)
        np.random.seed(seed)

        self.config = config

        # Create grid and disable force-based optimization
        self.grid = Grid(
            width=config.grid_size[0],
            height=config.grid_size[1],
            config=config
        )

        # Disable continuous biological adaptation
        self.grid.biological_optimization_enabled = False
        self.grid.continuous_adaptation_disabled = True

        # Enable energy tracking if biological optimization was enabled
        if getattr(config, 'biological_optimization_enabled', True):
            print("🔋 Enabling automatic energy tracking for biological optimization...")
            self.grid.enable_energy_tracking()
            self.energy_tracking_enabled = True
        else:
            self.energy_tracking_enabled = False

        # Initialize model components
        self.models = {}

        if config.enable_temporal_dynamics:
            self.models['temporal'] = TemporalDynamicsModel(config)

        if config.enable_population_dynamics:
            self.models['population'] = PopulationDynamicsModel(config)

        if config.enable_spatial_properties:
            temporal_model = self.models.get('temporal', None)
            self.models['spatial'] = SpatialPropertiesModel(config, temporal_model)

        # Initialize event-driven components
        from endothelial_simulation.core.event_system import EventDetector
        from endothelial_simulation.management.configuration_manager import ConfigurationManager
        from endothelial_simulation.management.transition_controller import TransitionController

        self.event_detector = EventDetector(config)
        self.configuration_manager = ConfigurationManager(self.grid, config)
        self.transition_controller = TransitionController(
            self.grid,
            temporal_model=self.models.get('temporal', None)
        )

        # Give configuration manager access to current conditions
        self.configuration_manager.simulator = self

        # Simulation state
        self.time = 0.0
        self.step_count = 0
        self.history = []

        # Input pattern information
        self.input_pattern = {
            'type': 'constant',
            'value': 0.0,
            'params': {}
        }

        # Event processing
        self.pending_events = []
        self.event_history = []
        self.last_reconfiguration_time = 0.0

        # Configuration tracking
        self.configuration_history = []
        self.current_configuration_id = 0

        # Animation settings
        self.record_frames = config.create_animations
        self.frame_data = []
        self.record_interval = 10

        # Mosaic-specific parameters (kept for compatibility)
        self.tessellation_update_interval = 5
        self.position_optimization_interval = 20
        self.last_tessellation_update = 0
        self.last_position_optimization = 0

    # =============================================================================
    # INITIALIZATION METHODS
    # =============================================================================

    def initialize(self, cell_count=None):
        """Initialize with standard single configuration."""
        if cell_count is None:
            cell_count = self.config.initial_cell_count

        print(f"🔄 Initializing event-driven simulation with {cell_count} cells...")

        # Calculate base area per cell
        total_area = self.grid.width * self.grid.height
        base_area_per_cell = total_area / cell_count

        def area_distribution():
            return np.random.uniform(base_area_per_cell * 0.7, base_area_per_cell * 1.3)

        # Populate grid with initial cells
        self.grid.populate_grid(cell_count, area_distribution=area_distribution)

        # Initialize cell properties for current pressure
        self._initialize_cell_properties_for_pressure()

        # Initial adaptation
        self.grid.adapt_cell_properties()

        # Record initial state
        self._record_state()

        # Record initial energy state if tracking is enabled
        if self.energy_tracking_enabled:
            print("🔋 Recording initial energy state...")
            self.grid.record_energy_state(self.step_count, label="initialization")

        print(f"✅ Initialized with energy: {self.grid.calculate_biological_energy():.4f}")

    def initialize_with_multiple_configurations(self, cell_count=None, num_configurations=10,
                                                optimization_iterations=3, save_analysis=True):
        """
        Initialize by testing multiple configurations and selecting the best one.

        Parameters:
            cell_count: Number of cells to create (default: from config)
            num_configurations: Number of configurations to test
            optimization_iterations: Optimization steps per configuration
            save_analysis: Whether to save detailed analysis

        Returns:
            Dictionary with configuration selection results
        """
        if cell_count is None:
            cell_count = self.config.initial_cell_count

        print(f"🚀 Initializing simulation with multi-configuration selection:")
        print(f"   Target cells: {cell_count}")
        print(f"   Configurations to test: {num_configurations}")
        print(f"   Optimization iterations per config: {optimization_iterations}")

        # Calculate base area per cell for distribution
        total_area = self.grid.width * self.grid.height
        base_area_per_cell = total_area / cell_count

        def area_distribution():
            return np.random.uniform(base_area_per_cell * 0.7, base_area_per_cell * 1.3)

        def division_distribution():
            max_div = self.config.max_divisions
            r = np.random.random()
            return int(max_div * (1 - np.sqrt(r)))

        # Generate and test multiple configurations
        config_results = self.grid.generate_multiple_initial_configurations(
            cell_count=cell_count,
            num_configurations=num_configurations,
            division_distribution=division_distribution,
            area_distribution=area_distribution,
            optimization_iterations=optimization_iterations,
            verbose=True
        )

        # Initialize cell properties for current pressure
        self._initialize_cell_properties_for_pressure()

        # Final adaptation
        self.grid.adapt_cell_properties()

        # Record initial state
        self._record_state()

        # Record initial energy state if tracking is enabled
        if self.energy_tracking_enabled:
            print("🔋 Recording initial energy state...")
            self.grid.record_energy_state(self.step_count, label="initialization_best_config")

        # Save analysis if requested
        if save_analysis and hasattr(self.grid, 'save_configuration_analysis'):
            self.grid.save_configuration_analysis(config_results)

        print(f"\n✅ Initialization complete with best configuration selected!")
        print(f"   Final energy: {config_results['best_config']['energy']:.4f}")
        print(f"   Energy improvement: {config_results['energy_improvement']:.4f}")

        return config_results

    def initialize_smart(self, cell_count=None, **kwargs):
        """
        Smart initialization that automatically chooses between single and multi-configuration.
        """
        if cell_count is None:
            cell_count = self.config.initial_cell_count

        # Use multi-configuration for larger simulations or if requested
        use_multi_config = (
                cell_count >= 50 or  # Large simulations benefit more
                kwargs.get('force_multi_config', False) or
                getattr(self.config, 'use_multi_config_init', False)
        )

        if use_multi_config:
            # Set reasonable defaults based on simulation size
            default_configs = min(20, max(5, cell_count // 10))
            kwargs.setdefault('num_configurations', default_configs)
            kwargs.setdefault('optimization_iterations', 3)

            return self.initialize_with_multiple_configurations(cell_count, **kwargs)
        else:
            print(f"🚀 Using standard initialization for {cell_count} cells")
            return self.initialize(cell_count)

    def _initialize_cell_properties_for_pressure(self):
        """Initialize cell properties for current pressure."""
        current_pressure = self.input_pattern.get('value', 0.0)

        if 'spatial' not in self.models:
            return

        spatial_model = self.models['spatial']

        for cell_id, cell in self.grid.cells.items():
            target_area = spatial_model.calculate_target_area(
                current_pressure, cell.is_senescent, cell.senescence_cause
            )
            target_aspect_ratio = spatial_model.calculate_target_aspect_ratio(
                current_pressure, cell.is_senescent
            )
            target_orientation = spatial_model.calculate_target_orientation(
                current_pressure, cell.is_senescent
            )

            cell.target_area = target_area
            cell.target_aspect_ratio = target_aspect_ratio
            cell.target_orientation = target_orientation
            cell.actual_aspect_ratio = target_aspect_ratio
            cell.actual_orientation = target_orientation

        self.grid._update_voronoi_tessellation()

    # =============================================================================
    # INPUT PATTERN METHODS
    # =============================================================================

    def set_step_input(self, initial_value, final_value, step_time):
        """Set step input pattern."""
        self.input_pattern = {
            'type': 'step',
            'value': initial_value,
            'params': {
                'initial_value': initial_value,
                'final_value': final_value,
                'step_time': step_time
            }
        }

    def set_ramp_input(self, initial_value, final_value, ramp_start_time, ramp_end_time):
        """Set a ramp input pattern."""
        self.input_pattern = {
            'type': 'ramp',
            'value': initial_value,
            'params': {
                'initial_value': initial_value,
                'final_value': final_value,
                'ramp_start_time': ramp_start_time,
                'ramp_end_time': ramp_end_time
            }
        }

    def set_oscillatory_input(self, base_value, amplitude, frequency, phase=0):
        """Set an oscillatory input pattern."""
        self.input_pattern = {
            'type': 'oscillatory',
            'value': base_value,
            'params': {
                'base_value': base_value,
                'amplitude': amplitude,
                'frequency': frequency,
                'phase': phase
            }
        }

    def set_multi_step_input(self, step_schedule):
        """
        Set a multi-step input pattern with multiple step changes.

        Parameters:
            step_schedule: List of (time, value) tuples defining the schedule
        """
        self.input_pattern = {
            'type': 'multi_step',
            'value': step_schedule[0][1] if step_schedule else 0.0,
            'params': {
                'schedule': step_schedule
            }
        }

    def set_protocol_input(self, protocol_name, **kwargs):
        """
        Set a predefined protocol input pattern.

        Parameters:
            protocol_name: Name of the protocol to use
            **kwargs: Protocol-specific parameters
        """
        protocols = {
            'acute_stress': [(0, 0), (30, 2.0), (90, 0)],
            'chronic_stress': [(0, 0), (60, 1.0), (360, 1.0), (420, 0)],
            'stepwise_increase': [(0, 0), (60, 0.5), (120, 1.0), (180, 1.5), (240, 2.0)],
            'oscillatory_low': [(0, 0), (30, 1.0), (90, 0), (120, 1.0), (180, 0)],
            'high_stress_brief': [(0, 0), (45, 3.0), (75, 0)]
        }

        if protocol_name not in protocols:
            raise ValueError(f"Unknown protocol: {protocol_name}")

        # Get base schedule
        schedule = protocols[protocol_name].copy()

        # Apply scaling if requested
        scale_time = kwargs.get('scale_time', 1.0)
        scale_stress = kwargs.get('scale_stress', 1.0)
        max_stress = kwargs.get('max_stress', None)

        if scale_time != 1.0 or scale_stress != 1.0 or max_stress is not None:
            scaled_schedule = []
            for time_point, stress_value in schedule:
                new_time = time_point * scale_time
                new_stress = stress_value * scale_stress

                if max_stress is not None and new_stress > max_stress:
                    new_stress = max_stress

                scaled_schedule.append((new_time, new_stress))

            schedule = scaled_schedule

        self.set_multi_step_input(schedule)

    def update_input_value(self):
        """Update the current input value based on the input pattern and current time."""
        pattern_type = self.input_pattern['type']
        params = self.input_pattern['params']

        if pattern_type == 'constant':
            self.input_pattern['value'] = params.get('value', 0.0)

        elif pattern_type == 'step':
            if self.time < params['step_time']:
                self.input_pattern['value'] = params['initial_value']
            else:
                self.input_pattern['value'] = params['final_value']

        elif pattern_type == 'multi_step':
            schedule = params['schedule']
            current_value = schedule[0][1]  # Default to first value

            for i, (step_time, step_value) in enumerate(schedule):
                if self.time >= step_time:
                    current_value = step_value
                else:
                    break

            if self.input_pattern['value'] != current_value:
                old_value = self.input_pattern['value']
                self.input_pattern['value'] = current_value
                print(f"Step change at t={self.time:.1f}min: {old_value:.2f} → {current_value:.2f} Pa")

        elif pattern_type == 'ramp':
            if self.time < params['ramp_start_time']:
                self.input_pattern['value'] = params['initial_value']
            elif self.time > params['ramp_end_time']:
                self.input_pattern['value'] = params['final_value']
            else:
                progress = (self.time - params['ramp_start_time']) / (
                        params['ramp_end_time'] - params['ramp_start_time'])
                self.input_pattern['value'] = params['initial_value'] + progress * (
                        params['final_value'] - params['initial_value'])

        elif pattern_type == 'oscillatory':
            omega = 2 * np.pi * params['frequency']
            self.input_pattern['value'] = params['base_value'] + params['amplitude'] * np.sin(
                omega * self.time + params['phase'])

        return self.input_pattern['value']

    def _get_current_input(self):
        """Get current input value (for compatibility)."""
        return self.update_input_value()

    # =============================================================================
    # SIMULATION EXECUTION METHODS
    # =============================================================================

    def step(self, dt=None):
        """Execute one simulation step using event-driven approach."""
        if dt is None:
            dt = self.config.time_step

        self.time += dt
        self.step_count += 1
        current_input = self.update_input_value()

        # Detect events (pressure changes, senescence, holes)
        events = self.event_detector.detect_events(
            current_time=self.time,
            current_pressure=current_input,
            grid=self.grid,
            last_check_time=self.time - dt
        )

        # Process any detected events
        for event in events:
            self._process_event(event)

        # Update active transitions
        if self.transition_controller.is_transitioning():
            self.transition_controller.update_transition(self.time, dt)

        # Apply shear stress
        self._apply_shear_stress(current_input, dt)

        # Update models (population dynamics, temporal dynamics)
        if 'temporal' in self.models and self.config.enable_temporal_dynamics:
            model = self.models['temporal']
            model.update_cell_responses(self.grid.cells, current_input, dt)

        # Update spatial properties if in transition
        if 'spatial' in self.models and self.config.enable_spatial_properties:
            model = self.models['spatial']
            # Set transition mode if transitioning
            model._in_transition_mode = self.transition_controller.is_transitioning()

            for cell in self.grid.cells.values():
                dynamics_result = model.update_cell_properties(cell, current_input, dt, self.grid.cells)

        # Update population dynamics
        if 'population' in self.models and self.config.enable_population_dynamics:
            model = self.models['population']
            stem_cell_rate = 10 if self.config.enable_stem_cells else 0
            model.update_from_cells(self.grid.cells, dt, current_input, stem_cell_rate)
            actions = model.synchronize_cells(self.grid.cells)
            self._execute_population_actions(actions)

        # Update hole system
        if hasattr(self.grid, 'update_holes'):
            self.grid.update_holes(dt)

        # Record frames for animation
        if self.record_frames and self.step_count % self.record_interval == 0:
            self._record_frame()

        # Record state
        self._record_state()

        return {
            'time': self.time,
            'step_count': self.step_count,
            'cell_count': len(self.grid.cells),
            'input_value': current_input,
            'transitioning': self.transition_controller.is_transitioning()
        }

    def run(self, duration=None):
        """
        Run simulation for specified duration.

        Parameters:
            duration: Duration to run in simulation time units (default: from config)

        Returns:
            Dictionary with simulation results
        """
        if duration is None:
            duration = self.config.simulation_duration

        # Calculate number of steps
        dt = self.config.time_step
        num_steps = int(duration / dt)
        target_time = self.time + duration

        print(f"🚀 Running event-driven simulation for {duration} minutes ({num_steps} steps)...")
        start_time = time.time()

        # Run steps
        for i in range(num_steps):
            if self.time >= target_time:
                break

            step_info = self.step(dt)

            # Print progress periodically
            if (i + 1) % 100 == 0 or i == num_steps - 1:
                progress = min((i + 1) / num_steps * 100, 100)
                elapsed = time.time() - start_time
                estimated_total = elapsed / (i + 1) * num_steps if i > 0 else elapsed
                remaining = max(0, estimated_total - elapsed)

                # Get current grid statistics
                grid_stats = self.grid.get_grid_statistics()
                packing_eff = grid_stats.get('packing_efficiency', 0)

                print(f"Progress: {progress:.1f}% (Step {i + 1}/{num_steps}), "
                      f"Time: {elapsed:.1f}s, Remaining: {remaining:.1f}s, "
                      f"Cells: {step_info['cell_count']}, "
                      f"Packing: {packing_eff:.2f}, "
                      f"Transitioning: {step_info['transitioning']}")

        end_time = time.time()
        total_time = end_time - start_time

        print(f"✅ Event-driven simulation completed in {total_time:.1f} seconds")

        # Create animations if enabled
        if self.record_frames and self.frame_data:
            print("🎬 Creating animations...")
            self._create_animations()

        # Return results
        return {
            'duration': duration,
            'steps': num_steps,
            'final_time': self.time,
            'execution_time': total_time,
            'history': self.history,
            'time_points': [state['time'] for state in self.history],
            'animations_created': self.record_frames and len(self.frame_data) > 0,
            'final_grid_stats': self.grid.get_grid_statistics(),
            'configuration_history': getattr(self, 'configuration_history', []),
            'event_history': getattr(self, 'event_history', [])
        }

    # =============================================================================
    # EVENT-DRIVEN LOGIC
    # =============================================================================

    def _process_event(self, event):
        """Process a detected event by triggering reconfiguration."""
        print(f"🔍 Processing event: {event.event_type.name} at t={self.time/60:.1f}h")

        # Check minimum interval between reconfigurations
        time_since_last = self.time - self.last_reconfiguration_time
        min_interval = getattr(self.config, 'min_reconfiguration_interval', 30.0)

        if time_since_last < min_interval:
            print(f"   ⏳ Skipping - too soon (last: {time_since_last:.1f}min ago)")
            return

        # Generate new configuration for current conditions
        current_pressure = self.update_input_value()
        hole_count = len(self.grid.holes) if hasattr(self.grid, 'holes') else 0
        cell_counts = self.grid.count_cells_by_type()
        senescent_count = cell_counts.get('telomere_senescent', 0) + cell_counts.get('stress_senescent', 0)

        new_config = self.configuration_manager.generate_configuration_for_conditions(
            pressure=current_pressure,
            hole_count=hole_count,
            senescent_count=senescent_count
        )

        if new_config:
            # Start transition to new configuration
            self.transition_controller.start_transition(
                target_config=new_config,
                current_time=self.time
            )

            self.last_reconfiguration_time = self.time

            # Record event
            event_record = {
                'time': self.time,
                'event': event.event_type.name,
                'pressure': current_pressure,
                'energy': new_config.energy,
                'cell_count': len(self.grid.cells)
            }

            self.configuration_history.append(event_record)
            self.event_history.append(event_record)

            print(f"   ✅ Started transition to new configuration (energy: {new_config.energy:.4f})")

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def _apply_shear_stress(self, shear_stress, duration):
        """Apply shear stress to all cells."""
        if hasattr(self.grid, 'apply_shear_stress_field'):
            def shear_stress_function(x, y):
                return shear_stress
            self.grid.apply_shear_stress_field(shear_stress_function, duration)

    def _execute_population_actions(self, actions):
        """Execute population actions."""
        for action in actions:
            if action['type'] == 'add_cell':
                self.grid.add_cell(**action['params'])
            elif action['type'] == 'remove_cell':
                self.grid.remove_cell(action['params']['cell_id'])

    # =============================================================================
    # STATE RECORDING AND ANALYSIS
    # =============================================================================

    def _record_state(self):
        """Record current simulation state."""
        # Get cell properties
        cell_properties = self.grid.get_cell_properties()

        # Build basic state
        state = {
            'time': self.time,
            'step_count': self.step_count,
            'cell_count': len(self.grid.cells),
            'input_value': self.update_input_value(),
            'biological_energy': self.grid.calculate_biological_energy(),
            'is_transitioning': self.transition_controller.is_transitioning() if hasattr(self, 'transition_controller') else False
        }

        # Add cell properties
        state.update(cell_properties)

        # Add population dynamics if enabled
        if 'population' in self.models:
            pop_model = self.models['population']
            totals = pop_model.calculate_total_cells()
            avg_div = pop_model.calculate_average_division_age()
            tel_len = pop_model.calculate_telomere_length()

            state.update({
                'healthy_cells': totals['E_total'],
                'senescent_tel': totals['S_tel'],
                'senescent_stress': totals['S_stress'],
                'avg_division_age': avg_div,
                'telomere_length': tel_len
            })

        # Add spatial properties if enabled
        if 'spatial' in self.models:
            spatial_model = self.models['spatial']
            collective_props = spatial_model.calculate_collective_properties(
                self.grid.cells, self.update_input_value()
            )
            state.update(collective_props)

            alignment = spatial_model.calculate_alignment_index(self.grid.cells)
            shape_index = spatial_model.calculate_shape_index(self.grid.cells)
            packing_quality = spatial_model.calculate_packing_quality(self.grid.cells)

            state.update({
                'alignment_index': alignment,
                'shape_index': shape_index,
                'packing_quality': packing_quality,
                'confluency': self.grid.calculate_confluency()
            })

        # Calculate adaptation metrics
        if len(self.grid.cells) > 0:
            target_areas = cell_properties['target_areas']
            target_ars = cell_properties['target_aspect_ratios']
            actual_areas = cell_properties['areas']
            actual_ars = cell_properties['aspect_ratios']

            # Mean adaptation errors
            area_adaptation_error = np.mean([abs(t - a) / max(t, 1) for t, a in zip(target_areas, actual_areas)])
            ar_adaptation_error = np.mean([abs(t - a) / max(t, 1) for t, a in zip(target_ars, actual_ars)])

            state.update({
                'mean_target_area': np.mean(target_areas),
                'std_target_area': np.std(target_areas),
                'mean_target_aspect_ratio': np.mean(target_ars),
                'std_target_aspect_ratio': np.std(target_ars),
                'area_adaptation_error': area_adaptation_error,
                'ar_adaptation_error': ar_adaptation_error,
            })

        self.history.append(state)

    def _record_frame(self):
        """Record frame data for animation."""
        # Collect cell data for this frame
        cells_data = []

        for cell_id, cell in self.grid.cells.items():
            cells_data.append({
                'cell_id': cell_id,
                'position': cell.position,
                'orientation': cell.actual_orientation,
                'aspect_ratio': cell.actual_aspect_ratio,
                'area': cell.actual_area,
                'is_senescent': cell.is_senescent,
                'senescence_cause': cell.senescence_cause
            })

        # Store frame data
        frame_info = {
            'time': self.time,
            'input_value': self.input_pattern['value'],
            'cell_count': len(self.grid.cells),
            'cells': cells_data,
            'transitioning': self.transition_controller.is_transitioning()
        }

        self.frame_data.append(frame_info)

    # =============================================================================
    # RESULTS AND ANALYSIS METHODS
    # =============================================================================

    def save_results(self, filename):
        """Save simulation results to file."""
        # Ensure directory exists
        save_dir = self.config.plot_directory
        os.makedirs(save_dir, exist_ok=True)

        # Create full path
        if not filename.endswith('.npz'):
            filename += '.npz'
        filepath = os.path.join(save_dir, filename)

        # Prepare data for saving
        save_data = {
            'history': np.array(self.history, dtype=object),
            'time_points': np.array([state['time'] for state in self.history]),
            'configuration_history': np.array(getattr(self, 'configuration_history', []), dtype=object),
            'event_history': np.array(getattr(self, 'event_history', []), dtype=object),
            'final_stats': self.grid.get_grid_statistics(),
            'config_params': {
                'duration': self.time,
                'cell_count': len(self.grid.cells),
                'grid_size': self.config.grid_size,
                'time_step': self.config.time_step
            }
        }

        # Add frame data if available
        if self.frame_data:
            save_data['frame_data'] = np.array(self.frame_data, dtype=object)

        # Save to file
        np.savez_compressed(filepath, **save_data)
        print(f"💾 Results saved to: {filepath}")

        return filepath

    def get_best_config_parameters(self, save_excel=False, excel_path=None):
        """
        Show parameters for the best configuration.
        """
        if not hasattr(self, '_config_results') or not self._config_results:
            print("❌ No configuration results available.")
            return None

        config_results = self._config_results
        best_config = config_results['best_config']

        # Extract parameters from best configuration
        params = {
            'config_id': best_config['config_id'],
            'energy': best_config['energy'],
            'cell_count': len(best_config['cells']),
            'cells': []
        }

        # Get individual cell parameters
        for cell_id, cell_data in best_config['cells'].items():
            cell_params = {
                'cell_id': cell_id,
                'area': cell_data.get('area', 0),
                'aspect_ratio': cell_data.get('aspect_ratio', 1.0),
                'orientation_degrees': np.degrees(cell_data.get('orientation', 0)),
                'divisions': cell_data.get('divisions', 0),
                'is_senescent': cell_data.get('is_senescent', False)
            }
            params['cells'].append(cell_params)

        # Calculate averages
        areas = [cell['area'] for cell in params['cells']]
        ars = [cell['aspect_ratio'] for cell in params['cells']]
        orientations = [cell['orientation_degrees'] for cell in params['cells']]

        params['averages'] = {
            'area': np.mean(areas),
            'aspect_ratio': np.mean(ars),
            'orientation_degrees': np.mean(orientations)
        }

        # Save to Excel if requested
        if save_excel:
            try:
                import pandas as pd

                if excel_path is None:
                    excel_path = os.path.join(self.config.plot_directory, 'best_config_parameters.xlsx')

                # Create DataFrame
                df = pd.DataFrame(params['cells'])

                # Save to Excel
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Cell_Parameters', index=False)

                    # Add summary sheet
                    summary_df = pd.DataFrame([params['averages']], index=['Average'])
                    summary_df.to_excel(writer, sheet_name='Summary')

                print(f"📊 Parameters saved to Excel: {excel_path}")

            except ImportError:
                print("⚠️  pandas not available - Excel export skipped")

        return params

    # =============================================================================
    # ANIMATION AND VISUALIZATION METHODS
    # =============================================================================

    def _create_animations(self):
        """Create animations for the simulation."""
        if not self.frame_data:
            print("⚠️  No frame data available for animation")
            return

        print(f"🎬 Creating animations from {len(self.frame_data)} frames...")

        try:
            # Create plotter
            plotter = Plotter(self.config)

            # Create detailed cell animation
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            animation_path = os.path.join(self.config.plot_directory, f"cell_animation_{timestamp}.mp4")

            ani = create_detailed_cell_animation(
                plotter, self.frame_data, self,
                save_path=animation_path,
                fps=10, dpi=100
            )

            if ani:
                print(f"✅ Cell animation saved to: {animation_path}")

            # Create metrics animation if history is available
            if self.history:
                metrics_path = os.path.join(self.config.plot_directory, f"metrics_animation_{timestamp}.mp4")
                metrics_ani = create_metrics_animation(
                    plotter, self.history,
                    save_path=metrics_path,
                    fps=10
                )

                if metrics_ani:
                    print(f"✅ Metrics animation saved to: {metrics_path}")

        except Exception as e:
            print(f"⚠️  Animation creation failed: {e}")

    def plot_energy_evolution(self, save_path=None):
        """Plot energy evolution if tracking is enabled."""
        if self.energy_tracking_enabled and hasattr(self.grid, 'plot_energy_evolution'):
            return self.grid.plot_energy_evolution(save_path)
        else:
            print("Energy tracking not enabled")
            return None

    def create_comprehensive_plots(self, save_individual=True, prefix="simulation"):
        """Create comprehensive plots using the Plotter class."""
        try:
            plotter = Plotter(self.config)
            figures = plotter.create_comprehensive_plots(self.history, save_individual=save_individual)
            print(f"📊 Created {len(figures)} comprehensive plots")
            return figures
        except Exception as e:
            print(f"⚠️  Plot creation failed: {e}")
            return []