"""
Main simulator class for endothelial cell mechanotransduction.
"""
import numpy as np
import time
import os
from endothelial_simulation.core.cell import Cell
from endothelial_simulation.core.grid import Grid
from endothelial_simulation.models.temporal_dynamics import TemporalDynamicsModel
from endothelial_simulation.models.population_dynamics import PopulationDynamicsModel
from endothelial_simulation.models.spatial_properties import SpatialPropertiesModel
# Add imports for animation
from endothelial_simulation.visualization import Plotter
from endothelial_simulation.visualization.animations import create_detailed_cell_animation, create_metrics_animation


class Simulator:
    """
    Main simulator class that integrates different model components and handles time evolution.
    """

    def __init__(self, config):
        """
        Initialize the simulator with configuration parameters.

        Parameters:
            config: SimulationConfig object with parameter settings
        """
        self.config = config

        # Create grid for spatial representation
        self.grid = Grid(
            width=config.grid_size[0],
            height=config.grid_size[1],
            config=config
        )

        # Initialize model components based on configuration
        self.models = {}

        if config.enable_temporal_dynamics:
            self.models['temporal'] = TemporalDynamicsModel(config)

        if config.enable_population_dynamics:
            self.models['population'] = PopulationDynamicsModel(config)

        if config.enable_spatial_properties:
            self.models['spatial'] = SpatialPropertiesModel(config)

        # Simulation state
        self.time = 0.0
        self.step_count = 0
        self.history = []

        # Input pattern information
        self.input_pattern = {
            'type': 'constant',  # 'constant', 'step', 'ramp', 'oscillatory'
            'value': 0.0,  # Current value
            'params': {}  # Pattern-specific parameters
        }

        # Animation settings
        self.record_frames = config.create_animations
        self.frame_data = []
        self.record_interval = 10  # Record every 10th frame

    def initialize(self, cell_count=None):
        """
        Initialize the simulation with cells.

        Parameters:
            cell_count: Number of cells to create (default: from config)
        """
        # Use default from config if not specified
        if cell_count is None:
            cell_count = self.config.initial_cell_count

        # Populate grid with initial cells
        self.grid.populate_grid(cell_count)

        # Record initial state
        self._record_state()

    def set_constant_input(self, value):
        """
        Set a constant input pattern.

        Parameters:
            value: Constant shear stress value (Pa)
        """
        self.input_pattern = {
            'type': 'constant',
            'value': value,
            'params': {
                'value': value
            }
        }

    def set_step_input(self, initial_value, final_value, step_time):
        """
        Set a step input pattern.

        Parameters:
            initial_value: Initial shear stress value (Pa)
            final_value: Final shear stress value after step (Pa)
            step_time: Time at which the step occurs
        """
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
        """
        Set a ramp input pattern.

        Parameters:
            initial_value: Initial shear stress value (Pa)
            final_value: Final shear stress value after ramp (Pa)
            ramp_start_time: Time at which the ramp begins
            ramp_end_time: Time at which the ramp ends
        """
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
        """
        Set an oscillatory input pattern.

        Parameters:
            base_value: Base shear stress value (Pa)
            amplitude: Oscillation amplitude (Pa)
            frequency: Oscillation frequency (Hz)
            phase: Initial phase (radians)
        """
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

    def update_input_value(self):
        """
        Update the current input value based on the input pattern and current time.

        Returns:
            Current input value
        """
        pattern_type = self.input_pattern['type']
        params = self.input_pattern['params']

        if pattern_type == 'constant':
            self.input_pattern['value'] = params['value']

        elif pattern_type == 'step':
            if self.time < params['step_time']:
                self.input_pattern['value'] = params['initial_value']
            else:
                self.input_pattern['value'] = params['final_value']

        elif pattern_type == 'ramp':
            if self.time < params['ramp_start_time']:
                self.input_pattern['value'] = params['initial_value']
            elif self.time > params['ramp_end_time']:
                self.input_pattern['value'] = params['final_value']
            else:
                # Linear interpolation during ramp
                progress = (self.time - params['ramp_start_time']) / (
                            params['ramp_end_time'] - params['ramp_start_time'])
                self.input_pattern['value'] = params['initial_value'] + progress * (
                            params['final_value'] - params['initial_value'])

        elif pattern_type == 'oscillatory':
            omega = 2 * np.pi * params['frequency']
            self.input_pattern['value'] = params['base_value'] + params['amplitude'] * np.sin(
                omega * self.time + params['phase'])

        return self.input_pattern['value']

    def step(self):
        """
        Advance the simulation by one time step.

        Returns:
            Dictionary with updated state information
        """
        # Get current time step
        dt = self.config.time_step

        # Update input value
        current_input = self.update_input_value()

        # Apply shear stress to cells
        self._apply_shear_stress(current_input, dt)

        # Update models
        self._update_models(current_input, dt)

        # Update time and step count
        self.time += dt
        self.step_count += 1

        # Record state periodically
        if self.step_count % self.config.plot_interval == 0:
            self._record_state()

        # Record frame data if animation is enabled
        if self.record_frames and self.step_count % self.record_interval == 0:
            # Collect cell data for this frame
            cells_data = []

            for cell_id, cell in self.grid.cells.items():
                cells_data.append({
                    'cell_id': cell_id,
                    'position': cell.position,
                    'orientation': cell.orientation,
                    'aspect_ratio': cell.aspect_ratio,
                    'area': cell.area,
                    'is_senescent': cell.is_senescent,
                    'senescence_cause': cell.senescence_cause
                })

            # Store frame data
            self.frame_data.append({
                'time': self.time,
                'input_value': self.input_pattern['value'],
                'cell_count': len(self.grid.cells),
                'cells': cells_data
            })

        return {
            'time': self.time,
            'step_count': self.step_count,
            'input_value': current_input,
            'cell_count': len(self.grid.cells)
        }

    def run(self, duration=None):
        """
        Run the simulation for the specified duration.

        Parameters:
            duration: Duration to run in simulation time units (default: from config)

        Returns:
            Dictionary with simulation results
        """
        # Use default from config if not specified
        if duration is None:
            duration = self.config.simulation_duration

        # Calculate number of steps
        num_steps = int(duration / self.config.time_step)

        print(f"Running simulation for {duration} time units ({num_steps} steps)...")
        start_time = time.time()

        # Run steps
        for i in range(num_steps):
            step_info = self.step()

            # Print progress periodically
            if (i + 1) % 100 == 0 or i == num_steps - 1:
                progress = (i + 1) / num_steps * 100
                elapsed = time.time() - start_time
                estimated_total = elapsed / (i + 1) * num_steps
                remaining = estimated_total - elapsed

                print(f"Progress: {progress:.1f}% (Step {i + 1}/{num_steps}), "
                      f"Time: {elapsed:.1f}s, Remaining: {remaining:.1f}s, "
                      f"Cells: {step_info['cell_count']}")

        end_time = time.time()
        total_time = end_time - start_time

        print(f"Simulation completed in {total_time:.1f} seconds")

        # Create animations if enabled
        if self.record_frames and self.frame_data:
            print("Creating animations...")

            # Create plotter
            plotter = Plotter(self.config)

            # Generate animation filenames based on input pattern
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            pattern_type = self.input_pattern['type']
            pattern_value = self.input_pattern['value']

            # Create detailed cell animation
            cell_animation_path = os.path.join(
                self.config.plot_directory,
                f"cell_animation_{pattern_type}_{pattern_value}_{timestamp}.mp4"
            )
            create_detailed_cell_animation(
                plotter,
                self.frame_data,
                self,
                save_path=cell_animation_path
            )
            print(f"Cell animation created: {cell_animation_path}")

            # Create metrics animation
            metrics_animation_path = os.path.join(
                self.config.plot_directory,
                f"metrics_animation_{pattern_type}_{pattern_value}_{timestamp}.mp4"
            )
            create_metrics_animation(
                plotter,
                self,
                save_path=metrics_animation_path
            )
            print(f"Metrics animation created: {metrics_animation_path}")

        # Return results
        return {
            'duration': duration,
            'steps': num_steps,
            'final_time': self.time,
            'execution_time': total_time,
            'history': self.history,
            'animations_created': self.record_frames and len(self.frame_data) > 0
        }

    def _apply_shear_stress(self, shear_stress, duration):
        """
        Apply shear stress to all cells.

        Parameters:
            shear_stress: Shear stress value (Pa)
            duration: Duration of exposure
        """

        # Create a uniform shear stress field
        def shear_stress_function(x, y):
            return shear_stress

        # Apply to cells
        self.grid.apply_shear_stress_field(shear_stress_function, duration)

    def _update_models(self, current_input, dt):
        """
        Update all active models.

        Parameters:
            current_input: Current input value (shear stress in Pa)
            dt: Time step
        """
        # Update temporal dynamics if enabled
        if 'temporal' in self.models and self.config.enable_temporal_dynamics:
            model = self.models['temporal']
            model.update_cell_responses(self.grid.cells, current_input, dt)

        # Update spatial properties if enabled
        if 'spatial' in self.models and self.config.enable_spatial_properties:
            model = self.models['spatial']

            # Update each cell's spatial properties
            # Pass the shear stress value as pressure to the spatial model
            # Also pass all cells so the model can calculate population senescence level
            for cell in self.grid.cells.values():
                model.update_cell_properties(cell, current_input, dt)

        # Update population dynamics if enabled
        if 'population' in self.models and self.config.enable_population_dynamics:
            model = self.models['population']

            # Determine stem cell rate
            stem_cell_rate = 10 if self.config.enable_stem_cells else 0

            # Update population state (tau parameter is shear stress)
            model.update_from_cells(self.grid.cells, dt, current_input, stem_cell_rate)

            # Synchronize cells with population state
            actions = model.synchronize_cells(self.grid.cells)

            # Execute actions
            self._execute_population_actions(actions)

    # Also, in the _record_state method, add morphometry metrics:
    # Find the section "# Add spatial statistics if available"
    # And add these additional metrics:

    def _record_state(self):
        """
        Record the current simulation state for later analysis.
        """
        # Collect state information
        state = {
            'time': self.time,
            'step_count': self.step_count,
            'input_value': self.input_pattern['value'],
            'cells': len(self.grid.cells)
        }

        # Add population statistics if available
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

        # Add spatial statistics if available
        if 'spatial' in self.models:
            spatial_model = self.models['spatial']

            # Calculate alignment index
            alignment = spatial_model.calculate_alignment_index(self.grid.cells)

            # Calculate shape index
            shape_index = spatial_model.calculate_shape_index(self.grid.cells)

            state.update({
                'alignment_index': alignment,
                'shape_index': shape_index,
                'confluency': self.grid.calculate_confluency()
            })

            # Add morphometry metrics
            # Calculate average cell properties
            if self.grid.cells:
                areas = [cell.area for cell in self.grid.cells.values()]
                aspect_ratios = [cell.aspect_ratio for cell in self.grid.cells.values()]
                orientations = [np.degrees(cell.orientation) for cell in self.grid.cells.values()]
                eccentricities = [cell.eccentricity for cell in self.grid.cells.values() if
                                  hasattr(cell, 'eccentricity')]
                circularities = [cell.circularity for cell in self.grid.cells.values() if hasattr(cell, 'circularity')]

                state.update({
                    'avg_area': np.mean(areas) if areas else 0,
                    'avg_aspect_ratio': np.mean(aspect_ratios) if aspect_ratios else 0,
                    'avg_orientation': np.mean(orientations) if orientations else 0,
                    'avg_eccentricity': np.mean(eccentricities) if eccentricities else 0,
                    'avg_circularity': np.mean(circularities) if circularities else 0,
                    'population_senescence_level': spatial_model.get_population_senescence_level(self.grid.cells)
                })

        # Add to history
        self.history.append(state)

    def save_results(self, filename=None):
        """
        Save simulation results to a file.

        Parameters:
            filename: Name of the file to save to (default: auto-generated)

        Returns:
            Path to the saved file
        """
        # Use auto-generated filename if not specified
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"results_{timestamp}.npz"

        # Ensure directory exists
        os.makedirs(self.config.plot_directory, exist_ok=True)
        filepath = os.path.join(self.config.plot_directory, filename)

        # Convert history to NumPy arrays
        history_data = {}

        if self.history:
            for key in self.history[0].keys():
                values = [state.get(key) for state in self.history]
                history_data[key] = np.array(values)

        # Save data
        np.savez(
            filepath,
            history=history_data,
            config_params={
                'simulation_duration': self.config.simulation_duration,
                'time_step': self.config.time_step,
                'grid_size': self.config.grid_size,
                'initial_cell_count': self.config.initial_cell_count
            },
            input_pattern=self.input_pattern
        )

        print(f"Results saved to {filepath}")

        return filepath

    def get_cell_data(self):
        """
        Get data for all cells.

        Returns:
            List of cell data dictionaries
        """
        return [cell.get_state_dict() for cell in self.grid.cells.values()]