"""
Main simulator class for endothelial cell mechanotransduction with mosaic structure.
"""
import numpy as np
import time
import os
import random
from endothelial_simulation.core.cell import Cell
from endothelial_simulation.core.grid import Grid
from endothelial_simulation.models.temporal_dynamics import TemporalDynamicsModel
from endothelial_simulation.models.population_dynamics import PopulationDynamicsModel
from endothelial_simulation.models.spatial_properties import SpatialPropertiesModel
from endothelial_simulation.visualization import Plotter
from endothelial_simulation.visualization.animations import create_detailed_cell_animation, create_metrics_animation


class Simulator:
    """
    Main simulator class that integrates different model components and handles time evolution
    with mosaic cell structure.
    """

    def __init__(self, config):
        """
        Initialize the simulator with configuration parameters.

        Parameters:
            config: SimulationConfig object with parameter settings
        """
        self.config = config

        # Create grid for spatial representation with mosaic structure
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
            # CHANGE: Pass temporal model to spatial model for shared τ calculation
            temporal_model = self.models.get('temporal', None)
            self.models['spatial'] = SpatialPropertiesModel(config, temporal_model)

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

        # Animation settings
        self.record_frames = config.create_animations
        self.frame_data = []
        self.record_interval = 10

        # Mosaic-specific parameters
        self.tessellation_update_interval = 5  # Update tessellation every N steps
        self.position_optimization_interval = 20  # Optimize positions every N steps
        self.last_tessellation_update = 0
        self.last_position_optimization = 0

    def initialize(self, cell_count=None):
        """
        Initialize the simulation with cells using mosaic structure.

        Parameters:
            cell_count: Number of cells to create (default: from config)
        """
        if cell_count is None:
            cell_count = self.config.initial_cell_count

        # Calculate base area per cell
        total_area = self.grid.width * self.grid.height
        base_area_per_cell = total_area / cell_count

        # Create area distribution function
        def area_distribution():
            return np.random.uniform(base_area_per_cell * 0.7, base_area_per_cell * 1.3)

        # Populate grid with initial cells using improved distribution
        self.grid.populate_grid(cell_count, area_distribution=area_distribution)

        # Initial adaptation
        self.grid.adapt_cell_properties()

        # Record initial state
        self._record_state()

    def set_constant_input(self, value):
        """Set a constant input pattern."""
        self.input_pattern = {
            'type': 'constant',
            'value': value,
            'params': {
                'value': value
            }
        }

    def set_step_input(self, initial_value, final_value, step_time):
        """Set a step input pattern."""
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

    def update_input_value(self):
        """Update the current input value based on the input pattern and current time."""
        pattern_type = self.input_pattern['type']
        params = self.input_pattern['params']

        if pattern_type == 'constant':
            self.input_pattern['value'] = params['value']

        elif pattern_type == 'step':
            if self.time < params['step_time']:
                self.input_pattern['value'] = params['initial_value']
            else:
                self.input_pattern['value'] = params['final_value']

        elif pattern_type == 'multi_step':
            schedule = params['schedule']
            current_value = schedule[0][1]  # Default to first value

            # Find the appropriate value for current time
            for i, (step_time, step_value) in enumerate(schedule):
                if self.time >= step_time:
                    current_value = step_value
                else:
                    break

            # Update the value if it changed
            if self.input_pattern['value'] != current_value:
                old_value = self.input_pattern['value']
                self.input_pattern['value'] = current_value

                # Optional: Print step changes (you can remove this if you don't want the output)
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

    def set_multi_step_input(self, step_schedule):
        """
        Set a multi-step input pattern with multiple step changes.

        Parameters:
            step_schedule: List of tuples (time, value) defining when to change input
                          Example: [(0, 0.0), (60, 1.4), (180, 0.7), (300, 2.0)]
                          Times should be in minutes, values in Pa
        """
        # Validate input
        if not step_schedule or not isinstance(step_schedule, list):
            raise ValueError("step_schedule must be a non-empty list of (time, value) tuples")

        # Ensure all entries are tuples with 2 elements
        for i, step in enumerate(step_schedule):
            if not isinstance(step, (list, tuple)) or len(step) != 2:
                raise ValueError(f"Step {i} must be a (time, value) tuple, got: {step}")

        # Sort schedule by time to ensure proper order
        sorted_schedule = sorted(step_schedule, key=lambda x: x[0])

        # Validate that first time point is 0 or at start
        if sorted_schedule[0][0] > 0:
            print(f"Warning: First time point is {sorted_schedule[0][0]}, adding baseline at time 0")
            sorted_schedule.insert(0, (0, sorted_schedule[0][1]))

        self.input_pattern = {
            'type': 'multi_step',
            'value': sorted_schedule[0][1],  # Start with first value
            'params': {
                'schedule': sorted_schedule
            }
        }

        print(f"Multi-step input pattern set with {len(sorted_schedule)} steps:")
        for time_point, value in sorted_schedule:
            print(f"  {time_point:6.1f} min ({time_point / 60:5.2f}h): {value:5.2f} Pa")

    def set_protocol_input(self, protocol_name, **kwargs):
        """
        Set predefined experimental protocols.

        Parameters:
            protocol_name: Name of the protocol
            **kwargs: Additional parameters like scale_time, scale_stress

        Available protocols:
            - 'baseline': Constant low shear
            - 'acute_stress': Short stress pulse with recovery
            - 'chronic_stress': Sustained stress
            - 'stepwise_increase': Gradual stress increase
            - 'stress_recovery': Stress with recovery cycles
            - 'oscillating': On/off stress pattern
        """
        # Define protocol templates (times in minutes, stress in Pa)
        protocols = {
            'baseline': [
                (0, 0.0)  # Constant baseline
            ],

            'acute_stress': [
                (0, 0.0),  # Baseline (1 hour)
                (60, 1.4),  # Acute stress (2 hours)
                (180, 0.0)  # Recovery (remainder)
            ],

            'chronic_stress': [
                (0, 0.0),  # Baseline (1 hour)
                (60, 1.4),  # Start chronic stress
                (1020, 1.4)  # Maintain until end (16 hours of stress)
            ],

            'stepwise_increase': [
                (0, 0.0),  # Baseline
                (60, 0.5),  # Low stress (1h)
                (120, 1.0),  # Medium stress (1h)
                (180, 1.4),  # High stress (2h)
                (300, 0.0)  # Recovery
            ],

            'stress_recovery': [
                (0, 0.0),  # Baseline
                (60, 1.4),  # Stress episode 1 (2h)
                (180, 0.0),  # Recovery 1 (2h)
                (300, 1.4),  # Stress episode 2 (2h)
                (420, 0.0)  # Final recovery
            ],

            'oscillating': [
                (0, 0.0),  # Start
                (30, 1.0),  # Stress on (1h)
                (90, 0.0),  # Stress off (1h)
                (150, 1.0),  # Stress on (1h)
                (210, 0.0),  # Stress off (1h)
                (270, 1.0),  # Stress on (1h)
                (330, 0.0)  # Final off
            ]
        }

        if protocol_name not in protocols:
            available = ', '.join(protocols.keys())
            raise ValueError(f"Unknown protocol: {protocol_name}. Available: {available}")

        # Get base schedule
        schedule = protocols[protocol_name].copy()

        # Apply scaling if requested
        if 'scale_time' in kwargs:
            scale = kwargs['scale_time']
            schedule = [(t * scale, v) for t, v in schedule]
            print(f"Time scaling applied: {scale}x")

        if 'scale_stress' in kwargs:
            scale = kwargs['scale_stress']
            schedule = [(t, v * scale) for t, v in schedule]
            print(f"Stress scaling applied: {scale}x")

        # Custom modifications
        if 'max_stress' in kwargs:
            max_stress = kwargs['max_stress']
            schedule = [(t, min(v, max_stress) if v > 0 else v) for t, v in schedule]
            print(f"Maximum stress limited to: {max_stress} Pa")

        print(f"Using predefined protocol: '{protocol_name}'")
        self.set_multi_step_input(schedule)

    def get_current_step_info(self):
        """
        Get information about the current step in a multi-step pattern.

        Returns:
            Dictionary with current step information or None if not multi-step
        """
        if self.input_pattern['type'] != 'multi_step':
            return None

        schedule = self.input_pattern['params']['schedule']
        current_time = self.time

        # Find current step
        current_step_idx = 0
        for i, (step_time, step_value) in enumerate(schedule):
            if current_time >= step_time:
                current_step_idx = i
            else:
                break

        # Calculate time in current step and time to next step
        current_step_time, current_step_value = schedule[current_step_idx]
        time_in_step = current_time - current_step_time

        if current_step_idx < len(schedule) - 1:
            next_step_time, next_step_value = schedule[current_step_idx + 1]
            time_to_next = next_step_time - current_time
        else:
            next_step_time, next_step_value = None, None
            time_to_next = None

        return {
            'step_number': current_step_idx + 1,
            'total_steps': len(schedule),
            'current_value': current_step_value,
            'step_start_time': current_step_time,
            'time_in_step': time_in_step,
            'next_step_time': next_step_time,
            'next_step_value': next_step_value,
            'time_to_next_step': time_to_next
        }

    def step(self):
        """
        Advance the simulation by one time step.

        Returns:
            Dictionary with updated state information
        """
        dt = self.config.time_step
        current_input = self.update_input_value()

        # Apply shear stress to cells
        self._apply_shear_stress(current_input, dt)

        # Update models
        self._update_models(current_input, dt)

        # Update mosaic structure periodically
        self._update_mosaic_structure()

        # Update time and step count
        self.time += dt
        self.step_count += 1

        # Record state periodically
        if self.step_count % self.config.plot_interval == 0:
            self._record_state()

        # Record frame data if animation is enabled
        if self.record_frames and self.step_count % self.record_interval == 0:
            self._record_frame_data()

        return {
            'time': self.time,
            'step_count': self.step_count,
            'input_value': current_input,
            'cell_count': len(self.grid.cells)
        }

    def _update_mosaic_structure(self):
        """Update the mosaic structure periodically."""
        # Update tessellation periodically
        if self.step_count - self.last_tessellation_update >= self.tessellation_update_interval:
            self.grid.adapt_cell_properties()
            self.grid.add_controlled_variability()
            self.grid._update_voronoi_tessellation()
            self.last_tessellation_update = self.step_count

        # Optimize positions less frequently
        if self.step_count - self.last_position_optimization >= self.position_optimization_interval:
            self.grid.optimize_cell_positions(iterations=2)
            self.last_position_optimization = self.step_count

    def _record_frame_data(self):
        """Record frame data for animation."""
        cells_data = []

        for cell_id, cell in self.grid.cells.items():
            cells_data.append({
                'cell_id': cell_id,
                'position': cell.position,
                'centroid': cell.centroid,
                'territory_pixels': cell.territory_pixels,
                'boundary_points': cell.boundary_points,
                'actual_orientation': cell.actual_orientation,
                'target_orientation': cell.target_orientation,
                'actual_aspect_ratio': cell.actual_aspect_ratio,
                'actual_area': cell.actual_area,
                'target_area': cell.target_area,
                'is_senescent': cell.is_senescent,
                'senescence_cause': cell.senescence_cause,
                'compression_ratio': cell.compression_ratio
            })

        self.frame_data.append({
            'time': self.time,
            'input_value': self.input_pattern['value'],
            'cell_count': len(self.grid.cells),
            'cells': cells_data,
            'grid_stats': self.grid.get_grid_statistics()
        })

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

        print(f"Running mosaic simulation for {duration} time units ({num_steps} steps)...")
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

                # Get current grid statistics
                grid_stats = self.grid.get_grid_statistics()
                packing_eff = grid_stats.get('packing_efficiency', 0)
                global_pressure = grid_stats.get('global_pressure', 1.0)

                print(f"Progress: {progress:.1f}% (Step {i + 1}/{num_steps}), "
                      f"Time: {elapsed:.1f}s, Remaining: {remaining:.1f}s, "
                      f"Cells: {step_info['cell_count']}, "
                      f"Packing: {packing_eff:.2f}, Pressure: {global_pressure:.2f}")

        end_time = time.time()
        total_time = end_time - start_time

        print(f"Mosaic simulation completed in {total_time:.1f} seconds")

        # Create animations if enabled
        if self.record_frames and self.frame_data:
            print("Creating mosaic animations...")
            self._create_animations()

        # Return results
        return {
            'duration': duration,
            'steps': num_steps,
            'final_time': self.time,
            'execution_time': total_time,
            'history': self.history,
            'animations_created': self.record_frames and len(self.frame_data) > 0,
            'final_grid_stats': self.grid.get_grid_statistics()
        }

    def _create_animations(self):
        """Create animations for the mosaic simulation."""
        plotter = Plotter(self.config)

        # Generate animation filenames based on input pattern
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        pattern_type = self.input_pattern['type']
        pattern_value = self.input_pattern['value']

        # Create detailed cell animation (mosaic version)
        cell_animation_path = os.path.join(
            self.config.plot_directory,
            f"mosaic_animation_{pattern_type}_{pattern_value}_{timestamp}.mp4"
        )
        self._create_mosaic_animation(cell_animation_path)
        print(f"Mosaic animation created: {cell_animation_path}")

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

    def _create_mosaic_animation(self, save_path):
        """Create a mosaic-specific animation showing cell territories."""
        try:
            import matplotlib.animation as animation
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon
            from scipy.spatial import ConvexHull

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_xlim(0, self.grid.width)
            ax.set_ylim(0, self.grid.height)
            ax.set_aspect('equal')

            # Color mapping
            color_map = {'healthy': 'green', 'telomere': 'red', 'stress': 'blue'}

            def update_frame(frame_idx):
                ax.clear()
                ax.set_xlim(0, self.grid.width)
                ax.set_ylim(0, self.grid.height)
                ax.set_aspect('equal')

                # Get frame data
                frame = self.frame_data[min(frame_idx, len(self.frame_data) - 1)]

                # Plot each cell's territory
                for cell_data in frame['cells']:
                    # Determine cell color
                    if not cell_data['is_senescent']:
                        color = color_map['healthy']
                        alpha = 0.6
                    elif cell_data['senescence_cause'] == 'telomere':
                        color = color_map['telomere']
                        alpha = 0.8
                    else:
                        color = color_map['stress']
                        alpha = 0.8

                    # Plot territory
                    territory_pixels = cell_data['territory_pixels']
                    if len(territory_pixels) > 10:  # Only for larger territories
                        try:
                            # Create convex hull for visualization
                            points = np.array(territory_pixels)
                            hull = ConvexHull(points)
                            hull_points = points[hull.vertices]

                            polygon = Polygon(hull_points, facecolor=color, alpha=alpha,
                                            edgecolor='black', linewidth=0.5)
                            ax.add_patch(polygon)
                        except:
                            # Fallback: scatter plot
                            points = np.array(territory_pixels)
                            ax.scatter(points[:, 0], points[:, 1], c=color, alpha=alpha, s=1, marker='s')

                    # Plot orientation vector
                    if cell_data['centroid'] is not None:
                        cx, cy = cell_data['centroid']
                        orientation = cell_data['actual_orientation']
                        vector_length = np.sqrt(cell_data['actual_area']) * 0.2
                        dx = vector_length * np.cos(orientation)
                        dy = vector_length * np.sin(orientation)

                        ax.arrow(cx, cy, dx, dy, head_width=vector_length*0.1,
                                head_length=vector_length*0.1, fc='white', ec='black',
                                alpha=0.8, width=vector_length*0.02)

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='green', edgecolor='black', alpha=0.6, label='Healthy'),
                    Patch(facecolor='red', edgecolor='black', alpha=0.8, label='Telomere-Senescent'),
                    Patch(facecolor='blue', edgecolor='black', alpha=0.8, label='Stress-Senescent')
                ]
                ax.legend(handles=legend_elements, loc='upper right')

                # Add info text
                info_text = (
                    f"Time: {frame['time']:.1f} {self.config.time_unit}\n"
                    f"Shear Stress: {frame['input_value']:.2f} Pa\n"
                    f"Cells: {frame['cell_count']}\n"
                    f"Packing: {frame['grid_stats'].get('packing_efficiency', 0):.2f}\n"
                    f"Pressure: {frame['grid_stats'].get('global_pressure', 1.0):.2f}"
                )
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

                # Add flow direction indicator
                arrow_length = self.grid.width * 0.08
                arrow_x = self.grid.width * 0.5
                arrow_y = self.grid.height * 0.05

                ax.arrow(arrow_x - arrow_length / 2, arrow_y, arrow_length, 0,
                         head_width=arrow_length * 0.3, head_length=arrow_length * 0.2,
                         fc='black', ec='black', width=arrow_length * 0.08)

                ax.text(arrow_x, arrow_y - arrow_length * 0.5, "Flow Direction",
                        ha='center', va='top', fontsize=12, weight='bold')

                ax.set_title('Endothelial Cell Mosaic Evolution', fontsize=16)
                ax.set_xlabel('X Position (pixels)', fontsize=12)
                ax.set_ylabel('Y Position (pixels)', fontsize=12)

            # Create animation
            ani = animation.FuncAnimation(
                fig, update_frame, frames=len(self.frame_data),
                interval=100, repeat=True
            )

            # Save animation
            if 'ffmpeg' in animation.writers.list():
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=10, metadata=dict(artist='Endothelial Mosaic Simulation'), bitrate=1800)
                ani.save(save_path, writer=writer, dpi=100)
            else:
                print("Warning: ffmpeg not available for animation creation")

            plt.close(fig)

        except Exception as e:
            print(f"Error creating mosaic animation: {e}")

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
        REPLACE this method in your existing simulator.py
        """
        # Update temporal dynamics if enabled (UNCHANGED from your original)
        if 'temporal' in self.models and self.config.enable_temporal_dynamics:
            model = self.models['temporal']
            model.update_cell_responses(self.grid.cells, current_input, dt)

        # Update spatial properties if enabled (CHANGED - now uses temporal dynamics)
        if 'spatial' in self.models and self.config.enable_spatial_properties:
            model = self.models['spatial']
            # NEW: Use temporal dynamics for all cells
            for cell in self.grid.cells.values():
                # This now uses temporal dynamics instead of instant calculation
                dynamics_result = model.update_cell_properties(cell, current_input, dt, self.grid.cells)

                # Optional: Store dynamics info for monitoring/debugging
                if hasattr(cell, 'last_dynamics_info'):
                    cell.last_dynamics_info = dynamics_result.get('dynamics_info', {})

        # Update population dynamics if enabled (UNCHANGED from your original)
        if 'population' in self.models and self.config.enable_population_dynamics:
            model = self.models['population']
            stem_cell_rate = 10 if self.config.enable_stem_cells else 0
            model.update_from_cells(self.grid.cells, dt, current_input, stem_cell_rate)
            actions = model.synchronize_cells(self.grid.cells)
            self._execute_population_actions(actions)

    def _execute_population_actions(self, actions):
        """
        Execute population actions returned by the population dynamics model.

        Parameters:
            actions: Dictionary with 'births', 'deaths', and 'senescence' actions
        """
        # Process birth actions
        for birth_action in actions.get('births', []):
            if birth_action['type'] == 'healthy':
                # Create a new healthy cell
                divisions = birth_action.get('divisions', 0)
                # Calculate appropriate target area
                total_area = self.grid.width * self.grid.height
                current_cell_count = len(self.grid.cells)
                target_area = total_area / max(1, current_cell_count + 1)

                self.grid.add_cell(
                    position=None,  # Random position
                    divisions=divisions,
                    is_senescent=False,
                    senescence_cause=None,
                    target_area=target_area
                )
            elif birth_action['type'] == 'senescent':
                # Create a new senescent cell
                cause = birth_action.get('cause', 'stress')
                divisions = self.config.max_divisions if cause == 'telomere' else 0
                # Senescent cells typically larger
                total_area = self.grid.width * self.grid.height
                current_cell_count = len(self.grid.cells)
                target_area = total_area / max(1, current_cell_count + 1) * 1.5

                self.grid.add_cell(
                    position=None,  # Random position
                    divisions=divisions,
                    is_senescent=True,
                    senescence_cause=cause,
                    target_area=target_area
                )

        # Process death actions
        for death_action in actions.get('deaths', []):
            death_type = death_action['type']
            count = death_action['count']

            # Find cells to remove based on type
            candidates = []

            if death_type == 'healthy':
                target_divisions = death_action.get('divisions', 0)
                candidates = [
                    cell_id for cell_id, cell in self.grid.cells.items()
                    if not cell.is_senescent and cell.divisions == target_divisions
                ]
            elif death_type == 'senescent':
                target_cause = death_action.get('cause', 'stress')
                candidates = [
                    cell_id for cell_id, cell in self.grid.cells.items()
                    if cell.is_senescent and cell.senescence_cause == target_cause
                ]

            # Remove random cells from candidates (up to count)
            cells_to_remove = random.sample(candidates, min(count, len(candidates)))

            for cell_id in cells_to_remove:
                self.grid.remove_cell(cell_id)

        # Process senescence actions (cells changing from healthy to senescent)
        for senescence_action in actions.get('senescence', []):
            # Find healthy cells to make senescent
            healthy_cells = [
                cell_id for cell_id, cell in self.grid.cells.items()
                if not cell.is_senescent
            ]

            if healthy_cells:
                # Select a random healthy cell
                cell_id = random.choice(healthy_cells)
                cell = self.grid.cells[cell_id]

                # Make it senescent
                cause = senescence_action.get('cause', 'stress')
                cell.induce_senescence(cause)

    def _record_state(self):
        """
        MODIFY this method in your existing simulator.py to include temporal dynamics info
        """
        # Keep all your existing state recording code...

        # Collect state information (your existing code)
        state = {
            'time': self.time,
            'step_count': self.step_count,
            'input_value': self.input_pattern['value'],
            'cells': len(self.grid.cells)
        }

        # Add grid statistics (your existing code)
        grid_stats = self.grid.get_grid_statistics()
        state.update(grid_stats)

        # ADD: Temporal dynamics monitoring
        if 'temporal' in self.models and self.config.enable_temporal_dynamics:
            temporal_model = self.models['temporal']
            current_pressure = self.input_pattern['value']

            # Get time constants for different properties
            tau_biochem, A_max = temporal_model.get_scaled_tau_and_amax(current_pressure, 'biochemical')
            tau_area, _ = temporal_model.get_scaled_tau_and_amax(current_pressure, 'area')
            tau_orientation, _ = temporal_model.get_scaled_tau_and_amax(current_pressure, 'orientation')
            tau_aspect_ratio, _ = temporal_model.get_scaled_tau_and_amax(current_pressure, 'aspect_ratio')

            state.update({
                'current_A_max': A_max,
                'tau_biochemical': tau_biochem,
                'tau_area': tau_area,
                'tau_orientation': tau_orientation,
                'tau_aspect_ratio': tau_aspect_ratio,
                'temporal_dynamics_active': True
            })

        # Keep all your existing population statistics code...
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

        # Keep all your existing spatial statistics code...
        if 'spatial' in self.models:
            spatial_model = self.models['spatial']
            collective_props = spatial_model.calculate_collective_properties(self.grid.cells,
                                                                             self.input_pattern['value'])
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

        # Add to history (your existing code)
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
            filename = f"mosaic_results_{timestamp}.npz"

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
            input_pattern=self.input_pattern,
            final_grid_stats=self.grid.get_grid_statistics()
        )

        print(f"Mosaic simulation results saved to {filepath}")

        return filepath

    def get_cell_data(self):
        """
        Get data for all cells including mosaic-specific properties.

        Returns:
            List of cell data dictionaries
        """
        return [cell.get_state_dict() for cell in self.grid.cells.values()]

    def get_mosaic_summary(self):
        """
        Get a summary of the mosaic structure state.

        Returns:
            Dictionary with mosaic summary information
        """
        grid_stats = self.grid.get_grid_statistics()

        # Calculate additional mosaic metrics
        total_target_area = sum(cell.target_area for cell in self.grid.cells.values())
        total_actual_area = sum(cell.actual_area for cell in self.grid.cells.values())

        # Orientation statistics
        orientations = [cell.actual_orientation for cell in self.grid.cells.values()]
        target_orientations = [cell.target_orientation for cell in self.grid.cells.values()]

        return {
            'total_cells': len(self.grid.cells),
            'total_target_area': total_target_area,
            'total_actual_area': total_actual_area,
            'area_utilization': total_actual_area / (self.grid.width * self.grid.height),
            'mean_orientation': np.mean(orientations) if orientations else 0,
            'std_orientation': np.std(orientations) if orientations else 0,
            'mean_target_orientation': np.mean(target_orientations) if target_orientations else 0,
            'orientation_adaptation': 1.0 - np.mean([abs(a - t) for a, t in zip(orientations, target_orientations)]) / np.pi if orientations else 0,
            'grid_statistics': grid_stats
        }