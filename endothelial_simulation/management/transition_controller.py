"""
Transition controller for spring-based transitions between configurations.
Uses temporal dynamics and compression/expansion mechanics.
"""
import numpy as np
from typing import Dict, Optional, List
import time


class TransitionState:
    """Represents the current state of a transition."""
    
    def __init__(self):
        self.is_active = False
        self.start_time = 0.0
        self.start_configuration = None
        self.target_configuration = None
        self.progress = 0.0
        self.completion_threshold = 0.95
        self.trajectory_checkpoints = []
        self.last_checkpoint_time = 0.0
        self.checkpoint_interval = 20.0  # minutes
        
    def is_complete(self) -> bool:
        """Check if transition is complete."""
        return self.progress >= self.completion_threshold
        
    def should_checkpoint(self, current_time: float) -> bool:
        """Check if we should create a trajectory checkpoint."""
        return (current_time - self.last_checkpoint_time) >= self.checkpoint_interval


class CompressionSpring:
    """
    Manages compression and expansion of cells during transitions.
    Acts like a spring system finding equilibrium.
    """
    
    def __init__(self, cell_id: str, start_properties: Dict, target_properties: Dict, tau: float):
        self.cell_id = cell_id
        self.start_area = start_properties['area']
        self.target_area = target_properties['area']
        self.start_aspect_ratio = start_properties['aspect_ratio']
        self.target_aspect_ratio = target_properties['aspect_ratio']
        self.start_orientation = start_properties['orientation']
        self.target_orientation = target_properties['orientation']
        
        self.tau = tau  # Time constant from temporal dynamics
        self.max_compression = 0.7  # Can compress to 70% of target
        self.current_compression_factor = 1.0
        
    def calculate_current_properties(self, t: float, space_pressure: float = 1.0) -> Dict:
        """
        Calculate current cell properties during transition.
        
        Parameters:
            t: Time since transition start (minutes)
            space_pressure: Global space pressure (>1 means crowded)
            
        Returns:
            Dictionary with current target properties
        """
        # Standard exponential approach to target (temporal dynamics)
        progress = 1.0 - np.exp(-t / self.tau)
        
        # Apply compression if space is contested
        if space_pressure > 1.0:
            compression_factor = min(self.max_compression, 1.0 / space_pressure)
            self.current_compression_factor = compression_factor
        else:
            # Gradually release compression as space becomes available
            release_rate = 0.1  # 10% release per time step
            self.current_compression_factor = min(1.0, 
                self.current_compression_factor + release_rate * (1.0 - self.current_compression_factor))
        
        # Calculate intermediate properties
        current_area = self._interpolate_property(
            self.start_area, self.target_area, progress
        ) * self.current_compression_factor
        
        current_aspect_ratio = self._interpolate_property(
            self.start_aspect_ratio, self.target_aspect_ratio, progress
        )
        
        current_orientation = self._interpolate_angle(
            self.start_orientation, self.target_orientation, progress
        )
        
        return {
            'area': current_area,
            'aspect_ratio': current_aspect_ratio,
            'orientation': current_orientation,
            'compression_factor': self.current_compression_factor,
            'progress': progress
        }
    
    def _interpolate_property(self, start: float, target: float, progress: float) -> float:
        """Interpolate between start and target values."""
        return start + (target - start) * progress
    
    def _interpolate_angle(self, start: float, target: float, progress: float) -> float:
        """Interpolate between angles, handling wrapping."""
        # Handle angle wrapping
        diff = target - start
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        
        return start + diff * progress


class TransitionController:
    """
    Controls transitions between configurations using spring-like mechanics.
    """
    
    def __init__(self, grid, temporal_model=None):
        self.grid = grid
        self.temporal_model = temporal_model
        self.current_transition = TransitionState()
        self.springs = {}  # cell_id -> CompressionSpring
        
    def start_transition(self, reconfiguration_result: Dict, current_time: float):
        """
        Start a transition to a new configuration.
        
        Parameters:
            reconfiguration_result: Result from ConfigurationManager.generate_reconfiguration()
            current_time: Current simulation time
        """
        print(f"🔄 Starting transition at t={current_time:.1f} min")
        
        # Set up transition state
        self.current_transition.is_active = True
        self.current_transition.start_time = current_time
        self.current_transition.start_configuration = reconfiguration_result['current_configuration']
        self.current_transition.target_configuration = reconfiguration_result['target_configuration']
        self.current_transition.progress = 0.0
        self.current_transition.trajectory_checkpoints = []
        self.current_transition.last_checkpoint_time = current_time
        
        # Create compression springs for each cell
        self._create_compression_springs(reconfiguration_result)
        
        # Apply target configuration to grid (this sets the target state)
        self._apply_target_configuration()
        
        print(f"   Energy improvement expected: {reconfiguration_result['energy_improvement']:.4f}")
        print(f"   Transition will use temporal dynamics with compression")
    
    def update_transition(self, current_time: float, dt: float) -> bool:
        """
        Update the ongoing transition.
        
        Parameters:
            current_time: Current simulation time
            dt: Time step
            
        Returns:
            True if transition is complete, False otherwise
        """
        if not self.current_transition.is_active:
            return True
        
        # Calculate time since transition start
        t_transition = current_time - self.current_transition.start_time
        
        # Calculate global space pressure
        space_pressure = self._calculate_space_pressure()
        
        # Update all cell springs
        total_progress = 0.0
        for cell_id, spring in self.springs.items():
            if cell_id in self.grid.cells:
                cell = self.grid.cells[cell_id]
                
                # Get current properties from spring
                current_props = spring.calculate_current_properties(t_transition, space_pressure)
                
                # Update cell targets (this is what cells will adapt toward)
                cell.target_area = current_props['area']
                cell.target_aspect_ratio = current_props['aspect_ratio']
                cell.target_orientation = current_props['orientation']
                
                total_progress += current_props['progress']
        
        # Calculate overall progress
        self.current_transition.progress = total_progress / max(1, len(self.springs))
        
        # Create trajectory checkpoint if needed
        if self.current_transition.should_checkpoint(current_time):
            self._create_trajectory_checkpoint(current_time)
        
        # Check for completion
        if self.current_transition.is_complete():
            self._complete_transition(current_time)
            return True
        
        # Update tessellation to reflect new targets
        self.grid._update_voronoi_tessellation()
        
        return False
    
    def _create_compression_springs(self, reconfiguration_result: Dict):
        """Create compression springs for the transition."""
        current_config = reconfiguration_result['current_configuration']
        target_config = reconfiguration_result['target_configuration']
        
        self.springs.clear()
        
        # Get time constant from temporal model
        if self.temporal_model:
            # Use current pressure for time constant calculation
            current_pressure = 1.0  # Will be set by simulator
            tau, _ = self.temporal_model.get_scaled_tau_and_amax(current_pressure, 'biochemical')
        else:
            tau = 30.0  # Default 30 minutes
        
        # Create springs for each cell
        for cell_id in current_config.cell_data:
            if cell_id in target_config.cell_data:
                start_props = {
                    'area': current_config.cell_data[cell_id]['target_area'],
                    'aspect_ratio': current_config.cell_data[cell_id]['target_aspect_ratio'],
                    'orientation': current_config.cell_data[cell_id]['target_orientation']
                }
                
                target_props = {
                    'area': target_config.cell_data[cell_id]['target_area'],
                    'aspect_ratio': target_config.cell_data[cell_id]['target_aspect_ratio'],
                    'orientation': target_config.cell_data[cell_id]['target_orientation']
                }
                
                spring = CompressionSpring(cell_id, start_props, target_props, tau)
                self.springs[cell_id] = spring
    
    def _apply_target_configuration(self):
        """Apply the target configuration to the grid."""
        target_config = self.current_transition.target_configuration
        
        # Clear and recreate cells according to target configuration
        self.grid.cells.clear()
        self.grid.cell_seeds.clear()
        self.grid.territory_map.clear()
        self.grid.pixel_ownership.fill(-1)
        self.grid.next_cell_id = 0
        
        # Recreate cells from target configuration
        for cell_id, cell_data in target_config.cell_data.items():
            cell = self.grid.add_cell(
                position=cell_data['position'],
                divisions=cell_data['divisions'],
                is_senescent=cell_data['is_senescent'],
                senescence_cause=cell_data['senescence_cause'],
                target_area=cell_data['target_area']
            )
            
            # Set target properties (these will be modified by springs during transition)
            cell.target_orientation = cell_data['target_orientation']
            cell.target_aspect_ratio = cell_data['target_aspect_ratio']
        
        # Update tessellation
        self.grid._update_voronoi_tessellation()
    
    def _calculate_space_pressure(self) -> float:
        """Calculate global space pressure for compression calculation."""
        if not self.grid.cells:
            return 1.0
        
        total_target_area = sum(cell.target_area for cell in self.grid.cells.values())
        total_available_area = self.grid.comp_width * self.grid.comp_height
        
        # Subtract hole areas if holes are present
        if self.grid.holes_enabled and self.grid.hole_manager:
            hole_stats = self.grid.hole_manager.get_hole_statistics()
            total_available_area -= hole_stats['total_hole_area']
        
        space_pressure = total_target_area / max(1, total_available_area)
        return space_pressure
    
    def _create_trajectory_checkpoint(self, current_time: float):
        """Create a trajectory checkpoint for monitoring."""
        checkpoint = {
            'time': current_time,
            'progress': self.current_transition.progress,
            'space_pressure': self._calculate_space_pressure(),
            'energy': self.grid.calculate_biological_energy(),
            'compression_factors': {
                cell_id: spring.current_compression_factor 
                for cell_id, spring in self.springs.items()
            }
        }
        
        self.current_transition.trajectory_checkpoints.append(checkpoint)
        self.current_transition.last_checkpoint_time = current_time
        
        print(f"📍 Trajectory checkpoint: t={current_time:.1f}, "
              f"progress={self.current_transition.progress:.1%}, "
              f"pressure={checkpoint['space_pressure']:.2f}")
    
    def _complete_transition(self, current_time: float):
        """Complete the transition."""
        print(f"✅ Transition completed at t={current_time:.1f} min")
        print(f"   Duration: {current_time - self.current_transition.start_time:.1f} minutes")
        print(f"   Final energy: {self.grid.calculate_biological_energy():.4f}")
        
        # Ensure all cells reach their final targets
        for cell_id, spring in self.springs.items():
            if cell_id in self.grid.cells:
                cell = self.grid.cells[cell_id]
                # Set final target values (no compression)
                target_config = self.current_transition.target_configuration
                if cell_id in target_config.cell_data:
                    cell_data = target_config.cell_data[cell_id]
                    cell.target_area = cell_data['target_area']
                    cell.target_aspect_ratio = cell_data['target_aspect_ratio']
                    cell.target_orientation = cell_data['target_orientation']
        
        # Clear transition state
        self.current_transition.is_active = False
        self.springs.clear()
        
        # Final tessellation update
        self.grid._update_voronoi_tessellation()
    
    def is_transitioning(self) -> bool:
        """Check if a transition is currently active."""
        return self.current_transition.is_active
    
    def get_transition_progress(self) -> float:
        """Get current transition progress (0.0 to 1.0)."""
        return self.current_transition.progress if self.current_transition.is_active else 1.0
    
    def get_transition_info(self) -> Dict:
        """Get detailed information about current transition."""
        if not self.current_transition.is_active:
            return {'active': False}
        
        return {
            'active': True,
            'progress': self.current_transition.progress,
            'start_time': self.current_transition.start_time,
            'checkpoints': len(self.current_transition.trajectory_checkpoints),
            'space_pressure': self._calculate_space_pressure(),
            'num_springs': len(self.springs)
        }