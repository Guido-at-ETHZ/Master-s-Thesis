"""
Configuration manager for event-driven reconfigurations.
Preserves cell types and generates optimal configurations.
"""
import numpy as np
import random
from typing import Dict, List, Tuple, Optional


class ConfigurationState:
    """Represents a complete configuration state."""
    
    def __init__(self):
        self.energy = float('inf')
        self.fitness = 0.0
        self.cell_data = {}
        self.packing_efficiency = 0.0
        self.biological_fitness = 0.0
        self.is_current_state = False
        
    def __repr__(self):
        return f"ConfigurationState(energy={self.energy:.4f}, fitness={self.fitness:.3f})"


class ConfigurationManager:
    """
    Manages configuration generation and selection for event-driven reconfiguration.
    Preserves cell types and uses current positions as starting points.
    """
    
    def __init__(self, grid, config):
        self.grid = grid
        self.config = config
        self.current_configuration = None
        self.target_configuration = None
        
    def generate_reconfiguration(self, event: 'ConfigurationEvent', 
                                num_configurations: int = 10,
                                optimization_iterations: int = 3) -> Dict:
        """
        Generate new configuration candidates in response to an event.
        Preserves cell types and counts from current state.
        
        Parameters:
            event: The triggering event
            num_configurations: Number of candidate configurations
            optimization_iterations: Optimization steps per candidate
            
        Returns:
            Dictionary with reconfiguration results
        """
        print(f"🔄 Generating reconfiguration for {event}")
        
        # Store current state
        original_cells = self.grid.cells.copy()
        original_seeds = self.grid.cell_seeds.copy()
        original_territories = self.grid.territory_map.copy()
        
        # Extract current cell information to preserve
        cell_inventory = self._extract_cell_inventory()
        
        print(f"   Preserving: {len(cell_inventory['healthy'])} healthy, "
              f"{len(cell_inventory['senescent_tel'])} tel-sen, "
              f"{len(cell_inventory['senescent_stress'])} stress-sen cells")
        
        configurations = []
        
        # Configuration 0: Current state (always include as baseline)
        current_config = self._create_current_state_configuration()
        configurations.append(current_config)
        
        # Generate additional candidate configurations
        for config_idx in range(1, num_configurations):
            print(f"   Testing configuration {config_idx + 1}/{num_configurations}...")
            
            # Clear grid for new configuration
            self._clear_grid_state()
            
            # Recreate cells with preserved types but new positions
            self._recreate_cells_with_preserved_types(cell_inventory, config_idx)
            
            # Update targets based on current conditions (pressure, holes, etc.)
            self._update_cell_targets_for_current_conditions()
            
            # Optimize this configuration
            for _ in range(optimization_iterations):
                self._optimize_configuration_biological()
            
            # Evaluate configuration
            config_data = self._evaluate_configuration(config_idx)
            configurations.append(config_data)
        
        # Select best configuration
        best_config = min(configurations, key=lambda x: x.energy)
        best_idx = configurations.index(best_config)
        
        print(f"✅ Best reconfiguration found: Config #{best_idx + 1}")
        print(f"   Energy: {best_config.energy:.4f}")
        print(f"   Energy improvement: {(current_config.energy - best_config.energy):.4f}")
        
        # Restore grid state temporarily
        self._restore_grid_state(original_cells, original_seeds, original_territories)
        
        return {
            'current_configuration': current_config,
            'target_configuration': best_config,
            'all_configurations': configurations,
            'energy_improvement': current_config.energy - best_config.energy,
            'selected_idx': best_idx,
            'event': event,
            'cell_inventory': cell_inventory
        }
    
    def _extract_cell_inventory(self) -> Dict:
        """Extract current cell inventory to preserve types."""
        inventory = {
            'healthy': [],
            'senescent_tel': [],
            'senescent_stress': []
        }
        
        for cell_id, cell in self.grid.cells.items():
            cell_info = {
                'cell_id': cell_id,
                'position': cell.position,
                'divisions': cell.divisions,
                'target_area': getattr(cell, 'target_area', cell.actual_area),
                'target_orientation': getattr(cell, 'target_orientation', cell.actual_orientation),
                'target_aspect_ratio': getattr(cell, 'target_aspect_ratio', cell.actual_aspect_ratio),
                'age': getattr(cell, 'age', 0.0),
                'senescent_growth_factor': getattr(cell, 'senescent_growth_factor', 1.0)
            }
            
            if not cell.is_senescent:
                inventory['healthy'].append(cell_info)
            elif cell.senescence_cause == 'telomere':
                inventory['senescent_tel'].append(cell_info)
            else:  # stress senescent
                inventory['senescent_stress'].append(cell_info)
        
        return inventory
    
    def _create_current_state_configuration(self) -> ConfigurationState:
        """Create configuration object for current state."""
        config = ConfigurationState()
        config.energy = self.grid.calculate_biological_energy()
        config.fitness = self.grid.get_biological_fitness()
        config.is_current_state = True
        
        # Store current cell data
        config.cell_data = {
            cell_id: {
                'position': cell.position,
                'divisions': cell.divisions,
                'is_senescent': cell.is_senescent,
                'senescence_cause': cell.senescence_cause,
                'target_area': getattr(cell, 'target_area', cell.actual_area),
                'target_orientation': getattr(cell, 'target_orientation', cell.actual_orientation),
                'target_aspect_ratio': getattr(cell, 'target_aspect_ratio', cell.actual_aspect_ratio)
            }
            for cell_id, cell in self.grid.cells.items()
        }
        
        grid_stats = self.grid.get_grid_statistics()
        config.packing_efficiency = grid_stats.get('packing_efficiency', 0)
        config.biological_fitness = grid_stats.get('biological_fitness', 0)
        
        return config
    
    def _recreate_cells_with_preserved_types(self, cell_inventory: Dict, config_idx: int):
        """Recreate cells with preserved types but new positions."""
        # Create new positions for all cells while preserving types
        total_cells = (len(cell_inventory['healthy']) + 
                      len(cell_inventory['senescent_tel']) + 
                      len(cell_inventory['senescent_stress']))
        
        # Generate new positions with some variation from original
        new_positions = self._generate_varied_positions(cell_inventory, config_idx)
        
        pos_idx = 0
        
        # Recreate healthy cells
        for cell_info in cell_inventory['healthy']:
            new_pos = new_positions[pos_idx]
            pos_idx += 1
            
            cell = self.grid.add_cell(
                position=new_pos,
                divisions=cell_info['divisions'],
                is_senescent=False,
                senescence_cause=None,
                target_area=cell_info['target_area']
            )
            
            # Set additional properties
            cell.age = cell_info.get('age', 0.0)
            cell.target_orientation = cell_info['target_orientation']
            cell.target_aspect_ratio = cell_info['target_aspect_ratio']
        
        # Recreate senescent cells
        for senescent_type in ['senescent_tel', 'senescent_stress']:
            cause = 'telomere' if senescent_type == 'senescent_tel' else 'stress'
            
            for cell_info in cell_inventory[senescent_type]:
                new_pos = new_positions[pos_idx]
                pos_idx += 1
                
                cell = self.grid.add_cell(
                    position=new_pos,
                    divisions=cell_info['divisions'],
                    is_senescent=True,
                    senescence_cause=cause,
                    target_area=cell_info['target_area']
                )
                
                # Set additional properties
                cell.age = cell_info.get('age', 0.0)
                cell.senescent_growth_factor = cell_info.get('senescent_growth_factor', 1.0)
                cell.target_orientation = cell_info['target_orientation']
                cell.target_aspect_ratio = cell_info['target_aspect_ratio']
    
    def _generate_varied_positions(self, cell_inventory: Dict, config_idx: int) -> List[Tuple[float, float]]:
        """Generate varied positions based on current positions."""
        all_positions = []
        
        # Collect all current positions
        for cell_type in ['healthy', 'senescent_tel', 'senescent_stress']:
            for cell_info in cell_inventory[cell_type]:
                all_positions.append(cell_info['position'])
        
        # Create variations of positions
        varied_positions = []
        
        for i, orig_pos in enumerate(all_positions):
            if config_idx == 0:
                # Configuration 0: keep original positions
                varied_positions.append(orig_pos)
            else:
                # Add controlled variation
                variation_radius = min(50, self.grid.width * 0.05)  # 5% of grid width or 50 pixels
                
                # Deterministic variation based on config_idx and position
                random.seed(config_idx * 1000 + i)
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0, variation_radius)
                
                new_x = orig_pos[0] + radius * np.cos(angle)
                new_y = orig_pos[1] + radius * np.sin(angle)
                
                # Constrain to grid bounds
                new_x = max(20, min(self.grid.width - 20, new_x))
                new_y = max(20, min(self.grid.height - 20, new_y))
                
                varied_positions.append((new_x, new_y))
        
        return varied_positions
    
    def _update_cell_targets_for_current_conditions(self):
        """Update cell targets based on current pressure and conditions."""
        if not hasattr(self, 'simulator') or not hasattr(self.simulator, 'models') or 'spatial' not in self.simulator.models:
            return
            
        # This will be called with access to simulator to get current pressure
        # For now, we'll update this when we integrate with the simulator
        pass

    def _optimize_configuration_biological(self):
        """Optimize the current configuration using biological energy."""
        # Just update tessellation without optimization
        self.grid._update_voronoi_tessellation()
    
    def _evaluate_configuration(self, config_idx: int) -> ConfigurationState:
        """Evaluate a configuration and return its metrics."""
        config = ConfigurationState()
        
        config.energy = self.grid.calculate_biological_energy()
        config.fitness = self.grid.get_biological_fitness()
        
        # Store cell data
        config.cell_data = {
            cell_id: {
                'position': cell.position,
                'divisions': cell.divisions,
                'is_senescent': cell.is_senescent,
                'senescence_cause': cell.senescence_cause,
                'target_area': getattr(cell, 'target_area', cell.actual_area),
                'target_orientation': getattr(cell, 'target_orientation', cell.actual_orientation),
                'target_aspect_ratio': getattr(cell, 'target_aspect_ratio', cell.actual_aspect_ratio)
            }
            for cell_id, cell in self.grid.cells.items()
        }
        
        grid_stats = self.grid.get_grid_statistics()
        config.packing_efficiency = grid_stats.get('packing_efficiency', 0)
        config.biological_fitness = grid_stats.get('biological_fitness', 0)
        
        return config
    
    def _clear_grid_state(self):
        """Clear the current grid state."""
        self.grid.cells.clear()
        self.grid.cell_seeds.clear()
        self.grid.territory_map.clear()
        self.grid.pixel_ownership.fill(-1)
        self.grid.next_cell_id = 0
    
    def _restore_grid_state(self, original_cells, original_seeds, original_territories):
        """Restore the grid to its original state."""
        self.grid.cells = original_cells
        self.grid.cell_seeds = original_seeds
        self.grid.territory_map = original_territories
        self.grid._update_voronoi_tessellation()