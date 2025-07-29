"""
Event system for triggering configuration recomputations.
"""
from enum import Enum
from typing import Dict, Any, List
import numpy as np


class EventType(Enum):
    """Types of events that trigger reconfiguration."""
    PRESSURE_CHANGE = "pressure_change"
    HOLE_CREATED = "hole_created"
    HOLE_FILLED = "hole_filled"
    SENESCENCE_EVENT = "senescence_event"
    DIVISION_EVENT = "division_event"
    DEATH_EVENT = "death_event"
    THRESHOLD_REACHED = "threshold_reached"


class ConfigurationEvent:
    """Represents an event that requires reconfiguration."""
    
    def __init__(self, event_type: EventType, time: float, data: Dict[str, Any]):
        self.event_type = event_type
        self.time = time
        self.data = data
        
    def __repr__(self):
        return f"ConfigurationEvent({self.event_type.value} at t={self.time:.1f})"


class EventDetector:
    """Detects events that require reconfiguration."""
    
    def __init__(self, config):
        self.config = config
        self.last_pressure = 0.0
        self.last_hole_count = 0
        self.last_senescent_count = 0
        self.last_cell_count = 0
        
        # Thresholds for triggering events
        self.pressure_change_threshold = 0.1  # Pa
        self.senescence_threshold_change = 0.05  # 5% change
        self.cell_count_change_threshold = 5  # cells
        
    def detect_events(self, simulator) -> List[ConfigurationEvent]:
        """
        Detect events that require reconfiguration.
        
        Returns:
            List of events to process
        """
        events = []
        current_time = simulator.time
        
        # 1. Pressure change event
        current_pressure = simulator.input_pattern['value']
        if abs(current_pressure - self.last_pressure) >= self.pressure_change_threshold:
            events.append(ConfigurationEvent(
                EventType.PRESSURE_CHANGE,
                current_time,
                {
                    'old_pressure': self.last_pressure,
                    'new_pressure': current_pressure,
                    'pressure_change': current_pressure - self.last_pressure
                }
            ))
            self.last_pressure = current_pressure
            
        # 2. Hole events
        current_hole_count = len(simulator.grid.hole_manager.holes) if simulator.grid.hole_manager else 0
        if current_hole_count > self.last_hole_count:
            events.append(ConfigurationEvent(
                EventType.HOLE_CREATED,
                current_time,
                {
                    'hole_count': current_hole_count,
                    'new_holes': current_hole_count - self.last_hole_count
                }
            ))
        elif current_hole_count < self.last_hole_count:
            events.append(ConfigurationEvent(
                EventType.HOLE_FILLED,
                current_time,
                {
                    'hole_count': current_hole_count,
                    'filled_holes': self.last_hole_count - current_hole_count
                }
            ))
        self.last_hole_count = current_hole_count
        
        # 3. Senescence events
        cell_counts = simulator.grid.count_cells_by_type()
        current_senescent = cell_counts['telomere_senescent'] + cell_counts['stress_senescent']
        
        if current_senescent > self.last_senescent_count:
            events.append(ConfigurationEvent(
                EventType.SENESCENCE_EVENT,
                current_time,
                {
                    'new_senescent_cells': current_senescent - self.last_senescent_count,
                    'total_senescent': current_senescent,
                    'cell_counts': cell_counts
                }
            ))
        self.last_senescent_count = current_senescent
        
        # 4. Significant cell count changes (division/death)
        current_total = cell_counts['total']
        cell_change = abs(current_total - self.last_cell_count)
        
        if cell_change >= self.cell_count_change_threshold:
            event_type = EventType.DIVISION_EVENT if current_total > self.last_cell_count else EventType.DEATH_EVENT
            events.append(ConfigurationEvent(
                event_type,
                current_time,
                {
                    'cell_change': current_total - self.last_cell_count,
                    'total_cells': current_total,
                    'cell_counts': cell_counts
                }
            ))
        self.last_cell_count = current_total
        
        return events