"""
Grid module for managing the spatial arrangement of cells.
"""
import numpy as np
from .cell import Cell


class Grid:
    """
    Class representing the spatial grid for cell placement and interactions.
    """

    def __init__(self, width, height, config):
        """
        Initialize the grid with dimensions and configuration.

        Parameters:
            width: Width of the grid in pixels
            height: Height of the grid in pixels
            config: SimulationConfig object
        """
        self.width = width
        self.height = height
        self.config = config

        # Dictionary to store cells by ID
        self.cells = {}

        # Grid representation for spatial queries
        self.grid_resolution = 20  # Size of each grid cell in pixels
        self.grid_width = int(np.ceil(width / self.grid_resolution))
        self.grid_height = int(np.ceil(height / self.grid_resolution))
        self.grid = [[[] for _ in range(self.grid_height)] for _ in range(self.grid_width)]

        # Tracking the next available cell ID
        self.next_cell_id = 0

        # Store gaps/holes between cells
        self.gaps = []  # List of (x, y, area) tuples representing intercellular gaps

    def add_cell(self, position=None, divisions=0, is_senescent=False, senescence_cause=None):
        """
        Add a new cell to the grid.

        Parameters:
            position: (x, y) coordinates, random if None
            divisions: Number of divisions the cell has undergone
            is_senescent: Boolean indicating if the cell is senescent
            senescence_cause: 'telomere' or 'stress' indicating the cause of senescence

        Returns:
            The newly created Cell object
        """
        # Generate a random position if none provided
        if position is None:
            position = (np.random.uniform(0, self.width),
                        np.random.uniform(0, self.height))

        # Create a new cell
        cell_id = self.next_cell_id
        self.next_cell_id += 1

        cell = Cell(cell_id, position, divisions, is_senescent, senescence_cause)

        # Add cell to the dictionary
        self.cells[cell_id] = cell

        # Add cell to the spatial grid
        self._add_cell_to_grid(cell)

        return cell

    def remove_cell(self, cell_id):
        """
        Remove a cell from the grid.

        Parameters:
            cell_id: ID of the cell to remove

        Returns:
            Boolean indicating if removal was successful
        """
        if cell_id not in self.cells:
            return False

        cell = self.cells[cell_id]

        # Remove cell from the spatial grid
        self._remove_cell_from_grid(cell)

        # Remove cell from the dictionary
        del self.cells[cell_id]

        return True

    def get_cell(self, cell_id):
        """
        Get a cell by its ID.

        Parameters:
            cell_id: ID of the cell to retrieve

        Returns:
            Cell object or None if not found
        """
        return self.cells.get(cell_id)

    def get_cells_in_region(self, x1, y1, x2, y2):
        """
        Get all cells whose centers lie within the specified rectangular region.

        Parameters:
            x1, y1: Coordinates of the top-left corner
            x2, y2: Coordinates of the bottom-right corner

        Returns:
            List of Cell objects in the region
        """
        result = []

        # Convert coordinates to grid indices
        grid_x1 = max(0, int(x1 / self.grid_resolution))
        grid_y1 = max(0, int(y1 / self.grid_resolution))
        grid_x2 = min(self.grid_width - 1, int(x2 / self.grid_resolution))
        grid_y2 = min(self.grid_height - 1, int(y2 / self.grid_resolution))

        # Check cells in each grid cell
        for gx in range(grid_x1, grid_x2 + 1):
            for gy in range(grid_y1, grid_y2 + 1):
                for cell_id in self.grid[gx][gy]:
                    cell = self.cells[cell_id]
                    x, y = cell.position
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        result.append(cell)

        return result

    def get_cells_in_radius(self, center_x, center_y, radius):
        """
        Get all cells whose centers lie within the specified circular region.

        Parameters:
            center_x, center_y: Coordinates of the circle center
            radius: Radius of the circle

        Returns:
            List of Cell objects in the region
        """
        # First get cells in the bounding box
        x1 = center_x - radius
        y1 = center_y - radius
        x2 = center_x + radius
        y2 = center_y + radius

        box_cells = self.get_cells_in_region(x1, y1, x2, y2)

        # Filter cells by actual distance
        result = []
        radius_squared = radius ** 2

        for cell in box_cells:
            x, y = cell.position
            distance_squared = (x - center_x) ** 2 + (y - center_y) ** 2
            if distance_squared <= radius_squared:
                result.append(cell)

        return result

    def get_neighboring_cells(self, cell_id, max_distance=None):
        """
        Get cells neighboring the specified cell.

        Parameters:
            cell_id: ID of the cell to find neighbors for
            max_distance: Maximum distance for neighboring cells, default is 2x the cell diameter

        Returns:
            List of neighboring Cell objects
        """
        if cell_id not in self.cells:
            return []

        cell = self.cells[cell_id]
        x, y = cell.position

        # Default max distance is based on typical cell size
        if max_distance is None:
            cell_radius = np.sqrt(cell.area / np.pi)
            max_distance = 2 * cell_radius * 2  # 2x the cell diameter

        neighbors = self.get_cells_in_radius(x, y, max_distance)

        # Remove the cell itself from neighbors
        neighbors = [n for n in neighbors if n.cell_id != cell_id]

        return neighbors

    def update_cell_position(self, cell_id, new_position):
        """
        Update a cell's position and update the spatial grid.

        Parameters:
            cell_id: ID of the cell to update
            new_position: New (x, y) coordinates

        Returns:
            Boolean indicating if update was successful
        """
        if cell_id not in self.cells:
            return False

        cell = self.cells[cell_id]

        # Remove cell from current grid position
        self._remove_cell_from_grid(cell)

        # Update cell position
        cell.update_position(new_position)

        # Add cell to new grid position
        self._add_cell_to_grid(cell)

        return True

    def populate_grid(self, count, division_distribution=None):
        """
        Populate the grid with initial cells.

        Parameters:
            count: Number of cells to create
            division_distribution: Optional function that returns division counts,
                                  default is exponential distribution

        Returns:
            List of created Cell objects
        """
        created_cells = []

        # Default division distribution
        if division_distribution is None:
            # Exponential distribution of division counts
            max_div = self.config.max_divisions

            def division_distribution():
                # Random divisions with exponential decay
                # More cells at lower division counts
                r = np.random.random()
                return int(max_div * (1 - np.sqrt(r)))

        # Create cells
        for _ in range(count):
            # Random position
            position = (np.random.uniform(0, self.width),
                        np.random.uniform(0, self.height))

            # Get division count from distribution
            divisions = division_distribution()

            # Determine if cell is senescent based on divisions
            is_senescent = False
            senescence_cause = None

            if divisions >= self.config.max_divisions:
                is_senescent = True
                senescence_cause = 'telomere'
            elif np.random.random() < 0.05:  # 5% chance of stress-induced senescence
                is_senescent = True
                senescence_cause = 'stress'

            # Create the cell
            cell = self.add_cell(position, divisions, is_senescent, senescence_cause)
            created_cells.append(cell)

        return created_cells

    def apply_shear_stress_field(self, shear_stress_function, duration):
        """
        Apply a shear stress field to all cells.

        Parameters:
            shear_stress_function: Function that takes (x, y) and returns shear stress value
            duration: Duration of exposure in simulation time units
        """
        for cell in self.cells.values():
            x, y = cell.position
            shear_stress = shear_stress_function(x, y)
            cell.apply_shear_stress(shear_stress, duration)

    def calculate_confluency(self):
        """
        Calculate the confluency (percentage of area covered by cells).

        Returns:
            Confluency as a value between 0 and 1
        """
        total_area = self.width * self.height
        covered_area = sum(cell.area for cell in self.cells.values())

        # Account for overlapping (approximate)
        coverage_factor = 0.9  # Adjustment for potential overlap
        covered_area *= coverage_factor

        return min(1.0, covered_area / total_area)

    def count_cells_by_type(self):
        """
        Count cells by type (normal, telomere-senescent, stress-senescent).

        Returns:
            Dictionary with counts for each cell type
        """
        normal_count = 0
        tel_sen_count = 0
        stress_sen_count = 0

        for cell in self.cells.values():
            if not cell.is_senescent:
                normal_count += 1
            elif cell.senescence_cause == 'telomere':
                tel_sen_count += 1
            elif cell.senescence_cause == 'stress':
                stress_sen_count += 1

        return {
            'normal': normal_count,
            'telomere_senescent': tel_sen_count,
            'stress_senescent': stress_sen_count,
            'total': len(self.cells)
        }

    def _add_cell_to_grid(self, cell):
        """
        Add a cell to the spatial grid data structure.

        Parameters:
            cell: Cell object to add
        """
        x, y = cell.position
        gx = min(self.grid_width - 1, max(0, int(x / self.grid_resolution)))
        gy = min(self.grid_height - 1, max(0, int(y / self.grid_resolution)))

        self.grid[gx][gy].append(cell.cell_id)

    def _remove_cell_from_grid(self, cell):
        """
        Remove a cell from the spatial grid data structure.

        Parameters:
            cell: Cell object to remove
        """
        x, y = cell.position
        gx = min(self.grid_width - 1, max(0, int(x / self.grid_resolution)))
        gy = min(self.grid_height - 1, max(0, int(y / self.grid_resolution)))

        if cell.cell_id in self.grid[gx][gy]:
            self.grid[gx][gy].remove(cell.cell_id)