from typing import List, Tuple, Optional, Dict, Set
import numpy as np

class CellType:
    EMPTY = 0
    OBSTACLE = 1
    SHELF = 2
    HEAVY_TRAFFIC = 3
    START = 4
    GOAL = 5

    NAMES = {
        EMPTY: "Floor",
        OBSTACLE: "Obstacle",
        SHELF: "Shelf Rack",
        HEAVY_TRAFFIC: "Heavy Traffic (Cost: 3)",
        START: "Robot Start",
        GOAL: "Pickup Goal"
    }

class WarehouseGrid:
    """
    Represents the warehouse environment grid graph.
    Nodes are (row, col) grid cells.
    Edges connect adjacent orthogonal cells (4-directional: Up, Down, Left, Right).
    """

    DEFAULT_COSTS = {
        CellType.EMPTY: 1.0,
        CellType.HEAVY_TRAFFIC: 3.0,
        CellType.START: 1.0,
        CellType.GOAL: 1.0,
        CellType.OBSTACLE: float('inf'),
        CellType.SHELF: float('inf')
    }

    def __init__(self, rows: int = 20, cols: int = 20):
        self.rows = max(1, rows)
        self.cols = max(1, cols)
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.costs = np.ones((self.rows, self.cols), dtype=float)
        self.start: Tuple[int, int] = (0, 0)
        self.goal: Tuple[int, int] = (self.rows - 1, self.cols - 1)

    def is_valid(self, row: int, col: int) -> bool:
        """Check if row and col are within grid boundaries."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_walkable(self, row: int, col: int) -> bool:
        """Check if cell is within boundaries and passable."""
        if not self.is_valid(row, col):
            return False
        cell_type = self.grid[row, col]
        return cell_type not in (CellType.OBSTACLE, CellType.SHELF)

    def get_cost(self, row: int, col: int) -> float:
        """Get traversal cost for entering cell (row, col)."""
        if not self.is_walkable(row, col):
            return float('inf')
        return float(self.costs[row, col])

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Get valid 4-directional adjacent walkable neighbors (Up, Down, Left, Right).
        Returns list of (r, c) tuples.
        """
        neighbors = []
        # Up, Down, Left, Right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if self.is_walkable(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def set_cell(self, row: int, col: int, cell_type: int, cost: Optional[float] = None):
        """Set cell type and cost at (row, col)."""
        if not self.is_valid(row, col):
            return
        self.grid[row, col] = cell_type
        if cost is not None:
            self.costs[row, col] = cost
        else:
            self.costs[row, col] = self.DEFAULT_COSTS.get(cell_type, 1.0)
        
        if cell_type == CellType.START:
            self.start = (row, col)
        elif cell_type == CellType.GOAL:
            self.goal = (row, col)

    def clear(self):
        """Reset grid to empty floor with default start and goal."""
        self.grid.fill(CellType.EMPTY)
        self.costs.fill(1.0)
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)
        self.set_cell(self.start[0], self.start[1], CellType.START)
        self.set_cell(self.goal[0], self.goal[1], CellType.GOAL)
