from warehouse.grid import WarehouseGrid, CellType

class WarehousePresets:
    """Preconfigured warehouse layout generators for demonstration and comparison."""

    @staticmethod
    def empty_arena(rows: int = 20, cols: int = 20) -> WarehouseGrid:
        """Open grid with start at top-left and goal at bottom-right."""
        grid = WarehouseGrid(rows, cols)
        grid.set_cell(0, 0, CellType.START)
        grid.set_cell(rows - 1, cols - 1, CellType.GOAL)
        return grid

    @staticmethod
    def amazon_fulfillment(rows: int = 25, cols: int = 35) -> WarehouseGrid:
        """
        Amazon-style Fulfillment Warehouse with parallel rack blocks,
        main central logistics highway, and cross-aisles.
        """
        grid = WarehouseGrid(rows, cols)
        
        # Draw shelf rack blocks (pairs of rows with aisles in between)
        for r in range(3, rows - 3):
            if r in (7, 8, 12, 13, 17, 18):
                for c in range(4, cols - 4):
                    # Leave cross-aisles open at columns 12, 22
                    if c not in (12, 13, 22, 23):
                        grid.set_cell(r, c, CellType.SHELF)
        
        # Add a heavy traffic zone along main central aisle
        for r in range(1, rows - 1):
            if grid.grid[r, 18] == CellType.EMPTY:
                grid.set_cell(r, 18, CellType.HEAVY_TRAFFIC, cost=3.0)

        # Set Start (Robot Dock) and Goal (Pickup Hub)
        grid.set_cell(1, 1, CellType.START)
        grid.set_cell(rows - 2, cols - 2, CellType.GOAL)
        return grid

    @staticmethod
    def dense_storage_rack(rows: int = 20, cols: int = 20) -> WarehouseGrid:
        """Grid of 2x2 shelf blocks with 1-cell aisles."""
        grid = WarehouseGrid(rows, cols)
        for r in range(2, rows - 2, 3):
            for c in range(2, cols - 2, 3):
                grid.set_cell(r, c, CellType.SHELF)
                if r + 1 < rows - 2:
                    grid.set_cell(r + 1, c, CellType.SHELF)
                if c + 1 < cols - 2:
                    grid.set_cell(r, c + 1, CellType.SHELF)
                if r + 1 < rows - 2 and c + 1 < cols - 2:
                    grid.set_cell(r + 1, c + 1, CellType.SHELF)
        
        grid.set_cell(0, 0, CellType.START)
        grid.set_cell(rows - 1, cols - 1, CellType.GOAL)
        return grid

    @staticmethod
    def bottleneck_maze(rows: int = 20, cols: int = 20) -> WarehouseGrid:
        """Multiple wall barriers creating winding bottlenecks."""
        grid = WarehouseGrid(rows, cols)
        
        # Wall 1 (top down with gap at bottom)
        for r in range(0, int(rows * 0.75)):
            grid.set_cell(r, int(cols * 0.3), CellType.OBSTACLE)
            
        # Wall 2 (bottom up with gap at top)
        for r in range(int(rows * 0.25), rows):
            grid.set_cell(r, int(cols * 0.6), CellType.OBSTACLE)
            
        # Wall 3 (top down with gap at bottom)
        for r in range(0, int(rows * 0.75)):
            grid.set_cell(r, int(cols * 0.8), CellType.OBSTACLE)

        grid.set_cell(rows // 2, 0, CellType.START)
        grid.set_cell(rows // 2, cols - 1, CellType.GOAL)
        return grid

    @staticmethod
    def weighted_traffic_depot(rows: int = 20, cols: int = 20) -> WarehouseGrid:
        """
        Layout highlighting difference between unweighted BFS (shortest hops)
        and weighted A*/Dijkstra (shortest cost).
        Direct path has heavy traffic (cost=5), while long bypass path is clear (cost=1).
        """
        grid = WarehouseGrid(rows, cols)
        
        # Middle direct corridor is heavy traffic
        mid_r = rows // 2
        for c in range(2, cols - 2):
            grid.set_cell(mid_r - 1, c, CellType.OBSTACLE)
            grid.set_cell(mid_r + 1, c, CellType.OBSTACLE)
            grid.set_cell(mid_r, c, CellType.HEAVY_TRAFFIC, cost=5.0)

        grid.set_cell(mid_r, 0, CellType.START)
        grid.set_cell(mid_r, cols - 1, CellType.GOAL)
        return grid
