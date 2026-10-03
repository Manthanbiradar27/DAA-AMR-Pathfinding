from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
import time
from warehouse.grid import WarehouseGrid

@dataclass
class AlgorithmStep:
    step_index: int
    current: Optional[Tuple[int, int]]
    frontier: List[Tuple[int, int]]
    explored: List[Tuple[int, int]]
    g_scores: Dict[Tuple[int, int], float]
    f_scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    h_scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    parent: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)
    level: Dict[Tuple[int, int], int] = field(default_factory=dict)

@dataclass
class PathResult:
    algorithm_name: str
    path: List[Tuple[int, int]]
    explored_cells: List[Tuple[int, int]]
    distance: float
    hop_count: int
    execution_time_ms: float
    steps: List[AlgorithmStep]
    found: bool
    g_scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    f_scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    h_scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    level_map: Dict[Tuple[int, int], int] = field(default_factory=dict)

class BaseSolver(ABC):
    """
    Abstract Base Class for all pathfinding solvers.
    Common interface: solve(grid, start, goal) -> PathResult
    """

    @abstractmethod
    def solve(self, grid: WarehouseGrid, start: Tuple[int, int], goal: Tuple[int, int]) -> PathResult:
        """Execute pathfinding algorithm on grid from start to goal."""
        pass

    def reconstruct_path(self, parent: Dict[Tuple[int, int], Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from start to goal using parent map."""
        if goal not in parent and start != goal:
            return []
        path = []
        curr = goal
        while curr is not None:
            path.append(curr)
            curr = parent.get(curr)
        path.reverse()
        if path and path[0] == start:
            return path
        return []
