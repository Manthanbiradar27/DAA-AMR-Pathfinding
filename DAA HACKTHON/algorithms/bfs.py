from collections import deque
import time
from typing import Tuple, List, Dict, Set
from warehouse.grid import WarehouseGrid
from algorithms.base import BaseSolver, PathResult, AlgorithmStep

class BFSSolver(BaseSolver):
    """
    Breadth-First Search (BFS) Implementation.
    Explores grid level-by-level in concentric waves using FIFO queue.
    Guarantees minimum-hop path on unweighted graphs.
    """

    def solve(self, grid: WarehouseGrid, start: Tuple[int, int], goal: Tuple[int, int]) -> PathResult:
        t0 = time.perf_counter()

        if not grid.is_walkable(start[0], start[1]) or not grid.is_walkable(goal[0], goal[1]):
            t1 = time.perf_counter()
            return PathResult(
                algorithm_name="Breadth-First Search",
                path=[],
                explored_cells=[],
                distance=float('inf'),
                hop_count=0,
                execution_time_ms=(t1 - t0) * 1000.0,
                steps=[],
                found=False
            )

        if start == goal:
            t1 = time.perf_counter()
            step = AlgorithmStep(
                step_index=0,
                current=start,
                frontier=[],
                explored=[start],
                g_scores={start: 0.0},
                level={start: 0},
                parent={start: None}
            )
            return PathResult(
                algorithm_name="Breadth-First Search",
                path=[start],
                explored_cells=[start],
                distance=0.0,
                hop_count=0,
                execution_time_ms=(t1 - t0) * 1000.0,
                steps=[step],
                found=True,
                g_scores={start: 0.0},
                level_map={start: 0}
            )

        queue = deque([start])
        visited: Set[Tuple[int, int]] = {start}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {start: None}
        level_map: Dict[Tuple[int, int], int] = {start: 0}
        g_scores: Dict[Tuple[int, int], float] = {start: 0.0}

        explored_order: List[Tuple[int, int]] = []
        steps: List[AlgorithmStep] = []
        step_count = 0
        found = False

        steps.append(AlgorithmStep(
            step_index=0,
            current=None,
            frontier=list(queue),
            explored=[],
            g_scores=dict(g_scores),
            level=dict(level_map),
            parent=dict(parent)
        ))

        while queue:
            current = queue.popleft()
            explored_order.append(current)

            step_count += 1
            steps.append(AlgorithmStep(
                step_index=step_count,
                current=current,
                frontier=list(queue),
                explored=list(explored_order),
                g_scores=dict(g_scores),
                level=dict(level_map),
                parent=dict(parent)
            ))

            if current == goal:
                found = True
                break

            current_level = level_map[current]
            current_g = g_scores[current]

            for neighbor in grid.get_neighbors(current[0], current[1]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    level_map[neighbor] = current_level + 1
                    cost = grid.get_cost(neighbor[0], neighbor[1])
                    g_scores[neighbor] = current_g + cost
                    queue.append(neighbor)

        t1 = time.perf_counter()
        final_path = self.reconstruct_path(parent, start, goal) if found else []
        
        # Calculate actual weighted distance along final path
        if found and final_path:
            distance = 0.0
            for i in range(1, len(final_path)):
                r, c = final_path[i]
                distance += grid.get_cost(r, c)
            hop_count = len(final_path) - 1
        else:
            distance = float('inf')
            hop_count = 0

        return PathResult(
            algorithm_name="Breadth-First Search",
            path=final_path,
            explored_cells=explored_order,
            distance=distance,
            hop_count=hop_count,
            execution_time_ms=(t1 - t0) * 1000.0,
            steps=steps,
            found=found,
            g_scores=g_scores,
            level_map=level_map
        )
