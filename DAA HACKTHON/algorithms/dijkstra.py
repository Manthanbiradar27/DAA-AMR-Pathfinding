import heapq
import time
from typing import Tuple, List, Dict, Set
from warehouse.grid import WarehouseGrid
from algorithms.base import BaseSolver, PathResult, AlgorithmStep

class DijkstraSolver(BaseSolver):
    """
    Dijkstra's Algorithm Implementation.
    Repeatedly selects node with smallest tentative cost g(n) from start.
    Guarantees shortest path in non-negatively weighted graphs.
    """

    def solve(self, grid: WarehouseGrid, start: Tuple[int, int], goal: Tuple[int, int]) -> PathResult:
        t0 = time.perf_counter()
        
        # Edge cases check
        if not grid.is_walkable(start[0], start[1]) or not grid.is_walkable(goal[0], goal[1]):
            t1 = time.perf_counter()
            return PathResult(
                algorithm_name="Dijkstra's Algorithm",
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
                parent={start: None}
            )
            return PathResult(
                algorithm_name="Dijkstra's Algorithm",
                path=[start],
                explored_cells=[start],
                distance=0.0,
                hop_count=0,
                execution_time_ms=(t1 - t0) * 1000.0,
                steps=[step],
                found=True,
                g_scores={start: 0.0}
            )

        g_scores: Dict[Tuple[int, int], float] = {start: 0.0}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {start: None}
        explored_order: List[Tuple[int, int]] = []
        explored_set: Set[Tuple[int, int]] = set()

        # Priority Queue: (g_score, node)
        pq = [(0.0, start)]
        frontier_set = {start}

        steps: List[AlgorithmStep] = []
        step_count = 0
        found = False

        # Initial step
        steps.append(AlgorithmStep(
            step_index=0,
            current=None,
            frontier=list(frontier_set),
            explored=[],
            g_scores=dict(g_scores),
            parent=dict(parent)
        ))

        while pq:
            current_g, current = heapq.heappop(pq)

            if current in explored_set:
                continue

            explored_set.add(current)
            explored_order.append(current)
            if current in frontier_set:
                frontier_set.remove(current)

            step_count += 1
            steps.append(AlgorithmStep(
                step_index=step_count,
                current=current,
                frontier=list(frontier_set),
                explored=list(explored_order),
                g_scores=dict(g_scores),
                parent=dict(parent)
            ))

            if current == goal:
                found = True
                break

            for neighbor in grid.get_neighbors(current[0], current[1]):
                if neighbor in explored_set:
                    continue
                
                cost = grid.get_cost(neighbor[0], neighbor[1])
                tentative_g = current_g + cost

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    parent[neighbor] = current
                    heapq.heappush(pq, (tentative_g, neighbor))
                    frontier_set.add(neighbor)

        t1 = time.perf_counter()
        final_path = self.reconstruct_path(parent, start, goal) if found else []
        distance = g_scores.get(goal, float('inf')) if found else float('inf')
        hop_count = len(final_path) - 1 if final_path else 0

        return PathResult(
            algorithm_name="Dijkstra's Algorithm",
            path=final_path,
            explored_cells=explored_order,
            distance=distance,
            hop_count=hop_count,
            execution_time_ms=(t1 - t0) * 1000.0,
            steps=steps,
            found=found,
            g_scores=g_scores
        )
