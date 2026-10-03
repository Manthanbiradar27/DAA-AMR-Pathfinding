import heapq
import time
from typing import Tuple, List, Dict, Set
from warehouse.grid import WarehouseGrid
from algorithms.base import BaseSolver, PathResult, AlgorithmStep

class AStarSolver(BaseSolver):
    """
    A* (A-Star) Search Algorithm Implementation.
    Uses evaluation function f(n) = g(n) + h(n), where g(n) is path cost from start,
    and h(n) is admissible heuristic distance (Manhattan distance for grid navigation).
    """

    def manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Admissible heuristic function for orthogonal 4-directional grid."""
        return float(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))

    def solve(self, grid: WarehouseGrid, start: Tuple[int, int], goal: Tuple[int, int]) -> PathResult:
        t0 = time.perf_counter()

        if not grid.is_walkable(start[0], start[1]) or not grid.is_walkable(goal[0], goal[1]):
            t1 = time.perf_counter()
            return PathResult(
                algorithm_name="A* Search",
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
                f_scores={start: 0.0},
                h_scores={start: 0.0},
                parent={start: None}
            )
            return PathResult(
                algorithm_name="A* Search",
                path=[start],
                explored_cells=[start],
                distance=0.0,
                hop_count=0,
                execution_time_ms=(t1 - t0) * 1000.0,
                steps=[step],
                found=True,
                g_scores={start: 0.0},
                f_scores={start: 0.0},
                h_scores={start: 0.0}
            )

        h_start = self.manhattan_distance(start, goal)
        g_scores: Dict[Tuple[int, int], float] = {start: 0.0}
        h_scores: Dict[Tuple[int, int], float] = {start: h_start}
        f_scores: Dict[Tuple[int, int], float] = {start: h_start}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {start: None}
        
        explored_order: List[Tuple[int, int]] = []
        explored_set: Set[Tuple[int, int]] = set()

        # Priority Queue: (f_score, h_score, node)
        # Using h_score as tie-breaker favors nodes closer to goal
        pq = [(h_start, h_start, start)]
        frontier_set = {start}

        steps: List[AlgorithmStep] = []
        step_count = 0
        found = False

        steps.append(AlgorithmStep(
            step_index=0,
            current=None,
            frontier=list(frontier_set),
            explored=[],
            g_scores=dict(g_scores),
            f_scores=dict(f_scores),
            h_scores=dict(h_scores),
            parent=dict(parent)
        ))

        while pq:
            current_f, current_h, current = heapq.heappop(pq)

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
                f_scores=dict(f_scores),
                h_scores=dict(h_scores),
                parent=dict(parent)
            ))

            if current == goal:
                found = True
                break

            for neighbor in grid.get_neighbors(current[0], current[1]):
                if neighbor in explored_set:
                    continue

                cost = grid.get_cost(neighbor[0], neighbor[1])
                tentative_g = g_scores[current] + cost

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h_val = self.manhattan_distance(neighbor, goal)
                    h_scores[neighbor] = h_val
                    f_val = tentative_g + h_val
                    f_scores[neighbor] = f_val
                    parent[neighbor] = current

                    heapq.heappush(pq, (f_val, h_val, neighbor))
                    frontier_set.add(neighbor)

        t1 = time.perf_counter()
        final_path = self.reconstruct_path(parent, start, goal) if found else []
        distance = g_scores.get(goal, float('inf')) if found else float('inf')
        hop_count = len(final_path) - 1 if final_path else 0

        return PathResult(
            algorithm_name="A* Search",
            path=final_path,
            explored_cells=explored_order,
            distance=distance,
            hop_count=hop_count,
            execution_time_ms=(t1 - t0) * 1000.0,
            steps=steps,
            found=found,
            g_scores=g_scores,
            f_scores=f_scores,
            h_scores=h_scores
        )
