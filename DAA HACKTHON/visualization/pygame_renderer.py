import os
import math
from typing import Tuple, List, Dict, Set, Optional
import numpy as np

# Set headless mode if SDL video driver is not specified
if "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame

class PygameGridRenderer:
    """
    Pygame-based high-performance warehouse grid renderer.
    Renders rich glowing visuals, heatmaps, numerical cost overlays,
    frontier sets, explored paths, and robot sprites.
    Converts surfaces to RGB numpy arrays for Streamlit rendering.
    """

    # Color Palette (Slate Dark Mode + Neon Accents)
    COLOR_BG = (15, 23, 42)          # Deep slate navy
    COLOR_GRID = (30, 41, 59)        # Subtle grid line
    COLOR_FLOOR = (24, 32, 53)       # Clean floor tile
    COLOR_OBSTACLE = (51, 65, 85)    # Steel dark wall
    COLOR_SHELF = (180, 83, 9)       # Warm amber/wood shelf rack
    COLOR_SHELF_BORDER = (245, 158, 11) # Bright amber outline
    COLOR_TRAFFIC = (180, 83, 9, 80) # Caution amber for heavy traffic
    
    # Node state colors
    COLOR_START = (34, 197, 94)      # Neon Emerald Green
    COLOR_GOAL = (239, 68, 68)       # Neon Crimson Red
    COLOR_FRONTIER = (234, 179, 8)   # Electric Yellow
    COLOR_CURRENT = (59, 130, 246)   # Electric Blue (Active Robot Head)
    
    # Path colors
    COLOR_PATH_DIJKSTRA = (6, 182, 212) # Cyan
    COLOR_PATH_ASTAR = (168, 85, 247)   # Neon Purple
    COLOR_PATH_BFS = (249, 115, 22)     # Vivid Orange

    # Heatmaps / Gradient stops
    GRADIENT_EXPLORED = [
        (30, 58, 138),   # Dark blue
        (59, 130, 246),  # Sky blue
        (147, 197, 253)  # Soft cyan
    ]

    def __init__(self, cell_size: int = 24):
        pygame.init()
        pygame.font.init()
        self.cell_size = max(12, cell_size)
        self.font_small = pygame.font.SysFont("monospace", max(8, self.cell_size // 3), bold=True)
        self.font_large = pygame.font.SysFont("sans-serif", max(10, self.cell_size // 2), bold=True)

    def _get_explored_color(self, step_idx: int, total_steps: int, algo_type: str = "A*") -> Tuple[int, int, int]:
        """Generate smooth color gradient for explored cells based on exploration order."""
        t = min(1.0, step_idx / max(1, total_steps))
        if "A*" in algo_type:
            # Purple to Pink gradient
            r = int(76 + t * (236 - 76))
            g = int(29 + t * (72 - 29))
            b = int(149 + t * (153 - 149))
            return (r, g, b)
        elif "Dijkstra" in algo_type:
            # Blue to Cyan gradient
            r = int(14 + t * (56 - 14))
            g = int(116 + t * (189 - 116))
            b = int(144 + t * (248 - 144))
            return (r, g, b)
        else:
            # Emerald to Yellow gradient for BFS
            r = int(16 + t * (202 - 16))
            g = int(185 + t * (138 - 185))
            b = int(129 + t * (4 - 129))
            return (r, g, b)

    def render_frame(
        self,
        grid,
        current_step: Optional[object] = None,
        final_path: Optional[List[Tuple[int, int]]] = None,
        algo_name: str = "A* Search",
        show_values: bool = False,
        value_mode: str = "g_score", # "g_score", "f_score", "h_score", "level"
        robot_pos: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Renders a complete frame surface and returns as RGB numpy array (height, width, 3).
        """
        width = grid.cols * self.cell_size
        height = grid.rows * self.cell_size
        surface = pygame.Surface((width, height))
        surface.fill(self.COLOR_BG)

        # 1. Base grid & obstacles
        for r in range(grid.rows):
            for c in range(grid.cols):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                cell_type = grid.grid[r, c]

                if cell_type == 1: # OBSTACLE
                    pygame.draw.rect(surface, self.COLOR_OBSTACLE, rect)
                    pygame.draw.rect(surface, (30, 41, 59), rect, 1)
                elif cell_type == 2: # SHELF
                    pygame.draw.rect(surface, self.COLOR_SHELF, rect)
                    pygame.draw.rect(surface, self.COLOR_SHELF_BORDER, rect, 1)
                    # Draw subtle shelf rack cross lines
                    mid_x = c * self.cell_size + self.cell_size // 2
                    mid_y = r * self.cell_size + self.cell_size // 2
                    pygame.draw.line(surface, (120, 53, 15), (rect.left+2, mid_y), (rect.right-2, mid_y), 1)
                elif cell_type == 3: # HEAVY_TRAFFIC
                    pygame.draw.rect(surface, (120, 53, 15), rect)
                    pygame.draw.rect(surface, self.COLOR_GRID, rect, 1)
                    # Hatching pattern
                    pygame.draw.line(surface, (180, 83, 9), (rect.left, rect.bottom), (rect.right, rect.top), 1)
                else:
                    pygame.draw.rect(surface, self.COLOR_FLOOR, rect)
                    pygame.draw.rect(surface, self.COLOR_GRID, rect, 1)

        # 2. Explored cells
        if current_step and current_step.explored:
            total_exp = len(current_step.explored)
            for idx, cell in enumerate(current_step.explored):
                if cell == grid.start or cell == grid.goal:
                    continue
                r, c = cell
                rect = pygame.Rect(c * self.cell_size + 1, r * self.cell_size + 1, self.cell_size - 2, self.cell_size - 2)
                col = self._get_explored_color(idx, total_exp, algo_name)
                pygame.draw.rect(surface, col, rect, border_radius=3)

        # 3. Frontier set
        if current_step and current_step.frontier:
            for cell in current_step.frontier:
                if cell == grid.start or cell == grid.goal:
                    continue
                r, c = cell
                rect = pygame.Rect(c * self.cell_size + 1, r * self.cell_size + 1, self.cell_size - 2, self.cell_size - 2)
                pygame.draw.rect(surface, (234, 179, 8, 100), rect, border_radius=3)
                pygame.draw.rect(surface, self.COLOR_FRONTIER, rect, 2, border_radius=3)

        # 4. Final Path
        path_color = self.COLOR_PATH_ASTAR if "A*" in algo_name else (self.COLOR_PATH_DIJKSTRA if "Dijkstra" in algo_name else self.COLOR_PATH_BFS)
        if final_path and len(final_path) > 1:
            points = [(c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2) for r, c in final_path]
            pygame.draw.lines(surface, path_color, False, points, max(3, self.cell_size // 5))
            
            # Draw glowing nodes along path
            for r, c in final_path:
                if (r, c) not in (grid.start, grid.goal):
                    cx = c * self.cell_size + self.cell_size // 2
                    cy = r * self.cell_size + self.cell_size // 2
                    pygame.draw.circle(surface, path_color, (cx, cy), max(3, self.cell_size // 6))

        # 5. Cell values overlay (numeric cost / heuristic / level)
        if show_values and current_step:
            for r in range(grid.rows):
                for c in range(grid.cols):
                    cell = (r, c)
                    val_str = None
                    if value_mode == "g_score" and cell in current_step.g_scores:
                        val_str = f"{current_step.g_scores[cell]:.0f}"
                    elif value_mode == "f_score" and cell in current_step.f_scores:
                        val_str = f"{current_step.f_scores[cell]:.0f}"
                    elif value_mode == "h_score" and cell in current_step.h_scores:
                        val_str = f"{current_step.h_scores[cell]:.0f}"
                    elif value_mode == "level" and cell in current_step.level:
                        val_str = f"{current_step.level[cell]}"

                    if val_str and self.cell_size >= 16:
                        txt = self.font_small.render(val_str, True, (255, 255, 255))
                        txt_rect = txt.get_rect(center=(c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2))
                        surface.blit(txt, txt_rect)

        # 6. Heuristic vector line (A* search direction)
        if "A*" in algo_name and current_step and current_step.current:
            cur_r, cur_c = current_step.current
            goal_r, goal_c = grid.goal
            start_pos = (cur_c * self.cell_size + self.cell_size // 2, cur_r * self.cell_size + self.cell_size // 2)
            goal_pos = (goal_c * self.cell_size + self.cell_size // 2, goal_r * self.cell_size + self.cell_size // 2)
            pygame.draw.line(surface, (236, 72, 153), start_pos, goal_pos, 1)

        # 7. Start & Goal markers
        sr, sc = grid.start
        start_rect = pygame.Rect(sc * self.cell_size, sr * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(surface, self.COLOR_START, start_rect, border_radius=4)
        s_txt = self.font_large.render("S", True, (255, 255, 255))
        surface.blit(s_txt, s_txt.get_rect(center=start_rect.center))

        gr, gc = grid.goal
        goal_rect = pygame.Rect(gc * self.cell_size, gr * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(surface, self.COLOR_GOAL, goal_rect, border_radius=4)
        g_txt = self.font_large.render("G", True, (255, 255, 255))
        surface.blit(g_txt, g_txt.get_rect(center=goal_rect.center))

        # 8. Robot AGV Sprite (Current expanding node or playback robot position)
        active_pos = robot_pos or (current_step.current if current_step else None)
        if active_pos and active_pos not in (grid.start, grid.goal):
            rr, rc = active_pos
            rcx = rc * self.cell_size + self.cell_size // 2
            rcy = rr * self.cell_size + self.cell_size // 2
            radius = max(4, self.cell_size // 3)
            # Glowing halo
            pygame.draw.circle(surface, (59, 130, 246, 150), (rcx, rcy), radius + 3)
            # Robot core
            pygame.draw.circle(surface, (255, 255, 255), (rcx, rcy), radius)
            pygame.draw.circle(surface, (37, 99, 235), (rcx, rcy), radius - 2)

        # Convert surface to numpy array (H, W, 3) RGB
        img_array = pygame.surfarray.array3d(surface)
        # Transpose array from (W, H, 3) Pygame format to (H, W, 3) Standard image format
        return np.transpose(img_array, (1, 0, 2))
