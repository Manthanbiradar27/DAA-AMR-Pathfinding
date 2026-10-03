import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List
from algorithms.base import PathResult
from warehouse.grid import WarehouseGrid, CellType

class PerformanceChartGenerator:
    """Generates comparison charts and visual overlays using Matplotlib."""

    @staticmethod
    def create_comparison_bar_chart(results: Dict[str, PathResult]) -> plt.Figure:
        """
        Creates a side-by-side grouped bar chart comparing Distance Traveled
        and Explored Cells for Dijkstra, A*, and BFS.
        """
        fig, ax1 = plt.subplots(figsize=(8, 4.5), facecolor='#0f172a')
        ax1.set_facecolor('#1e293b')

        algos = list(results.keys())
        distances = [res.distance if res.found else 0 for res in results.values()]
        explored = [len(res.explored_cells) for res in results.values()]

        x = np.arange(len(algos))
        width = 0.35

        # Bar colors
        color_dist = '#38bdf8'   # Sky blue
        color_exp = '#f43f5e'    # Vivid rose red

        rects1 = ax1.bar(x - width/2, distances, width, label='Total Path Cost', color=color_dist, edgecolor='#0284c7', linewidth=1.5)
        
        # Dual axis for explored cells
        ax2 = ax1.twinx()
        rects2 = ax2.bar(x + width/2, explored, width, label='Explored Cells', color=color_exp, edgecolor='#e11d48', linewidth=1.5)

        # Style left axis
        ax1.set_ylabel('Path Distance Cost', color=color_dist, fontsize=11, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color_dist, colors='#94a3b8')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algos, color='#f8fafc', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='x', colors='#f8fafc')

        # Style right axis
        ax2.set_ylabel('Cells Explored Count', color=color_exp, fontsize=11, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_exp, colors='#94a3b8')
        ax2.spines['top'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_color('#334155')
        ax2.spines['right'].set_color('#334155')
        ax1.spines['bottom'].set_color('#334155')
        ax1.grid(True, linestyle='--', alpha=0.15, color='#94a3b8')

        # Add data labels
        for rect in rects1:
            h = rect.get_height()
            if h > 0:
                ax1.annotate(f'{h:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', color='#e0f2fe', fontweight='bold', fontsize=9)

        for rect in rects2:
            h = rect.get_height()
            if h > 0:
                ax2.annotate(f'{h}',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', color='#fecdd3', fontweight='bold', fontsize=9)

        plt.title("Algorithm Performance Comparison", color='#f8fafc', fontsize=13, fontweight='bold', pad=12)
        fig.tight_layout()
        return fig

    @staticmethod
    def create_overlay_map(grid: WarehouseGrid, results: Dict[str, PathResult]) -> plt.Figure:
        """
        Overlays final paths from all 3 algorithms on the same warehouse grid
        to highlight trajectory overlaps and pathing differences.
        """
        fig, ax = plt.subplots(figsize=(8, 7), facecolor='#0f172a')
        ax.set_facecolor('#0f172a')

        grid_matrix = np.zeros((grid.rows, grid.cols, 3))
        # Fill floor
        grid_matrix[:, :] = [0.094, 0.125, 0.207]

        # Draw obstacles and shelves
        for r in range(grid.rows):
            for c in range(grid.cols):
                val = grid.grid[r, c]
                if val == CellType.OBSTACLE:
                    grid_matrix[r, c] = [0.2, 0.25, 0.33] # Dark steel
                elif val == CellType.SHELF:
                    grid_matrix[r, c] = [0.7, 0.32, 0.03] # Amber wood
                elif val == CellType.HEAVY_TRAFFIC:
                    grid_matrix[r, c] = [0.5, 0.3, 0.1] # Caution zone

        ax.imshow(grid_matrix, origin='upper')

        # Colors for each algorithm path
        colors = {
            "Dijkstra": '#06b6d4',  # Cyan
            "A*": '#a855f7',        # Purple
            "BFS": '#f97316'        # Orange
        }

        # Plot paths
        for name, res in results.items():
            if res.found and res.path:
                cols = [p[1] for p in res.path]
                rows = [p[0] for p in res.path]
                col = colors.get(name, '#ffffff')
                ax.plot(cols, rows, label=f"{name} (Cost: {res.distance:.1f})", color=col, linewidth=3.0, alpha=0.85, marker='o', markersize=4)

        # Draw Start and Goal
        sr, sc = grid.start
        gr, gc = grid.goal
        ax.plot(sc, sr, marker='s', color='#22c55e', markersize=12, label='Start (Dock)')
        ax.plot(gc, gr, marker='*', color='#ef4444', markersize=16, label='Goal (Pickup)')

        ax.set_xticks(np.arange(-0.5, grid.cols, 5))
        ax.set_yticks(np.arange(-0.5, grid.rows, 5))
        ax.grid(True, color='#334155', linestyle=':', linewidth=0.7)
        ax.tick_params(colors='#94a3b8')

        legend = ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', loc='upper right')
        ax.set_title("Multi-Algorithm Path Trajectory Overlay", color='#f8fafc', fontsize=13, fontweight='bold', pad=12)
        fig.tight_layout()
        return fig
