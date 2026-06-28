"""A* pathfinding on a randomly generated grid maze.

Self-contained (stdlib heapq + numpy + matplotlib), seeded for reproducibility.
Saves results/path.png and results/RESULTS.md, and prints a self-check.
"""
import os
import heapq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

H, W = 30, 40


def neighbors(node, grid):
    h, w = grid.shape
    r, c = node
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
            yield (nr, nc)


def heuristic(a, b):  # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    open_heap = [(heuristic(start, goal), start)]
    came_from = {}
    g = {start: 0}
    visited = set()
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1], visited
        if cur in visited:
            continue
        visited.add(cur)
        for nb in neighbors(cur, grid):
            ng = g[cur] + 1
            if ng < g.get(nb, float("inf")):
                g[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_heap, (ng + heuristic(nb, goal), nb))
    return None, visited


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(here, "results")
    os.makedirs(res, exist_ok=True)

    start, goal = (0, 0), (H - 1, W - 1)
    base = np.random.random((H, W))  # fixed random field; lower threshold => fewer walls
    path, visited, used_p = None, set(), None
    for wall_p in (0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.15):
        grid = (base < wall_p).astype(int)
        grid[start], grid[goal] = 0, 0
        path, visited = astar(grid, start, goal)
        if path:
            used_p = wall_p
            break

    # Visualization: walls black, visited light blue, path red, start/goal markers
    img = np.ones((H, W, 3))
    img[grid == 1] = (0.15, 0.15, 0.18)
    for (r, c) in visited:
        img[r, c] = (0.75, 0.85, 1.0)
    for (r, c) in path:
        img[r, c] = (0.90, 0.20, 0.20)
    plt.figure(figsize=(8, 6))
    plt.imshow(img, interpolation="nearest")
    plt.scatter([start[1]], [start[0]], c="green", s=60, marker="o", label="start")
    plt.scatter([goal[1]], [goal[0]], c="black", s=60, marker="*", label="goal")
    plt.title(f"A* path — length {len(path)}, {len(visited)} nodes expanded")
    plt.legend(loc="upper right")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(res, "path.png"), dpi=110)
    plt.close()

    with open(os.path.join(res, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("# A* Pathfinding — Results\n\n")
        f.write(f"- Grid: **{H}x{W}**, wall density used: **{used_p}**\n")
        f.write(f"- Path length: **{len(path)}** cells\n")
        f.write(f"- Nodes expanded: **{len(visited)}**\n")
        f.write(f"- Efficiency (path / expanded): **{len(path) / len(visited):.2f}**\n\n")
        f.write("See `path.png` for the visualization (walls, expanded nodes, final path).\n")

    print(f"path length = {len(path)}, expanded = {len(visited)}")
    print("self-check:", "OK" if path and path[0] == start and path[-1] == goal else "FAIL")


if __name__ == "__main__":
    main()
