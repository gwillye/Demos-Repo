# A* Pathfinding (grid maze)

A self-contained **algorithms demo**: A* search finds the shortest path through a randomly generated grid maze, and visualizes the walls, the explored frontier, and the final path.

## What it does
1. Generates a seeded grid maze (auto-tunes wall density so a path exists).
2. Runs **A\*** with a Manhattan-distance heuristic (`heapq` priority queue).
3. Reports path length, nodes expanded, and search efficiency.
4. Renders the maze, expanded nodes and path to `results/path.png`.

## Results (reproducible, seed = 42)
See [`results/RESULTS.md`](results/RESULTS.md) and `results/path.png`.

## Run
```bash
pip install -r requirements.txt
python astar_demo.py
```

## Stack
Python · NumPy · matplotlib (`heapq` from the standard library)
