import time
import json
from robot_path_optimization import optimize_path, START, GOAL, compute_path_length, check_collision
from app import serialize_obstacle, serialize_path

t0 = time.time()
optimized_path, initial_path, cost_history, path_snapshots, obstacles, used_seed, grid_data = optimize_path(START, GOAL, [])
t1 = time.time()
x_grid, y_grid, V, _, _, _, _ = grid_data

data = {
    "start": [float(START[0]), float(START[1])],
    "goal": [float(GOAL[0]), float(GOAL[1])],
    "seed": int(used_seed),
    "obstacles": [serialize_obstacle(obs) for obs in obstacles],
    "initial_path": serialize_path(initial_path),
    "optimized_path": serialize_path(optimized_path),
    "snapshots": [serialize_path(p) for p in path_snapshots],
    "initial_length": float(compute_path_length(initial_path)),
    "final_length": float(compute_path_length(optimized_path)),
    "iterations": len(cost_history),
    "final_collision": bool(check_collision(optimized_path, obstacles)),
    "cost_history": [float(c) for c in cost_history],
    "x_grid": [float(v) for v in x_grid[::2]],
    "y_grid": [float(v) for v in y_grid[::2]],
    "V": V[::2, ::2].tolist(),
}
t2 = time.time()
js = json.dumps(data)
t3 = time.time()

print(f"Optimize time: {t1-t0:.3f}s")
print(f"Dict prep time: {t2-t1:.3f}s")
print(f"JSON dump time: {t3-t2:.3f}s")
print(f"JSON length: {len(js)} bytes")
