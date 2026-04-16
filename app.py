from functools import lru_cache

from flask import Flask, jsonify, render_template

from robot_path_optimization import (
    START,
    GOAL,
    OBSTACLES,
    compute_path_length,
    check_collision,
    optimize_path,
)

app = Flask(__name__, static_folder="static", template_folder="templates")


def serialize_path(path):
    return [[float(point[0]), float(point[1])] for point in path]


@lru_cache(maxsize=1)
def get_simulation_data():
    optimized_path, initial_path, cost_history, path_snapshots = optimize_path(
        START, GOAL, OBSTACLES
    )
    return {
        "start": [float(START[0]), float(START[1])],
        "goal": [float(GOAL[0]), float(GOAL[1])],
        "obstacles": [[float(cx), float(cy), float(r)] for cx, cy, r in OBSTACLES],
        "initial_path": serialize_path(initial_path),
        "optimized_path": serialize_path(optimized_path),
        "snapshots": [serialize_path(path) for path in path_snapshots],
        "initial_length": float(compute_path_length(initial_path)),
        "final_length": float(compute_path_length(optimized_path)),
        "iterations": len(cost_history),
        "final_collision": bool(check_collision(optimized_path, OBSTACLES)),
        "cost_history": [float(cost) for cost in cost_history],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run")
def api_run():
    return jsonify(get_simulation_data())


if __name__ == "__main__":
    app.run(debug=True)
