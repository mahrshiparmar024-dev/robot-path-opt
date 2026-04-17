from flask import Flask, jsonify, render_template, request

from robot_path_optimization import (
    START,
    GOAL,
    compute_path_length,
    check_collision,
    optimize_path,
)

app = Flask(__name__, static_folder="static", template_folder="templates")


def serialize_path(path):
    return [[float(point[0]), float(point[1])] for point in path]


def serialize_obstacle(obs):
    """Serialize an obstacle dict for JSON transport."""
    result = {"type": obs["type"]}
    if obs["type"] == "circle":
        result["cx"] = float(obs["cx"])
        result["cy"] = float(obs["cy"])
        result["r"] = float(obs["r"])
    elif obs["type"] == "rect":
        result["cx"] = float(obs["cx"])
        result["cy"] = float(obs["cy"])
        result["w"] = float(obs["w"])
        result["h"] = float(obs["h"])
        result["angle"] = float(obs["angle"])
    elif obs["type"] == "polygon":
        result["vertices"] = [[float(v[0]), float(v[1])] for v in obs["vertices"]]
    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run")
def api_run():
    # Accept optional seed; if absent, each run is random
    seed = request.args.get("seed", default=None, type=int)

    optimized_path, initial_path, cost_history, path_snapshots, obstacles, used_seed = (
        optimize_path(START, GOAL, [], seed=seed)
    )

    return jsonify(
        {
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
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
