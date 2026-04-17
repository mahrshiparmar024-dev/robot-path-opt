with open("robot_path_optimization.py", "r") as f:
    lines = f.readlines()

new_opt = """def optimize_path(start, goal, obstacles, seed=None):
    \"\"\"
    Simulate path using Euler integration over the Artificial Potential Field (APF).

    Vector Calculus: INTEGRAL CURVE of the force field
        q_{t+1} = q_t + α * F(q_t)
        Simulates the robot moving down the potential gradient.

    Returns:
        optimized_path, initial_path, cost_history, path_snapshots, obstacles, seed, grid_data
    \"\"\"
    if not obstacles or (isinstance(obstacles[0], (tuple, list)) and len(obstacles[0]) == 3):
        obstacles, seed = generate_random_obstacles(seed)
    elif seed is None:
        seed = int(time.time() * 1000) % (2**31)

    circle_primitives = obstacles_to_circle_primitives(obstacles)
    
    # Pre-compute the force field over a grid for interpolation
    grid_res = 100
    x_grid = np.linspace(-1.5, 11.5, grid_res)
    y_grid = np.linspace(-1.5, 11.5, grid_res)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    V, Fx, Fy, divergence, curl = compute_potential_gradient_field(X, Y, goal, circle_primitives)
    
    # Setup RegularGridInterpolator (note: meshgrid xy means axes are y_grid, x_grid)
    interp_Fx = RegularGridInterpolator((y_grid, x_grid), Fx, bounds_error=False, fill_value=0)
    interp_Fy = RegularGridInterpolator((y_grid, x_grid), Fy, bounds_error=False, fill_value=0)
    
    path = [start.copy()]
    q = start.copy()
    
    path_snapshots = [[start.copy()]]
    
    # Euler integration parameters
    alpha = LEARNING_RATE * 5.0  # Step size along the gradient
    max_steps = 1500
    
    for iteration in range(max_steps):
        dist_to_goal = np.linalg.norm(q - goal)
        if dist_to_goal < 0.2:
            path.append(goal.copy())
            break
            
        # Interpolate force at current continuous position q
        # Ensure we pass (y, x) to the interpolator because V is (y_grid, x_grid)
        fx = interp_Fx((q[1], q[0]))
        fy = interp_Fy((q[1], q[0]))
        
        # Normalize force to ensure bounded steps
        F = np.array([fx, fy])
        mag = np.linalg.norm(F)
        if mag > 1e-5:
            F = F / mag * min(mag, 5.0)
            
        # q_{t+1} = q_t + α·F(q_t)
        q = q + alpha * F
        
        # Clamp to bounds
        q[0] = np.clip(q[0], -1, 11)
        q[1] = np.clip(q[1], -1, 11)
        
        path.append(q.copy())
        
        if iteration % 15 == 0:
            path_snapshots.append(np.array(path).copy())
            
    path_arr = np.array(path)
    initial_path = create_straight_path(start, goal, N_WAYPOINTS)
    
    path_snapshots.append(path_arr.copy())
    
    # Cost history is tracked roughly as distance to goal for plotting structure compatibility
    cost_history = [np.linalg.norm(p[-1] if isinstance(p, list) else p - goal) for p in path_snapshots]
    cost_history = [np.linalg.norm(p[-1] - goal) if len(p.shape) > 1 else np.linalg.norm(p - goal) for p in path_snapshots]
    
    # We return V to help the web app
    grid_data = (x_grid, y_grid, V, Fx, Fy, divergence, curl)
    
    return path_arr, initial_path, cost_history, path_snapshots, obstacles, seed, grid_data
"""

# Replace lines 489 to 661 inside the list
# Note: Python lists are 0-indexed. Line 489 corresponds to index 488. Line 661 corresponds to index 660.
# So we replace lines[488:661]

lines[488:661] = [new_opt + "\\n"]

with open("robot_path_optimization.py", "w") as f:
    f.writelines(lines)

print("Patch via python line index applied successfully.")
