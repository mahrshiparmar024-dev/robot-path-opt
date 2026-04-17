import re
import sys

with open("robot_path_optimization.py", "r") as f:
    text = f.read()

# 1. Imports
text = text.replace(
    "from matplotlib import cm\nimport time",
    "from matplotlib import cm\nimport time\nfrom scipy.interpolate import RegularGridInterpolator"
)

# 2. compute_potential_gradient_field
old_pot = """def compute_potential_gradient_field(X, Y, goal, circle_primitives):
    \"\"\"
    Compute the negative gradient (force field) F = -∇V over the grid.

    Vector Calculus: GRADIENT of scalar field
        ∇V = (∂V/∂x, ∂V/∂y)  computed via np.gradient (central differences)
        F = -∇V points toward lower potential (goal) and away from obstacles.

    Also computes DIVERGENCE: div(F) = ∂Fx/∂x + ∂Fy/∂y
    \"\"\"
    V = total_potential(X, Y, goal, circle_primitives)
    dV_dy, dV_dx = np.gradient(V)
    Fx, Fy = -dV_dx, -dV_dy  # Force = negative gradient

    # Divergence: div(F) = ∂Fx/∂x + ∂Fy/∂y
    _, dFx_dx = np.gradient(Fx)
    dFy_dy, _ = np.gradient(Fy)
    divergence = dFx_dx + dFy_dy

    return V, Fx, Fy, divergence"""

new_pot = """def compute_potential_gradient_field(X, Y, goal, circle_primitives):
    \"\"\"
    Compute the negative gradient (force field) F = -∇V over the grid.

    Vector Calculus: GRADIENT of scalar field
        ∇V = (∂V/∂x, ∂V/∂y)  computed via np.gradient (central differences)
        F = -∇V points toward lower potential (goal) and away from obstacles.

    Also computes DIVERGENCE: div(F) = ∂Fx/∂x + ∂Fy/∂y
    And CURL: curl(F) = ∂Fy/∂x - ∂Fx/∂y
    \"\"\"
    V = total_potential(X, Y, goal, circle_primitives)
    
    # Calculate grid spacings explicitly to make np.gradient robust
    dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
    dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
    
    # np.gradient returns [d/d_axis0, d/d_axis1]. With default indexing='xy', axis 0 is Y, axis 1 is X.
    dV_dy, dV_dx = np.gradient(V, dy, dx)
    Fx, Fy = -dV_dx, -dV_dy  # Force = negative gradient

    # Divergence: div(F) = ∂Fx/∂x + ∂Fy/∂y and Curl: ∂Fy/∂x - ∂Fx/∂y
    dFx_dy, dFx_dx = np.gradient(Fx, dy, dx)
    dFy_dy, dFy_dx = np.gradient(Fy, dy, dx)
    divergence = dFx_dx + dFy_dy
    curl = dFy_dx - dFx_dy # Exact physical definition of curl for 2D vector field

    return V, Fx, Fy, divergence, curl"""
text = text.replace(old_pot, new_pot)

# 3. Replace optimize_path completely and delete compute_total_gradient and post_process_collision_repair
# We'll use a regex that matches from `# ==============================================================================
# GRADIENT DESCENT OPTIMIZATION` down to `# VISUALIZATION`
import re
pattern = re.compile(r'# ==============================================================================\n# GRADIENT DESCENT OPTIMIZATION (Vector Calculus: Gradient)\n# ==============================================================================.*?# ==============================================================================\n# VISUALIZATION\n# ==============================================================================', re.DOTALL)

new_opt = """# ==============================================================================
# APF TRAJECTORY SIMULATION (Vector Calculus: Integral Curves)
# ==============================================================================

def optimize_path(start, goal, obstacles, seed=None):
    \"\"\"
    Simulate path using Euler integration over the Artificial Potential Field (APF).

    Vector Calculus: INTEGRAL CURVE of the force field
        q_{t+1} = q_t + α * F(q_t)
        Simulates the robot moving down the potential gradient.

    Returns:
        optimized_path, initial_path, cost_history, path_snapshots, obstacles, seed, (grid_data)
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
    
    path_snapshots = []
    
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
    cost_history = [np.linalg.norm(p - goal) for p in path_snapshots]
    
    grid_data = (x_grid, y_grid, V, Fx, Fy, divergence, curl)
    
    return path_arr, initial_path, cost_history, path_snapshots, obstacles, seed, grid_data


# ==============================================================================
# VISUALIZATION
# =============================================================================="""

text = pattern.sub(new_opt.replace('\\', '\\\\'), text)

# Now fix plot_all_results
# 1. Update signature to accept curl
text = text.replace(
    "def plot_all_results(X, Y, V, Fx, Fy, initial_path, optimized_path,",
    "def plot_all_results(X, Y, V, Fx, Fy, curl, initial_path, optimized_path,"
)

# 2. Replace the Convergence Graph plot 4 with Curl heatmap and fix tight_layout
old_plot4 = """    # ── PLOT 4: Convergence Graph (Cost vs Iteration) ──
    ax4 = axes[1, 1]
    iters = np.arange(len(cost_history))

    # Color gradient line (red → green)
    points = np.array([iters, cost_history]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(cost_history) - 1)
    lc = LineCollection(segments, cmap='RdYlGn', norm=norm)
    lc.set_array(iters[:-1])
    lc.set_linewidth(2.5)
    ax4.add_collection(lc)

    # Reference lines
    ax4.axhline(y=initial_length, color='red', linestyle='--', alpha=0.6,
                label=f'Initial Length ({initial_length:.2f})')
    ax4.axhline(y=final_length, color='green', linestyle='--', alpha=0.6,
                label=f'Final Length ({final_length:.2f})')

    ax4.set_title('Convergence: Path Length vs Iteration', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Iteration Number', fontsize=11)
    ax4.set_ylabel('Path Length (Line Integral)', fontsize=11)
    ax4.set_xlim(0, len(cost_history) - 1)
    ax4.set_ylim(min(cost_history) * 0.95, max(cost_history) * 1.05)
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3)"""

new_plot4 = """    # ── PLOT 4: Curl Magnitude Map (∇×F) ──
    ax4 = axes[1, 1]
    
    # Render the curl to show it is almost zero everywhere (conservative field)
    c_plot = ax4.contourf(X, Y, curl, levels=30, cmap='coolwarm', alpha=0.8)
    fig.colorbar(c_plot, ax=ax4, fraction=0.046, pad=0.04)

    draw_obstacles(ax4, obstacles)
    ax4.plot(*start, '*', color='limegreen', markersize=15, markeredgecolor='black', zorder=6)
    ax4.plot(*goal, '*', color='red', markersize=15, markeredgecolor='black', zorder=6)

    ax4.set_title(r'Conservative Verification: Curl $\nabla \\times \mathbf{F} \approx 0$', fontsize=13, fontweight='bold')
    ax4.set_xlabel('X', fontsize=11)
    ax4.set_ylabel('Y', fontsize=11)
    ax4.set_xlim(-0.5, 10.5)
    ax4.set_ylim(-0.5, 10.5)
    ax4.set_aspect('equal')"""

text = text.replace(old_plot4, new_plot4)

# 3. Update main() references
# Fix compute_potential_gradient_field unpacking
text = text.replace(
    "V, Fx, Fy, divergence = compute_potential_gradient_field(X, Y, GOAL, circle_primitives)",
    "V, Fx, Fy, divergence, curl = compute_potential_gradient_field(X, Y, GOAL, circle_primitives)"
)

# Replace optimize_path unpack
text = text.replace(
    "optimized_path, _, cost_history, path_snapshots, obstacles, seed = optimize_path(START, GOAL, obstacles)",
    "optimized_path, _, cost_history, path_snapshots, obstacles, seed, grid_data = optimize_path(START, GOAL, obstacles)"
)

# Update plot_all_results call in main
text = text.replace(
    "plot_all_results(X, Y, V, Fx, Fy, initial_path, optimized_path,",
    "plot_all_results(X, Y, V, Fx, Fy, curl, initial_path, optimized_path,"
)

with open("robot_path_optimization.py", "w") as f:
    f.write(text)

print("Patch applied successfully.")
