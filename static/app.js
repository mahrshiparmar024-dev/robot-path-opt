const runButton = document.getElementById('runButton');
const status = document.getElementById('status');
const metrics = document.getElementById('metrics');
const plotElement = document.getElementById('plot');
const seedInput = document.getElementById('seedInput');

/* ─── Shape drawing helpers ──────────────────────────────────────── */

function makeCircle(cx, cy, radius, resolution = 64) {
    const xs = [], ys = [];
    for (let i = 0; i <= resolution; i++) {
        const theta = (Math.PI * 2 * i) / resolution;
        xs.push(cx + radius * Math.cos(theta));
        ys.push(cy + radius * Math.sin(theta));
    }
    return { x: xs, y: ys };
}

function makeRect(cx, cy, w, h, angleDeg) {
    const corners = [
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2],
        [-w / 2, -h / 2],          // close the shape
    ];
    const rad = (angleDeg * Math.PI) / 180;
    const cosA = Math.cos(rad), sinA = Math.sin(rad);
    const xs = [], ys = [];
    for (const [lx, ly] of corners) {
        xs.push(cx + lx * cosA - ly * sinA);
        ys.push(cy + lx * sinA + ly * cosA);
    }
    return { x: xs, y: ys };
}

function makePolygon(vertices) {
    const xs = vertices.map(v => v[0]);
    const ys = vertices.map(v => v[1]);
    xs.push(vertices[0][0]);            // close the shape
    ys.push(vertices[0][1]);
    return { x: xs, y: ys };
}

/* ─── Obstacle colours by type ───────────────────────────────────── */

const OBSTACLE_PALETTE = {
    circle:  { fill: 'rgba(229, 57, 53, 0.30)', line: '#D32F2F' },
    rect:    { fill: 'rgba(171, 71, 188, 0.30)', line: '#8E24AA' },
    polygon: { fill: 'rgba(255, 112, 67, 0.30)', line: '#E64A19' },
};

/* ─── Build Plotly traces for obstacles ──────────────────────────── */

function buildObstacleTraces(obstacles) {
    return obstacles.map((obs, idx) => {
        let shape;
        if (obs.type === 'circle') {
            shape = makeCircle(obs.cx, obs.cy, obs.r);
        } else if (obs.type === 'rect') {
            shape = makeRect(obs.cx, obs.cy, obs.w, obs.h, obs.angle);
        } else if (obs.type === 'polygon') {
            shape = makePolygon(obs.vertices);
        } else {
            return null;
        }

        const palette = OBSTACLE_PALETTE[obs.type] || OBSTACLE_PALETTE.circle;
        const label = obs.type === 'circle' ? 'Circle'
                    : obs.type === 'rect'   ? 'Rect'
                    : 'Polygon';

        return {
            x: shape.x,
            y: shape.y,
            mode: 'lines',
            fill: 'toself',
            fillcolor: palette.fill,
            line: { color: palette.line, width: 2 },
            hoverinfo: 'text',
            text: `${label} #${idx + 1}`,
            name: `${label} ${idx + 1}`,
            showlegend: false,
        };
    }).filter(Boolean);
}

/* ─── Metrics display ────────────────────────────────────────────── */

function renderMetrics(data) {
    const circleCount  = data.obstacles.filter(o => o.type === 'circle').length;
    const rectCount    = data.obstacles.filter(o => o.type === 'rect').length;
    const polyCount    = data.obstacles.filter(o => o.type === 'polygon').length;
    const lengthDelta  = ((data.final_length - data.initial_length) / data.initial_length * 100).toFixed(1);

    metrics.textContent =
        `Start:               (${data.start[0].toFixed(1)}, ${data.start[1].toFixed(1)})\n` +
        `Goal:                (${data.goal[0].toFixed(1)}, ${data.goal[1].toFixed(1)})\n` +
        `Seed:                ${data.seed}\n` +
        `─────────────────────────────────────\n` +
        `Obstacles:           ${data.obstacles.length} total\n` +
        `  • Circles:         ${circleCount}\n` +
        `  • Rectangles:      ${rectCount}\n` +
        `  • Polygons:        ${polyCount}\n` +
        `─────────────────────────────────────\n` +
        `Initial path length: ${data.initial_length.toFixed(3)}\n` +
        `Optimized length:    ${data.final_length.toFixed(3)}\n` +
        `Length overhead:     ${lengthDelta}%\n` +
        `Iterations:          ${data.iterations}\n` +
        `Collision-free:      ${data.final_collision ? 'NO ✗' : 'YES ✓'}\n` +
        `Snapshots:           ${data.snapshots.length}`;
}

/* ─── Main plot ──────────────────────────────────────────────────── */

function drawSimulation(data) {
    const obstacleTraces = buildObstacleTraces(data.obstacles);

    const contourTrace = {
        z: data.V,
        x: data.x_grid,
        y: data.y_grid,
        type: 'contour',
        colorscale: 'RdYlGn',
        reversescale: true,
        opacity: 0.6,
        showscale: false,
        name: 'Potential Field',
        hoverinfo: 'none',
        contours: {
            coloring: 'heatmap'
        }
    };

    const endpointTrace = {
        x: [data.start[0], data.goal[0]],
        y: [data.start[1], data.goal[1]],
        mode: 'markers+text',
        marker: { size: 14, color: ['#4CAF50', '#FF9800'], line: { width: 1, color: '#333' } },
        text: ['START', 'GOAL'],
        textposition: ['top left', 'bottom right'],
        textfont: { size: 12, color: '#333', family: 'Inter, sans-serif' },
        name: 'Endpoints',
        hoverinfo: 'text',
    };

    const initialTrace = {
        x: data.initial_path.map(p => p[0]),
        y: data.initial_path.map(p => p[1]),
        mode: 'lines',
        line: { color: '#F44336', dash: 'dash', width: 2 },
        name: `Initial Path (L=${data.initial_length.toFixed(2)})`,
    };

    const animatedTrace = {
        x: data.snapshots[0].map(p => p[0]),
        y: data.snapshots[0].map(p => p[1]),
        mode: 'lines+markers',
        line: { color: '#1976D2', width: 4 },
        marker: { size: 5, color: '#1565C0' },
        name: `Optimized Path (L=${data.final_length.toFixed(2)})`,
    };

    const traceIdx = obstacleTraces.length + 2;  // after contour(1) + obstacles(N) + initialTrace(1)

    const frames = data.snapshots.map((snap, i) => ({
        name: `step-${i}`,
        data: [{ x: snap.map(p => p[0]), y: snap.map(p => p[1]) }],
        traces: [traceIdx]
    }));

    const sliderSteps = data.snapshots.map((_, i) => ({
        method: 'animate',
        label: `${i}`,
        args: [[`step-${i}`], { mode: 'immediate', frame: { duration: 0, redraw: true }, transition: { duration: 0 } }],
    }));

    const layout = {
        title: {
            text: 'Robot Path Optimization — Gradient Descent Simulation',
            font: { size: 16, family: 'Inter, sans-serif' },
        },
        xaxis: { title: 'X', range: [-1.5, 11.5], zeroline: false },
        yaxis: { title: 'Y', range: [-1.5, 11.5], scaleanchor: 'x', scaleratio: 1, zeroline: false },
        legend: { orientation: 'h', y: -0.15 },
        margin: { t: 55, b: 80, l: 60, r: 20 },
        updatemenus: [{
            type: 'buttons', showactive: false, y: 1.05, x: 1.02, xanchor: 'right',
            buttons: [
                { label: '▶ Play', method: 'animate', args: [null, { fromcurrent: true, frame: { duration: 400, redraw: true }, transition: { duration: 0 } }] },
                { label: '⏸ Pause', method: 'animate', args: [[null], { mode: 'immediate', frame: { duration: 0, redraw: false }, transition: { duration: 0 } }] },
            ],
        }],
        sliders: [{
            active: 0, pad: { t: 30 },
            currentvalue: { prefix: 'Snapshot: ' },
            steps: sliderSteps,
        }],
    };

    // animatedTrace must be at the right index so frames update it
    const allTraces = [contourTrace, ...obstacleTraces, initialTrace, animatedTrace, endpointTrace];

    Plotly.newPlot(plotElement, allTraces, layout, { responsive: true });
    Plotly.addFrames(plotElement, frames);
}

/* ─── Load / reload ──────────────────────────────────────────────── */

function loadSimulation() {
    status.textContent = 'Running simulation — generating random obstacles…';
    runButton.disabled = true;

    const seedValue = seedInput.value.trim();
    const url = seedValue ? `/api/run?seed=${encodeURIComponent(seedValue)}` : '/api/run';

    fetch(url)
        .then(r => r.json())
        .then(data => {
            status.textContent = data.final_collision
                ? 'Simulation loaded — WARNING: path has collisions.'
                : 'Simulation loaded — collision-free path found ✓';
            renderMetrics(data);
            drawSimulation(data);
            runButton.disabled = false;
            // Show the seed so user can reproduce
            seedInput.placeholder = `Last seed: ${data.seed}`;
        })
        .catch(err => {
            status.textContent = 'Unable to load the simulation. Check the server.';
            console.error(err);
            runButton.disabled = false;
        });
}

runButton.addEventListener('click', loadSimulation);
window.addEventListener('DOMContentLoaded', () => {
    loadSimulation();
});
