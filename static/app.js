const runButton = document.getElementById('runButton');
const status = document.getElementById('status');
const metrics = document.getElementById('metrics');
const plotElement = document.getElementById('plot');

function makeCircle(cx, cy, radius, resolution = 64) {
    const circle = { x: [], y: [] };
    for (let i = 0; i <= resolution; i += 1) {
        const theta = (Math.PI * 2 * i) / resolution;
        circle.x.push(cx + radius * Math.cos(theta));
        circle.y.push(cy + radius * Math.sin(theta));
    }
    return circle;
}

function renderMetrics(data) {
    metrics.textContent = `Start: (${data.start[0].toFixed(2)}, ${data.start[1].toFixed(2)})\n` +
        `Goal: (${data.goal[0].toFixed(2)}, ${data.goal[1].toFixed(2)})\n` +
        `Obstacles: ${data.obstacles.length}\n` +
        `Initial path length: ${data.initial_length.toFixed(3)}\n` +
        `Optimized path length: ${data.final_length.toFixed(3)}\n` +
        `Iterations used: ${data.iterations}\n` +
        `Collision-free: ${data.final_collision ? 'YES' : 'NO'}\n` +
        `Snapshots: ${data.snapshots.length}`;
}

function drawSimulation(data) {
    const obstacleTraces = data.obstacles.map((obstacle, index) => {
        const [cx, cy, radius] = obstacle;
        const circle = makeCircle(cx, cy, radius);
        return {
            x: circle.x,
            y: circle.y,
            mode: 'lines',
            fill: 'toself',
            fillcolor: 'rgba(229, 57, 53, 0.25)',
            line: { color: '#D32F2F', width: 2 },
            hoverinfo: 'none',
            name: `Obstacle ${index + 1}`,
        };
    });

    const endpointTrace = {
        x: [data.start[0], data.goal[0]],
        y: [data.start[1], data.goal[1]],
        mode: 'markers+text',
        marker: { size: 14, color: ['#4CAF50', '#FF9800'], line: { width: 1, color: '#333' } },
        text: ['Start', 'Goal'],
        textposition: ['top left', 'bottom right'],
        name: 'Endpoints',
        hoverinfo: 'text',
    };

    const initialTrace = {
        x: data.initial_path.map(point => point[0]),
        y: data.initial_path.map(point => point[1]),
        mode: 'lines',
        line: { color: '#F44336', dash: 'dash', width: 2 },
        name: 'Initial Path',
    };

    const animatedTrace = {
        x: data.snapshots[0].map(point => point[0]),
        y: data.snapshots[0].map(point => point[1]),
        mode: 'lines+markers',
        line: { color: '#1976D2', width: 4 },
        marker: { size: 6, color: '#1565C0' },
        name: 'Optimized Path',
    };

    const frames = data.snapshots.map((snapshot, index) => ({
        name: `step-${index}`,
        data: [{
            x: snapshot.map(point => point[0]),
            y: snapshot.map(point => point[1]),
        }],
    }));

    const sliderSteps = data.snapshots.map((snapshot, index) => ({
        method: 'animate',
        label: `${index}`,
        args: [[`step-${index}`], { mode: 'immediate', frame: { duration: 0, redraw: true }, transition: { duration: 0 } }],
    }));

    const layout = {
        title: 'Robot Path Optimization — Gradient Descent Simulation',
        xaxis: { title: 'X', range: [-1, 11], zeroline: false },
        yaxis: { title: 'Y', range: [-1, 11], scaleanchor: 'x', scaleratio: 1, zeroline: false },
        legend: { orientation: 'h', y: -0.15 },
        margin: { t: 50, b: 80, l: 60, r: 20 },
        updatemenus: [
            {
                type: 'buttons',
                showactive: false,
                y: 1.05,
                x: 1.02,
                xanchor: 'right',
                buttons: [
                    {
                        label: 'Play',
                        method: 'animate',
                        args: [null, { fromcurrent: true, frame: { duration: 400, redraw: true }, transition: { duration: 0 } }],
                    },
                    {
                        label: 'Pause',
                        method: 'animate',
                        args: [[null], { mode: 'immediate', frame: { duration: 0, redraw: false }, transition: { duration: 0 } }],
                    },
                ],
            },
        ],
        sliders: [
            {
                active: 0,
                pad: { t: 30 },
                currentvalue: { prefix: 'Snapshot: ' },
                steps: sliderSteps,
            },
        ],
    };

    Plotly.newPlot(plotElement, [...obstacleTraces, initialTrace, animatedTrace, endpointTrace], layout, { responsive: true });
    Plotly.addFrames(plotElement, frames);
}

function loadSimulation() {
    status.textContent = 'Running simulation...';
    runButton.disabled = true;
    fetch('/api/run')
        .then(response => response.json())
        .then(data => {
            status.textContent = 'Simulation loaded successfully.';
            renderMetrics(data);
            drawSimulation(data);
        })
        .catch(error => {
            status.textContent = 'Unable to load the simulation. Check the server.';
            console.error(error);
            runButton.disabled = false;
        });
}

runButton.addEventListener('click', loadSimulation);
window.addEventListener('DOMContentLoaded', () => {
    loadSimulation();
});
