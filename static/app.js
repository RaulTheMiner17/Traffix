document.addEventListener('DOMContentLoaded', function () {
    const loadingIndicator = document.getElementById('loading-indicator');
    const videoTab = document.getElementById('video-tab');
    const videoTabButton = document.getElementById('video-tab-button');
    const closeVideoTab = document.getElementById('close-video-tab');
    const videoGrid = document.getElementById('video-grid');

    // Dashboard elements
    const dashboardTab = document.getElementById('dashboard-tab');
    const dashboardTabButton = document.getElementById('dashboard-tab-button');
    const closeDashboardTab = document.getElementById('close-dashboard-tab');
    const cameraSelector = document.getElementById('camera-selector');

    // Dashboard Tab Toggle
    const toggleDashboardTab = () => dashboardTab.classList.toggle('active');
    dashboardTabButton.addEventListener('click', toggleDashboardTab);
    closeDashboardTab.addEventListener('click', toggleDashboardTab);

    // =====================================================
    //  REAL-TIME METRICS DASHBOARD
    // =====================================================
    const chartOpts = {
        responsive: true,
        maintainAspectRatio: true,
        animation: { duration: 400 },
        plugins: { legend: { labels: { color: '#b3b3b3', boxWidth: 12 } } },
        scales: {
            y: { beginAtZero: true, grid: { color: '#282828' }, ticks: { color: '#727272' } },
            x: { grid: { color: '#282828' }, ticks: { color: '#727272', maxRotation: 45, maxTicksLimit: 15 } }
        }
    };
    const timelineChartOpts = {
        responsive: true,
        maintainAspectRatio: true,
        animation: { duration: 400 },
        plugins: { legend: { labels: { color: '#b3b3b3', boxWidth: 12 } } },
        scales: {
            throughput: {
                type: 'linear',
                position: 'left',
                beginAtZero: true,
                suggestedMax: 120,
                grid: { color: '#282828' },
                ticks: { color: '#727272' },
                title: { display: true, text: 'vehicles/min', color: '#727272' }
            },
            queue: {
                type: 'linear',
                position: 'right',
                beginAtZero: true,
                suggestedMax: 20,
                grid: { drawOnChartArea: false },
                ticks: { color: '#727272' },
                title: { display: true, text: 'queue', color: '#727272' }
            },
            x: { grid: { color: '#282828' }, ticks: { color: '#727272', maxRotation: 45, maxTicksLimit: 15 } }
        }
    };
    const densityWaitChartOpts = {
        responsive: true,
        maintainAspectRatio: true,
        animation: { duration: 400 },
        plugins: { legend: { labels: { color: '#b3b3b3', boxWidth: 12 } } },
        scales: {
            density: {
                type: 'linear',
                position: 'left',
                beginAtZero: true,
                suggestedMax: 100,
                max: 100,
                grid: { color: '#282828' },
                ticks: { color: '#727272' },
                title: { display: true, text: 'density %', color: '#727272' }
            },
            wait: {
                type: 'linear',
                position: 'right',
                beginAtZero: true,
                suggestedMax: 120,
                max: 180,
                grid: { drawOnChartArea: false },
                ticks: { color: '#727272' },
                title: { display: true, text: 'wait seconds', color: '#727272' }
            },
            x: { grid: { color: '#282828' }, ticks: { color: '#727272', maxRotation: 45, maxTicksLimit: 15 } }
        }
    };

    let timelineChart = null;
    let densityWaitChart = null;

    function ensureCharts() {
        if (timelineChart) return;
        const ctx1 = document.getElementById('timelineChart');
        const ctx2 = document.getElementById('densityWaitChart');
        if (!ctx1 || !ctx2) return;

        timelineChart = new Chart(ctx1, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Throughput vpm (RL)', yAxisID: 'throughput', data: [], borderColor: '#1DB954', backgroundColor: 'rgba(29,185,84,0.08)', fill: false, tension: 0.25 },
                    { label: 'Avg Queue (RL)', yAxisID: 'queue', data: [], borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,0.08)', fill: false, tension: 0.25 },
                    { label: 'Throughput vpm (Static)', yAxisID: 'throughput', data: [], borderColor: '#1DB954', borderDash: [5, 5], backgroundColor: 'rgba(0,0,0,0)', fill: false, tension: 0.25 },
                    { label: 'Avg Queue (Static)', yAxisID: 'queue', data: [], borderColor: '#f97316', borderDash: [5, 5], backgroundColor: 'rgba(0,0,0,0)', fill: false, tension: 0.25 }
                ]
            },
            options: timelineChartOpts
        });

        densityWaitChart = new Chart(ctx2, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Density (%) (RL)', yAxisID: 'density', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.08)', fill: false, tension: 0.25 },
                    { label: 'Wait Time (s) (RL)', yAxisID: 'wait', data: [], borderColor: '#fbbf24', backgroundColor: 'rgba(251,191,36,0.08)', fill: false, tension: 0.25 },
                    { label: 'Density (%) (Static)', yAxisID: 'density', data: [], borderColor: '#ef4444', borderDash: [5, 5], backgroundColor: 'rgba(0,0,0,0)', fill: false, tension: 0.25 },
                    { label: 'Wait Time (s) (Static)', yAxisID: 'wait', data: [], borderColor: '#fbbf24', borderDash: [5, 5], backgroundColor: 'rgba(0,0,0,0)', fill: false, tension: 0.25 }
                ]
            },
            options: densityWaitChartOpts
        });
    }

    function formatUptime(s) {
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        const sec = Math.floor(s % 60);
        if (h > 0) return `${h}h ${m}m`;
        if (m > 0) return `${m}m ${sec}s`;
        return `${sec}s`;
    }

    function alignSeries(values, targetLength) {
        const arr = Array.isArray(values) ? values.slice(-targetLength) : [];
        while (arr.length < targetLength) arr.unshift(null);
        return arr;
    }

    function setImprovementText(metricId, text) {
        const valueEl = document.getElementById(metricId);
        const subtextEl = valueEl?.closest('.stat-card')?.querySelector('.stat-subtext');
        if (subtextEl) subtextEl.textContent = text;
    }

    function lowerIsBetterText(rlValue, staticValue) {
        if (!Number.isFinite(rlValue) || !Number.isFinite(staticValue) || staticValue <= 0) return '';
        const improvement = Math.max(0, ((staticValue - rlValue) / staticValue) * 100);
        return `${improvement.toFixed(1)}% lower than Static`;
    }

    function higherIsBetterText(rlValue, staticValue) {
        if (!Number.isFinite(rlValue) || !Number.isFinite(staticValue) || staticValue <= 0) return '';
        const improvement = Math.max(0, ((rlValue - staticValue) / staticValue) * 100);
        return `${improvement.toFixed(1)}% higher than Static`;
    }

    function updateDashboard() {
        fetch('/api/metrics')
            .then(r => r.json())
            .then(d => {
                const el = id => document.getElementById(id);
                // Model/controller info
                const nameEl = el('m-model-name');
                if (nameEl) {
                    nameEl.textContent = `${d.rl.name} / ${d.static.name}`;
                    nameEl.style.color = d.rl.active ? '#1DB954' : '#f97316';
                }
                const typeEl = el('m-model-type');
                if (typeEl) typeEl.textContent = `${d.rl.type} / ${d.static.type}`;
                const cardEl = document.getElementById('model-card');
                if (cardEl) cardEl.style.borderColor = d.rl.active ? '#1DB954' : '#f97316';

                // Counters: uptime same for both, switches shown as rl/static
                if (el('m-uptime')) el('m-uptime').textContent = formatUptime(d.uptime_seconds);
                if (el('m-switches')) el('m-switches').textContent = `${d.rl.phase_switches} / ${d.static.phase_switches}`;
                if (el('m-detections')) el('m-detections').textContent = d.total_vehicles_now;

                // Primary metrics: compare RL vs static
                const rt_rl = d.rl.realtime;
                const rt_static = d.static.realtime;
                if (el('m-wait')) el('m-wait').textContent = `${rt_rl.est_wait_time_s.toFixed(1)}s / ${rt_static.est_wait_time_s.toFixed(1)}s`;
                if (el('m-queue')) el('m-queue').textContent = `${rt_rl.avg_queue_length.toFixed(1)} / ${rt_static.avg_queue_length.toFixed(1)}`;
                const rlThroughput = rt_rl.throughput_vpm ?? (rt_rl.avg_speed_factor * d.total_vehicles_now * 3);
                const staticThroughput = rt_static.throughput_vpm ?? (rt_static.avg_speed_factor * d.total_vehicles_now * 3);
                if (el('m-throughput')) el('m-throughput').textContent = `${rlThroughput.toFixed(1)} / ${staticThroughput.toFixed(1)}`;
                setImprovementText('m-wait', lowerIsBetterText(rt_rl.est_wait_time_s, rt_static.est_wait_time_s));
                setImprovementText('m-queue', lowerIsBetterText(rt_rl.avg_queue_length, rt_static.avg_queue_length));
                setImprovementText('m-throughput', higherIsBetterText(rlThroughput, staticThroughput));

                // Secondary metrics
                if (el('m-density')) el('m-density').textContent = `${rt_rl.avg_density_pct.toFixed(1)}% / ${rt_static.avg_density_pct.toFixed(1)}%`;
                if (el('m-speed')) el('m-speed').textContent = `${rt_rl.avg_speed_factor.toFixed(2)} / ${rt_static.avg_speed_factor.toFixed(2)}`;
                if (el('m-emissions')) el('m-emissions').textContent = `${rt_rl.idle_emissions_factor.toFixed(3)} / ${rt_static.idle_emissions_factor.toFixed(3)}`;
                setImprovementText('m-speed', higherIsBetterText(rt_rl.avg_speed_factor, rt_static.avg_speed_factor));
                setImprovementText('m-emissions', lowerIsBetterText(rt_rl.idle_emissions_factor, rt_static.idle_emissions_factor));

                // Per-direction counts (same for both systems)
                ['N', 'S', 'E', 'W'].forEach(dir => {
                    const dirEl = el('m-dir-' + dir);
                    if (dirEl) dirEl.textContent = (d.rl.per_direction[dir] || 0).toLocaleString();
                });

                // Update charts with RL vs static series
                ensureCharts();
                const tl_rl = d.rl.timeline;
                const tl_static = d.static.timeline;
                if (timelineChart && tl_rl.labels.length > 0) {
                    const labels = tl_rl.labels;
                    const n = labels.length;
                    timelineChart.data.labels = labels;
                    timelineChart.data.datasets[0].data = alignSeries(tl_rl.throughput || tl_rl.vehicles, n);
                    timelineChart.data.datasets[1].data = alignSeries(tl_rl.queue, n);
                    timelineChart.data.datasets[2].data = alignSeries(tl_static.throughput || tl_static.vehicles, n);
                    timelineChart.data.datasets[3].data = alignSeries(tl_static.queue, n);
                    timelineChart.update('none');
                }
                if (densityWaitChart && tl_rl.labels.length > 0) {
                    const labels = tl_rl.labels;
                    const n = labels.length;
                    densityWaitChart.data.labels = labels;
                    densityWaitChart.data.datasets[0].data = alignSeries(tl_rl.density, n);
                    densityWaitChart.data.datasets[1].data = alignSeries(tl_rl.wait_time, n);
                    densityWaitChart.data.datasets[2].data = alignSeries(tl_static.density, n);
                    densityWaitChart.data.datasets[3].data = alignSeries(tl_static.wait_time, n);
                    densityWaitChart.update('none');
                }
            })
            .catch(e => console.error('Dashboard update error:', e));
    }

    // Poll every 3 seconds
    setInterval(updateDashboard, 3000);
    setTimeout(updateDashboard, 1500);

    // MAP INITIALIZE
    const JUHU_CENTER = [19.10711753017639, 72.82994390098155];

    const map = L.map('map', {
        center: JUHU_CENTER,
        zoom: 17,
        zoomControl: true,
        attributionControl: false
    });

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 20,
        className: 'map-tiles'
    }).addTo(map);

    // =====================================================
    //  TRAFFIC SIGNAL MARKERS — from OpenStreetMap (Overpass)
    // =====================================================
    let signalLayer = L.layerGroup().addTo(map);
    let lastSignalBounds = null;
    let signalFetchController = null;

    function debounce(fn, ms) {
        let timer;
        return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
    }

    const trafficSignalIcon = L.divIcon({
        html: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20">
                    <rect x="7" y="1" width="10" height="22" rx="2" fill="#1a202c" stroke="#4a5568" stroke-width="1"/>
                    <circle cx="12" cy="7" r="2.2" fill="#ef4444"/>
                    <circle cx="12" cy="12" r="2.2" fill="#4a5568" fill-opacity="0.3"/>
                    <circle cx="12" cy="17" r="2.2" fill="#4a5568" fill-opacity="0.3"/>
                </svg>`,
        className: 'traffic-signal-icon',
        iconSize: [20, 20],
        iconAnchor: [10, 20]
    });

    function expandBounds(b, f) {
        const dy = (b.getNorth() - b.getSouth()) * f;
        const dx = (b.getEast() - b.getWest()) * f;
        return L.latLngBounds([b.getSouth() - dy, b.getWest() - dx], [b.getNorth() + dy, b.getEast() + dx]);
    }

    async function fetchSignalMarkers() {
        const zoom = map.getZoom();
        if (zoom < 14) { signalLayer.clearLayers(); return; }

        const currentBounds = map.getBounds();
        if (lastSignalBounds && lastSignalBounds.contains(currentBounds)) return;

        if (signalFetchController) signalFetchController.abort();
        signalFetchController = new AbortController();

        const fetchBounds = expandBounds(currentBounds, 0.4);
        const bbox = `${fetchBounds.getSouth()},${fetchBounds.getWest()},${fetchBounds.getNorth()},${fetchBounds.getEast()}`;

        const query = `[out:json][timeout:15];(node["highway"="traffic_signals"](${bbox});node["crossing"="traffic_signals"](${bbox}););out body;`;

        try {
            const resp = await fetch(
                `https://overpass-api.de/api/interpreter?data=${encodeURIComponent(query)}`,
                { signal: signalFetchController.signal }
            );
            const data = await resp.json();
            const newSignals = L.layerGroup();
            data.elements.forEach(el => {
                if (el.type === 'node') {
                    L.marker([el.lat, el.lon], { icon: trafficSignalIcon }).addTo(newSignals);
                }
            });
            map.removeLayer(signalLayer);
            signalLayer = newSignals.addTo(map);
            lastSignalBounds = fetchBounds;
        } catch (e) {
            if (e.name !== 'AbortError') console.error('Signal fetch error:', e);
        }
    }

    function updateTime() {
        const timeEl = document.getElementById('live-time');
        if (timeEl) timeEl.textContent = new Date().toLocaleTimeString('en-US');
    }

    function createVideoPlayer(camera) {
        const videoItem = document.createElement('div');
        videoItem.className = 'video-item';
        // Add random query param to force image refresh when switching sources
        const streamSrc = `/video_feed?id=${camera.id}&t=${Date.now()}`;

        videoItem.innerHTML = `
            <div class="video-container">
                <img src="${streamSrc}" alt="${camera.name}" class="video-stream" onerror="this.onerror=null;this.src='placeholder.jpg';"/>
            </div>
            <div class="pressure-bar-container" data-camera-id="${camera.id}">
                <div class="pressure-bar-label">
                    <span>Vehicle Pressure</span>
                    <span class="pressure-bar-count" data-count>0</span>
                </div>
                <div class="pressure-bar-track">
                    <div class="pressure-bar-fill low" data-fill style="width: 0%"></div>
                </div>
            </div>
            <div class="video-info">
                <div class="signal-row">
                    <div class="video-title">${camera.name} <span class="pressure-signal-badge red" data-signal-badge>RED</span></div>
                    <div class="signal-timer" data-signal-timer>0s</div>
                </div>
                <div class="turn-indicators" data-turn-indicators>
                    <span class="turn-tag left" data-turn-left>Left</span>
                    <span class="turn-tag straight" data-turn-straight>Straight</span>
                    <span class="turn-tag right" data-turn-right>Right</span>
                </div>
                <div class="video-status">
                    <div class="status-dot"></div>
                    <span>Online</span>
                </div>
            </div>
        `;
        return videoItem;
    }

    async function initializeVideoGrid() {
        if (!videoGrid) return;
        videoGrid.innerHTML = '<p class="text-gray-400 p-4">Loading cameras...</p>';
        try {
            const response = await fetch('/api/cameras');
            if (!response.ok) throw new Error(`API request failed: ${response.status}`);
            const cameras = await response.json();
            videoGrid.innerHTML = '';
            cameras.forEach(camera => {
                videoGrid.appendChild(createVideoPlayer(camera));
            });
        } catch (error) {
            console.error("Failed to fetch camera data:", error);
            videoGrid.innerHTML = `<p class="text-red-500 p-4">Could not load cameras.</p>`;
        }
    }

    const toggleVideoTab = () => videoTab.classList.toggle('active');
    videoTabButton.addEventListener('click', toggleVideoTab);
    closeVideoTab.addEventListener('click', toggleVideoTab);

    setInterval(updateTime, 1000);
    updateTime();

    // Traffic tiles load automatically via Leaflet — just fetch signal markers
    fetchSignalMarkers();
    map.on('moveend', debounce(() => fetchSignalMarkers(), 400));
    if (loadingIndicator) loadingIndicator.style.display = 'none';

    initializeVideoGrid();

    // --- Pressure Bar Live Polling ---
    const MAX_CAPACITY = 20;
    function updatePressureBars() {
        fetch('/api/signal-status')
            .then(res => res.json())
            .then(data => {
                const liveData = data.live_data || {};
                // use RL controller state for live video overlays
                const phase = data.rl ? data.rl.current_phase_code : data.current_phase_code;
                const isYellowFlag = data.rl ? data.rl.is_yellow : data.is_yellow;
                const currentSubPhase = data.rl ? data.rl.sub_phase : data.sub_phase;
                const currentTimer = Math.floor(data.rl ? (data.rl.state_timer || 0) : (data.state_timer || 0));

                document.querySelectorAll('.pressure-bar-container').forEach(container => {
                    const camId = container.dataset.cameraId;
                    const camData = liveData[camId];
                    if (!camData) return;

                    const count = camData.count || 0;
                    const pct = Math.min((count / MAX_CAPACITY) * 100, 100);

                    const countEl = container.querySelector('[data-count]');
                    const fillEl = container.querySelector('[data-fill]');
                    if (countEl) countEl.textContent = count;
                    if (fillEl) {
                        fillEl.style.width = pct + '%';
                        fillEl.className = 'pressure-bar-fill';
                        if (pct < 30) fillEl.classList.add('low');
                        else if (pct < 60) fillEl.classList.add('medium');
                        else if (pct < 85) fillEl.classList.add('high');
                        else fillEl.classList.add('critical');
                    }

                    // Update signal badge, timer, and turn indicators
                    const videoItem = container.parentElement;
                    const badge = videoItem.querySelector('[data-signal-badge]');
                    const timerEl = videoItem.querySelector('[data-signal-timer]');
                    const dir = camId.replace('camera-', '');
                    const isGreenDir = (phase === 0 && (dir === 'N' || dir === 'S')) ||
                        (phase === 1 && (dir === 'E' || dir === 'W'));
                    const isYellow = isYellowFlag;
                    const subPhase = currentSubPhase;
                    const timer = currentTimer;

                    if (badge) {
                        if (isYellow && isGreenDir) {
                            badge.textContent = 'YELLOW';
                            badge.className = 'pressure-signal-badge yellow';
                        } else if (isGreenDir && subPhase === 'left_turn') {
                            badge.textContent = 'LEFT TURN';
                            badge.className = 'pressure-signal-badge left-turn';
                        } else if (isGreenDir && subPhase === 'right_turn') {
                            badge.textContent = 'RIGHT TURN';
                            badge.className = 'pressure-signal-badge right-turn';
                        } else if (isGreenDir) {
                            badge.textContent = 'GREEN';
                            badge.className = 'pressure-signal-badge green';
                        } else {
                            badge.textContent = 'RED';
                            badge.className = 'pressure-signal-badge red';
                        }
                    }

                    // Timer
                    if (timerEl) {
                        if (isGreenDir) {
                            timerEl.textContent = timer + 's';
                            timerEl.className = 'signal-timer';
                            if (isYellow) timerEl.classList.add('yellow');
                            else if (subPhase === 'left_turn') timerEl.classList.add('left-turn');
                            else if (subPhase === 'right_turn') timerEl.classList.add('right-turn');
                            else timerEl.classList.add('green');
                        } else {
                            timerEl.textContent = timer + 's';
                            timerEl.className = 'signal-timer red';
                        }
                    }

                    // Turn indicators
                    const turnLeft = videoItem.querySelector('[data-turn-left]');
                    const turnStraight = videoItem.querySelector('[data-turn-straight]');
                    const turnRight = videoItem.querySelector('[data-turn-right]');
                    if (turnLeft && turnStraight && turnRight) {
                        turnLeft.classList.remove('active');
                        turnStraight.classList.remove('active');
                        turnRight.classList.remove('active');
                        if (isGreenDir && !isYellow) {
                            if (subPhase === 'left_turn') turnLeft.classList.add('active');
                            else if (subPhase === 'straight') turnStraight.classList.add('active');
                            else if (subPhase === 'right_turn') turnRight.classList.add('active');
                        }
                    }
                });
            })
            .catch(err => console.error('Pressure bar update error:', err));
    }
    setInterval(updatePressureBars, 1000);
    setTimeout(updatePressureBars, 2000);

    // =====================================================
    //  TRAFFIC OVERLAY — on main map (8 road segments)
    // =====================================================
    // 8 segments: incoming + outgoing for each of 4 directions
    // Coordinates from OpenStreetMap inspection. India = left-hand drive.
    // N-S road: Guru Nanak Rd — center lon ~72.82992, lanes offset ±0.00005
    // E-W road: Indravadan Oza Rd — center lat ~19.10715, lanes offset ±0.00004
    const ROAD_SEGMENTS = [
        // NORTH ARM — Guru Nanak Rd
        { name: 'North Incoming', direction: 'N', flow: 'in', coords: [[19.10793, 72.82998], [19.10760, 72.82998], [19.10740, 72.82998], [19.10725, 72.82998]] },
        { name: 'North Outgoing', direction: 'N', flow: 'out', coords: [[19.10725, 72.82987], [19.10740, 72.82987], [19.10760, 72.82987], [19.10793, 72.82987]] },
        // SOUTH ARM — Guru Nanak Rd
        { name: 'South Incoming', direction: 'S', flow: 'in', coords: [[19.10642, 72.82986], [19.10660, 72.82986], [19.10685, 72.82987], [19.10705, 72.82987]] },
        { name: 'South Outgoing', direction: 'S', flow: 'out', coords: [[19.10705, 72.829985], [19.10685, 72.829985], [19.10660, 72.829980], [19.10642, 72.829975]] },
        // EAST ARM — Indravadan Oza Rd
        { name: 'East Incoming', direction: 'E', flow: 'in', coords: [[19.107094, 72.83085], [19.107094, 72.83055], [19.107090, 72.83025], [19.107090, 72.83002]] },
        { name: 'East Outgoing', direction: 'E', flow: 'out', coords: [[19.10717, 72.83002], [19.10717, 72.83025], [19.10717, 72.83055], [19.10717, 72.83085]] },
        // WEST ARM — Indravadan Oza Rd
        { name: 'West Incoming', direction: 'W', flow: 'in', coords: [[19.10717, 72.82922], [19.10717, 72.82945], [19.10717, 72.82965], [19.10717, 72.82982]] },
        { name: 'West Outgoing', direction: 'W', flow: 'out', coords: [[19.10710, 72.82982], [19.10710, 72.82965], [19.10710, 72.82945], [19.10710, 72.82922]] },
    ];

    let trafficLines = [];
    let currentMapMode = 'ai';

    function getDensityForDirection(metricsData, direction, flow, mode) {
        if (!metricsData) return 50;
        let density = mode === 'ai'
            ? (metricsData.rl?.realtime?.avg_density_pct ?? 50)
            : (metricsData.static?.realtime?.avg_density_pct ?? 50);
        const perDir = metricsData.rl?.per_direction || {};
        const total = Object.values(perDir).reduce((a, b) => a + b, 0) || 1;
        const ratio = (perDir[direction] || 0) / total;
        let dirDensity = density * 0.6 + (ratio * 4 * density * 0.4);
        // Outgoing lanes have lower density (traffic flowing out)
        if (flow === 'out') dirDensity *= 0.5;
        return mode === 'static' ? Math.min(dirDensity * 1.38, 100) : dirDensity;
    }

    function densityToColor(d) {
        if (d < 35) return '#22c55e';
        if (d < 60) return '#f97316';
        return '#ef4444';
    }

    function drawTrafficOverlay(metricsData) {
        trafficLines.forEach(l => map.removeLayer(l));
        trafficLines = [];
        ROAD_SEGMENTS.forEach(seg => {
            const density = getDensityForDirection(metricsData, seg.direction, seg.flow, currentMapMode);
            const color = densityToColor(density);
            const weight = density < 35 ? 3 : density < 60 ? 4 : 5;
            const shadow = L.polyline(seg.coords, { color: 'rgba(0,0,0,0.45)', weight: weight + 2, lineCap: 'round', lineJoin: 'round', interactive: false }).addTo(map);
            const line = L.polyline(seg.coords, { color, weight, lineCap: 'round', lineJoin: 'round', opacity: 0.93 }).addTo(map);
            line.bindTooltip(`<b>${seg.name}</b><br>Density: ${density.toFixed(1)}%<br>${currentMapMode === 'ai' ? '\ud83e\udde0 AI (PPO)' : '\u23f1 Fixed Timer'}`, { sticky: true });
            trafficLines.push(shadow, line);
        });
    }

    function refreshTrafficOverlay() {
        fetch('/api/metrics')
            .then(r => r.json())
            .then(data => drawTrafficOverlay(data))
            .catch(() => drawTrafficOverlay(null));
    }

    // Toggle buttons
    const overlayToggleAi = document.getElementById('overlay-toggle-ai');
    const overlayToggleStatic = document.getElementById('overlay-toggle-static');

    function setOverlayMode(mode) {
        currentMapMode = mode;
        overlayToggleAi && overlayToggleAi.classList.toggle('active', mode === 'ai');
        overlayToggleStatic && overlayToggleStatic.classList.toggle('active', mode === 'static');
        refreshTrafficOverlay();
    }

    if (overlayToggleAi) overlayToggleAi.addEventListener('click', () => setOverlayMode('ai'));
    if (overlayToggleStatic) overlayToggleStatic.addEventListener('click', () => setOverlayMode('static'));

    refreshTrafficOverlay();
    setInterval(refreshTrafficOverlay, 4000);
});
