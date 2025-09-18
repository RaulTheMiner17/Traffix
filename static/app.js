document.addEventListener('DOMContentLoaded', function () {
    const loadingIndicator = document.getElementById('loading-indicator');
    const videoTab = document.getElementById('video-tab');
    const videoTabButton = document.getElementById('video-tab-button');
    const closeVideoTab = document.getElementById('close-video-tab');
    const videoGrid = document.getElementById('video-grid');

    // MAP INITIALIZE
    const map = L.map('map', {
        center: [19.0760, 72.8777], // Mumbai Coordinates
        zoom: 12,
        zoomControl: true,
        attributionControl: false
    });

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 20,
        className: 'map-tiles'
    }).addTo(map);

    let trafficLayer = L.layerGroup().addTo(map);
    let signalLayer = L.layerGroup().addTo(map); 

    // 
    //  SVG icon for a traffic light.
    const createSignalIcon = (color) => {
        const redFill = color === 'red' ? '#ef4444' : '#4a5568';
        const redOpacity = color === 'red' ? '1' : '0.3';

        return L.divIcon({
            html: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                        <path d="M8 2h8a2 2 0 0 1 2 2v16a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z" fill="#1a202c" stroke="#4a5568" stroke-width="1"/>
                        <circle cx="12" cy="7" r="2.5" fill="${redFill}" fill-opacity="${redOpacity}"/>
                        <circle cx="12" cy="12" r="2.5" fill="#4a5568" fill-opacity="0.3"/>
                        <circle cx="12" cy="17" r="2.5" fill="#4a5568" fill-opacity="0.3"/>
                    </svg>`,
            className: 'traffic-signal-icon',
            iconSize: [24, 24],
            iconAnchor: [12, 24]
        });
    };
    const trafficSignalIcon = createSignalIcon('red');


    // OSM DATA FETCHING include signals
    async function fetchAndProcessMapData() {
        loadingIndicator.style.display = 'block';
        trafficLayer.clearLayers();
        signalLayer.clearLayers(); 

        const bounds = map.getBounds();
        const bbox = `${bounds.getSouth()},${bounds.getWest()},${bounds.getNorth()},${bounds.getEast()}`;
        const zoom = map.getZoom();

        const highwayTypes = zoom < 13 ? "^(primary|trunk|motorway)$" : "^(primary|secondary|tertiary|trunk|motorway|primary_link|secondary_link|trunk_link)$";
        
        // query part only runs at higher zoom levels
        let signalQueryPart = "";
        if (zoom >= 14) { 
            signalQueryPart = `
                node["highway"="traffic_signals"](${bbox});
                node["crossing"="traffic_signals"](${bbox});
            `;
        }
        
        // query to include the signal 
        const query = `
            [out:json][timeout:25];
            (
                way["highway"~"${highwayTypes}"](${bbox});
                ${signalQueryPart}
            );
            out body;
            >;
            out skel qt;
        `;

        const url = `https://overpass-api.de/api/interpreter?data=${encodeURIComponent(query)}`;

        try {
            const response = await fetch(url);
            const data = await response.json();
            
            const nodes = {};
            data.elements.forEach(el => {
                if (el.type === 'node') nodes[el.id] = [el.lat, el.lon];
            });

            const roadData = [];
            data.elements.forEach(el => {
                // Process ways into roads 
                if (el.type === 'way' && el.nodes) {
                    const path = el.nodes.map(nodeId => nodes[nodeId]).filter(Boolean);
                    if (path.length > 1) roadData.push(path);
                }
                
                // Process nodes into traffic signals
                if (el.type === 'node' && (el.tags?.highway === 'traffic_signals' || el.tags?.crossing === 'traffic_signals')) {
                    L.marker([el.lat, el.lon], { icon: trafficSignalIcon })
                        .addTo(signalLayer)
                        .bindPopup('Traffic Signal');
                }
            });
            
            drawTrafficLines(roadData);
        } catch (error) {
            console.error("Error fetching OSM data:", error);
        } finally {
            loadingIndicator.style.display = 'none';
        }
    }

    //DRAW TRAFFIC LINES ON MAP 
    function drawTrafficLines(roadData) {
        const zoom = map.getZoom();
        const trafficStyles = ['#22c55e', '#f97316', '#ef4444'];
        const baseWeight = (zoom < 13) ? 2 : (zoom < 15) ? 3 : 4;

        roadData.forEach(path => {
            const style = {
                color: trafficStyles[Math.floor(Math.random() * trafficStyles.length)],
                weight: baseWeight + (Math.random() > 0.8 ? 2 : 0),
                opacity: 0.8
            };
            L.polyline(path, style).addTo(trafficLayer);
        });
    }

    // LIVE CLOCK 
    function updateTime() {
        const timeEl = document.getElementById('live-time');
        if (timeEl) timeEl.textContent = new Date().toLocaleTimeString('en-US');
    }

    // VIDEO PLAYER CREATION
    function createVideoPlayer(camera) {
        const videoItem = document.createElement('div');
        videoItem.className = 'video-item';
        const streamSrc = camera.streamUrl ? `/video_feed?url=${encodeURIComponent(camera.streamUrl)}` : '';
        videoItem.innerHTML = `
            <div class="video-container">
                ${streamSrc ? `<img src="${streamSrc}" alt="${camera.name}" class="video-stream" />` : `<div class="flex items-center justify-center h-full text-gray-400">No stream URL</div>`}
            </div>
            <div class="video-info">
                <div class="video-title">${camera.name}</div>
                <div class="video-location">${camera.location}</div>
                <div class="video-status">
                    <div class="status-dot"></div>
                    <span>${camera.status}</span>
                </div>
            </div>
        `;
        return videoItem;
    }

    // FETCH CAMERA DATA & INITIALIZE GRID 
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
            videoGrid.innerHTML = `<p class="text-red-500 p-4">Could not load cameras. Is the backend running?</p>`;
        }
    }

    // EVENT LISTENERS INITIALIZATION 
    const toggleVideoTab = () => videoTab.classList.toggle('active');
    videoTabButton.addEventListener('click', toggleVideoTab);
    closeVideoTab.addEventListener('click', toggleVideoTab);

    setInterval(updateTime, 1000);
    updateTime();
    
    fetchAndProcessMapData();
    map.on('moveend', fetchAndProcessMapData);

    initializeVideoGrid();
});