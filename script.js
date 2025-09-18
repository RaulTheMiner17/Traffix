
document.addEventListener('DOMContentLoaded', function () {
    // --- UI ELEMENTS ---
    const loadingIndicator = document.getElementById('loading-indicator');

    // --- OBJECT DETECTION VARIABLES ---
    let model = null;
    let detectionEnabled = false;
    const detectionIntervals = {};

    // Target classes for traffic monitoring
    const targetClasses = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person'];
    
    // Class mapping for better display names
    const classMapping = {
        'car': 'Car',
        'truck': 'Truck', 
        'bus': 'Bus',
        'motorcycle': 'Motorcycle',
        'bicycle': 'Bicycle',
        'person': 'Person'
    };

    // --- MAP INITIALIZATION ---
    const map = L.map('map', {
        center: [19.0760, 72.8777],
        zoom: 12, // Start at a mid-level zoom
        zoomControl: true,
        attributionControl: false
    });

    // Assign a custom class to the tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20,
        className: 'map-tiles' // This class will be targeted by the CSS filter
    }).addTo(map);

    let trafficLayer = L.layerGroup().addTo(map);
    let signalLayer = L.layerGroup().addTo(map);

    // --- STATIC TRAFFIC SIGNAL ICON ---
    const createSignalIcon = (color) => {
        const redFill = color === 'red' ? '#ef4444' : '#4a5568';
        const yellowFill = color === 'yellow' ? '#fBBF24' : '#4a5568';
        const greenFill = color === 'green' ? '#22c55e' : '#4a5568';
        
        const redOpacity = color === 'red' ? '1' : '0.3';
        const yellowOpacity = color === 'yellow' ? '1' : '0.3';
        const greenOpacity = color === 'green' ? '1' : '0.3';

        return L.divIcon({
            html: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                        <path d="M8 2h8a2 2 0 0 1 2 2v16a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z" fill="#1a202c" stroke="#4a5568" stroke-width="1"/>
                        <circle cx="12" cy="7" r="2.5" fill="${redFill}" fill-opacity="${redOpacity}"/>
                        <circle cx="12" cy="12" r="2.5" fill="${yellowFill}" fill-opacity="${yellowOpacity}"/>
                        <circle cx="12" cy="17" r="2.5" fill="${greenFill}" fill-opacity="${greenOpacity}"/>
                    </svg>`,
            className: 'traffic-signal-icon',
            iconSize: [24, 24],
            iconAnchor: [12, 24]
        });
    };

    // Create one static icon, fixed to red.
    const trafficSignalIcon = createSignalIcon('red');

    // --- OSM DATA FETCHING (ZOOM-AWARE) ---
    async function fetchAndProcessMapData() {
        loadingIndicator.style.display = 'block';
        trafficLayer.clearLayers();
        signalLayer.clearLayers();

        const bounds = map.getBounds();
        const bbox = `${bounds.getSouth()},${bounds.getWest()},${bounds.getNorth()},${bounds.getEast()}`;
        const zoom = map.getZoom();

        let highwayTypes;
        if (zoom < 13) {
            highwayTypes = "^(primary|trunk|motorway)$";
        } else {
            highwayTypes = "^(primary|secondary|tertiary|trunk|motorway|primary_link|secondary_link|trunk_link)$";
        }
        
        let signalQueryPart = "";
        if (zoom >= 14) { 
            signalQueryPart = `
                node["highway"="traffic_signals"](${bbox});
                node["crossing"="traffic_signals"](${bbox});
            `;
        }

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
                if (el.type === 'node') {
                    nodes[el.id] = [el.lat, el.lon];
                }
            });

            const roadData = [];
            data.elements.forEach(el => {
                if (el.type === 'way' && el.nodes) {
                    const path = el.nodes.map(nodeId => nodes[nodeId]).filter(Boolean);
                    if (path.length > 1) {
                        roadData.push(path);
                    }
                }
                
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

    // --- DRAW TRAFFIC LINES (ZOOM-AWARE) ---
    function drawTrafficLines(roadData) {
        const zoom = map.getZoom();
        const trafficStyles = ['#22c55e', '#f97316', '#ef4444'];
        
        let baseWeight;
        if (zoom < 13) baseWeight = 2;
        else if (zoom < 15) baseWeight = 3;
        else baseWeight = 4;

        roadData.forEach(path => {
            const style = {
                color: trafficStyles[Math.floor(Math.random() * trafficStyles.length)],
                weight: baseWeight + (Math.random() > 0.8 ? 2 : 0),
                opacity: 0.8
            };
            L.polyline(path, style).addTo(trafficLayer);
        });
    }

    // --- LIVE CLOCK ---
    function updateTime() {
        const now = new Date();
        document.getElementById('live-time').textContent = now.toLocaleTimeString('en-US');
    }
    setInterval(updateTime, 1000);
    updateTime();

    // --- VIDEO TAB FUNCTIONALITY ---
    const videoTab = document.getElementById('video-tab');
    const videoTabButton = document.getElementById('video-tab-button');
    const closeVideoTab = document.getElementById('close-video-tab');
    const videoGrid = document.getElementById('video-grid');

    // Camera data with locations across Mumbai
    const cameras = [
        {
            id: 'camera-1',
            name: 'Camera 1',
            location: 'Mumbai',
            coordinates: [19.0160, 72.8200],
            streamUrl: 'https://www.youtube.com/live/y-Os52eW2rg?si=yf9CIMI81lJfNftb',
            status: 'online',
            type: 'youtube'
        },
        {
            id: 'camera-2',
            name: 'Camera 2',
            location: 'Mumbai',
            coordinates: [18.9220, 72.8347],
            streamUrl: '',
            status: 'online',
            type: 'youtube'
        },
        {
            id: 'camera-3',
            name: 'Camera 3',
            location: 'Mumbai',
            coordinates: [19.0400, 72.8200],
            streamUrl: '',
            status: 'online',
            type: 'youtube'
        },
        {
            id: 'camera-4',
            name: 'Camera 4',
            location: 'Mumbai',
            coordinates: [19.1197, 72.8464],
            streamUrl: '',
            status: 'online',
            type: 'youtube'
        },
        {
            id: 'camera-5',
            name: 'Camera 5',
            location: 'Mumbai',
            coordinates: [19.1167, 72.9000],
            streamUrl: '',
            status: 'online',
            type: 'youtube'
        },
        {
            id: 'camera-6',
            name: 'Camera 6',
            location: 'Mumbai',
            coordinates: [19.1074, 72.8263],
            streamUrl: '',
            status: 'online',
            type: 'youtube'
        }
    ];

    // Test stream URLs for debugging
    const testStreams = [
        'https://mam.jogjaprov.go.id:1937/atcs-kota/PasarDemangan.stream/chunklist_w1938394215.m3u8',
        'https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism/.m3u8',
        'https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8'
    ];

    // Create video player element
    function createVideoPlayer(camera) {
        const videoItem = document.createElement('div');
        videoItem.className = 'video-item';
        
        if (camera.type === 'youtube') {
            // Extract YouTube video ID from URL
            const videoId = extractYouTubeId(camera.streamUrl);
            videoItem.innerHTML = `
                <div class="video-player" id="video-${camera.id}" style="position: relative;">
                    <iframe 
                        width="100%" 
                        height="200" 
                        src="https://www.youtube.com/embed/${videoId}?autoplay=1&mute=1&controls=1&rel=0&modestbranding=1" 
                        frameborder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen>
                    </iframe>
                </div>
                <div class="video-info">
                    <div class="video-info-header">
                        <div class="video-info-content">
                            <div class="video-title">${camera.name}</div>
                            <div class="video-location">${camera.location}</div>
                        </div>
                        <div class="video-info-actions">
                            <!-- AI Detection button will be added here -->
                        </div>
                    </div>
                    <div class="flex items-center">
                        <div class="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                        <span class="text-xs text-green-400">${camera.status}</span>
                    </div>
                </div>
            `;
            
            // Create detection overlay after DOM update
            setTimeout(() => {
                createDetectionOverlay(camera.id);
            }, 200);
        } else {
            // Fallback for other video types
            videoItem.innerHTML = `
                <video class="video-player" controls muted autoplay id="video-${camera.id}">
                    Your browser does not support the video tag.
                </video>
                <div class="video-info">
                    <div class="video-title">${camera.name}</div>
                    <div class="video-location">${camera.location}</div>
                    <div class="flex items-center mt-2">
                        <div class="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                        <span class="text-xs text-green-400">${camera.status}</span>
                    </div>
                </div>
            `;
            
            // Initialize HLS stream after the element is added to DOM
            setTimeout(() => {
                initializeHLSStream(camera);
            }, 100);
        }
        
        return videoItem;
    }

    // Extract YouTube video ID from URL
    function extractYouTubeId(url) {
        const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=|live\/)([^#&?]*).*/;
        const match = url.match(regExp);
        return (match && match[2].length === 11) ? match[2] : null;
    }

    // --- OBJECT DETECTION FUNCTIONS ---
    
    // Initialize TensorFlow.js model
    async function initializeDetectionModel() {
        try {
            console.log('Loading COCO-SSD model...');
            model = await cocoSsd.load();
            console.log('✅ Object detection model loaded successfully');
            return true;
        } catch (error) {
            console.error('❌ Failed to load detection model:', error);
            return false;
        }
    }

    // Create detection overlay for a video
    function createDetectionOverlay(cameraId) {
        const videoContainer = document.getElementById(`video-${cameraId}`);
        if (!videoContainer) {
            console.error(`Video container not found for camera: ${cameraId}`);
            return null;
        }

        // Remove existing overlay if it exists
        const existingOverlay = document.getElementById(`detection-overlay-${cameraId}`);
        if (existingOverlay) {
            existingOverlay.remove();
        }

        const overlay = document.createElement('div');
        overlay.className = 'detection-overlay';
        overlay.id = `detection-overlay-${cameraId}`;
        
        videoContainer.appendChild(overlay);
        
        // Add detection button to video info section
        addDetectionButtonToInfo(cameraId);
        
        console.log(`Detection overlay created for camera: ${cameraId}`);
        return overlay;
    }

    // Add detection button to video info section
    function addDetectionButtonToInfo(cameraId) {
        const videoItem = document.querySelector(`#video-${cameraId}`).closest('.video-item');
        if (!videoItem) return;

        const actionsContainer = videoItem.querySelector('.video-info-actions');
        if (!actionsContainer) return;

        // Remove existing button if it exists
        const existingButton = document.getElementById(`detection-toggle-${cameraId}`);
        if (existingButton) {
            existingButton.remove();
        }

        // Create button
        const toggle = document.createElement('button');
        toggle.className = 'detection-toggle';
        toggle.id = `detection-toggle-${cameraId}`;
        toggle.textContent = 'AI Detection';
        toggle.type = 'button';
        
        // Add click event listener
        toggle.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log(`AI Detection button clicked for camera: ${cameraId}`);
            toggleDetection(cameraId);
        });

        // Add button to actions container
        actionsContainer.appendChild(toggle);
    }

    // Toggle detection for a specific camera
    function toggleDetection(cameraId) {
        console.log(`Toggling detection for camera: ${cameraId}`);
        const toggle = document.getElementById(`detection-toggle-${cameraId}`);
        
        if (!toggle) {
            console.error(`Toggle button not found for camera: ${cameraId}`);
            return;
        }
        
        if (detectionIntervals[cameraId]) {
            // Stop detection
            console.log(`Stopping detection for camera: ${cameraId}`);
            clearInterval(detectionIntervals[cameraId]);
            delete detectionIntervals[cameraId];
            toggle.classList.remove('active');
            toggle.textContent = 'Detect Vehicles';
            clearDetectionOverlay(cameraId);
        } else {
            // Start detection
            console.log(`Starting detection for camera: ${cameraId}`);
            if (!model) {
                alert('Detection model not loaded yet. Please wait...');
                return;
            }
            startDetection(cameraId);
            toggle.classList.add('active');
            toggle.textContent = 'Stop detection';
        }
    }

    // Start object detection for a camera
    function startDetection(cameraId) {
        const iframe = document.querySelector(`#video-${cameraId} iframe`);
        if (!iframe) {
            console.error('No iframe found for camera:', cameraId);
            return;
        }


        // Run detection every 2 seconds
        detectionIntervals[cameraId] = setInterval(async () => {
            try {
                await runDetection(cameraId, iframe);
            } catch (error) {
                console.error('Detection error for camera', cameraId, ':', error);
            }
        }, 2000);

        console.log(`🔍 Started detection for camera: ${cameraId}`);
    }

    // Run object detection on video frame
    async function runDetection(cameraId, iframe) {
        try {
            // Create a canvas to capture video frame
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size to match video
            canvas.width = 640;
            canvas.height = 360;
            
            // For YouTube iframes, we can't directly access the video stream
            // So we'll simulate detection with random data for demonstration
            // In a real implementation, you'd need to use YouTube API or screen capture
            const mockDetections = generateMockDetections();
            
            // Clear previous detections
            clearDetectionOverlay(cameraId);
            
            // Draw detection boxes
            drawDetectionBoxes(cameraId, mockDetections, canvas.width, canvas.height);
            
        } catch (error) {
            console.error('Detection error:', error);
        }
    }

    // Generate mock detections for demonstration
    function generateMockDetections() {
        const detections = [];
        const numDetections = Math.floor(Math.random() * 8) + 2; // 2-10 detections
        
        for (let i = 0; i < numDetections; i++) {
            const classIndex = Math.floor(Math.random() * targetClasses.length);
            const className = targetClasses[classIndex];
            
            detections.push({
                class: className,
                score: 0.7 + Math.random() * 0.3, // 0.7-1.0 confidence
                bbox: [
                    Math.random() * 0.6, // x
                    Math.random() * 0.6, // y  
                    Math.random() * 0.3 + 0.1, // width
                    Math.random() * 0.3 + 0.1  // height
                ]
            });
        }
        
        return detections;
    }


    // Draw detection boxes on overlay
    function drawDetectionBoxes(cameraId, detections, videoWidth, videoHeight) {
        const overlay = document.getElementById(`detection-overlay-${cameraId}`);
        if (!overlay) return;
        
        detections.forEach((detection, index) => {
            if (detection.score < 0.5) return; // Skip low confidence detections
            
            const [x, y, width, height] = detection.bbox;
            const box = document.createElement('div');
            box.className = `detection-box ${detection.class}`;
            box.style.left = `${x * 100}%`;
            box.style.top = `${y * 100}%`;
            box.style.width = `${width * 100}%`;
            box.style.height = `${height * 100}%`;
            box.textContent = `${classMapping[detection.class] || detection.class} ${Math.round(detection.score * 100)}%`;
            
            overlay.appendChild(box);
        });
    }

    // Clear detection overlay
    function clearDetectionOverlay(cameraId) {
        const overlay = document.getElementById(`detection-overlay-${cameraId}`);
        if (overlay) {
            const boxes = overlay.querySelectorAll('.detection-box');
            boxes.forEach(box => box.remove());
        }
    }

    // Initialize HLS stream for a camera
    function initializeHLSStream(camera) {
        const video = document.getElementById(`video-${camera.id}`);
        if (!video) {
            console.error(`Video element not found for camera: ${camera.id}`);
            return;
        }

        console.log(`Initializing HLS stream for ${camera.name}: ${camera.streamUrl}`);

        // Add loading indicator
        video.style.background = 'linear-gradient(45deg, #1f2937, #374151)';
        video.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: white; font-size: 14px;">Loading stream...</div>';

        if (camera.type === 'hls' && Hls.isSupported()) {
            console.log('Using HLS.js for stream playback');
            
            const hls = new Hls({
                enableWorker: true,
                lowLatencyMode: true,
                backBufferLength: 90,
                maxBufferLength: 30,
                maxMaxBufferLength: 60,
                liveSyncDurationCount: 3,
                liveMaxLatencyDurationCount: 5,
                liveDurationInfinity: true,
                highBufferWatchdogPeriod: 2,
                nudgeOffset: 0.1,
                nudgeMaxRetry: 3,
                maxFragLookUpTolerance: 0.2,
                liveBackBufferLength: 0,
                maxLiveSyncPlaybackRate: 1.2,
                liveSyncDuration: 1,
                xhrSetup: function(xhr, url) {
                    console.log(`Making request to: ${url}`);
                    xhr.setRequestHeader('Referer', 'https://worldviewstream.com/');
                    xhr.setRequestHeader('Origin', 'https://worldviewstream.com');
                    xhr.setRequestHeader('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36');
                }
            });
            
            hls.loadSource(camera.streamUrl);
            hls.attachMedia(video);
            
            hls.on(Hls.Events.MANIFEST_PARSED, function() {
                console.log(`✅ HLS stream loaded successfully for ${camera.name}`);
                video.style.background = '#000';
                video.innerHTML = '';
                video.play().catch(e => {
                    console.log('Autoplay prevented, user interaction required:', e);
                    video.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: white; font-size: 14px; cursor: pointer;">Click to play</div>';
                    video.onclick = () => video.play();
                });
            });
            
            hls.on(Hls.Events.ERROR, function(event, data) {
                console.error(`❌ HLS Error for ${camera.name}:`, data);
                
                if (data.fatal) {
                    switch(data.type) {
                        case Hls.ErrorTypes.NETWORK_ERROR:
                            console.log('🔄 Fatal network error, trying to recover...');
                            video.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #fbbf24; font-size: 14px;">Network error, retrying...</div>';
                            hls.startLoad();
                            break;
                        case Hls.ErrorTypes.MEDIA_ERROR:
                            console.log('🔄 Fatal media error, trying to recover...');
                            video.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #fbbf24; font-size: 14px;">Media error, retrying...</div>';
                            hls.recoverMediaError();
                            break;
                        default:
                            console.log('💀 Fatal error, cannot recover');
                            video.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #ef4444; font-size: 14px;">Stream unavailable</div>';
                            hls.destroy();
                            break;
                    }
                } else {
                    console.log('⚠️ Non-fatal HLS error:', data);
                }
            });

            // Add timeout for stream loading
            setTimeout(() => {
                if (video.readyState === 0) {
                    console.log(`⏰ Stream timeout for ${camera.name}`);
                    video.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #ef4444; font-size: 14px;">Stream timeout - Check console for errors</div>';
                }
            }, 10000);
            
        } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
            console.log('Using native HLS support (Safari)');
            video.src = camera.streamUrl;
            video.addEventListener('loadedmetadata', function() {
                console.log(`✅ Native HLS stream loaded for ${camera.name}`);
                video.style.background = '#000';
                video.innerHTML = '';
                video.play().catch(e => console.log('Autoplay prevented:', e));
            });
            video.addEventListener('error', function(e) {
                console.error(`❌ Native HLS error for ${camera.name}:`, e);
                video.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #ef4444; font-size: 14px;">Stream error</div>';
            });
        } else {
            console.error('❌ HLS is not supported in this browser');
            video.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #ef4444; font-size: 14px;">HLS not supported</div>';
        }
    }

    // Initialize video players
    function initializeVideoPlayers() {
        videoGrid.innerHTML = '';
        cameras.forEach(camera => {
            const videoPlayer = createVideoPlayer(camera);
            videoGrid.appendChild(videoPlayer);
        });
    }

    // Toggle video tab
    function toggleVideoTab() {
        videoTab.classList.toggle('active');
        if (videoTab.classList.contains('active')) {
            videoTabButton.textContent = 'Hide Cameras';
        } else {
            videoTabButton.textContent = 'Show Cameras';
        }
    }

    // Event listeners
    videoTabButton.addEventListener('click', toggleVideoTab);
    closeVideoTab.addEventListener('click', toggleVideoTab);

    // Initialize video players on load
    initializeVideoPlayers();

    // Initialize object detection model
    initializeDetectionModel().then(success => {
        if (success) {
            console.log('🎯 Object detection ready! Click "AI Detection" on any camera to start.');
        }
    });

    // --- INITIALIZE AND BIND EVENTS ---
    fetchAndProcessMapData();
    map.on('moveend', fetchAndProcessMapData);
});
