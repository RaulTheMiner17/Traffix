import cv2
import threading
import time
import numpy as np
import os
import warnings
import collections
import subprocess
import sys
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import torch

# --- RL & GYM IMPORTS ---
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- SUMO IMPORTS (for static controller baseline) ---
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
    try:
        import traci
        SUMO_AVAILABLE = True
    except ImportError:
        SUMO_AVAILABLE = False
else:
    SUMO_AVAILABLE = False
    print("⚠️  SUMO_HOME not set. Static baseline will be disabled.")

warnings.filterwarnings("ignore")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == 'cuda':
    print(f"🚀 Success! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CUDA not found. Falling back to CPU. (Performance may be slower)")

model = YOLO('yolo11n.pt').to(device)
CLASS_IDS_TO_DETECT = [2, 3, 5, 7]
CLASS_NAMES = model.names
VEHICLE_WEIGHTS = {
    2: 1.0,  # Car
    3: 0.3,  # Motorcycle
    5: 3.0,  # Bus
    7: 2.5   # Truck
}

# Define the invisible lane divider lines for each camera: ((top_x, top_y), (bottom_x, bottom_y))
# You can rotate or shift the lanes by changing the x coordinates here!
LANE_DIVIDERS = {
    'camera-E': ((250, 0), (350, 360)),
    'camera-W': ((270, 0), (360, 360)),
    'camera-N': ((320, 0), (320, 360)),
    'camera-S': ((320, 0), (320, 360)),
}
yolo_lock = threading.Lock()

CAMERAS = [
    {
        'id': 'camera-N',
        'name': 'North',
        'direction': 'N',
        'localPath': os.path.join(BASE_DIR, 'videos', 'north_traffic.mp4'),
    },
    {
        'id': 'camera-S',
        'name': 'South',
        'direction': 'S',
        'localPath': os.path.join(BASE_DIR, 'videos', 'south_traffic.mp4'),
    },
    {
        'id': 'camera-E',
        'name': 'East',
        'direction': 'E',
        'localPath': os.path.join(BASE_DIR, 'videos', 'east_traffic.mp4'),
    },
    {
        'id': 'camera-W',
        'name': 'West',
        'direction': 'W',
        'localPath': os.path.join(BASE_DIR, 'videos', 'west_traffic.mp4'),
    },
]

output_frames = {}
vehicle_data = {} 
lock = threading.Lock()

for cam in CAMERAS:
    vehicle_data[cam['id']] = {'count': 0, 'queue': 0, 'density': 0.0, 'speed': 1.0}

class TrafficLightSystem:
    def __init__(self):
        self.current_phase = 0        
        self.sub_phase = 'left_turn'  
        self.last_switch_time = time.time()
        self.state_start_time = time.time()  
        self.min_green_time = 15
        self.max_green_time = 60
        self.yellow_duration = 4
        self.left_turn_duration = 12
        self.straight_duration = 20
        self.right_turn_duration = 8
        self.is_yellow = False
        self.yellow_start_time = 0
        self.rl_model = None
        self.vec_env = None
        self.load_brain()

    def load_brain(self):
        print("🧠 Attempting to load AI Brain...")
        if not os.path.exists("vec_normalize.pkl") or not os.path.exists("ppo_traffic_final.zip"):
            print("⚠️  FILES MISSING. Using Logic Fallback.")
            return

        try:
            dummy_env = DummyVecEnv([lambda: gym.make("CartPole-v1")]) 
            dummy_env.observation_space = spaces.Box(low=0, high=np.inf, shape=(15,), dtype=np.float32)
            self.vec_env = VecNormalize.load("vec_normalize.pkl", dummy_env)
            self.vec_env.training = False
            self.vec_env.norm_reward = False
            self.rl_model = PPO.load("ppo_traffic_final", env=self.vec_env)
            print("✅ Traffic AI: Model LOADED SUCCESSFULLY!")
        except Exception as e:
            print(f"❌ Traffic AI Load Error: {e}")
            self.rl_model = None

    def get_observation_vector(self):
        order = ['camera-N', 'camera-S', 'camera-E', 'camera-W']
        queues, densities, speeds = [], [], []
        with lock:
            for cid in order:
                data = vehicle_data[cid]
                queues.append(data['queue'])
                densities.append(data['density'])
                speeds.append(data['speed'])
        
        phase_oh = [1.0, 0.0] if self.current_phase == 0 else [0.0, 1.0]
        time_diff = time.time() - self.last_switch_time
        time_norm = min(time_diff / self.max_green_time, 1.0)
        
        obs = np.concatenate([queues, densities, speeds, phase_oh, [time_norm]])
        return obs.astype(np.float32)

    def decide(self):
        while True:
            time.sleep(0.01) 
            obs = self.get_observation_vector()
            elapsed = time.time() - self.last_switch_time
            action = self.current_phase 
            
            if self.rl_model:
                try:
                    action, _ = self.rl_model.predict(obs, deterministic=True)
                except: pass
            else:
                ns_density = obs[4] + obs[5]
                ew_density = obs[6] + obs[7]
                if self.current_phase == 0 and ew_density > ns_density: action = 1
                elif self.current_phase == 1 and ns_density > ew_density: action = 0

            # Only enforce min_green_time check logic inside the straight phase now

            sub_elapsed = time.time() - self.state_start_time

            if not self.is_yellow:
                if self.sub_phase == 'left_turn' and sub_elapsed >= self.left_turn_duration:
                    self.sub_phase = 'straight'
                    self.state_start_time = time.time()
                    print(f"🚦 STRAIGHT phase (AI controlled)")
                
                elif self.sub_phase == 'straight':
                    time_in_straight = sub_elapsed
                    total_green = time.time() - self.last_switch_time
                    
                    rl_wants_switch = (action != self.current_phase)
                    met_min_green = (total_green >= self.min_green_time)
                    hit_max_green = (total_green >= (self.max_green_time - self.right_turn_duration))
                    min_straight_met = (time_in_straight >= 5)

                    if (rl_wants_switch and met_min_green and min_straight_met) or hit_max_green:
                        self.sub_phase = 'right_turn'
                        self.state_start_time = time.time()
                        trigger_reason = 'Max Time limit' if hit_max_green else 'AI Traffic Optimization'
                        print(f"🚦 RIGHT TURN phase (Triggered by {trigger_reason})")
                        
                elif self.sub_phase == 'right_turn' and sub_elapsed >= self.right_turn_duration:
                    print(f"🚦 YELLOW: Preparing to switch")
                    self.is_yellow = True
                    self.yellow_start_time = time.time()
                    self.state_start_time = time.time()
            
            if self.is_yellow and (time.time() - self.yellow_start_time) >= self.yellow_duration:
                new_phase = 1 if self.current_phase == 0 else 0
                print(f"🚦 SWITCHING: {'EW Green' if new_phase==1 else 'NS Green'}")
                self.current_phase = new_phase
                self.last_switch_time = time.time()
                self.is_yellow = False
                self.sub_phase = 'left_turn'
                self.state_start_time = time.time()
                if hasattr(self, '_metrics_cb') and self._metrics_cb:
                    self._metrics_cb()

class StaticTrafficSystem:
    def __init__(self):
        self.current_phase = 0
        self.sub_phase = 'left_turn'
        self.last_switch_time = time.time()
        self.state_start_time = time.time()
        self.min_green_time = 15
        self.max_green_time = 60
        self.yellow_duration = 4
        self.left_turn_duration = 12
        self.straight_duration = 20
        self.right_turn_duration = 8
        self.is_yellow = False
        self.yellow_start_time = 0

    def decide(self):
        while True:
            time.sleep(0.2)
            elapsed = time.time() - self.last_switch_time
            action = self.current_phase
            
            target_green_time = 60 if self.current_phase == 0 else 120
            
            if elapsed >= target_green_time and elapsed >= self.min_green_time:
                action = 1 - self.current_phase

            sub_elapsed = time.time() - self.state_start_time
            if not self.is_yellow:
                if self.sub_phase == 'left_turn' and sub_elapsed >= self.left_turn_duration:
                    self.sub_phase = 'straight'
                    self.state_start_time = time.time()
                elif self.sub_phase == 'straight' and sub_elapsed >= self.straight_duration:
                    self.sub_phase = 'right_turn'
                    self.state_start_time = time.time()
                elif self.sub_phase == 'right_turn' and sub_elapsed >= self.right_turn_duration:
                    self.is_yellow = True
                    self.yellow_start_time = time.time()
                    self.state_start_time = time.time()

            if self.is_yellow and (time.time() - self.yellow_start_time) >= self.yellow_duration:
                new_phase = 1 if self.current_phase == 0 else 0
                self.current_phase = new_phase
                self.last_switch_time = time.time()
                self.is_yellow = False
                self.sub_phase = 'left_turn'
                self.state_start_time = time.time()
                if hasattr(self, '_metrics_cb') and self._metrics_cb:
                    self._metrics_cb()

traffic_brain = TrafficLightSystem()
threading.Thread(target=traffic_brain.decide, daemon=True).start()

try:
    from static_sumo_controller import StaticSUMOController
    # UPDATED: Passing the YOLO vehicle_data directly to the Static Controller
    static_brain = StaticSUMOController(max_steps=1000, vehicle_data=vehicle_data, data_lock=lock)
    threading.Thread(target=static_brain.run, daemon=True).start()
    print("🚦 Static SUMO-simulation baseline started.")
except Exception as e:
    print(f"⚠️  SUMO baseline unavailable ({e}), using simple static logic.")
    static_brain = StaticTrafficSystem()
    threading.Thread(target=static_brain.decide, daemon=True).start()

def _wire_metrics_callback():
    traffic_brain._metrics_cb = lambda: metrics.record_phase_switch('rl')
    if hasattr(static_brain, '_metrics_cb'):
        static_brain._metrics_cb = lambda: metrics.record_phase_switch('static')

class MetricsTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.phase_switches = {'rl': 0, 'static': 0}
        self._current = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        self._current_queue = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        self._current_density = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}
        self._current_speed = {'N': 1.0, 'S': 1.0, 'E': 1.0, 'W': 1.0}
        self._timeline_labels = collections.deque(maxlen=30)
        self._timeline_vehicles = collections.deque(maxlen=30)
        self._timeline_queue = collections.deque(maxlen=30)
        self._timeline_density = collections.deque(maxlen=30)
        self._timeline_wait = collections.deque(maxlen=30)

    def record_frame(self, camera_id, count, queue, density, speed):
        direction = camera_id.replace('camera-', '')
        with self.lock:
            self._current[direction] = count
            self._current_queue[direction] = queue
            self._current_density[direction] = density
            self._current_speed[direction] = speed

    def record_phase_switch(self, system='rl'):
        with self.lock:
            if system not in self.phase_switches:
                self.phase_switches[system] = 0
            self.phase_switches[system] += 1

    def snapshot_timeline(self):
        with self.lock:
            now = time.strftime('%H:%M:%S')
            total_v = sum(self._current.values())
            avg_q = sum(self._current_queue.values()) / 4
            avg_d = sum(self._current_density.values()) / 4
            avg_s = sum(self._current_speed.values()) / 4
            wait = avg_d * (1 - avg_s) * 120

            self._timeline_labels.append(now)
            self._timeline_vehicles.append(total_v)
            self._timeline_queue.append(round(avg_q, 2))
            self._timeline_density.append(round(avg_d * 100, 1))
            self._timeline_wait.append(round(wait, 1))

    def get_metrics(self):
        with self.lock:
            uptime = time.time() - self.start_time
            total_vehicles = sum(self._current.values())
            avg_q = sum(self._current_queue.values()) / 4
            avg_d = sum(self._current_density.values()) / 4
            avg_s = sum(self._current_speed.values()) / 4
            wait_proxy = avg_d * (1 - avg_s) * 120
            idle_factor = avg_d * (1 - avg_s)

            rl_timeline = {
                'labels': list(self._timeline_labels),
                'vehicles': list(self._timeline_vehicles),
                'queue': list(self._timeline_queue),
                'density': list(self._timeline_density),
                'wait_time': list(self._timeline_wait),
            }
            rl_bundle = {
                'name': 'PPO (ppo_traffic_final)' if traffic_brain.rl_model else 'Max-Pressure Fallback',
                'active': traffic_brain.rl_model is not None,
                'type': 'Reinforcement Learning (PPO)' if traffic_brain.rl_model else 'Heuristic',
                'phase_switches': self.phase_switches.get('rl', 0),
                'current_phase': 'NS Green' if traffic_brain.current_phase == 0 else 'EW Green',
                'sub_phase': traffic_brain.sub_phase,
                'is_yellow': traffic_brain.is_yellow,
                'realtime': {
                    'avg_queue_length': round(avg_q, 2),
                    'avg_density_pct': round(avg_d * 100, 1),
                    'avg_speed_factor': round(avg_s, 2),
                    'est_wait_time_s': round(wait_proxy, 1),
                    'idle_emissions_factor': round(idle_factor, 3),
                },
                'per_direction': dict(self._current),
                'timeline': rl_timeline
            }

            if hasattr(static_brain, 'get_metrics') and callable(getattr(static_brain, 'get_metrics')):
                static_bundle = static_brain.get_metrics()
                static_bundle['current_phase'] = 'NS Green' if getattr(static_brain, 'current_phase', 0) == 0 else 'EW Green'
                static_bundle['sub_phase'] = getattr(static_brain, 'sub_phase', 'straight')
                static_bundle['is_yellow'] = getattr(static_brain, 'is_yellow', False)
            else:
                static_bundle = {
                    'name': 'fixed time',
                    'active': True,
                    'type': 'Static Baseline',
                    'phase_switches': self.phase_switches.get('static', 0),
                    'current_phase': 'NS Green' if static_brain.current_phase == 0 else 'EW Green',
                    'sub_phase': static_brain.sub_phase,
                    'is_yellow': static_brain.is_yellow,
                    'realtime': {
                        'avg_queue_length': round(avg_q, 2),
                        'avg_density_pct': round(avg_d * 100, 1),
                        'avg_speed_factor': round(avg_s, 2),
                        'est_wait_time_s': round(wait_proxy, 1),
                        'idle_emissions_factor': round(idle_factor, 3),
                    },
                    'per_direction': dict(self._current),
                    'timeline': rl_timeline
                }

            return {
                'uptime_seconds': round(uptime, 0),
                'total_vehicles_now': total_vehicles,
                'rl': rl_bundle,
                'static': static_bundle
            }

metrics = MetricsTracker()
_wire_metrics_callback()

def _metrics_timeline_loop():
    while True:
        time.sleep(5)
        metrics.snapshot_timeline()

threading.Thread(target=_metrics_timeline_loop, daemon=True).start()

def calculate_metrics(vehicle_count):
    MAX_CAPACITY = 20.0 
    density = min(vehicle_count / MAX_CAPACITY, 1.0)
    speed = max(0.1, 1.0 - density)
    queue = vehicle_count if density > 0.6 else 0
    return queue, density, speed

class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.objects = {} # id: (rect, cls)
        self.history = {} # id: [(cx, cy), ...]
        self.disappeared = {} # id: count

    def update(self, rects_data):
        new_objects = {}
        active_objects = {}
        available_objects = dict(self.objects)
        
        for item in rects_data:
            rect = item[:4]
            cls = item[4]
            cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
            matched_id = None
            min_dist = 120
            
            for obj_id, (o_rect, o_cls) in available_objects.items():
                ocx, ocy = (o_rect[0] + o_rect[2]) / 2, (o_rect[1] + o_rect[3]) / 2
                dist = ((cx - ocx)**2 + (cy - ocy)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    matched_id = obj_id
            
            if matched_id is not None:
                new_objects[matched_id] = (rect, cls)
                active_objects[matched_id] = (rect, cls)
                self.history[matched_id].append((cx, cy))
                if len(self.history[matched_id]) > 10:
                    self.history[matched_id].pop(0)
                self.disappeared[matched_id] = 0
                del available_objects[matched_id]
            else:
                new_objects[self.next_id] = (rect, cls)
                active_objects[self.next_id] = (rect, cls)
                self.history[self.next_id] = [(cx, cy)]
                self.disappeared[self.next_id] = 0
                self.next_id += 1
                
        for obj_id, (o_rect, o_cls) in available_objects.items():
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] <= 2:
                new_objects[obj_id] = (o_rect, o_cls) 
            else:
                if obj_id in self.history: del self.history[obj_id]
                del self.disappeared[obj_id]
                
        self.objects = new_objects
        return active_objects, self.history

def process_camera_stream(camera_info):
    global output_frames, lock, vehicle_data
    camera_id = camera_info['id']
    tracker = SimpleTracker()
    
    while True:
        src = camera_info['localPath']
        if not os.path.exists(src):
            print(f"[{camera_id}] Local file not found: {src}")
            time.sleep(2)
            continue

        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[{camera_id}] Failed to open video.")
            time.sleep(5)
            continue

        FRAME_SKIP = 6 
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1
            if frame_count % FRAME_SKIP != 0: continue

            frame_resized = cv2.resize(frame, (640, 360))

            results = []
            try:
                with yolo_lock:
                    results = model.predict(
                        frame_resized,
                        classes=CLASS_IDS_TO_DETECT,
                        verbose=False,
                        device=device,
                    )
            except Exception as e:
                print(f"[{camera_id}] YOLO error, skipping frame: {e}")
                results = []

            count = 0
            count_incoming = 0
            weighted_incoming_count = 0.0
            count_outgoing = 0
            is_green = False
            if len(results) > 0:
                result = results[0]
                
                rects_data = []
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0].item())
                    rects_data.append((x1, y1, x2, y2, cls))
                
                objects, history = tracker.update(rects_data)
                
                if camera_info['direction'] in ['N', 'S'] and traffic_brain.current_phase == 0: is_green = True
                elif camera_info['direction'] in ['E', 'W'] and traffic_brain.current_phase == 1: is_green = True
                
                if traffic_brain.is_yellow:
                    base_color = (0, 255, 255)
                elif is_green and traffic_brain.sub_phase == 'left_turn':
                    base_color = (0, 255, 128)
                elif is_green and traffic_brain.sub_phase == 'right_turn':
                    base_color = (0, 191, 255)
                elif is_green:
                    base_color = (0, 255, 0)
                else:
                    base_color = (0, 0, 255)
                    
                # The invisible lane divider line (uncomment the line below if you ever need to visually debug them again)
                pt1, pt2 = LANE_DIVIDERS.get(camera_id, ((320, 0), (320, 360)))
                # cv2.line(frame_resized, pt1, pt2, (255, 0, 255), 1)
                
                for obj_id, (rect, cls) in objects.items():
                    x1, y1, x2, y2 = [int(v) for v in rect]
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    hist = history.get(obj_id, [])
                    
                    is_moving = False
                    dy = 0
                    if len(hist) >= 3:
                        dy = hist[-1][1] - hist[0][1]
                        dx = hist[-1][0] - hist[0][0]
                        if abs(dy) > 3 or abs(dx) > 3:
                            is_moving = True
                            
                    # Calculate dynamic divider_x based on the configurable LANE_DIVIDERS
                    divider_x = pt1[0] + (cy / 360.0) * (pt2[0] - pt1[0])
                            
                    # Filter out non-moving vehicles parked on the far edges (relative to the actual road center)
                    if not is_moving and abs(cx - divider_x) > 220:
                        continue
                        
                    # Primary Logic: The Invisible Line (Lane Position)
                    if cx < divider_x:
                        direction = "Outgoing"
                    else:
                        direction = "Incoming"
                        
                    # Override ONLY if they are blatantly driving on the wrong side of the road (India traffic)
                    if is_moving:
                        if direction == "Outgoing" and dy > 6:
                            direction = "Incoming" # Wrong way towards intersection
                        elif direction == "Incoming" and dy < -6:
                            direction = "Outgoing" # Wrong way away from intersection
                    
                    if direction == "Outgoing":
                        count_outgoing += 1
                        color = (128, 128, 128)
                        label = "Out"
                    else:
                        count_incoming += 1
                        weight = VEHICLE_WEIGHTS.get(cls, 1.0)
                        weighted_incoming_count += weight
                        color = base_color
                        cls_name = CLASS_NAMES.get(cls, 'Veh')
                        label = f"In ({cls_name})"
                        
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(frame_resized, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            count = count_incoming
            queue, density, speed = calculate_metrics(weighted_incoming_count)
            with lock:
                vehicle_data[camera_id] = {'count': count, 'queue': queue, 'density': density, 'speed': speed}
            metrics.record_frame(camera_id, count, queue, density, speed)

            with lock:
                flag, encodedImage = cv2.imencode(".jpg", frame_resized)
                if flag: output_frames[camera_id] = encodedImage.tobytes()
        
        cap.release()
        time.sleep(1)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/cameras')
def get_cameras(): return jsonify(CAMERAS)

@app.route('/video_feed')
def video_feed():
    camera_id = request.args.get('id')
    return Response(generate_frame_for_request(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frame_for_request(camera_id):
    global output_frames, lock
    while True:
        time.sleep(0.05)
        with lock:
            if camera_id not in output_frames: continue
            frame_bytes = output_frames[camera_id]
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/api/signal-status')
def get_signal_status():
    phase_name = "NS Green" if traffic_brain.current_phase == 0 else "EW Green"
    
    if traffic_brain.is_yellow:
        target_time = traffic_brain.yellow_duration
        elapsed = time.time() - traffic_brain.yellow_start_time
    else:
        elapsed = time.time() - traffic_brain.state_start_time
        if traffic_brain.sub_phase == 'left_turn': target_time = traffic_brain.left_turn_duration
        elif traffic_brain.sub_phase == 'straight': 
            target_time = traffic_brain.max_green_time - traffic_brain.left_turn_duration - traffic_brain.right_turn_duration
        else: target_time = traffic_brain.right_turn_duration
        
    rl_state_timer = max(0.0, round(target_time - elapsed, 1))

    rl_state = {
        'current_phase_code': traffic_brain.current_phase,
        'current_phase_text': phase_name,
        'is_yellow': traffic_brain.is_yellow,
        'sub_phase': traffic_brain.sub_phase,
        'state_timer': rl_state_timer,
    }
    
    s_phase_code = getattr(static_brain, 'current_phase', 0)
    s_phase = "NS Green" if s_phase_code == 0 else "EW Green"
    
    s_is_yellow = getattr(static_brain, 'is_yellow', False)
    s_sub_phase = getattr(static_brain, 'sub_phase', 'straight')
    s_yellow_start = getattr(static_brain, 'yellow_start_time', 0)
    s_state_start = getattr(static_brain, 'state_start_time', time.time())
    
    if s_is_yellow:
        s_target = getattr(static_brain, 'yellow_duration', 4)
        s_elapsed = time.time() - s_yellow_start
    else:
        s_elapsed = time.time() - s_state_start
        if s_sub_phase == 'left_turn': s_target = getattr(static_brain, 'left_turn_duration', 12)
        elif s_sub_phase == 'straight': s_target = getattr(static_brain, 'straight_duration', 20)
        else: s_target = getattr(static_brain, 'right_turn_duration', 8)
        
    s_state_timer = max(0.0, round(s_target - s_elapsed, 1))
    
    static_state = {
        'current_phase_code': s_phase_code,
        'current_phase_text': s_phase,
        'is_yellow': s_is_yellow,
        'sub_phase': s_sub_phase,
        'state_timer': s_state_timer,
    }
    
    return jsonify({
        'rl': rl_state,
        'static': static_state,
        'live_data': vehicle_data
    })

@app.route('/api/metrics')
def get_metrics():
    return jsonify(metrics.get_metrics())

if __name__ == '__main__':
    for camera in CAMERAS:
        threading.Thread(target=process_camera_stream, args=(camera,), daemon=True).start()
    print("🚀 Server Started.")
    app.run(debug=False, threaded=True, port=5000)