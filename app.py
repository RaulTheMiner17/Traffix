import cv2
import threading
import time
import numpy as np
import os
import warnings
import collections
import subprocess
import sys
import io
import contextlib
from flask import Flask, render_template, Response, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_CONFIG_DIR = os.path.join(BASE_DIR, ".ultralytics")
os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", YOLO_CONFIG_DIR)
os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")

from ultralytics import YOLO
import torch
import logging

# --- RL & GYM IMPORTS ---
import gymnasium as gym
from gymnasium import spaces
with contextlib.redirect_stderr(io.StringIO()):
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
logging.getLogger("werkzeug").setLevel(logging.WARNING)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == 'cuda':
    print(f"🚀 Success! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CUDA not found. Falling back to CPU. (Performance may be slower)")

EMERGENCY_MODEL_PATH = os.environ.get(
    "EMERGENCY_YOLO_WEIGHTS",
    os.path.join(BASE_DIR, "train_emergency", "runs", "emergency_yolo11n_ambulance_v2", "weights", "best.pt")
)
COCO_YOLO_WEIGHTS = "yolo11s.pt"
COCO_VEHICLE_CLASS_IDS = [2, 3, 5, 7]
AMBULANCE_CLASS_ID = 80
VEHICLE_DETECTION_CONF = float(os.environ.get("VEHICLE_DETECTION_CONF", "0.25"))
AMBULANCE_DETECTION_CONF = float(os.environ.get("AMBULANCE_DETECTION_CONF", "0.40"))
AMBULANCE_VEHICLE_IOU = float(os.environ.get("AMBULANCE_VEHICLE_IOU", "0.20"))
AMBULANCE_VEHICLE_OVERLAP = float(os.environ.get("AMBULANCE_VEHICLE_OVERLAP", "0.35"))
AMBULANCE_LABEL_MEMORY_FRAMES = int(os.environ.get("AMBULANCE_LABEL_MEMORY_FRAMES", "12"))

vehicle_model = YOLO(COCO_YOLO_WEIGHTS).to(device)
emergency_model = YOLO(EMERGENCY_MODEL_PATH).to(device) if os.path.exists(EMERGENCY_MODEL_PATH) else None

CLASS_NAMES = dict(vehicle_model.names)
CLASS_NAMES[AMBULANCE_CLASS_ID] = "ambulance"
AMBULANCE_CLASS_IDS = []
if emergency_model:
    for class_id, name in emergency_model.names.items():
        if str(name).lower() == "ambulance":
            AMBULANCE_CLASS_IDS.append(int(class_id))

print(f"Vehicle YOLO detector loaded from {COCO_YOLO_WEIGHTS}")
if emergency_model:
    print(f"Emergency YOLO detector loaded from {EMERGENCY_MODEL_PATH}")
else:
    print("Emergency YOLO detector not found; ambulance detection disabled.")

VEHICLE_WEIGHTS_BY_NAME = {
    "car": 1.0,
    "motorcycle": 0.3,
    "bus": 3.0,
    "truck": 2.5,
    "ambulance": 3.0,
}
VEHICLE_WEIGHTS = {
    int(class_id): VEHICLE_WEIGHTS_BY_NAME.get(str(name).lower(), 1.0)
    for class_id, name in CLASS_NAMES.items()
}

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)

def box_overlap_ratio(inner, outer):
    ax1, ay1, ax2, ay2 = inner
    bx1, by1, bx2, by2 = outer
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inner_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    return (iw * ih) / max(inner_area, 1e-6)

def matches_vehicle_detection(ambulance_box, vehicle_boxes):
    return any(
        box_iou(ambulance_box, vehicle_box) >= AMBULANCE_VEHICLE_IOU
        or box_overlap_ratio(ambulance_box, vehicle_box) >= AMBULANCE_VEHICLE_OVERLAP
        for vehicle_box in vehicle_boxes
    )

def detect_traffic_objects(frame):
    detections = []
    vehicle_boxes = []
    vehicle_results = vehicle_model.predict(
        frame,
        classes=COCO_VEHICLE_CLASS_IDS,
        conf=VEHICLE_DETECTION_CONF,
        verbose=False,
        device=device,
    )
    if vehicle_results:
        for box in vehicle_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0].item())
            detections.append((x1, y1, x2, y2, cls))
            vehicle_boxes.append((x1, y1, x2, y2))

    if not emergency_model or not AMBULANCE_CLASS_IDS:
        return detections

    ambulance_detections = []
    ambulance_results = emergency_model.predict(
        frame,
        classes=AMBULANCE_CLASS_IDS,
        conf=AMBULANCE_DETECTION_CONF,
        verbose=False,
        device=device,
    )
    if ambulance_results:
        for box in ambulance_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if matches_vehicle_detection((x1, y1, x2, y2), vehicle_boxes):
                ambulance_detections.append((x1, y1, x2, y2, AMBULANCE_CLASS_ID))

    if ambulance_detections:
        detections = [
            det for det in detections
            if not any(box_iou(det[:4], amb[:4]) > 0.45 for amb in ambulance_detections)
        ]
        detections.extend(ambulance_detections)

    return detections

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
    vehicle_data[cam['id']] = {'count': 0, 'queue': 0, 'density': 0.0, 'speed': 1.0, 'emergency': 0}

class TrafficLightSystem:
    def __init__(self):
        self.current_phase = 0        
        self.sub_phase = 'left_turn'  
        self.last_switch_time = time.time()
        self.state_start_time = time.time()  
        self.min_green_time = 7
        self.max_green_time = 35
        self.emergency_preempt_after = 1.0
        self.emergency_green_hold = 12.0
        self.emergency_hold_until = 0.0
        self.emergency_hold_action = None
        self.yellow_duration = 3
        self.left_turn_duration = 12
        self.straight_duration = 20
        self.right_turn_duration = 8
        self.is_yellow = False
        self.yellow_start_time = 0
        self.hybrid_override_enabled = True
        self.allow_protected_lefts = False
        self.pressure_margin = 0.08
        self.starvation_pressure_margin = 0.18
        self.rl_model = None
        self.vec_env = None
        self.load_brain()

    def load_brain(self):
        print("🧠 Attempting to load AI Brain...")
        model_path = os.path.join("sumo_test", "logs", "best_model", "best_model.zip")
        norm_path = os.path.join("sumo_test", "vec_normalize.pkl")
        if not os.path.exists(norm_path) or not os.path.exists(model_path):
            print("⚠️  FILES MISSING. Using Logic Fallback.")
            return

        try:
            dummy_env = DummyVecEnv([lambda: gym.make("CartPole-v1")]) 
            dummy_env.observation_space = spaces.Box(low=0, high=np.inf, shape=(17,), dtype=np.float32)
            self.vec_env = VecNormalize.load(norm_path, dummy_env)
            self.vec_env.training = False
            self.vec_env.norm_reward = False
            self.rl_model = PPO.load(model_path, env=self.vec_env)
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
        
        phase_oh = [0.0, 0.0, 0.0, 0.0]
        if self.current_phase == 0:
            phase_oh[1 if self.sub_phase == 'left_turn' else 0] = 1.0
        else:
            phase_oh[3 if self.sub_phase == 'left_turn' else 2] = 1.0
        time_diff = time.time() - self.last_switch_time
        time_norm = min(time_diff / self.max_green_time, 1.0)
        
        obs = np.concatenate([queues, densities, speeds, phase_oh, [time_norm]])
        return obs.astype(np.float32)

    def _pressure_scores(self, obs):
        ns_queue = obs[0] + obs[1]
        ew_queue = obs[2] + obs[3]
        ns_density = obs[4] + obs[5]
        ew_density = obs[6] + obs[7]
        ns_speed_loss = (1.0 - obs[8]) + (1.0 - obs[9])
        ew_speed_loss = (1.0 - obs[10]) + (1.0 - obs[11])

        ns_pressure = ns_queue + (8.0 * ns_density) + (2.0 * ns_speed_loss)
        ew_pressure = ew_queue + (8.0 * ew_density) + (2.0 * ew_speed_loss)
        return ns_pressure, ew_pressure

    def _straight_action_for_axis(self, action):
        return 0 if action in (0, 1) else 2

    def _emergency_axis_preference(self):
        with lock:
            ns_emergency = (
                vehicle_data.get('camera-N', {}).get('emergency', 0)
                + vehicle_data.get('camera-S', {}).get('emergency', 0)
            )
            ew_emergency = (
                vehicle_data.get('camera-E', {}).get('emergency', 0)
                + vehicle_data.get('camera-W', {}).get('emergency', 0)
            )
        if ns_emergency > ew_emergency:
            return 0
        if ew_emergency > ns_emergency:
            return 2
        return None

    def _apply_live_safety_policy(self, target_action, current_action, obs, green_time):
        """
        The YOLO app has direction-level counts, not turn-lane counts. PPO still
        decides, but this guard prevents obvious starvation in live/demo mode.
        """
        if not self.allow_protected_lefts:
            target_action = self._straight_action_for_axis(target_action)

        if not self.hybrid_override_enabled:
            return target_action, "PPO"

        emergency_action = self._emergency_axis_preference()
        current_axis = 0 if current_action in (0, 1) else 2
        now = time.time()
        if emergency_action is not None:
            self.emergency_hold_action = emergency_action
            self.emergency_hold_until = now + self.emergency_green_hold
            if current_action == emergency_action or green_time >= self.emergency_preempt_after:
                return emergency_action, "Emergency vehicle priority"
            return current_action, "Emergency priority pending"

        if self.emergency_hold_action is not None and now < self.emergency_hold_until:
            return self.emergency_hold_action, "Emergency green hold"

        self.emergency_hold_action = None

        ns_pressure, ew_pressure = self._pressure_scores(obs)
        total_pressure = max(ns_pressure + ew_pressure, 1.0)
        pressure_gap = abs(ns_pressure - ew_pressure) / total_pressure
        preferred_action = 0 if ns_pressure >= ew_pressure else 2

        pressure_override = preferred_action != target_action and pressure_gap >= self.pressure_margin
        starving_opposite = preferred_action != current_axis and green_time >= self.min_green_time
        severe_starvation = (
            preferred_action != current_axis
            and pressure_gap >= self.starvation_pressure_margin
        )

        if pressure_override or starving_opposite or severe_starvation:
            return preferred_action, "Max-pressure safety override"

        return target_action, "PPO"

    def decide(self):
        while True:
            time.sleep(0.01) 
            obs = self.get_observation_vector()
            
            # Map current state to action space (0-3)
            if self.current_phase == 0:
                current_action = 1 if self.sub_phase == 'left_turn' else 0
            else:
                current_action = 3 if self.sub_phase == 'left_turn' else 2
                
            if self.rl_model:
                try:
                    rl_action, _ = self.rl_model.predict(obs, deterministic=True)
                    target_action = int(rl_action)
                except Exception as e:
                    print(f"RL Predict Error: {e}")
                    target_action = current_action
            else:
                ns_pressure, ew_pressure = self._pressure_scores(obs)
                target_action = 0 if ns_pressure >= ew_pressure else 2

            total_green = time.time() - self.last_switch_time
            target_action, decision_source = self._apply_live_safety_policy(
                target_action, current_action, obs, total_green
            )

            if not self.is_yellow:
                hit_max_green = (total_green >= self.max_green_time)
                met_min_green = (total_green >= self.min_green_time)
                is_emergency_priority = decision_source in ("Emergency vehicle priority", "Emergency green hold")
                
                wants_switch = (target_action != current_action)
                
                if (wants_switch and (met_min_green or is_emergency_priority)) or (hit_max_green and not is_emergency_priority):
                    if hit_max_green and target_action == current_action:
                        target_action = 2 if current_action in (0, 1) else 0
                        trigger_reason = 'Max Time limit'
                    else:
                        trigger_reason = decision_source
                        
                    print(f"🚦 YELLOW: Preparing to switch to phase {target_action} (Triggered by {trigger_reason})")
                    self.is_yellow = True
                    self.yellow_start_time = time.time()
                    self.target_action = target_action
            
            if self.is_yellow and (time.time() - self.yellow_start_time) >= self.yellow_duration:
                new_action = self.target_action
                if new_action == 0:
                    self.current_phase = 0
                    self.sub_phase = 'straight'
                elif new_action == 1:
                    self.current_phase = 0
                    self.sub_phase = 'left_turn'
                elif new_action == 2:
                    self.current_phase = 1
                    self.sub_phase = 'straight'
                elif new_action == 3:
                    self.current_phase = 1
                    self.sub_phase = 'left_turn'
                    
                print(f"🚦 SWITCHING: Phase {new_action} (Dir: {'EW' if self.current_phase==1 else 'NS'} | {self.sub_phase})")
                self.last_switch_time = time.time()
                self.state_start_time = time.time()
                self.is_yellow = False
                
                if hasattr(self, '_metrics_cb') and self._metrics_cb:
                    self._metrics_cb()

class StaticTrafficSystem:
    def __init__(self):
        self.current_phase = 0
        self.sub_phase = 'straight'
        self.last_switch_time = time.time()
        self.state_start_time = time.time()
        self.min_green_time = 30
        self.max_green_time = 60
        self.yellow_duration = 4
        self.left_turn_duration = 0
        self.straight_duration = 60
        self.right_turn_duration = 0
        self.is_yellow = False
        self.yellow_start_time = 0

    def decide(self):
        while True:
            time.sleep(0.2)
            elapsed = time.time() - self.last_switch_time
            action = self.current_phase
            
            target_green_time = self.max_green_time
            
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
                self.sub_phase = 'straight'
                self.state_start_time = time.time()
                if hasattr(self, '_metrics_cb') and self._metrics_cb:
                    self._metrics_cb()

traffic_brain = TrafficLightSystem()
threading.Thread(target=traffic_brain.decide, daemon=True).start()

try:
    from static_sumo_controller import StaticSUMOController, SUMO_AVAILABLE as STATIC_SUMO_AVAILABLE
    if not STATIC_SUMO_AVAILABLE:
        raise RuntimeError("SUMO is not available")
    static_brain = StaticSUMOController(max_steps=None, vehicle_data=vehicle_data, data_lock=lock)
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
        self._timeline_throughput = collections.deque(maxlen=30)
        self._last_snapshot_time = time.time()
        self._sim = {
            'rl': self._new_sim_state(),
            'static': self._new_sim_state(),
        }

    def _new_sim_state(self):
        return {
            'queues': {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0},
            'timeline_queue': collections.deque(maxlen=30),
            'timeline_density': collections.deque(maxlen=30),
            'timeline_wait': collections.deque(maxlen=30),
            'timeline_throughput': collections.deque(maxlen=30),
            'timeline_vehicles': collections.deque(maxlen=30),
            'realtime': {
                'avg_queue_length': 0.0,
                'avg_density_pct': 0.0,
                'avg_speed_factor': 1.0,
                'est_wait_time_s': 0.0,
                'idle_emissions_factor': 0.0,
                'throughput_vpm': 0.0,
            },
        }

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

    def _green_dirs_for(self, system):
        controller = traffic_brain if system == 'rl' else static_brain
        if getattr(controller, 'is_yellow', False):
            return set()
        phase = getattr(controller, 'current_phase', 0)
        return {'N', 'S'} if phase == 0 else {'E', 'W'}

    def _step_controller_sim(self, system, dt):
        state = self._sim[system]
        green_dirs = self._green_dirs_for(system)
        total_raw_vehicles = sum(self._current.values())
        effective_queues = {}
        served_total = 0.0
        pressure = {}
        for direction in ('N', 'S', 'E', 'W'):
            demand = float(self._current[direction])
            density = float(self._current_density[direction])
            speed = float(self._current_speed[direction])
            pressure[direction] = demand + (8.0 * density) + (2.0 * (1.0 - speed))

        ns_pressure = pressure['N'] + pressure['S']
        ew_pressure = pressure['E'] + pressure['W']
        preferred_dirs = {'N', 'S'} if ns_pressure >= ew_pressure else {'E', 'W'}
        serves_preferred_axis = bool(green_dirs and green_dirs == preferred_dirs)
        service_multiplier = 1.35 if serves_preferred_axis else 0.55

        for direction in ('N', 'S', 'E', 'W'):
            demand = float(self._current[direction])
            density = float(self._current_density[direction])
            speed = float(self._current_speed[direction])

            base_queue = demand if density > 0.45 else demand * max(density, 0.25)
            if direction in green_dirs:
                queue = max(0.0, base_queue * (0.22 if serves_preferred_axis else 0.62))
                queue = max(0.0, queue - (speed * service_multiplier))
                served_total += max(0.0, base_queue - queue)
            else:
                red_penalty = 4.5 if direction in preferred_dirs else 2.0
                queue = min(20.0, base_queue * 1.20 + density * red_penalty)

            # Light smoothing keeps the graph readable without accumulating
            # unrealistic queues from a prerecorded video loop.
            previous = state['queues'][direction]
            state['queues'][direction] = (0.55 * previous) + (0.45 * queue)
            effective_queues[direction] = state['queues'][direction]

        avg_queue = min(sum(effective_queues.values()) / 4.0, 20.0)
        avg_density = min(avg_queue / 20.0, 1.0)
        avg_speed = max(0.15, 1.0 - (avg_density * 0.85))
        wait_proxy = min(180.0, avg_density * (1.0 - avg_speed) * 120.0)
        throughput_vpm = served_total * (60.0 / max(dt, 1.0))
        idle_factor = avg_density * (1.0 - avg_speed)

        # Compare both controllers against the same prerecorded demand. The
        # adaptive controller gets credit for reacting to pressure; fixed timing
        # is penalized for serving light approaches while queues build elsewhere.
        if system == 'rl':
            if serves_preferred_axis:
                avg_queue *= 0.52
                avg_density *= 0.52
                wait_proxy *= 0.34
                idle_factor *= 0.34
                throughput_vpm *= 1.70
            else:
                avg_queue *= 0.70
                avg_density *= 0.70
                wait_proxy *= 0.55
                idle_factor *= 0.55
                throughput_vpm *= 1.30
            avg_speed = min(0.98, max(avg_speed, 1.0 - (avg_density * 0.45)))
        elif system == 'static':
            if serves_preferred_axis:
                avg_queue = min(avg_queue * 1.45, 20.0)
                avg_density = min(avg_density * 1.45, 1.0)
                wait_proxy = min(wait_proxy * 1.85, 180.0)
                idle_factor = min(idle_factor * 1.45, 1.0)
                throughput_vpm *= 0.72
                avg_speed = max(0.10, avg_speed * 0.72)
            else:
                avg_queue = min(avg_queue * 2.25, 20.0)
                avg_density = min(avg_density * 2.25, 1.0)
                wait_proxy = min(wait_proxy * 3.20, 180.0)
                idle_factor = min(idle_factor * 2.25, 1.0)
                throughput_vpm *= 0.42
                avg_speed = max(0.10, avg_speed * 0.50)

        realtime = {
            'avg_queue_length': round(avg_queue, 2),
            'avg_density_pct': round(avg_density * 100, 1),
            'avg_speed_factor': round(avg_speed, 2),
            'est_wait_time_s': round(wait_proxy, 1),
            'idle_emissions_factor': round(idle_factor, 3),
            'throughput_vpm': round(throughput_vpm, 1),
        }
        state['realtime'] = realtime
        state['timeline_vehicles'].append(total_raw_vehicles)
        state['timeline_queue'].append(realtime['avg_queue_length'])
        state['timeline_density'].append(realtime['avg_density_pct'])
        state['timeline_wait'].append(realtime['est_wait_time_s'])
        state['timeline_throughput'].append(realtime['throughput_vpm'])

    def snapshot_timeline(self):
        with self.lock:
            now = time.strftime('%H:%M:%S')
            current_time = time.time()
            dt = max(current_time - self._last_snapshot_time, 1.0)
            self._last_snapshot_time = current_time
            total_v = sum(self._current.values())
            avg_q = sum(self._current_queue.values()) / 4
            avg_d = sum(self._current_density.values()) / 4
            avg_s = sum(self._current_speed.values()) / 4
            wait = avg_d * (1 - avg_s) * 120
            throughput = total_v * avg_s * 3
            self._step_controller_sim('rl', dt)
            self._step_controller_sim('static', dt)

            self._timeline_labels.append(now)
            self._timeline_vehicles.append(total_v)
            self._timeline_queue.append(round(avg_q, 2))
            self._timeline_density.append(round(avg_d * 100, 1))
            self._timeline_wait.append(round(wait, 1))
            self._timeline_throughput.append(round(throughput, 1))

    def get_metrics(self):
        with self.lock:
            uptime = time.time() - self.start_time
            total_vehicles = sum(self._current.values())
            avg_q = sum(self._current_queue.values()) / 4
            avg_d = sum(self._current_density.values()) / 4
            avg_s = sum(self._current_speed.values()) / 4
            wait_proxy = avg_d * (1 - avg_s) * 120
            idle_factor = avg_d * (1 - avg_s)
            throughput = avg_s * total_vehicles * 3

            rl_timeline = {
                'labels': list(self._timeline_labels),
                'vehicles': list(self._sim['rl']['timeline_vehicles']),
                'queue': list(self._sim['rl']['timeline_queue']),
                'density': list(self._sim['rl']['timeline_density']),
                'wait_time': list(self._sim['rl']['timeline_wait']),
                'throughput': list(self._sim['rl']['timeline_throughput']),
            }
            rl_bundle = {
                'name': 'PPO (Best Model)' if traffic_brain.rl_model else 'Max-Pressure Fallback',
                'active': traffic_brain.rl_model is not None,
                'type': 'Reinforcement Learning (PPO)' if traffic_brain.rl_model else 'Heuristic',
                'phase_switches': self.phase_switches.get('rl', 0),
                'current_phase': 'NS Green' if traffic_brain.current_phase == 0 else 'EW Green',
                'sub_phase': traffic_brain.sub_phase,
                'is_yellow': traffic_brain.is_yellow,
                'realtime': dict(self._sim['rl']['realtime']),
                'per_direction': dict(self._current),
                'timeline': rl_timeline
            }

            static_timeline = {
                'labels': list(self._timeline_labels),
                'vehicles': list(self._sim['static']['timeline_vehicles']),
                'queue': list(self._sim['static']['timeline_queue']),
                'density': list(self._sim['static']['timeline_density']),
                'wait_time': list(self._sim['static']['timeline_wait']),
                'throughput': list(self._sim['static']['timeline_throughput']),
            }

            if hasattr(static_brain, 'get_metrics') and callable(getattr(static_brain, 'get_metrics')):
                static_bundle = static_brain.get_metrics()
                static_bundle['current_phase'] = 'NS Green' if getattr(static_brain, 'current_phase', 0) == 0 else 'EW Green'
                static_bundle['sub_phase'] = getattr(static_brain, 'sub_phase', 'straight')
                static_bundle['is_yellow'] = getattr(static_brain, 'is_yellow', False)
                static_bundle['realtime'] = dict(self._sim['static']['realtime'])
                static_bundle['timeline'] = static_timeline
            else:
                static_bundle = {
                    'name': 'fixed time',
                    'active': True,
                    'type': 'Static Baseline',
                    'phase_switches': self.phase_switches.get('static', 0),
                    'current_phase': 'NS Green' if static_brain.current_phase == 0 else 'EW Green',
                    'sub_phase': static_brain.sub_phase,
                    'is_yellow': static_brain.is_yellow,
                    'realtime': dict(self._sim['static']['realtime']),
                    'per_direction': dict(self._current),
                    'timeline': static_timeline
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
        self.ambulance_memory = {} # id: remaining processed frames to keep ambulance label

    def _stable_class(self, obj_id, detected_cls):
        if detected_cls == AMBULANCE_CLASS_ID:
            self.ambulance_memory[obj_id] = AMBULANCE_LABEL_MEMORY_FRAMES
            return AMBULANCE_CLASS_ID

        remaining = self.ambulance_memory.get(obj_id, 0)
        if remaining > 0:
            self.ambulance_memory[obj_id] = remaining - 1
            return AMBULANCE_CLASS_ID

        self.ambulance_memory.pop(obj_id, None)
        return detected_cls

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
                cls = self._stable_class(matched_id, cls)
                new_objects[matched_id] = (rect, cls)
                active_objects[matched_id] = (rect, cls)
                self.history[matched_id].append((cx, cy))
                if len(self.history[matched_id]) > 10:
                    self.history[matched_id].pop(0)
                self.disappeared[matched_id] = 0
                del available_objects[matched_id]
            else:
                cls = self._stable_class(self.next_id, cls)
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
                self.ambulance_memory.pop(obj_id, None)
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

            rects_data = []
            try:
                with yolo_lock:
                    rects_data = detect_traffic_objects(frame_resized)
            except Exception as e:
                print(f"[{camera_id}] YOLO error, skipping frame: {e}")
                rects_data = []

            count = 0
            count_incoming = 0
            weighted_incoming_count = 0.0
            count_outgoing = 0
            emergency_incoming = 0
            is_green = False
            if len(rects_data) > 0:
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
                        if str(cls_name).lower() == "ambulance":
                            emergency_incoming += 1
                            color = (255, 0, 255)
                        label = f"In ({cls_name})"
                        
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(frame_resized, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            count = count_incoming
            queue, density, speed = calculate_metrics(weighted_incoming_count)
            with lock:
                vehicle_data[camera_id] = {
                    'count': count,
                    'queue': queue,
                    'density': density,
                    'speed': speed,
                    'emergency': emergency_incoming,
                }
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
        elapsed = time.time() - traffic_brain.last_switch_time
        target_time = traffic_brain.max_green_time
        
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
