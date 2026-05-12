"""
Static SUMO-based controller for baseline comparison.
Runs in a separate process to generate independent metrics.
"""
import os
import sys
import time
import threading
import collections
import subprocess
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SUMO_AVAILABLE = False
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
    try:
        import traci
        SUMO_AVAILABLE = True
    except ImportError:
        pass

class StaticSUMOController:
    # UPDATED: Added vehicle_data and data_lock to the constructor
    def __init__(self, sumo_config="sumo_test/cross.sumocfg", max_steps=3600, vehicle_data=None, data_lock=None):
        self.sumo_config = os.path.join(BASE_DIR, sumo_config)
        self.max_steps = max_steps
        self.vehicle_data = vehicle_data
        self.data_lock = data_lock
        self.proc = None
        self.current_phase = 0
        self.sub_phase = "straight"
        self.is_yellow = False
        self.yellow_start_time = 0
        self.state_start_time = time.time()
        self.min_green_time = 30
        self.max_green_time = 60
        self.yellow_duration = 3
        self.straight_duration = 60
        self.left_turn_duration = 0
        self.right_turn_duration = 0
        self.time_since_switch = 0
        self.lanes = ["N2J_0", "S2J_0", "E2J_0", "W2J_0"]
        self.tls_id = "J0"
        self.is_running = False
        
        self.label = "static_baseline"
        self.conn = None 
        
        self.queue_history = collections.deque(maxlen=30)
        self.density_history = collections.deque(maxlen=30)
        self.wait_time_history = collections.deque(maxlen=30)
        self.vehicle_count_history = collections.deque(maxlen=30)
        self.latest_queue = 0.0
        self.latest_density = 0.0
        self.latest_wait = 0.0
        self.latest_speed = 1.0
        self.latest_vehicle_count = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.phase_switches = 0

    def run(self):
        if not SUMO_AVAILABLE: return

        try:
            sumo_cmd = [
                "sumo",
                "-c", self.sumo_config,
                "--no-step-log", "true",
                "--waiting-time-memory", "1000",
                "--remote-port", "8874",
            ]
            self.proc = subprocess.Popen(sumo_cmd, stdout=subprocess.DEVNULL, stderr=sys.stderr)

            for attempt in range(5):
                try:
                    time.sleep(1) 
                    traci.init(8874, label=self.label)
                    self.conn = traci.getConnection(self.label)
                    break
                except Exception:
                    pass
            else:
                return

            self.is_running = True
            self.conn.trafficlight.setPhase(self.tls_id, 0)
            sim_step = 0
            loop_sleep_s = 0.1
            yellow_duration_steps = max(1, int(self.yellow_duration / loop_sleep_s))
            is_yellow = False
            yellow_start = 0
            
            while self.is_running and (self.max_steps is None or sim_step < self.max_steps):
                self.conn.simulationStep()
                
                time.sleep(loop_sleep_s)
                
                sim_step += 1
                self.time_since_switch += 1

                # ==========================================
                # METRICS COLLECTION (YOLO VIDEO OVERRIDE)
                # ==========================================
                if self.vehicle_data is not None and self.data_lock is not None:
                    # Pull directly from app.py YOLO detections
                    with self.data_lock:
                        queues = [self.vehicle_data[cam]['queue'] for cam in self.vehicle_data]
                        densities = [self.vehicle_data[cam]['density'] for cam in self.vehicle_data]
                        speeds = [self.vehicle_data[cam]['speed'] for cam in self.vehicle_data]
                        total_v = sum([self.vehicle_data[cam]['count'] for cam in self.vehicle_data])
                    
                    avg_q = sum(queues) / len(queues) if queues else 0
                    avg_d = sum(densities) / len(densities) if densities else 0
                    avg_s = sum(speeds) / len(speeds) if speeds else 1.0
                    avg_w = avg_d * (1 - avg_s) * 120 # Uniform wait proxy
                
                else:
                    # Fallback to SUMO simulation metrics if YOLO is offline
                    queues, densities, wait_times = [], [], []
                    for lane in self.lanes:
                        q = self.conn.lane.getLastStepHaltingNumber(lane)
                        vc = self.conn.lane.getLastStepVehicleNumber(lane)
                        density = min(vc / 20.0, 1.0)
                        wait = self.conn.lane.getWaitingTime(lane) if hasattr(self.conn.lane, 'getWaitingTime') else density * 30
                        queues.append(q)
                        densities.append(density)
                        wait_times.append(wait)

                    avg_q = sum(queues) / len(queues) if queues else 0
                    avg_d = sum(densities) / len(densities) if densities else 0
                    avg_w = sum(wait_times) / len(wait_times) if wait_times else 0
                    avg_s = max(0.1, 1.0 - avg_d)
                    total_v = sum([self.conn.lane.getLastStepVehicleNumber(lane) for lane in self.lanes])

                with self.lock:
                    self.latest_queue = float(avg_q)
                    self.latest_density = float(avg_d * 100)
                    self.latest_wait = float(avg_w)
                    self.latest_speed = float(avg_s if self.vehicle_data is not None else max(0.1, 1.0 - avg_d))
                    self.latest_vehicle_count = int(total_v)
                    self.queue_history.append(round(avg_q, 2))
                    self.density_history.append(round(avg_d * 100, 1))
                    self.wait_time_history.append(round(avg_w, 1))
                    self.vehicle_count_history.append(total_v)

                # ==========================================
                # SIGNAL LOGIC (FIXED 60s CYCLIC TIMING)
                # ==========================================
                target_green_time = max(1, int(self.max_green_time / loop_sleep_s))

                if self.time_since_switch >= target_green_time and not is_yellow:
                    is_yellow = True
                    self.is_yellow = True
                    yellow_start = sim_step
                    self.yellow_start_time = time.time()
                    self.state_start_time = time.time()
                    yellow_phase_idx = 1 if self.current_phase == 0 else 3
                    self.conn.trafficlight.setPhase(self.tls_id, yellow_phase_idx)

                if is_yellow and (sim_step - yellow_start) >= yellow_duration_steps:
                    self.current_phase = 1 - self.current_phase
                    self.sub_phase = "straight"
                    green_phase_idx = 0 if self.current_phase == 0 else 2
                    self.conn.trafficlight.setPhase(self.tls_id, green_phase_idx)
                    self.time_since_switch = 0
                    is_yellow = False
                    self.is_yellow = False
                    self.state_start_time = time.time()
                    with self.lock:
                        self.phase_switches += 1
                    if hasattr(self, '_metrics_cb') and self._metrics_cb:
                        self._metrics_cb()

            self.conn.close()
            
        except Exception:
            pass
        finally:
            self.is_running = False
            if self.proc:
                try: self.proc.terminate()
                except: pass

    def get_metrics(self):
        with self.lock:
            avg_q = self.latest_queue if self.queue_history else 0
            avg_d = self.latest_density if self.density_history else 0
            avg_w = self.latest_wait if self.wait_time_history else 0
            avg_s = self.latest_speed if self.vehicle_count_history else 1.0
            total_v = self.latest_vehicle_count if self.vehicle_count_history else 0
            uptime = time.time() - self.start_time

            return {
                'name': 'fixed time',
                'active': self.is_running,
                'type': 'Static Baseline',
                'phase_switches': self.phase_switches,
                'uptime_seconds': round(uptime, 0),
                'total_vehicles': total_v,
                'realtime': {
                    'avg_queue_length': round(avg_q, 2),
                    'avg_density_pct': round(avg_d, 1),
                    'avg_speed_factor': round(max(0.1, avg_s), 2),
                    'est_wait_time_s': round(avg_w, 1),
                    'idle_emissions_factor': round((avg_d / 100.0) * (1.0 - max(0.1, avg_s)), 3),
                    'throughput_vpm': round(max(0.1, avg_s) * total_v * 3, 1),
                },
                'per_direction': {'N': 0, 'S': 0, 'E': 0, 'W': 0},
                'timeline': {
                    'queue': list(self.queue_history),
                    'density': list(self.density_history),
                    'wait_time': list(self.wait_time_history),
                    'vehicles': list(self.vehicle_count_history),
                    'throughput': [
                        round(max(0.1, 1.0 - (density / 100.0)) * vehicles * 3, 1)
                        for density, vehicles in zip(self.density_history, self.vehicle_count_history)
                    ],
                    'labels': [f'{i}s' for i in range(len(self.queue_history))]
                }
            }

if __name__ == "__main__":
    controller = StaticSUMOController(max_steps=1000)
    controller.run()
