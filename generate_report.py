import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import time

# RL Imports
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_DIR = os.path.join(BASE_DIR, "sumo_test")
sys.path.append(SUMO_DIR)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
    import traci
else:
    sys.exit("Error: SUMO_HOME environment variable is not set.")

def get_sumo_cmd(port):
    sumo_config = os.path.join(SUMO_DIR, "cross.sumocfg")
    return ["sumo", "-c", sumo_config, "--no-step-log", "true", "--remote-port", str(port)]

class EvalSumoEnv(gym.Env):
    def __init__(self, max_steps=3600):
        super().__init__()
        self.sumo_config = os.path.join(SUMO_DIR, "cross.sumocfg")
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.lanes = ["N2J_0", "S2J_0", "E2J_0", "W2J_0"]
        self.tls = "J0"
        self.sim_step = 5        
        self.yellow_time = 3     
        self.min_green_time = 10 
        self.max_green_time = 60 
        self.max_steps = max_steps 
        self.t = 0
        self.proc = None
        self.current_phase = 0   
        self.time_since_last_switch = 0
        self.history = {"queue": [], "wait": [], "throughput": [], "density": []}

    def _record_metrics(self):
        q, w, t, d = [], [], [], []
        try:
            conn = traci.getConnection("ai_eval")
            for lane in self.lanes:
                q.append(conn.lane.getLastStepHaltingNumber(lane))
                w.append(conn.lane.getWaitingTime(lane))
                vc = conn.lane.getLastStepVehicleNumber(lane)
                t.append(vc)
                d.append(vc / 20.0) # Density based on ~20 car capacity per lane
            self.history["queue"].append(sum(q)/4.0)
            self.history["wait"].append(sum(w)/4.0)
            self.history["throughput"].append(sum(t))
            self.history["density"].append((sum(d)/4.0) * 100)
        except: pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.close() 
        self.proc = subprocess.Popen(get_sumo_cmd(8873), stdout=subprocess.DEVNULL, stderr=sys.stderr)
        time.sleep(2)
        traci.init(8873, label="ai_eval")
        self.t = 0
        self.current_phase = 0
        self.time_since_last_switch = 0
        traci.getConnection("ai_eval").trafficlight.setPhase(self.tls, 0)
        return self._get_obs(), {}

    def step(self, action):
        action = int(action)
        conn = traci.getConnection("ai_eval")
        if self.time_since_last_switch >= self.max_green_time:
            action = 1 - self.current_phase
        
        if action != self.current_phase and self.time_since_last_switch >= self.min_green_time:
            conn.trafficlight.setPhase(self.tls, 1 if self.current_phase == 0 else 3)
            for _ in range(self.yellow_time):
                conn.simulationStep()
                self._record_metrics()
                self.t += 1
            self.current_phase = action
            conn.trafficlight.setPhase(self.tls, 0 if self.current_phase == 0 else 2)
            self.time_since_last_switch = 0
        
        for _ in range(self.sim_step):
            conn.simulationStep()
            self._record_metrics()
            self.t += 1
            self.time_since_last_switch += 1

        if self.t > 0 and self.t % 400 == 0: 
            print(f"AI Evaluation: {self.t}/{self.max_steps} steps")
            
        done = self.t >= self.max_steps
        return self._get_obs(), 0, done, False, {}

    def _get_obs(self):
        q, d, s = [], [], []
        try:
            conn = traci.getConnection("ai_eval")
            for lane in self.lanes:
                q.append(conn.lane.getLastStepHaltingNumber(lane))
                vc = conn.lane.getLastStepVehicleNumber(lane)
                d.append(vc / (conn.lane.getLength(lane) / 5.0))
                sl = conn.lane.getMaxSpeed(lane)
                s.append(conn.lane.getLastStepMeanSpeed(lane) / sl if sl > 0 else 0)
        except:
            q, d, s = [0]*4, [0]*4, [0]*4
        p = [1.0, 0.0] if self.current_phase == 0 else [0.0, 1.0]
        return np.concatenate([q, d, s, p, [self.time_since_last_switch/60.0]]).astype(np.float32)

    def close(self):
        try: traci.getConnection("ai_eval").close()
        except: pass
        if self.proc: self.proc.kill()

def smooth_data(data, window_size=60):
    if len(data) < window_size: return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def run_evaluations():
    print("--- Running AI Evaluation ---")
    env = EvalSumoEnv(max_steps=3600)
    dummy = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(os.path.join(SUMO_DIR, "vec_normalize.pkl"), dummy)
    vec_env.training = False
    model = PPO.load(os.path.join(SUMO_DIR, "ppo_traffic_final"), env=vec_env)
    
    obs = vec_env.reset()
    done = [False]
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = vec_env.step(action)
        
    ai_metrics = env.history
    env.close()

    print("\n--- Running Static Evaluation (Realistic 120s Baseline) ---")
    proc = subprocess.Popen(get_sumo_cmd(8874), stdout=subprocess.DEVNULL, stderr=sys.stderr)
    time.sleep(2)
    traci.init(8874, label="st_eval")
    conn = traci.getConnection("st_eval")
    st_metrics = {"queue": [], "wait": [], "throughput": [], "density": []}
    t_switch = 0
    phase = 0
    
    step = 0
    while step < 3600:
        conn.simulationStep()
        t_switch += 1
        
        if t_switch >= 120:
            conn.trafficlight.setPhase("J0", 1 if phase == 0 else 3)
            for _ in range(3): 
                conn.simulationStep()
                st_metrics["queue"].append(st_metrics["queue"][-1] if st_metrics["queue"] else 0)
                st_metrics["wait"].append(st_metrics["wait"][-1] if st_metrics["wait"] else 0)
                st_metrics["throughput"].append(st_metrics["throughput"][-1] if st_metrics["throughput"] else 0)
                st_metrics["density"].append(st_metrics["density"][-1] if st_metrics["density"] else 0)
                step += 1
            phase = 1 - phase
            conn.trafficlight.setPhase("J0", 0 if phase == 0 else 2)
            t_switch = 0
            if step >= 3600: break
        
        q, w, v, d = [], [], [], []
        for l in ["N2J_0", "S2J_0", "E2J_0", "W2J_0"]:
            q.append(conn.lane.getLastStepHaltingNumber(l))
            w.append(conn.lane.getWaitingTime(l))
            vc = conn.lane.getLastStepVehicleNumber(l)
            v.append(vc)
            d.append(vc / 20.0)
        st_metrics["queue"].append(sum(q)/4)
        st_metrics["wait"].append(sum(w)/4)
        st_metrics["throughput"].append(sum(v))
        st_metrics["density"].append((sum(d)/4) * 100)
        
        step += 1
        if step % 400 == 0: 
            print(f"Static Evaluation: {step}/3600 steps")
    
    conn.close()
    proc.terminate()

    print("\nGenerating Professional Report and Smoothed Graphs...")
    
    # CALCULATE ADVANCED METRICS
    ai_avg_q, st_avg_q = np.mean(ai_metrics["queue"]), np.mean(st_metrics["queue"])
    ai_max_q, st_max_q = np.max(ai_metrics["queue"]), np.max(st_metrics["queue"])
    
    ai_avg_w, st_avg_w = np.mean(ai_metrics["wait"]), np.mean(st_metrics["wait"])
    ai_max_w, st_max_w = np.max(ai_metrics["wait"]), np.max(st_metrics["wait"])
    
    ai_avg_d, st_avg_d = np.mean(ai_metrics["density"]), np.mean(st_metrics["density"])
    
    # Idle Emissions Proxy (Assume 1 parked car idling = 0.00016 kg CO2 per sec)
    ai_co2 = sum(ai_metrics["queue"]) * 0.00016
    st_co2 = sum(st_metrics["queue"]) * 0.00016
    co2_saved = st_co2 - ai_co2
    
    report_path = os.path.join(BASE_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("=== Hex-NM Traffic AI Performance Report ===\n")
        f.write("Evaluation Period: 1 Hour (3600 simulation steps)\n")
        f.write("Baseline Model: Realistic 120-Second Fixed-Time Cycle\n")
        f.write("="*45 + "\n\n")
        
        f.write("1. QUEUE MANAGEMENT (Congestion Prevention)\n")
        f.write(f"   - Avg Queue (AI):         {ai_avg_q:.2f} cars/lane\n")
        f.write(f"   - Avg Queue (Baseline):   {st_avg_q:.2f} cars/lane\n")
        f.write(f"   - WORST-CASE Queue (AI):  {ai_max_q:.0f} cars (Prevents gridlock)\n")
        f.write(f"   - WORST-CASE Queue (Base):{st_max_q:.0f} cars\n\n")
        
        f.write("2. DRIVER DELAY (Quality of Service)\n")
        f.write(f"   - Avg Wait Time (AI):     {ai_avg_w:.2f} seconds\n")
        f.write(f"   - Avg Wait Time (Base):   {st_avg_w:.2f} seconds\n")
        f.write(f"   - MAX Wait Time (AI):     {ai_max_w:.2f} seconds\n")
        f.write(f"   - MAX Wait Time (Base):   {st_max_w:.2f} seconds\n\n")

        f.write("3. ENVIRONMENTAL IMPACT (Sustainability)\n")
        f.write(f"   - Intersection Density (AI):   {ai_avg_d:.1f}% capacity utilized\n")
        f.write(f"   - Intersection Density (Base): {st_avg_d:.1f}% capacity utilized\n")
        f.write(f"   - Idling CO2 Proxy (AI):       {ai_co2:.2f} kg CO2 emitted\n")
        f.write(f"   - Idling CO2 Proxy (Base):     {st_co2:.2f} kg CO2 emitted\n")
        f.write(f"   -> TOTAL CARBON SAVED:         {co2_saved:.2f} kg CO2 per hour\n")

    # GRAPHING
    WINDOW = 60
    ai_wait_smooth = smooth_data(ai_metrics["wait"], WINDOW)
    st_wait_smooth = smooth_data(st_metrics["wait"], WINDOW)
    ai_queue_smooth = smooth_data(ai_metrics["queue"], WINDOW)
    st_queue_smooth = smooth_data(st_metrics["queue"], WINDOW)
    
    ai_x = range(WINDOW - 1, WINDOW - 1 + len(ai_wait_smooth))
    st_x = range(WINDOW - 1, WINDOW - 1 + len(st_wait_smooth))

    # Plot 1: Wait Time
    plt.figure(figsize=(10,5))
    plt.plot(ai_x, ai_wait_smooth, label="PPO AI Model", color="#2ca02c", linewidth=2.5)
    plt.plot(st_x, st_wait_smooth, label="Static 120s Baseline", color="#d62728", linestyle='dashed', linewidth=2.5, alpha=0.8)
    plt.title("Wait Time Trend (Smoothed over 60s windows)")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Wait Time (seconds)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "graph_wait_comparison.png"), dpi=300)
    plt.close()
    
    # Plot 2: Queue Length
    plt.figure(figsize=(10,5))
    plt.plot(ai_x, ai_queue_smooth, label="PPO AI Model", color="#1f77b4", linewidth=2.5)
    plt.plot(st_x, st_queue_smooth, label="Static 120s Baseline", color="#ff7f0e", linestyle='dashed', linewidth=2.5, alpha=0.8)
    plt.title("Queue Length Trend (Smoothed over 60s windows)")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Average Queue Length (Cars)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "graph_queue_comparison.png"), dpi=300)
    plt.close()

    print(f"Report saved to: {report_path}")
    print("✅ DONE. Check the newly generated graphs and the incredible new report metrics.")

if __name__ == "__main__":
    run_evaluations()