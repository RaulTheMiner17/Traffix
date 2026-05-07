import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import subprocess
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- SUMO CONFIGURATION ---
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

class SumoEnv(gym.Env):
    def __init__(self, sumo_config="cross.sumocfg", gui=True, max_steps=1000):
        super().__init__()
        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.sumo_config = sumo_config
        self.gui = gui
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(15,), dtype=np.float32)
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        cmd = [
            self.sumo_binary, "-c", self.sumo_config,
            "--no-step-log", "true", "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1"
        ]
        if self.gui:
            cmd.extend(["--quit-on-end", "--start"])

        self.proc = subprocess.Popen(
            cmd + ["--remote-port", "8873"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        import time
        for _ in range(5):
            try:
                time.sleep(1)
                traci.init(8873)
                break
            except:
                continue

        self.t = 0
        self.current_phase = 0
        self.time_since_last_switch = 0
        traci.trafficlight.setPhase(self.tls, 0)
        return self._get_obs(), {}

    def step(self, action):
        action = int(action)
        can_switch = self.time_since_last_switch >= self.min_green_time
        must_switch = self.time_since_last_switch >= self.max_green_time
        
        switch = False
        if must_switch and action == self.current_phase:
            switch = True
            action = 1 - self.current_phase
        elif can_switch and action != self.current_phase:
            switch = True
        
        if switch:
            self._do_yellow_phase()
            self.current_phase = action
            traci.trafficlight.setPhase(self.tls, 0 if self.current_phase == 0 else 2)
            self.time_since_last_switch = 0
        
        reward = 0
        for _ in range(self.sim_step):
            traci.simulationStep()
            self.t += 1
            self.time_since_last_switch += 1

        obs = self._get_obs()
        terminated = self.t >= self.max_steps
        return obs, reward, terminated, False, {}

    def _do_yellow_phase(self):
        traci.trafficlight.setPhase(self.tls, 1 if self.current_phase == 0 else 3)
        for _ in range(self.yellow_time):
            traci.simulationStep()
            self.t += 1

    def _get_obs(self):
        queues, densities, speeds = [], [], []
        for lane in self.lanes:
            queues.append(traci.lane.getLastStepHaltingNumber(lane))
            veh_count = traci.lane.getLastStepVehicleNumber(lane)
            densities.append(veh_count / (traci.lane.getLength(lane) / 5.0))
            speed_limit = traci.lane.getMaxSpeed(lane)
            speeds.append(traci.lane.getLastStepMeanSpeed(lane) / speed_limit if speed_limit > 0 else 0)

        phase_oh = [1.0, 0.0] if self.current_phase == 0 else [0.0, 1.0]
        time_norm = self.time_since_last_switch / self.max_green_time
        return np.concatenate([queues, densities, speeds, phase_oh, [time_norm]]).astype(np.float32)

    def close(self):
        try: traci.close()
        except: pass
        if self.proc: self.proc.kill()

def run_demo():
    print("\n" + "="*50)
    print("🚦 RUNNING AI SUMO CONTROLLER (HIGH SPEED)")
    print("="*50 + "\n")

    env = SumoEnv(gui=True, max_steps=1000)
    env = DummyVecEnv([lambda: env])
    
    try:
        env = VecNormalize.load("vec_normalize.pkl", env)
        env.training = False  
        env.norm_reward = False 
    except: pass

    try: model = PPO.load("ppo_traffic_final", env=env)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    obs = env.reset()
    last_print_step = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        current_step = env.envs[0].unwrapped.t
        
        if current_step - last_print_step >= 100:
            last_print_step = (current_step // 100) * 100
            print(f"[INFO] demo_model running... Step: {last_print_step}/1000")
            
        if done[0]:
            print(f"[INFO] demo_model running... Step: 1000/1000")
            print("[INFO] AI Simulation completed successfully.")
            break

if __name__ == "__main__":
    run_demo()