import os
import sys
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import subprocess
import tempfile
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

# ---------------------------------------------------------------------------
# Phase / action mapping
# 4 agent actions → 4 green SUMO phase indices (yellow = green_idx + 1)
# ---------------------------------------------------------------------------
# Action 0 → SUMO phase 0  : NS through + right (permissive left)
# Action 1 → SUMO phase 2  : NS protected left turn
# Action 2 → SUMO phase 4  : EW through + right (permissive left)
# Action 3 → SUMO phase 6  : EW protected left turn
ACTION_TO_SUMO_GREEN = {0: 0, 1: 2, 2: 4, 3: 6}
NUM_PHASES = 4

ROUTE_DEFINITIONS = [
    ("N2S", "N2J J2S"),
    ("S2N", "S2J J2N"),
    ("E2W", "E2J J2W"),
    ("W2E", "W2J J2E"),
    ("N2W_R", "N2J J2W"),
    ("S2E_R", "S2J J2E"),
    ("E2N_R", "E2J J2N"),
    ("W2S_R", "W2J J2S"),
    ("N2E_L", "N2J J2E"),
    ("S2W_L", "S2J J2W"),
    ("E2S_L", "E2J J2S"),
    ("W2N_L", "W2J J2N"),
]

ROUTE_GROUPS = {
    "ns_straight": ["N2S", "S2N"],
    "ew_straight": ["E2W", "W2E"],
    "ns_right": ["N2W_R", "S2E_R"],
    "ew_right": ["E2N_R", "W2S_R"],
    "ns_left": ["N2E_L", "S2W_L"],
    "ew_left": ["E2S_L", "W2N_L"],
}

# --- SUMO CONFIGURATION ---
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

class SumoEnv(gym.Env):
    def __init__(self, sumo_config="cross.sumocfg", gui=False, max_steps=3600,
                 port=None, randomize_routes=True):
        super().__init__()

        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.sumo_config = sumo_config
        if not os.path.isabs(self.sumo_config):
            self.sumo_config = os.path.join(self.base_dir, self.sumo_config)
        self.gui = gui
        self.port = port or random.randint(9000, 9999)
        self.randomize_routes = randomize_routes
        self.route_file = None

        # ------------------------------------------------------------------
        # Action space: 4 phases
        #   0 = NS through+right   (SUMO phase 0)
        #   1 = NS protected left  (SUMO phase 2)
        #   2 = EW through+right   (SUMO phase 4)
        #   3 = EW protected left  (SUMO phase 6)
        # ------------------------------------------------------------------
        self.action_space = spaces.Discrete(NUM_PHASES)

        # Observation (17 features):
        #  0-3  : Queue length (N, S, E, W)
        #  4-7  : Vehicle density (N, S, E, W)
        #  8-11 : Normalised avg speed (N, S, E, W)
        #  12-15: Current phase one-hot (4 bins)
        #  16   : Normalised time-since-last-switch
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(17,),
            dtype=np.float32
        )

        self.lanes = ["N2J_0", "S2J_0", "E2J_0", "W2J_0"]
        self.tls = "J0"
        self.sim_step = 5
        self.yellow_time = 3
        self.min_green_time = 10
        self.max_green_time = 60
        self.max_steps = max_steps

        self.t = 0
        self.proc = None
        self.current_phase = 0   # agent action index (0-3)
        self.time_since_last_switch = 0
        self.last_action = 0

    def _make_random_route_file(self):
        """Create one episode's demand profile so PPO does not overfit static timing."""
        scenario = random.choice(["balanced", "ns_peak", "ew_peak", "left_heavy", "bursty"])
        base = {
            "ns_straight": 0.12,
            "ew_straight": 0.12,
            "ns_right": 0.04,
            "ew_right": 0.04,
            "ns_left": 0.03,
            "ew_left": 0.03,
        }

        if scenario == "ns_peak":
            base["ns_straight"] *= random.uniform(1.5, 2.4)
            base["ns_right"] *= random.uniform(1.3, 2.0)
            base["ns_left"] *= random.uniform(1.3, 2.2)
            base["ew_straight"] *= random.uniform(0.45, 0.85)
        elif scenario == "ew_peak":
            base["ew_straight"] *= random.uniform(1.5, 2.4)
            base["ew_right"] *= random.uniform(1.3, 2.0)
            base["ew_left"] *= random.uniform(1.3, 2.2)
            base["ns_straight"] *= random.uniform(0.45, 0.85)
        elif scenario == "left_heavy":
            base["ns_left"] *= random.uniform(2.0, 3.5)
            base["ew_left"] *= random.uniform(2.0, 3.5)
        elif scenario == "bursty":
            axis = random.choice(["ns", "ew"])
            base[f"{axis}_straight"] *= random.uniform(2.0, 3.0)
            base[f"{axis}_right"] *= random.uniform(1.5, 2.5)
            base[f"{axis}_left"] *= random.uniform(1.5, 2.5)

        route_prob = {}
        for group, routes in ROUTE_GROUPS.items():
            for route in routes:
                route_prob[route] = min(base[group] * random.uniform(0.75, 1.25), 0.45)

        fd, path = tempfile.mkstemp(prefix="traffix_routes_", suffix=".rou.xml", dir=self.base_dir, text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
            f.write('  <vType id="car" accel="2.0" decel="4.5" sigma="0.5" length="5.0" maxSpeed="13.9"/>\n\n')
            for route_id, edges in ROUTE_DEFINITIONS:
                f.write(f'  <route id="{route_id}" edges="{edges}"/>\n')
            f.write("\n")
            for route_id, prob in route_prob.items():
                f.write(
                    f'  <flow id="f_{route_id}" type="car" route="{route_id}" '
                    f'begin="0" end="{self.max_steps}" probability="{prob:.4f}"/>\n'
                )
            f.write("</routes>\n")
        return path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        if self.randomize_routes:
            self.route_file = self._make_random_route_file()
        
        # Start SUMO
        cmd = [
            self.sumo_binary,
            "-c", self.sumo_config,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1"
        ]
        if self.route_file:
            cmd.extend(["--route-files", self.route_file])
        if self.gui:
            cmd.append("--quit-on-end")
            cmd.append("--start")

        self.proc = subprocess.Popen(
            cmd + ["--remote-port", str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Retry connection with back-off
        for attempt in range(10):
            try:
                time.sleep(0.5 + attempt * 0.3)
                traci.init(self.port)
                break
            except Exception:
                continue

        self.t = 0
        self.current_phase = 0  # start with NS through phase
        self.time_since_last_switch = 0
        self.last_action = 0

        traci.trafficlight.setPhase(self.tls, ACTION_TO_SUMO_GREEN[0])

        return self._get_obs(), {}

    def step(self, action):
        action = int(action)

        can_switch = self.time_since_last_switch >= self.min_green_time
        must_switch = self.time_since_last_switch >= self.max_green_time

        switch = False
        switch_penalty = 0.0
        if must_switch and action == self.current_phase:
            # Forced rotation to prevent starvation
            switch = True
            action = (self.current_phase + 1) % NUM_PHASES
        elif can_switch and action != self.current_phase:
            switch = True

        if switch:
            self._do_yellow_phase()
            self.current_phase = action
            traci.trafficlight.setPhase(self.tls, ACTION_TO_SUMO_GREEN[self.current_phase])
            self.time_since_last_switch = 0
            switch_penalty = 0.25

        # Run simulation steps and accumulate reward
        reward = 0.0
        for _ in range(self.sim_step):
            traci.simulationStep()
            self.t += 1
            self.time_since_last_switch += 1
            reward -= self._calculate_pressure_reward()
            reward += 0.08 * traci.simulation.getArrivedNumber()

        reward -= switch_penalty

        obs = self._get_obs()
        terminated = self.t >= self.max_steps
        truncated = False
        self.last_action = action

        return obs, reward, terminated, truncated, {}

    def _do_yellow_phase(self):
        # Yellow phase index = green phase index + 1
        yellow_sumo_idx = ACTION_TO_SUMO_GREEN[self.current_phase] + 1
        traci.trafficlight.setPhase(self.tls, yellow_sumo_idx)
        for _ in range(self.yellow_time):
            traci.simulationStep()
            self.t += 1

    def _get_obs(self):
        queues = []
        densities = []
        speeds = []

        for lane in self.lanes:
            # Queue: halting vehicles
            q = traci.lane.getLastStepHaltingNumber(lane)
            queues.append(float(q))

            # Density: vehicles per unit length
            veh_count = traci.lane.getLastStepVehicleNumber(lane)
            lane_len = traci.lane.getLength(lane)
            densities.append(veh_count / max(lane_len / 5.0, 1.0))

            # Normalised average speed
            avg_speed = traci.lane.getLastStepMeanSpeed(lane)
            speed_limit = traci.lane.getMaxSpeed(lane)
            speeds.append(avg_speed / speed_limit if speed_limit > 0 else 0.0)

        # Current phase one-hot (4 bins)
        phase_oh = [0.0] * NUM_PHASES
        phase_oh[self.current_phase] = 1.0

        # Normalised time in current phase
        time_norm = min(self.time_since_last_switch / self.max_green_time, 1.0)

        obs = np.concatenate([
            queues,       # 4
            densities,    # 4
            speeds,       # 4
            phase_oh,     # 4
            [time_norm],  # 1
        ])
        return obs.astype(np.float32)

    def _calculate_pressure_reward(self):
        """
        Multi-objective penalty aligned with the dashboard metrics:
          1. Average queue length per approach
          2. Average waiting time per approach
          3. Density pressure
          4. Left-turn starvation protection
        """
        total_queue = 0
        total_wait = 0
        total_density = 0.0
        max_veh_wait = 0.0
        lane_queues = {}
        lane_waits = {}

        for lane in self.lanes:
            queue = traci.lane.getLastStepHaltingNumber(lane)
            wait = traci.lane.getWaitingTime(lane)
            lane_queues[lane] = queue
            lane_waits[lane] = wait
            total_queue += queue
            total_wait += wait
            veh_count = traci.lane.getLastStepVehicleNumber(lane)
            lane_len = traci.lane.getLength(lane)
            total_density += veh_count / max(lane_len / 5.0, 1.0)

        # Worst single-vehicle wait (detects left-turn starvation)
        for veh_id in traci.vehicle.getIDList():
            w = traci.vehicle.getWaitingTime(veh_id)
            if w > max_veh_wait:
                max_veh_wait = w

        avg_queue = total_queue / len(self.lanes)
        avg_wait = total_wait / len(self.lanes)
        avg_density = total_density / len(self.lanes)
        if self.current_phase in (0, 1):
            red_lanes = ["E2J_0", "W2J_0"]
        else:
            red_lanes = ["N2J_0", "S2J_0"]
        red_queue = sum(lane_queues[lane] for lane in red_lanes) / len(red_lanes)
        red_wait = sum(lane_waits[lane] for lane in red_lanes) / len(red_lanes)

        # Weighted combination.
        # Smaller values are better, so the PPO policy is trained to reduce
        # the same traffic indicators the dashboard displays.
        penalty = (
            0.55 * avg_queue
            + 0.25 * (avg_wait / 60.0)
            + 0.15 * avg_density
            + 0.18 * red_queue
            + 0.08 * (red_wait / 60.0)
            + 0.04 * (max_veh_wait / 60.0)
        )
        return penalty

    def close(self):
        try:
            traci.close()
        except:
            pass
        if self.proc:
            self.proc.kill()
            self.proc = None
        if self.route_file and os.path.exists(self.route_file):
            try:
                os.remove(self.route_file)
            except OSError:
                pass
            self.route_file = None


# ===============================================================
#                       OPTIMIZED TRAINING
# ===============================================================

def make_env(port=None):
    """Utility to create and wrap the environment."""
    def _init():
        _port = port or random.randint(9000, 9999)
        env = SumoEnv(gui=False, max_steps=3600, port=_port)
        return Monitor(env)
    return _init

def get_num_train_envs():
    requested = os.environ.get("TRAFFIX_NUM_ENVS")
    if requested:
        return max(1, int(requested))

    cpu_count = os.cpu_count() or 1
    # SUMO is CPU-bound. Use nearly the whole machine, leaving one core for
    # Windows, the terminal, and file IO so training does not choke itself.
    return max(1, cpu_count - 1)

def get_train_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def cleanup_temp_route_files():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in os.listdir(base_dir):
        if fname.startswith("traffix_routes_") and fname.endswith(".rou.xml"):
            try:
                os.remove(os.path.join(base_dir, fname))
            except OSError:
                pass

def linear_schedule(initial_lr: float, final_lr: float = 1e-5):
    """Returns a callable that linearly decays the learning rate."""
    def schedule(progress_remaining: float) -> float:
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule

def latest_checkpoint(checkpoint_dir="./logs/checkpoints"):
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for fname in os.listdir(checkpoint_dir):
        if not fname.startswith("ppo_traffic_") or not fname.endswith("_steps.zip"):
            continue
        try:
            steps = int(fname.replace("ppo_traffic_", "").replace("_steps.zip", ""))
        except ValueError:
            continue
        candidates.append((steps, os.path.join(checkpoint_dir, fname)))
    return max(candidates, default=(None, None))[1]


def train_best_model():
    TOTAL_TIMESTEPS = 2_000_000
    num_envs = get_num_train_envs()
    train_device = get_train_device()
    base_port = 9100
    rollout_steps = 1024 if num_envs > 1 else 4096
    batch_size = 512 if num_envs > 1 else 256
    n_epochs = 10 if num_envs > 1 else 15
    cleanup_temp_route_files()

    if train_device == "cuda":
        torch.set_num_threads(1)
    else:
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // max(num_envs, 1)))

    print(
        f"Training with {num_envs} parallel SUMO workers on device={train_device} "
        f"(n_steps={rollout_steps}, batch_size={batch_size}, n_epochs={n_epochs})"
    )

    # ------------------------------------------------------------------
    # 1. Vectorised training environment
    # ------------------------------------------------------------------
    env_fns = [make_env(port=base_port + i) for i in range(num_envs)]
    env = SubprocVecEnv(env_fns, start_method="spawn") if num_envs > 1 else DummyVecEnv(env_fns)
    norm_path = "vec_normalize.pkl"
    if os.path.exists(norm_path):
        print(f"Loading existing normalization stats from {norm_path}")
        env = VecNormalize.load(norm_path, env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ------------------------------------------------------------------
    # 2. Separate evaluation environment (must share norm stats)
    # ------------------------------------------------------------------
    eval_env = DummyVecEnv([make_env()])
    if os.path.exists(norm_path):
        eval_env = VecNormalize.load(norm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                                clip_obs=10.0, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/best_model',
        log_path='./logs/',
        eval_freq=max(3000 // num_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(25_000 // num_envs, 1),
        save_path='./logs/checkpoints/',
        name_prefix='ppo_traffic',
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 3. PPO – high-accuracy hyperparameters
    # ------------------------------------------------------------------
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
    )

    checkpoint_path = latest_checkpoint()
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}")
        model = PPO.load(
            checkpoint_path,
            env=env,
            tensorboard_log="./traffic_tensorboard/",
            device=train_device,
            custom_objects={
                "n_steps": rollout_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
            },
        )
        # Keep some learning signal so the policy can move past the plateau.
        fine_tune_lr = linear_schedule(5e-5, 5e-6)
        model.learning_rate = fine_tune_lr
        if hasattr(model, "lr_schedule"):
            model.lr_schedule = fine_tune_lr
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=linear_schedule(3e-4, 1e-5),
            n_steps=rollout_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=train_device,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./traffic_tensorboard/",
        )

    print("Starting Training…")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_callback, checkpoint_callback],
            tb_log_name="PPO_Phase2",
            reset_num_timesteps=False,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted – saving current model.")

    model.save("ppo_traffic_final")
    env.save("vec_normalize.pkl")
    print("Training complete. Model and normalisation stats saved.")
    env.close()

def test_best_model():
    """Run a visual demo with the best saved model."""
    inner_env = SumoEnv(gui=True, max_steps=1000)
    env = DummyVecEnv([lambda: Monitor(inner_env)])

    # IMPORTANT: Load normalisation statistics from training.
    # Without this the agent sees completely different input distributions.
    env = VecNormalize.load("vec_normalize.pkl", env)
    env.training   = False   # freeze running stats
    env.norm_reward = False  # display raw reward

    model = PPO.load("./logs/best_model/best_model", env=env)

    obs = env.reset()
    total_reward = 0.0
    step_count = 0
    print("Running demo (close SUMO window to stop) …")
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        step_count += 1
        if step_count % 100 == 0:
            print(f"  step={step_count:>5}  cumulative_reward={total_reward:.1f}")
        time.sleep(0.05)

if __name__ == "__main__":
    os.makedirs("./logs/best_model", exist_ok=True)
    os.makedirs("./logs/checkpoints", exist_ok=True)

    # --- Train ---
    train_best_model()

    # --- Test (uncomment after training) ---
    # test_best_model()
