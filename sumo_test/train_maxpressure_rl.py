import os
import sys
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import subprocess
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

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

# --- SUMO CONFIGURATION ---
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

class SumoEnv(gym.Env):
    def __init__(self, sumo_config="cross.sumocfg", gui=False, max_steps=3600,
                 port=None):
        super().__init__()

        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.sumo_config = sumo_config
        self.gui = gui
        self.port = port or random.randint(9000, 9999)

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        
        # Start SUMO
        cmd = [
            self.sumo_binary,
            "-c", self.sumo_config,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1"
        ]
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

        # Run simulation steps and accumulate reward
        reward = 0.0
        for _ in range(self.sim_step):
            traci.simulationStep()
            self.t += 1
            self.time_since_last_switch += 1
            reward -= self._calculate_pressure_reward()

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
        Multi-objective penalty that captures:
          1. Queue length per approach
          2. Total accumulated waiting time (normalised)
          3. Left-turn starvation bonus: extra penalty when left-turners
             have been waiting a long time without a protected phase.
        """
        total_queue = 0
        total_wait = 0
        max_veh_wait = 0.0

        for lane in self.lanes:
            total_queue += traci.lane.getLastStepHaltingNumber(lane)
            total_wait += traci.lane.getWaitingTime(lane)

        # Worst single-vehicle wait (detects left-turn starvation)
        for veh_id in traci.vehicle.getIDList():
            w = traci.vehicle.getWaitingTime(veh_id)
            if w > max_veh_wait:
                max_veh_wait = w

        # Weighted combination
        alpha = 0.5   # weight for total accumulated wait
        beta  = 0.02  # weight for max single-vehicle wait (starvation)
        penalty = (total_queue
                   + alpha * (total_wait / 60.0)
                   + beta  * (max_veh_wait / 60.0))
        return penalty

    def close(self):
        try:
            traci.close()
        except:
            pass
        if self.proc:
            self.proc.kill()


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

def linear_schedule(initial_lr: float, final_lr: float = 1e-5):
    """Returns a callable that linearly decays the learning rate."""
    def schedule(progress_remaining: float) -> float:
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule


def train_best_model():
    TOTAL_TIMESTEPS = 500_000

    # ------------------------------------------------------------------
    # 1. Vectorised training environment
    # ------------------------------------------------------------------
    env = DummyVecEnv([make_env()])
    # Normalise observations AND rewards (critical for stable PPO training)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ------------------------------------------------------------------
    # 2. Separate evaluation environment (must share norm stats)
    # ------------------------------------------------------------------
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/best_model',
        log_path='./logs/',
        eval_freq=3000,          # evaluate every 3 000 steps
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path='./logs/checkpoints/',
        name_prefix='ppo_traffic',
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 3. PPO – high-accuracy hyperparameters
    # ------------------------------------------------------------------
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        # Larger network: 512→512→256 for both actor and critic
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        # Linearly decaying LR from 3e-4 down to 1e-5
        learning_rate=linear_schedule(3e-4, 1e-5),
        n_steps=4096,           # larger rollout buffer → more stable gradients
        batch_size=256,         # larger mini-batches
        n_epochs=15,            # more optimisation passes per rollout
        gamma=0.995,            # long planning horizon (traffic needs lookahead)
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,         # mild entropy → exploit learned policy
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./traffic_tensorboard/",
    )

    print("Starting Optimised Training (1 000 000 steps) …")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
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