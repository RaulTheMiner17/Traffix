"""
plot_results.py
===============
Compares two PPO models by:
  1. Plotting the training reward curve from evaluations.npz
  2. Running both models head-to-head in the SUMO environment
  3. Generating comparison charts (bar + line)

Models:
  - "Final"  (ppo_traffic_final.zip)          — the last checkpoint (1M steps)
  - "Best"   (logs/best_model/best_model.zip) — saved by EvalCallback at peak

Output: ./plots/
"""

import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FINAL_MODEL = os.path.join(BASE_DIR, "logs", "checkpoints", "ppo_traffic_3000000_steps.zip")
BEST_MODEL  = os.path.join(BASE_DIR, "logs", "best_model", "best_model.zip")
EVAL_NPZ    = os.path.join(BASE_DIR, "logs", "evaluations.npz")
VEC_NORM    = os.path.join(BASE_DIR, "vec_normalize.pkl")
OUT_DIR     = os.path.join(BASE_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "final": "#E07B39",   # orange
    "best":  "#2E8BC0",   # blue
    "green": "#22c55e",
}

def smooth(values, weight=0.85):
    """Exponential moving average."""
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return np.array(smoothed)

def fmt_ts(ts):
    if ts is None: return "??"
    if ts >= 1000000: return f"{ts/1000000:.1f}M".replace(".0M", "M")
    if ts >= 1000: return f"{ts/1000:.0f}k"
    return str(ts)


# ═════════════════════════════════════════════════════════════════════════════
# PART 1: Training curve from evaluations.npz
# ═════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Plotting training reward curve ...")
best_ts = None
final_ts = None
best_val = None

if os.path.isfile(EVAL_NPZ):
    data      = np.load(EVAL_NPZ)
    timesteps = data["timesteps"]
    results   = data["results"]
    mean_rew  = results.mean(axis=1)
    std_rew   = results.std(axis=1)
    best_idx  = int(np.argmax(mean_rew))
    best_ts   = timesteps[best_idx]
    final_ts  = timesteps[-1]
    best_val  = mean_rew[best_idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timesteps, smooth(mean_rew), color=COLORS["best"],
            linewidth=2, label="Smoothed mean reward", zorder=3)
    ax.fill_between(timesteps, mean_rew - std_rew, mean_rew + std_rew,
                    color=COLORS["best"], alpha=0.15, label="+-1 std")
    ax.plot(timesteps, mean_rew, color=COLORS["best"], alpha=0.2, linewidth=0.6)

    # Best checkpoint marker
    ax.axvline(best_ts, color=COLORS["green"], linestyle="--", linewidth=1.5,
               label=f"Optimal Stop Point @ step {best_ts:,}")
    ax.scatter([best_ts], [best_val], color=COLORS["green"], s=80, zorder=5)
    ax.annotate(f"Optimal Stop:\n{best_val:.0f} @ step {best_ts:,}",
                xy=(best_ts, best_val), xytext=(15, 15),
                textcoords="offset points", fontsize=9, fontweight="bold",
                color=COLORS["green"],
                arrowprops=dict(arrowstyle="->", color=COLORS["green"]))

    # Final model marker
    final_val = mean_rew[-1]
    final_ts  = timesteps[-1]
    ax.scatter([final_ts], [final_val], color=COLORS["final"], s=80, zorder=5, marker="D")
    ax.annotate(f"Final: {final_val:.0f}\n@ step {final_ts:,}",
                xy=(final_ts, final_val), xytext=(15, -25),
                textcoords="offset points", fontsize=9, fontweight="bold",
                color=COLORS["final"],
                arrowprops=dict(arrowstyle="->", color=COLORS["final"]))

    ax.set_title("PPO Training Curve - Evaluation Reward Over Time",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "01_training_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")
else:
    print("  [SKIP] evaluations.npz not found")


# ═════════════════════════════════════════════════════════════════════════════
# PART 2: Head-to-head SUMO simulation
# ═════════════════════════════════════════════════════════════════════════════
print("\n[2/3] Running head-to-head comparison in SUMO ...")

# Check SUMO
if "SUMO_HOME" not in os.environ:
    print("  [SKIP] SUMO_HOME not set — cannot run simulation comparison.")
    sim_results = None
else:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
    import traci

    from stable_baselines3 import PPO as PPOLoader
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor

    # Import env from training script
    sys.path.insert(0, BASE_DIR)
    from train_maxpressure_rl import SumoEnv

    def evaluate_model(model_path, label, n_episodes=5):
        """Run a model for n_episodes, collect per-step metrics."""
        print(f"    Evaluating: {label} ({os.path.basename(model_path)})")
        model = PPOLoader.load(model_path)

        all_rewards = []
        all_queues  = []
        all_waits   = []

        for ep in range(n_episodes):
            port = random.randint(10000, 19999)
            inner_env = SumoEnv(gui=False, max_steps=3600, port=port)
            env = DummyVecEnv([lambda: Monitor(inner_env)])
            
            if os.path.exists(VEC_NORM):
                env = VecNormalize.load(VEC_NORM, env)
                env.training = False
                env.norm_reward = False
            else:
                print("      [WARNING] vec_normalize.pkl not found. Results will be inaccurate.")

            obs = env.reset()

            ep_reward = 0
            ep_queues = []
            ep_waits  = []
            done = False
            step = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done_arr, info = env.step(action)
                done = done_arr[0]
                ep_reward += float(reward[0])

                # Get unnormalized observation directly from the inner environment
                raw_obs = inner_env._get_obs()

                # Collect queue lengths from observation (indices 0-3)
                ep_queues.append(raw_obs[0:4].mean())
                # Collect density from observation (indices 4-7)
                ep_waits.append(raw_obs[4:8].mean())
                step += 1

            all_rewards.append(ep_reward)
            all_queues.append(np.mean(ep_queues))
            all_waits.append(np.mean(ep_waits))
            env.close()
            print(f"      Episode {ep+1}/{n_episodes}: reward={ep_reward:.0f}, "
                  f"avg_queue={np.mean(ep_queues):.2f}, avg_density={np.mean(ep_waits):.3f}")

        return {
            "label": label,
            "rewards": all_rewards,
            "avg_queues": all_queues,
            "avg_densities": all_waits,
            "mean_reward": np.mean(all_rewards),
            "mean_queue": np.mean(all_queues),
            "mean_density": np.mean(all_waits),
        }

    sim_results = {}
    if os.path.isfile(BEST_MODEL):
        sim_results["best"] = evaluate_model(BEST_MODEL, "Best Model (EvalCallback)")
    else:
        print(f"  [SKIP] Best model not found: {BEST_MODEL}")

    if os.path.isfile(FINAL_MODEL):
        sim_results["final"] = evaluate_model(FINAL_MODEL, f"Final Model ({fmt_ts(final_ts)} steps)")
    else:
        print(f"  [SKIP] Final model not found: {FINAL_MODEL}")


# ═════════════════════════════════════════════════════════════════════════════
# PART 3: Comparison plots
# ═════════════════════════════════════════════════════════════════════════════
print("\n[3/3] Generating comparison charts ...")

if sim_results and len(sim_results) == 2:
    best_r  = sim_results["best"]
    final_r = sim_results["final"]

    # ── 3a. Bar chart comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("mean_reward",  "Mean Episode Reward",  "higher is better"),
        ("mean_queue",   "Mean Queue Length",     "lower is better"),
        ("mean_density", "Mean Density",          "lower is better"),
    ]

    for ax, (key, title, hint) in zip(axes, metrics):
        vals = [best_r[key], final_r[key]]
        bars = ax.bar([f"Best\n(step {fmt_ts(best_ts)})", f"Final\n(step {fmt_ts(final_ts)})"],
                      vals, color=[COLORS["best"], COLORS["final"]],
                      width=0.5, edgecolor="white", linewidth=1.5)


        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

        # Winner highlight
        if key == "mean_reward":
            winner = 0 if vals[0] > vals[1] else 1
        else:
            winner = 0 if vals[0] < vals[1] else 1
        bars[winner].set_edgecolor(COLORS["green"])
        bars[winner].set_linewidth(3)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.text(0.98, 0.02, hint, transform=ax.transAxes,
                fontsize=8, color="grey", ha="right", va="bottom")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Head-to-Head: Best Model vs Final Model",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "02_bar_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    # ── 3b. Episode-by-episode reward comparison ──
    fig, ax = plt.subplots(figsize=(10, 5))
    eps = range(1, len(best_r["rewards"]) + 1)
    ax.plot(eps, best_r["rewards"],  "o-", color=COLORS["best"],
            linewidth=2, markersize=8, label=f"Best (mean={best_r['mean_reward']:.0f})")
    ax.plot(eps, final_r["rewards"], "D-", color=COLORS["final"],
            linewidth=2, markersize=8, label=f"Final (mean={final_r['mean_reward']:.0f})")
    ax.axhline(best_r["mean_reward"],  color=COLORS["best"],  linestyle="--", alpha=0.5)
    ax.axhline(final_r["mean_reward"], color=COLORS["final"], linestyle="--", alpha=0.5)
    ax.set_title("Episode Rewards: Best vs Final Model", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Episode Reward", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "03_episode_rewards.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    # ── 3c. Combined dashboard ──
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("PPO Traffic Model Comparison Dashboard",
                 fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Training curve
    if os.path.isfile(EVAL_NPZ):
        ax1 = fig.add_subplot(gs[0, :])
        data = np.load(EVAL_NPZ)
        ts = data["timesteps"]
        mr = data["results"].mean(axis=1)
        ax1.plot(ts, smooth(mr), color=COLORS["best"], linewidth=2)
        ax1.fill_between(ts, mr - data["results"].std(axis=1),
                         mr + data["results"].std(axis=1),
                         color=COLORS["best"], alpha=0.1)
        if best_ts is not None:
            ax1.axvline(best_ts, color=COLORS["green"], linestyle="--", linewidth=1.5,
                        label=f"Optimal Stop Point ({fmt_ts(best_ts)})")
        ax1.set_title("Training Reward Curve", fontsize=12, fontweight="bold")

        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Mean Reward")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

    # Panel 2: Bar comparison
    ax2 = fig.add_subplot(gs[1, 0])
    vals = [best_r["mean_reward"], final_r["mean_reward"]]
    bars = ax2.bar([f"Best\n({fmt_ts(best_ts)})", f"Final\n({fmt_ts(final_ts)})"], vals,
                   color=[COLORS["best"], COLORS["final"]], width=0.5)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{v:.0f}", ha="center", va="bottom", fontweight="bold")
    ax2.set_title("Mean Reward Comparison", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: Queue comparison
    ax3 = fig.add_subplot(gs[1, 1])
    vals_q = [best_r["mean_queue"], final_r["mean_queue"]]
    bars_q = ax3.bar([f"Best\n({fmt_ts(best_ts)})", f"Final\n({fmt_ts(final_ts)})"], vals_q,
                     color=[COLORS["best"], COLORS["final"]], width=0.5)
    for bar, v in zip(bars_q, vals_q):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
    ax3.set_title("Mean Queue Length Comparison", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # Summary text
    diff_pct = ((best_r["mean_reward"] - final_r["mean_reward"])
                / abs(final_r["mean_reward"]) * 100)
    summary = (f"Best model outperforms Final by {abs(diff_pct):.1f}%  |  "
               f"Best reward: {best_r['mean_reward']:.0f}  |  "
               f"Final reward: {final_r['mean_reward']:.0f}")
    fig.text(0.5, 0.01, summary, ha="center", fontsize=11, fontweight="bold",
             color=COLORS["green"] if diff_pct > 0 else COLORS["final"],
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", alpha=0.9))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = os.path.join(OUT_DIR, "00_dashboard.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

elif sim_results:
    print("  [SKIP] Need both models for comparison (only found one).")
else:
    # No SUMO — still generate the training curve plot if we have eval data
    print("  [SKIP] No simulation results to plot.")

print(f"\nDone! All plots saved to: {OUT_DIR}")
print("Files generated:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
