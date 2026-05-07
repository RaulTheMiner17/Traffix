"""
plot_results.py
===============
Generates PNG comparison charts between:
  - Old PPO run  (ppo_traffic_tensorboard/)   – 2-phase: NS / EW only
  - New PPO run  (traffic_tensorboard/)        – 4-phase: NS-str, NS-left, EW-str, EW-left

Also reads logs/evaluations.npz produced by EvalCallback.

Output files (saved to ./plots/):
  01_reward_comparison.png
  02_policy_loss_comparison.png
  03_value_loss_comparison.png
  04_entropy_comparison.png
  05_eval_rewards.png
  06_eval_episode_lengths.png
  07_training_summary.png        ← single combined dashboard
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no GUI needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ── optional TensorBoard reader ──────────────────────────────────────────────
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("[WARN] 'tensorboard' package not found – TensorBoard curves will be skipped.")
    print("       Install with:  pip install tensorboard")

# ─────────────────────────────────────────────────────────────────────────────
# Paths (relative to this file's directory)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OLD_TB    = os.path.join(BASE_DIR, "ppo_traffic_tensorboard")   # 2-phase runs
NEW_TB    = os.path.join(BASE_DIR, "traffic_tensorboard")       # 4-phase runs
EVAL_NPZ  = os.path.join(BASE_DIR, "logs", "evaluations.npz")
OUT_DIR   = os.path.join(BASE_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "old": "#E07B39",   # orange  – old 2-phase model
    "new": "#2E8BC0",   # blue    – new 4-phase model
}
STYLE = {"linewidth": 1.8, "alpha": 0.9}

def smooth(values, weight=0.85):
    """Exponential moving average (same as TensorBoard's smoothing slider)."""
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return np.array(smoothed)


def load_tb_scalar(log_dir: str, tag: str):
    """
    Walk all event files under log_dir, collect (step, value) pairs for `tag`.
    Merges multiple PPO_* sub-runs by offsetting steps.
    Returns (steps_array, values_array) or (None, None) if unavailable.
    """
    if not HAS_TB or not os.path.isdir(log_dir):
        return None, None

    event_files = sorted(glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"),
                                   recursive=True))
    all_steps, all_vals = [], []
    step_offset = 0
    prev_max_step = 0

    for ef in event_files:
        ea = EventAccumulator(ef)
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        events = ea.Scalars(tag)
        for e in events:
            all_steps.append(e.step + step_offset)
            all_vals.append(e.value)
        if events:
            prev_max_step = max(e.step for e in events)
        step_offset += prev_max_step

    if not all_steps:
        return None, None

    # Sort by step (multiple files can interleave)
    pairs = sorted(zip(all_steps, all_vals))
    steps = np.array([p[0] for p in pairs])
    vals  = np.array([p[1] for p in pairs])
    return steps, vals


def plot_scalar_comparison(tag, label, filename, ylabel=None, lower_is_better=False):
    """Plot one TB scalar: old (orange) vs new (blue), save PNG."""
    old_steps, old_vals = load_tb_scalar(OLD_TB, tag)
    new_steps, new_vals = load_tb_scalar(NEW_TB, tag)

    if old_steps is None and new_steps is None:
        print(f"  [SKIP] No data for tag '{tag}'")
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))

    if old_steps is not None:
        ax.plot(old_steps, smooth(old_vals), color=COLORS["old"],
                label="Old model (2-phase)", **STYLE)
        ax.plot(old_steps, old_vals, color=COLORS["old"], alpha=0.15, linewidth=0.8)

    if new_steps is not None:
        ax.plot(new_steps, smooth(new_vals), color=COLORS["new"],
                label="New model (4-phase + left turns)", **STYLE)
        ax.plot(new_steps, new_vals, color=COLORS["new"], alpha=0.15, linewidth=0.8)

    # Highlight final-value annotations
    def annotate_final(steps, vals, color):
        if steps is None:
            return
        final_step, final_val = steps[-1], smooth(vals)[-1]
        ax.annotate(f"{final_val:.3f}",
                    xy=(final_step, final_val),
                    xytext=(8, 0), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold")

    annotate_final(old_steps, old_vals, COLORS["old"])
    annotate_final(new_steps, new_vals, COLORS["new"])

    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_xlabel("Training steps", fontsize=11)
    ax.set_ylabel(ylabel or label, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)

    hint = "↓ lower is better" if lower_is_better else "↑ higher is better"
    ax.text(0.98, 0.02, hint, transform=ax.transAxes,
            fontsize=9, color="grey", ha="right", va="bottom")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 1-6  Individual comparison plots
# ─────────────────────────────────────────────────────────────────────────────
TAGS = [
    # (TB tag,                              title,                     filename,                       ylabel,           lower_better)
    ("rollout/ep_rew_mean",      "Episode Reward (Mean)",       "01_reward_comparison.png",      "Mean reward",     False),
    ("train/policy_gradient_loss","Policy Loss",               "02_policy_loss_comparison.png", "Policy loss",     True),
    ("train/value_loss",         "Value Loss",                  "03_value_loss_comparison.png",  "Value loss",      True),
    ("train/entropy_loss",       "Entropy (Exploration Level)", "04_entropy_comparison.png",     "Entropy",         False),
    ("rollout/ep_len_mean",      "Mean Episode Length",         "05_ep_len_comparison.png",      "Steps / episode", False),
]

print("\n[1/3] Generating per-metric comparison charts …")
for tag, title, fname, ylbl, low in TAGS:
    plot_scalar_comparison(tag, title, fname, ylbl, low)

# ─────────────────────────────────────────────────────────────────────────────
# 5 & 6  EvalCallback NPZ plots
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/3] Generating evaluation reward plots from evaluations.npz …")
if os.path.isfile(EVAL_NPZ):
    data = np.load(EVAL_NPZ)
    timesteps   = data["timesteps"]              # (n_evals,)
    ep_rewards  = data["results"]                # (n_evals, n_eval_episodes)
    ep_lengths  = data.get("ep_lengths", None)   # (n_evals, n_eval_episodes) or None

    mean_rew = ep_rewards.mean(axis=1)
    std_rew  = ep_rewards.std(axis=1)
    best_ts  = timesteps[np.argmax(mean_rew)]
    best_rew = mean_rew.max()

    # -- Eval rewards over time --
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(timesteps, mean_rew, color=COLORS["new"], label="New model (eval mean)", **STYLE)
    ax.fill_between(timesteps, mean_rew - std_rew, mean_rew + std_rew,
                    color=COLORS["new"], alpha=0.2, label="±1 std")
    ax.axvline(best_ts, color="green", linestyle="--", linewidth=1.2,
               label=f"Best eval @ step {best_ts:,}")
    ax.set_title("Evaluation Reward Over Training (New 4-Phase Model)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Training steps", fontsize=11)
    ax.set_ylabel("Episode reward", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    ax.text(0.98, 0.02, f"Best mean reward: {best_rew:.1f}",
            transform=ax.transAxes, fontsize=10, color="green",
            ha="right", va="bottom", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "06_eval_rewards.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    # -- Eval episode lengths --
    if ep_lengths is not None:
        mean_len = ep_lengths.mean(axis=1)
        std_len  = ep_lengths.std(axis=1)
        fig, ax  = plt.subplots(figsize=(10, 4.5))
        ax.plot(timesteps, mean_len, color=COLORS["new"], **STYLE)
        ax.fill_between(timesteps, mean_len - std_len, mean_len + std_len,
                        color=COLORS["new"], alpha=0.2)
        ax.set_title("Evaluation Episode Length Over Training",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Training steps", fontsize=11)
        ax.set_ylabel("Steps per episode", fontsize=11)
        ax.grid(True, alpha=0.35)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, "07_eval_ep_lengths.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out}")
else:
    print("  [SKIP] evaluations.npz not found (run training first).")

# ─────────────────────────────────────────────────────────────────────────────
# 7  Combined dashboard (4-panel)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/3] Generating combined dashboard …")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Traffic RL Model – Old (2-Phase)  vs  New (4-Phase with Left Turns)",
             fontsize=15, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.32)

dashboard_tags = [
    ("rollout/ep_rew_mean",       "Episode Reward (↑ better)", False),
    ("train/value_loss",          "Value Loss (↓ better)",     True),
    ("train/policy_gradient_loss","Policy Loss (↓ better)",    True),
    ("train/entropy_loss",        "Entropy (exploration)",     False),
]

for idx, (tag, title, low) in enumerate(dashboard_tags):
    row, col = divmod(idx, 2)
    ax = fig.add_subplot(gs[row, col])

    old_steps, old_vals = load_tb_scalar(OLD_TB, tag)
    new_steps, new_vals = load_tb_scalar(NEW_TB, tag)

    if old_steps is not None:
        ax.plot(old_steps, smooth(old_vals), color=COLORS["old"],
                label="Old (2-phase)", linewidth=1.6, alpha=0.9)
    if new_steps is not None:
        ax.plot(new_steps, smooth(new_vals), color=COLORS["new"],
                label="New (4-phase)", linewidth=1.6, alpha=0.9)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Steps", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Append eval reward curve as a 5th sub-plot spanning full bottom width
if os.path.isfile(EVAL_NPZ):
    data      = np.load(EVAL_NPZ)
    timesteps = data["timesteps"]
    mean_rew  = data["results"].mean(axis=1)
    std_rew   = data["results"].std(axis=1)

    # Summary text block below the 2×2 grid
    summary_lines = []
    for tag, label, _ in dashboard_tags:
        s, v = load_tb_scalar(NEW_TB, tag)
        if s is not None:
            summary_lines.append(f"{label.split('(')[0].strip()}: final={smooth(v)[-1]:.4f}")
    if summary_lines:
        fig.text(0.5, -0.01, "  |  ".join(summary_lines),
                 ha="center", va="top", fontsize=9, color="#444", style="italic",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4f8", alpha=0.8))
else:
    summary_lines = []
    for tag, label, _ in dashboard_tags:
        s, v = load_tb_scalar(NEW_TB, tag)
        if s is not None:
            summary_lines.append(f"{label.split('(')[0].strip()}: final={smooth(v)[-1]:.4f}")
    if summary_lines:
        fig.text(0.5, -0.01, "  |  ".join(summary_lines),
                 ha="center", va="top", fontsize=9, color="#444", style="italic",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4f8", alpha=0.8))

plt.tight_layout()
out = os.path.join(OUT_DIR, "00_dashboard.png")
plt.savefig(out, dpi=160, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

print(f"\nDone! All plots saved to:  {OUT_DIR}")
print("Files generated:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
