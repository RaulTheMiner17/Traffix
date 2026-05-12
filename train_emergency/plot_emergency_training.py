import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs" / "emergency_yolo11n_ambulance_v2"


def read_results(csv_path):
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({key.strip(): float(value) for key, value in row.items()})
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def series(rows, key):
    return [row[key] for row in rows]


def plot_training_curve(run_dir):
    csv_path = run_dir / "results.csv"
    rows = read_results(csv_path)
    epochs = [int(row["epoch"]) for row in rows]

    precision = series(rows, "metrics/precision(B)")
    recall = series(rows, "metrics/recall(B)")
    map50 = series(rows, "metrics/mAP50(B)")
    map5095 = series(rows, "metrics/mAP50-95(B)")

    train_loss = [
        row["train/box_loss"] + row["train/cls_loss"] + row["train/dfl_loss"]
        for row in rows
    ]
    val_loss = [
        row["val/box_loss"] + row["val/cls_loss"] + row["val/dfl_loss"]
        for row in rows
    ]

    best_idx = max(range(len(rows)), key=lambda idx: map50[idx])
    final_idx = len(rows) - 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Emergency Vehicle Detection - YOLO Training Results",
                 fontsize=16, fontweight="bold")

    ax1.plot(epochs, precision, linewidth=2.4, label="Precision")
    ax1.plot(epochs, recall, linewidth=2.4, label="Recall")
    ax1.plot(epochs, map50, linewidth=3.0, label="mAP50")
    ax1.plot(epochs, map5095, linewidth=2.0, label="mAP50-95", alpha=0.8)
    ax1.scatter([epochs[best_idx]], [map50[best_idx]], s=80, color="#22c55e", zorder=5)
    ax1.annotate(f"Best mAP50: {map50[best_idx]:.2f}\nEpoch {epochs[best_idx]}",
                 xy=(epochs[best_idx], map50[best_idx]), xytext=(10, -35),
                 textcoords="offset points", fontsize=9, fontweight="bold",
                 color="#15803d",
                 arrowprops=dict(arrowstyle="->", color="#15803d"))
    ax1.set_title("Detection Metrics", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    ax2.plot(epochs, train_loss, linewidth=2.6, label="Train loss")
    ax2.plot(epochs, val_loss, linewidth=2.6, label="Validation loss")
    ax2.scatter([epochs[final_idx]], [val_loss[final_idx]], s=70,
                color="#E07B39", zorder=5)
    ax2.annotate(f"Final val loss: {val_loss[final_idx]:.2f}",
                 xy=(epochs[final_idx], val_loss[final_idx]), xytext=(-95, 20),
                 textcoords="offset points", fontsize=9, fontweight="bold",
                 color="#C2410C",
                 arrowprops=dict(arrowstyle="->", color="#C2410C"))
    ax2.set_title("Training Loss", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Total Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out_path = run_dir / "emergency_training_curve.png"
    plt.savefig(out_path, dpi=180)
    plt.close()

    return out_path, {
        "epochs": len(rows),
        "best_epoch": epochs[best_idx],
        "best_map50": map50[best_idx],
        "final_precision": precision[final_idx],
        "final_recall": recall[final_idx],
        "final_map50": map50[final_idx],
        "final_map5095": map5095[final_idx],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=str(DEFAULT_RUN), help="YOLO run directory")
    args = parser.parse_args()

    out_path, summary = plot_training_curve(Path(args.run))
    print(f"Saved: {out_path}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
