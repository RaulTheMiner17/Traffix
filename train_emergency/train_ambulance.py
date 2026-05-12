import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

import torch
from ultralytics import YOLO


DEFAULT_RUN_NAME = "emergency_yolo11n_ambulance_v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the standalone ambulance YOLO detector.")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "yolo11n.pt"), help="Base YOLO weights to fine-tune.")
    parser.add_argument("--data", default=str(ROOT / "data.yaml"), help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    device = 0 if torch.cuda.is_available() else "cpu"
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Base model not found: {model_path}")

    model = YOLO(str(model_path))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        workers=args.workers,
        patience=10,
        cache=False,
    )

    best_weights = ROOT / "runs" / args.name / "weights" / "best.pt"
    print(f"Ambulance detector ready: {best_weights}")


if __name__ == "__main__":
    main()
