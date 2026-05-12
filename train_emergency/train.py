import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATASET_URL = "https://universe.roboflow.com/himank-vpetc/ambulance-4bova/1"
AMBULANCE_CLASS_ID = 80
COCO_VEHICLE_CLASS_IDS = {2, 3, 5, 7}
KEEP_CLASS_IDS = COCO_VEHICLE_CLASS_IDS | {AMBULANCE_CLASS_ID}
COCO128_URL = "https://ultralytics.com/assets/coco128.zip"

# Windows + CUDA can exhaust the page file when dataloader workers spawn and
# each imports torch. Keep the default conservative; users can raise --workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_image_and_label(image_path: Path, label_path: Path, out_images: Path, out_labels: Path):
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    target_image = out_images / image_path.name
    target_label = out_labels / f"{image_path.stem}.txt"
    shutil.copy2(image_path, target_image)
    shutil.copy2(label_path, target_label)


def ensure_yolo_dirs(dataset_dir: Path):
    for split in ("train", "val", "test"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def count_images(dataset_dir: Path, split: str) -> int:
    image_dir = dataset_dir / "images" / split
    if not image_dir.exists():
        return 0
    return sum(1 for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def filter_label_file(src: Path, dst: Path, allowed_ids: set[int], class_remap: dict[int, int] | None = None) -> bool:
    kept = []
    if not src.exists():
        return False

    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue

        if class_remap and class_id in class_remap:
            class_id = class_remap[class_id]

        if class_id in allowed_ids:
            kept.append(" ".join([str(class_id), *parts[1:]]))

    if not kept:
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return True


def roboflow_class_remap(raw_dir: Path) -> dict[int, int]:
    data_yaml = raw_dir / "data.yaml"
    if not data_yaml.exists():
        return {0: AMBULANCE_CLASS_ID, 1: AMBULANCE_CLASS_ID}

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    remap = {}
    for idx, name in enumerate(names):
        if str(name).strip().lower() == "ambulance":
            remap[idx] = AMBULANCE_CLASS_ID

    return remap or {0: AMBULANCE_CLASS_ID, 1: AMBULANCE_CLASS_ID}


def download_roboflow_ambulance(api_key: str, out_dir: Path) -> Path:
    reset_dir(out_dir)
    try:
        import roboflow

        dataset = roboflow.download_dataset(
            dataset_url=DATASET_URL,
            model_format="yolov11",
            location=str(out_dir),
            overwrite=True,
            api_key=api_key,
        )
        return Path(getattr(dataset, "location", out_dir))
    except Exception:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        project = rf.workspace("himank-vpetc").project("ambulance-4bova")
        dataset = project.version(1).download("yolov11", location=str(out_dir), overwrite=True)
        return Path(dataset.location)


def ensure_coco_source(coco_root: Path | None) -> Path:
    if coco_root and coco_root.exists():
        return coco_root

    coco128_dir = ROOT / "data" / "coco128"
    if (coco128_dir / "images").exists() and (coco128_dir / "labels").exists():
        return coco128_dir

    zip_path = ROOT / "data" / "coco128.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print("COCO_DATA_ROOT not set. Downloading coco128 vehicle anchors...")
    urllib.request.urlretrieve(COCO128_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(ROOT / "data")
    return coco128_dir


def merge_ambulance_dataset(raw_dir: Path, merged_dir: Path):
    class_remap = roboflow_class_remap(raw_dir)
    for split in ("train", "valid", "test"):
        source_split = raw_dir / split
        target_split = "val" if split == "valid" else split
        image_dir = source_split / "images"
        label_dir = source_split / "labels"
        if not image_dir.exists():
            continue

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            src_label = label_dir / f"{image_path.stem}.txt"
            dst_label = merged_dir / "labels" / target_split / f"rf_{image_path.stem}.txt"
            if filter_label_file(src_label, dst_label, {AMBULANCE_CLASS_ID}, class_remap):
                dst_image = merged_dir / "images" / target_split / f"rf_{image_path.name}"
                dst_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, dst_image)


def backfill_validation_split(merged_dir: Path, val_fraction: float = 0.1, min_val: int = 50):
    if count_images(merged_dir, "val") > 0:
        return

    train_images = sorted(
        p for p in (merged_dir / "images" / "train").iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not train_images:
        return

    move_count = min(max(min_val, int(len(train_images) * val_fraction)), max(1, len(train_images) // 5))
    for image_path in train_images[:move_count]:
        label_path = merged_dir / "labels" / "train" / f"{image_path.stem}.txt"
        target_image = merged_dir / "images" / "val" / image_path.name
        target_label = merged_dir / "labels" / "val" / f"{image_path.stem}.txt"
        shutil.move(str(image_path), str(target_image))
        if label_path.exists():
            shutil.move(str(label_path), str(target_label))
    print(f"Validation split was empty; moved {move_count} training images into val.")


def merge_coco_vehicle_dataset(coco_root: Path, merged_dir: Path, max_images: int | None):
    split_pairs = [
        ("train2017", "train"),
        ("val2017", "val"),
        ("train", "train"),
        ("val", "val"),
    ]
    copied = {"train": 0, "val": 0}

    for coco_split, target_split in split_pairs:
        image_dir = coco_root / "images" / coco_split
        label_dir = coco_root / "labels" / coco_split
        if not image_dir.exists() or not label_dir.exists():
            continue

        for image_path in image_dir.iterdir():
            if max_images and copied[target_split] >= max_images:
                break
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            src_label = label_dir / f"{image_path.stem}.txt"
            dst_label = merged_dir / "labels" / target_split / f"coco_{image_path.stem}.txt"
            if filter_label_file(src_label, dst_label, COCO_VEHICLE_CLASS_IDS):
                dst_image = merged_dir / "images" / target_split / f"coco_{image_path.name}"
                dst_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, dst_image)
                copied[target_split] += 1

    print(f"COCO vehicle anchors copied: train={copied['train']} val={copied['val']}")


def write_dataset_yaml(merged_dir: Path):
    template = yaml.safe_load((ROOT / "dataset.yaml").read_text(encoding="utf-8"))
    template["path"] = str(merged_dir.resolve())
    (ROOT / "dataset.yaml").write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")


def prepare_dataset(args):
    merged_dir = ROOT / "data" / "merged"
    raw_rf_dir = ROOT / "data" / "roboflow_ambulance"
    reset_dir(merged_dir)
    ensure_yolo_dirs(merged_dir)

    api_key = args.roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Set ROBOFLOW_API_KEY or pass --roboflow-api-key to download the ambulance dataset.")

    print("Downloading Roboflow ambulance dataset...")
    rf_location = download_roboflow_ambulance(api_key, raw_rf_dir)
    merge_ambulance_dataset(rf_location, merged_dir)

    coco_root = ensure_coco_source(Path(args.coco_data_root) if args.coco_data_root else None)
    merge_coco_vehicle_dataset(coco_root, merged_dir, args.max_coco_images)
    backfill_validation_split(merged_dir)
    write_dataset_yaml(merged_dir)
    return ROOT / "dataset.yaml"


def train(args):
    data_yaml = prepare_dataset(args)
    model = YOLO(args.weights)

    common = dict(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        amp=True,
        cos_lr=True,
        patience=args.patience,
        save=True,
        save_period=args.save_period,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        plots=True,
        pretrained=True,
        optimizer="AdamW",
    )

    print("Stage 1: freeze backbone and fine-tune detection head.")
    model.train(
        epochs=args.freeze_epochs,
        freeze=args.freeze_layers,
        lr0=args.head_lr,
        close_mosaic=5,
        **common,
    )

    stage1_best = ROOT / "runs" / args.name / "weights" / "best.pt"
    model = YOLO(str(stage1_best if stage1_best.exists() else args.weights))

    print("Stage 2: unfreeze and fine-tune the full detector.")
    model.train(
        epochs=args.finetune_epochs,
        freeze=0,
        lr0=args.finetune_lr,
        close_mosaic=10,
        resume=False,
        **common,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11s for COCO vehicles + ambulance.")
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--roboflow-api-key", default=None)
    parser.add_argument("--coco-data-root", default=os.environ.get("COCO_DATA_ROOT"))
    parser.add_argument("--max-coco-images", type=int, default=5000)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--freeze-epochs", type=int, default=25)
    parser.add_argument("--finetune-epochs", type=int, default=75)
    parser.add_argument("--freeze-layers", type=int, default=10)
    parser.add_argument("--head-lr", type=float, default=0.001)
    parser.add_argument("--finetune-lr", type=float, default=0.0002)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--name", default="emergency_yolo11s")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
