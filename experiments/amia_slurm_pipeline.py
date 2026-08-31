
import os
import sys
import gc
import ctypes
import json
import math
import time
import random
import shutil
import warnings
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from tqdm.auto import tqdm

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ============================================================
# Configuration from environment variables
# ============================================================

def env_bool(name, default=False):
    return os.environ.get(name, str(default)).lower() in {"1", "true", "yes", "y"}

def env_int(name, default):
    return int(os.environ.get(name, default))

def env_float(name, default):
    return float(os.environ.get(name, default))

SEED = env_int("AMIA_SEED", 42)
IMG_SIZE = env_int("AMIA_IMG_SIZE", 1024)
VAL_SIZE = env_float("AMIA_VAL_SIZE", 0.20)
WBF_IOU_THR = env_float("AMIA_WBF_IOU", 0.40)
EVAL_IOU_THR = env_float("AMIA_EVAL_IOU", 0.40)

FAST_DEV_RUN = env_bool("AMIA_FAST_DEV_RUN", False)

YOLO_EPOCHS = env_int("AMIA_YOLO_EPOCHS", 40 if not FAST_DEV_RUN else 1)
FRCNN_EPOCHS = env_int("AMIA_FRCNN_EPOCHS", 10 if not FAST_DEV_RUN else 1)
CLS_EPOCHS = env_int("AMIA_CLS_EPOCHS", 8 if not FAST_DEV_RUN else 1)

YOLO_BATCH = env_int("AMIA_YOLO_BATCH", 4)
YOLO_INFER_BATCH = env_int("AMIA_YOLO_INFER_BATCH", 1)
YOLO_INFER_CHUNK = env_int("AMIA_YOLO_INFER_CHUNK", 64)
FRCNN_BATCH = env_int("AMIA_FRCNN_BATCH", 2)
CLS_BATCH = env_int("AMIA_CLS_BATCH", 16)

# DataLoader workers are configurable for a normal Linux/SLURM compute node.
FRCNN_NUM_WORKERS = env_int("AMIA_FRCNN_NUM_WORKERS", 4)
OTHER_NUM_WORKERS = env_int("AMIA_OTHER_NUM_WORKERS", 4)

YOLO_CONF = env_float("AMIA_YOLO_CONF", 0.03)
FRCNN_CONF = env_float("AMIA_FRCNN_CONF", 0.03)
DEFAULT_FINAL_CONF = env_float("AMIA_FINAL_CONF", 0.05)

RUN_TEST_INFERENCE = env_bool("AMIA_RUN_TEST", True)

ROOT = Path(os.environ.get("AMIA_WORK_DIR", "./work")).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("AMIA_DATA_DIR", "./data/amia-public-challenge-2026")).expanduser().resolve()
WEIGHTS_DIR = Path(os.environ.get("AMIA_WEIGHTS_DIR", "./weights")).expanduser().resolve()
OUT_DIR = ROOT / "amia_results"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
CKPT_DIR = ROOT / "amia_checkpoints"
STATE_DIR = ROOT / "amia_state"

for d in [ROOT, OUT_DIR, FIG_DIR, TABLE_DIR, CKPT_DIR, STATE_DIR, WEIGHTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

STATUS_PATH = STATE_DIR / "status.json"
RESULTS_JSON = OUT_DIR / "metrics.json"

def update_status(stage, message="", progress=None, extra=None):
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "message": message,
        "progress": progress,
    }
    if extra:
        payload.update(extra)
    STATUS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[STATUS] {stage}: {message}", flush=True)

def stage_done(name):
    return (STATE_DIR / f"{name}.done").exists()

def mark_stage_done(name):
    (STATE_DIR / f"{name}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))

def release_cuda_memory(label=""):
    """Release Python objects, host allocator pages, and PyTorch CUDA caches."""
    gc.collect()

    # On Linux, Python/PyTorch can return objects while glibc still holds large
    # arenas. malloc_trim() gives those pages back to the OS, which matters
    # under a strict SLURM --mem limit.
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    if label:
        rss_gib = None
        try:
            import psutil
            rss_gib = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        except Exception:
            pass

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            gpu_msg = (
                f"GPU free={free_b / 1024**3:.2f} GiB / "
                f"total={total_b / 1024**3:.2f} GiB"
            )
        else:
            gpu_msg = "GPU unavailable"

        ram_msg = f", host RSS={rss_gib:.2f} GiB" if rss_gib is not None else ""
        print(f"[MEM CLEANUP] {label}: {gpu_msg}{ram_msg}", flush=True)

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True

seed_everything()

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()
# SLURM requests exactly one GPU. Keep every model on cuda:0 and never
# auto-enable Ultralytics DDP merely because a cluster environment exposes
# more than one device. This makes the job behavior match the one-GPU GRES.
YOLO_DEVICE = 0 if N_GPUS >= 1 else "cpu"
YOLO_PRED_DEVICE = 0 if N_GPUS >= 1 else "cpu"

CLASS_NAMES = {
    0: "Aortic enlargement",
    1: "Atelectasis",
    2: "Calcification",
    3: "Cardiomegaly",
    4: "Consolidation",
    5: "ILD",
    6: "Infiltration",
    7: "Lung Opacity",
    8: "Nodule/Mass",
    9: "Other lesion",
    10: "Pleural effusion",
    11: "Pleural thickening",
    12: "Pneumothorax",
    13: "Pulmonary fibrosis",
    14: "No finding",
}
DETECTION_CLASS_NAMES = [CLASS_NAMES[i] for i in range(14)]
BBOX_COLS = ["x_min", "y_min", "x_max", "y_max"]

# ============================================================
# Data discovery and preprocessing
# ============================================================

def locate_data():
    input_root = DATA_ROOT
    if not input_root.exists():
        raise FileNotFoundError(
            f"AMIA_DATA_DIR does not exist: {input_root}. "
            "Run download_kaggle_data.sh first or set AMIA_DATA_DIR."
        )

    candidates = list(input_root.rglob("train.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No train.csv found under {input_root}. "
            "Check that the Kaggle archive has been extracted."
        )

    data_dir = None
    for p in candidates:
        parent = p.parent
        if all((parent / f).exists() for f in ["test.csv", "img_size.csv", "sample_submission.csv"]):
            data_dir = parent
            break
    if data_dir is None:
        raise FileNotFoundError(
            "Found train.csv but could not find test.csv, img_size.csv, and "
            "sample_submission.csv in the same directory."
        )

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    img_size_df = pd.read_csv(data_dir / "img_size.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

    return input_root, data_dir, train_df, test_df, img_size_df, sample_submission

def build_image_maps(input_root, data_dir, train_df, sample_submission):
    all_pngs = list(data_dir.rglob("*.png"))
    if not all_pngs:
        all_pngs = list(input_root.rglob("*.png"))

    train_ids = set(train_df["image_id"].astype(str).unique())
    test_ids = set(sample_submission["image_id"].astype(str).unique())

    train_map, test_map = {}, {}
    for p in all_pngs:
        stem = p.stem
        low = str(p).lower()
        if stem in train_ids and ("train" in low or stem not in test_ids):
            train_map.setdefault(stem, p)
        if stem in test_ids and ("test" in low or stem not in train_ids):
            test_map.setdefault(stem, p)

    for p in all_pngs:
        stem = p.stem
        if stem in train_ids:
            train_map.setdefault(stem, p)
        if stem in test_ids:
            test_map.setdefault(stem, p)

    missing_train = train_ids - set(train_map)
    missing_test = test_ids - set(test_map)
    if missing_train:
        raise FileNotFoundError(f"Missing {len(missing_train)} training PNGs.")
    if missing_test:
        raise FileNotFoundError(f"Missing {len(missing_test)} test PNGs.")

    return train_map, test_map

def preprocess_annotations(train_df, img_size_df):
    train_df = train_df.copy()
    train_df["image_id"] = train_df["image_id"].astype(str)
    train_df["class_id"] = train_df["class_id"].astype(int)

    img_size_df = img_size_df.copy()
    img_size_df["image_id"] = img_size_df["image_id"].astype(str)

    det_df = train_df[train_df["class_id"].between(0, 13)].copy()
    for c in BBOX_COLS:
        det_df[c] = pd.to_numeric(det_df[c], errors="coerce")
    det_df = det_df.dropna(subset=BBOX_COLS)

    det_df = det_df.merge(
        img_size_df[["image_id", "dim0", "dim1"]],
        on="image_id",
        how="left",
    )
    if det_df[["dim0", "dim1"]].isna().any().any():
        raise ValueError("Some annotations do not have image-size metadata.")

    det_df["x_min"] = det_df["x_min"] / det_df["dim1"] * IMG_SIZE
    det_df["x_max"] = det_df["x_max"] / det_df["dim1"] * IMG_SIZE
    det_df["y_min"] = det_df["y_min"] / det_df["dim0"] * IMG_SIZE
    det_df["y_max"] = det_df["y_max"] / det_df["dim0"] * IMG_SIZE

    for c in BBOX_COLS:
        det_df[c] = det_df[c].clip(0, IMG_SIZE)

    det_df = det_df[
        (det_df["x_max"] > det_df["x_min"]) &
        (det_df["y_max"] > det_df["y_min"])
    ].reset_index(drop=True)

    return train_df, det_df

def box_iou_np(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, float(a[2]-a[0])) * max(0.0, float(a[3]-a[1]))
    area_b = max(0.0, float(b[2]-b[0])) * max(0.0, float(b[3]-b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def fuse_group_boxes(group, iou_thr=WBF_IOU_THR):
    rows = group.reset_index(drop=True)
    clusters = []

    for _, row in rows.iterrows():
        box = np.array([row.x_min, row.y_min, row.x_max, row.y_max], dtype=float)
        best_idx, best_iou = None, -1.0

        for i, cl in enumerate(clusters):
            fused = np.mean(np.stack(cl["boxes"]), axis=0)
            iou = box_iou_np(box, fused)
            if iou >= iou_thr and iou > best_iou:
                best_idx, best_iou = i, iou

        if best_idx is None:
            rad = str(row.rad_id) if "rad_id" in rows.columns else "unknown"
            clusters.append({"boxes": [box], "rad_ids": [rad]})
        else:
            clusters[best_idx]["boxes"].append(box)
            if "rad_id" in rows.columns:
                clusters[best_idx]["rad_ids"].append(str(row.rad_id))

    output = []
    for cl in clusters:
        fused = np.mean(np.stack(cl["boxes"]), axis=0)
        output.append({
            "x_min": fused[0],
            "y_min": fused[1],
            "x_max": fused[2],
            "y_max": fused[3],
            "n_boxes_fused": len(cl["boxes"]),
            "n_radiologists": len(set(cl["rad_ids"])),
        })
    return output

def build_consensus(det_df):
    records = []
    for (image_id, class_id), grp in tqdm(
        det_df.groupby(["image_id", "class_id"]),
        desc="Consensus annotation fusion"
    ):
        for item in fuse_group_boxes(grp):
            item.update({"image_id": image_id, "class_id": int(class_id)})
            records.append(item)

    cols = [
        "image_id", "class_id", "x_min", "y_min", "x_max", "y_max",
        "n_boxes_fused", "n_radiologists"
    ]
    return pd.DataFrame(records)[cols]

# ============================================================
# EDA visualizations
# ============================================================

def save_eda(train_df, det_df, consensus_df, train_ids, val_ids):
    class_count = (
        train_df.groupby(["class_id", "class_name"])
        .size()
        .reset_index(name="count")
        .sort_values("class_id")
    )
    class_count.to_csv(TABLE_DIR / "class_counts.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(class_count["class_name"], class_count["count"])
    plt.xlabel("Number of annotation rows")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_class_distribution.png", dpi=160)
    plt.close()

    box_w = det_df["x_max"] - det_df["x_min"]
    box_h = det_df["y_max"] - det_df["y_min"]

    plt.figure(figsize=(7, 5))
    plt.hist(np.sqrt(box_w * box_h), bins=50)
    plt.xlabel("sqrt(box area) in pixels at 1024 scale")
    plt.ylabel("Annotation count")
    plt.title("Lesion Size Distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_lesion_size_distribution.png", dpi=160)
    plt.close()

    raw_n = len(det_df)
    fused_n = len(consensus_df)
    plt.figure(figsize=(6, 4))
    plt.bar(["Raw boxes", "Consensus boxes"], [raw_n, fused_n])
    plt.ylabel("Number of boxes")
    plt.title("Effect of Multi-Radiologist Box Fusion")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_annotation_fusion.png", dpi=160)
    plt.close()

    split_stats = pd.DataFrame({
        "split": ["Train", "Validation"],
        "images": [len(train_ids), len(val_ids)],
    })
    split_stats.to_csv(TABLE_DIR / "split_stats.csv", index=False)

# ============================================================
# YOLO
# ============================================================

def prepare_yolo_dataset(train_ids, val_ids, train_image_map, consensus_df):
    yolo_dir = ROOT / "yolo_chest_xray"

    # Reuse the prepared folder if it already exists and appears complete.
    yaml_path = yolo_dir / "data.yaml"
    if yaml_path.exists():
        return yolo_dir, yaml_path

    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)

    for split in ["train", "val"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    consensus_by_image = {
        k: g.copy() for k, g in consensus_df.groupby("image_id")
    }

    def prepare_split(ids, split):
        for image_id in tqdm(sorted(ids), desc=f"Preparing YOLO {split}"):
            src = train_image_map[image_id]
            dst = yolo_dir / "images" / split / f"{image_id}.png"
            try:
                os.symlink(src, dst)
            except Exception:
                shutil.copy2(src, dst)

            rows = consensus_by_image.get(image_id)
            lines = []
            if rows is not None:
                for r in rows.itertuples(index=False):
                    xc = ((r.x_min + r.x_max) / 2) / IMG_SIZE
                    yc = ((r.y_min + r.y_max) / 2) / IMG_SIZE
                    w = (r.x_max - r.x_min) / IMG_SIZE
                    h = (r.y_max - r.y_min) / IMG_SIZE
                    if w > 0 and h > 0:
                        lines.append(
                            f"{int(r.class_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                        )

            (yolo_dir / "labels" / split / f"{image_id}.txt").write_text(
                "\n".join(lines)
            )

    prepare_split(train_ids, "train")
    prepare_split(val_ids, "val")

    yaml_lines = [
        f"path: {yolo_dir}",
        "train: images/train",
        "val: images/val",
        "nc: 14",
        "names:",
    ]
    for i, name in enumerate(DETECTION_CLASS_NAMES):
        safe_name = name.replace("'", "")
        yaml_lines.append(f"  {i}: '{safe_name}'")

    yaml_path.write_text("\n".join(yaml_lines))
    return yolo_dir, yaml_path

def find_yolo_seed_weight(input_root):
    preferred = WEIGHTS_DIR / "yolov8s.pt"
    if preferred.exists():
        return str(preferred)
    matches = list(input_root.rglob("yolov8s.pt"))
    if matches:
        return str(matches[0])
    return "yolov8s.pt"

def train_yolo(input_root, yaml_path):
    from ultralytics import YOLO

    run_dir = ROOT / "yolo_runs" / "yolov8s_1024_consensus"
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"

    if stage_done("yolo_train") and best.exists():
        print("YOLO training already complete. Reusing best.pt.")
        return YOLO(str(best)), best

    if last.exists():
        print("Resuming YOLO from:", last)
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(find_yolo_seed_weight(input_root))
        model.train(
            data=str(yaml_path),
            imgsz=IMG_SIZE,
            epochs=YOLO_EPOCHS,
            batch=YOLO_BATCH,
            workers=OTHER_NUM_WORKERS,
            device=YOLO_DEVICE,
            amp=True,
            cache=False,
            seed=SEED,
            deterministic=False,
            patience=12,
            close_mosaic=10,
            project=str(ROOT / "yolo_runs"),
            name="yolov8s_1024_consensus",
            exist_ok=True,
            verbose=True,
        )

    if not best.exists():
        raise FileNotFoundError("YOLO training finished but best.pt was not found.")

    mark_stage_done("yolo_train")

    # Ultralytics' training object can retain optimizer/scaler/model tensors on
    # the GPU after model.train() returns. On an H100 this can still occupy
    # tens of GiB and make the following validation warmup OOM. Destroy the
    # training object before creating a fresh inference-only YOLO instance.
    try:
        if "model" in locals():
            try:
                if getattr(model, "trainer", None) is not None:
                    model.trainer.optimizer = None
            except Exception:
                pass
            del model
    except Exception:
        pass

    release_cuda_memory("after YOLO training")
    inference_model = YOLO(str(best))
    return inference_model, best

def save_yolo_curves():
    csv_path = ROOT / "yolo_runs" / "yolov8s_1024_consensus" / "results.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df.to_csv(TABLE_DIR / "yolo_training_log.csv", index=False)

    metric_cols = [
        c for c in df.columns
        if "metrics/mAP50(B)" in c or "metrics/mAP50-95(B)" in c
    ]
    loss_cols = [c for c in df.columns if "train/" in c and "loss" in c]

    if metric_cols:
        plt.figure(figsize=(8, 5))
        for c in metric_cols:
            plt.plot(df["epoch"], df[c], label=c)
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("YOLO Validation Metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "04_yolo_metrics.png", dpi=160)
        plt.close()

    if loss_cols:
        plt.figure(figsize=(8, 5))
        for c in loss_cols:
            plt.plot(df["epoch"], df[c], label=c)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("YOLO Training Losses")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "05_yolo_losses.png", dpi=160)
        plt.close()

def predict_yolo(image_ids, image_map, model, conf=YOLO_CONF):
    """
    Memory-safe YOLO inference using small source chunks.

    Passing thousands of paths to Ultralytics in one model.predict() call can
    inflate host RAM before results are yielded. Chunking keeps RSS bounded.
    """
    preds = []
    ordered_ids = list(sorted(image_ids))
    total = len(ordered_ids)

    release_cuda_memory("before YOLO inference")
    pbar = tqdm(total=total, desc="YOLO inference")

    for chunk_start in range(0, total, YOLO_INFER_CHUNK):
        chunk_ids = ordered_ids[chunk_start:chunk_start + YOLO_INFER_CHUNK]
        chunk_paths = [str(image_map[i]) for i in chunk_ids]

        results = model.predict(
            source=chunk_paths,
            imgsz=IMG_SIZE,
            conf=conf,
            iou=0.50,
            device=YOLO_PRED_DEVICE,
            batch=YOLO_INFER_BATCH,
            half=torch.cuda.is_available(),
            verbose=False,
            stream=True,
        )

        for res in results:
            image_id = Path(res.path).stem

            if res.boxes is not None:
                xyxy = res.boxes.xyxy.detach().cpu().numpy()
                scores = res.boxes.conf.detach().cpu().numpy()
                classes = res.boxes.cls.detach().cpu().numpy().astype(int)

                for box, score, cid in zip(xyxy, scores, classes):
                    preds.append({
                        "image_id": image_id,
                        "class_id": int(cid),
                        "score": float(score),
                        "box": box.astype(float),
                    })

            pbar.update(1)
            del res

        del results
        del chunk_paths
        del chunk_ids

        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if chunk_start == 0 or ((chunk_start // YOLO_INFER_CHUNK) + 1) % 10 == 0:
            try:
                import psutil
                rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
                done = min(chunk_start + YOLO_INFER_CHUNK, total)
                print(
                    f"[YOLO CHUNK] {done}/{total}: host RSS={rss:.2f} GiB",
                    flush=True,
                )
            except Exception:
                pass

    pbar.close()
    release_cuda_memory("after YOLO inference")
    return preds


# ============================================================
# Faster R-CNN
# ============================================================

class CXRDetectionDataset(Dataset):
    def __init__(self, image_ids, image_map, ann_df, train=False):
        self.image_ids = list(sorted(image_ids))
        self.image_map = image_map
        self.ann_by_image = {
            k: g.copy()
            for k, g in ann_df[ann_df.image_id.isin(self.image_ids)].groupby("image_id")
        }
        self.train = train

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = Image.open(self.image_map[image_id]).convert("RGB")
        image = TF.to_tensor(image)

        rows = self.ann_by_image.get(image_id)
        if rows is None or len(rows) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(
                rows[["x_min", "y_min", "x_max", "y_max"]].values,
                dtype=torch.float32,
            )
            labels = torch.as_tensor(
                rows["class_id"].values + 1,
                dtype=torch.int64,
            )

        if self.train and random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            if len(boxes):
                width = image.shape[2]
                old_xmin = boxes[:, 0].clone()
                old_xmax = boxes[:, 2].clone()
                boxes[:, 0] = width - old_xmax
                boxes[:, 2] = width - old_xmin

        area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if len(boxes)
            else torch.zeros(0)
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return image, target, image_id

def detection_collate(batch):
    images, targets, ids = zip(*batch)
    return list(images), list(targets), list(ids)

def build_frcnn():
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    local_weights = WEIGHTS_DIR / "fasterrcnn_resnet50_fpn_coco.pth"
    if local_weights.exists():
        print("Loading local Faster R-CNN COCO weights:", local_weights)
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=IMG_SIZE,
            max_size=IMG_SIZE,
            trainable_backbone_layers=3,
        )
        state = torch.load(local_weights, map_location="cpu")
        model.load_state_dict(state, strict=True)
    else:
        try:
            model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                min_size=IMG_SIZE,
                max_size=IMG_SIZE,
                trainable_backbone_layers=3,
            )
        except Exception as e:
            print("Faster R-CNN pretrained weights unavailable:", e)
            model = fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                min_size=IMG_SIZE,
                max_size=IMG_SIZE,
                trainable_backbone_layers=3,
            )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 15)
    return model

def move_targets_to_device(targets, device):
    return [
        {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in t.items()
        }
        for t in targets
    ]

def frcnn_validation_loss(model, loader):
    was_training = model.training
    model.train()
    losses = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = [img.to(DEVICE, non_blocking=True) for img in images]
            targets = move_targets_to_device(targets, DEVICE)
            loss_dict = model(images, targets)
            losses.append(float(sum(loss_dict.values()).item()))

    model.train(was_training)
    return float(np.mean(losses)) if losses else float("inf")

def train_frcnn(train_ids, val_ids, train_image_map, consensus_df):
    train_ds = CXRDetectionDataset(
        train_ids, train_image_map, consensus_df, train=True
    )
    val_ds = CXRDetectionDataset(
        val_ids, train_image_map, consensus_df, train=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=FRCNN_BATCH,
        shuffle=True,
        num_workers=FRCNN_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=FRCNN_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate,
    )

    model = build_frcnn().to(DEVICE)

    last_ckpt = CKPT_DIR / "frcnn_last.pth"
    best_ckpt = CKPT_DIR / "frcnn_best.pth"
    log_csv = TABLE_DIR / "frcnn_training_log.csv"

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    start_epoch = 0
    best_val = float("inf")
    history = []

    if last_ckpt.exists():
        print("Resuming Faster R-CNN from:", last_ckpt)
        ckpt = torch.load(last_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt["best_val_loss"])
        history = ckpt.get("history", [])

    if stage_done("frcnn_train") and best_ckpt.exists():
        print("Faster R-CNN training already complete.")
    else:
        for epoch in range(start_epoch, FRCNN_EPOCHS):
            model.train()
            running = []

            pbar = tqdm(
                train_loader,
                desc=f"FRCNN epoch {epoch+1}/{FRCNN_EPOCHS}",
            )

            for batch_idx, (images, targets, _) in enumerate(pbar):
                images = [
                    img.to(DEVICE, non_blocking=True)
                    for img in images
                ]
                targets = move_targets_to_device(targets, DEVICE)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    "cuda",
                    enabled=torch.cuda.is_available(),
                ):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running.append(float(loss.item()))
                pbar.set_postfix(
                    train_loss=f"{np.mean(running[-50:]):.4f}"
                )

                # Lightweight heartbeat so monitoring can see that the
                # long-running SLURM job is making progress.
                if batch_idx % 250 == 0:
                    update_status(
                        "frcnn_train",
                        f"epoch {epoch+1}/{FRCNN_EPOCHS}, batch {batch_idx}/{len(train_loader)}",
                    )

            val_loss = frcnn_validation_loss(model, val_loader)
            train_loss = float(np.mean(running))
            scheduler.step()

            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            })

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val,
                "history": history,
            }
            torch.save(checkpoint, last_ckpt)

            if is_best:
                torch.save(checkpoint, best_ckpt)

            pd.DataFrame(history).to_csv(log_csv, index=False)

            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, best_val={best_val:.4f}",
                flush=True,
            )

        mark_stage_done("frcnn_train")

    if not best_ckpt.exists():
        raise FileNotFoundError("Faster R-CNN best checkpoint not found.")

    best_state = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(best_state["model_state_dict"])

    if log_csv.exists():
        h = pd.read_csv(log_csv)
        plt.figure(figsize=(8, 5))
        plt.plot(h["epoch"], h["train_loss"], marker="o", label="Train loss")
        plt.plot(h["epoch"], h["val_loss"], marker="o", label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Faster R-CNN Training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "06_frcnn_training.png", dpi=160)
        plt.close()

    return model, train_loader, val_loader, best_ckpt

def predict_frcnn(model, loader, score_thr=FRCNN_CONF):
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _, ids in tqdm(loader, desc="Faster R-CNN validation inference"):
            images = [x.to(DEVICE, non_blocking=True) for x in images]
            outputs = model(images)

            for image_id, out in zip(ids, outputs):
                boxes = out["boxes"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()
                labels = out["labels"].detach().cpu().numpy() - 1

                for box, score, cid in zip(boxes, scores, labels):
                    if score >= score_thr and 0 <= cid <= 13:
                        preds.append({
                            "image_id": image_id,
                            "class_id": int(cid),
                            "score": float(score),
                            "box": box.astype(float),
                        })
    return preds

# ============================================================
# DenseNet classifier
# ============================================================

class CXRClassificationDataset(Dataset):
    def __init__(self, image_ids, image_map, target_map, train=False):
        self.image_ids = list(sorted(image_ids))
        self.image_map = image_map
        self.target_map = target_map

        ops = [transforms.Resize((512, 512))]
        if train:
            ops += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ]
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = Image.open(self.image_map[image_id]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(
            float(self.target_map.get(image_id, 0)),
            dtype=torch.float32,
        )
        return x, y, image_id

def build_densenet():
    from torchvision.models import densenet121, DenseNet121_Weights

    local_weights = WEIGHTS_DIR / "densenet121_imagenet.pth"
    if local_weights.exists():
        print("Loading local DenseNet121 ImageNet weights:", local_weights)
        model = densenet121(weights=None)
        state = torch.load(local_weights, map_location="cpu")
        model.load_state_dict(state, strict=True)
    else:
        try:
            model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        except Exception as e:
            print("DenseNet pretrained weights unavailable:", e)
            model = densenet121(weights=None)

    model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
    return model

def evaluate_classifier(model, loader):
    model.eval()
    ys, ps, ids_all = [], [], []

    with torch.no_grad():
        for x, y, ids in loader:
            x = x.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
            ):
                logits = model(x).squeeze(1)

            ys.extend(y.numpy().tolist())
            ps.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            ids_all.extend(ids)

    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    return auc, np.asarray(ys), np.asarray(ps), list(ids_all)

def train_classifier(train_ids, val_ids, train_image_map, abnormal_target_map):
    train_ds = CXRClassificationDataset(
        train_ids, train_image_map, abnormal_target_map, train=True
    )
    val_ds = CXRClassificationDataset(
        val_ids, train_image_map, abnormal_target_map, train=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CLS_BATCH,
        shuffle=True,
        num_workers=OTHER_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CLS_BATCH * 2,
        shuffle=False,
        num_workers=OTHER_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_densenet().to(DEVICE)
    if N_GPUS > 1:
        model = torch.nn.DataParallel(model)

    train_targets = np.array(
        [abnormal_target_map[i] for i in train_ids]
    )
    pos = train_targets.sum()
    neg = len(train_targets) - pos
    pos_weight = torch.tensor(
        [neg / max(pos, 1)],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, CLS_EPOCHS),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    last_ckpt = CKPT_DIR / "densenet_last.pth"
    best_ckpt = CKPT_DIR / "densenet_best.pth"
    log_csv = TABLE_DIR / "densenet_training_log.csv"

    start_epoch = 0
    best_auc = -1.0
    history = []

    if last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=DEVICE)
        target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        target_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_auc = float(ckpt["best_auc"])
        history = ckpt.get("history", [])

    if not (stage_done("classifier_train") and best_ckpt.exists()):
        for epoch in range(start_epoch, CLS_EPOCHS):
            model.train()
            losses = []

            pbar = tqdm(
                train_loader,
                desc=f"DenseNet epoch {epoch+1}/{CLS_EPOCHS}",
            )

            for x, y, _ in pbar:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    "cuda",
                    enabled=torch.cuda.is_available(),
                ):
                    logits = model(x).squeeze(1)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                losses.append(float(loss.item()))
                pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")

            scheduler.step()
            auc, _, _, _ = evaluate_classifier(model, val_loader)

            history.append({
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_auc": float(auc),
                "lr": optimizer.param_groups[0]["lr"],
            })

            is_best = auc > best_auc
            if is_best:
                best_auc = auc

            target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            payload = {
                "epoch": epoch,
                "model_state_dict": target_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_auc": best_auc,
                "history": history,
            }
            torch.save(payload, last_ckpt)
            if is_best:
                torch.save(payload, best_ckpt)

            pd.DataFrame(history).to_csv(log_csv, index=False)

            print(
                f"Epoch {epoch+1}: train_loss={np.mean(losses):.4f}, "
                f"val_auc={auc:.4f}, best_auc={best_auc:.4f}",
                flush=True,
            )

        mark_stage_done("classifier_train")

    if not best_ckpt.exists():
        raise FileNotFoundError("DenseNet best checkpoint not found.")

    state = torch.load(best_ckpt, map_location=DEVICE)
    target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    target_model.load_state_dict(state["model_state_dict"])

    if log_csv.exists():
        h = pd.read_csv(log_csv)

        plt.figure(figsize=(8, 5))
        plt.plot(h["epoch"], h["train_loss"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.title("DenseNet121 Training Loss")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "07_densenet_loss.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(h["epoch"], h["val_auc"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Validation ROC-AUC")
        plt.title("DenseNet121 Validation ROC-AUC")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "08_densenet_auc.png", dpi=160)
        plt.close()

    return model, train_loader, val_loader, best_ckpt

# ============================================================
# VOC 2010-style AP@0.40
# ============================================================

def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(
        np.sum(
            (mrec[idx+1] - mrec[idx]) * mpre[idx+1]
        )
    )

def evaluate_map40(predictions, gt_df, class_ids=range(14), iou_thr=EVAL_IOU_THR):
    gt_lookup = defaultdict(list)

    for r in gt_df.itertuples(index=False):
        gt_lookup[(str(r.image_id), int(r.class_id))].append(
            np.array([r.x_min, r.y_min, r.x_max, r.y_max], dtype=float)
        )

    ap_by_class = {}

    for cid in class_ids:
        cls_preds = [
            p for p in predictions
            if int(p["class_id"]) == cid
        ]
        cls_preds.sort(key=lambda x: x["score"], reverse=True)

        n_gt = sum(
            len(v)
            for (img, c), v in gt_lookup.items()
            if c == cid
        )

        if n_gt == 0:
            ap_by_class[cid] = np.nan
            continue

        used = {
            k: np.zeros(len(v), dtype=bool)
            for k, v in gt_lookup.items()
            if k[1] == cid
        }

        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        for i, p in enumerate(cls_preds):
            key = (str(p["image_id"]), cid)
            candidates = gt_lookup.get(key, [])

            if not candidates:
                fp[i] = 1
                continue

            ious = np.array([
                box_iou_np(np.asarray(p["box"], dtype=float), g)
                for g in candidates
            ])
            j = int(np.argmax(ious))

            if ious[j] >= iou_thr and not used[key][j]:
                tp[i] = 1
                used[key][j] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / max(n_gt, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        ap_by_class[cid] = (
            voc_ap(rec, prec) if len(rec) else 0.0
        )

    valid = [v for v in ap_by_class.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan"), ap_by_class

# ============================================================
# Ensembling and threshold tuning
# ============================================================

def ensemble_predictions(pred_lists, iou_thr=0.40, min_score=0.01):
    grouped = defaultdict(list)

    for preds in pred_lists:
        for p in preds:
            if p["score"] >= min_score:
                grouped[(p["image_id"], int(p["class_id"]))].append(p)

    fused_all = []

    for (image_id, cid), items in grouped.items():
        items = sorted(items, key=lambda z: z["score"], reverse=True)
        clusters = []

        for p in items:
            box = np.asarray(p["box"], dtype=float)
            best_idx, best_iou = None, -1.0

            for i, cl in enumerate(clusters):
                weights = np.asarray(cl["scores"], dtype=float)
                fused_box = np.average(
                    np.stack(cl["boxes"]),
                    axis=0,
                    weights=weights,
                )
                iou = box_iou_np(box, fused_box)

                if iou >= iou_thr and iou > best_iou:
                    best_idx, best_iou = i, iou

            if best_idx is None:
                clusters.append({
                    "boxes": [box],
                    "scores": [float(p["score"])],
                })
            else:
                clusters[best_idx]["boxes"].append(box)
                clusters[best_idx]["scores"].append(float(p["score"]))

        for cl in clusters:
            scores = np.asarray(cl["scores"], dtype=float)
            box = np.average(
                np.stack(cl["boxes"]),
                axis=0,
                weights=scores,
            )
            fused_all.append({
                "image_id": image_id,
                "class_id": cid,
                "score": float(np.mean(scores)),
                "box": box,
            })

    return fused_all

def threshold_predictions(preds, thr):
    return [p for p in preds if p["score"] >= thr]

def calibrate_with_classifier(preds, prob_map):
    out = []
    for p in preds:
        q = dict(p)
        q["score"] = float(
            p["score"] * prob_map.get(p["image_id"], 1.0)
        )
        out.append(q)
    return out

def tune_strategy(preds, gt_df, conf_grid=None):
    if conf_grid is None:
        conf_grid = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

    rows = []
    for thr in conf_grid:
        m, _ = evaluate_map40(
            threshold_predictions(preds, thr),
            gt_df,
        )
        rows.append({"threshold": thr, "mAP40": m})

    df = pd.DataFrame(rows)
    best = df.iloc[df["mAP40"].argmax()]
    return float(best["threshold"]), float(best["mAP40"]), df

# ============================================================
# Visualization helpers
# ============================================================

def save_classifier_visuals(y_true, y_prob, threshold):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("DenseNet121 Normal/Abnormal ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_classifier_roc.png", dpi=160)
    plt.close()

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 5))
    plt.imshow(cm)
    plt.xticks([0, 1], ["Normal", "Abnormal"])
    plt.yticks([0, 1], ["Normal", "Abnormal"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("DenseNet121 Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_classifier_confusion_matrix.png", dpi=160)
    plt.close()

def save_ap_comparison(yolo_ap, frcnn_ap, ensemble_ap, hybrid_ap):
    df = pd.DataFrame({
        "class_id": range(14),
        "class_name": [CLASS_NAMES[i] for i in range(14)],
        "YOLO_AP40": [yolo_ap[i] for i in range(14)],
        "FRCNN_AP40": [frcnn_ap[i] for i in range(14)],
        "Ensemble_AP40": [ensemble_ap[i] for i in range(14)],
        "Hybrid_AP40": [hybrid_ap[i] for i in range(14)],
    })
    df.to_csv(TABLE_DIR / "per_class_ap40.csv", index=False)

    x = np.arange(14)
    width = 0.20

    plt.figure(figsize=(16, 7))
    plt.bar(x - 1.5*width, df["YOLO_AP40"], width, label="YOLOv8s")
    plt.bar(x - 0.5*width, df["FRCNN_AP40"], width, label="Faster R-CNN")
    plt.bar(x + 0.5*width, df["Ensemble_AP40"], width, label="Detector ensemble")
    plt.bar(x + 1.5*width, df["Hybrid_AP40"], width, label="Hybrid + DenseNet")
    plt.xticks(x, df["class_name"], rotation=70, ha="right")
    plt.ylabel("AP@0.40")
    plt.title("Per-Class AP@0.40 Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_per_class_ap_comparison.png", dpi=160)
    plt.close()

def save_model_summary_plot(metrics):
    labels = ["YOLOv8s", "Faster R-CNN", "Ensemble", "Hybrid"]
    vals = [
        metrics["yolo_map40"],
        metrics["frcnn_map40"],
        metrics["ensemble_map40"],
        metrics["hybrid_map40"],
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals)
    plt.ylabel("Lesion mAP@0.40")
    plt.title("Detector System Comparison")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_model_summary.png", dpi=160)
    plt.close()

def save_qualitative_examples(image_ids, image_map, gt_df, preds, n=6):
    chosen = list(sorted(image_ids))[:]
    random.Random(SEED).shuffle(chosen)
    chosen = chosen[:min(n, len(chosen))]

    for idx, image_id in enumerate(chosen, 1):
        img = np.array(Image.open(image_map[image_id]).convert("RGB"))

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(img)

        gt_rows = gt_df[gt_df.image_id == image_id]
        for r in gt_rows.itertuples(index=False):
            rect = patches.Rectangle(
                (r.x_min, r.y_min),
                r.x_max-r.x_min,
                r.y_max-r.y_min,
                fill=False,
                linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(
                r.x_min,
                r.y_min,
                "GT: " + CLASS_NAMES[int(r.class_id)],
                fontsize=7,
                backgroundcolor="white",
            )

        for p in preds:
            if p["image_id"] != image_id or p["score"] < 0.10:
                continue

            x1, y1, x2, y2 = p["box"]
            rect = patches.Rectangle(
                (x1, y1),
                x2-x1,
                y2-y1,
                fill=False,
                linewidth=1.5,
                linestyle="--",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y2,
                f"P: {CLASS_NAMES[p['class_id']]} {p['score']:.2f}",
                fontsize=7,
                backgroundcolor="white",
            )

        ax.set_title(f"Qualitative validation example: {image_id}")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            FIG_DIR / f"qualitative_{idx:02d}_{image_id}.png",
            dpi=150,
        )
        plt.close()

# ============================================================
# Test inference
# ============================================================

class CXRTestDetectionDataset(Dataset):
    def __init__(self, ids, image_map):
        self.ids = list(ids)
        self.image_map = image_map

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img = TF.to_tensor(
            Image.open(self.image_map[image_id]).convert("RGB")
        )
        return img, image_id

def test_det_collate(batch):
    images, ids = zip(*batch)
    return list(images), list(ids)

def predict_frcnn_test(model, ids, image_map, score_thr=FRCNN_CONF):
    ds = CXRTestDetectionDataset(ids, image_map)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=test_det_collate,
    )

    model.eval()
    preds = []

    with torch.no_grad():
        for images, image_ids in tqdm(loader, desc="Faster R-CNN test inference"):
            outputs = model([
                x.to(DEVICE, non_blocking=True)
                for x in images
            ])

            for image_id, out in zip(image_ids, outputs):
                boxes = out["boxes"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()
                labels = out["labels"].detach().cpu().numpy() - 1

                for box, score, cid in zip(boxes, scores, labels):
                    if score >= score_thr and 0 <= cid <= 13:
                        preds.append({
                            "image_id": image_id,
                            "class_id": int(cid),
                            "score": float(score),
                            "box": box.astype(float),
                        })

    return preds

class TestClassificationDataset(Dataset):
    def __init__(self, image_ids, image_map, transform):
        self.image_ids = list(image_ids)
        self.image_map = image_map
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = Image.open(self.image_map[image_id]).convert("RGB")
        return self.transform(img), image_id

def predict_classifier_test(model, ids, image_map, transform):
    ds = TestClassificationDataset(ids, image_map, transform)
    loader = DataLoader(
        ds,
        batch_size=CLS_BATCH * 2,
        shuffle=False,
        num_workers=OTHER_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    result = {}

    with torch.no_grad():
        for x, batch_ids in tqdm(loader, desc="DenseNet test inference"):
            x = x.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
            ):
                prob = torch.sigmoid(model(x).squeeze(1)).detach().cpu().numpy()

            result.update({
                image_id: float(p)
                for image_id, p in zip(batch_ids, prob)
            })

    return result

# ============================================================
# Submission and report
# ============================================================

def format_prediction_string(preds, image_id, final_thr):
    selected = [
        p for p in preds
        if p["image_id"] == image_id and p["score"] >= final_thr
    ]
    selected.sort(key=lambda p: p["score"], reverse=True)

    if not selected:
        # Competition instructions explicitly request this exact No Finding form.
        return "14 1.0 0 0 1 1"

    parts = []
    for p in selected:
        x1, y1, x2, y2 = np.asarray(p["box"], dtype=float)
        x1 = float(np.clip(x1, 0, IMG_SIZE))
        y1 = float(np.clip(y1, 0, IMG_SIZE))
        x2 = float(np.clip(x2, 0, IMG_SIZE))
        y2 = float(np.clip(y2, 0, IMG_SIZE))

        if x2 <= x1 or y2 <= y1:
            continue

        parts.extend([
            str(int(p["class_id"])),
            f"{p['score']:.6f}",
            f"{x1:.2f}",
            f"{y1:.2f}",
            f"{x2:.2f}",
            f"{y2:.2f}",
        ])

    return " ".join(parts) if parts else "14 1.0 0 0 1 1"

def write_report(metrics, dataset_stats, classifier_report_text, strategy_name, final_thr):
    per_class = pd.read_csv(TABLE_DIR / "per_class_ap40.csv")

    headers = list(per_class.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in per_class.itertuples(index=False, name=None):
        vals = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                vals.append(f"{float(value):.4f}")
            else:
                vals.append(str(value))
        table_lines.append("| " + " | ".join(vals) + " |")
    table_md = "\n".join(table_lines)

    report = f"""# AMIA Public Challenge 2026 — SLURM Experiment Report

## 1. Project Goal

The project localizes and classifies 14 thoracic abnormalities in 1024×1024 chest radiographs. The system combines multi-radiologist consensus annotation, a one-stage YOLOv8s detector, a two-stage Faster R-CNN detector, and a DenseNet121 Normal/Abnormal classifier.

## 2. Dataset and Annotation Issues

- Training images: **{dataset_stats['n_train_images']}**
- Validation images: **{dataset_stats['n_val_images']}**
- Test images: **{dataset_stats['n_test_images']}**
- Raw abnormal annotation boxes: **{dataset_stats['raw_boxes']}**
- Consensus boxes after fusion: **{dataset_stats['consensus_boxes']}**
- Annotation-box reduction: **{dataset_stats['fusion_reduction_pct']:.2f}%**
- Median abnormal box width at 1024 scale: **{dataset_stats['median_box_width']:.2f} px**
- Median abnormal box height at 1024 scale: **{dataset_stats['median_box_height']:.2f} px**

Multiple radiologists can mark the same lesion with slightly different boxes. Highly overlapping boxes from the same image and class were therefore fused using an IoU threshold of {WBF_IOU_THR:.2f}. This reduces duplicated detector targets while retaining spatially separated lesions.

![Class distribution](figures/01_class_distribution.png)

![Lesion size distribution](figures/02_lesion_size_distribution.png)

![Annotation fusion](figures/03_annotation_fusion.png)

## 3. Model Design

### YOLOv8s
YOLOv8s provides the one-stage detector baseline. It was trained at {IMG_SIZE}×{IMG_SIZE} resolution to preserve small radiographic findings.

### Faster R-CNN + ResNet50-FPN
Faster R-CNN provides a two-stage comparison model. This architecture tests whether region-proposal-based detection offers complementary localization behavior.

### DenseNet121
DenseNet121 performs image-level Normal/Abnormal classification. It provides global evidence that is independent from the local detectors.

## 4. Competition-Compatible Evaluation

The lesion detectors were evaluated using a custom PASCAL-VOC-style AP calculation at **IoU = {EVAL_IOU_THR:.2f}**, averaged over abnormality classes 0–13.

## 5. Main Results

| System | Lesion mAP@0.40 |
|---|---:|
| YOLOv8s | {metrics['yolo_map40']:.4f} |
| Faster R-CNN | {metrics['frcnn_map40']:.4f} |
| YOLO + Faster R-CNN ensemble | {metrics['ensemble_map40']:.4f} |
| Ensemble + DenseNet calibration | {metrics['hybrid_map40']:.4f} |

DenseNet validation ROC-AUC: **{metrics['classifier_auc']:.4f}**

DenseNet selected F1 threshold: **{metrics['classifier_threshold']:.4f}**

Automatically selected final validation strategy: **{strategy_name}**

Selected final confidence threshold: **{final_thr:.4f}**

![Model comparison](figures/12_model_summary.png)

![Per-class comparison](figures/11_per_class_ap_comparison.png)

## 6. Per-Class AP@0.40

{table_md}

## 7. Training Behavior

![YOLO validation metrics](figures/04_yolo_metrics.png)

![YOLO losses](figures/05_yolo_losses.png)

![Faster R-CNN training](figures/06_frcnn_training.png)

![DenseNet training loss](figures/07_densenet_loss.png)

![DenseNet validation ROC-AUC](figures/08_densenet_auc.png)

## 8. Normal/Abnormal Classification

![Classifier ROC](figures/09_classifier_roc.png)

![Classifier confusion matrix](figures/10_classifier_confusion_matrix.png)

Classifier validation report:

```text
{classifier_report_text}
```

## 9. Qualitative Examples

The qualitative figures use solid rectangles for consensus ground truth and dashed rectangles for model predictions. These examples should be reviewed for true positives, false positives, false negatives, and localization errors.

See the `figures/qualitative_*.png` files produced by the pipeline.

## 10. Interpretation Guide for the Final Paper

The final discussion should answer:

1. Did multi-radiologist fusion reduce redundant targets?
2. Did Faster R-CNN outperform YOLO for any specific classes?
3. Did the detector ensemble improve mAP@0.40?
4. Did DenseNet calibration improve or reduce detector mAP?
5. Which rare or small-lesion classes remained difficult?
6. Which final strategy was selected by validation rather than by model complexity?

The final submission uses the competition-required **`14 1.0 0 0 1 1`** string whenever no abnormal detection survives post-processing.

## 11. Reproducibility

- Seed: {SEED}
- PyTorch: {torch.__version__}
- Torchvision: {torchvision.__version__}
- CUDA available: {torch.cuda.is_available()}
- GPUs: {", ".join(torch.cuda.get_device_name(i) for i in range(N_GPUS))}
- YOLO epochs: {YOLO_EPOCHS}
- Faster R-CNN epochs: {FRCNN_EPOCHS}
- DenseNet epochs: {CLS_EPOCHS}
- Detection image size: {IMG_SIZE}
- FRCNN DataLoader workers: {FRCNN_NUM_WORKERS}

All numerical result tables are also saved under `tables/`, and machine-readable metrics are saved in `metrics.json`.
"""

    (OUT_DIR / "report.md").write_text(report)

def package_results():
    archive_base = ROOT / "amia_report_bundle"
    if (ROOT / "amia_report_bundle.zip").exists():
        (ROOT / "amia_report_bundle.zip").unlink()
    shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=OUT_DIR,
    )

# ============================================================
# Main
# ============================================================

def main():
    update_status("startup", "Pipeline started")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Submit this pipeline to a SLURM GPU node.")

    print("SLURM_JOB_ID:", os.environ.get("SLURM_JOB_ID", "not-set"))
    print("Work directory:", ROOT)
    print("Data directory:", DATA_ROOT)
    print("Weights directory:", WEIGHTS_DIR)

    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("GPU count:", N_GPUS)
    for i in range(N_GPUS):
        print(
            f"GPU {i}: {torch.cuda.get_device_name(i)} | "
            f"{torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} GB"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    update_status("data", "Locating dataset")
    input_root, data_dir, train_df, test_df, img_size_df, sample_submission = locate_data()
    train_image_map, test_image_map = build_image_maps(
        input_root, data_dir, train_df, sample_submission
    )
    train_df, det_df = preprocess_annotations(train_df, img_size_df)

    update_status("data", "Building multi-radiologist consensus annotations")
    consensus_df = build_consensus(det_df)
    consensus_df.to_csv(TABLE_DIR / "consensus_annotations.csv", index=False)

    image_abnormal = (
        train_df.groupby("image_id")["class_id"]
        .apply(lambda s: int((s != 14).any()))
    )
    abnormal_target_map = image_abnormal.to_dict()

    image_table = pd.DataFrame({
        "image_id": sorted(train_df["image_id"].astype(str).unique())
    })
    image_table["abnormal"] = (
        image_table["image_id"]
        .map(abnormal_target_map)
        .fillna(0)
        .astype(int)
    )

    train_ids_list, val_ids_list = train_test_split(
        image_table["image_id"].tolist(),
        test_size=VAL_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=image_table["abnormal"].values,
    )

    if FAST_DEV_RUN:
        train_ids_list = train_ids_list[:300]
        val_ids_list = val_ids_list[:100]

    train_ids = set(train_ids_list)
    val_ids = set(val_ids_list)

    pd.DataFrame({"image_id": sorted(train_ids)}).to_csv(
        TABLE_DIR / "train_ids.csv", index=False
    )
    pd.DataFrame({"image_id": sorted(val_ids)}).to_csv(
        TABLE_DIR / "val_ids.csv", index=False
    )

    save_eda(train_df, det_df, consensus_df, train_ids, val_ids)

    dataset_stats = {
        "n_train_images": len(train_ids),
        "n_val_images": len(val_ids),
        "n_test_images": len(sample_submission),
        "raw_boxes": len(det_df),
        "consensus_boxes": len(consensus_df),
        "fusion_reduction_pct": 100.0 * (1 - len(consensus_df) / max(len(det_df), 1)),
        "median_box_width": float((det_df["x_max"] - det_df["x_min"]).median()),
        "median_box_height": float((det_df["y_max"] - det_df["y_min"]).median()),
    }

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------
    update_status("yolo", "Preparing and training YOLOv8s")
    _, yaml_path = prepare_yolo_dataset(
        train_ids, val_ids, train_image_map, consensus_df
    )
    yolo_model, yolo_best = train_yolo(input_root, yaml_path)
    save_yolo_curves()

    val_gt = consensus_df[consensus_df.image_id.isin(val_ids)].copy()

    update_status("evaluation", "YOLO validation inference")
    yolo_val_preds = predict_yolo(
        val_ids, train_image_map, yolo_model, YOLO_CONF
    )
    yolo_map40, yolo_ap = evaluate_map40(yolo_val_preds, val_gt)

    # --------------------------------------------------------
    # Faster R-CNN
    # --------------------------------------------------------
    update_status("frcnn", "Training Faster R-CNN")
    frcnn_model, _, frcnn_val_loader, frcnn_best = train_frcnn(
        train_ids, val_ids, train_image_map, consensus_df
    )

    update_status("evaluation", "Faster R-CNN validation inference")
    frcnn_val_preds = predict_frcnn(
        frcnn_model, frcnn_val_loader, FRCNN_CONF
    )
    frcnn_map40, frcnn_ap = evaluate_map40(frcnn_val_preds, val_gt)

    # --------------------------------------------------------
    # Classifier
    # --------------------------------------------------------
    update_status("classifier", "Training DenseNet121")
    cls_model, _, cls_val_loader, cls_best = train_classifier(
        train_ids,
        val_ids,
        train_image_map,
        abnormal_target_map,
    )

    cls_auc, y_true_cls, y_prob_cls, cls_val_ids = evaluate_classifier(
        cls_model, cls_val_loader
    )

    precision, recall, thresholds = precision_recall_curve(
        y_true_cls, y_prob_cls
    )

    cls_threshold = 0.5
    if len(thresholds):
        f1 = (
            2 * precision[:-1] * recall[:-1] /
            np.maximum(precision[:-1] + recall[:-1], 1e-12)
        )
        cls_threshold = float(thresholds[int(np.nanargmax(f1))])

    y_pred_cls = (y_prob_cls >= cls_threshold).astype(int)
    classifier_report_text = classification_report(
        y_true_cls, y_pred_cls, digits=4
    )
    (OUT_DIR / "classifier_report.txt").write_text(classifier_report_text)

    save_classifier_visuals(
        y_true_cls, y_prob_cls, cls_threshold
    )

    cls_val_prob_map = {
        image_id: float(prob)
        for image_id, prob in zip(cls_val_ids, y_prob_cls)
    }

    # --------------------------------------------------------
    # Tune ensemble and hybrid on validation
    # --------------------------------------------------------
    update_status("evaluation", "Tuning detector ensemble")

    ensemble_grid_rows = []
    best_ensemble = None
    best_ensemble_map = -1
    best_ensemble_iou = 0.40
    best_ensemble_thr = DEFAULT_FINAL_CONF

    for iou_thr in [0.30, 0.40, 0.50]:
        fused = ensemble_predictions(
            [yolo_val_preds, frcnn_val_preds],
            iou_thr=iou_thr,
            min_score=0.01,
        )
        thr, m, grid_df = tune_strategy(fused, val_gt)
        for r in grid_df.to_dict("records"):
            r["ensemble_iou"] = iou_thr
            ensemble_grid_rows.append(r)

        if m > best_ensemble_map:
            best_ensemble_map = m
            best_ensemble = fused
            best_ensemble_iou = iou_thr
            best_ensemble_thr = thr

    pd.DataFrame(ensemble_grid_rows).to_csv(
        TABLE_DIR / "ensemble_threshold_grid.csv",
        index=False,
    )

    ensemble_map40, ensemble_ap = evaluate_map40(
        threshold_predictions(best_ensemble, best_ensemble_thr),
        val_gt,
    )

    hybrid_val = calibrate_with_classifier(
        best_ensemble,
        cls_val_prob_map,
    )
    hybrid_thr, hybrid_map40, hybrid_grid = tune_strategy(
        hybrid_val, val_gt
    )
    hybrid_grid.to_csv(
        TABLE_DIR / "hybrid_threshold_grid.csv",
        index=False,
    )
    _, hybrid_ap = evaluate_map40(
        threshold_predictions(hybrid_val, hybrid_thr),
        val_gt,
    )

    # Also tune individual detector thresholds so the final selection is fair.
    yolo_thr, yolo_tuned_map, yolo_grid = tune_strategy(
        yolo_val_preds, val_gt
    )
    frcnn_thr, frcnn_tuned_map, frcnn_grid = tune_strategy(
        frcnn_val_preds, val_gt
    )
    yolo_grid.to_csv(TABLE_DIR / "yolo_threshold_grid.csv", index=False)
    frcnn_grid.to_csv(TABLE_DIR / "frcnn_threshold_grid.csv", index=False)

    # Report raw AP ranking but select final submission strategy from tuned mAP.
    strategies = {
        "YOLOv8s": {
            "map": yolo_tuned_map,
            "threshold": yolo_thr,
        },
        "Faster R-CNN": {
            "map": frcnn_tuned_map,
            "threshold": frcnn_thr,
        },
        "YOLO + Faster R-CNN ensemble": {
            "map": ensemble_map40,
            "threshold": best_ensemble_thr,
        },
        "Hybrid ensemble + DenseNet": {
            "map": hybrid_map40,
            "threshold": hybrid_thr,
        },
    }
    strategy_name = max(strategies, key=lambda k: strategies[k]["map"])
    final_thr = float(strategies[strategy_name]["threshold"])

    metrics = {
        "yolo_map40": float(yolo_map40),
        "frcnn_map40": float(frcnn_map40),
        "ensemble_map40": float(ensemble_map40),
        "hybrid_map40": float(hybrid_map40),
        "classifier_auc": float(cls_auc),
        "classifier_threshold": float(cls_threshold),
        "best_ensemble_iou": float(best_ensemble_iou),
        "best_ensemble_threshold": float(best_ensemble_thr),
        "hybrid_threshold": float(hybrid_thr),
        "selected_strategy": strategy_name,
        "selected_final_threshold": float(final_thr),
        "tuned_strategy_map40": {
            k: float(v["map"]) for k, v in strategies.items()
        },
    }
    RESULTS_JSON.write_text(json.dumps(metrics, indent=2))

    save_ap_comparison(
        yolo_ap, frcnn_ap, ensemble_ap, hybrid_ap
    )
    save_model_summary_plot(metrics)

    selected_val_preds = {
        "YOLOv8s": threshold_predictions(yolo_val_preds, yolo_thr),
        "Faster R-CNN": threshold_predictions(frcnn_val_preds, frcnn_thr),
        "YOLO + Faster R-CNN ensemble": threshold_predictions(
            best_ensemble, best_ensemble_thr
        ),
        "Hybrid ensemble + DenseNet": threshold_predictions(
            hybrid_val, hybrid_thr
        ),
    }[strategy_name]

    save_qualitative_examples(
        val_ids,
        train_image_map,
        val_gt,
        selected_val_preds,
        n=6,
    )

    # --------------------------------------------------------
    # Test inference and submission
    # --------------------------------------------------------
    if RUN_TEST_INFERENCE:
        update_status("test", "Running test inference")

        # Validation tuning is complete. Free large validation prediction
        # containers and DataLoader references before processing 6k+ test
        # images. The previous run hit the 32 GiB SLURM host-RAM limit here.
        try:
            del yolo_val_preds
            del frcnn_val_preds
            del best_ensemble
            del hybrid_val
            del selected_val_preds
            del ensemble_grid_rows
            del yolo_grid
            del frcnn_grid
            del hybrid_grid
            del cls_val_loader
            del y_true_cls
            del y_prob_cls
            del y_pred_cls
            del precision
            del recall
            del thresholds
        except Exception:
            pass

        release_cuda_memory("before full test inference")

        test_ids = sample_submission["image_id"].astype(str).tolist()
        if FAST_DEV_RUN:
            test_ids = test_ids[:100]

        # DenseNet probabilities are only required for hybrid strategy,
        # but generating them is also useful for analysis.
        cls_val_tf = CXRClassificationDataset(
            val_ids,
            train_image_map,
            abnormal_target_map,
            train=False,
        ).tf

        test_abnormal_prob = predict_classifier_test(
            cls_model,
            test_ids,
            test_image_map,
            cls_val_tf,
        )

        # DenseNet is no longer needed after test probabilities are materialized.
        try:
            del cls_model
            del cls_val_tf
        except Exception:
            pass
        release_cuda_memory("after DenseNet test inference")

        yolo_test_preds = predict_yolo(
            test_ids,
            test_image_map,
            yolo_model,
            YOLO_CONF,
        )
        frcnn_test_preds = predict_frcnn_test(
            frcnn_model,
            test_ids,
            test_image_map,
            FRCNN_CONF,
        )

        test_ensemble = ensemble_predictions(
            [yolo_test_preds, frcnn_test_preds],
            iou_thr=best_ensemble_iou,
            min_score=0.01,
        )

        test_hybrid = calibrate_with_classifier(
            test_ensemble,
            test_abnormal_prob,
        )

        if strategy_name == "YOLOv8s":
            final_test_preds = yolo_test_preds
        elif strategy_name == "Faster R-CNN":
            final_test_preds = frcnn_test_preds
        elif strategy_name == "YOLO + Faster R-CNN ensemble":
            final_test_preds = test_ensemble
        else:
            final_test_preds = test_hybrid

        preds_by_image = defaultdict(list)
        for p in final_test_preds:
            preds_by_image[p["image_id"]].append(p)

        submission = pd.DataFrame({"image_id": test_ids})
        submission["PredictionString"] = [
            format_prediction_string(
                preds_by_image[image_id],
                image_id,
                final_thr,
            )
            for image_id in test_ids
        ]

        if not FAST_DEV_RUN:
            assert len(submission) == len(sample_submission)
            assert submission["image_id"].is_unique

        submission.to_csv(OUT_DIR / "submission.csv", index=False)

        # Save compact test classifier scores for later analysis.
        pd.DataFrame({
            "image_id": list(test_abnormal_prob.keys()),
            "p_abnormal": list(test_abnormal_prob.values()),
        }).to_csv(
            TABLE_DIR / "test_abnormal_probabilities.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Report + metadata + bundle
    # --------------------------------------------------------
    write_report(
        metrics,
        dataset_stats,
        classifier_report_text,
        strategy_name,
        final_thr,
    )

    metadata = {
        "seed": SEED,
        "image_size": IMG_SIZE,
        "validation_fraction": VAL_SIZE,
        "annotation_fusion_iou": WBF_IOU_THR,
        "evaluation_iou": EVAL_IOU_THR,
        "yolo_epochs": YOLO_EPOCHS,
        "frcnn_epochs": FRCNN_EPOCHS,
        "classifier_epochs": CLS_EPOCHS,
        "frcnn_num_workers": FRCNN_NUM_WORKERS,
        "other_num_workers": OTHER_NUM_WORKERS,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "data_dir": str(DATA_ROOT),
        "work_dir": str(ROOT),
        "weights_dir": str(WEIGHTS_DIR),
        "yolo_batch": YOLO_BATCH,
        "frcnn_batch": FRCNN_BATCH,
        "classifier_batch": CLS_BATCH,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [
            torch.cuda.get_device_name(i)
            for i in range(N_GPUS)
        ],
    }

    try:
        import ultralytics
        metadata["ultralytics"] = ultralytics.__version__
    except Exception:
        pass

    (OUT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    package_results()
    mark_stage_done("pipeline")
    update_status(
        "complete",
        "Pipeline complete. report.md, submission.csv, figures, tables, and zip bundle are ready.",
        progress=1.0,
        extra={"selected_strategy": strategy_name},
    )

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("Report:", OUT_DIR / "report.md")
    print("Submission:", OUT_DIR / "submission.csv")
    print("Bundle:", ROOT / "amia_report_bundle.zip")
    print("Selected strategy:", strategy_name)
    print("="*70, flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        update_status(
            "failed",
            f"{type(e).__name__}: {e}",
            extra={"traceback": traceback.format_exc()},
        )
        traceback.print_exc()
        raise
