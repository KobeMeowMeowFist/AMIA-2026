#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMIA Public Challenge 2026 — v6 RT-DETR raw-label H100 experiment

This is an independently implemented controlled ablation:
  - preserve ALL abnormal radiologist boxes from train.csv (class 0..13)
  - no consensus / WBF label fusion during training
  - RT-DETR-L
  - 640x640 training
  - single-GPU AutoBatch targeting ~75% VRAM
  - longer training with early stopping
  - deterministic=False for speed
  - test inference saved once at low confidence
  - multiple submission thresholds generated without rerunning inference
  - final boxes are always mapped back to original dim0/dim1 coordinates

This does not copy the uploaded public notebook code; it uses the public
configuration only as a benchmark hypothesis to test against our prior v5.
"""

from __future__ import annotations

import os
import gc
import json
import ctypes
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from ultralytics import RTDETR


def env_int(name, default):
    return int(os.getenv(name, str(default)))

def env_float(name, default):
    return float(os.getenv(name, str(default)))

def env_str(name, default):
    return os.getenv(name, default)

def env_bool(name, default=False):
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "y"}


SEED = env_int("AMIA_V6_SEED", 42)
IMG_SIZE = env_int("AMIA_V6_IMG_SIZE", 640)
MAX_EPOCHS = env_int("AMIA_V6_EPOCHS", 220)
PATIENCE = env_int("AMIA_V6_PATIENCE", 30)
WORKERS = env_int("AMIA_V6_WORKERS", 8)

# Ultralytics accepts a float in (0,1) as desired GPU-memory fraction.
BATCH_FRACTION = env_float("AMIA_V6_BATCH_FRACTION", 0.75)

PRED_CONF_FLOOR = env_float("AMIA_V6_PRED_CONF_FLOOR", 0.02)
PRED_IOU = env_float("AMIA_V6_PRED_IOU", 0.40)
PRED_BATCH = env_int("AMIA_V6_PRED_BATCH", 16)
PRED_CHUNK = env_int("AMIA_V6_PRED_CHUNK", 128)

DATA_DIR = Path(env_str(
    "AMIA_DATA_DIR",
    "/scratch/fanm01/CV/amia_slurm_project_curta/data/amia-public-challenge-2026",
))
WEIGHTS_DIR = Path(env_str(
    "AMIA_WEIGHTS_DIR",
    "/scratch/fanm01/CV/amia_slurm_project_curta/weights",
))
WORK_DIR = Path(env_str(
    "AMIA_V6_WORK_DIR",
    "/scratch/xiaolil02/amia_2026_work/v6_rtdetr_raw640",
))

PRETRAIN = Path(env_str(
    "AMIA_V6_RTDETR_PRETRAIN",
    str(WEIGHTS_DIR / "rtdetr-l.pt"),
))

DATASET_DIR = WORK_DIR / "dataset"
RUN_DIR = WORK_DIR / "runs"
RESULTS_DIR = WORK_DIR / "results"
STATE_DIR = WORK_DIR / "state"

for p in [WORK_DIR, DATASET_DIR, RUN_DIR, RESULTS_DIR, STATE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

RUN_NAME = "rtdetr_l_raw640"
NO_FINDING = 14
N_CLASSES = 14

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
}

SUB_THRESHOLDS = [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def status(stage, message):
    print(f"[STATUS] {stage}: {message}", flush=True)
    (STATE_DIR / "status.json").write_text(
        json.dumps({"stage": stage, "message": message}, indent=2),
        encoding="utf-8",
    )


def cleanup(label=""):
    gc.collect()
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
        msg = f"[MEM] {label}"
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            msg += f" | GPU free={free_b/1024**3:.2f}/{total_b/1024**3:.2f} GiB"
        try:
            import psutil
            rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
            msg += f" | host RSS={rss:.2f} GiB"
        except Exception:
            pass
        print(msg, flush=True)


def load_tables():
    train_csv = DATA_DIR / "train.csv"
    test_csv = DATA_DIR / "test.csv"
    size_csv = DATA_DIR / "img_size.csv"
    sample_csv = DATA_DIR / "sample_submission.csv"

    for p in [train_csv, test_csv, size_csv, sample_csv]:
        if not p.exists():
            raise FileNotFoundError(p)

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    sizes = pd.read_csv(size_csv)
    sample = pd.read_csv(sample_csv)

    train["image_id"] = train["image_id"].astype(str).map(lambda x: Path(x).stem)
    test["image_id"] = test["image_id"].astype(str).map(lambda x: Path(x).stem)
    sizes["image_id"] = sizes["image_id"].astype(str).map(lambda x: Path(x).stem)
    sample["image_id"] = sample["image_id"].astype(str).map(lambda x: Path(x).stem)

    return train, test, sizes, sample


def discover_images(train_ids, test_ids):
    train_ids = set(map(str, train_ids))
    test_ids = set(map(str, test_ids))

    train_map, test_map = {}, {}

    status("data", "Scanning PNG files")
    for p in DATA_DIR.rglob("*.png"):
        sid = p.stem
        low = str(p).lower()
        if sid in train_ids:
            if sid not in train_map or "train" in low:
                train_map[sid] = p
        if sid in test_ids:
            if sid not in test_map or "test" in low:
                test_map[sid] = p

    missing_tr = train_ids - set(train_map)
    missing_te = test_ids - set(test_map)

    if missing_tr:
        raise RuntimeError(f"Missing {len(missing_tr)} train PNGs; examples={list(missing_tr)[:5]}")
    if missing_te:
        raise RuntimeError(f"Missing {len(missing_te)} test PNGs; examples={list(missing_te)[:5]}")

    print(f"Train PNGs mapped: {len(train_map)}")
    print(f"Test PNGs mapped:  {len(test_map)}")
    return train_map, test_map


def make_strata(all_ids, abnormal):
    """Stratify by the rarest abnormal class present in each image; normals separate."""
    class_freq = abnormal["class_id"].astype(int).value_counts().to_dict()
    by_image = (
        abnormal.groupby("image_id")["class_id"]
        .apply(lambda x: sorted(set(map(int, x))))
        .to_dict()
    )

    labels = []
    for iid in all_ids:
        classes = by_image.get(str(iid), [])
        if not classes:
            labels.append("normal")
        else:
            rarest = min(classes, key=lambda c: class_freq.get(c, 10**9))
            labels.append(f"class_{rarest}")

    s = pd.Series(labels)
    counts = s.value_counts()
    labels = [x if counts[x] >= 4 else "rare_other" for x in labels]
    return np.asarray(labels)


def symlink_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)


def build_dataset(train, sizes, train_map):
    marker = DATASET_DIR / ".prepared"
    yaml_path = DATASET_DIR / "data.yaml"
    split_path = RESULTS_DIR / "split_v6.csv"

    if marker.exists() and yaml_path.exists() and split_path.exists():
        print("Dataset already prepared; reusing.")
        return yaml_path

    status("data", "Building raw-radiologist-box dataset")

    abnormal = train[train["class_id"].astype(int).between(0, 13)].copy()

    for c in ["x_min", "y_min", "x_max", "y_max"]:
        abnormal[c] = pd.to_numeric(abnormal[c], errors="coerce")
    abnormal = abnormal.dropna(subset=["x_min", "y_min", "x_max", "y_max"])

    sizes2 = sizes[["image_id", "dim0", "dim1"]].copy()
    abnormal = abnormal.merge(sizes2, on="image_id", how="left")

    if abnormal[["dim0", "dim1"]].isna().any().any():
        raise RuntimeError("Missing image sizes for training annotations")

    valid = (
        (abnormal["x_max"] > abnormal["x_min"]) &
        (abnormal["y_max"] > abnormal["y_min"])
    )
    abnormal = abnormal[valid].copy()

    all_ids = np.asarray(sorted(train["image_id"].unique()))
    strata = make_strata(all_ids, abnormal)

    tr_ids, va_ids = train_test_split(
        all_ids,
        test_size=0.15,
        random_state=SEED,
        shuffle=True,
        stratify=strata,
    )

    split_df = pd.DataFrame({
        "image_id": np.concatenate([tr_ids, va_ids]),
        "split": ["train"] * len(tr_ids) + ["val"] * len(va_ids),
    })
    split_df.to_csv(split_path, index=False)

    ann = defaultdict(list)
    for r in abnormal.itertuples(index=False):
        ann[str(r.image_id)].append(r)

    # IMPORTANT: every abnormal CSV row is kept. No radiologist fusion here.
    for split, ids in [("train", tr_ids), ("val", va_ids)]:
        for iid in tqdm(ids, desc=f"Prepare raw labels {split}"):
            iid = str(iid)
            src = train_map[iid]
            symlink_image(src, DATASET_DIR / "images" / split / src.name)

            label_path = DATASET_DIR / "labels" / split / f"{iid}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)

            lines = []
            for r in ann.get(iid, []):
                orig_h = float(r.dim0)
                orig_w = float(r.dim1)

                x1 = np.clip(float(r.x_min), 0, orig_w)
                x2 = np.clip(float(r.x_max), 0, orig_w)
                y1 = np.clip(float(r.y_min), 0, orig_h)
                y2 = np.clip(float(r.y_max), 0, orig_h)

                if x2 <= x1 or y2 <= y1:
                    continue

                xc = ((x1 + x2) / 2.0) / orig_w
                yc = ((y1 + y2) / 2.0) / orig_h
                bw = (x2 - x1) / orig_w
                bh = (y2 - y1) / orig_h

                lines.append(
                    f"{int(r.class_id)} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}"
                )

            label_path.write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )

    yaml_data = {
        "path": str(DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": {i: CLASS_NAMES[i] for i in range(N_CLASSES)},
    }
    yaml_path.write_text(
        yaml.safe_dump(yaml_data, sort_keys=False),
        encoding="utf-8",
    )

    stats = {
        "train_images": int(len(tr_ids)),
        "val_images": int(len(va_ids)),
        "raw_abnormal_rows": int(len(abnormal)),
        "normal_images": int(sum(iid not in ann for iid in all_ids)),
        "imgsz": IMG_SIZE,
    }
    (RESULTS_DIR / "dataset_stats_v6.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    marker.write_text("prepared\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return yaml_path


def train_model(data_yaml):
    weights_dir = RUN_DIR / RUN_NAME / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    complete_marker = STATE_DIR / "train_complete.done"

    if complete_marker.exists() and best_pt.exists():
        print(f"Training already complete. Reusing: {best_pt}")
        return best_pt

    if not PRETRAIN.exists() and not last_pt.exists():
        raise FileNotFoundError(
            f"Missing pretrained weight {PRETRAIN}. "
            "Reuse the rtdetr-l.pt already downloaded for v5."
        )

    status("train", "RT-DETR-L raw-label 640 training")

    cleanup("before training")

    if last_pt.exists():
        print(f"Resuming interrupted run from: {last_pt}", flush=True)
        model = RTDETR(str(last_pt))
        model.train(resume=True)
    else:
        model = RTDETR(str(PRETRAIN))
        model.train(
            data=str(data_yaml),
            epochs=MAX_EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_FRACTION,
            workers=WORKERS,
            device=0,
            project=str(RUN_DIR),
            name=RUN_NAME,
            exist_ok=True,
            pretrained=True,
            optimizer="auto",
            cos_lr=True,
            patience=PATIENCE,
            amp=True,
            cache=False,
            plots=True,
            save=True,
            save_period=10,
            seed=SEED,
            deterministic=False,
            verbose=True,
        )

    del model
    cleanup("after training")

    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found after training: {best_pt}")

    complete_marker.write_text("done\n", encoding="utf-8")
    return best_pt


def predict_test(best_pt, test_ids, test_map, sizes):
    raw_csv = RESULTS_DIR / "test_predictions_v6_raw.csv"
    complete_marker = STATE_DIR / "test_inference_complete.done"

    if complete_marker.exists() and raw_csv.exists():
        print(f"Test inference already complete. Reusing: {raw_csv}")
        return pd.read_csv(raw_csv)

    status("test", "Running RT-DETR test inference once at low confidence")
    model = RTDETR(str(best_pt))

    size_lookup = (
        sizes.set_index("image_id")[["dim0", "dim1"]]
        .to_dict("index")
    )

    rows = []
    test_ids = list(map(str, test_ids))

    cleanup("before test inference")
    pbar = tqdm(total=len(test_ids), desc="RT-DETR v6 test")

    for start in range(0, len(test_ids), PRED_CHUNK):
        chunk_ids = test_ids[start:start + PRED_CHUNK]
        paths = [str(test_map[i]) for i in chunk_ids]

        results = model.predict(
            source=paths,
            imgsz=IMG_SIZE,
            conf=PRED_CONF_FLOOR,
            iou=PRED_IOU,
            device=0,
            batch=PRED_BATCH,
            half=torch.cuda.is_available(),
            stream=True,
            verbose=False,
        )

        for res in results:
            iid = Path(res.path).stem

            # Ultralytics returns boxes in SOURCE-image coordinates, not 640-space.
            with Image.open(res.path) as im:
                src_w, src_h = im.size

            orig_h = float(size_lookup[iid]["dim0"])
            orig_w = float(size_lookup[iid]["dim1"])

            if res.boxes is not None and len(res.boxes):
                boxes = res.boxes.xyxy.detach().cpu().numpy()
                scores = res.boxes.conf.detach().cpu().numpy()
                classes = res.boxes.cls.detach().cpu().numpy().astype(int)

                for b, score, cid in zip(boxes, scores, classes):
                    if not (0 <= int(cid) <= 13):
                        continue

                    x1, y1, x2, y2 = map(float, b)

                    # SOURCE PNG coordinates -> ORIGINAL radiograph coordinates.
                    x1 = np.clip(x1 * orig_w / src_w, 0, orig_w)
                    x2 = np.clip(x2 * orig_w / src_w, 0, orig_w)
                    y1 = np.clip(y1 * orig_h / src_h, 0, orig_h)
                    y2 = np.clip(y2 * orig_h / src_h, 0, orig_h)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    rows.append({
                        "image_id": iid,
                        "class_id": int(cid),
                        "score": float(score),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    })

            pbar.update(1)
            del res

        del results, paths, chunk_ids
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()
    del model
    cleanup("after test inference")

    pred_df = pd.DataFrame(
        rows,
        columns=["image_id", "class_id", "score", "x1", "y1", "x2", "y2"],
    )
    pred_df.to_csv(raw_csv, index=False)
    complete_marker.write_text("done\n", encoding="utf-8")
    return pred_df


def make_submission(pred_df, sample, threshold, out_csv):
    pred = pred_df[pred_df["score"] >= threshold].copy()
    pred = pred.sort_values(["image_id", "score"], ascending=[True, False])

    grouped = {iid: g for iid, g in pred.groupby("image_id", sort=False)}
    rows = []

    for iid in sample["image_id"].astype(str):
        g = grouped.get(iid)

        if g is None or g.empty:
            rows.append({
                "image_id": iid,
                "PredictionString": "14 1.0 0 0 1 1",
            })
            continue

        tokens = []
        for r in g.itertuples(index=False):
            tokens.extend([
                str(int(r.class_id)),
                f"{float(r.score):.6f}",
                f"{float(r.x1):.2f}",
                f"{float(r.y1):.2f}",
                f"{float(r.x2):.2f}",
                f"{float(r.y2):.2f}",
            ])

        rows.append({
            "image_id": iid,
            "PredictionString": " ".join(tokens) if tokens else "14 1.0 0 0 1 1",
        })

    out = pd.DataFrame(rows)

    if len(out) != 6427:
        raise RuntimeError(f"Submission row count={len(out)}, expected 6427")
    if out["image_id"].nunique() != 6427:
        raise RuntimeError("Submission does not contain 6427 unique image IDs")

    out.to_csv(out_csv, index=False)

    counts = []
    for s in out["PredictionString"]:
        t = str(s).split()
        counts.append(0 if (not t or t[0] == "14") else len(t) // 6)

    counts = pd.Series(counts)
    print(
        f"{out_csv.name}: threshold={threshold} | "
        f"no_finding={(counts==0).sum()} | "
        f"abnormal={(counts>0).sum()} | "
        f"total_boxes={int(counts.sum())} | "
        f"mean={counts.mean():.2f} | "
        f"median={counts.median():.1f} | "
        f"max={int(counts.max())}",
        flush=True,
    )


def main():
    seed_everything()

    print("=" * 76)
    print("AMIA v6 — RT-DETR-L RAW LABELS / 640 / LONG TRAINING / H100 AUTOBATCH")
    print("=" * 76)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Target AutoBatch VRAM fraction: {BATCH_FRACTION:.2f}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Patience: {PATIENCE}")
    print(f"Workers: {WORKERS}")
    print(f"Work dir: {WORK_DIR}")

    train, test, sizes, sample = load_tables()

    train_ids = sorted(train["image_id"].unique())
    test_ids = sample["image_id"].astype(str).tolist()

    if len(test_ids) != 6427 or len(set(test_ids)) != 6427:
        raise RuntimeError(
            f"Expected 6427 unique sample_submission IDs; "
            f"got rows={len(test_ids)}, unique={len(set(test_ids))}"
        )

    train_map, test_map = discover_images(train_ids, test_ids)
    data_yaml = build_dataset(train, sizes, train_map)
    best_pt = train_model(data_yaml)

    pred_df = predict_test(best_pt, test_ids, test_map, sizes)

    status("submission", "Generating confidence-threshold candidates")
    for thr in SUB_THRESHOLDS:
        tag = str(thr).replace(".", "p")
        make_submission(
            pred_df,
            sample,
            threshold=thr,
            out_csv=RESULTS_DIR / f"submission_v6_conf{tag}.csv",
        )

    summary = {
        "model": "RT-DETR-L",
        "training_labels": "raw radiologist boxes, no fusion",
        "imgsz": IMG_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "batch_fraction": BATCH_FRACTION,
        "workers": WORKERS,
        "deterministic": False,
        "test_prediction_floor": PRED_CONF_FLOOR,
        "test_iou": PRED_IOU,
        "threshold_candidates": SUB_THRESHOLDS,
        "best_pt": str(best_pt),
    }
    (RESULTS_DIR / "experiment_v6.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    (STATE_DIR / "pipeline_complete.done").write_text("done\n", encoding="utf-8")
    status("complete", "v6 pipeline complete")

    print("=" * 76)
    print("PIPELINE COMPLETE")
    print(f"Best model: {best_pt}")
    print(f"Results: {RESULTS_DIR}")
    print("Recommended first Kaggle candidates:")
    print("  submission_v6_conf0p05.csv")
    print("  submission_v6_conf0p075.csv")
    print("  submission_v6_conf0p1.csv")
    print("=" * 76)


if __name__ == "__main__":
    main()
