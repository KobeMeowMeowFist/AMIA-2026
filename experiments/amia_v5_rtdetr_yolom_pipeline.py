#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMIA Public Challenge 2026 — stronger v5 pipeline
=================================================
Goal:
  * 3-fold RT-DETR-L
  * 3-fold YOLOv8m
  * relaxed multi-radiologist fusion
  * pooled out-of-fold (OOF) VOC mAP@0.40
  * per-class family selection (RT-DETR vs YOLOv8m)
  * per-class confidence threshold tuning on OOF only
  * 3-fold test-time model ensemble with WBF-like fusion
  * output bounding boxes converted BACK to original image coordinates

This script intentionally does NOT reuse the old Faster R-CNN ensemble because
the previous experiment showed that it reduced the final validation mAP.

Environment variables are documented in run_v5_h100.sbatch.
"""

from __future__ import annotations

import os
import gc
import json
import math
import random
import shutil
import ctypes
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import torch
from ultralytics import YOLO, RTDETR


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
def env_int(name, default):
    return int(os.getenv(name, str(default)))

def env_float(name, default):
    return float(os.getenv(name, str(default)))

def env_str(name, default):
    return os.getenv(name, default)

def env_bool(name, default=False):
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "y"}

SEED = env_int("AMIA_SEED", 42)
IMG_SIZE = env_int("AMIA_V5_IMG_SIZE", 1024)
N_FOLDS = env_int("AMIA_V5_FOLDS", 3)

RT_EPOCHS = env_int("AMIA_RTDETR_EPOCHS", 45)
YOLO_EPOCHS = env_int("AMIA_YOLOM_EPOCHS", 55)
RT_BATCH = env_int("AMIA_RTDETR_BATCH", 8)
YOLO_BATCH = env_int("AMIA_YOLOM_BATCH", 12)
WORKERS = env_int("AMIA_V5_WORKERS", 4)

PRED_CONF = env_float("AMIA_V5_PRED_CONF", 0.005)
PRED_BATCH = env_int("AMIA_V5_PRED_BATCH", 1)
PRED_CHUNK = env_int("AMIA_V5_PRED_CHUNK", 64)
WBF_IOU = env_float("AMIA_V5_WBF_IOU", 0.35)
LABEL_FUSION_IOU = env_float("AMIA_V5_LABEL_FUSION_IOU", 0.35)
EVAL_IOU = 0.40

DATA_DIR = Path(env_str(
    "AMIA_DATA_DIR",
    "/scratch/fanm01/CV/amia_slurm_project_curta/data/amia-public-challenge-2026",
))
WORK_DIR = Path(env_str(
    "AMIA_V5_WORK_DIR",
    "/scratch/xiaolil02/amia_2026_work/v5_rtdetr_yolom",
))
WEIGHTS_DIR = Path(env_str(
    "AMIA_WEIGHTS_DIR",
    "/scratch/fanm01/CV/amia_slurm_project_curta/weights",
))
RTDETR_PRETRAIN = Path(env_str(
    "AMIA_RTDETR_PRETRAIN",
    str(WEIGHTS_DIR / "rtdetr-l.pt"),
))
YOLOM_PRETRAIN = Path(env_str(
    "AMIA_YOLOM_PRETRAIN",
    str(WEIGHTS_DIR / "yolov8m.pt"),
))

RUN_RTDETR = env_bool("AMIA_RUN_RTDETR", True)
RUN_YOLOM = env_bool("AMIA_RUN_YOLOM", True)
RUN_TEST = env_bool("AMIA_RUN_TEST", True)

STATE_DIR = WORK_DIR / "state"
DATASET_DIR = WORK_DIR / "fold_datasets"
RUNS_DIR = WORK_DIR / "runs"
OOF_DIR = WORK_DIR / "oof"
RESULTS_DIR = WORK_DIR / "results"

for p in [WORK_DIR, STATE_DIR, DATASET_DIR, RUNS_DIR, OOF_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

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
BBOX_COLS = ["x_min", "y_min", "x_max", "y_max"]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        gpu = ""
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            gpu = f" GPU_free={free_b/1024**3:.2f}/{total_b/1024**3:.2f} GiB"
        rss = ""
        try:
            import psutil
            rss = f" host_RSS={psutil.Process(os.getpid()).memory_info().rss/1024**3:.2f} GiB"
        except Exception:
            pass
        print(f"[MEM] {label}:{gpu}{rss}", flush=True)

def status(stage, msg):
    print(f"[STATUS] {stage}: {msg}", flush=True)
    (STATE_DIR / "status.json").write_text(
        json.dumps({"stage": stage, "message": msg}, indent=2),
        encoding="utf-8",
    )

def done(name):
    return (STATE_DIR / f"{name}.done").exists()

def mark_done(name):
    (STATE_DIR / f"{name}.done").write_text("done\n", encoding="utf-8")

def find_pngs():
    pngs = list(DATA_DIR.rglob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG images found under {DATA_DIR}")
    return pngs

def locate_files():
    train_csv = DATA_DIR / "train.csv"
    test_csv = DATA_DIR / "test.csv"
    size_csv = DATA_DIR / "img_size.csv"
    sample_csv = DATA_DIR / "sample_submission.csv"
    for p in [train_csv, test_csv, size_csv, sample_csv]:
        if not p.exists():
            raise FileNotFoundError(p)
    return (
        pd.read_csv(train_csv),
        pd.read_csv(test_csv),
        pd.read_csv(size_csv),
        pd.read_csv(sample_csv),
    )

def build_image_maps(train_df, sample_df):
    pngs = find_pngs()
    train_ids = set(train_df["image_id"].astype(str).unique())
    test_ids = set(sample_df["image_id"].astype(str).unique())

    tr, te = {}, {}
    for p in pngs:
        sid = p.stem
        low = str(p).lower()
        if sid in train_ids and ("train" in low or sid not in test_ids):
            tr.setdefault(sid, p)
        if sid in test_ids and ("test" in low or sid not in train_ids):
            te.setdefault(sid, p)
    for p in pngs:
        sid = p.stem
        if sid in train_ids:
            tr.setdefault(sid, p)
        if sid in test_ids:
            te.setdefault(sid, p)

    if len(tr) != len(train_ids):
        raise RuntimeError(f"Missing train PNGs: {len(train_ids)-len(tr)}")
    if len(te) != len(test_ids):
        raise RuntimeError(f"Missing test PNGs: {len(test_ids)-len(te)}")
    return tr, te


# ---------------------------------------------------------------------
# Annotation processing
# ---------------------------------------------------------------------
def box_iou(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, float(a[2]-a[0])) * max(0.0, float(a[3]-a[1]))
    ab = max(0.0, float(b[2]-b[0])) * max(0.0, float(b[3]-b[1]))
    u = aa + ab - inter
    return inter/u if u > 0 else 0.0

def preprocess_annotations(train_df, size_df):
    t = train_df.copy()
    t["image_id"] = t["image_id"].astype(str)
    t["class_id"] = t["class_id"].astype(int)

    s = size_df.copy()
    s["image_id"] = s["image_id"].astype(str)

    det = t[t["class_id"].between(0, 13)].copy()
    for c in BBOX_COLS:
        det[c] = pd.to_numeric(det[c], errors="coerce")
    det = det.dropna(subset=BBOX_COLS)
    det = det.merge(s[["image_id", "dim0", "dim1"]], on="image_id", how="left")
    if det[["dim0", "dim1"]].isna().any().any():
        raise ValueError("Missing img_size rows for training annotations")

    det["x_min"] = det["x_min"] / det["dim1"] * IMG_SIZE
    det["x_max"] = det["x_max"] / det["dim1"] * IMG_SIZE
    det["y_min"] = det["y_min"] / det["dim0"] * IMG_SIZE
    det["y_max"] = det["y_max"] / det["dim0"] * IMG_SIZE
    for c in BBOX_COLS:
        det[c] = det[c].clip(0, IMG_SIZE)
    det = det[(det.x_max > det.x_min) & (det.y_max > det.y_min)].reset_index(drop=True)
    return t, det

def relaxed_fusion(det_df, iou_thr=LABEL_FUSION_IOU):
    """
    Merge duplicate annotations from different radiologists but NEVER discard
    singleton lesions. Lower IoU than the original strict version makes this
    a relaxed radiologist-consensus representation.
    """
    records = []
    for (image_id, cid), grp in tqdm(
        det_df.groupby(["image_id", "class_id"]),
        desc="Relaxed radiologist fusion",
    ):
        clusters = []
        for r in grp.itertuples(index=False):
            box = np.array([r.x_min, r.y_min, r.x_max, r.y_max], dtype=float)
            best = None
            best_iou = -1
            for j, cl in enumerate(clusters):
                center = np.average(
                    np.stack(cl["boxes"]),
                    axis=0,
                    weights=np.asarray(cl["weights"]),
                )
                i = box_iou(box, center)
                if i >= iou_thr and i > best_iou:
                    best, best_iou = j, i
            rad = str(getattr(r, "rad_id", "unknown"))
            if best is None:
                clusters.append({"boxes": [box], "weights": [1.0], "rads": [rad]})
            else:
                clusters[best]["boxes"].append(box)
                clusters[best]["weights"].append(1.0)
                clusters[best]["rads"].append(rad)

        for cl in clusters:
            fused = np.average(
                np.stack(cl["boxes"]),
                axis=0,
                weights=np.asarray(cl["weights"]),
            )
            records.append({
                "image_id": str(image_id),
                "class_id": int(cid),
                "x_min": fused[0],
                "y_min": fused[1],
                "x_max": fused[2],
                "y_max": fused[3],
                "n_boxes_fused": len(cl["boxes"]),
                "n_radiologists": len(set(cl["rads"])),
            })
    return pd.DataFrame(records)

def make_stratification_labels(all_ids, fused_df):
    """Use the rarest abnormal class in each image as the stratification label."""
    counts = fused_df["class_id"].value_counts().to_dict()
    by_img = fused_df.groupby("image_id")["class_id"].apply(lambda x: sorted(set(map(int, x)))).to_dict()
    labels = []
    for iid in all_ids:
        cls = by_img.get(str(iid), [])
        if not cls:
            labels.append("normal")
        else:
            rarest = min(cls, key=lambda c: counts.get(c, 10**9))
            labels.append(f"c{rarest}")
    # Collapse ultra-rare strata that cannot support K folds.
    vc = pd.Series(labels).value_counts()
    labels = [x if vc[x] >= N_FOLDS else "rare_other" for x in labels]
    return np.asarray(labels)


# ---------------------------------------------------------------------
# Fold dataset creation
# ---------------------------------------------------------------------
def safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)

def write_yolo_label(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in rows:
        xc = ((r.x_min + r.x_max) / 2) / IMG_SIZE
        yc = ((r.y_min + r.y_max) / 2) / IMG_SIZE
        w = (r.x_max - r.x_min) / IMG_SIZE
        h = (r.y_max - r.y_min) / IMG_SIZE
        xc, yc, w, h = [float(np.clip(v, 0, 1)) for v in [xc, yc, w, h]]
        if w > 0 and h > 0:
            lines.append(f"{int(r.class_id)} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def prepare_fold_dataset(fold, train_ids, val_ids, image_map, fused_df):
    root = DATASET_DIR / f"fold{fold}"
    marker = root / ".prepared"
    yaml_path = root / "data.yaml"
    if marker.exists() and yaml_path.exists():
        return yaml_path

    ann = defaultdict(list)
    for r in fused_df.itertuples(index=False):
        ann[str(r.image_id)].append(r)

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for iid in tqdm(ids, desc=f"Prepare fold{fold} {split}"):
            src = image_map[str(iid)]
            safe_symlink(src, root / "images" / split / src.name)
            write_yolo_label(root / "labels" / split / f"{iid}.txt", ann.get(str(iid), []))

    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {i: CLASS_NAMES[i] for i in range(14)},
    }
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    marker.write_text("prepared\n", encoding="utf-8")
    return yaml_path


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def best_path(family, fold):
    return RUNS_DIR / family / f"fold{fold}" / "weights" / "best.pt"

def train_one(family, fold, data_yaml):
    out_best = best_path(family, fold)
    if out_best.exists():
        print(f"{family} fold {fold}: reusing {out_best}")
        return out_best

    cleanup(f"before {family} fold{fold} training")
    project = RUNS_DIR / family

    if family == "rtdetr":
        if not RTDETR_PRETRAIN.exists():
            raise FileNotFoundError(
                f"Missing {RTDETR_PRETRAIN}. Run prepare_v5_weights.py on the login node."
            )
        model = RTDETR(str(RTDETR_PRETRAIN))
        model.train(
            data=str(data_yaml),
            imgsz=IMG_SIZE,
            epochs=RT_EPOCHS,
            batch=RT_BATCH,
            workers=WORKERS,
            device=0,
            project=str(project),
            name=f"fold{fold}",
            exist_ok=True,
            pretrained=True,
            optimizer="auto",
            cos_lr=True,
            patience=12,
            amp=True,
            cache=False,
            plots=False,
            seed=SEED + fold,
            deterministic=True,
            verbose=True,
        )
    elif family == "yolom":
        if not YOLOM_PRETRAIN.exists():
            raise FileNotFoundError(
                f"Missing {YOLOM_PRETRAIN}. Run prepare_v5_weights.py on the login node."
            )
        model = YOLO(str(YOLOM_PRETRAIN))
        model.train(
            data=str(data_yaml),
            imgsz=IMG_SIZE,
            epochs=YOLO_EPOCHS,
            batch=YOLO_BATCH,
            workers=WORKERS,
            device=0,
            project=str(project),
            name=f"fold{fold}",
            exist_ok=True,
            pretrained=True,
            optimizer="auto",
            cos_lr=True,
            patience=15,
            amp=True,
            cache=False,
            plots=False,
            seed=SEED + 100 + fold,
            deterministic=True,
            verbose=True,
            close_mosaic=10,
        )
    else:
        raise ValueError(family)

    del model
    cleanup(f"after {family} fold{fold} training")
    if not out_best.exists():
        raise FileNotFoundError(f"Training finished but best.pt missing: {out_best}")
    return out_best


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------
def load_model(family, weight):
    if family == "rtdetr":
        return RTDETR(str(weight))
    return YOLO(str(weight))

def predict_model(family, weight, ids, image_map, desc="inference"):
    model = load_model(family, weight)
    preds = []
    ids = list(map(str, ids))
    cleanup(f"before {family} {desc}")

    pbar = tqdm(total=len(ids), desc=f"{family} {desc}")
    for st in range(0, len(ids), PRED_CHUNK):
        chunk_ids = ids[st:st+PRED_CHUNK]
        paths = [str(image_map[i]) for i in chunk_ids]
        results = model.predict(
            source=paths,
            imgsz=IMG_SIZE,
            conf=PRED_CONF,
            device=0,
            batch=PRED_BATCH,
            half=torch.cuda.is_available(),
            stream=True,
            verbose=False,
        )
        for res in results:
            iid = Path(res.path).stem
            if res.boxes is not None and len(res.boxes):
                boxes = res.boxes.xyxy.detach().cpu().numpy()
                scores = res.boxes.conf.detach().cpu().numpy()
                cls = res.boxes.cls.detach().cpu().numpy().astype(int)
                for b, s, c in zip(boxes, scores, cls):
                    if 0 <= int(c) <= 13:
                        preds.append({
                            "image_id": iid,
                            "class_id": int(c),
                            "score": float(s),
                            "box": np.asarray(b, dtype=float),
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
    cleanup(f"after {family} {desc}")
    return preds

def preds_to_df(preds):
    if not preds:
        return pd.DataFrame(columns=["image_id","class_id","score","x1","y1","x2","y2"])
    return pd.DataFrame([{
        "image_id": p["image_id"],
        "class_id": p["class_id"],
        "score": p["score"],
        "x1": p["box"][0],
        "y1": p["box"][1],
        "x2": p["box"][2],
        "y2": p["box"][3],
    } for p in preds])

def df_to_preds(df):
    return [{
        "image_id": str(r.image_id),
        "class_id": int(r.class_id),
        "score": float(r.score),
        "box": np.array([r.x1, r.y1, r.x2, r.y2], dtype=float),
    } for r in df.itertuples(index=False)]


# ---------------------------------------------------------------------
# VOC mAP@0.40
# ---------------------------------------------------------------------
def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1]-mrec[idx]) * mpre[idx+1]))

def evaluate_class(predictions, gt_df, cid, min_score=0.0):
    gt_lookup = defaultdict(list)
    cls_gt = gt_df[gt_df.class_id.astype(int) == int(cid)]
    for r in cls_gt.itertuples(index=False):
        gt_lookup[str(r.image_id)].append(
            np.array([r.x_min, r.y_min, r.x_max, r.y_max], dtype=float)
        )
    n_gt = sum(map(len, gt_lookup.values()))
    if n_gt == 0:
        return np.nan

    pp = [
        p for p in predictions
        if int(p["class_id"]) == int(cid) and float(p["score"]) >= min_score
    ]
    pp.sort(key=lambda x: x["score"], reverse=True)
    used = {k: np.zeros(len(v), dtype=bool) for k, v in gt_lookup.items()}
    tp, fp = np.zeros(len(pp)), np.zeros(len(pp))

    for i, p in enumerate(pp):
        iid = str(p["image_id"])
        candidates = gt_lookup.get(iid, [])
        if not candidates:
            fp[i] = 1
            continue
        ious = np.asarray([box_iou(np.asarray(p["box"]), g) for g in candidates])
        j = int(np.argmax(ious))
        if ious[j] >= EVAL_IOU and not used[iid][j]:
            tp[i] = 1
            used[iid][j] = True
        else:
            fp[i] = 1

    if len(pp) == 0:
        return 0.0
    tpc = np.cumsum(tp)
    fpc = np.cumsum(fp)
    rec = tpc / max(n_gt, 1)
    prec = tpc / np.maximum(tpc + fpc, 1e-12)
    return voc_ap(rec, prec)

def evaluate_map(predictions, gt_df, thresholds=None):
    aps = {}
    for cid in range(14):
        thr = 0.0 if thresholds is None else float(thresholds.get(cid, 0.0))
        aps[cid] = evaluate_class(predictions, gt_df, cid, thr)
    vals = [x for x in aps.values() if not np.isnan(x)]
    return float(np.mean(vals)), aps

def tune_thresholds(predictions, gt_df):
    grid = [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
    best_thr, best_ap = {}, {}
    for cid in range(14):
        pairs = []
        for thr in grid:
            ap = evaluate_class(predictions, gt_df, cid, thr)
            pairs.append((ap, thr))
        pairs = [x for x in pairs if not np.isnan(x[0])]
        if not pairs:
            best_thr[cid], best_ap[cid] = 0.01, np.nan
        else:
            ap, thr = max(pairs, key=lambda x: x[0])
            best_thr[cid], best_ap[cid] = float(thr), float(ap)
    return best_thr, best_ap


# ---------------------------------------------------------------------
# Fold ensemble / class-wise selector
# ---------------------------------------------------------------------
def wbf_like(pred_lists, iou_thr=WBF_IOU):
    """
    Lightweight WBF-style fusion. Clusters per image/class, weighted coordinates,
    and confidence boosted modestly when multiple fold models agree.
    """
    grouped = defaultdict(list)
    n_models = len(pred_lists)
    for model_idx, preds in enumerate(pred_lists):
        for p in preds:
            q = dict(p)
            q["model_idx"] = model_idx
            grouped[(str(p["image_id"]), int(p["class_id"]))].append(q)

    out = []
    for (iid, cid), items in grouped.items():
        items = sorted(items, key=lambda x: x["score"], reverse=True)
        clusters = []
        for p in items:
            best = None
            best_iou = -1
            for j, cl in enumerate(clusters):
                b = np.average(
                    np.stack([x["box"] for x in cl]),
                    axis=0,
                    weights=np.asarray([max(x["score"], 1e-6) for x in cl]),
                )
                ii = box_iou(p["box"], b)
                if ii >= iou_thr and ii > best_iou:
                    best, best_iou = j, ii
            if best is None:
                clusters.append([p])
            else:
                clusters[best].append(p)

        for cl in clusters:
            scores = np.asarray([x["score"] for x in cl], dtype=float)
            boxes = np.stack([x["box"] for x in cl])
            fused_box = np.average(boxes, axis=0, weights=np.maximum(scores, 1e-6))
            agreeing = len(set(x["model_idx"] for x in cl))
            mean_score = float(scores.mean())
            # Small consensus boost, capped. Keeps ranking stable but rewards agreement.
            fused_score = min(1.0, mean_score * (1.0 + 0.10 * (agreeing - 1)))
            out.append({
                "image_id": iid,
                "class_id": cid,
                "score": fused_score,
                "box": fused_box,
            })
    return out

def classwise_select(rtdetr_preds, yolom_preds, family_by_class):
    out = []
    for p in rtdetr_preds:
        if family_by_class.get(int(p["class_id"])) == "rtdetr":
            out.append(p)
    for p in yolom_preds:
        if family_by_class.get(int(p["class_id"])) == "yolom":
            out.append(p)
    return out


# ---------------------------------------------------------------------
# Submission — IMPORTANT: original coordinate system
# ---------------------------------------------------------------------
def build_submission(preds, thresholds, sample_df, size_df, out_csv):
    size = size_df.copy()
    size["image_id"] = size["image_id"].astype(str)
    size_lookup = size.set_index("image_id")[["dim0","dim1"]].to_dict("index")

    grouped = defaultdict(list)
    for p in preds:
        cid = int(p["class_id"])
        if float(p["score"]) >= float(thresholds.get(cid, 0.01)):
            grouped[str(p["image_id"])].append(p)

    rows = []
    for iid in sample_df["image_id"].astype(str):
        pp = sorted(grouped.get(iid, []), key=lambda x: x["score"], reverse=True)
        if not pp:
            rows.append({"image_id": iid, "PredictionString": "14 1.0 0 0 1 1"})
            continue

        h = float(size_lookup[iid]["dim0"])
        w = float(size_lookup[iid]["dim1"])
        parts = []
        for p in pp:
            x1, y1, x2, y2 = map(float, p["box"])
            # MODEL 1024 SPACE -> ORIGINAL IMAGE SPACE.
            x1 = np.clip(x1 * w / IMG_SIZE, 0, w)
            x2 = np.clip(x2 * w / IMG_SIZE, 0, w)
            y1 = np.clip(y1 * h / IMG_SIZE, 0, h)
            y2 = np.clip(y2 * h / IMG_SIZE, 0, h)
            if x2 <= x1 or y2 <= y1:
                continue
            parts.extend([
                str(int(p["class_id"])),
                f"{float(p['score']):.6f}",
                f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
            ])
        pred_str = " ".join(parts) if parts else "14 1.0 0 0 1 1"
        rows.append({"image_id": iid, "PredictionString": pred_str})

    sub = pd.DataFrame(rows)
    sub.to_csv(out_csv, index=False)
    return sub


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    seed_everything()
    status("startup", "AMIA v5 stronger pipeline started")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"WORK_DIR: {WORK_DIR}")
    print(f"IMG_SIZE={IMG_SIZE}, folds={N_FOLDS}")

    train_df, test_df, size_df, sample_df = locate_files()
    train_df, det_df = preprocess_annotations(train_df, size_df)
    train_map, test_map = build_image_maps(train_df, sample_df)

    fused_path = WORK_DIR / "relaxed_fused_annotations.csv"
    if fused_path.exists():
        fused = pd.read_csv(fused_path)
        fused["image_id"] = fused["image_id"].astype(str)
    else:
        fused = relaxed_fusion(det_df)
        fused.to_csv(fused_path, index=False)

    print(f"Raw abnormal boxes: {len(det_df)}")
    print(f"Relaxed fused boxes: {len(fused)}")
    print(f"Singleton retained: {(fused.n_radiologists == 1).sum()}")

    all_ids = np.asarray(sorted(train_df["image_id"].astype(str).unique()))
    strat = make_stratification_labels(all_ids, fused)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_rows = []
    fold_assign = {}
    splits = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(all_ids, strat)):
        tr_ids = all_ids[tr_idx].tolist()
        va_ids = all_ids[va_idx].tolist()
        splits.append((tr_ids, va_ids))
        for iid in va_ids:
            fold_assign[iid] = fold
        fold_rows.append({"fold": fold, "train_images": len(tr_ids), "val_images": len(va_ids)})
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "fold_sizes.csv", index=False)
    pd.DataFrame({
        "image_id": list(fold_assign.keys()),
        "fold": list(fold_assign.values()),
    }).to_csv(RESULTS_DIR / "fold_assignments.csv", index=False)

    families = []
    if RUN_RTDETR:
        families.append("rtdetr")
    if RUN_YOLOM:
        families.append("yolom")
    if not families:
        raise RuntimeError("No model family enabled")

    # Train + OOF predict
    family_oof = {}
    family_fold_weights = {f: [] for f in families}

    for family in families:
        all_oof = []
        for fold, (tr_ids, va_ids) in enumerate(splits):
            status(family, f"fold {fold+1}/{N_FOLDS}")
            data_yaml = prepare_fold_dataset(fold, tr_ids, va_ids, train_map, fused)
            w = train_one(family, fold, data_yaml)
            family_fold_weights[family].append(w)

            oof_csv = OOF_DIR / f"{family}_fold{fold}.csv"
            if oof_csv.exists():
                fold_preds = df_to_preds(pd.read_csv(oof_csv))
            else:
                fold_preds = predict_model(family, w, va_ids, train_map, desc=f"OOF fold{fold}")
                preds_to_df(fold_preds).to_csv(oof_csv, index=False)
            all_oof.extend(fold_preds)

        family_oof[family] = all_oof
        preds_to_df(all_oof).to_csv(OOF_DIR / f"{family}_all_oof.csv", index=False)

    # OOF evaluation before threshold tuning
    oof_metrics = {}
    ap_table = pd.DataFrame({
        "class_id": range(14),
        "class_name": [CLASS_NAMES[i] for i in range(14)],
    })

    for family in families:
        m, aps = evaluate_map(family_oof[family], fused)
        oof_metrics[f"{family}_oof_map40_raw"] = m
        ap_table[f"{family}_ap40_raw"] = [aps[i] for i in range(14)]
        print(f"{family} OOF mAP@0.40 raw = {m:.5f}")

    # Per-class family selection from unbiased pooled OOF AP.
    if set(families) == {"rtdetr", "yolom"}:
        family_by_class = {}
        for cid in range(14):
            ra = float(ap_table.loc[ap_table.class_id == cid, "rtdetr_ap40_raw"].iloc[0])
            ya = float(ap_table.loc[ap_table.class_id == cid, "yolom_ap40_raw"].iloc[0])
            family_by_class[cid] = "rtdetr" if ra >= ya else "yolom"
    else:
        family_by_class = {cid: families[0] for cid in range(14)}

    selected_oof = classwise_select(
        family_oof.get("rtdetr", []),
        family_oof.get("yolom", []),
        family_by_class,
    )
    raw_selected_map, raw_selected_aps = evaluate_map(selected_oof, fused)
    thresholds, tuned_aps = tune_thresholds(selected_oof, fused)
    tuned_map, tuned_ap_eval = evaluate_map(selected_oof, fused, thresholds)

    ap_table["selected_family"] = [family_by_class[i] for i in range(14)]
    ap_table["selected_ap40_raw"] = [raw_selected_aps[i] for i in range(14)]
    ap_table["tuned_threshold"] = [thresholds[i] for i in range(14)]
    ap_table["selected_ap40_tuned"] = [tuned_ap_eval[i] for i in range(14)]
    ap_table.to_csv(RESULTS_DIR / "per_class_oof_ap40.csv", index=False)

    oof_metrics["classwise_oof_map40_raw"] = raw_selected_map
    oof_metrics["classwise_oof_map40_tuned"] = tuned_map
    print(f"Class-wise OOF mAP@0.40 raw   = {raw_selected_map:.5f}")
    print(f"Class-wise OOF mAP@0.40 tuned = {tuned_map:.5f}")

    # Test inference: three models per family -> fold WBF, then class-wise selector.
    if RUN_TEST:
        status("test", "3-fold model inference and WBF")
        family_test_fused = {}
        test_ids = sample_df["image_id"].astype(str).tolist()

        for family in families:
            fold_test_preds = []
            for fold, w in enumerate(family_fold_weights[family]):
                cache_csv = RESULTS_DIR / f"{family}_test_fold{fold}.csv"
                if cache_csv.exists():
                    pp = df_to_preds(pd.read_csv(cache_csv))
                else:
                    pp = predict_model(family, w, test_ids, test_map, desc=f"test fold{fold}")
                    preds_to_df(pp).to_csv(cache_csv, index=False)
                fold_test_preds.append(pp)
            fused_test = wbf_like(fold_test_preds, iou_thr=WBF_IOU)
            family_test_fused[family] = fused_test
            preds_to_df(fused_test).to_csv(RESULTS_DIR / f"{family}_test_wbf.csv", index=False)
            del fold_test_preds
            cleanup(f"after {family} test WBF")

        final_preds = classwise_select(
            family_test_fused.get("rtdetr", []),
            family_test_fused.get("yolom", []),
            family_by_class,
        )
        preds_to_df(final_preds).to_csv(RESULTS_DIR / "final_test_predictions_1024.csv", index=False)

        submission_path = RESULTS_DIR / "submission_v5.csv"
        sub = build_submission(final_preds, thresholds, sample_df, size_df, submission_path)

        # Submission sanity checks.
        assert len(sub) == len(sample_df)
        assert sub["image_id"].nunique() == len(sample_df)
        assert sub["PredictionString"].notna().all()

        abnormal = ~sub["PredictionString"].str.startswith("14 ")
        print(f"Submission: {submission_path}")
        print(f"Rows: {len(sub)} | abnormal rows: {int(abnormal.sum())}")

    metrics = {
        **oof_metrics,
        "img_size": IMG_SIZE,
        "folds": N_FOLDS,
        "label_fusion_iou": LABEL_FUSION_IOU,
        "wbf_iou": WBF_IOU,
        "prediction_conf_floor": PRED_CONF,
        "family_by_class": {str(k): v for k, v in family_by_class.items()},
        "thresholds": {str(k): float(v) for k, v in thresholds.items()},
        "raw_abnormal_boxes": int(len(det_df)),
        "relaxed_fused_boxes": int(len(fused)),
        "singleton_fused_boxes": int((fused.n_radiologists == 1).sum()),
    }
    (RESULTS_DIR / "metrics_v5.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Compact human-readable report.
    lines = [
        "# AMIA v5 Results",
        "",
        f"- RT-DETR enabled: {RUN_RTDETR}",
        f"- YOLOv8m enabled: {RUN_YOLOM}",
        f"- Folds: {N_FOLDS}",
        f"- Image size: {IMG_SIZE}",
        f"- Relaxed label fusion IoU: {LABEL_FUSION_IOU}",
        f"- WBF IoU: {WBF_IOU}",
        "",
    ]
    for k, v in oof_metrics.items():
        lines.append(f"- {k}: **{v:.5f}**")
    lines += [
        "",
        "## Per-class selector",
        "",
        ap_table.to_markdown(index=False),
        "",
        "## Submission",
        "",
        f"`{RESULTS_DIR / 'submission_v5.csv'}`",
        "",
        "Important: submission boxes are converted from 1024 model space back to each image's original dim0/dim1 coordinates.",
    ]
    (RESULTS_DIR / "report_v5.md").write_text("\n".join(lines), encoding="utf-8")

    mark_done("pipeline")
    status("complete", "AMIA v5 pipeline complete")
    print("=" * 72)
    print("PIPELINE COMPLETE")
    print(f"Metrics:    {RESULTS_DIR / 'metrics_v5.json'}")
    print(f"Per-class:  {RESULTS_DIR / 'per_class_oof_ap40.csv'}")
    print(f"Report:     {RESULTS_DIR / 'report_v5.md'}")
    if RUN_TEST:
        print(f"Submission: {RESULTS_DIR / 'submission_v5.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
