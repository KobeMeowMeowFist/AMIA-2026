#!/usr/bin/env python3
"""
Generate two additional low-confidence single-fold RT-DETR submissions
from the already cached v5 fold1 predictions. No GPU and no training.
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("/scratch/fanm01/CV/amia_slurm_project_curta/data/amia-public-challenge-2026")
WORK = Path("/scratch/xiaolil02/amia_2026_work/v5_rtdetr_yolom")
SRC = WORK / "results" / "rtdetr_test_fold1.csv"
OUT = WORK / "results" / "v52_candidates"
OUT.mkdir(parents=True, exist_ok=True)

sample = pd.read_csv(DATA / "sample_submission.csv")
sizes = pd.read_csv(DATA / "img_size.csv")

sample["image_id"] = sample["image_id"].astype(str)
sizes["image_id"] = sizes["image_id"].astype(str)
pred = pd.read_csv(SRC)
pred["image_id"] = pred["image_id"].astype(str)

lookup = sizes.set_index("image_id")[["dim0", "dim1"]].to_dict("index")

# v5 fold predictions are in the 1024 source-PNG coordinate system.
for thr in [0.05, 0.075]:
    p = pred[pred.score >= thr].copy().sort_values(
        ["image_id", "score"], ascending=[True, False]
    )
    grouped = {k: g for k, g in p.groupby("image_id", sort=False)}
    rows = []

    for iid in sample.image_id:
        g = grouped.get(iid)
        if g is None or g.empty:
            rows.append((iid, "14 1.0 0 0 1 1"))
            continue

        h = float(lookup[iid]["dim0"])
        w = float(lookup[iid]["dim1"])
        tok = []

        for r in g.itertuples(index=False):
            x1 = np.clip(float(r.x1) * w / 1024.0, 0, w)
            x2 = np.clip(float(r.x2) * w / 1024.0, 0, w)
            y1 = np.clip(float(r.y1) * h / 1024.0, 0, h)
            y2 = np.clip(float(r.y2) * h / 1024.0, 0, h)
            if x2 <= x1 or y2 <= y1:
                continue
            tok += [
                str(int(r.class_id)), f"{float(r.score):.6f}",
                f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}"
            ]

        rows.append((iid, " ".join(tok) if tok else "14 1.0 0 0 1 1"))

    out = pd.DataFrame(rows, columns=["image_id", "PredictionString"])
    dst = OUT / f"v52_rtdetr_fold1_conf{str(thr).replace('.', 'p')}.csv"
    out.to_csv(dst, index=False)

    def nbox(s):
        t = str(s).split()
        return 0 if (not t or t[0] == "14") else len(t)//6

    n = out.PredictionString.map(nbox)
    print(
        f"{dst.name}: no_finding={(n==0).sum()}, abnormal={(n>0).sum()}, "
        f"total_boxes={int(n.sum())}, mean={n.mean():.2f}, max={int(n.max())}"
    )
