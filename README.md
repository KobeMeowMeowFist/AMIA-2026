# AMIA Public Challenge 2026 — Curta/SLURM GPU Version

This version is adapted to the known-working SLURM pattern supplied for the school cluster. SLURM owns the training process; there is no Kaggle/Jupyter background `Popen` layer.

## Cluster request used by this project

The main job follows this resource profile:

```bash
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --qos=standard
```

The job checks the allocated card at runtime and accepts **H100 or A5000 only**. It exits immediately if a different GPU is assigned.

The Python pipeline deliberately uses **one GPU (`cuda:0`)** for YOLO, Faster R-CNN, and DenseNet. It will not auto-enable Ultralytics multi-GPU DDP even if the node exposes additional devices.

## Files

```text
amia_slurm_pipeline.py     full training/evaluation pipeline
run_curta_gpu.sbatch       recommended full job; accepts H100 or A5000
run_h100.sbatch            full job that requires an H100 at runtime
run_a5000.sbatch           full job using the known A5000 GRES spelling
run_smoke_test.sbatch      one-epoch end-to-end test
download_kaggle_data.sh    Kaggle competition downloader
prepare_pretrained_weights.py
monitor.sh
requirements.txt
```

## 1. Put the project on the server

```bash
unzip amia_slurm_project_curta.zip
cd amia_slurm_project_curta
```

Keep the project code wherever you normally store repositories. Large competition data and checkpoints default to `/scratch/$USER/...`.

## 2. Python environment

Prefer the school's CUDA/PyTorch module or environment. Example if PyTorch is already available system-wide:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
pip install -r requirements.txt
```

Do not blindly replace the cluster's working PyTorch/CUDA stack.

## 3. Kaggle authentication and data download

First accept the competition rules on Kaggle. Configure Kaggle CLI credentials on the login node, then run:

```bash
source .venv/bin/activate
./download_kaggle_data.sh
```

Default destination:

```text
/scratch/$USER/amia-public-challenge-2026
```

You can override it:

```bash
./download_kaggle_data.sh /scratch/$USER/my_amia_data
export AMIA_DATA_DIR=/scratch/$USER/my_amia_data
```

## 4. Download pretrained weights before compute jobs

If compute nodes have no Internet:

```bash
python prepare_pretrained_weights.py --weights-dir ./weights
```

This prepares YOLOv8s, Faster R-CNN/ResNet50-FPN, and DenseNet121 pretrained weights.

## 5. Smoke test

```bash
sbatch run_smoke_test.sbatch
```

Then:

```bash
squeue -u $USER
./monitor.sh JOB_ID
```

The smoke test uses one epoch per model and exists only to validate the complete path from data loading through submission/report generation.

## 6. Recommended full submission

Use the generic Curta job:

```bash
sbatch run_curta_gpu.sbatch
```

It requests one GPU from `scavenger`, then checks that the card is H100 or A5000. Batch sizes are selected from the actual GPU:

| GPU | YOLO batch | Faster R-CNN batch | DenseNet batch |
|---|---:|---:|---:|
| H100 | 8 | 4 | 32 |
| A5000 | 4 | 2 | 16 |

The full experiment uses YOLO 40 epochs, Faster R-CNN 10 epochs, and DenseNet121 8 epochs.

If the scheduler gives a GPU other than H100/A5000, the script exits before model training. Resubmit using your cluster's node-selection helper or a suitable `sbatch --nodelist=...` option.

## 7. A5000-specific request

The supplied working cluster script documents this GRES spelling for A5000, so you can explicitly request it with:

```bash
sbatch run_a5000.sbatch
```

or equivalently:

```bash
sbatch --gres=gpu:a5000:1 run_curta_gpu.sbatch
```

## 8. H100-specific request

The reference cluster script intentionally avoids hardcoding an H100 node. `run_h100.sbatch` therefore uses generic `--gres=gpu:1` and verifies at runtime that the allocated GPU is H100.

If your existing `find_best_gpu.sh` selects an H100 node, use it with `run_h100.sbatch` or submit with your known-valid node selector. Do not permanently hardcode a node into the project unless the cluster administrator recommends it.

## 9. Work/checkpoint locations

Generic job default:

```text
/scratch/$USER/amia_2026_work/amia_cv
```

H100 profile:

```text
/scratch/$USER/amia_2026_work/h100
```

A5000 profile:

```text
/scratch/$USER/amia_2026_work/a5000
```

All checkpoints and stage markers stay on shared `/scratch`, so SSH disconnects do not stop training. Re-submitting with the same `AMIA_WORK_DIR` allows the pipeline to reuse completed stages/checkpoints.

## 10. Final outputs

Under the selected work directory:

```text
amia_results/
  report.md
  submission.csv
  metrics.json
  experiment_metadata.json
  figures/
  tables/
amia_report_bundle.zip
```

The pipeline performs multi-radiologist annotation fusion, YOLOv8s, Faster R-CNN + ResNet50-FPN, DenseNet121 Normal/Abnormal classification, VOC-style lesion mAP@IoU=0.40, detector fusion, classifier calibration, validation threshold tuning, test inference, plots, and report generation.
