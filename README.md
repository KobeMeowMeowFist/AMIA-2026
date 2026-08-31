# AMIA 2026 Chest X-ray Abnormality Detection

Course project for multi-class abnormality detection on chest X-rays using **RT-DETR-L**.

The final pipeline uses the original radiologist bounding-box annotations without custom box fusion. The 8,573 labeled training images are split at the image level into 7,287 training and 1,286 validation images. RT-DETR-L is trained with 640×640 inputs and evaluated with standard detection metrics. Test predictions are filtered by confidence and mapped back to the original image coordinates.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── amia_v6_rtdetr_raw_h100.py
├── scripts/
│   └── run_v6_rtdetr_h100.sbatch
├── configs/
│   ├── args.yaml
│   ├── experiment_v6.json
│   └── dataset_stats_v6.json
└── results/
    └── results.csv
```

## Final configuration

- Model: RT-DETR-L
- Input size: 640×640
- Training labels: raw radiologist boxes, no custom fusion
- Train / validation split: 7,287 / 1,286 images
- Maximum epochs: 220
- Early-stopping patience: 30
- Batch: Ultralytics AutoBatch, target 75% GPU memory
- Optimizer: auto
- Learning-rate schedule: cosine
- AMP: enabled
- Workers: 8
- Seed: 42
- Final confidence threshold: 0.10

## Environment

The final run used:

```text
Python 3.14.7
PyTorch 2.13.0+cu130
Ultralytics 8.4.129
NVIDIA H100 80GB
```

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

On the university cluster, use the existing CUDA/PyTorch environment when possible rather than replacing the system CUDA stack.

## Data

The dataset is based on the AMIA Public Challenge 2026 / VinDr-CXR data.

Expected data:

```text
8,573 labeled training images
6,427 test images
14 abnormality classes
```

The final preprocessing keeps all abnormal annotation rows for classes 0–13. Normal images are retained with empty detection labels.

Set the dataset location before running if required by the script, for example:

```bash
export AMIA_DATA_DIR=/scratch/$USER/amia-public-challenge-2026
```

Do not upload the image dataset to this repository.

## Training

Submit the final training job with:

```bash
sbatch scripts/run_v6_rtdetr_h100.sbatch
```

The job trains one RT-DETR-L model on a single GPU and uses early stopping. Checkpoints and intermediate outputs are stored on shared scratch storage.

## Inference and post-processing

The final model produces:

```text
class
confidence score
bounding box
```

Test inference first keeps low-confidence candidates so that post-processing can be evaluated without rerunning the network. The final predictions use a confidence threshold of 0.10.

Predicted boxes are converted from the source image coordinates to the original radiograph coordinates before evaluation.

## Evaluation

Validation analysis includes:

- Precision
- Recall
- F1 score
- mAP@0.50
- mAP@0.50:0.95
- normalized confusion matrix
- qualitative comparison of ground-truth boxes and predictions

The challenge test-set metric is mAP at IoU 0.40.

## Code references

RT-DETR-L is used through the Ultralytics implementation. A public RT-DETR notebook for the same AMIA task was consulted as an implementation reference during the later experiments. The final v6 training, splitting, post-processing, checkpoint handling, and coordinate-conversion pipeline was implemented separately for this project.

See the report for the full experimental comparison, including the earlier YOLO, Faster R-CNN, DenseNet, and ensemble experiments.
