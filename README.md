# Dog Aggression Detection System

A two-stage ML pipeline that detects dogs in video/images and classifies their behavior as **aggressive** or **non-aggressive** in real-time.

Built with YOLOv8 (detection) + ResNet18 (classification) and includes an **AI Training Agent** for automated hyperparameter tuning.

---

## Architecture

```
Input (image/video frame)
        │
        ▼
┌─────────────────────┐
│  Stage 1: Detector   │   YOLOv8n fine-tuned
│  → Locate dogs       │   Output: bounding boxes
└────────┬────────────┘
         │  crop each dog
         ▼
┌─────────────────────┐
│  Stage 2: Classifier │   ResNet18 / MobileNetV3
│  → Aggressive or     │   Output: class + confidence
│    Non-aggressive    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Decision Engine     │   confidence > 0.7 → ALERT
└─────────────────────┘
```

---

## Project Structure

```
dog_detection/
├── config/
│   ├── config.yaml          # Master config (all hyperparameters)
│   └── data.yaml            # Dataset paths (Roboflow YOLO format)
├── src/
│   ├── data/prepare.py      # Data loading, validation, splitting, augmentation
│   ├── models/
│   │   ├── detector.py      # YOLOv8 dog detector wrapper
│   │   └── classifier.py    # Aggression classifier (ResNet/MobileNet)
│   ├── training/
│   │   ├── train_detector.py    # Stage 1: YOLO fine-tuning
│   │   ├── train_classifier.py  # Stage 2: classifier with live graphs
│   │   └── callbacks.py         # EarlyStopping, MetricsLogger
│   ├── evaluation/evaluate.py   # mAP, F1, recall, AUC-ROC, confusion matrix
│   ├── inference/predict.py     # Image/video/webcam inference pipeline
│   └── agent/training_agent.py  # AI auto-tuning agent
├── app.py                   # Desktop GUI application (Tkinter)
├── train.py                 # Single training entry point
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/tedo001/dog_detection.git
cd dog_detection
pip install -r requirements.txt
```

### 2. Configure Dataset

Edit `config/data.yaml` and set the path to your YOLO-format dataset:

```yaml
path: D:/dog_cnn
train: train/images
val: valid/images
test: test/images
```

### 3. Prepare Data

```bash
python train.py --stage prepare
```

Validates dataset, creates train/val/test splits, and crops dog regions for classifier training.

### 4. Train Models

```bash
# Train dog detector (YOLOv8)
python train.py --stage detector

# Train aggression classifier (with live training graphs)
python train.py --stage classifier

# Or run the full pipeline
python train.py
```

### 5. AI Agent (Auto-Tune)

```bash
python train.py --stage agent --max-experiments 10
```

Automatically searches for the best hyperparameters using explore/exploit strategy.

### 6. Launch GUI Application

```bash
python app.py
```

Desktop app with:
- Browse/enter local video path
- Real-time annotated video with bounding boxes
- Live stats (frames, dogs detected, aggressive alerts, FPS)
- Alert log with timestamps
- Export annotated video and alert JSON

### 7. Command-Line Inference

```bash
# Single image
python train.py --stage predict --source path/to/image.jpg --output result.jpg

# Video
python train.py --stage predict --source path/to/video.mp4 --output output.mp4

# Webcam
python train.py --stage predict --source webcam
```

---

## Training Features

- **Live graphs**: Loss, accuracy, F1, precision, recall, and learning rate curves update in real-time during training
- **Progress tracking**: Epoch-by-epoch progress bar with percentage and ETA
- **Early stopping**: Automatically stops when validation F1 plateaus
- **Class weighting**: Handles imbalanced datasets (aggressive dogs are minority class)
- **Transfer learning**: Pretrained ImageNet backbones with optional backbone freezing
- **Experiment tracking**: Each run saves config, metrics, and checkpoints to `experiments/`

---

## Evaluation Metrics

| Metric | Stage | Purpose |
|--------|-------|---------|
| mAP@50 | Detector | Can we find dogs reliably? |
| mAP@50-95 | Detector | How precise are bounding boxes? |
| F1-Score | Classifier | Balance of precision and recall |
| Recall | Classifier | Do we catch all aggressive dogs? |
| AUC-ROC | Classifier | Performance across thresholds |
| Miss Rate | End-to-end | Missed aggressive dogs (safety metric) |

**Priority**: Recall > Precision (missing an aggressive dog is worse than a false alarm).

---

## Configuration

All hyperparameters are in `config/config.yaml`:

```yaml
detector:
  model: yolov8n.pt
  epochs: 50
  batch_size: 16
  lr: 0.01

classifier:
  model: resnet18
  epochs: 30
  batch_size: 32
  lr: 0.001
  class_weights: [1.0, 3.0]  # upweight aggressive class

inference:
  detection_conf: 0.5
  aggression_threshold: 0.7
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- Ultralytics (YOLOv8)
- Albumentations
- OpenCV
- scikit-learn

GPU: NVIDIA GPU with CUDA support recommended. Tested on RTX 3050.

---

## Tech Stack

- **Detection**: YOLO(yolo26x,yolo26n....ect.....) (Ultralytics)
- **Classification**: ResNet18 / MobileNetV3 (torchvision)
- **Augmentation**: Albumentations
- **GUI**: Tkinter + OpenCV + Pillow
- **Training Agent**: Custom explore/exploit hyperparameter search (inspired by n-autoresearch)
