@echo off
REM train_classifier.bat - Windows training script for dog aggression classifier

echo Setting up environment...
cd /d "%~dp0"

REM Set Python path (adjust if needed)
set PYTHON_PATH=python

REM Create dataset directory if it doesn't exist
if not exist "..\datasets\dog_aggression" (
    echo Creating dataset directory...
    mkdir "..\datasets\dog_aggression"
)

echo Starting YOLOv8 classification training...
echo.

REM Train classification model
%PYTHON_PATH% -c "
from ultralytics import YOLO

# Load a pretrained model (YOLOv8n-cls is recommended for classification)
model = YOLO('yolov8n-cls.pt')

# Train the model
results = model.train(
    data='../datasets/dog_aggression',
    epochs=50,
    imgsz=224,
    batch=16,
    device='cpu',  # or '0' for GPU if available
    workers=0,  # Windows-safe (0 for CPU)
    project='../models',
    name='dog_aggression_cls',
    exist_ok=True,
    verbose=True
)

print('Training complete!')
print(f'Best model saved to: {results.save_dir}')
"

pause