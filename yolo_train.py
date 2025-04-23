from ultralytics import YOLO
import subprocess
import os

# Load model
model = YOLO('yolov8s.pt')  

# Train
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=500,
    batch=16,
    project='yolo_digit_detection',
    device='cpu',
    name='exp3'
)

