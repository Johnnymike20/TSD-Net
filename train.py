from ultralytics import YOLO

# Load a model
model = YOLO(model="ultralytics/cfg/models/11/TSD-Net.yaml")

# Load pretrained-weight
model.load("yolo11s.pt")

# Train the model
# TT100k
model.train(data="ultralytics/cfg/datasets/TT100K.yaml", batch=32, epochs=200, imgsz=640)

# CCTSDB2021
# model.train(data="ultralytics/cfg/datasets/CCTSDB.yaml", batch=32, epochs=200, imgsz=640)