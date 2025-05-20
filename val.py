from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train/weights/best.pt")

# Validate the model
metrics = model.val(imgsz=640)  


