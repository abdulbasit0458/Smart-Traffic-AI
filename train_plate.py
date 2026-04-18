from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/licence/data.yaml",
    epochs=80,
    imgsz=960,
    batch=8,
    conf=0.25,
    name="license_plate_model"
)