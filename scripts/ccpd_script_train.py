from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="ccpd_yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    project="runs_ccpd",
    name="yolov8n_ccpd_10k"
)