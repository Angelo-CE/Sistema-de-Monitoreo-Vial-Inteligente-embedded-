from ultralytics import YOLO

# 1. Cargar pesos preentrenados universales (Transfer Learning)
model = YOLO("models/yolov8n.pt")

# 2. Entrenar el detector de placas costarricenses
results = model.train(
    data="dataset_yolo_cr/data.yaml",  # Ruta del script anterior
    epochs=60,                          # 60 épocas son suficientes
    imgsz=480,                          # Optimizado para TensorRT en Jetson Nano
    batch=16,                           # Ajustar a 8 si te quedas sin VRAM en GPU local
    device=0,                           # GPU ID (0)
    workers=4,
    patience=10,                        # Early stopping: frena si no mejora en 10 épocas
    close_mosaic=10,                    # Desactiva aumento Mosaic las últimas 10 épocas para afinar bordes
    project="runs_cr_plates",
    name="yolov8n_cr_v2",
    exist_ok=True,                      # Sobrescribe la carpeta si haces pruebas rápidas
    save=True
)