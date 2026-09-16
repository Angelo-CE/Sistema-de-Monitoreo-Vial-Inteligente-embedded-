import os
import glob
import time
import cv2
import torch
from ultralytics import YOLO


def test_real_images(input_dir, output_dir, model_path, conf_thresh=0.25, imgsz=480):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        model = YOLO(model_path)
        print(f"Modelo cargado: {model_path}")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return

    # Buscar imágenes en el directorio de entrada
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not img_paths:
        print(f"[ERROR] No se encontraron imágenes en: {input_dir}")
        return

    print(f"[INFO] Procesando {len(img_paths)} imágenes reales...")

    total_inference_time = 0.0
    images_with_detections = 0
    total_detections = 0

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        start = time.perf_counter()
        results = model.predict(
            source=img,
            imgsz=imgsz,       # 480x480 para mantener paridad con Jetson Nano
            conf=conf_thresh,
            device=device,
            verbose=False
        )
        inference_time = time.perf_counter() - start
        total_inference_time += inference_time

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

        if len(boxes) > 0:
            images_with_detections += 1
            total_detections += len(boxes)

        for box, conf in zip(boxes, confs):
            px1, py1, px2, py2 = map(int, box)

            # Centro exacto de la placa
            p_center_x = (px1 + px2) / 2
            p_center_y = (py1 + py2) / 2

            # Rectángulo en verde
            cv2.rectangle(
                img,
                (px1, py1),
                (px2, py2),
                (0, 255, 0),
                2
            )

            # Punto central en rojo
            cv2.circle(
                img,
                (int(p_center_x), int(p_center_y)),
                4,
                (0, 0, 255),
                -1
            )

            # Etiqueta con porcentaje de confianza
            label = f"PLATE {conf * 100:.1f}%"
            cv2.putText(
                img,
                label,
                (px1, max(py1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Guardar resultado con el mismo nombre en la carpeta de salida
        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_dir, filename), img)

    # ---------------------------------------------------------
    # Estadísticas finales
    # ---------------------------------------------------------
    if img_paths:
        avg_time = (total_inference_time / len(img_paths)) * 1000
        print("\n========== RESULTADOS EN FOTOS REALES ==========")
        print(f"Imágenes evaluadas: {len(img_paths)}")
        print(f"Imágenes con detección: {images_with_detections} ({images_with_detections / len(img_paths) * 100:.1f}%)")
        print(f"Detecciones totales: {total_detections}")
        print(f"Tiempo promedio/inferencia: {avg_time:.2f} ms")
        print(f"Imágenes guardadas en: {output_dir}")
        print("=================================================")


if __name__ == "__main__":
    # Ajusta las rutas según tu estructura
    INPUT_DIR = "./ccpd_cr/ccpd_irl"                           # Fotos reales sin editar
    OUTPUT_DIR = "./output_test_irl"                           # Salida visual
    MODEL_PATH = "models/3.0/best.pt" # Tu modelo recién entrenado

    test_real_images(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        model_path=MODEL_PATH, 
        conf_thresh=0.25,
        imgsz=480
    )