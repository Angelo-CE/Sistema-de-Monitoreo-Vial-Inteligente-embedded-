import time
import cv2
import torch
from ultralytics import YOLO


def main(path):

    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Cargar modelo 
    model_path = "models/1.0/best.pt"

    try:
        model_plates = YOLO(model_path)
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    print(f"Modelo cargado: {model_path}")


    # Abrir el video
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.2f} FPS")

    
    # Output Config
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        "output.mp4",
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        print("Error creando output.mp4")
        cap.release()
        return

    frame_count = 0
    total_inference_time = 0.0
    total_detections = 0

    # ---------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        start = time.perf_counter()

        results = model_plates.predict(
            source=frame,
            imgsz=960, # Establecer tamaño de imagen a 640x640
            conf=0.15, # Establecer umbral de confianza a 0.5
            device=device,
            verbose=False
        )

        inference_time = time.perf_counter() - start
        total_inference_time += inference_time

        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:

            boxes = result.boxes.xyxy.cpu().numpy()

            total_detections += len(boxes)

            for box in boxes:

                px1, py1, px2, py2 = map(int, box)

                # Center of plate
                p_center_x = (px1 + px2) / 2
                p_center_y = (py1 + py2) / 2

                cv2.rectangle(
                    frame,
                    (px1, py1),
                    (px2, py2),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (int(p_center_x), int(p_center_y)),
                    4,
                    (0, 0, 255),
                    -1
                )

                # Optional label
                cv2.putText(
                    frame,
                    "PLATE",
                    (px1, max(py1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        out.write(frame)

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    cap.release()
    out.release()

    cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------
    if frame_count > 0:

        avg_inference = total_inference_time / frame_count
        inference_fps = 1.0 / avg_inference

        print("\n========== RESULTS ==========")
        print(f"Frames procesados: {frame_count}")
        print(f"Detecciones totales: {total_detections}")
        print(f"Tiempo promedio/frame: {avg_inference * 1000:.2f} ms")
        print(f"FPS de inferencia: {inference_fps:.2f}")
        print("=============================")


if __name__ == "__main__":

    video_path = "videos_test/highway23.mp4"

    main(video_path)