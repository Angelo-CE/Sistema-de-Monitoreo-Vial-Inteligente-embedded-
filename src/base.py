import ctypes
import multiprocessing as mp
import numpy as np
import cv2
import time
import sys
import os

# ====================================================================
# CONFIGURACIÓN GENERAL
# ====================================================================
MODE = "JETSON"  # "PC" o "JETSON"
#SOURCE = "videos_test/highway23.mp4"  # "camera" o ruta del archivo
SOURCE = "highway23.mp4"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CHANNELS = 3
BUFFER_SIZE = 8
FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * CHANNELS


def get_video_reader(source, mode="PC"):
    """Abstracción de entrada de video según plataforma."""
    if mode == "PC":
        if source == "camera":
            return cv2.VideoCapture(0)
        return cv2.VideoCapture(source)

    elif mode == "JETSON":
        if source == "camera":
            # En CÁMARA sí se usa drop=true para evitar latencia
            gst_pipeline = (
                "nvarguscamerasrc sensor-id=0 ! "
                f"video/x-raw(memory:NVMM), width=(int){FRAME_WIDTH}, height=(int){FRAME_HEIGHT}, framerate=(fraction)30/1 ! "
                "nvvidconv flip-method=0 ! "
                f"video/x-raw, width=(int){FRAME_WIDTH}, height=(int){FRAME_HEIGHT}, format=(string)BGRx ! "
                "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true"
            )
            print("[READER] Abriendo cámara CSI por GStreamer...")
            return cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        else:
            # En ARCHIVO se usa drop=false sync=false para NO perder cuadros
            gst_pipeline = (
                f"filesrc location={source} ! qtdemux ! h264parse ! "
                "nvv4l2decoder ! nvvidconv ! "
                f"video/x-raw, width=(int){FRAME_WIDTH}, height=(int){FRAME_HEIGHT}, format=(string)BGRx ! "
                "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=false sync=false"
            )
            print("[READER] Abriendo archivo mediante NVDEC por GStreamer...")
            return cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    raise ValueError(f"Modo no reconocido: {mode}")


# ====================================================================
# WORKERS (PROCESOS INDEPENDIENTES)
# ====================================================================

def reader_process(source, mode, raw_array, free_queue, ai_queue, config_queue, stop_event):
    """Proceso 1: Extrae frames y escribe en la memoria compartida (Zero-Copy)."""
    print(f"[READER] Iniciado en modo {mode}.")
    
    shared_array = np.frombuffer(raw_array, dtype=np.uint8).reshape(
        (BUFFER_SIZE, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)
    )
    
    cap = get_video_reader(source, mode=mode)
    if not cap.isOpened():
        print(f"[READER ERROR] No se pudo abrir la fuente: {source}")
        config_queue.put(30.0)
        ai_queue.put(None)
        return

    # Obtener FPS reales de la fuente
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # Fallback si GStreamer no reporta los FPS
    
    print(f"[READER] Fuente detectada a {fps:.2f} FPS.")
    config_queue.put(fps)

    frame_count = 0

    try:
        while not stop_event.is_set():
            try:
                index = free_queue.get(timeout=1.0)
            except mp.queues.Empty:
                continue

            ret, frame = cap.read()
            if not ret:
                free_queue.put(index)
                print(f"[READER] Fin del stream. Total cuadros leídos: {frame_count}")
                break

            if frame.shape[0] != FRAME_HEIGHT or frame.shape[1] != FRAME_WIDTH:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            shared_array[index] = frame
            frame_count += 1

            ai_queue.put((frame_count, index))

    except Exception as e:
        print(f"[READER ERROR]: {e}")
    finally:
        ai_queue.put(None)
        cap.release()
        print("[READER] Proceso finalizado.")


def ai_process(raw_array, ai_queue, render_queue, stop_event):
    """Proceso 2: Inferencia de IA leyendo directo de la memoria compartida."""
    print("[AI] Iniciado.")
    
    shared_array = np.frombuffer(raw_array, dtype=np.uint8).reshape(
        (BUFFER_SIZE, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)
    )

    try:
        while not stop_event.is_set():
            try:
                data = ai_queue.get(timeout=1.0)
            except mp.queues.Empty:
                continue

            if data is None:
                break

            frame_count, index = data
            
            # Simulación de inferencia
            time.sleep(0.01)
            
            metadata = {
                "frame_id": frame_count,
                "detections": [
                    {"label": "Vehiculo", "bbox": [100, 100, 300, 300], "confidence": 0.92}
                ]
            }

            render_queue.put((frame_count, index, metadata))

    except Exception as e:
        print(f"[AI ERROR]: {e}")
    finally:
        render_queue.put(None)
        print("[AI] Proceso finalizado.")


def render_process(raw_array, render_queue, free_queue, config_queue, stop_event):
    """Proceso 3: Dibuja metadata sobre la imagen compartida y genera el archivo."""
    print("[RENDER] Esperando configuración de FPS...")
    
    fps_out = config_queue.get()
    print(f"[RENDER] Configurando salida de video a {fps_out:.2f} FPS.")

    shared_array = np.frombuffer(raw_array, dtype=np.uint8).reshape(
        (BUFFER_SIZE, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, fps_out, (FRAME_WIDTH, FRAME_HEIGHT))

    processed_count = 0

    try:
        while not stop_event.is_set():
            try:
                data = render_queue.get(timeout=1.0)
            except mp.queues.Empty:
                continue

            if data is None:
                break

            frame_count, index, metadata = data
            frame = shared_array[index]

            for det in metadata["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{det['label']} {det['confidence']:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"Frame: {frame_count}", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)
            processed_count += 1

            free_queue.put(index)

    except Exception as e:
        print(f"[RENDER ERROR]: {e}")
    finally:
        out.release()
        print(f"[RENDER] Proceso finalizado. Cuadros guardados: {processed_count}")


# ====================================================================
# MAIN
# ====================================================================

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    print(f"=== INICIANDO MOTOR MULTIPROCESAMIENTO EN MODO [{MODE}] ===")

    total_bytes = BUFFER_SIZE * FRAME_BYTES
    raw_array = mp.RawArray(ctypes.c_uint8, total_bytes)

    free_queue = mp.Queue()
    ai_queue = mp.Queue()
    render_queue = mp.Queue()
    config_queue = mp.Queue()
    stop_event = mp.Event()

    for i in range(BUFFER_SIZE):
        free_queue.put(i)

    p_reader = mp.Process(target=reader_process, args=(SOURCE, MODE, raw_array, free_queue, ai_queue, config_queue, stop_event))
    p_ai = mp.Process(target=ai_process, args=(raw_array, ai_queue, render_queue, stop_event))
    p_render = mp.Process(target=render_process, args=(raw_array, render_queue, free_queue, config_queue, stop_event))

    try:
        p_reader.start()
        p_ai.start()
        p_render.start()

        p_reader.join()
        p_ai.join()
        p_render.join()

    except KeyboardInterrupt:
        print("\n[MAIN] Apagado de emergencia activado por el usuario...")
        stop_event.set()
        p_reader.join()
        p_ai.join()
        p_render.join()
    finally:
        print("[MAIN] Memoria compartida liberada. Sistema apagado limpiamente.")