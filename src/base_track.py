import ctypes
import multiprocessing as mp
import numpy as np
import cv2
import time
import sys
import os
from ultralytics import YOLO
import torch
import LPRnetOCR as ocr

# ====================================================================
# CONFIGURACIÓN GENERAL
# ====================================================================
MODE = "PC"
SOURCE = "../system-monitoring-files/videos_test/highway932.mp4" #447 y 483
PLATES_MODEL = "./models/3.0/best.pt"
TRAFFIC_MODEL = "./models/yolov8n.pt"

# ====================================================================
# CONFIGURACIÓN DE LOS CUADROS
# ====================================================================
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CHANNELS = 3
BUFFER_SIZE = 8
FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * CHANNELS

CONF_THRESH = 0.35
IMGSZ_YOLO = 640 # 

# --- UMBRALES DE LA ZONA DULCE (SWEET SPOT) ---
MIN_PLATE_WIDTH = 50 #Cambiar 
MAX_PLATE_WIDTH = 250 

def get_video_reader(source, mode="PC"):
    if mode == "PC":
        return cv2.VideoCapture(source)
    elif mode == "JETSON":
        if source == "camera":
            gst_pipeline = (
                "nvarguscamerasrc sensor-id=0 ! "
                f"video/x-raw(memory:NVMM), width=(int){FRAME_WIDTH}, height=(int){FRAME_HEIGHT}, framerate=(fraction)30/1 ! "
                "nvvidconv flip-method=0 ! "
                f"video/x-raw, width=(int){FRAME_WIDTH}, height=(int){FRAME_HEIGHT}, format=(string)BGRx ! "
                "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true"
            )
            return cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        else:
            gst_pipeline = (
                f"filesrc location={source} ! qtdemux ! h264parse ! "
                "nvv4l2decoder ! nvvidconv ! "
                f"video/x-raw, width=(int){FRAME_WIDTH}, height=(int){FRAME_HEIGHT}, format=(string)BGRx ! "
                "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=false sync=false"
            )
            return cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    raise ValueError(f"Modo no reconocido: {mode}")

# ====================================================================
# PROCESO 1: LECTOR (Zero-Copy)
# ====================================================================
def reader_process(source, mode, raw_array, free_queue, ai_queue, config_queue, stop_event):
    print(f"[READER] Iniciado en modo {mode}.")
    shared_array = np.frombuffer(raw_array, dtype=np.uint8).reshape((BUFFER_SIZE, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS))
    cap = get_video_reader(source, mode=mode)
    
    if not cap.isOpened():
        print(f"[READER ERROR] No se pudo abrir la fuente: {source}")
        config_queue.put(30.0)
        ai_queue.put(None)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if (fps <= 0 or np.isnan(fps)) else fps
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
                break

            if frame.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            shared_array[index] = frame
            frame_count += 1
            ai_queue.put((frame_count, index))
    except Exception as e:
        print(f"[READER ERROR]: {e}")
    finally:
        ai_queue.put(None)
        cap.release()
        print("[READER] Finalizado.")

# ====================================================================
# PROCESO 2: IA (Matemáticas y Máquina de Estados)
# ====================================================================
def traffic_track(raw_array, traffic_queue, render_queue, stop_event):
    print("[TRAFFIC] Iniciado y cargando modelo...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    traffic_ai = YOLO(TRAFFIC_MODEL)

    shared_array = np.frombuffer(raw_array, dtype=np.uint8).reshape((BUFFER_SIZE, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS))

    try:
        while not stop_event.is_set():
            try:
                data = traffic_queue.get(timeout=1.0)
            except mp.queues.Empty:
                continue

            if data is None:
                break

            frame_count, index = data
            frame = shared_array[index]

            results = traffic_ai.track(
                source=frame, persist=True, conf=CONF_THRESH,
                imgsz=IMGSZ_YOLO, device=device, tracker="bytetrack.yaml", verbose=False
            )
            result = results[0]

            current_detections = []
            if result.boxes is not None and len(result.boxes) > 0 and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    current_detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "track_id": track_id
                    })

            metadata = {
                "frame_id": frame_count,
                "detections": current_detections
            }
            render_queue.put((frame_count, index, metadata))

    except Exception as e:
        print(f"[TRAFFIC ERROR]: {e}")
    finally:
        render_queue.put(None)
        print("[TRAFFIC] Finalizado.")


def ai_process(raw_array, ai_queue, render_queue, stop_event):
    print("[AI] Iniciando y cargando modelos...")
    
    # REGLA DE ORO: Inicialización CUDA estrictamente adentro del proceso hijo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    plate_ai = YOLO(PLATES_MODEL)

    ocr_model = ocr.LPRNet(num_classes=len(ocr.CHARS)).to(device)
    if os.path.exists(ocr.LPRNET_MODEL_PATH):
        ocr_model.load_state_dict(torch.load(ocr.LPRNET_MODEL_PATH, map_location=device, weights_only=True))
    else:
        print(f"[AI ERROR] No se encontró el modelo LPRNet en {ocr.LPRNET_MODEL_PATH}")
        stop_event.set()
        render_queue.put(None)
        return
    ocr_model.eval()
    
    # Memoria de estados aislada en este proceso
    vehicle_states = {}
    track_ocr_final = {}

    shared_array = np.frombuffer(raw_array, dtype=np.uint8).reshape((BUFFER_SIZE, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS))

    try:
        while not stop_event.is_set():
            try:
                data = ai_queue.get(timeout=1.0)
            except mp.queues.Empty:
                continue

            if data is None:
                break

            frame_count, index = data
            frame = shared_array[index]
            
            # Lista de detecciones empaquetadas para enviar al Proceso 3
            current_detections = []

            results = plate_ai.track(
                source=frame, persist=True, conf=CONF_THRESH,
                imgsz=IMGSZ_YOLO, device=device, tracker="bytetrack.yaml", verbose=False
            )
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0 and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    raw_w = x2 - x1
                    raw_h = y2 - y1
                    print(f"[AI] raw_w: {raw_w}, raw_h: {raw_h}, track_id: {track_id}")

                    if track_id not in vehicle_states:
                        vehicle_states[track_id] = 0

                    # --- MÁQUINA DE ESTADOS CORRECTA ---
                    if vehicle_states[track_id] == 0:
                        if MIN_PLATE_WIDTH <= raw_w <= MAX_PLATE_WIDTH and raw_h <= raw_w*0.7: 
                            
                            # CROP PADDING (Aquí es donde extraes la imagen para OCR)
                            #if raw_w >= 200: 
                                #pad_x = int(raw_w * 0.9)
                            #else:
                            pad_x = int(raw_w * 0.08)
                            pad_y = int(raw_h * 0.08)
                            crop_x1 = max(0, x1 - pad_x)
                            crop_y1 = max(0, y1 - pad_y)
                            crop_x2 = min(FRAME_WIDTH, x2 + pad_x)
                            crop_y2 = min(FRAME_HEIGHT, y2 + pad_y)
                                
                                
                            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                                
                            # El crop NO se envia bien al render, es mas pequeno
                            track_ocr_final[track_id] = ocr.run_lprnet_inference(crop, ocr_model, device)
                            vehicle_states[track_id] = 1 
                                
                        elif raw_w > MAX_PLATE_WIDTH:
                            vehicle_states[track_id] = 2

                    # Empaquetar la info del vehículo para que Render dibuje
                    current_detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "track_id": track_id,
                        "state": vehicle_states[track_id],
                        "width": raw_w,
                        "text": track_ocr_final.get(track_id, "")
                    })

            # Enviar metadatos reales al proceso de Render
            metadata = {
                "frame_id": frame_count,
                "detections": current_detections
            }
            render_queue.put((frame_count, index, metadata))

    except Exception as e:
        print(f"[AI ERROR]: {e}")
    finally:
        render_queue.put(None)
        print("[AI] Finalizado.")

# ====================================================================
# PROCESO 3: RENDER (Dibujado y Codificación de Salida)
# ====================================================================
def render_process(raw_array, render_queue, free_queue, config_queue, stop_event):
    fps_out = config_queue.get()
    shared_array = np.frombuffer(raw_array, dtype=np.uint8).reshape((BUFFER_SIZE, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("../system-monitoring-files/output.mp4", fourcc, fps_out, (FRAME_WIDTH, FRAME_HEIGHT))
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
            frame = shared_array[index].copy() # Hacemos copia para no ensuciar la RAM si hay lecturas cruzadas

            # Dibujar cada detección en base al estado reportado por AI
            for det in metadata["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                state = det["state"]
                t_id = det["track_id"]
                
                if state == 1:
                    color = (0, 255, 0)
                    label = f"ID:{t_id} | {det['text']}"
                elif state == 2:
                    color = (0, 0, 255)
                    label = f"ID:{t_id} | FAIL"
                else:
                    color = (0, 255, 255)
                    label = f"ID:{t_id} | W: {det['width']}px"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + len(label) * 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            out.write(frame)
            processed_count += 1
            free_queue.put(index) # Devolver índice para el proceso Lector

    except Exception as e:
        print(f"[RENDER ERROR]: {e}")
    finally:
        out.release()
        print(f"[RENDER] Finalizado. Cuadros guardados: {processed_count}")

# ====================================================================
# MAIN (Controlador de Hilos)
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
        print("\n[MAIN] Apagado de emergencia...")
        stop_event.set()
        p_reader.join()
        p_ai.join()
        p_render.join()
    finally:
        print("[MAIN] Sistema apagado limpiamente.")