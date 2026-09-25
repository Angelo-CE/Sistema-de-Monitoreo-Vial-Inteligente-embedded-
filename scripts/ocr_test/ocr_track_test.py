import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# ==========================================
# CONFIGURACIÓN Y PARÁMETROS
# ==========================================
YOLO_MODEL_PATH = "./models/3.0/best.pt"  
LPRNET_MODEL_PATH = "./models/ocr/lprnet_cr_best_v2.pth"
#VIDEO_INPUT_PATH = "./videos_test/highway932.mp4"
VIDEO_INPUT_PATH = "./videos_test/IMG_0483.MOV"
VIDEO_OUTPUT_PATH = "output_pipeline.mp4"

CONF_THRESH = 0.35
IMGSZ_YOLO = 640

# --- UMBRALES DE LA ZONA DULCE (SWEET SPOT) ---
MIN_PLATE_WIDTH = 70   # Muy lejos = píxeles basura
MAX_PLATE_WIDTH = 100  # Muy cerca = distorsión/desenfoque de borde

CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', '-'
]

# ==========================================
# ARQUITECTURA LPRNET (T=20)
# ==========================================
class LPRNet(nn.Module):
    def __init__(self, num_classes=37):
        super(LPRNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(256, 512, kernel_size=(1, 4), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        )

    def forward(self, x):
        x = self.backbone(x)
        return torch.mean(x, dim=2)


def decode_greedy(preds):
    pred_labels = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy()
    char_list = []
    prev_idx = -1
    blank_idx = len(CHARS) - 1
    
    for idx in pred_labels:
        if idx != prev_idx and idx != blank_idx:
            char_list.append(CHARS[idx])
        prev_idx = idx
    return "".join(char_list)


def preprocess_crop_clahe(crop_bgr):
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def run_lprnet_inference(crop_bgr, model, device):
    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 10:
        return ""

    crop_clahe = preprocess_crop_clahe(crop_bgr)

    # CORREGIDO: Ahora sí se redimensiona la imagen con CLAHE aplicado
    img = cv2.resize(crop_clahe, (94, 24))
    img = (img.astype(np.float32) - 127.5) / 128.0
    tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor)
        text = decode_greedy(preds)
    return text


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo de ejecución: {device}")

    yolo_model = YOLO(YOLO_MODEL_PATH)
    lprnet_model = LPRNet(num_classes=len(CHARS)).to(device)
    
    if os.path.exists(LPRNET_MODEL_PATH):
        lprnet_model.load_state_dict(torch.load(LPRNET_MODEL_PATH, map_location=device, weights_only=True))
        print(f"[INFO] LPRNet cargado correctamente desde {LPRNET_MODEL_PATH}")
    else:
        print(f"[ERROR] No se encontró el modelo {LPRNET_MODEL_PATH}")
        return

    lprnet_model.eval()

    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el video: {VIDEO_INPUT_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (width, height))

    # MÁQUINA DE ESTADOS
    # 0 = Tracking (Esperando zona dulce)
    # 1 = Placa Capturada con éxito
    # 2 = Descartado (Se acercó demasiado y falló)
    vehicle_states = {}
    track_ocr_final = {}

    frame_count = 0
    total_time = 0.0

    print("[INFO] Procesando video con State Machine (Best-Shot)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t_start = time.perf_counter()

        results = yolo_model.track(
            source=frame,
            persist=True,
            conf=CONF_THRESH,
            imgsz=IMGSZ_YOLO,
            device=device,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0 and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Tamaño puro de la caja para la lógica de estados
                raw_w = x2 - x1
                raw_h = y2 - y1

                # Inicializar nuevo vehículo
                if track_id not in vehicle_states:
                    vehicle_states[track_id] = 0

                # --- LÓGICA DE ESTADOS ---
                if vehicle_states[track_id] == 0:
                    #if MIN_PLATE_WIDTH <= raw_w <= MAX_PLATE_WIDTH:
                        
                        # APLICAR PADDING SOLO PARA EL RECORTE DEL OCR
                        pad_x = int(raw_w * 0.08)
                        pad_y = int(raw_h * 0.08)
                        crop_x1 = max(0, x1 - pad_x)
                        crop_y1 = max(0, y1 - pad_y)
                        crop_x2 = min(width, x2 + pad_x)
                        crop_y2 = min(height, y2 + pad_y)

                        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        # INFERENCIA ÚNICA
                        plate_text = run_lprnet_inference(crop, lprnet_model, device)

                        # Validación estricta: Placas CR tienen mínimo 5 o 6 caracteres útiles
                        #if len(plate_text) >= 5:
                        track_ocr_final[track_id] = plate_text
                        vehicle_states[track_id] = 1  # ÉXITO
                        #else:
                            # Sigue en estado 0, intentará en el siguiente frame si no sale de la zona
                            #pass
                            
                    #elif raw_w > MAX_PLATE_WIDTH:
                        # Se acercó mucho y cruzó el límite sin una lectura exitosa
                        #vehicle_states[track_id] = 2  # DESCARTADO

                # --- RENDERIZADO VISUAL ---
                if vehicle_states[track_id] == 1:
                    box_color = (0, 255, 0) # Verde
                    display_text = f"ID:{track_id} | {track_ocr_final[track_id]}"
                elif vehicle_states[track_id] == 2:
                    box_color = (0, 0, 255) # Rojo
                    display_text = f"ID:{track_id} | FAIL"
                else:
                    # Estado 0 (Tracking)
                    box_color = (0, 255, 255) # Amarillo
                    display_text = f"ID:{track_id} | W: {raw_w}px"

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + len(display_text) * 10, y1), box_color, -1)
                cv2.putText(frame, display_text, (x1 + 3, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        t_end = time.perf_counter()
        total_time += (t_end - t_start)
        out.write(frame)

    cap.release()
    out.release()

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print("\n========== RESUMEN DE EJECUCIÓN ==========")
    print(f"Frames procesados: {frame_count}")
    print(f"Velocidad promedio: {avg_fps:.2f} FPS")
    print(f"Placas exitosas: {sum(1 for v in vehicle_states.values() if v == 1)} / {len(vehicle_states)}")
    print(f"Video exportado en: {VIDEO_OUTPUT_PATH}")
    print("==========================================")

if __name__ == "__main__":
    main()