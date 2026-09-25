import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn


# ==========================================
# CONFIGURACIÓN Y PARÁMETROS
# ==========================================
  
LPRNET_MODEL_PATH = "./models/ocr/lprnet_cr_best_v2.pth"


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