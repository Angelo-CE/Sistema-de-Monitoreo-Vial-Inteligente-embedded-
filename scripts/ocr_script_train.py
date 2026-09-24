import os
import glob
import cv2
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 0. PARÁMETROS GENERALES
# ==========================================
DATASET_IMG_DIR = "./Dataset_LPRNet/images"

DATASET_LBL_DIR = "./Dataset_LPRNet/labels"
EPOCHS = 50                            # Variable para definir las épocas
BATCH_SIZE = 8
LEARNING_RATE = 0.0001                # LR ligeramente más bajo para ajustes/fine-tuning
PRETRAINED_MODEL = "./models/ocr/lprnet_cr_best_v1.pth"  # Archivo a cargar (dejar en None si es desde cero)
SAVE_MODEL_PATH = "./models/ocr/lprnet_cr_best_v2.pth"

# ==========================================
# 1. DICCIONARIO Y CONFIGURACIÓN
# ==========================================
CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', '-'
]
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
BLANK_IDX = CHAR_TO_IDX['-']

class LPRNet(nn.Module):
    def __init__(self, num_classes=37):
        super(LPRNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            nn.Dropout(0.5),
            # KERNEL 1x4 NUEVO (Amplía la secuencia a T=20)
            nn.Conv2d(256, 512, kernel_size=(1, 4), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        )

    def forward(self, x):
        x = self.backbone(x)
        return torch.mean(x, dim=2)

# ==========================================
# 3. DATASET PYTORCH CON FILTRADO DE SEGURIDAD
# ==========================================
class LPRDatasetAugmented(Dataset):
    def __init__(self, img_dir, lbl_dir, is_train=True):
        raw_img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.img_paths = []
        self.lbl_dir = lbl_dir
        self.is_train = is_train

        # Filtrar etiquetas válidas no vacías
        for img_path in raw_img_paths:
            lbl_path = os.path.join(self.lbl_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as f:
                    if len(f.read().strip()) > 0:
                        self.img_paths.append(img_path)

    def __len__(self):
        return len(self.img_paths)

    def apply_augmentation(self, img):
        """Aplica transformaciones físicas reales a los recortes de 94x24."""
        h, w = img.shape[:2]

        # 1. Variación aleatoria de Brillo y Contraste
        if random.random() > 0.3:
            alpha = random.uniform(0.7, 1.3)  # Contraste
            beta = random.uniform(-30, 30)    # Brillo
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # 2. Rotación leve (simula cámara inclinada: -5° a +5°)
        if random.random() > 0.4:
            angle = random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 3. Blur o Desenfoque por movimiento vehicular
        if random.random() > 0.5:
            k = random.choice([3])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 4. Ruido de sensor (Ruido Gaussiano)
        if random.random() > 0.5:
            sigma = random.uniform(3, 10)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 5. Mimetización de sombras laterales (Corte aleatorio de brillo)
        if random.random() > 0.6:
            mask = np.ones_like(img, dtype=np.float32)
            cut_x = random.randint(0, w)
            factor = random.uniform(0.5, 0.8)
            mask[:, :cut_x] *= factor
            img = np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)

        return img

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        lbl_path = os.path.join(self.lbl_dir, os.path.basename(img_path).replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        
        # Aplicar aumentación en entrenamiento antes de redimensionar
        if self.is_train:
            img = self.apply_augmentation(img)

        img = cv2.resize(img, (94, 24))
        img = (img.astype(np.float32) - 127.5) / 128.0
        img = img.transpose(2, 0, 1)

        with open(lbl_path, "r") as f:
            label_str = f.read().strip().upper()

        label_targets = [CHAR_TO_IDX[c] for c in label_str if c in CHAR_TO_IDX]

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label_targets, dtype=torch.long), len(label_targets)


def collate_fn(batch):
    imgs, targets, target_lengths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return imgs, targets, target_lengths

# ==========================================
# 4. BUCLE DE ENTRENAMIENTO
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = LPRDatasetAugmented(DATASET_IMG_DIR, DATASET_LBL_DIR)
    print(f"[INFO] Muestras válidas encontradas en el dataset: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = LPRNet(num_classes=len(CHARS)).to(device)

    # CARGA DE PESOS PREVIOS (Fine-Tuning / Continuación)
    if PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL):
        try:
            model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=device, weights_only=True))
            print(f"[ÉXITO] Pesos del modelo anterior cargados desde: '{PRETRAINED_MODEL}'")
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudieron cargar los pesos: {e}. Entrenando desde cero.")
    else:
        print("[INFO] No se encontró modelo previo o no se especificó. Entrenando desde cero.")

    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"[INFO] Iniciando entrenamiento por {EPOCHS} épocas en {device}...")
    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for imgs, targets, target_lengths in dataloader:
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            
            preds = preds.permute(2, 0, 1)
            preds_log_softmax = preds.log_softmax(2)
            
            input_lengths = torch.full(size=(imgs.size(0),), fill_value=preds.size(0), dtype=torch.long)
            
            loss = ctc_loss(preds_log_softmax, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / len(dataloader)
            print(f"Época {epoch:03d}/{EPOCHS} | Pérdida CTC Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"[ÉXITO] Entrenamiento completado. Modelo actualizado guardado en '{SAVE_MODEL_PATH}'")

if __name__ == "__main__":
    train()