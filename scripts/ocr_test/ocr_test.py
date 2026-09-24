import cv2
import numpy as np
import torch
import torch.nn as nn

CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', '-'
]

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
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)), # W: 94 -> 47
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)), # W: 47 -> 23
            nn.Dropout(0.5),
            # CORRECCIÓN: Kernel 1x4 amplía la secuencia temporal T de 11 a 20
            nn.Conv2d(256, 512, kernel_size=(1, 4), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        )

    def forward(self, x):
        x = self.backbone(x)
        return torch.mean(x, dim=2)

def decode_greedy(preds):
    """Greedy Decoder: Remueve duplicados consecutivos y filtra el token blank (-)"""
    pred_labels = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy()
    char_list = []
    prev_idx = -1
    blank_idx = len(CHARS) - 1
    
    for idx in pred_labels:
        if idx != prev_idx and idx != blank_idx:
            char_list.append(CHARS[idx])
        prev_idx = idx
    return "".join(char_list)

def test_inference(img_path, model_path="./models/ocr/lprnet_cr_best_v2.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LPRNet(num_classes=len(CHARS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    img = cv2.imread(img_path)
    if img is None:
        print(f"No se encontró la imagen en {img_path}")
        return

    img = cv2.resize(img, (94, 24))
    img = (img.astype(np.float32) - 127.5) / 128.0
    tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor)
        text = decode_greedy(preds)
        print(f"Predicción para {img_path}: '{text}'")

if __name__ == "__main__":
    test_inference("./Dataset_LPRNet_Synth/images/cr_synth_00039.jpg")
    #test_inference("./Dataset_LPRNet/images/00020.jpg")