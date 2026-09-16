#!/usr/bin/env python3
import os
import re
import random
import shutil
from pathlib import Path
import cv2

# ============================================================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ============================================================
CCPD_CR_DIR = Path("./ccpd_cr")       # Carpeta origen con las subcarpetas
OUTPUT_DIR = Path("./dataset_yolo_cr") # Carpeta destino formateada para YOLO

TRAIN_RATIO = 0.80
SEED = 42

EXTENSIONS = {".jpg", ".jpeg", ".png"}

SUBFOLDERS = [
    "ccpd_base",
    "ccpd_blur",
    "ccpd_tilt",
    "ccpd_db",
    "ccpd_fn",
    "ccpd_np",
    "ccpd_irl"
]

# Directorios YOLO
TRAIN_IMAGES = OUTPUT_DIR / "images" / "train"
VAL_IMAGES = OUTPUT_DIR / "images" / "val"
TRAIN_LABELS = OUTPUT_DIR / "labels" / "train"
VAL_LABELS = OUTPUT_DIR / "labels" / "val"


def create_directories():
    for directory in [TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS]:
        directory.mkdir(parents=True, exist_ok=True)


def parse_bbox(filename):
    """Extrae la caja envolvente real leyendo directamente los 4 vértices (bloque 3)."""
    parts = filename.split("-")
    if len(parts) < 4:
        raise ValueError(f"Formato no reconocido: {filename}")

    # Forzar la lectura de la parte 3 (vértices reales rb_lb_lt_rt)
    vertices_raw = parts[3].split("_")
    if len(vertices_raw) == 4:  
        pts = [list(map(int, pt.split("&"))) for pt in vertices_raw]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    raise ValueError(f"Imposible parsear vértices en: {filename}")


def bbox_to_yolo(x1, y1, x2, y2, width, height):
    """Convierte píxeles a formato YOLO normalizado (cx, cy, w, h)."""
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    return cx / width, cy / height, bw / width, bh / height



def process_image(image_path, folder_name, dest_img_dir, dest_lbl_dir):
    new_stem = f"{folder_name}_{image_path.stem}"
    dest_img = dest_img_dir / f"{new_stem}{image_path.suffix}"
    dest_lbl = dest_lbl_dir / f"{new_stem}.txt"

    # Prioridad 1: Si existe el archivo .txt generado por el etiquetador interactivo
    existing_txt = image_path.with_suffix(".txt")
    if existing_txt.exists():
        shutil.copy2(image_path, dest_img)
        shutil.copy2(existing_txt, dest_lbl)
        return

    # Prioridad 2: Si es muestra negativa
    if folder_name == "ccpd_np":
        shutil.copy2(image_path, dest_img)
        dest_lbl.touch()
        return

    # Prioridad 3: Parseo corregido por vértices
    image = cv2.imread(str(image_path))
    if image is None:
        return

    h, w = image.shape[:2]
    x1, y1, x2, y2 = parse_bbox(image_path.name)

    # Convertir a YOLO
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    shutil.copy2(image_path, dest_img)
    with open(dest_lbl, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

def main():
    if not CCPD_CR_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta origen: {CCPD_CR_DIR.resolve()}")

    create_directories()
    random.seed(SEED)

    all_train, all_val = [], []

    print("========================================")
    print("ESCANEO Y DIVISIÓN DE SUB-DATASETS")
    print("========================================")

    for folder in SUBFOLDERS:
        folder_path = CCPD_CR_DIR / folder
        if not folder_path.exists():
            print(f"[SKIP] Subcarpeta no encontrada: {folder}")
            continue

        images = [p for p in folder_path.glob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS]
        random.shuffle(images)

        train_count = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:train_count]
        val_imgs = images[train_count:]

        print(f"-> {folder:15s}: {len(images):5d} total | Train: {len(train_imgs):5d} | Val: {len(val_imgs):5d}")

        for img in train_imgs:
            all_train.append((img, folder))
        for img in val_imgs:
            all_val.append((img, folder))

    print("\n========================================")
    print("PROCESANDO ARCHIVOS PARA YOLO")
    print("========================================")

    print(f"Procesando Entrenamiento ({len(all_train)} imágenes)...")
    for i, (img_path, folder_name) in enumerate(all_train, 1):
        process_image(img_path, folder_name, TRAIN_IMAGES, TRAIN_LABELS)
        if i % 1000 == 0:
            print(f"Train: {i}/{len(all_train)}")

    print(f"\nProcesando Validación ({len(all_val)} imágenes)...")
    for i, (img_path, folder_name) in enumerate(all_val, 1):
        process_image(img_path, folder_name, VAL_IMAGES, VAL_LABELS)
        if i % 500 == 0:
            print(f"Val: {i}/{len(all_val)}")

    # --------------------------------------------------------
    # GENERACIÓN DEL ARCHIVO DATA.YAML
    # --------------------------------------------------------
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        f.write("  0: license_plate\n")

    print("\n========================================")
    print("DATASET CREADO CON ÉXITO")
    print("========================================")
    print("Carpeta Destino:", OUTPUT_DIR.resolve())
    print("Archivo YAML:   ", yaml_path.resolve())


if __name__ == "__main__":
    main()