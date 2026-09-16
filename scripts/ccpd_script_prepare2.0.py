#!/usr/bin/env python3
import os
import random
import re
import shutil
from pathlib import Path
import cv2

# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================
CCPD_CR_DIR = Path("./ccpd_cr")       # Carpeta raíz con las subcarpetas
OUTPUT_DIR = Path("./dataset_yolo_cr") # Carpeta destino formateada para YOLO

EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Subcarpetas sintéticas -> VAN DERECHO A TRAIN
SYNTHETIC_FOLDERS = [
    "ccpd_base",
    "ccpd_blur",
    "ccpd_tilt",
    "ccpd_db",
    "ccpd_fn",
    "ccpd_np"
]

# Subcarpeta real -> VA DERECHO A VAL
REAL_FOLDER = "ccpd_irl"

# Directorios de salida
TRAIN_IMAGES = OUTPUT_DIR / "images" / "train"
VAL_IMAGES = OUTPUT_DIR / "images" / "val"
TRAIN_LABELS = OUTPUT_DIR / "labels" / "train"
VAL_LABELS = OUTPUT_DIR / "labels" / "val"


def create_directories():
    """Limpia la estructura anterior y crea los directorios limpios."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for directory in [TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS]:
        directory.mkdir(parents=True, exist_ok=True)


def parse_bbox_from_vertices(filename):
    """
    CORRECCIÓN CRÍTICA: Extrae las coordenadas reales leyendo los 4 vértices
    del bloque 3 (parts[3]) e ignora el bloque 2 estático.
    """
    parts = filename.split("-")
    if len(parts) < 4:
        raise ValueError(f"Formato no reconocido: {filename}")

    # Forzar la lectura estricta del bloque 3 (rb_lb_lt_rt)
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


def process_file(image_path, folder_name, dest_img_dir, dest_lbl_dir):
    """Procesa una imagen y genera/copia su etiqueta correspondiente."""
    new_stem = f"{folder_name}_{image_path.stem}"
    dest_img = dest_img_dir / f"{new_stem}{image_path.suffix}"
    dest_lbl = dest_lbl_dir / f"{new_stem}.txt"

    # CASO 1: Si existe el archivo .txt generado manualmente por el etiquetador
    existing_txt = image_path.with_suffix(".txt")
    if existing_txt.exists():
        shutil.copy2(image_path, dest_img)
        shutil.copy2(existing_txt, dest_lbl)
        return

    # CASO 2: Muestra negativa (ccpd_np) -> Genera un .txt vacío
    if folder_name == "ccpd_np":
        shutil.copy2(image_path, dest_img)
        dest_lbl.touch()
        return

    # CASO 3: Imagen con coordenadas codificadas en el nombre
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] No se pudo leer la imagen: {image_path}")
        return

    h, w = image.shape[:2]
    try:
        x1, y1, x2, y2 = parse_bbox_from_vertices(image_path.name)
        cx, cy, bw, bh = bbox_to_yolo(x1, y1, x2, y2, w, h)

        shutil.copy2(image_path, dest_img)
        with open(dest_lbl, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    except Exception as e:
        print(f"[WARN] Error procesando etiqueta para {image_path.name}: {e}")


def main():
    if not CCPD_CR_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta origen: {CCPD_CR_DIR.resolve()}")

    create_directories()

    print("==================================================")
    print("PROCESANDO DATASET: SINTÉTICO (TRAIN) / REAL (VAL)")
    print("==================================================")

    # 1. PROCESAR SINTÉTICOS -> TRAIN
    train_count = 0
    for folder in SYNTHETIC_FOLDERS:
        folder_path = CCPD_CR_DIR / folder
        if not folder_path.exists():
            print(f"[SKIP] Carpeta sintética no encontrada: {folder}")
            continue

        images = [p for p in folder_path.glob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS]
        print(f"-> Cargando {len(images):5d} imágenes de '{folder}' hacia TRAIN...")

        for img in images:
            process_file(img, folder, TRAIN_IMAGES, TRAIN_LABELS)
            train_count += 1

    # 2. PROCESAR REALES -> VAL
    # PROCESAR REALES -> SPLIT 70% TRAIN / 30% VAL
    real_path = CCPD_CR_DIR / REAL_FOLDER
    if real_path.exists():
        real_images = [p for p in real_path.glob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS]
        random.shuffle(real_images)
        
        split_idx = int(len(real_images) * 0.70)
        real_train = real_images[:split_idx]
        real_val = real_images[split_idx:]

        for img in real_train:
            process_file(img, REAL_FOLDER, TRAIN_IMAGES, TRAIN_LABELS)
        for img in real_val:
            process_file(img, REAL_FOLDER, VAL_IMAGES, VAL_LABELS)
            
        print(f"-> REALES (`ccpd_irl`): {len(real_train)} a TRAIN | {len(real_val)} a VAL")

    # 3. GENERAR ARCHIVO DATA.YAML
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        f.write("  0: license_plate\n")

    print("\n==================================================")
    print("RESUMEN DE ORGANIZACIÓN DEL DATASET")
    print("==================================================")
    print(f"Entrenamiento (Sintético): {train_count} imágenes")
    print(f"Validación    (Real):      {len(real_val)} imágenes")
    print(f"Directorio YOLO creado en: {OUTPUT_DIR.resolve()}")
    print(f"Archivo YAML:              {yaml_path.resolve()}")


if __name__ == "__main__":
    main()