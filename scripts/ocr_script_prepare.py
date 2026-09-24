import os
import glob
import cv2
import numpy as np

INPUT_DIR = "./ccpd_cr/ccpd_irl"  # Carpeta con imágenes y coordenadas en nombre o .txt
OUTPUT_IMG_DIR = "./Dataset_LPRNet/images"
OUTPUT_LBL_DIR = "./Dataset_LPRNet/labels"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

def align_plate_94x24(img, pts):
    """Proyecta la placa a una imagen plana de 94x24 píxeles."""
    src_pts = np.array(pts, dtype="float32")
    dst_pts = np.array([[0, 0], [93, 0], [93, 23], [0, 23]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (94, 24))

def parse_vertices_from_filename(filename):
    parts = filename.split("-")
    if len(parts) >= 4:
        vertices_raw = parts[3].split("_")
        if len(vertices_raw) == 4:
            pts = [list(map(int, pt.split("&"))) for pt in vertices_raw]
            # Orden de puntos: LT, RT, RB, LB
            return [pts[2], pts[3], pts[0], pts[1]]
    return None

img_files = glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_DIR, "*.png"))

for idx, img_path in enumerate(img_files):
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    pts = parse_vertices_from_filename(os.path.basename(img_path))
    if pts is not None:
        crop = align_plate_94x24(img, pts)
        
        name_id = f"{idx:04d}"
        out_img_path = os.path.join(OUTPUT_IMG_DIR, f"{name_id}.jpg")
        out_lbl_path = os.path.join(OUTPUT_LBL_DIR, f"{name_id}.txt")
        
        cv2.imwrite(out_img_path, crop)
        
        # Crear txt para que coloques el texto manualmente (si es real)
        if not os.path.exists(out_lbl_path):
            with open(out_lbl_path, "w") as f:
                f.write("") # Escribir el texto real aquí (ej: CLB123)

print("[ÉXITO] Recortes 94x24 generados en ./Dataset_LPRNet")