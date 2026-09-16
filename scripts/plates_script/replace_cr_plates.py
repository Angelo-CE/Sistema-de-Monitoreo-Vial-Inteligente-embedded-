import os
import glob
import cv2
import numpy as np
import random
import string
from PIL import Image, ImageDraw, ImageFont

# ====================================================================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ====================================================================
CCPD_DIR = "./datasets/CCPD2019/ccpd_np"
#CCPD_DIR = "./fotos_reales_ccpd/images"
TEMPLATES_DIR = "./scripts/plates_script/cr_plates_template"
OUTPUT_DIR = "./ccpd_cr/ccpd_np"

FONT_MAIN_PATH = os.path.join(TEMPLATES_DIR, "roadgeek-2005-engschrift.ttf")
FONT_BAR_PATH = os.path.join(TEMPLATES_DIR, "biosolid-regular.ttf")

NUM_SAMPLES = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_host_plate_roi(car_img, dst_pts):
    """Analiza el nivel de brillo y desenfoque del área original de la placa en CCPD."""
    x, y, w, h = cv2.boundingRect(dst_pts.astype(np.int32))
    
    # Asegurar coordenadas dentro de la imagen
    h_img, w_img = car_img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    roi = car_img[y1:y2, x1:x2]
    if roi.size == 0:
        return 120.0, 300.0  # Valores por defecto si falla el recorte

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 1. Medir Brillo Promedio (0 a 255)
    mean_brightness = np.mean(gray_roi)

    # 2. Medir Desenfoque usando Varianza del Laplaciano (Valores < 100 indican imagen borrosa)
    blur_score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()

    return mean_brightness, blur_score


def adapt_synthetic_plate(cr_plate, target_brightness, blur_score):
    """Adapta el brillo, contraste y desenfoque de la placa sintética para que coincida con el auto."""
    # --- 1. AJUSTE DE BRILLO Y CONTRASTE ---
    current_brightness = np.mean(cv2.cvtColor(cr_plate, cv2.COLOR_BGR2GRAY))
    
    if current_brightness > 0:
        # Factor de escala para igualar la iluminación del auto objetivo
        beta = target_brightness - current_brightness
        # Ajustar manteniendo contraste natural
        cr_plate = cv2.convertScaleAbs(cr_plate, alpha=0.85, beta=beta)

    # --- 2. APLICACIÓN DE DESENFOQUE ADAPTATIVO (BLUR) ---
    # Si la imagen original es borrosa (blur_score bajo), aplicamos desenfoque proporcional
    if blur_score < 80:
        k_size = 9  # Muy borroso
    elif blur_score < 200:
        k_size = 7  # Borroso moderado
    elif blur_score < 400:
        k_size = 5  # Leve desenfoque
    else:
        k_size = 3  # Nitidez normal

    if k_size > 3:
        # Alternar entre Gaussian Blur y Motion Blur aleatoriamente
        if random.random() > 0.5:
            cr_plate = cv2.GaussianBlur(cr_plate, (k_size, k_size), 0)
        else:
            # Simular Motion Blur (desenfoque por movimiento vehicular)
            kernel_motion = np.zeros((k_size, k_size))
            kernel_motion[int((k_size - 1) / 2), :] = np.ones(k_size)
            kernel_motion /= k_size
            cr_plate = cv2.filter2D(cr_plate, -1, kernel_motion)

    # --- 3. RUIDO GAUSSIANO EN ESCENAS OSCURAS ---
    if target_brightness < 70:
        row, col, ch = cr_plate.shape
        mean = 0
        sigma = 15
        gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)
        noisy_plate = cr_plate.astype(np.float32) + gauss
        cr_plate = np.clip(noisy_plate, 0, 255).astype(np.uint8)

    return cr_plate


def get_text_dimensions(font, text, draw):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1], bbox[0], bbox[1]
    except AttributeError:
        w, h = font.getsize(text)
        return w, h, 0, 0


def draw_text_autofit(draw, text, target_rect, font_path, max_font_size, color_rgb):
    x_min, y_min, x_max, y_max = target_rect
    target_w, target_h = x_max - x_min, y_max - y_min

    font_size = max_font_size
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    while font_size > 15:
        text_w, text_h, _, _ = get_text_dimensions(font, text, draw)
        if text_w <= target_w * 0.92 and text_h <= target_h * 0.85:
            break
        font_size -= 4
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            break

    text_w, text_h, offset_x, offset_y = get_text_dimensions(font, text, draw)
    x = x_min + (target_w - text_w) / 2 - offset_x
    y = y_min + (target_h - text_h) / 2 - offset_y
    draw.text((x, y), text, fill=color_rgb, font=font)


def draw_vertical_bar_text(draw, letters, target_rect, color_rgb):
    x_min, y_min, x_max, y_max = target_rect
    target_h = y_max - y_min
    num_letters = len(letters)

    if num_letters == 0:
        return

    slot_h = target_h / num_letters
    for i, char in enumerate(letters):
        slot_ymin = y_min + (i * slot_h)
        slot_ymax = slot_ymin + slot_h
        draw_text_autofit(draw, char, (x_min, slot_ymin, x_max, slot_ymax), FONT_BAR_PATH, int(slot_h * 0.75), color_rgb)


def generate_plate_config():
    types = ["car-new", "car-old", "carga-liviana-bar", "carga-liviana", "disabled", "official-black"]
    weights = [0.45, 0.25, 0.15, 0.10, 0.03, 0.02]
    selected_type = random.choices(types, weights=weights)[0]

    if selected_type == "car-new":
        template, color, layout, bar_letters = "cr-blue-flag.jpg", (21, 45, 98), "full", None
        letters = ''.join(random.choices(string.ascii_uppercase.replace('O', '').replace('I', ''), k=3))
        numbers = ''.join(random.choices(string.digits, k=3))
        text = f"{letters}-{numbers}"

    elif selected_type == "car-old":
        template, color, layout, bar_letters = random.choice(["cr-blue-flag.jpg", "cr-blue-no-flag.jpg"]), (21, 45, 98), "full", None
        text = ''.join(random.choices(string.digits, k=6))

    elif selected_type == "carga-liviana-bar":
        template = random.choice(["cr-red-flag-bar.jpg", "cr-black-flag-bar.jpg"])
        color = (173, 24, 13) if "red" in template else (30, 30, 30)
        text = ''.join(random.choices(string.digits, k=6))
        layout, bar_letters = "bar", ["C", "L"]

    elif selected_type == "carga-liviana":
        template = random.choice(["cr-red-flag.jpg", "cr-red-no-flag.jpg"])
        numbers = ''.join(random.choices(string.digits, k=5))
        text, color, layout, bar_letters = f"CL-{numbers}", (173, 24, 13), "full", None

    elif selected_type == "disabled":
        template, text, color, layout, bar_letters = "cr-disable.jpg", f"D-{''.join(random.choices(string.digits, k=3))}", (21, 45, 98), "disabled", None

    else:
        template = random.choice(["cr-black-flag.jpg", "cr-black-flag-bar.jpg"])
        color = (30, 30, 30)
        if "bar" in template:
            text, layout, bar_letters = ''.join(random.choices(string.digits, k=5)), "bar", ["M", "1"]
        else:
            text, layout, bar_letters = f"{random.randint(10, 99)}-{random.randint(1000, 9999)}", "full", None

    return template, text, color, layout, bar_letters


def create_cr_plate_from_template():
    template_file, text, color_rgb, layout, bar_letters = generate_plate_config()
    template_path = os.path.join(TEMPLATES_DIR, template_file)

    if os.path.exists(template_path):
        img_pil = Image.open(template_path).convert("RGB")
    else:
        img_pil = Image.new('RGB', (1200, 600), color=(240, 240, 240))

    w, h = img_pil.size
    draw = ImageDraw.Draw(img_pil)

    if layout == "disabled":
        main_target = (int(w * 0.36), int(h * 0.18), int(w * 0.96), int(h * 0.85))
    elif layout == "bar":
        main_target = (int(w * 0.18), int(h * 0.18), int(w * 0.96), int(h * 0.85))
        bar_target = (int(w * 0.02), int(h * 0.12), int(w * 0.16), int(h * 0.88))
        draw_vertical_bar_text(draw, bar_letters, bar_target, color_rgb)
    else:
        main_target = (int(w * 0.03), int(h * 0.18), int(w * 0.97), int(h * 0.85))

    draw_text_autofit(draw, text, main_target, FONT_MAIN_PATH, int(h * 0.55), color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def parse_ccpd_vertices(filename):
    base_name = os.path.basename(filename).replace(".jpg", "").replace(".png", "")
    parts = base_name.split("-")
    vertices_raw = parts[3].split("_")

    pts = []
    for pt in vertices_raw:
        x, y = map(int, pt.split("&"))
        pts.append([x, y])

    rb, lb, lt, rt = pts[0], pts[1], pts[2], pts[3]
    return np.float32([lt, rt, rb, lb])


def overlay_plate(car_img, cr_plate, dst_pts):
    """
    Funde la placa sintética en la carrocería difuminando bordes y 
    homogeneizando el ruido de compresión de la escena completa.
    """
    h_plate, w_plate = cr_plate.shape[:2]
    src_pts = np.float32([[0, 0], [w_plate, 0], [w_plate, h_plate], [0, h_plate]])

    # 1. Homografía de la placa
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    h_car, w_car = car_img.shape[:2]
    warped_plate = cv2.warpPerspective(cr_plate, M, (w_car, h_car))

    # 2. Crear máscara base y encogerla para eliminar bordes blancos de corte
    mask = np.zeros((h_plate, w_plate), dtype=np.uint8) + 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=2)

    # 3. Mapear máscara a la escena y aplicar FEATHERING (suavizado de bordes)
    warped_mask = cv2.warpPerspective(mask, M, (w_car, h_car))
    feathered_mask = cv2.GaussianBlur(warped_mask, (7, 7), 0).astype(np.float32) / 255.0
    feathered_mask_3ch = cv2.merge([feathered_mask, feathered_mask, feathered_mask])

    # 4. Fusión por canales usando matriz Alpha en flotantes
    car_float = car_img.astype(np.float32)
    plate_float = warped_plate.astype(np.float32)

    blended = (plate_float * feathered_mask_3ch) + (car_float * (1.0 - feathered_mask_3ch))
    blended_img = np.clip(blended, 0, 255).astype(np.uint8)

    # 5. DESENFOQUE LIGERO EN LA ZONA DE FUSIÓN (Elimina bordes rectangulares)
    x, y, w, h = cv2.boundingRect(dst_pts.astype(np.int32))
    x1, y1 = max(0, x - 5), max(0, y - 5)
    x2, y2 = min(w_car, x + w + 5), min(h_car, y + h + 5)
    
    roi = blended_img[y1:y2, x1:x2]
    blended_img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (3, 3), 0)

    # 6. MIMETIZACIÓN DE COMPRESIÓN JPEG (Aplica grano a toda la escena en conjunto)
    jpeg_quality = random.randint(35, 65)
    _, encoded_jpg = cv2.imencode('.jpg', blended_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    final_img = cv2.imdecode(encoded_jpg, cv2.IMREAD_COLOR)

    return final_img


def process_dataset():
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    ccpd_files = []
    for ext in extensions:
        ccpd_files.extend(glob.glob(os.path.join(CCPD_DIR, ext)))

    if not ccpd_files:
        print(f"[ERROR CRÍTICO] No hay archivos en: {CCPD_DIR}")
        return

    samples_to_process = min(NUM_SAMPLES, len(ccpd_files))
    print(f"[INFO] Procesando {samples_to_process} imágenes con degradación adaptativa...")

    for ccpd_path in ccpd_files[:samples_to_process]:
        try:
            car_img = cv2.imread(ccpd_path)
            dst_pts = parse_ccpd_vertices(ccpd_path)

            # 1. Medir métricas visuales del auto en CCPD
            brightness, blur_score = analyze_host_plate_roi(car_img, dst_pts)

            # 2. Generar placa costarricense
            cr_plate = create_cr_plate_from_template()

            # 3. Aplicar degradación adaptativa para igualar el fondo
            cr_plate = adapt_synthetic_plate(cr_plate, brightness, blur_score)

            # 4. Superponer sobre el auto
            final_img = overlay_plate(car_img, cr_plate, dst_pts)

            filename = os.path.basename(ccpd_path)
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), final_img)

        except Exception as e:
            print(f"[WARN] Error procesando {ccpd_path}: {e}")

    print(f"[ÉXITO] Dataset sintético adaptativo generado en '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    process_dataset()