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
TEMPLATES_DIR = "./scripts/plates_script/cr_plates_template"
OUTPUT_IMG_DIR = "./Dataset_LPRNet_Synth/images"
OUTPUT_LBL_DIR = "./Dataset_LPRNet_Synth/labels"

FONT_MAIN_PATH = os.path.join(TEMPLATES_DIR, "roadgeek-2005-engschrift.ttf")
FONT_BAR_PATH = os.path.join(TEMPLATES_DIR, "biosolid-regular.ttf")

NUM_SAMPLES = 4000 # Cantidad ideal para entrenar LPRNet desde cero

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)


# ====================================================================
# RENDERIZADO DE TEXTO EN PLANTILLAS PIL
# ====================================================================
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

    # Extraer texto de etiqueta limpia para LPRNet (sin espacios ni guiones)
    if bar_letters:
        gt_label = "".join(bar_letters) + text.replace("-", "").replace(" ", "")
    else:
        gt_label = text.replace("-", "").replace(" ", "")

    return template, text, gt_label, color, layout, bar_letters


def create_cr_plate_image():
    template_file, text, gt_label, color_rgb, layout, bar_letters = generate_plate_config()
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
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return img_bgr, gt_label


# ====================================================================
# PROYECCIÓN A 94x24 + DEGRADACIÓN ADAPTATIVA PARA LPRNET
# ====================================================================
def warp_and_degrade_to_94x24(img_bgr):
    """Proyecta la imagen a 94x24 aplicando perspectiva, ruido y desenfoque."""
    h, w = img_bgr.shape[:2]

    # Simular imprecisión del bounding box/perspectiva en el detector
    dx = w * 0.03
    dy = h * 0.05
    src_pts = np.float32([
        [random.uniform(-dx, dx), random.uniform(-dy, dy)],
        [w + random.uniform(-dx, dx), random.uniform(-dy, dy)],
        [w + random.uniform(-dx, dx), h + random.uniform(-dy, dy)],
        [random.uniform(-dx, dx), h + random.uniform(-dy, dy)]
    ])
    dst_pts = np.float32([[0, 0], [93, 0], [93, 23], [0, 23]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    crop = cv2.warpPerspective(img_bgr, M, (94, 24))

    # 1. Variación de Brillo y Contraste
    alpha = random.uniform(0.65, 1.15)
    beta = random.uniform(-35, 35)
    crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)

    # 2. Desenfoque Gaussiano / Movimiento aleatorio
    if random.random() > 0.4:
        crop = cv2.GaussianBlur(crop, (3, 3), 0)

    # 3. Ruido de sensor (Ruido Gaussiano)
    if random.random() > 0.5:
        sigma = random.uniform(4, 12)
        noise = np.random.normal(0, sigma, crop.shape).astype(np.float32)
        crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Artefactos de compresión JPEG
    if random.random() > 0.3:
        quality = random.randint(30, 75)
        _, encoded = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        crop = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return crop


# ====================================================================
# BUCLE PRINCIPAL
# ====================================================================
def main():
    print(f"[INFO] Generando {NUM_SAMPLES} recortes de 94x24 para LPRNet...")

    for idx in range(1, NUM_SAMPLES + 1):
        # 1. Crear placa base usando plantillas y tipografía costarricense
        plate_bgr, gt_label = create_cr_plate_image()

        # 2. Transformar a 94x24 con distorsiones reales
        crop_94x24 = warp_and_degrade_to_94x24(plate_bgr)

        # 3. Guardar imagen y archivo de texto
        file_id = f"cr_synth_{idx:05d}"
        img_path = os.path.join(OUTPUT_IMG_DIR, f"{file_id}.jpg")
        lbl_path = os.path.join(OUTPUT_LBL_DIR, f"{file_id}.txt")

        cv2.imwrite(img_path, crop_94x24)
        with open(lbl_path, "w") as f:
            f.write(gt_label)

        if idx % 500 == 0:
            print(f"Procesadas: {idx}/{NUM_SAMPLES}")

    print(f"[ÉXITO] {NUM_SAMPLES} recortes generados en '{OUTPUT_IMG_DIR}'.")


if __name__ == "__main__":
    main()