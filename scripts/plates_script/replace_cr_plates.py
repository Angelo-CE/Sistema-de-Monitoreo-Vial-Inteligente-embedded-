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
CCPD_DIR = "./ccpd_yolo/images/train"
TEMPLATES_DIR = "./scripts/plates_script/cr_plates_template"
OUTPUT_DIR = "./output_dataset_plates_cr"

FONT_MAIN_PATH = os.path.join(TEMPLATES_DIR, "roadgeek-2005-engschrift.ttf")
FONT_BAR_PATH = os.path.join(TEMPLATES_DIR, "biosolid-regular.ttf")

NUM_SAMPLES = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_text_dimensions(font, text, draw):
    """Obtiene dimensiones exactas del texto compatibles con varias versiones de PIL."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1], bbox[0], bbox[1]
    except AttributeError:
        w, h = font.getsize(text)
        return w, h, 0, 0


def draw_text_autofit(draw, text, target_rect, font_path, max_font_size, color_rgb):
    """
    Centra y escala el texto automáticamente dentro de target_rect (x_min, y_min, x_max, y_max)
    para evitar colisiones con bordes u objetos de la plantilla.
    """
    x_min, y_min, x_max, y_max = target_rect
    target_w = x_max - x_min
    target_h = y_max - y_min

    font_size = max_font_size
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Reducir tamaño de fuente si el texto excede el área permitida
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

    # Calcular coordenadas de centrado exacto
    x = x_min + (target_w - text_w) / 2 - offset_x
    y = y_min + (target_h - text_h) / 2 - offset_y

    draw.text((x, y), text, fill=color_rgb, font=font)


def draw_vertical_bar_text(draw, letters, target_rect, color_rgb):
    """Dibuja letras apiladas verticalmente en la franja amarilla usando biosolid-regular.ttf."""
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
    """
    Configura el tipo de plantilla enfocándose prioritariamente en autos particulares 
    y cargas livianas (formato CCPD).
    """
    types = ["car-new", "car-old", "carga-liviana-bar", "carga-liviana", "disabled", "official-black"]
    weights = [0.45, 0.25, 0.15, 0.10, 0.03, 0.02]  # Distribución ponderada
    selected_type = random.choices(types, weights=weights)[0]

    if selected_type == "car-new":
        template = "cr-blue-flag.jpg"
        letters = ''.join(random.choices(string.ascii_uppercase.replace('O', '').replace('I', ''), k=3))
        numbers = ''.join(random.choices(string.digits, k=3))
        text = f"{letters}-{numbers}"
        color = (21, 45, 98) # Azul
        layout = "full"
        bar_letters = None

    elif selected_type == "car-old":
        template = random.choice(["cr-blue-flag.jpg", "cr-blue-no-flag.jpg"])
        text = ''.join(random.choices(string.digits, k=6))
        color = (21, 45, 98)
        layout = "full"
        bar_letters = None

    elif selected_type == "carga-liviana-bar":
        template = random.choice(["cr-red-flag-bar.jpg", "cr-black-flag-bar.jpg"])
        is_red = "red" in template
        color = (173, 24, 13) if is_red else (30, 30, 30)
        text = ''.join(random.choices(string.digits, k=6))
        layout = "bar"
        bar_letters = ["C", "L"]

    elif selected_type == "carga-liviana":
        template = random.choice(["cr-red-flag.jpg", "cr-red-no-flag.jpg"])
        numbers = ''.join(random.choices(string.digits, k=5))
        text = f"CL-{numbers}"
        color = (173, 24, 13)
        layout = "full"
        bar_letters = None

    elif selected_type == "disabled":
        template = "cr-disable.jpg"
        numbers = ''.join(random.choices(string.digits, k=3))
        text = f"D-{numbers}"
        color = (21, 45, 98)
        layout = "disabled"
        bar_letters = None

    else: # official-black
        template = random.choice(["cr-black-flag.jpg", "cr-black-flag-bar.jpg"])
        color = (30, 30, 30)
        if "bar" in template:
            text = ''.join(random.choices(string.digits, k=5))
            layout = "bar"
            bar_letters = ["M", "1"]
        else:
            text = f"{random.randint(10, 99)}-{random.randint(1000, 9999)}"
            layout = "full"
            bar_letters = None

    return template, text, color, layout, bar_letters


def create_cr_plate_from_template():
    """Carga la plantilla real y aplica la maquetación geométrica según la categoría."""
    template_file, text, color_rgb, layout, bar_letters = generate_plate_config()
    template_path = os.path.join(TEMPLATES_DIR, template_file)

    if os.path.exists(template_path):
        img_pil = Image.open(template_path).convert("RGB")
    else:
        # Fallback de seguridad en caso de ruta incorrecta
        img_pil = Image.new('RGB', (1200, 600), color=(240, 240, 240))

    w, h = img_pil.size
    draw = ImageDraw.Draw(img_pil)

    # Definir la zona permitida para el texto principal (evita solapamientos)
    if layout == "disabled":
        # Deja libre el primer 36% izquierdo (símbolo de silla de ruedas)
        main_target = (int(w * 0.36), int(h * 0.18), int(w * 0.96), int(h * 0.85))
    elif layout == "bar":
        # Deja libre el primer 18% izquierdo (franja amarilla)
        main_target = (int(w * 0.18), int(h * 0.18), int(w * 0.96), int(h * 0.85))
        # Dibuja letras en la franja amarilla con biosolid-regular.ttf
        bar_target = (int(w * 0.02), int(h * 0.12), int(w * 0.16), int(h * 0.88))
        draw_vertical_bar_text(draw, bar_letters, bar_target, color_rgb)
    else:
        # Área completa usable (respetando encabezado y pie de página)
        main_target = (int(w * 0.03), int(h * 0.18), int(w * 0.97), int(h * 0.85))

    # Dibujar texto principal escalado y centrado
    draw_text_autofit(draw, text, main_target, FONT_MAIN_PATH, int(h * 0.55), color_rgb)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def parse_ccpd_vertices(filename):
    """Extrae los 4 vértices exactos desde el nombre de archivo CCPD."""
    base_name = os.path.basename(filename).replace(".jpg", "").replace(".png", "")
    parts = base_name.split("-")
    vertices_raw = parts[3].split("_")

    pts = []
    for pt in vertices_raw:
        x, y = map(int, pt.split("&"))
        pts.append([x, y])

    # Reordenar vértices CCPD a estándar OpenCV: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    rb, lb, lt, rt = pts[0], pts[1], pts[2], pts[3]
    return np.float32([lt, rt, rb, lb])


def apply_degradations(plate_img):
    """Aplica blur y variaciones de luz/sombras para simular condiciones reales."""
    if random.random() > 0.4:
        k = random.choice([3, 5])
        plate_img = cv2.GaussianBlur(plate_img, (k, k), 0)

    alpha = random.uniform(0.65, 1.1)
    beta = random.randint(-30, 20)
    return cv2.convertScaleAbs(plate_img, alpha=alpha, beta=beta)


def overlay_plate(car_img, cr_plate, dst_pts):
    """Aplica homografía y sustitución precisa sobre el auto mediante máscaras."""
    h_plate, w_plate = cr_plate.shape[:2]
    src_pts = np.float32([[0, 0], [w_plate, 0], [w_plate, h_plate], [0, h_plate]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    h_car, w_car = car_img.shape[:2]

    warped_plate = cv2.warpPerspective(cr_plate, M, (w_car, h_car))

    mask = np.zeros((h_plate, w_plate), dtype=np.uint8) + 255
    warped_mask = cv2.warpPerspective(mask, M, (w_car, h_car))

    inv_mask = cv2.bitwise_not(warped_mask)
    car_bg = cv2.bitwise_and(car_img, car_img, mask=inv_mask)
    plate_fg = cv2.bitwise_and(warped_plate, warped_plate, mask=warped_mask)

    return cv2.add(car_bg, plate_fg)


def process_dataset():
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    ccpd_files = []
    for ext in extensions:
        ccpd_files.extend(glob.glob(os.path.join(CCPD_DIR, ext)))

    if not ccpd_files:
        print(f"[ERROR CRÍTICO] No hay archivos en: {CCPD_DIR}")
        return

    samples_to_process = min(NUM_SAMPLES, len(ccpd_files))
    print(f"[INFO] Procesando {samples_to_process} imágenes de CCPD...")

    for ccpd_path in ccpd_files[:samples_to_process]:
        try:
            car_img = cv2.imread(ccpd_path)
            dst_pts = parse_ccpd_vertices(ccpd_path)

            cr_plate = create_cr_plate_from_template()
            cr_plate = apply_degradations(cr_plate)

            final_img = overlay_plate(car_img, cr_plate, dst_pts)

            filename = os.path.basename(ccpd_path)
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), final_img)

        except Exception as e:
            print(f"[WARN] Error procesando {ccpd_path}: {e}")

    print(f"[ÉXITO] Dataset sintético generado en '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    process_dataset()