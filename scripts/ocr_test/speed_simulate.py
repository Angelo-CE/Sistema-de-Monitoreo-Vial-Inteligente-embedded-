import cv2
import numpy as np

INPUT_VIDEO = "./videos_test/IMG_0483.MOV"
OUTPUT_VIDEO = "./videos_test/highway941.mp4"

# Parámetros de simulación
SPEED_MULTIPLIER = 4       # Salta 3 de cada 4 frames
BLUR_INTENSITY = 9         # Longitud de la estela del motion blur (impar: 5, 7, 9, 11)

cap = cv2.VideoCapture(INPUT_VIDEO)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# El FPS del output se mantiene, pero el auto cruzará 4 veces más rápido
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# Kernel de Motion Blur Horizontal (Simula el auto moviéndose de lado a lado)
kernel_motion = np.zeros((BLUR_INTENSITY, BLUR_INTENSITY))
kernel_motion[int((BLUR_INTENSITY - 1) / 2), :] = np.ones(BLUR_INTENSITY)
kernel_motion /= BLUR_INTENSITY

frame_count = 0

print(f"Simulando velocidad {SPEED_MULTIPLIER}x con Motion Blur nivel {BLUR_INTENSITY}...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Simular la aceleración leyendo solo 1 de cada N frames
    if frame_count % SPEED_MULTIPLIER == 0:
        # Aplicar el defecto físico del sensor (Motion Blur)
        blurred_frame = cv2.filter2D(frame, -1, kernel_motion)
        out.write(blurred_frame)

    frame_count += 1

cap.release()
out.release()
print(f"Video simulado guardado en: {OUTPUT_VIDEO}")