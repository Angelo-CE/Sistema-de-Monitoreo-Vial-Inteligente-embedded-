import ctypes
import multiprocessing as mp
import numpy as np
import cv2
import time
import json

# ====================================================================
# CONFIGURACIÓN GENERAL
# ====================================================================
MODE = "JETSON"
SOURCE = "highway9.mp4"
ENGINE_PATH = "best_fp16.engine"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CHANNELS = 3
BUFFER_SIZE = 8

MODEL_INPUT_WIDTH = 480
MODEL_INPUT_HEIGHT = 480

# Umbrales drásticamente bajos para forzar detecciones y auditar el modelo
CONF_THRESHOLD = 0.25 
IOU_THRESHOLD = 0.45


def postprocess_yolov8(raw_output, conf_thresh, iou_thresh):
    """Parsea la salida de YOLOv8 escalando de 480x480 a la resolución original."""
    try:
        predictions = raw_output.reshape(5, -1).T
    except ValueError:
        return []

    boxes = []
    scores = []

    # Calcular la proporción matemática correcta
    scale_x = FRAME_WIDTH / MODEL_INPUT_WIDTH
    scale_y = FRAME_HEIGHT / MODEL_INPUT_HEIGHT

    for pred in predictions:
        score = pred[4]
        if score > conf_thresh:
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            
            # Escalar coordenadas a la dimensión del video
            x = int((cx - w / 2) * scale_x)
            y = int((cy - h / 2) * scale_y)
            bw = int(w * scale_x)
            bh = int(h * scale_y)

            boxes.append([x, y, bw, bh])
            scores.append(float(score))

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            
            # Limitar las cajas para que no se salgan del cuadro
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(FRAME_WIDTH, x + bw)
            y2 = min(FRAME_HEIGHT, y + bh)
            
            results.append({
                "bbox": [x1, y1, x2, y2],
                "score": scores[i],
                "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                "left": x1,
                "right": x2
            })

    return results



def get_video_reader(source, mode="JETSON"):
    gst_pipeline = (
        f"filesrc location={source} ! qtdemux ! h264parse ! "
        "nvv4l2decoder ! nvvidconv ! "
        f"video/x-raw, width=(int){FRAME_WIDTH}, height=(int){FRAME_HEIGHT}, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=false sync=false"
    )
    return cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)


def reader_process(source, mode, raw_array, free_queue, ai_queue, stop_event):
    shared_array = np.frombuffer(raw_array, dtype=np.uint8).reshape(
        (BUFFER_SIZE, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)
    )
    cap = get_video_reader(source, mode=mode)
    
    if not cap.isOpened():
        print("[READER ERROR] No se pudo abrir el video.")
        ai_queue.put(None)
        return

    frame_count = 0
    while not stop_event.is_set():
        try:
            index = free_queue.get(timeout=1.0)
        except mp.queues.Empty:
            continue

        ret, frame = cap.read()
        if not ret:
            free_queue.put(index)
            break

        if frame.shape[0] != FRAME_HEIGHT or frame.shape[1] != FRAME_WIDTH:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        shared_array[index] = frame
        frame_count += 1
        ai_queue.put((frame_count, index))

    ai_queue.put(None)
    cap.release()


def ai_process(raw_array, ai_queue, render_queue, stop_event):
    import tensorrt as trt
    import pycuda.driver as cuda

    cuda.init()
    dev = cuda.Device(0)
    cuda_ctx = dev.make_context()

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})

    shared_array = np.frombuffer(raw_array, dtype=np.uint8).reshape(
        (BUFFER_SIZE, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)
    )

    while not stop_event.is_set():
        try:
            data = ai_queue.get(timeout=1.0)
        except mp.queues.Empty:
            continue

        if data is None:
            break

        frame_count, index = data
        frame = shared_array[index]

        img = cv2.resize(frame, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img)

        np.copyto(inputs[0]["host"], img.ravel())
        cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
        stream.synchronize()

        detections = postprocess_yolov8(outputs[0]["host"], CONF_THRESHOLD, IOU_THRESHOLD)
        render_queue.put((frame_count, index, detections))

    cuda_ctx.pop()
    render_queue.put(None)


def render_process(raw_array, render_queue, free_queue, stop_event):
    shared_array = np.frombuffer(raw_array, dtype=np.uint8).reshape(
        (BUFFER_SIZE, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)
    )

    # Configuración de VideoWriter - Codificación estándar MP4V (usará CPU pero es seguro)
    writer = cv2.VideoWriter("output_placas.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (FRAME_WIDTH, FRAME_HEIGHT))
    
    json_data = []
    processed_count = 0
    start_time = time.time()

    print("[RENDER] Escribiendo video a 'output_placas.mp4' en disco. Cero GUI.")

    while not stop_event.is_set():
        try:
            data = render_queue.get(timeout=1.0)
        except mp.queues.Empty:
            continue

        if data is None:
            break

        frame_count, index, detections = data
        frame = shared_array[index].copy()  # Copia para no sobrescribir el buffer antes de liberar

        frame_info = {"frame_id": frame_count, "plates": []}

        #if detections:
           # print(f"[DEBUG] Frame {frame_count} - Encontradas {len(detections)} placas.")
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            score = det["score"]
            cx, cy = det["center"]
            left = det["left"]
            right = det["right"]

            # Datos para JSON
            frame_info["plates"].append({
                "center_pixel": [cx, cy],
                "left": left,
                "right": right,
                "top": y1,
                "bottom": y2,
                "confidence": round(score, 3)
            })

            # Dibujar en el video
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"P: {score:.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1) # Punto central

        # Escribir frame a disco inmediatamente
        writer.write(frame)
        if frame_info["plates"]:
            json_data.append(frame_info)

        processed_count += 1
        free_queue.put(index)

    # Limpieza final
    writer.release()
    
    # Exportar JSON
    with open("placas_data.json", "w") as f:
        json.dump(json_data, f, indent=4)

    total_time = time.time() - start_time
    fps_real = processed_count / total_time if total_time > 0 else 0
    
    print(f"\n[ÉXITO] Video guardado como 'output_placas.mp4'")
    print(f"[ÉXITO] Datos espaciales guardados en 'placas_data.json'")
    print(f"[METRICAS] Cuadros procesados: {processed_count} | Rendimiento de Pipeline: {fps_real:.2f} FPS")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    raw_array = mp.RawArray(ctypes.c_uint8, BUFFER_SIZE * FRAME_WIDTH * FRAME_HEIGHT * CHANNELS)
    free_queue, ai_queue, render_queue = mp.Queue(), mp.Queue(), mp.Queue()
    stop_event = mp.Event()

    for i in range(BUFFER_SIZE):
        free_queue.put(i)

    p_reader = mp.Process(target=reader_process, args=(SOURCE, MODE, raw_array, free_queue, ai_queue, stop_event))
    p_ai = mp.Process(target=ai_process, args=(raw_array, ai_queue, render_queue, stop_event))
    p_render = mp.Process(target=render_process, args=(raw_array, render_queue, free_queue, stop_event))

    p_reader.start()
    p_ai.start()
    p_render.start()

    p_reader.join()
    p_ai.join()
    p_render.join()