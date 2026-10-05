#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import queue
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from TrtInference import TRTPlates
from TrackerIoU import Tracker
import LPRnetOCR as ocr

MIN_PLATE_W = 30     # px (en la resolucion del pipeline)
MAX_PLATE_W = 250


# --Gstream pipeline para capturar video desde camara o archivo
def gst(source, w, h, codec):
    if source == "camera":
        return ("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1280, "
                "height=(int)720, framerate=(fraction)30/1 ! nvvidconv ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false" % (w, h))
    return ("filesrc location=%s ! qtdemux ! queue ! %sparse ! nvv4l2decoder ! nvvidconv ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink drop=false max-buffers=2 sync=false"
            % (source, codec, w, h))



# -- Hilo de captura para leer frames desde la camara o archivo y ponerlos en una cola
def capture_thread(args, q, stop):
    cap = cv2.VideoCapture(gst(args.source, args.width, args.height, args.codec), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[CAP] GStreamer fallo (si es iPhone prueba --codec h265 o transcodifica a h264)")
        q.put(None)
        return
    is_file = args.source != "camera"
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        if is_file:
            while not stop.is_set():
                try:
                    q.put(frame, timeout=0.5)
                    break
                except queue.Full:
                    pass
        else:
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put(frame)
    cap.release()
    q.put(None)


# -- Worker/Hilo de OCR que recibe recortes de placas y ejecuta la red LPRNet para reconocer el texto
def ocr_worker(args, q, ready, stats):
    dev = torch.device("cuda:0")
    model = ocr.LPRNet(num_classes=len(ocr.CHARS)).to(dev)
    model.load_state_dict(torch.load(ocr.LPRNET_MODEL_PATH, map_location=dev))
    model.eval().half()
    ready.set()
    while True:
        job = q.get()
        if job is None:
            break
        pid, crop = job
        t0 = time.perf_counter()
        with torch.no_grad():
            text = ocr.run_lprnet_inference(crop, model, dev)
        stats["ocr"] += time.perf_counter() - t0
        stats["n"] += 1
        text = text if text else "NADA"
        cv2.imwrite(os.path.join(args.out, "CROP_ID%d_%s.jpg" % (pid, text)), crop)
        print("[OCR] ID %d -> %s" % (pid, text))


# --Main---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="./video_test/highway720.mp4")
    ap.add_argument("--codec", default="h264", choices=["h264", "h265"])
    ap.add_argument("--plates", default="./models/3.0/best_416.engine")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", default="evidencias")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frames, ocr_q, stop, ready = queue.Queue(maxsize=2), queue.Queue(), threading.Event(), threading.Event()
    stats = {"ocr": 0.0, "n": 0}
    threading.Thread(target=capture_thread, args=(args, frames, stop), daemon=True).start()
    threading.Thread(target=ocr_worker, args=(args, ocr_q, ready, stats), daemon=True).start()

    det, trk = TRTPlates(args.plates, conf=args.conf), Tracker()
    ready.wait() # Espera al modelo este cargado

    n, t_start, t_det, t_trk = 0, None, 0.0, 0.0
    try:
        while True:
            frame = frames.get()
            if frame is None:
                break
            n += 1
            t0 = time.perf_counter()  # Inicio medicion 
            dets = det.detect(frame) # Ejecuta deteccion
            t1 = time.perf_counter()
            active, dead = trk.update(dets) # Actualiza tracker

            for pid, box in active:
                x1, y1, x2, y2 = [int(v) for v in box]
                w, h = x2 - x1, y2 - y1 # Convierte Wedth, Height
                tr = trk.t[pid] #Info interta de track
                if tr["sent"] or h > w * 0.7: # Filtro para agarrar bien la geometria rectangular
                    continue
                if MIN_PLATE_W <= w <= MAX_PLATE_W: # Filtro de tamano 
                    if w * h > tr["area"]:
                        tr["area"] = w * h #Actualiza el area maxima del track
                        px, py = int(w * 0.08), int(h * 0.08) # Margen de recorte
                        tr["crop"] = frame[max(0, y1 - py):y2 + py, max(0, x1 - px):x2 + px].copy() #Guarda Copia del recoporte
                elif w > MAX_PLATE_W and tr["crop"] is not None:     # muy cerca: disparar
                    tr["sent"] = True
                    ocr_q.put((pid, tr["crop"]))
            for pid, tr in dead:                                     # track terminado: disparar
                if not tr["sent"] and tr["crop"] is not None:
                    ocr_q.put((pid, tr["crop"]))
            t2 = time.perf_counter()

            if n == 30: #Ignorar los primeros 30 frames para estabilizar FPS
                t_start = t0
            if n > 30:
                t_det += t1 - t0
                t_trk += t2 - t1
                m = n - 30
                if m % 100 == 0:
                    el = time.perf_counter() - t_start
                    print("[FPS %.1f] detect %.1f ms | track+logica %.1f ms | ocr(worker) %.1f ms | cola_ocr %d"
                          % (m / el, 1000 * t_det / m, 1000 * t_trk / m,
                             1000 * stats["ocr"] / max(1, stats["n"]), ocr_q.qsize()))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for pid, tr in trk.t.items():                                # vaciar pendientes
            if not tr["sent"] and tr["crop"] is not None:
                ocr_q.put((pid, tr["crop"]))
        ocr_q.put(None)
        time.sleep(1.0)
        if t_start:
            print("[FIN] %d frames | %.2f FPS promedio" % (n - 30, (n - 30) / (time.perf_counter() - t_start)))


if __name__ == "__main__":
    main()