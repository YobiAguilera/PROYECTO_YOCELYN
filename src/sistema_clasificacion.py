"""
Sistema de Clasificación (YOLO -> CSV)
--------------------------------------

- Carga un modelo YOLO preentrenado.
- Procesa imágenes y videos desde:
    - data/raw/images/
    - data/raw/videos/
- Extrae todos los atributos requeridos por el enunciado.
- Escribe un CSV de detecciones en:
    - data/staging/detecciones/detecciones_YYYYMMDD_HHMMSS.csv

Este sistema **NO** se conecta a Hive ni a HDFS.
Su única responsabilidad es clasificar y generar CSVs de staging.
"""

import os
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

#rutas base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
RAW_VIDEOS_DIR = os.path.join(BASE_DIR, "data", "raw", "videos")
STAGING_DIR = os.path.join(BASE_DIR, "data", "staging", "detecciones")

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", os.path.join(BASE_DIR, "yolov8n.pt"))


#--------------------------------------------------------------------
#helpers
#---------------------------------------------------------------------
def asegurar_directorios() -> None:
    """Crea el directorio de staging si no existe."""
    os.makedirs(STAGING_DIR, exist_ok=True)


def _position_region(center_x_norm: float, center_y_norm: float) -> str:
    """
    Devuelve la región de posición:
    top/middle/bottom  x  left/center/right
    """
    if center_y_norm < 1 / 3:
        v = "top"
    elif center_y_norm < 2 / 3:
        v = "middle"
    else:
        v = "bottom"

    if center_x_norm < 1 / 3:
        h = "left"
    elif center_x_norm < 2 / 3:
        h = "center"
    else:
        h = "right"

    return f"{v}-{h}"


def _dominant_color_name_and_rgb(roi_bgr: np.ndarray) -> (str, int, int, int):
    """
    Calcula el color dominante promedio del ROI (en BGR) y lo mapea a un nombre simple.
    """
    if roi_bgr.size == 0:
        return "unknown", 0, 0, 0

    #promedio de color
    mean_bgr = roi_bgr.mean(axis=(0, 1))
    b, g, r = [int(x) for x in mean_bgr]
    #convertir a RGB para guardarlo
    dom_r, dom_g, dom_b = r, g, b

    #nombre de color
    max_c = max(dom_r, dom_g, dom_b)
    min_c = min(dom_r, dom_g, dom_b)

    if max_c < 40:
        name = "black"
    elif min_c > 215:
        name = "white"
    elif max_c - min_c < 25:
        name = "gray"
    elif dom_r == max_c and dom_g < 100 and dom_b < 100:
        name = "red"
    elif dom_g == max_c and dom_r < 100 and dom_b < 100:
        name = "green"
    elif dom_b == max_c and dom_r < 100 and dom_g < 100:
        name = "blue"
    elif dom_r == max_c and dom_g == max_c:
        name = "yellow"
    else:
        name = "other"

    return name, dom_r, dom_g, dom_b


def _build_detection(
    source_type: str,
    source_id: str,
    frame_number: int,
    frame_width: int,
    frame_height: int,
    cls_id: int,
    cls_name: str,
    conf: float,
    xyxy: np.ndarray,
    frame_bgr: np.ndarray,
    timestamp_sec: float,
    local_object_id: int,
    ingestion_date: str,
) -> Dict:
    """
    Construye el diccionario con todos los atributos de una detección.
    """
    x_min, y_min, x_max, y_max = [int(v) for v in xyxy]

    width = max(x_max - x_min, 1)
    height = max(y_max - y_min, 1)
    area_pixels = width * height

    center_x = x_min + width / 2.0
    center_y = y_min + height / 2.0

    frame_area = max(frame_width * frame_height, 1)
    bbox_area_ratio = area_pixels / frame_area

    center_x_norm = center_x / frame_width
    center_y_norm = center_y / frame_height

    position_region = _position_region(center_x_norm, center_y_norm)

    #color dominante
    x_min_clip = max(x_min, 0)
    y_min_clip = max(y_min, 0)
    x_max_clip = min(x_max, frame_width)
    y_max_clip = min(y_max, frame_height)

    roi = frame_bgr[y_min_clip:y_max_clip, x_min_clip:x_max_clip]
    dominant_color_name, dom_r, dom_g, dom_b = _dominant_color_name_and_rgb(roi)

    detection_id = f"{source_id}_{frame_number}_{local_object_id}"

    return {
        "source_type": source_type,
        "source_id": source_id,
        "frame_number": frame_number,
        "class_id": cls_id,
        "class_name": cls_name,
        "confidence": float(conf),
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "width": width,
        "height": height,
        "area_pixels": area_pixels,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "bbox_area_ratio": bbox_area_ratio,
        "center_x": center_x,
        "center_y": center_y,
        "center_x_norm": center_x_norm,
        "center_y_norm": center_y_norm,
        "position_region": position_region,
        "dominant_color_name": dominant_color_name,
        "dom_r": dom_r,
        "dom_g": dom_g,
        "dom_b": dom_b,
        "timestamp_sec": float(timestamp_sec),
        "ingestion_date": ingestion_date,
        "detection_id": detection_id,
    }


# ---------------------------------------------------------------------
#procesamiento de imagenes y videos
# ---------------------------------------------------------------------
def procesar_imagenes(model: YOLO) -> List[Dict]:
    """Procesa todas las imágenes en RAW_IMAGES_DIR y devuelve una lista de detecciones."""
    detections: List[Dict] = []
    ingestion_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for fname in sorted(os.listdir(RAW_IMAGES_DIR)):
        path = os.path.join(RAW_IMAGES_DIR, fname)
        if not os.path.isfile(path):
            continue
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] No se pudo leer imagen {path}")
            continue

        frame_height, frame_width = img.shape[:2]
        source_id = fname
        source_type = "image"
        frame_number = 0 
        timestamp_sec = 0.0

        results = model(img)[0] 
        boxes = results.boxes

        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            cls_name = model.names.get(cls_id, f"class_{cls_id}")

            det = _build_detection(
                source_type=source_type,
                source_id=source_id,
                frame_number=frame_number,
                frame_width=frame_width,
                frame_height=frame_height,
                cls_id=cls_id,
                cls_name=cls_name,
                conf=conf,
                xyxy=xyxy,
                frame_bgr=img,
                timestamp_sec=timestamp_sec,
                local_object_id=idx,
                ingestion_date=ingestion_date,
            )
            detections.append(det)

    return detections


def procesar_videos(model: YOLO) -> List[Dict]:
    """Procesa todos los videos en RAW_VIDEOS_DIR y devuelve una lista de detecciones."""
    detections: List[Dict] = []
    ingestion_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for fname in sorted(os.listdir(RAW_VIDEOS_DIR)):
        path = os.path.join(RAW_VIDEOS_DIR, fname)
        if not os.path.isfile(path):
            continue
        if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[WARN] No se pudo abrir video {path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 25.0

        source_id = fname
        source_type = "video"
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_height, frame_width = frame.shape[:2]
            timestamp_sec = frame_number / fps

            results = model(frame)[0]
            boxes = results.boxes

            for idx, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")

                det = _build_detection(
                    source_type=source_type,
                    source_id=source_id,
                    frame_number=frame_number,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    cls_id=cls_id,
                    cls_name=cls_name,
                    conf=conf,
                    xyxy=xyxy,
                    frame_bgr=frame,
                    timestamp_sec=timestamp_sec,
                    local_object_id=idx,
                    ingestion_date=ingestion_date,
                )
                detections.append(det)

            frame_number += 1

        cap.release()

    return detections


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main() -> None:
    print("[INFO] Iniciando sistema de clasificación (YOLO -> CSV)")
    asegurar_directorios()

    print(f"[INFO] Cargando modelo YOLO desde {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    all_detections: List[Dict] = []

    print("[INFO] Procesando imágenes...")
    all_detections.extend(procesar_imagenes(model))

    print("[INFO] Procesando videos...")
    all_detections.extend(procesar_videos(model))

    if not all_detections:
        print("[WARN] No se generaron detecciones.")
        return

    df = pd.DataFrame(all_detections)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(STAGING_DIR, f"detecciones_{ts}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[INFO] CSV de detecciones generado en: {out_path}")
    print(f"[INFO] Total de detecciones: {len(df)}")


if __name__ == "__main__":
    main()
