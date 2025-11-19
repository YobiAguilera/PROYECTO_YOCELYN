"""
Sistema Batch / ETL
-------------------

Lee los CSV generados por el sistema de clasificación en data/staging/detecciones/,
realiza limpieza y transformaciones básicas, agrupa las detecciones en lotes y las
carga a Hive a través de HDFS SIN DUPLICADOS.

- Imágenes: se envían en un solo lote.
- Videos: se agrupan en ventanas de 10 segundos (por video).

Estrategia contra duplicados:
- Cada CSV procesado se registra en un archivo de checkpoint:
  data/staging/detecciones/etl_checkpoints/processed_files.txt
- Si vuelves a ejecutar el ETL, los CSV ya registrados se saltan.
"""

import glob
import os
import subprocess
from datetime import datetime
from typing import List, Set

import pandas as pd

#rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING_DIR = os.path.join(BASE_DIR, "data", "staging", "detecciones")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINT_DIR = os.path.join(STAGING_DIR, "etl_checkpoints")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "processed_files.txt")


HDFS_BASE_DIR = os.getenv("HDFS_BASE_DIR", "/projects/yolo_objects/hive/csv")


def asegurar_directorios_locales() -> None:
    """Crea carpetas locales necesarias para ETL."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def listar_csv_staging() -> List[str]:
    """Lista todos los CSV en data/staging/detecciones/."""
    pattern = os.path.join(STAGING_DIR, "*.csv")
    return sorted(glob.glob(pattern))


def leer_checkpoint() -> Set[str]:
    """Lee el listado de CSV ya procesados (para no duplicar cargas)."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def escribir_checkpoint(procesados: Set[str]) -> None:
    """Guarda el listado de CSV procesados en disco."""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        for path in sorted(procesados):
            f.write(path + "\n")


def limpiar_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza básica:
    - Verifica columnas obligatorias.
    - Elimina nulos en campos clave.
    - Filtra confidences fuera de [0, 1].
    - Elimina bounding boxes inválidos.
    - Elimina duplicados por detection_id.
    """
    columnas_obligatorias = [
        "source_type",
        "source_id",
        "frame_number",
        "class_id",
        "class_name",
        "confidence",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "width",
        "height",
        "area_pixels",
        "frame_width",
        "frame_height",
        "bbox_area_ratio",
        "center_x",
        "center_y",
        "center_x_norm",
        "center_y_norm",
        "position_region",
        "dominant_color_name",
        "dom_r",
        "dom_g",
        "dom_b",
        "timestamp_sec",
        "ingestion_date",
        "detection_id",
    ]

    faltantes = set(columnas_obligatorias) - set(df.columns)
    if faltantes:
        raise ValueError(f"CSV no tiene todas las columnas esperadas, faltan: {faltantes}")

    #eliminar filas con nulos en campos clave
    df = df.dropna(
        subset=[
            "source_type",
            "source_id",
            "class_id",
            "class_name",
            "confidence",
            "detection_id",
        ]
    )

    #asegurar tipos básicos
    df["confidence"] = df["confidence"].astype(float)
    df["frame_number"] = df["frame_number"].astype(int)
    df["timestamp_sec"] = df["timestamp_sec"].astype(float)

    #filtrar confidencias fuera de rango 0, 1
    df = df[(df["confidence"] >= 0.0) & (df["confidence"] <= 1.0)]

    #counding boxes validos
    df = df[
        (df["x_min"] >= 0)
        & (df["y_min"] >= 0)
        & (df["x_max"] > df["x_min"])
        & (df["y_max"] > df["y_min"])
    ]

    #eliminar duplicados por detection_id dentro del CSV
    df = df.drop_duplicates(subset=["detection_id"])

    return df


def subir_a_hdfs(local_path: str, hdfs_dir: str) -> None:
    """
    Copia un CSV local a HDFS usando 'hdfs dfs -put -f'.
    No hace LOAD DATA porque la tabla es EXTERNAL: al estar el archivo en LOCATION,
    Hive lo ve automáticamente.
    """
    print(f"[INFO] Subiendo a HDFS: {local_path} -> {hdfs_dir}")
    subprocess.run(
        ["hdfs", "dfs", "-put", "-f", local_path, hdfs_dir],
        check=True,
    )


def procesar_csv(csv_path: str) -> None:
    """
    Procesa un archivo de detecciones:
    - Lee el CSV.
    - Limpia.
    - Separa imágenes y videos.
    - Imágenes: un solo CSV.
    - Videos: un CSV por ventana de 10 segundos (por video).
    - Sube todos los CSV resultantes a HDFS.
    """
    print(f"[INFO] Procesando CSV de staging: {csv_path}")
    df = pd.read_csv(csv_path)

    df = limpiar_df(df)
    if df.empty:
        print("[WARN] Después de limpiar no quedan filas, se omite.")
        return

    ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")

    #imagenes en lote
    df_img = df[df["source_type"] == "image"]
    if not df_img.empty:
        local_img = os.path.join(PROCESSED_DIR, f"imagenes_lote_{ts_run}.csv")
        df_img.to_csv(local_img, index=False)
        subir_a_hdfs(local_img, HDFS_BASE_DIR)
        print(f"[INFO]   Imágenes: {len(df_img)} filas cargadas.")

    #videos: ventanas de 10 segundos
    df_vid = df[df["source_type"] == "video"]
    if not df_vid.empty:
        df_vid = df_vid.copy()
        df_vid["window_10s"] = (df_vid["timestamp_sec"] // 10).astype(int)

        for (source_id, window_10s), grupo in df_vid.groupby(["source_id", "window_10s"]):
            if grupo.empty:
                continue
            safe_source = os.path.splitext(os.path.basename(source_id))[0]
            local_vid = os.path.join(
                PROCESSED_DIR, f"video_{safe_source}_win{int(window_10s)}_{ts_run}.csv"
            )
            grupo.to_csv(local_vid, index=False)
            subir_a_hdfs(local_vid, HDFS_BASE_DIR)
            print(
                f"[INFO]   Video {source_id}, ventana {window_10s*10}-{(window_10s+1)*10}s: "
                f"{len(grupo)} filas cargadas."
            )


def main() -> None:
    """Punto de entrada del sistema batch/ETL."""
    print("[INFO] Iniciando sistema Batch / ETL (CSV -> Hive)")
    asegurar_directorios_locales()

    todos_csv = listar_csv_staging()
    if not todos_csv:
        print("[INFO] No hay CSV en data/staging/detecciones, nada que hacer.")
        return

    procesados = leer_checkpoint()
    print(f"[INFO] CSV ya procesados (checkpoint): {len(procesados)}")

    for csv_path in todos_csv:
        if csv_path in procesados:
            print(f"[INFO] Saltando CSV ya procesado: {csv_path}")
            continue

        procesar_csv(csv_path)
        procesados.add(csv_path)
        escribir_checkpoint(procesados)

    print("[INFO] ETL completado sin duplicados (por checkpoint).")


if __name__ == "__main__":
    main()
