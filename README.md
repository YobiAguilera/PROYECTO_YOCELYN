# Proyecto Final – YOLO + Hive (Procesos ETL para IA)

## 1. Arquitectura

El proyecto tiene dos sistemas separados:

1. **Sistema de Clasificación (`src/sistema_clasificacion.py`)**
   - Carga un modelo YOLO pre-entrenado.
   - Procesa imágenes y videos de `data/raw/`.
   - Extrae atributos enriquecidos (Sección 6 del enunciado).
   - Escribe todas las detecciones en CSV en `data/staging/detecciones/`.

2. **Sistema Batch / ETL (`src/sistema_batch_etl.py`)**
   - Lee los CSV de `data/staging/detecciones/`.
   - Limpia, transforma y normaliza los datos.
   - Agrupa las detecciones:
     - Imágenes: un solo lote.
     - Videos: lotes por ventana de 10 segundos.
   - Elimina duplicados usando `detection_id` y un archivo de checkpoint.
   - Escribe CSV en `data/processed/etl_batches/` y los sube a HDFS:
     `/projects/yolo_objects/hive/csv`.
   - La tabla externa de Hive (`proyecto_yolo.yolo_objects`) apunta a este directorio.

## 2. Prerrequisitos

- Ubuntu 24.04
- Python 3.10
- Hadoop + HDFS
- Apache Hive
- GPU NVIDIA (opcional)

## 3. Instalación

```bash
git clone <repo>
cd proyecto_yolo_hive
make install
