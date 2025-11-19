# Guía del Proyecto Final  
**Detección de objetos con YOLO y análisis batch en Hive**

## 1. Descripción general

El proyecto implementa un *pipeline* completo de analítica batch sobre datos de visión por computadora.  
A partir de imágenes y videos se ejecuta YOLOv8 para detectar objetos y personas, se generan archivos CSV con atributos enriquecidos y luego se corre un proceso ETL que limpia y agrupa las detecciones, las sube a HDFS y las deja disponibles en una tabla externa de Hive para ejecutar consultas analíticas con MapReduce.

---

## 2. Objetivo del proyecto

- Demostrar el uso de **IA** (YOLOv8) para extraer información estructurada desde imágenes y videos.  
- Construir un **sistema batch/ETL** que procese los resultados, evite duplicados y cargue los datos en **Hive sobre HDFS**.  
- Ejecutar **consultas analíticas** en Hive (por clase, por video, por ventanas de tiempo, por color, etc.) y validar que el flujo completo funciona de punta a punta.

---

## 3. Arquitectura y organización del repositorio

Estructura principal:

- `Makefile`  
- `requirements.txt`  
- `yolov8n.pt` (modelo YOLOv8 pre-entrenado)  
- `README.md` (instrucciones de ejecución para el usuario)

- `src/`  
  - `sistema_clasificacion.py`  
  - `sistema_batch_etl.py`

- `data/`  
  - `raw/`
    - `images/`  (imágenes de entrada)
    - `videos/`  (videos de entrada)
  - `staging/detecciones/`
    - `etl_checkpoints/processed_files.txt`  
    - `detecciones_*.csv` (salida del sistema de clasificación)

- `scripts_hive/`  
  - `create_tables.sql` (crea la BD y la tabla externa en Hive)  
  - `consultas_analiticas.sql` (conjunto de consultas analíticas)

- `tests/`  
  - `test_etl.py` (pruebas básicas del proceso ETL)

- `.venv/` (entorno virtual de Python, creado localmente)  
- `.pytest_cache/`, `__pycache__/` (archivos generados por Python y pytest)

---

## 4. Componentes principales

### 4.1 Sistema de clasificación – `src/sistema_clasificacion.py`

Responsable de la **parte de IA**:

1. Lee imágenes y videos desde:
   - `data/raw/images/`
   - `data/raw/videos/`
2. Carga el modelo **YOLOv8n** (`yolov8n.pt`) usando la librería `ultralytics`.
3. Para cada imagen / frame de video:
   - Detecta objetos y obtiene:  
     `class_id`, `class_name`, `confidence`, `x_min`, `y_min`, `x_max`, `y_max`, etc.
   - Calcula atributos derivados:
     - `width`, `height`, `area_pixels`
     - `frame_width`, `frame_height`, `bbox_area_ratio`
     - `center_x`, `center_y`, posiciones normalizadas
     - `position_region` (zona de la imagen donde cae el objeto)
     - `dominant_color_name` y canales `dom_r`, `dom_g`, `dom_b`
     - `timestamp_sec` (segundos dentro del video)
   - Genera un identificador único `detection_id` por detección.
4. Escribe todas las detecciones en **CSV** dentro de `data/staging/detecciones/`, con nombres tipo:
   - `detecciones_YYYYMMDD_HHMMSS.csv`
   - `detecciones_imagenes_YYYYMMDD_HHMMSS.csv`

Estos CSV son la **fuente de datos** para el ETL.


### 4.2 Sistema Batch / ETL – `src/sistema_batch_etl.py`

Implementa el flujo **CSV → HDFS/Hive**:

1. Asegura la existencia de:
   - `data/processed/`
   - `data/staging/detecciones/etl_checkpoints/`
2. Lista todos los CSV de `data/staging/detecciones/`.
3. Lee el archivo de checkpoint `processed_files.txt` para saber qué CSV ya fueron procesados.
4. Para cada CSV **no procesado**:
   - Carga el DataFrame con `pandas`.
   - Llama a `limpiar_df(df)` que:
     - Verifica que existan todas las columnas esperadas.
     - Elimina filas con datos nulos en campos clave.
     - Filtra `confidence` fuera de `[0, 1]`.
     - Elimina *bounding boxes* inválidos.
     - Elimina duplicados por `detection_id`.
   - Separa por tipo de fuente:
     - **Imágenes (`source_type = 'image'`)**  
       → genera un único CSV de lote y lo guarda en `data/processed/`.
     - **Videos (`source_type = 'video'`)**  
       → crea una columna `window_10s` agrupando por ventanas de **10 segundos**.  
       → para cada (video, ventana) genera un CSV con solo esos registros.
   - Cada CSV resultante se sube a HDFS ejecutando:
     - `hdfs dfs -put -f <archivo_local> <HDFS_BASE_DIR>`
   - El path del CSV original se agrega a `processed_files.txt` para evitar reprocesarlo en futuras corridas.

El directorio HDFS usado es (configurable por variable de entorno):

- `HDFS_BASE_DIR = /projects/yolo_objects/hive/csv`

---

### 4.3 Capa de Hive – `scripts_hive/`

**`create_tables.sql`**

- Crea la base de datos:

```sql
CREATE DATABASE IF NOT EXISTS proyecto_yolo;
USE proyecto_yolo;
```

- Crea la tabla externa `yolo_objects` apuntando a `hdfs:///projects/yolo_objects/hive/csv` con todas las columnas generadas en el ETL, usando `OpenCSVSerde` para leer los CSV.

**`consultas_analiticas.sql`**

Incluye varias consultas ejemplo:

1. Conteo de objetos por clase (`class_name`).  
2. Número de detecciones de personas por `source_id` (nota: cuenta detecciones por frame, **no** personas únicas).  
3. Área promedio de *bounding box* por clase.  
4. Distribución de colores dominantes (`dominant_color_name`) por clase.  
5. Número de objetos por ventana de 10 segundos en cada video.

Al ejecutar este script desde `beeline` se lanzan *jobs* de MapReduce y se obtienen tablas de resultados como evidencia del análisis.

---

### 4.4 Makefile

Facilita la ejecución desde terminal:

- `make venv` – Crea el entorno virtual `.venv/`.  
- `make install` – Instala dependencias de `requirements.txt`.  
- `make clasificar` – Ejecuta `src/sistema_clasificacion.py`.  
- `make etl` – Ejecuta `src/sistema_batch_etl.py`.  
- `make hive-tables` – Llama a `beeline` para ejecutar `scripts_hive/create_tables.sql`.  
- (Opcionales) `make lint`, `make test` si se desea usar `pylint` y `pytest`.

---

### 4.5 Pruebas – `tests/test_etl.py`

Se incluye un archivo de pruebas básico para el módulo ETL, donde se valida que:

- El script puede leer un CSV de ejemplo de detecciones.  
- La etapa de limpieza elimina registros inválidos y duplicados.  
- El proceso genera un lote de salida coherente (sin datos nulos en campos clave).

Estas pruebas se pueden lanzar con:

```bash
make test
```

---

## 5. Flujo de ejecución del proyecto

### 5.1 Preparación del entorno

```bash
cd PROYECTO_YOLO_HIVE
python3 -m venv .venv
source .venv/bin/activate
make install
```

(Instala `ultralytics`, `opencv-python`, `pandas`, `pytest`, etc.)

### 5.2 Carga de datos de entrada

- Copiar imágenes a `data/raw/images/`.  
- Copiar videos a `data/raw/videos/`.

### 5.3 Sistema de clasificación (YOLO → CSV)

```bash
make clasificar
```

Salida:

- CSV de detecciones (`detecciones_*.csv`) en `data/staging/detecciones/`.

### 5.4 Proceso Batch / ETL (CSV → HDFS)

Antes, asegurarse de que **HDFS y YARN** estén levantados.

```bash
make etl
```

Salida:

- CSV procesados en `data/processed/`.  
- Los mismos CSV subidos a `hdfs:///projects/yolo_objects/hive/csv`.  
- Archivo `data/staging/detecciones/etl_checkpoints/processed_files.txt` actualizado.

Si se vuelve a ejecutar, los archivos ya listados en `processed_files.txt` se **saltan** para evitar duplicados en Hive.

### 5.5 Creación de tablas en Hive (primer uso)

```bash
make hive-tables
```

o desde `beeline`:

```sql
!run /ruta/al/proyecto/scripts_hive/create_tables.sql
```

### 5.6 Consultas analíticas en Hive

Desde `beeline`, con HiveServer2 en el puerto 10000:

```sql
!run /ruta/al/proyecto/scripts_hive/consultas_analiticas.sql
```

Las salidas muestran, por ejemplo:

- Clases más detectadas (`person`, `airplane`, etc.).  
- Número de detecciones de `person` por video o imagen.  
- Área promedio de los *bounding boxes* por clase.  
- Distribución de colores dominantes.  
- Cantidad de objetos por ventana de 10 segundos en cada video.


---

## 6. Conclusiones y trabajo futuro

- Se logró integrar un **modelo de visión por computadora (YOLOv8)** con un **pipeline de datos batch** sobre **Hadoop + Hive**, ejecutando consultas analíticas.  
- El uso de **tablas externas** y **checkpoints** permite recargar solo nuevos CSV sin duplicar datos.  
- Como trabajo futuro se podría:
  - Migrar la ejecución de Hive a **Tez o Spark** para mejorar tiempos.  
  - Calcular métricas más avanzadas (seguimiento de objetos, “personas únicas” por video, etc.).  
  - Conectar Hive con una capa de visualización (por ejemplo, dashboards en Power BI o Superset).
