-- 1)Crear base de datos (si no existe)
CREATE DATABASE IF NOT EXISTS proyecto_yolo;

-- 2)Usar esa base
USE proyecto_yolo;

-- 3)Borrar la tabla si ya existiera
DROP TABLE IF EXISTS yolo_objects;

-- 4)Crear tabla EXTERNA que apunta a CSV en HDFS
CREATE EXTERNAL TABLE yolo_objects (
  source_type           STRING,
  source_id             STRING,
  frame_number          INT,
  class_id              INT,
  class_name            STRING,
  confidence            DOUBLE,
  x_min                 INT,
  y_min                 INT,
  x_max                 INT,
  y_max                 INT,
  width                 INT,
  height                INT,
  area_pixels           INT,
  frame_width           INT,
  frame_height          INT,
  bbox_area_ratio       DOUBLE,
  center_x              DOUBLE,
  center_y              DOUBLE,
  center_x_norm         DOUBLE,
  center_y_norm         DOUBLE,
  position_region       STRING,
  dominant_color_name   STRING,
  dom_r                 INT,
  dom_g                 INT,
  dom_b                 INT,
  timestamp_sec         DOUBLE,
  ingestion_date        STRING,
  detection_id          STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  "separatorChar" = ",",
  "quoteChar"     = "\"",
  "escapeChar"    = "\\"
)
STORED AS TEXTFILE
LOCATION 'hdfs:///projects/yolo_objects/hive/csv';
