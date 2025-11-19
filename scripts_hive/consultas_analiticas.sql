USE proyecto_yolo;

-- 1) Conteo de objetos por clase
SELECT
  class_name,
  COUNT(*) AS total_objetos
FROM yolo_objects
WHERE class_name <> 'class_name'
GROUP BY class_name
ORDER BY total_objetos DESC;


-- 2A) Total de DETECCIONES de personas por video
-- (cuántas personas detectadas por frame)
SELECT
  source_id,
  COUNT(*) AS total_detecciones_persona
FROM yolo_objects
WHERE class_name = 'person'
GROUP BY source_id
ORDER BY total_detecciones_persona DESC;


-- 2B) Máximo de personas simultáneas en un frame por video
SELECT
  source_id,
  MAX(personas_en_frame) AS max_personas_simultaneas
FROM (
  SELECT
    source_id,
    frame_number,
    COUNT(*) AS personas_en_frame
  FROM yolo_objects
  WHERE class_name = 'person'
  GROUP BY source_id, frame_number
) t
GROUP BY source_id
ORDER BY max_personas_simultaneas DESC;


-- 3) Área promedio de bounding box por clase
SELECT
  class_name,
  AVG(area_pixels)     AS area_media_pixels,
  AVG(bbox_area_ratio) AS ratio_area_medio
FROM yolo_objects
WHERE class_name <> 'class_name'
GROUP BY class_name
ORDER BY area_media_pixels DESC;


-- 4) Distribución de colores dominantes por clase
SELECT
  class_name,
  dominant_color_name,
  COUNT(*) AS total
FROM yolo_objects
WHERE class_name <> 'class_name'
GROUP BY class_name, dominant_color_name
ORDER BY total DESC;


-- 5) Objetos por ventana de 10 segundos en cada video
SELECT
  source_id,
  FLOOR(timestamp_sec / 10) * 10 AS window_start_sec,
  COUNT(*)                        AS objetos_en_ventana
FROM yolo_objects
WHERE source_type = 'video'
GROUP BY source_id, FLOOR(timestamp_sec / 10) * 10
ORDER BY source_id, window_start_sec;
