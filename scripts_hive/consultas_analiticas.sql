USE proyecto_yolo;

-- 1)Conteo de objetos por clase
SELECT
  class_name,
  COUNT(*) AS total_objetos
FROM yolo_objects
GROUP BY class_name
ORDER BY total_objetos DESC;

-- 2) Nro. de personas por video
SELECT
  source_id,
  COUNT(*) AS personas_por_video
FROM yolo_objects
WHERE class_name = 'person'
GROUP BY source_id
ORDER BY personas_por_video DESC;

-- 3) Area promedio de bounding box por clase
SELECT
  class_name,
  AVG(area_pixels)     AS area_media_pixels,
  AVG(bbox_area_ratio) AS ratio_area_medio
FROM yolo_objects
GROUP BY class_name
ORDER BY area_media_pixels DESC;

-- 4) Distribucion coloress dominantes por clase
SELECT
  class_name,
  dominant_color_name,
  COUNT(*) AS total
FROM yolo_objects
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
