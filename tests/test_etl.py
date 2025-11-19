import pandas as pd
from src.sistema_batch_etl import limpiar_df


def test_limpiar_df_elimina_nulos_y_duplicados():
    data = {
        "source_type": ["image", None],
        "source_id": ["img1.jpg", "img2.jpg"],
        "frame_number": [0, 0],
        "class_id": [0, 0],
        "class_name": ["person", "person"],
        "confidence": [0.9, 1.1],  
        "x_min": [10, 10],
        "y_min": [20, 20],
        "x_max": [50, 50],
        "y_max": [80, 80],
        "width": [40, 40],
        "height": [60, 60],
        "area_pixels": [2400, 2400],
        "frame_width": [640, 640],
        "frame_height": [480, 480],
        "bbox_area_ratio": [0.01, 0.01],
        "center_x": [30, 30],
        "center_y": [50, 50],
        "center_x_norm": [0.5, 0.5],
        "center_y_norm": [0.5, 0.5],
        "position_region": ["middle-center", "middle-center"],
        "dominant_color_name": ["red", "red"],
        "dom_r": [255, 255],
        "dom_g": [0, 0],
        "dom_b": [0, 0],
        "timestamp_sec": [0.0, 0.0],
        "ingestion_date": ["2025-01-01", "2025-01-01"],
        "detection_id": ["A", "A"],  # duplicado
    }
    df = pd.DataFrame(data)

    cleaned = limpiar_df(df)

    #solo debe quedar 1 fila valida
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["detection_id"] == "A"
    assert cleaned.iloc[0]["confidence"] == 0.9
