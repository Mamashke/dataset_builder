# modules/exporter.py — экспорт финального датасета в форматы YOLO и COCO.
#
# Не копирует файлы изображений — только создаёт файлы конфигурации/аннотаций,
# которые указывают на уже существующий датасет в data/processed/dataset/.
#
# Публичный интерфейс:
#   export(format)  — "yolo" или "coco", запускает соответствующий экспортёр.

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from modules.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Вспомогательная функция чтения изображения (поддержка Unicode-путей)
# ---------------------------------------------------------------------------

def _imread(path: Path) -> Optional[np.ndarray]:
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Экспорт в формат YOLO
# ---------------------------------------------------------------------------

def export_yolo() -> dict:
    """Создаёт data.yaml для обучения YOLO-моделей.

    Файл указывает на папки images/ и labels/ существующего датасета.
    Файлы изображений не копируются.

    Returns:
        {"yaml_path": str, "images": int}
    """
    out_dir = config.EXPORT_DIR / "yolo"
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = config.DATASET_IMAGES_DIR
    images_count = len(list(images_dir.glob("*.jpg"))) + \
                   len(list(images_dir.glob("*.jpeg"))) + \
                   len(list(images_dir.glob("*.png")))

    # data.yaml в формате, который принимает YOLOv8/YOLOv5
    yaml_path = out_dir / "data.yaml"
    yaml_content = (
        f"path: {config.DATASET_IMAGES_DIR.parent.as_posix()}\n"
        f"train: images\n"
        f"val: images\n"
        f"\n"
        f"nc: {len(config.CLASS_NAMES)}\n"
        f"names: {config.CLASS_NAMES}\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")

    logger.info(f"YOLO: data.yaml → {yaml_path}")
    logger.info(f"YOLO: изображений в датасете = {images_count}")

    return {"yaml_path": str(yaml_path), "images": images_count}


# ---------------------------------------------------------------------------
# Экспорт в формат COCO
# ---------------------------------------------------------------------------

def export_coco() -> dict:
    """Создаёт annotations.json в формате COCO Detection.

    Читает изображения из dataset/images/ и разметку из dataset/labels/.
    Конвертирует bbox из YOLO [x_c, y_c, w, h] (нормализованный)
    в COCO [x_min, y_min, w_px, h_px] (абсолютный).

    Returns:
        {"json_path": str, "images": int, "annotations": int}
    """
    out_dir = config.EXPORT_DIR / "coco"
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir  = config.DATASET_IMAGES_DIR
    labels_dir  = config.DATASET_LABELS_DIR

    # Собираем все изображения, сортируем для воспроизводимости
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    coco_images      = []  # список словарей image
    coco_annotations = []  # список словарей annotation
    ann_id = 1             # глобальный счётчик аннотаций (COCO требует уникальный id)

    logger.info(f"COCO: обрабатываю {len(image_paths)} изображений...")

    for img_id, img_path in enumerate(image_paths, start=1):
        # Читаем изображение только ради размеров (width, height)
        img = _imread(img_path)
        if img is None:
            logger.warning(f"Не удалось прочитать: {img_path.name} — пропускаем")
            continue

        h, w = img.shape[:2]

        coco_images.append({
            "id":        img_id,
            "file_name": img_path.name,
            "width":     w,
            "height":    h,
        })

        # Ищем соответствующий txt-файл разметки
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue  # негативный пример — аннотаций нет

        label_text = label_path.read_text(encoding="utf-8").strip()
        if not label_text:
            continue  # пустой файл — тоже негативный

        for line in label_text.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # пропускаем некорректные строки

            class_id = int(parts[0])
            x_c_n, y_c_n, w_n, h_n = map(float, parts[1:])

            # Конвертируем из нормализованного YOLO в абсолютный COCO
            w_px = w_n * w
            h_px = h_n * h
            x_min = (x_c_n - w_n / 2) * w
            y_min = (y_c_n - h_n / 2) * h

            coco_annotations.append({
                "id":          ann_id,
                "image_id":    img_id,
                "category_id": class_id,
                "bbox":        [round(x_min, 2), round(y_min, 2),
                                round(w_px, 2),  round(h_px, 2)],
                "area":        round(w_px * h_px, 2),
                "iscrowd":     0,
            })
            ann_id += 1

        if img_id % 500 == 0:
            logger.info(f"  Обработано {img_id}/{len(image_paths)}...")

    # Категории объектов
    categories = [
        {"id": i, "name": name}
        for i, name in enumerate(config.CLASS_NAMES)
    ]

    coco_data = {
        "images":      coco_images,
        "annotations": coco_annotations,
        "categories":  categories,
    }

    json_path = out_dir / "annotations.json"
    json_path.write_text(
        json.dumps(coco_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(f"COCO: annotations.json → {json_path}")
    logger.info(f"COCO: изображений={len(coco_images)}, аннотаций={len(coco_annotations)}")

    return {
        "json_path":   str(json_path),
        "images":      len(coco_images),
        "annotations": len(coco_annotations),
    }


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def export(format: str = "yolo") -> dict:
    """Экспортирует финальный датасет в указанный формат.

    Args:
        format: "yolo" — создаёт export/yolo/data.yaml
                "coco" — создаёт export/coco/annotations.json

    Returns:
        Словарь со статистикой экспорта.

    Raises:
        ValueError: если передан неизвестный формат.
    """
    supported = {"yolo", "coco"}
    if format not in supported:
        raise ValueError(f"Неизвестный формат '{format}'. Доступные: {supported}")

    logger.info("=" * 50)
    logger.info(f"Exporter | формат={format}")
    logger.info("=" * 50)

    if format == "yolo":
        stats = export_yolo()
    else:
        stats = export_coco()

    logger.info("ИТОГ: " + str(stats))
    logger.info("=" * 50)

    return stats
