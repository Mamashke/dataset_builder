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
from modules.project import Project

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

def export_yolo(images_dir: Path, out_dir: Path) -> dict:
    """Создаёт data.yaml для обучения YOLO-моделей.

    Файл указывает на папку images/ существующего датасета проекта.
    Файлы изображений не копируются.

    Args:
        images_dir: папка с изображениями датасета (project.dataset_images_dir).
        out_dir:    куда записать data.yaml (project.export_dir / "yolo").

    Returns:
        {"yaml_path": str, "images": int}
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    images_count = len(list(images_dir.glob("*.jpg"))) + \
                   len(list(images_dir.glob("*.jpeg"))) + \
                   len(list(images_dir.glob("*.png")))

    # data.yaml в формате, который принимает YOLOv8/YOLOv5
    yaml_path = out_dir / "data.yaml"
    yaml_content = (
        f"path: {images_dir.parent.as_posix()}\n"
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

def export_coco(images_dir: Path, labels_dir: Path, out_dir: Path) -> dict:
    """Создаёт annotations.json в формате COCO Detection.

    Читает изображения и разметку из папок датасета проекта.
    Конвертирует bbox из YOLO [x_c, y_c, w, h] (нормализованный)
    в COCO [x_min, y_min, w_px, h_px] (абсолютный).

    Args:
        images_dir: папка с изображениями датасета (project.dataset_images_dir).
        labels_dir: папка с метками датасета (project.dataset_labels_dir).
        out_dir:    куда записать annotations.json (project.export_dir / "coco").

    Returns:
        {"json_path": str, "images": int, "annotations": int}
    """
    out_dir.mkdir(parents=True, exist_ok=True)

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

def export(project: Project, format: str = "yolo") -> dict:
    """Экспортирует финальный датасет проекта в указанный формат.

    Args:
        project: объект Project — определяет пути к датасету и папке экспорта.
        format:  "yolo" — создаёт export/yolo/data.yaml
                 "coco" — создаёт export/coco/annotations.json

    Returns:
        Словарь со статистикой экспорта.

    Raises:
        ValueError: если передан неизвестный формат.
    """
    # Подключаем файловый лог проекта — все записи logger попадут
    # в project.logs_dir/exporter.log
    get_logger(__name__, project.logs_dir)

    supported = {"yolo", "coco"}
    if format not in supported:
        raise ValueError(f"Неизвестный формат '{format}'. Доступные: {supported}")

    logger.info("=" * 50)
    logger.info(f"Exporter | project={project.name} | формат={format}")
    logger.info("=" * 50)

    # Пути к датасету и выходной папке берём из проекта
    images_dir = project.dataset_images_dir
    labels_dir = project.dataset_labels_dir
    out_dir    = project.export_dir / format

    if format == "yolo":
        stats = export_yolo(images_dir, out_dir)
    else:
        stats = export_coco(images_dir, labels_dir, out_dir)

    logger.info("ИТОГ: " + str(stats))
    logger.info("=" * 50)

    # Обновляем метаданные проекта
    project.update_step("export")
    project.update_stats({
        "exported_format": format,
        "exported_images": stats.get("images", 0),
    })

    return stats
