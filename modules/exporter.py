# modules/exporter.py — экспорт финального датасета в форматы YOLO и COCO.
#
# YOLO: делит датасет 80/20 (train/val), копирует файлы в export/yolo/,
#       создаёт data.yaml с абсолютным путём к корню экспорта.
# COCO: читает изображения и разметку, создаёт export/coco/annotations.json.
#
# Публичный интерфейс:
#   export(project, format)  — "yolo" или "coco", запускает соответствующий экспортёр.

import json
import random
import shutil
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

def export_yolo(images_dir: Path, labels_dir: Path, out_dir: Path) -> dict:
    """Делит датасет 80/20, копирует файлы в export/yolo/ и создаёт data.yaml.

    Структура выходной папки:
        out_dir/
        ├── images/train/   — 80% изображений
        ├── images/val/     — 20% изображений
        ├── labels/train/   — соответствующие txt-метки
        ├── labels/val/
        └── data.yaml

    Args:
        images_dir: папка с изображениями датасета (project.dataset_images_dir).
        labels_dir: папка с метками датасета (project.dataset_labels_dir).
        out_dir:    корневая папка экспорта (project.export_dir / "yolo").

    Returns:
        {"yaml_path": str, "images": int, "train": int, "val": int}
    """
    # Собираем все изображения, сортируем для детерминированного порядка
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    # Перемешиваем с фиксированным seed — одинаковое разделение при каждом запуске
    random.seed(42)
    random.shuffle(image_paths)

    split_idx  = int(len(image_paths) * 0.8)
    train_imgs = image_paths[:split_idx]
    val_imgs   = image_paths[split_idx:]

    # Создаём структуру папок export/yolo/images/{train,val} и labels/{train,val}
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _copy_split(img_list: list, split: str) -> int:
        """Копирует изображения и метки в подпапку split, возвращает число скопированных."""
        for img_path in img_list:
            shutil.copy2(img_path, out_dir / "images" / split / img_path.name)

            label_src = labels_dir / (img_path.stem + ".txt")
            label_dst = out_dir / "labels" / split / (img_path.stem + ".txt")
            if label_src.exists():
                shutil.copy2(label_src, label_dst)
            else:
                # Негативный пример без txt — создаём пустой файл для совместимости с YOLO
                label_dst.touch()

        return len(img_list)

    n_train = _copy_split(train_imgs, "train")
    n_val   = _copy_split(val_imgs,   "val")
    total   = n_train + n_val

    # data.yaml с абсолютным путём к корню экспорта
    yaml_path = out_dir / "data.yaml"
    yaml_content = (
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(config.CLASS_NAMES)}\n"
        f"names: {config.CLASS_NAMES}\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")

    logger.info(f"YOLO: train={n_train}, val={n_val}, всего={total}")
    logger.info(f"YOLO: data.yaml → {yaml_path}")

    pct_train = n_train * 100 // total if total else 0
    pct_val   = n_val   * 100 // total if total else 0
    print(f"Экспорт YOLO завершён:")
    print(f"  train: {n_train} изображений ({pct_train}%)")
    print(f"  val:   {n_val} изображений ({pct_val}%)")
    print(f"  data.yaml: {yaml_path}")

    return {"yaml_path": str(yaml_path), "images": total, "train": n_train, "val": n_val}


# ---------------------------------------------------------------------------
# Экспорт в формат COCO
# ---------------------------------------------------------------------------

def _parse_coco_annotations(img_id: int, img_w: int, img_h: int,
                             label_path: Path, ann_id_start: int) -> list:
    """Читает YOLO-метки и возвращает список COCO-аннотаций для одного изображения.

    Args:
        img_id:       id изображения в COCO.
        img_w:        ширина изображения в пикселях.
        img_h:        высота изображения в пикселях.
        label_path:   путь к txt-файлу разметки (YOLO-формат).
        ann_id_start: начальный id для аннотаций этого изображения.

    Returns:
        Список словарей COCO annotation; пустой список для негативных примеров.
    """
    if not label_path.exists():
        return []

    label_text = label_path.read_text(encoding="utf-8").strip()
    if not label_text:
        return []

    annotations = []
    ann_id = ann_id_start
    for line in label_text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x_c_n, y_c_n, w_n, h_n = map(float, parts[1:])

        # Конвертируем из нормализованного YOLO в абсолютный COCO
        w_px  = w_n * img_w
        h_px  = h_n * img_h
        x_min = (x_c_n - w_n / 2) * img_w
        y_min = (y_c_n - h_n / 2) * img_h

        annotations.append({
            "id":          ann_id,
            "image_id":    img_id,
            "category_id": class_id,
            "bbox":        [round(x_min, 2), round(y_min, 2),
                            round(w_px, 2),  round(h_px, 2)],
            "area":        round(w_px * h_px, 2),
            "iscrowd":     0,
        })
        ann_id += 1

    return annotations


def export_coco(images_dir: Path, labels_dir: Path, out_dir: Path) -> dict:
    """Делит датасет 80/20, копирует изображения и создаёт два COCO JSON.

    Структура выходной папки:
        out_dir/
        ├── images/
        │   ├── train/         — 80% изображений
        │   └── val/           — 20% изображений
        └── annotations/
            ├── train.json     — аннотации train-части
            └── val.json       — аннотации val-части

    Конвертирует bbox из YOLO [x_c, y_c, w, h] (нормализованный)
    в COCO [x_min, y_min, w_px, h_px] (абсолютный).

    Args:
        images_dir: папка с изображениями датасета (project.dataset_images_dir).
        labels_dir: папка с метками датасета (project.dataset_labels_dir).
        out_dir:    корневая папка экспорта (project.export_dir / "coco").

    Returns:
        {"train_json": str, "val_json": str,
         "images": int, "annotations": int}
    """
    # Собираем все изображения, сортируем для детерминированного порядка
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    # Перемешиваем с фиксированным seed — одинаковое разделение при каждом запуске
    random.seed(42)
    random.shuffle(image_paths)

    split_idx = int(len(image_paths) * 0.8)
    splits = {
        "train": image_paths[:split_idx],
        "val":   image_paths[split_idx:],
    }

    # Создаём структуру папок
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)

    # Категории одинаковы в обоих JSON
    categories = [
        {"id": i, "name": name}
        for i, name in enumerate(config.CLASS_NAMES)
    ]

    split_stats = {}   # {"train": {"images": N, "annotations": N}, "val": {...}}
    json_paths  = {}   # {"train": Path, "val": Path}

    for split, img_list in splits.items():
        coco_images      = []
        coco_annotations = []
        ann_id = 1  # счётчик сбрасывается в каждом JSON — id уникальны внутри файла

        logger.info(f"COCO {split}: обрабатываю {len(img_list)} изображений...")

        for img_id, img_path in enumerate(img_list, start=1):
            # Копируем изображение в подпапку split
            shutil.copy2(img_path, out_dir / "images" / split / img_path.name)

            # Читаем размеры изображения
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

            # Парсим аннотации и добавляем к списку сплита
            label_path = labels_dir / (img_path.stem + ".txt")
            anns = _parse_coco_annotations(img_id, w, h, label_path, ann_id)
            coco_annotations.extend(anns)
            ann_id += len(anns)

            if img_id % 500 == 0:
                logger.info(f"  {split}: обработано {img_id}/{len(img_list)}...")

        coco_data = {
            "images":      coco_images,
            "annotations": coco_annotations,
            "categories":  categories,
        }

        json_path = out_dir / "annotations" / f"{split}.json"
        json_path.write_text(
            json.dumps(coco_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        json_paths[split]  = json_path
        split_stats[split] = {
            "images":      len(coco_images),
            "annotations": len(coco_annotations),
        }

        logger.info(
            f"COCO {split}: изображений={split_stats[split]['images']}, "
            f"аннотаций={split_stats[split]['annotations']}"
        )
        logger.info(f"COCO {split}: {split}.json → {json_path}")

    total_images      = split_stats["train"]["images"]      + split_stats["val"]["images"]
    total_annotations = split_stats["train"]["annotations"] + split_stats["val"]["annotations"]

    print(f"Экспорт COCO завершён:")
    print(f"  train: {split_stats['train']['images']} изображений, "
          f"{split_stats['train']['annotations']} аннотаций")
    print(f"  val:   {split_stats['val']['images']} изображений, "
          f"{split_stats['val']['annotations']} аннотаций")

    return {
        "train_json":  str(json_paths["train"]),
        "val_json":    str(json_paths["val"]),
        "images":      total_images,
        "annotations": total_annotations,
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

    # Пути к датасету берём из data_sources; если не задан — fallback на фиксированный путь
    images_dir = project.get_source("dataset", "images") or project.dataset_images_dir
    labels_dir = project.get_source("dataset", "labels") or project.dataset_labels_dir
    out_dir    = project.export_dir / format

    if format == "yolo":
        stats = export_yolo(images_dir, labels_dir, out_dir)
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
