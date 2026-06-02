# modules/compositor.py — синтетическая аугментация методом Copy-Paste.
#
# Вырезает фигуры людей из позитивных кадров датасета и вставляет их
# на негативные кадры, формируя синтетические обучающие примеры с
# автоматически сгенерированными YOLO-аннотациями.
#
# Публичный интерфейс:
#   extract_persons(project)         — вырезать людей из dataset/images/ → persons/
#   compose(project, count)          — собрать синтетические кадры comp_*.jpg
#
# Формат выходных аннотаций (YOLO): class x_c y_c w h (нормализованные 0–1).
# Класс людей: 0.

import random
from pathlib import Path
from typing import List

import cv2
import numpy as np

from modules.logger import get_logger
from modules.project import Project

logger = get_logger(__name__)

# Допустимые расширения изображений
_IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Диапазон высоты человека при масштабировании (вид с БПЛА)
_PERSON_HEIGHT_MIN = 10
_PERSON_HEIGHT_MAX = 30

# Диапазон угла поворота
_ROTATE_MIN = -15.0
_ROTATE_MAX =  15.0

# Диапазон коэффициента яркости
_BRIGHTNESS_MIN = 0.8
_BRIGHTNESS_MAX = 1.2


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _read_image(path: Path):
    """Читает изображение через numpy+cv2 (поддержка Unicode-путей на Windows)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _write_image(path: Path, img: np.ndarray, quality: int = 90) -> bool:
    """Сохраняет изображение в файл с заданным JPEG-качеством."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        path.write_bytes(buf.tobytes())
    return ok


def _rotate_crop(img: np.ndarray, angle: float) -> np.ndarray:
    """Поворачивает прямоугольный фрагмент на заданный угол.

    Граничные пиксели заполняются репликацией края (без чёрных артефактов).
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Публичная функция: извлечение фигур людей
# ---------------------------------------------------------------------------

def extract_persons(project: Project) -> List[Path]:
    """Вырезает прямоугольники с людьми из позитивных кадров датасета.

    Для каждого изображения из dataset/images/ читает соответствующий
    файл разметки из dataset/labels/. Кадры без аннотаций (негативные)
    и пустые файлы пропускаются. Для каждой bbox вырезает регион и
    сохраняет в project.persons_dir.

    Args:
        project: объект Project с путями к датасету и папке persons.

    Returns:
        Список путей к сохранённым вырезанным фигурам.

    Raises:
        FileNotFoundError: если dataset_images_dir не существует или пуста.
    """
    images_dir  = project.dataset_images_dir
    labels_dir  = project.dataset_labels_dir
    persons_dir = project.persons_dir

    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise FileNotFoundError(
            f"Папка с изображениями датасета не найдена или пуста: {images_dir}\n"
            f"Сначала запустите шаг balance."
        )

    persons_dir.mkdir(parents=True, exist_ok=True)

    # Собираем все изображения датасета
    image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in _IMG_EXTS]
    if not image_files:
        raise FileNotFoundError(f"Изображения не найдены в {images_dir}")

    logger.info(
        f"extract_persons | проект={project.name} | "
        f"изображений={len(image_files)} | выход={persons_dir}"
    )

    saved_paths: List[Path] = []
    extracted   = 0
    skipped     = 0

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")

        # Пропускаем негативные кадры (нет аннотации или пустой файл)
        if not label_path.exists():
            skipped += 1
            continue
        content = label_path.read_text(encoding="utf-8").strip()
        if not content:
            skipped += 1
            continue

        img = _read_image(img_path)
        if img is None:
            logger.warning(f"Не удалось прочитать: {img_path.name}")
            skipped += 1
            continue

        h, w = img.shape[:2]

        # Парсим YOLO-аннотации: class x_c y_c w_n h_n (нормализованные)
        for obj_idx, line in enumerate(content.splitlines()):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                x_c = float(parts[1])
                y_c = float(parts[2])
                bw  = float(parts[3])
                bh  = float(parts[4])
            except ValueError:
                continue

            # Переводим из нормализованных в пиксельные координаты
            x1 = max(0, int((x_c - bw / 2) * w))
            y1 = max(0, int((y_c - bh / 2) * h))
            x2 = min(w,  int((x_c + bw / 2) * w))
            y2 = min(h,  int((y_c + bh / 2) * h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_path = persons_dir / f"{img_path.stem}_obj{obj_idx}.jpg"
            if _write_image(out_path, crop):
                saved_paths.append(out_path)
                extracted += 1

    logger.info(
        f"extract_persons завершён: вырезано={extracted}, "
        f"пропущено кадров={skipped}"
    )
    print(f"Вырезано фигур: {extracted} → {persons_dir}")

    return saved_paths


# ---------------------------------------------------------------------------
# Публичная функция: компоновка синтетических кадров
# ---------------------------------------------------------------------------

def compose(project: Project, count: int = 200) -> dict:
    """Генерирует синтетические кадры методом Copy-Paste.

    Алгоритм для каждого синтетического кадра:
      1. Берёт случайный негативный кадр из dataset/images/ как фон.
      2. Вставляет 1–3 случайных человека из persons_dir.
      3. Для каждого человека:
         - масштабирует до высоты 10–30 пикселей (вид с БПЛА),
         - применяет случайный поворот -15..+15°,
         - лёгкое гауссово размытие (ядро 3–7),
         - небольшое изменение яркости ±20%,
         - вставляет в случайную позицию на фоне.
      4. Записывает YOLO-аннотацию с координатами вставки.

    Сохраняет:
      - изображение:  project.frames_real_dir / "comp_NNNNNN.jpg"
      - аннотацию:    project.annotations_dir / "real" / "comp_NNNNNN.txt"

    Args:
        project: объект Project.
        count:   количество синтетических кадров (по умолчанию 200).

    Returns:
        {"composed": N} — фактическое число созданных кадров.

    Raises:
        FileNotFoundError: если нет негативных кадров или фигур людей.
    """
    images_dir  = project.dataset_images_dir
    labels_dir  = project.dataset_labels_dir
    persons_dir = project.persons_dir
    output_dir  = project.frames_real_dir
    # Аннотации comp_ кадров кладём туда же, куда annotator сохраняет real-разметку
    ann_dir     = project.annotations_dir / "real"

    # Собираем негативные кадры (фоны) — пустые или отсутствующие аннотации
    backgrounds: List[Path] = []
    if images_dir.exists():
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                backgrounds.append(img_path)
                continue
            if not label_path.read_text(encoding="utf-8").strip():
                backgrounds.append(img_path)

    if not backgrounds:
        raise FileNotFoundError(
            f"Негативные кадры (пустые аннотации) не найдены в {images_dir}.\n"
            f"Для фонов нужны кадры без объектов. Сначала запустите balance."
        )

    # Собираем вырезанных людей
    person_files: List[Path] = []
    if persons_dir.exists():
        person_files = [
            p for p in persons_dir.iterdir()
            if p.suffix.lower() in _IMG_EXTS
        ]

    if not person_files:
        raise FileNotFoundError(
            f"Фигуры людей не найдены в {persons_dir}.\n"
            f"Сначала запустите: --extract-persons"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Стартовый номер — не перезаписываем существующие comp-кадры
    existing_nums: set = set()
    for f in output_dir.glob("comp_*.jpg"):
        try:
            existing_nums.add(int(f.stem[5:]))
        except ValueError:
            pass
    next_num = max(existing_nums, default=-1) + 1

    logger.info(
        f"compose | проект={project.name} | фонов={len(backgrounds)} | "
        f"фигур={len(person_files)} | count={count} | нумерация с comp_{next_num:06d}"
    )
    print(
        f"Компоновка {count} кадров | фонов={len(backgrounds)} | "
        f"фигур={len(person_files)}..."
    )

    composed = 0

    for _ in range(count):
        # Загружаем случайный фон
        bg_path = random.choice(backgrounds)
        bg = _read_image(bg_path)
        if bg is None:
            continue
        bg = bg.copy()  # не модифицируем оригинал

        # Приводим фон к стандартному разрешению — защита от нерезайзнутых исходников
        if bg.shape[0] != 640 or bg.shape[1] != 640:
            bg = cv2.resize(bg, (640, 640), interpolation=cv2.INTER_AREA)

        bg_h, bg_w = bg.shape[:2]

        yolo_lines: List[str] = []

        # Вставляем 1–3 человека на один кадр
        n_persons = random.randint(1, 3)
        for _ in range(n_persons):
            person = _read_image(random.choice(person_files))
            if person is None:
                continue

            p_h_orig, p_w_orig = person.shape[:2]
            if p_h_orig == 0 or p_w_orig == 0:
                continue

            # Масштабируем до целевой высоты 10–30 пикселей
            target_h = random.randint(_PERSON_HEIGHT_MIN, _PERSON_HEIGHT_MAX)
            scale    = target_h / p_h_orig
            target_w = max(1, int(p_w_orig * scale))
            person   = cv2.resize(
                person, (target_w, target_h), interpolation=cv2.INTER_AREA
            )

            # Случайный поворот -15..+15 градусов
            angle = random.uniform(_ROTATE_MIN, _ROTATE_MAX)
            if abs(angle) > 0.5:
                person = _rotate_crop(person, angle)

            # Лёгкое гауссово размытие (ядра 3, 5, 7 — 1, 2, 3 пикселя размытия)
            blur_r = random.randint(1, 3)
            ksize  = blur_r * 2 + 1  # 3, 5, 7
            person = cv2.GaussianBlur(person, (ksize, ksize), 0)

            # Небольшое изменение яркости ±20%
            brightness = random.uniform(_BRIGHTNESS_MIN, _BRIGHTNESS_MAX)
            person = np.clip(
                person.astype(np.float32) * brightness, 0, 255
            ).astype(np.uint8)

            p_h, p_w = person.shape[:2]

            # Пропускаем, если фигура не помещается в кадр
            if p_w >= bg_w or p_h >= bg_h:
                continue

            # Случайная позиция вставки
            x = random.randint(0, bg_w - p_w)
            y = random.randint(0, bg_h - p_h)

            # Вставляем фигуру на фон (прямоугольное наложение)
            bg[y : y + p_h, x : x + p_w] = person

            # Формируем YOLO-аннотацию (нормализованные координаты центра и размеров)
            x_c = (x + p_w / 2) / bg_w
            y_c = (y + p_h / 2) / bg_h
            w_n = p_w / bg_w
            h_n = p_h / bg_h
            yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        # Не сохраняем кадр, если ни один человек не был вставлен
        if not yolo_lines:
            continue

        img_out_path = output_dir / f"comp_{next_num:06d}.jpg"
        ann_out_path = ann_dir    / f"comp_{next_num:06d}.txt"

        if not _write_image(img_out_path, bg):
            logger.warning(f"Не удалось сохранить кадр: {img_out_path.name}")
            continue

        ann_out_path.write_text("\n".join(yolo_lines), encoding="utf-8")

        next_num += 1
        composed  += 1

    logger.info(f"compose завершён: создано={composed} кадров → {output_dir}")
    print(f"Создано кадров: {composed} → {output_dir}")

    project.update_stats({"composed": composed})
    return {"composed": composed}
