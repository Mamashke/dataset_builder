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

import json
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from modules.logger import get_logger
from modules.project import Project

logger = get_logger(__name__)

# Допустимые расширения изображений
_IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Диапазон угла поворота
_ROTATE_MIN = -15.0
_ROTATE_MAX =  15.0

# Диапазон коэффициента яркости
_BRIGHTNESS_MIN = 0.8
_BRIGHTNESS_MAX = 1.2

# Допустимое отклонение масштаба от среднего (±20 %)
_SCALE_VARIATION = 0.20

# Размер ядра гауссова размытия краёв маски вставки (нечётное число)
_EDGE_BLUR_KERNEL = 15


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


def _blend_person(bg: np.ndarray, person: np.ndarray, x: int, y: int) -> None:
    """Вставляет фигуру человека на фон с размытыми краями (alpha-blending).

    Создаёт бинарную маску (1 — человек, 0 — фон), размывает её края
    гауссовым фильтром (_EDGE_BLUR_KERNEL × _EDGE_BLUR_KERNEL), чтобы
    получить плавный переход. Изменяет bg на месте.

    Формула: result = фон * (1 − маска) + человек * маска
    """
    p_h, p_w = person.shape[:2]

    # Создаём белую маску размером с вставляемый патч
    mask = np.ones((p_h, p_w), dtype=np.float32)
    # Адаптивный kernel: не больше патча и не больше _EDGE_BLUR_KERNEL, всегда нечётный >= 3
    k = min(p_h - 1, p_w - 1, _EDGE_BLUR_KERNEL)
    k = k if k % 2 == 1 else k - 1
    k = max(k, 3)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    # Нормализуем обратно к 1.0 в центре (после GaussianBlur максимум < 1)
    mask /= mask.max()
    # Расширяем для поканального умножения: (H, W) → (H, W, 1)
    mask3 = mask[:, :, np.newaxis]

    bg_roi   = bg[y : y + p_h, x : x + p_w].astype(np.float32)
    person_f = person.astype(np.float32)

    blended = bg_roi * (1.0 - mask3) + person_f * mask3
    bg[y : y + p_h, x : x + p_w] = np.clip(blended, 0, 255).astype(np.uint8)


def _load_metadata(persons_dir: Path) -> list:
    """Загружает метаданные вырезанных людей из persons/metadata.json."""
    meta_path = persons_dir / "metadata.json"
    if not meta_path.exists():
        return []
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _avg_person_height(metadata: list) -> Optional[float]:
    """Возвращает среднюю высоту людей из метаданных, или None если нет данных."""
    heights = [m["original_height_px"] for m in metadata if "original_height_px" in m]
    return sum(heights) / len(heights) if heights else None


def _bg_color_around_bbox(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: int = 5
) -> List[int]:
    """Считает средний BGR цвет фона вокруг bbox (полоса шириной pad пикселей).

    Берёт четыре полосы за границей bbox (сверху, снизу, слева, справа),
    объединяет все пиксели и возвращает средний цвет как [B, G, R].
    Полосы автоматически обрезаются до границ изображения.
    """
    h, w = img.shape[:2]
    strips = []

    # Полоса сверху — от (y1-pad) до y1
    r0, r1 = max(0, y1 - pad), y1
    if r1 > r0 and x2 > x1:
        strips.append(img[r0:r1, x1:x2])

    # Полоса снизу — от y2 до (y2+pad)
    r0, r1 = y2, min(h, y2 + pad)
    if r1 > r0 and x2 > x1:
        strips.append(img[r0:r1, x1:x2])

    # Полоса слева — от (x1-pad) до x1
    c0, c1 = max(0, x1 - pad), x1
    if c1 > c0 and y2 > y1:
        strips.append(img[y1:y2, c0:c1])

    # Полоса справа — от x2 до (x2+pad)
    c0, c1 = x2, min(w, x2 + pad)
    if c1 > c0 and y2 > y1:
        strips.append(img[y1:y2, c0:c1])

    if not strips:
        return [0, 0, 0]

    # Объединяем пиксели всех полос и считаем среднее по каждому каналу
    pixels = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0)
    mean   = pixels.mean(axis=0)
    return [int(mean[0]), int(mean[1]), int(mean[2])]


def _pick_person_by_color(
    person_files: List[Path],
    meta_by_name: dict,
    bg_color: List[int],
    top_k: int = 5,
) -> Path:
    """Выбирает человека из списка, чей цвет фона наиболее близок к bg_color.

    Вычисляет евклидово расстояние в BGR-пространстве между bg_color и
    bg_color из метаданных каждого человека. Берёт top_k ближайших и
    возвращает случайного из них — разнообразие сохраняется, но цветовая
    согласованность улучшается.

    Если ни для одного человека нет bg_color в метаданных — возвращает
    случайного из всего списка (обратная совместимость со старыми persons/).
    """
    # Отбираем только тех, для кого записан bg_color
    with_color = [
        p for p in person_files
        if p.name in meta_by_name and "bg_color" in meta_by_name[p.name]
    ]
    if not with_color:
        return random.choice(person_files)

    bg = np.array(bg_color, dtype=np.float32)

    # Считаем расстояние: sqrt((b1-b2)² + (g1-g2)² + (r1-r2)²)
    distances = []
    for p in with_color:
        pc   = np.array(meta_by_name[p.name]["bg_color"], dtype=np.float32)
        dist = float(np.linalg.norm(bg - pc))
        distances.append((dist, p))

    # Сортируем по расстоянию — ближайшие первые
    distances.sort(key=lambda x: x[0])

    # Берём топ-k и выбираем случайного — баланс между точностью и разнообразием
    k    = min(top_k, len(distances))
    top  = [p for _, p in distances[:k]]
    return random.choice(top)


# ---------------------------------------------------------------------------
# Публичная функция: извлечение фигур людей
# ---------------------------------------------------------------------------

def extract_persons(project: Project) -> List[Path]:
    """Вырезает прямоугольники с людьми из позитивных кадров датасета.

    Источники данных определяются через data_sources проекта (приоритет),
    с fallback на стандартные папки проекта, и в крайнем случае — на dataset/.
    Кадры без аннотаций (негативные) и пустые файлы пропускаются.
    Для каждой bbox вырезает регион и сохраняет в project.persons_dir.

    Дополнительно записывает persons/metadata.json с оригинальной высотой
    каждой вырезанной фигуры — используется в compose() для нормализации масштаба.

    Args:
        project: объект Project с путями к датасету и папке persons.

    Returns:
        Список путей к сохранённым вырезанным фигурам.

    Raises:
        FileNotFoundError: если не найдена ни одна папка с изображениями.
    """
    persons_dir = project.persons_dir

    # Кадры — приоритет data_sources, fallback на папку проекта
    frames_dir = project.get_source("frames", "real") or project.frames_real_dir

    # Аннотации — приоритет data_sources, fallback на папку проекта
    annotations_dir = (
        project.get_source("annotations", "real")
        or (project.annotations_dir / "real")
    )

    # Если из data_sources/проекта данных нет — берём готовый датасет
    if not frames_dir.exists() or not any(frames_dir.iterdir()):
        frames_dir      = project.dataset_images_dir
        annotations_dir = project.dataset_labels_dir

    logger.info(f"Кадры: {frames_dir}")
    logger.info(f"Аннотации: {annotations_dir}")

    if not frames_dir.exists() or not any(frames_dir.iterdir()):
        raise FileNotFoundError(
            f"Папка с кадрами не найдена или пуста: {frames_dir}\n"
            f"Проверьте data_sources проекта или запустите шаг balance."
        )

    persons_dir.mkdir(parents=True, exist_ok=True)

    # Собираем все изображения из выбранной папки
    image_files = [p for p in frames_dir.iterdir() if p.suffix.lower() in _IMG_EXTS]
    if not image_files:
        raise FileNotFoundError(f"Изображения не найдены в {frames_dir}")

    logger.info(
        f"extract_persons | проект={project.name} | "
        f"изображений={len(image_files)} | выход={persons_dir}"
    )

    saved_paths: List[Path] = []
    # Список записей для persons/metadata.json
    metadata: list = []
    extracted = 0
    skipped   = 0
    processed = 0   # кадры, дошедшие до разбора bbox (не пропущенные)

    for img_path in image_files:
        label_path = annotations_dir / (img_path.stem + ".txt")

        # Пропускаем аугментированные и синтетические кадры — люди нужны только
        # из оригинальных кадров, чтобы не дублировать одни и те же патчи
        _stem = img_path.stem
        if (_stem.endswith(("_fog", "_rain", "_noise", "_blur", "_brightness"))
                or "_sd_" in _stem
                or _stem.startswith("gan_")
                or _stem.startswith("comp_")):
            skipped += 1
            continue

        # Первые 5 кадров — диагностика пути к аннотации
        if processed + skipped < 5:
            logger.info(
                f"Кадр: {img_path.name} | "
                f"Ищу аннотацию: {label_path} | "
                f"Существует: {label_path.exists()}"
            )

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
        processed += 1   # кадр прошёл все проверки, идём разбирать bbox

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

            patch_h = y2 - y1
            patch_w = x2 - x1

            # Пропускаем слишком мелкие патчи — нет смысла вырезать нечитаемые пиксели
            if patch_h < 8 or patch_w < 5:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_name = f"{img_path.stem}_obj{obj_idx}.jpg"
            out_path = persons_dir / out_name
            if _write_image(out_path, crop):
                saved_paths.append(out_path)
                # Средний цвет фона вокруг bbox — для подбора людей по цвету в compose()
                bg_color = _bg_color_around_bbox(img, x1, y1, x2, y2)
                metadata.append({
                    "filename":           out_name,
                    "original_height_px": patch_h,
                    "patch_h":            patch_h,
                    "patch_w":            patch_w,
                    "bg_color":           bg_color,
                })
                extracted += 1

    # Сохраняем метаданные рядом с вырезанными фигурами
    meta_path = persons_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Метаданные записаны → {meta_path} ({len(metadata)} записей)")

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
         - масштабирует до среднего размера из metadata.json ±20%
           (или 10–30 px, если metadata.json отсутствует),
         - применяет случайный поворот -15..+15°,
         - лёгкое гауссово размытие (ядро 3–7),
         - небольшое изменение яркости ±20%,
         - вставляет с размытыми краями (alpha-blending через маску).
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
    persons_dir = project.persons_dir
    output_dir  = project.frames_real_dir
    # Аннотации comp_ кадров кладём туда же, куда annotator сохраняет real-разметку
    ann_dir = project.annotations_dir / "real"

    # Выбираем источник фонов.
    # frames/real/ — приоритет: туда пишут generate_sd, load и augment.
    # dataset/images/ — запасной: используется если frames/real/ пуст
    # (например, пользователь запускает compose отдельно после полного pipeline).
    _frames_dir = project.frames_real_dir
    _dataset_dir = project.dataset_images_dir
    if _frames_dir.exists() and any(_frames_dir.iterdir()):
        bg_frames_dir = _frames_dir
        bg_labels_dir = project.annotations_dir / "real"
        logger.info(f"Фоны берём из frames/real: {bg_frames_dir}")
    else:
        bg_frames_dir = _dataset_dir
        bg_labels_dir = project.dataset_labels_dir
        logger.info(f"frames/real/ пуст — фоны берём из dataset: {bg_frames_dir}")

    # Собираем негативные кадры (фоны) — пустые или отсутствующие аннотации.
    # sd_forest_ исключаем: в лесных сценах человек плохо различим,
    # используем только sd_open_ и обычные негативные кадры.
    backgrounds: List[Path] = []
    if bg_frames_dir.exists():
        for img_path in bg_frames_dir.iterdir():
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            if img_path.stem.startswith("sd_forest_"):
                continue
            label_path = bg_labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                backgrounds.append(img_path)
                continue
            if not label_path.read_text(encoding="utf-8").strip():
                backgrounds.append(img_path)

    if not backgrounds:
        raise FileNotFoundError(
            f"Негативные кадры (пустые аннотации) не найдены.\n"
            f"Проверено: {bg_frames_dir}\n"
            f"Убедитесь что generate_sd выполнен или запустите balance."
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

    # Загружаем метаданные для нормализации масштаба и подбора по цвету
    metadata   = _load_metadata(persons_dir)
    avg_height = _avg_person_height(metadata)
    if avg_height is not None:
        logger.info(f"Средняя высота людей из metadata.json: {avg_height:.1f} px")
        print(
            f"Средняя высота людей: {avg_height:.1f} px "
            f"(±{int(_SCALE_VARIATION * 100)}%)"
        )
    else:
        # Если metadata.json отсутствует — используем фиксированный диапазон
        logger.warning(
            "metadata.json не найден, масштаб будет случайным в диапазоне 10–30 px"
        )

    # Индекс метаданных по имени файла — для быстрого поиска bg_color в цикле
    meta_by_name = {m["filename"]: m for m in metadata}
    has_color = sum(1 for m in metadata if "bg_color" in m)
    logger.info(
        f"Метаданные цвета: {has_color}/{len(metadata)} фигур имеют bg_color"
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

        # Средний BGR цвет всего фонового кадра — основа для подбора человека
        bg_mean_color = [
            int(bg[:, :, 0].mean()),  # B
            int(bg[:, :, 1].mean()),  # G
            int(bg[:, :, 2].mean()),  # R
        ]

        yolo_lines: List[str] = []

        # Вставляем 1–3 человека на один кадр
        n_persons = random.randint(1, 3)
        for _ in range(n_persons):
            # Подбираем человека чей исходный фон близок по цвету к текущему фону
            chosen_path = _pick_person_by_color(person_files, meta_by_name, bg_mean_color)
            person = _read_image(chosen_path)
            if person is None:
                continue

            p_h_orig, p_w_orig = person.shape[:2]
            if p_h_orig == 0 or p_w_orig == 0:
                continue

            # Берём оригинальный размер патча из metadata — он точнее чем shape
            # загруженного JPEG (тот мог быть пересохранён с потерями)
            meta_entry = meta_by_name.get(chosen_path.name, {})
            patch_h = meta_entry.get("patch_h", p_h_orig)
            patch_w = meta_entry.get("patch_w", p_w_orig)

            # Всегда масштабируем патч вниз до TARGET_HEIGHT — реалистичный размер
            # человека на кадре 640×640 с дроновой высоты (15–35 px)
            target_h = random.randint(15, 35)
            scale    = target_h / patch_h
            new_w    = max(1, int(patch_w * scale))
            person   = cv2.resize(
                person, (new_w, target_h), interpolation=cv2.INTER_AREA
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

            # Вставляем с размытыми краями для плавного перехода
            _blend_person(bg, person, x, y)

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
