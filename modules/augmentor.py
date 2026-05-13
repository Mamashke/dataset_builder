# modules/augmentor.py — модуль аугментации кадров датасета.
#
# Имитирует сложные условия съёмки: туман, дождь, шум, размытие, изменение яркости.
# Каждый тип аугментации — отдельная функция, принимающая numpy-массив изображения
# и параметр intensity ∈ [0.0, 1.0], управляющий силой эффекта.
#
# Публичный интерфейс:
#   add_fog()          — белый полупрозрачный overlay + снижение контраста
#   add_rain()         — случайные диагональные полупрозрачные линии
#   add_noise()        — гауссовский шум поверх изображения
#   add_blur()         — размытие через Gaussian blur
#   add_brightness()   — изменение яркости (темнее или светлее)
#   augment_dataset()  — применяет выбранные аугментации ко всему датасету

import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import config  # Централизованные настройки и пути проекта
from modules.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Реестр поддерживаемых аугментаций: имя → функция
# Заполняется после определения функций через декоратор-регистратор.
# ---------------------------------------------------------------------------

AUGMENTATIONS = {}

# ---------------------------------------------------------------------------
# Вспомогательная утилита
# ---------------------------------------------------------------------------

def _imread(path: Path) -> Optional[np.ndarray]:
    """Читает изображение с поддержкой кириллических и Unicode-путей.

    cv2.imread на Windows не поддерживает пути с символами вне ASCII.
    Обходим это через np.fromfile + cv2.imdecode.

    Args:
        path: путь к изображению.

    Returns:
        BGR-массив или None, если файл не удалось открыть.
    """
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _imwrite(path: Path, image: np.ndarray, quality: int = 95) -> bool:
    """Сохраняет изображение с поддержкой кириллических и Unicode-путей.

    cv2.imwrite на Windows не поддерживает пути с символами вне ASCII.
    Обходим через cv2.imencode + ndarray.tofile.

    Args:
        path:    путь для сохранения (расширение определяет формат).
        image:   BGR-массив.
        quality: качество JPEG (0–100); игнорируется для форматов без потерь.

    Returns:
        True при успехе, False при ошибке кодирования.
    """
    ext = path.suffix.lower()  # например ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in {".jpg", ".jpeg"} else []
    ok, buf = cv2.imencode(ext, image, params)
    if ok:
        buf.tofile(str(path))
    return ok


def _clip(image: np.ndarray) -> np.ndarray:
    """Обрезает значения пикселей до допустимого диапазона [0, 255].

    После арифметических операций над uint8-массивом значения могут выйти
    за границы. np.clip + astype возвращают корректный uint8-массив.

    Args:
        image: массив формата (H, W, C) или (H, W) с произвольным dtype.

    Returns:
        uint8-массив с теми же размерами.
    """
    return np.clip(image, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Аугментация 1: туман
# ---------------------------------------------------------------------------

def add_fog(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Добавляет эффект тумана: белый overlay + снижение контраста.

    Туман реализован двумя шагами:
    1. Линейное смешение (alpha-blending) изображения с белым фоном —
       чем больше intensity, тем сильнее картинка «выцветает» к белому.
    2. Снижение контраста: пиксели сжимаются к серому значению 128,
       что имитирует размытие цветов в плотном тумане.

    Args:
        image:     исходное изображение (H, W, 3), uint8, BGR.
        intensity: сила эффекта от 0.0 (без изменений) до 1.0 (максимальный туман).

    Returns:
        Аугментированное изображение того же размера и типа.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))

    # Белый холст того же размера, что и исходное изображение
    fog_layer = np.full_like(image, 255, dtype=np.uint8)

    # alpha-blending: result = (1 - alpha) * image + alpha * fog_layer
    # При intensity=0.5 alpha=0.4 — заметный, но не непрозрачный туман
    alpha = intensity * 0.8
    fogged = cv2.addWeighted(image, 1.0 - alpha, fog_layer, alpha, 0)

    # Сжимаем контраст: приближаем пиксели к нейтральному серому (128).
    # Формула: pixel = 128 + (pixel - 128) * contrast_factor
    # При contrast_factor < 1 диапазон яркостей сужается → туманный вид.
    contrast_factor = 1.0 - intensity * 0.4
    fogged = fogged.astype(np.float32)
    fogged = 128 + (fogged - 128) * contrast_factor

    return _clip(fogged)


# ---------------------------------------------------------------------------
# Аугментация 2: дождь
# ---------------------------------------------------------------------------

def add_rain(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Реалистичный дождь с перспективой вертикально-вниз камеры БПЛА.

    Угол каждой капли радиальный от центра кадра:
      центр → вертикальные штрихи (|)
      края  → наклон до 30° (\ слева, / справа)
    Три слоя (крупные / средние / мелкие) с разным blur создают глубину.
    Финальный blur 3×3 имитирует влажный объектив.

    Args:
        image:     исходное изображение (H, W, 3), uint8, BGR.
        intensity: сила эффекта от 0.0 (слабый) до 1.0 (ливень).

    Returns:
        Аугментированное изображение того же размера и типа.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    # Максимальное расстояние от центра (угол кадра)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)

    num_drops = np.random.randint(2500, 3501)
    n_large  = int(num_drops * 0.15)
    n_medium = int(num_drops * 0.55)
    n_small  = num_drops - n_large - n_medium

    def _make_layer(n: int, len_lo: int, len_hi: int, thick: int) -> np.ndarray:
        """Рисует n капель на чёрном слое; радиальный угол вычислен векторно."""
        layer = np.zeros_like(image, dtype=np.uint8)

        # Генерируем все координаты и длины сразу — ускоряет numpy-вычисления
        xs      = np.random.randint(0, w, n).astype(np.float32)
        ys      = np.random.randint(0, h, n).astype(np.float32)
        lengths = np.random.randint(len_lo, len_hi + 1, n).astype(np.float32)

        # Вектор от центра до каждой капли
        dx_c = xs - cx
        dy_c = ys - cy
        dists = np.sqrt(dx_c ** 2 + dy_c ** 2)
        dists = np.where(dists < 1.0, 1.0, dists)  # избегаем деления на 0

        # Единичный вектор радиального направления (от центра наружу)
        nx = dx_c / dists
        ny = dy_c / dists

        # Угол наклона от вертикали: 0° в центре, до 30° у края
        tilts = np.deg2rad(30.0) * np.minimum(dists / max_dist, 1.0)

        # Смещение конца капли:
        #   вертикальная составляющая: length * cos(tilt)
        #   радиальная составляющая:   length * sin(tilt) * (-nx, -ny)
        #   минус — капля наклонена К центру, а не от него
        sin_t, cos_t = np.sin(tilts), np.cos(tilts)
        end_dx = (lengths * sin_t * (-nx)).astype(np.int32)
        end_dy = (lengths * cos_t + lengths * sin_t * (-ny)).astype(np.int32)

        x2s = np.clip(xs.astype(np.int32) + end_dx, 0, w - 1)
        y2s = np.clip(ys.astype(np.int32) + end_dy, 0, h - 1)
        x1s = xs.astype(np.int32)
        y1s = ys.astype(np.int32)

        # Чередуем чисто белый и голубоватый цвет
        use_blue = np.random.random(n) < 0.5

        for i in range(n):
            p1 = (x1s[i], y1s[i])
            p2 = (x2s[i], y2s[i])
            color = (160, 160, 190) if use_blue[i] else (170, 170, 170)
            # Серый контур (+1 px) — виден на светлом фоне
            cv2.line(layer, p1, p2, (60, 60, 60), thick + 1)
            # Приглушённая линия поверх
            cv2.line(layer, p1, p2, color, thick)

        return layer

    # Крупные капли: очень сильный blur — полупрозрачные «пятна» воды
    layer_large = _make_layer(n_large, 100, 190, 10)
    layer_large = cv2.GaussianBlur(layer_large, (51, 51), sigmaX=0)

    # Средние капли: заметный blur — размытые штрихи
    layer_medium = _make_layer(n_medium, 45, 95, 5)
    layer_medium = cv2.GaussianBlur(layer_medium, (21, 21), sigmaX=0)

    # Мелкие капли: лёгкий blur — далёкие капли тоже не чёткие
    layer_small = _make_layer(n_small, 5, 45, 1)
    layer_small = cv2.GaussianBlur(layer_small, (7, 7), sigmaX=0)

    rain_layer = cv2.add(cv2.add(layer_large, layer_medium), layer_small)
    result = cv2.addWeighted(image, 1.0, rain_layer, 0.38, 0)

    # Влажный объектив: лёгкое размытие всего кадра
    result = cv2.GaussianBlur(result, (3, 3), sigmaX=0)

    return _clip(result)


# ---------------------------------------------------------------------------
# Аугментация 3: гауссовский шум
# ---------------------------------------------------------------------------

def add_noise(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Добавляет гауссовский шум поверх изображения.

    Шум генерируется как нормально распределённые случайные значения
    с нулевым средним и стандартным отклонением, пропорциональным intensity.
    Имитирует зернистость сенсора камеры при плохом освещении.

    Args:
        image:     исходное изображение (H, W, 3), uint8, BGR.
        intensity: сила шума от 0.0 (без изменений) до 1.0 (сильный шум).

    Returns:
        Аугментированное изображение того же размера и типа.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))

    # Стандартное отклонение шума: от 0 до 50 уровней яркости.
    # При intensity=0.5 sigma≈25 — заметный, но реалистичный шум.
    sigma = intensity * 50.0

    # Генерируем шум той же формы, что и изображение
    noise = np.random.normal(0, sigma, image.shape)

    # Прибавляем шум к изображению и ограничиваем диапазон
    noisy = image.astype(np.float32) + noise

    return _clip(noisy)


# ---------------------------------------------------------------------------
# Аугментация 4: размытие
# ---------------------------------------------------------------------------

def add_blur(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Применяет Gaussian blur — размытие движением или расфокусировкой.

    Размер ядра размытия растёт пропорционально intensity.
    Ядро всегда нечётное (требование OpenCV GaussianBlur).

    Args:
        image:     исходное изображение (H, W, 3), uint8, BGR.
        intensity: сила размытия от 0.0 (без изменений) до 1.0 (сильное размытие).

    Returns:
        Аугментированное изображение того же размера и типа.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))

    if intensity == 0.0:
        return image.copy()

    # Размер ядра: от 3 до 31 пикселя.
    # Вычисляем нечётное целое: 2 * int(…) + 1 гарантирует нечётность.
    max_ksize = 31
    ksize = 2 * int(intensity * (max_ksize // 2)) + 1
    ksize = max(ksize, 3)  # минимальное допустимое ядро для GaussianBlur

    # sigma=0 означает автовычисление из размера ядра (стандартное поведение)
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=0)

    return blurred


# ---------------------------------------------------------------------------
# Аугментация 5: изменение яркости
# ---------------------------------------------------------------------------

def add_brightness(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Изменяет яркость изображения — темнее или светлее.

    intensity < 0.5 → затемнение (имитация ночи / пасмурной погоды)
    intensity > 0.5 → засветка (имитация яркого солнца / бликов)
    intensity = 0.5 → изображение без изменений

    Для плавного изменения яркости используется цветовое пространство HSV:
    канал V (Value) умножается на коэффициент, что сохраняет оттенок и насыщенность.

    Args:
        image:     исходное изображение (H, W, 3), uint8, BGR.
        intensity: от 0.0 (максимальное затемнение) до 1.0 (максимальная засветка).
                   0.5 — нейтральное значение.

    Returns:
        Аугментированное изображение того же размера и типа.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))

    # Переводим в HSV для работы только с каналом яркости
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Вычисляем коэффициент яркости:
    #   intensity=0.0 → factor=0.2 (очень тёмное)
    #   intensity=0.5 → factor=1.0 (без изменений)
    #   intensity=1.0 → factor=2.0 (очень яркое)
    factor = 0.2 + intensity * 1.8

    # Умножаем канал V (яркость) на коэффициент
    hsv[:, :, 2] *= factor

    # Конвертируем обратно в BGR
    hsv = _clip(hsv)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result


# ---------------------------------------------------------------------------
# Реестр аугментаций (заполняем после определения функций)
# ---------------------------------------------------------------------------

AUGMENTATIONS = {
    "fog":        add_fog,
    "rain":       add_rain,
    "noise":      add_noise,
    "blur":       add_blur,
    "brightness": add_brightness,
}

# ---------------------------------------------------------------------------
# Вспомогательные функции для augment_dataset
# ---------------------------------------------------------------------------

def _collect_source_frames(source_dir: Path) -> List[Path]:
    """Собирает оригинальные кадры из папки источника.

    Исключает файлы, которые сами являются результатом аугментации
    (их имя заканчивается на _fog, _rain, _noise, _blur, _brightness).
    Это важно при повторных запусках — чтобы не аугментировать аугментации.

    Args:
        source_dir: папка с кадрами одного источника (real или airsim).

    Returns:
        Отсортированный список путей к оригинальным кадрам.
    """
    suffixes = set(AUGMENTATIONS.keys())
    frames = []

    for p in sorted(source_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        # Проверяем, не является ли файл уже аугментированной копией.
        # Оригинальные файлы имеют формат: {source}_{video}_frame_{N}.ext
        # Аугментированные: {source}_{video}_frame_{N}_{aug_type}.ext
        stem_parts = p.stem.rsplit("_", 1)
        if len(stem_parts) == 2 and stem_parts[-1] in suffixes:
            continue  # пропускаем аугментированные копии

        frames.append(p)

    return frames


def _find_annotation(frame_path: Path, sources_map: dict) -> Optional[Path]:
    """Ищет txt-файл разметки для заданного кадра.

    Перебирает все подпапки ANNOTATIONS_DIR и ищет файл с именем кадра.
    Не зависит от префикса имени файла — работает с любыми именами.

    Args:
        frame_path:  путь к кадру.
        sources_map: словарь {имя_источника: Path(frames_dir)} — используется
                     для определения списка подпапок аннотаций.

    Returns:
        Путь к txt-файлу аннотации или None, если файл не найден.
    """
    target = frame_path.stem + ".txt"

    # Сначала ищем в подпапках, соответствующих известным источникам
    for source_name in sources_map:
        ann_file = config.ANNOTATIONS_DIR / source_name / target
        if ann_file.exists():
            return ann_file

    # Затем перебираем все остальные подпапки ANNOTATIONS_DIR
    if config.ANNOTATIONS_DIR.exists():
        for subdir in config.ANNOTATIONS_DIR.iterdir():
            if not subdir.is_dir():
                continue
            ann_file = subdir / target
            if ann_file.exists():
                return ann_file

    return None


# ---------------------------------------------------------------------------
# Главная функция модуля
# ---------------------------------------------------------------------------

def augment_dataset(
    augmentation_types: List[str],
    intensity: float = 0.5,
    sources: List[str] = None,
    overwrite: bool = False,
) -> dict:
    """Применяет аугментации ко всем кадрам датасета.

    Для каждого оригинального кадра и каждого типа аугментации создаётся
    новый файл рядом с оригиналом:
        {оригинальное_имя}_{тип_аугментации}.jpg

    Если для кадра есть файл разметки (txt), он копируется с аналогичным именем —
    геометрические координаты объектов при пиксельных аугментациях не меняются.

    Args:
        augmentation_types: список типов аугментации из набора:
                            ["fog", "rain", "noise", "blur", "brightness"].
        intensity:          единая сила эффекта для всех типов [0.0, 1.0].
        sources:            список источников ("real", "airsim").
                            По умолчанию — все существующие.
        overwrite:          если True, перезаписывать уже созданные копии.

    Returns:
        Словарь со статистикой:
        {"processed_frames": int, "created": int, "skipped": int, "errors": int}

    Raises:
        ValueError: если передан неизвестный тип аугментации.

    Пример использования:
        >>> from modules.augmentor import augment_dataset
        >>> stats = augment_dataset(["fog", "rain"], intensity=0.4)
        >>> print(stats)  # {"processed_frames": 125, "created": 250, ...}
    """
    # Проверяем корректность переданных типов аугментации
    unknown = set(augmentation_types) - set(AUGMENTATIONS)
    if unknown:
        raise ValueError(
            f"Неизвестные типы аугментации: {unknown}. "
            f"Допустимые: {list(AUGMENTATIONS)}"
        )

    # Убираем дубликаты, сохраняя порядок
    augmentation_types = list(dict.fromkeys(augmentation_types))

    # Карта источников: имя → папка с кадрами
    sources_map = {
        "real": config.FRAMES_REAL_DIR,
        "airsim": config.FRAMES_AIRSIM_DIR,
    }
    if sources is not None:
        sources_map = {k: v for k, v in sources_map.items() if k in sources}

    logger.info("=" * 50)
    logger.info(f"Augmentor | аугментации={augmentation_types} | intensity={intensity}")
    logger.info("=" * 50)

    processed_frames = 0  # оригинальных кадров обработано
    created = 0           # новых аугментированных файлов создано
    skipped = 0           # пропущено (уже существуют, overwrite=False)
    errors = 0            # ошибок при чтении/записи

    for source_name, frames_dir in sources_map.items():
        if not frames_dir.exists():
            logger.warning(f"Папка источника не найдена: {frames_dir} — пропускаем '{source_name}'")
            continue

        original_frames = _collect_source_frames(frames_dir)

        if not original_frames:
            logger.warning(f"Источник '{source_name}': оригинальные кадры не найдены в {frames_dir}")
            continue

        logger.info(f"Источник '{source_name}': {len(original_frames)} оригинальных кадров")

        for frame_path in original_frames:
            # Читаем изображение через _imread (поддержка кириллики в пути)
            image = _imread(frame_path)
            if image is None:
                logger.warning(f"Не удалось прочитать: {frame_path.name} — пропускаем")
                errors += 1
                continue

            processed_frames += 1

            # Ищем соответствующий файл разметки (может отсутствовать — это нормально)
            ann_src = _find_annotation(frame_path, sources_map)

            for aug_type in augmentation_types:
                # Формируем имя нового файла: оригинальное_имя + _тип.jpg
                aug_stem = f"{frame_path.stem}_{aug_type}"
                aug_path = frame_path.parent / f"{aug_stem}.jpg"

                image_exists = aug_path.exists()

                # Пропускаем картинку если она уже создана и перезапись не нужна
                if image_exists and not overwrite:
                    skipped += 1
                else:
                    try:
                        aug_func = AUGMENTATIONS[aug_type]
                        aug_image = aug_func(image, intensity)

                        # Ресайз до целевого разрешения перед сохранением
                        if config.AUGMENT_RESIZE:
                            aug_image = cv2.resize(
                                aug_image,
                                (config.TARGET_WIDTH, config.TARGET_HEIGHT),
                                interpolation=cv2.INTER_AREA,
                            )

                        _imwrite(aug_path, aug_image, quality=config.AUGMENT_QUALITY)
                        created += 1
                    except Exception as exc:
                        logger.error(f"Ошибка при '{aug_type}' для {frame_path.name}: {exc}")
                        errors += 1
                        continue

                # Копируем txt разметки независимо от того, была ли картинка создана
                # сейчас или уже существовала — это позволяет «дополнить» ранее
                # созданные аугментации недостающими аннотациями.
                if ann_src is not None:
                    ann_dst = ann_src.parent / f"{aug_stem}.txt"
                    if not ann_dst.exists() or overwrite:
                        try:
                            shutil.copy2(str(ann_src), str(ann_dst))
                        except Exception as exc:
                            logger.error(f"Ошибка копирования txt для {aug_stem}: {exc}")

        logger.info(
            f"Источник '{source_name}' завершён: "
            f"кадров={len(original_frames)}, "
            f"создано={created}, пропущено={skipped}"
        )

    # Итоговый отчёт
    stats = {
        "processed_frames": processed_frames,
        "created": created,
        "skipped": skipped,
        "errors": errors,
    }

    logger.info("=" * 50)
    logger.info("ИТОГОВЫЙ ОТЧЁТ:")
    logger.info(f"  Обработано оригинальных кадров : {processed_frames}")
    logger.info(f"  Создано аугментированных файлов: {created}")
    logger.info(f"  Пропущено (уже существуют)     : {skipped}")
    logger.info(f"  Ошибок                         : {errors}")
    logger.info("=" * 50)

    return stats
