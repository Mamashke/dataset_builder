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

import config  # Централизованные настройки (AUGMENT_RESIZE, AUGMENT_QUALITY, TARGET_*)
from modules.logger import get_logger
from modules.project import Project

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

    def _odd(n: int) -> int:
        """Возвращает ближайшее нечётное число >= 3 (требование GaussianBlur)."""
        n = max(3, n)
        return n if n % 2 == 1 else n + 1

    # Фиксированное число капель — одинаковая «наполненность» кадра
    # при любом разрешении; размеры капель уже масштабируются через h
    num_drops = np.random.randint(2000, 2501)
    n_large   = int(num_drops * 0.15)
    n_medium  = int(num_drops * 0.55)
    n_small   = num_drops - n_large - n_medium

    # Размеры капель и blur-ядер — относительно высоты кадра,
    # чтобы эффект не зависел от разрешения исходного видео
    lo_large,  hi_large  = int(h * 0.05),  int(h * 0.09)
    lo_medium, hi_medium = int(h * 0.025), int(h * 0.05)
    lo_small,  hi_small  = int(h * 0.008), int(h * 0.02)

    # Blur-ядра для трёх слоёв, сохраняем исходное соотношение 51:21:7
    k_large  = _odd(int(h * 0.014))         # ≈51 при h=3648, ≈9  при h=640
    k_medium = _odd(int(h * 0.0057))        # ≈21 при h=3648, ≈5  при h=640
    k_small  = _odd(int(h * 0.002))         # ≈7  при h=3648, ≈3  при h=640

    # Толщина штриха тоже масштабируется — иначе на маленьком кадре
    # крупные капли (thick=10) перекрывают всё изображение
    thick_large  = max(1, int(h * 0.003))   # ≈10 при h=3648
    thick_medium = max(1, int(h * 0.0014))  # ≈5  при h=3648
    thick_small  = 1

    def _make_layer(n: int, len_lo: int, len_hi: int, thick: int) -> np.ndarray:
        """Рисует n капель на чёрном слое; радиальный угол вычислен векторно."""
        layer = np.zeros_like(image, dtype=np.uint8)

        # Генерируем все координаты и длины сразу — ускоряет numpy-вычисления
        xs      = np.random.randint(0, w, n).astype(np.float32)
        ys      = np.random.randint(0, h, n).astype(np.float32)
        len_lo  = max(1, len_lo)
        len_hi  = max(len_lo + 1, len_hi)
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

    # Крупные капли: сильный blur — полупрозрачные «пятна» воды
    layer_large = _make_layer(n_large, lo_large, hi_large, thick_large)
    layer_large = cv2.GaussianBlur(layer_large, (k_large, k_large), sigmaX=0)

    # Средние капли: заметный blur — размытые штрихи
    layer_medium = _make_layer(n_medium, lo_medium, hi_medium, thick_medium)
    layer_medium = cv2.GaussianBlur(layer_medium, (k_medium, k_medium), sigmaX=0)

    # Мелкие капли: лёгкий blur — далёкие капли тоже не чёткие
    layer_small = _make_layer(n_small, lo_small, hi_small, thick_small)
    layer_small = cv2.GaussianBlur(layer_small, (k_small, k_small), sigmaX=0)

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

def _collect_source_frames(project: Project, sources: list) -> List[tuple]:
    """Собирает оригинальные кадры из источников проекта через data_sources.

    Для каждого источника:
      - входная папка берётся из project.get_source("frames", source);
      - если путь не задан — источник пропускается с сообщением;
      - выходная папка всегда project.frames_real_dir / frames_airsim_dir.

    Файлы, уже являющиеся результатом аугментации (_fog, _rain и т.д.),
    исключаются, чтобы не аугментировать аугментации при повторных запусках.

    Args:
        project: объект Project — входные пути читаются через data_sources.
        sources: список имён источников ("real", "airsim").

    Returns:
        Список кортежей (frame_path, output_dir, source_name):
          frame_path  — путь к оригинальному кадру во входной папке;
          output_dir  — папка проекта, куда сохранять результат;
          source_name — "real" или "airsim".
    """
    output_dirs = {
        "real":   project.frames_real_dir,
        "airsim": project.frames_airsim_dir,
    }
    suffixes = set(AUGMENTATIONS.keys())
    result = []

    for source in sources:
        # Входная папка берётся из data_sources — может быть внешней
        input_dir = project.get_source("frames", source)

        if input_dir is None:
            print(f"Путь к кадрам не задан для источника '{source}' — пропускаем")
            logger.info(f"Путь к кадрам не задан для источника '{source}' — пропускаем")
            continue

        if not input_dir.exists():
            logger.warning(
                f"Папка с кадрами не найдена: {input_dir} — пропускаем '{source}'"
            )
            continue

        output_dir = output_dirs.get(source, project.frames_real_dir)

        frames = []
        for p in sorted(input_dir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            # Пропускаем файлы, уже являющиеся результатом аугментации
            stem_parts = p.stem.rsplit("_", 1)
            if len(stem_parts) == 2 and stem_parts[-1] in suffixes:
                continue
            frames.append(p)

        for frame_path in frames:
            result.append((frame_path, output_dir, source))

        logger.info(f"Источник '{source}': {len(frames)} кадров → {output_dir}")

    return result


def _find_annotation(
    frame_path: Path,
    annotations_dir: Path,
    sources_map: dict,
) -> Optional[Path]:
    """Ищет txt-файл разметки для заданного кадра в папке аннотаций проекта.

    Сначала проверяет подпапки для каждого известного источника,
    затем — все остальные подпапки annotations_dir.
    Не зависит от префикса имени файла — работает с любыми именами.

    Args:
        frame_path:      путь к кадру.
        annotations_dir: корневая папка аннотаций проекта (project.annotations_dir).
        sources_map:     словарь {имя_источника: Path(frames_dir)} — определяет
                         список подпапок для приоритетного поиска.

    Returns:
        Путь к txt-файлу аннотации или None, если файл не найден.
    """
    target = frame_path.stem + ".txt"

    # Приоритетный поиск: подпапки известных источников (real, airsim)
    for source_name in sources_map:
        ann_file = annotations_dir / source_name / target
        if ann_file.exists():
            return ann_file

    # Резервный поиск: все остальные подпапки папки аннотаций
    if annotations_dir.exists():
        for subdir in annotations_dir.iterdir():
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
    project: Project,
    augmentation_types: List[str],
    intensity: float = 0.5,
    sources: List[str] = None,
    overwrite: bool = False,
) -> dict:
    """Применяет аугментации ко всем кадрам датасета проекта.

    Для каждого оригинального кадра и каждого типа аугментации создаётся
    новый файл рядом с оригиналом:
        {оригинальное_имя}_{тип_аугментации}.jpg

    Если для кадра есть файл разметки (txt), он копируется с аналогичным именем —
    геометрические координаты объектов при пиксельных аугментациях не меняются.

    Args:
        project:            объект Project — определяет пути к кадрам и аннотациям.
        augmentation_types: список типов аугментации из набора:
                            ["fog", "rain", "noise", "blur", "brightness"].
        intensity:          единая сила эффекта для всех типов [0.0, 1.0].
        sources:            список источников ("real", "airsim").
                            По умолчанию — все источники проекта.
        overwrite:          если True, перезаписывать уже созданные копии.

    Returns:
        Словарь со статистикой:
        {"processed_frames": int, "created": int, "skipped": int, "errors": int}

    Raises:
        ValueError: если передан неизвестный тип аугментации.
    """
    # Подключаем файловый лог проекта — все записи logger попадут
    # в project.logs_dir/augmentor.log
    get_logger(__name__, project.logs_dir)

    # Проверяем корректность переданных типов аугментации
    unknown = set(augmentation_types) - set(AUGMENTATIONS)
    if unknown:
        raise ValueError(
            f"Неизвестные типы аугментации: {unknown}. "
            f"Допустимые: {list(AUGMENTATIONS)}"
        )

    # Убираем дубликаты, сохраняя порядок
    augmentation_types = list(dict.fromkeys(augmentation_types))

    # Определяем активные источники с учётом фильтра sources
    active_sources = (
        [s for s in ("real", "airsim") if s in sources]
        if sources is not None
        else ["real", "airsim"]
    )

    logger.info("=" * 50)
    logger.info(
        f"Augmentor | project={project.name} | "
        f"аугментации={augmentation_types} | intensity={intensity}"
    )
    logger.info("=" * 50)

    processed_frames = 0  # оригинальных кадров обработано
    created = 0           # новых аугментированных файлов создано
    skipped = 0           # пропущено (уже существуют, overwrite=False)
    errors = 0            # ошибок при чтении/записи

    # Собираем кадры через data_sources; каждый элемент — (frame_path, output_dir, source)
    all_frames = _collect_source_frames(project, active_sources)

    if not all_frames:
        logger.warning("Кадры не найдены ни в одном источнике.")
        return {"processed_frames": 0, "created": 0, "skipped": 0, "errors": 0}

    # Группируем по источнику для per-source статистики и set_source
    frames_by_source: dict = {}
    source_output_dirs: dict = {}
    for frame_path, output_dir, source_name in all_frames:
        frames_by_source.setdefault(source_name, []).append(frame_path)
        source_output_dirs[source_name] = output_dir

    for source_name, original_frames in frames_by_source.items():
        output_dir = source_output_dirs[source_name]
        output_dir.mkdir(parents=True, exist_ok=True)

        # Путь к аннотациям берём из data_sources; None → копирование пропускается
        ann_dir = project.get_source("annotations", source_name)
        if ann_dir is None:
            logger.info(
                f"Источник '{source_name}': путь к аннотациям не задан — "
                "копирование аннотаций пропускается"
            )

        logger.info(f"Источник '{source_name}': {len(original_frames)} оригинальных кадров")

        source_created = 0
        source_skipped = 0

        for frame_path in original_frames:
            # Читаем изображение через _imread (поддержка кириллики в пути)
            image = _imread(frame_path)
            if image is None:
                logger.warning(f"Не удалось прочитать: {frame_path.name} — пропускаем")
                errors += 1
                continue

            processed_frames += 1

            # Ищем txt-аннотацию для кадра в папке аннотаций источника
            ann_src = None
            if ann_dir is not None:
                candidate = ann_dir / (frame_path.stem + ".txt")
                if candidate.exists():
                    ann_src = candidate

            # Ресайз оригинала и сохранение в папку проекта (сжатый оригинал без суффикса)
            if config.AUGMENT_RESIZE:
                orig_resized = cv2.resize(
                    image,
                    (config.TARGET_WIDTH, config.TARGET_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                orig_resized = image.copy()

            orig_dst = output_dir / f"{frame_path.stem}.jpg"
            if not orig_dst.exists() or overwrite:
                _imwrite(orig_dst, orig_resized, quality=config.AUGMENT_QUALITY)

            for aug_type in augmentation_types:
                # Имя аугментированного файла: оригинальное_имя + _тип.jpg
                aug_stem = f"{frame_path.stem}_{aug_type}"
                aug_path = output_dir / f"{aug_stem}.jpg"

                if aug_path.exists() and not overwrite:
                    skipped += 1
                    source_skipped += 1
                else:
                    try:
                        aug_func = AUGMENTATIONS[aug_type]
                        # Аугментация на оригинальном разрешении
                        aug_image = aug_func(image, intensity)
                        # Ресайз после аугментации — в целевое разрешение
                        if config.AUGMENT_RESIZE:
                            aug_image = cv2.resize(
                                aug_image,
                                (config.TARGET_WIDTH, config.TARGET_HEIGHT),
                                interpolation=cv2.INTER_AREA,
                            )
                        _imwrite(aug_path, aug_image, quality=config.AUGMENT_QUALITY)
                        created += 1
                        source_created += 1
                    except Exception as exc:
                        logger.error(f"Ошибка при '{aug_type}' для {frame_path.name}: {exc}")
                        errors += 1
                        continue

                # Копируем txt разметки (координаты не меняются при пиксельных аугментациях)
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
            f"создано={source_created}, пропущено={source_skipped}"
        )

        # Фиксируем путь к кадрам проекта в data_sources
        project.set_source("frames", source_name, output_dir)

    # Итоговый отчёт
    stats = {
        "processed_frames": processed_frames,
        "created":          created,
        "skipped":          skipped,
        "errors":           errors,
    }

    logger.info("=" * 50)
    logger.info("ИТОГОВЫЙ ОТЧЁТ:")
    logger.info(f"  Обработано оригинальных кадров : {processed_frames}")
    logger.info(f"  Создано аугментированных файлов: {created}")
    logger.info(f"  Пропущено (уже существуют)     : {skipped}")
    logger.info(f"  Ошибок                         : {errors}")
    logger.info("=" * 50)

    # Обновляем метаданные проекта
    project.update_step("augment")
    project.update_stats({"augmented_frames": created})

    return stats
