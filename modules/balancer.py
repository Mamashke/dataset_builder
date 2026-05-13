# modules/balancer.py — модуль фильтрации и балансировки датасета.
#
# Контекст: конструктор обучающей выборки для YOLO-моделей детекции БПЛА.
# Источники данных: кадры из реальных mp4-видео + синтетика из AirSim.
#
# Пайплайн (запускается через build()):
#   1. filter_frames()    — отбрасывает тёмные, размытые и битые кадры
#   2. balance_dataset()  — выравнивает соотношение позитивных/негативных примеров
#   3. collect_dataset()  — копирует итоговый набор в data/processed/dataset/
#
# Публичный интерфейс:
#   filter_frames(frames)       → List[Path]
#   balance_dataset(frames)     → List[Path]
#   collect_dataset(frames)     → dict
#   build(sources, overwrite)   → dict

import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from modules.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Карта источников: имя → папка с кадрами
# ---------------------------------------------------------------------------

SOURCES = {
    "real": config.FRAMES_REAL_DIR,
    "airsim": config.FRAMES_AIRSIM_DIR,
}

# Допустимые расширения изображений
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _imread(path: Path) -> Optional[np.ndarray]:
    """Читает изображение с поддержкой кириллических путей на Windows.

    cv2.imread не работает с не-ASCII путями — обходим через np.fromfile.

    Args:
        path: путь к файлу изображения.

    Returns:
        BGR-массив (H, W, 3) или None при ошибке чтения.
    """
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _find_annotation(frame_path: Path) -> Optional[Path]:
    """Ищет txt-файл разметки для кадра по имени файла.

    Перебирает все подпапки ANNOTATIONS_DIR и ищет файл с именем кадра.
    Не зависит от префикса — работает с любыми именами файлов (real_, airsim_, 01_ и т.д.).

    Args:
        frame_path: путь к файлу кадра.

    Returns:
        Путь к txt-файлу аннотации или None, если он не найден.
    """
    target = frame_path.stem + ".txt"

    # Сначала ищем в подпапках известных источников (быстрее для типичного случая)
    for source_name in SOURCES:
        ann = config.ANNOTATIONS_DIR / source_name / target
        if ann.exists():
            return ann

    # Затем перебираем все остальные подпапки ANNOTATIONS_DIR
    if config.ANNOTATIONS_DIR.exists():
        for subdir in config.ANNOTATIONS_DIR.iterdir():
            if not subdir.is_dir():
                continue
            ann = subdir / target
            if ann.exists():
                return ann

    return None


def _is_positive(ann_path: Optional[Path]) -> bool:
    """Определяет, является ли кадр позитивным примером.

    Позитивный кадр — тот, для которого существует непустой txt-файл разметки
    (т.е. на кадре есть хотя бы один размеченный объект).

    Args:
        ann_path: путь к файлу аннотации или None.

    Returns:
        True если кадр содержит объекты, False если негативный.
    """
    if ann_path is None:
        return False
    content = ann_path.read_text(encoding="utf-8").strip()
    return len(content) > 0


def _collect_all_frames(sources: List[str]) -> List[Path]:
    """Собирает все кадры изображений из указанных источников.

    Включает как оригинальные кадры, так и аугментированные копии —
    они участвуют в фильтрации и балансировке наравне с оригиналами.

    Args:
        sources: список имён источников ("real", "airsim").

    Returns:
        Отсортированный список путей ко всем найденным кадрам.
    """
    result = []
    for source in sources:
        if source not in SOURCES:
            logger.warning(f"Неизвестный источник '{source}', пропускаем.")
            continue
        frames_dir = SOURCES[source]
        if not frames_dir.exists():
            logger.warning(f"Папка не найдена: {frames_dir} — пропускаем '{source}'")
            continue
        frames = [
            p for p in sorted(frames_dir.iterdir())
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        result.extend(frames)
        logger.info(f"Источник '{source}': найдено {len(frames)} кадров")
    return result


# ---------------------------------------------------------------------------
# 1. Фильтрация некачественных кадров
# ---------------------------------------------------------------------------

def _check_brightness(gray: np.ndarray) -> Tuple[bool, float]:
    """Проверяет среднюю яркость кадра.

    Args:
        gray: одноканальное (grayscale) изображение.

    Returns:
        (прошёл_проверку, значение_яркости)
    """
    brightness = float(np.mean(gray))
    return brightness >= config.MIN_BRIGHTNESS, brightness


def _check_blur(gray: np.ndarray) -> Tuple[bool, float]:
    """Проверяет резкость кадра методом Лапласа (Variance of Laplacian).

    Оператор Лапласа выделяет края — резкое изображение даёт высокую дисперсию,
    размытое — низкую. Порог задаётся константой BLUR_THRESHOLD в config.py.

    Args:
        gray: одноканальное (grayscale) изображение.

    Returns:
        (прошёл_проверку, значение_дисперсии_лапласиана)
    """
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return laplacian_var >= config.BLUR_THRESHOLD, laplacian_var


def filter_frames(frames: List[Path]) -> List[Path]:
    """Фильтрует некачественные кадры из набора.

    Отбрасывает:
    - битые/нечитаемые файлы изображений
    - слишком тёмные кадры (средняя яркость < MIN_BRIGHTNESS)
    - слишком размытые кадры (variance of Laplacian < BLUR_THRESHOLD)

    Args:
        frames: список путей к кадрам для проверки.

    Returns:
        Список путей к кадрам, прошедшим все проверки.
    """
    passed = []
    rejected_broken = 0
    rejected_dark = 0
    rejected_blur = 0

    total = len(frames)

    for i, frame_path in enumerate(frames, start=1):
        # --- Проверка 1: читаемость файла ---
        image = _imread(frame_path)
        if image is None:
            logger.debug(f"[{i}/{total}] БИТЫЙ: {frame_path.name}")
            rejected_broken += 1
            continue

        # Переводим в grayscale — все метрики качества считаются по яркости
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # --- Проверка 2: яркость ---
        bright_ok, brightness = _check_brightness(gray)
        if not bright_ok:
            logger.debug(
                f"[{i}/{total}] ТЁМНЫЙ (brightness={brightness:.1f} < {config.MIN_BRIGHTNESS}): "
                f"{frame_path.name}"
            )
            rejected_dark += 1
            continue

        # --- Проверка 3: резкость ---
        # _blur-аугментации намеренно размыты — фильтр к ним не применяем
        is_blur_aug = frame_path.stem.endswith("_blur")
        if not is_blur_aug:
            sharp_ok, lap_var = _check_blur(gray)
            if not sharp_ok:
                logger.debug(
                    f"[{i}/{total}] РАЗМЫТЫЙ (laplacian={lap_var:.1f} < {config.BLUR_THRESHOLD}): "
                    f"{frame_path.name}"
                )
                rejected_blur += 1
                continue

        passed.append(frame_path)

    logger.info(
        f"Фильтрация: {total} → {len(passed)} кадров "
        f"(отброшено: битых={rejected_broken}, "
        f"тёмных={rejected_dark}, размытых={rejected_blur})"
    )

    return passed


# ---------------------------------------------------------------------------
# 2. Балансировка классов
# ---------------------------------------------------------------------------

def balance_dataset(frames: List[Path]) -> List[Path]:
    """Балансирует соотношение позитивных и негативных примеров.

    Делит кадры на:
    - позитивные: непустой txt-файл разметки (есть хотя бы один объект)
    - негативные: пустой или отсутствующий txt-файл (чистый фон)

    Если негативных примеров больше, чем позитивных × POS_NEG_RATIO,
    случайно отбирает нужное количество негативных (undersampling).
    Позитивные примеры не уменьшаются — их обычно меньше.

    Args:
        frames: список путей к кадрам (после фильтрации).

    Returns:
        Сбалансированный список кадров.
    """
    positives = []
    negatives = []

    for frame_path in frames:
        ann_path = _find_annotation(frame_path)
        if _is_positive(ann_path):
            positives.append(frame_path)
        else:
            negatives.append(frame_path)

    logger.info(
        f"До балансировки: позитивных={len(positives)}, "
        f"негативных={len(negatives)}, "
        f"целевое соотношение 1:{config.POS_NEG_RATIO}"
    )

    # Максимально допустимое количество негативных примеров
    max_negatives = len(positives) * config.POS_NEG_RATIO

    if len(negatives) > max_negatives:
        # Фиксируем seed для воспроизводимости — одинаковый датасет при повторном запуске
        random.seed(42)
        negatives = random.sample(negatives, max_negatives)
        logger.info(
            f"Негативных обрезано до {max_negatives} "
            f"({len(positives)} pos × {config.POS_NEG_RATIO})"
        )
    else:
        logger.info("Балансировка не требуется — негативных в пределах нормы.")

    balanced = positives + negatives
    logger.info(
        f"После балансировки: позитивных={len(positives)}, "
        f"негативных={len(negatives)}, итого={len(balanced)}"
    )

    return balanced


# ---------------------------------------------------------------------------
# 3. Сборка финального датасета
# ---------------------------------------------------------------------------

def collect_dataset(frames: List[Path], overwrite: bool = False) -> dict:
    """Копирует отфильтрованные и сбалансированные кадры в папку датасета.

    Структура вывода:
        data/processed/dataset/images/  — изображения
        data/processed/dataset/labels/  — txt-файлы разметки

    Для кадров без txt-аннотации создаётся пустой txt-файл —
    YOLO интерпретирует его как «негативный пример» (фон без объектов).

    Args:
        frames:    список путей к кадрам (результат balance_dataset).
        overwrite: если True, перезаписывать уже существующие файлы.

    Returns:
        Словарь со статистикой:
        {"copied": int, "skipped": int, "labels_created": int, "labels_copied": int}
    """
    images_dir = config.DATASET_IMAGES_DIR
    labels_dir = config.DATASET_LABELS_DIR

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    copied = 0          # изображений скопировано
    skipped = 0         # пропущено (уже существуют, overwrite=False)
    labels_copied = 0   # txt-файлов скопировано из аннотаций
    labels_created = 0  # пустых txt-файлов создано (негативные примеры)

    total = len(frames)

    for i, frame_path in enumerate(frames, start=1):
        dst_img = images_dir / frame_path.name
        dst_lbl = labels_dir / (frame_path.stem + ".txt")

        # Пропускаем уже существующие файлы, если перезапись не нужна
        if dst_img.exists() and not overwrite:
            skipped += 1
            continue

        # Копируем изображение
        shutil.copy2(str(frame_path), str(dst_img))
        copied += 1

        # Ищем соответствующую аннотацию
        ann_src = _find_annotation(frame_path)

        if ann_src is not None:
            # Копируем существующий txt-файл разметки
            shutil.copy2(str(ann_src), str(dst_lbl))
            labels_copied += 1
        else:
            # Создаём пустой txt — YOLO-стандарт для негативного примера
            dst_lbl.write_text("", encoding="utf-8")
            labels_created += 1

        if i % 100 == 0 or i == total:
            logger.info(f"  Скопировано {i}/{total} кадров...")

    logger.info(
        f"Сборка датасета завершена: "
        f"изображений={copied}, "
        f"меток скопировано={labels_copied}, "
        f"меток создано={labels_created}, "
        f"пропущено={skipped}"
    )

    return {
        "copied": copied,
        "skipped": skipped,
        "labels_copied": labels_copied,
        "labels_created": labels_created,
    }


# ---------------------------------------------------------------------------
# 4. Главная функция пайплайна
# ---------------------------------------------------------------------------

def build(sources: List[str] = None, overwrite: bool = False) -> dict:
    """Запускает полный пайплайн сборки датасета.

    Шаги:
        1. Собирает все кадры из указанных источников.
        2. filter_frames()    — фильтрует некачественные кадры.
        3. balance_dataset()  — балансирует позитивные/негативные примеры.
        4. collect_dataset()  — копирует финальный датасет на диск.

    Args:
        sources:   список источников для обработки ("real", "airsim").
                   По умолчанию — все зарегистрированные.
        overwrite: если True, перезаписывать уже существующие файлы датасета.

    Returns:
        Полный отчёт со статистикой всех этапов:
        {
            "total_input":    int,  # кадров на входе
            "after_filter":   int,  # после фильтрации
            "after_balance":  int,  # после балансировки
            "positives":      int,  # позитивных в финальном датасете
            "negatives":      int,  # негативных в финальном датасете
            "copied":         int,  # скопировано в dataset/images/
            "skipped":        int,
            "labels_copied":  int,
            "labels_created": int,
        }

    Пример использования:
        >>> from modules.balancer import build
        >>> report = build(sources=["real"], overwrite=True)
        >>> print(report)
    """
    if sources is None:
        sources = list(SOURCES.keys())

    logger.info("=" * 50)
    logger.info(f"Balancer | источники={sources} | перезапись={overwrite}")
    logger.info(f"Пороги: MIN_BRIGHTNESS={config.MIN_BRIGHTNESS}, "
                f"BLUR_THRESHOLD={config.BLUR_THRESHOLD}, "
                f"POS_NEG_RATIO=1:{config.POS_NEG_RATIO}")
    logger.info("=" * 50)

    # --- Шаг 1: сбор всех кадров ---
    all_frames = _collect_all_frames(sources)
    total_input = len(all_frames)

    if not all_frames:
        logger.warning("Кадры не найдены. Проверьте папки источников.")
        return {}

    logger.info(f"Всего кадров на входе: {total_input}")

    # --- Шаг 2: фильтрация ---
    logger.info("-" * 50)
    logger.info("Шаг 1/3: фильтрация некачественных кадров")
    filtered = filter_frames(all_frames)

    # --- Шаг 3: балансировка ---
    logger.info("-" * 50)
    logger.info("Шаг 2/3: балансировка позитивных/негативных примеров")
    balanced = balance_dataset(filtered)

    # Считаем финальное распределение для отчёта
    final_positives = sum(
        1 for p in balanced if _is_positive(_find_annotation(p))
    )
    final_negatives = len(balanced) - final_positives

    # --- Шаг 4: сборка датасета ---
    logger.info("-" * 50)
    logger.info("Шаг 3/3: сборка финального датасета")
    collect_stats = collect_dataset(balanced, overwrite=overwrite)

    # --- Итоговый отчёт ---
    report = {
        "total_input": total_input,
        "after_filter": len(filtered),
        "after_balance": len(balanced),
        "positives": final_positives,
        "negatives": final_negatives,
        **collect_stats,
    }

    logger.info("=" * 50)
    logger.info("ИТОГОВЫЙ ОТЧЁТ:")
    logger.info(f"  Кадров на входе          : {report['total_input']}")
    logger.info(f"  После фильтрации         : {report['after_filter']}"
                f"  (отброшено {total_input - report['after_filter']})")
    logger.info(f"  После балансировки       : {report['after_balance']}"
                f"  (отброшено {report['after_filter'] - report['after_balance']})")
    logger.info(f"  Позитивных примеров      : {report['positives']}")
    logger.info(f"  Негативных примеров      : {report['negatives']}")
    logger.info(f"  Скопировано изображений  : {report['copied']}")
    logger.info(f"  Меток скопировано        : {report['labels_copied']}")
    logger.info(f"  Меток создано (пустых)   : {report['labels_created']}")
    logger.info(f"  Пропущено (существуют)   : {report['skipped']}")
    logger.info(f"  Датасет: {config.DATASET_IMAGES_DIR}")
    logger.info("=" * 50)

    return report
