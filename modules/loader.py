# modules/loader.py — модуль загрузки и первичной обработки видеоданных.
#
# Основная задача: принять объект Project, извлечь из каждого видео кадры
# с заданным шагом, привести их к единому разрешению и сохранить
# в соответствующую папку внутри проекта.
#
# Публичный интерфейс модуля — единственная функция load_videos().
# Вспомогательные функции (_extract_frames, _select_videos) намеренно скрыты
# (префикс _), чтобы не засорять пространство имён при импорте.

from pathlib import Path

import cv2  # OpenCV — основная библиотека для работы с видео и изображениями

import config  # Централизованные настройки и пути проекта
from modules.logger import get_logger
from modules.project import Project

logger = get_logger(__name__)

# Глобальный перехватчик выбора видео.
# Если задан внешним кодом (например, GUI) — вызывается вместо интерактивного меню.
# Сигнатура: func(videos: list[Path]) -> list[Path]
_video_selector_override = None


# ---------------------------------------------------------------------------
# Вспомогательная функция: извлечение кадров из одного видеофайла
# ---------------------------------------------------------------------------

def _extract_frames(
    video_path: Path,   # путь к конкретному mp4-файлу
    source: str,        # имя источника (нужно для формирования имён файлов)
    output_dir: Path,   # куда сохранять кадры
    sample_rate: int,   # сохранять каждый N-й кадр
    width: int,         # целевая ширина кадра в пикселях
    height: int,        # целевая высота кадра в пикселях
    fmt: str,           # расширение выходного файла (jpg / png)
) -> int:
    """Извлекает кадры из одного видеофайла и сохраняет их на диск.

    Перебирает все кадры видео; каждый кадр с индексом, кратным sample_rate,
    масштабируется до (width × height) и записывается как отдельное изображение.

    Args:
        video_path:  путь к обрабатываемому mp4-файлу.
        source:      "real" или "airsim" — используется в имени выходного файла.
        output_dir:  директория для сохранения кадров.
        sample_rate: шаг выборки (каждый N-й кадр сохраняется).
        width:       ширина выходного изображения.
        height:      высота выходного изображения.
        fmt:         формат файла ("jpg" или "png").

    Returns:
        Количество сохранённых кадров (0, если видео не удалось открыть).
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path.name} — skipping")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_stem = video_path.stem

    saved = 0
    frame_idx = 0

    logger.info(f"Processing '{video_path.name}' ({total_frames} frames, step={sample_rate})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            filename = f"{source}_{video_stem}_frame_{frame_idx:06d}.{fmt}"
            cv2.imwrite(str(output_dir / filename), resized)
            saved += 1

        frame_idx += 1

    cap.release()

    logger.info(f"  Saved {saved} frames from '{video_path.name}'")
    return saved


# ---------------------------------------------------------------------------
# Вспомогательная функция: интерактивный выбор видеофайлов
# ---------------------------------------------------------------------------

def _select_videos(videos: list) -> list:
    """Предлагает пользователю выбрать файлы для обработки.

    Три сценария:
    - 0 файлов: сообщает об отсутствии и возвращает пустой список.
    - 1 файл:   запрашивает подтверждение перед обработкой.
    - N файлов: печатает нумерованный список, запрашивает ввод в цикле
                до получения корректного ответа.

    Args:
        videos: отсортированный список Path-объектов mp4-файлов.

    Returns:
        Список Path-объектов, выбранных для обработки.
    """
    # Если задан внешний перехватчик (GUI) — делегируем выбор ему
    if _video_selector_override is not None:
        return _video_selector_override(videos)

    if not videos:
        return []

    if len(videos) == 1:
        v = videos[0]
        size_mb = round(v.stat().st_size / (1024 * 1024))
        print(f"\nНайден 1 файл: {v.name} ({size_mb} МБ)")
        while True:
            raw = input("Начать обработку? (yes/no): ").strip().lower()
            if raw == "yes":
                logger.info(f"Пользователь подтвердил обработку: {v.name}")
                return videos
            if raw == "no":
                logger.info("Пользователь отказался от обработки.")
                return []
            print("  Введите 'yes' или 'no'.")

    # --- Несколько файлов: показываем меню выбора ---
    n = len(videos)
    print(f"\nНайдено {n} видеофайла:" if n in (2, 3, 4) else f"\nНайдено {n} видеофайлов:")
    for i, v in enumerate(videos, start=1):
        size_mb = round(v.stat().st_size / (1024 * 1024))
        print(f"  {i}. {v.name} ({size_mb} МБ)")

    while True:
        raw = input('\nКакие файлы обработать? (all / номера через запятую: 1,2): ').strip()

        if raw.lower() == "all":
            logger.info(f"Пользователь выбрал: все {n} файлов")
            return videos

        try:
            indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            print(f"  Неверный ввод. Введите 'all' или номера через запятую (например: 1,2).")
            continue

        invalid = [i for i in indices if i < 1 or i > n]
        if invalid:
            print(f"  Неверные номера: {invalid}. Допустимый диапазон: 1–{n}.")
            continue

        if not indices:
            print(f"  Список пуст. Введите 'all' или хотя бы один номер.")
            continue

        seen = set()
        selected = []
        for i in indices:
            if i not in seen:
                seen.add(i)
                selected.append(videos[i - 1])

        names = ", ".join(v.name for v in selected)
        logger.info(f"Пользователь выбрал {len(selected)} из {n}: {names}")
        return selected


# ---------------------------------------------------------------------------
# Публичная функция модуля
# ---------------------------------------------------------------------------

_VALID_SOURCES = ("real", "airsim")


def load_videos(
    project: Project,
    source: str,
    sample_rate: int = config.FRAME_SAMPLE_RATE,
    width: int = config.TARGET_WIDTH,
    height: int = config.TARGET_HEIGHT,
    fmt: str = config.FRAME_FORMAT,
) -> dict:
    """Извлекает кадры из выбранных mp4-файлов проекта.

    При наличии нескольких видео предлагает интерактивный выбор:
    пользователь вводит "all" или номера через запятую (1,2).
    При одном файле запрашивает подтверждение перед обработкой.

    Args:
        project:     объект Project — определяет пути к видео и кадрам.
        source:      идентификатор источника — "real" или "airsim".
                     Определяет папку с исходными видео, папку назначения
                     и префикс имён сохраняемых файлов.
        sample_rate: сохранять каждый N-й кадр (по умолчанию из config).
        width:       ширина выходных кадров в пикселях.
        height:      высота выходных кадров в пикселях.
        fmt:         формат изображения — "jpg" или "png".

    Returns:
        {"videos": <кол-во обработанных файлов>, "frames": <кол-во кадров>}

    Raises:
        ValueError:        если source не входит в допустимые значения.
        FileNotFoundError: если папка с исходными видео не существует.
    """
    if source not in _VALID_SOURCES:
        raise ValueError(f"Неизвестный источник '{source}'. Допустимые: {list(_VALID_SOURCES)}")

    # Путь к видео берём из data_sources проекта — задаётся при --new-project или --set-source
    input_dir  = project.get_source("videos", source)
    output_dir = project.frames_real_dir if source == "real" else project.frames_airsim_dir

    if input_dir is None:
        print(
            f"Путь к видео не указан для источника '{source}'.\n"
            f"Укажите путь: python main.py --project '{project.name}'"
            f" --set-source videos {source} C:/path/"
        )
        raise SystemExit(1)

    if not input_dir.exists():
        raise FileNotFoundError(f"Папка с видео не найдена: {input_dir}")

    all_videos = sorted(input_dir.glob("*.mp4"))

    # --- Случай 1: видеофайлов нет ---
    if not all_videos:
        logger.warning(f"Видеофайлы не найдены в папке: {input_dir}")
        stats = {"videos": 0, "frames": 0}
        project.update_stats({"load": stats})
        return stats

    # --- Случаи 2 и 3: интерактивный выбор ---
    selected = _select_videos(all_videos)

    if not selected:
        logger.warning("Ни один файл не выбран — завершаем работу.")
        stats = {"videos": 0, "frames": 0}
        project.update_stats({"load": stats})
        return stats

    output_dir.mkdir(parents=True, exist_ok=True)
    total_frames = 0

    for video_path in selected:
        total_frames += _extract_frames(
            video_path, source, output_dir, sample_rate, width, height, fmt
        )

    logger.info("=" * 50)
    logger.info(
        f"DONE | project={project.name} | source={source} | "
        f"обработано={len(selected)} | извлечено кадров={total_frames}"
    )
    logger.info(f"Результат: {output_dir}")
    logger.info("=" * 50)

    # Фиксируем путь к извлечённым кадрам в data_sources проекта
    project.set_source("frames", source, output_dir)

    stats = {"videos": len(selected), "frames": total_frames}
    project.update_stats({"load": stats})
    return stats
