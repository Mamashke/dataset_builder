# modules/loader.py — модуль загрузки и первичной обработки видеоданных.
#
# Основная задача: принять папку с mp4-файлами, извлечь из каждого видео
# кадры с заданным шагом, привести их к единому разрешению и сохранить
# в соответствующую папку внутри data/processed/frames/.
#
# Публичный интерфейс модуля — единственная функция load_videos().
# Вспомогательные функции (_output_dir, _extract_frames) намеренно скрыты
# (префикс _), чтобы не засорять пространство имён при импорте.

from pathlib import Path
from typing import Union

import cv2  # OpenCV — основная библиотека для работы с видео и изображениями

import config  # Централизованные настройки и пути проекта
from modules.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Таблица соответствия: имя источника → папка назначения
# ---------------------------------------------------------------------------

# Словарь связывает строковый идентификатор источника с путём из config.
# Добавление нового источника в будущем требует только одной строки здесь.
SOURCES = {
    "real": config.FRAMES_REAL_DIR,      # реальные видеозаписи
    "airsim": config.FRAMES_AIRSIM_DIR,  # записи из симулятора AirSim
}


# ---------------------------------------------------------------------------
# Вспомогательная функция: подготовка выходной директории
# ---------------------------------------------------------------------------

def _output_dir(source: str) -> Path:
    """Проверяет корректность имени источника и возвращает путь к папке вывода.

    Создаёт папку (и все промежуточные директории), если она не существует.

    Args:
        source: идентификатор источника — "real" или "airsim".

    Returns:
        Абсолютный путь к папке, куда нужно сохранять кадры.

    Raises:
        ValueError: если передано неизвестное имя источника.
    """
    # Проверяем, что источник входит в список допустимых значений.
    # Это защищает от опечаток вроде "Real" или "AirSim".
    if source not in SOURCES:
        raise ValueError(f"Unknown source '{source}'. Expected: {list(SOURCES)}")

    out = SOURCES[source]

    # parents=True  — создаёт все промежуточные папки (как mkdir -p)
    # exist_ok=True — не бросает исключение, если папка уже есть
    out.mkdir(parents=True, exist_ok=True)

    return out


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
    # Открываем видеофайл через OpenCV.
    # cv2.VideoCapture принимает строку, поэтому Path конвертируется через str().
    cap = cv2.VideoCapture(str(video_path))

    # Проверяем, что файл успешно открыт.
    # Это может не сработать при повреждённом файле или неверном пути.
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path.name} — skipping")
        return 0  # Возвращаем 0, чтобы не прерывать обработку остальных файлов

    # Общее число кадров в видео — используется только для информативного лога.
    # CAP_PROP_FRAME_COUNT возвращает float, поэтому приводим к int.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Имя файла без расширения — войдёт в имя каждого сохранённого кадра.
    video_stem = video_path.stem

    saved = 0      # счётчик сохранённых кадров для текущего видео
    frame_idx = 0  # порядковый номер текущего кадра (с нуля)

    logger.info(f"Processing '{video_path.name}' ({total_frames} frames, step={sample_rate})")

    # Основной цикл чтения кадров.
    # cap.read() возвращает (True, кадр) при успехе или (False, None) в конце файла.
    while True:
        ret, frame = cap.read()

        # ret == False означает конец видео или ошибку чтения — выходим из цикла
        if not ret:
            break

        # Сохраняем только каждый sample_rate-й кадр.
        # frame_idx % sample_rate == 0 истинно для кадров 0, 5, 10, 15, ...
        if frame_idx % sample_rate == 0:
            # Масштабируем кадр до целевого разрешения.
            # INTER_AREA — оптимальный алгоритм для уменьшения изображения:
            # усредняет пиксели области, что даёт более чёткий результат
            # по сравнению с билинейной или ближайшей интерполяцией.
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # Формируем имя файла: real_video1_frame_000025.jpg
            # :06d — номер кадра с ведущими нулями до 6 цифр,
            # что обеспечивает правильную сортировку по имени файла.
            filename = f"{source}_{video_stem}_frame_{frame_idx:06d}.{fmt}"

            # Сохраняем кадр на диск.
            # cv2.imwrite автоматически выбирает кодек по расширению файла.
            cv2.imwrite(str(output_dir / filename), resized)
            saved += 1

        frame_idx += 1  # переходим к следующему кадру

    # Освобождаем ресурсы видеодекодера (закрываем файл).
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
    - 1 файл:   сразу возвращает его без вопросов.
    - N файлов: печатает нумерованный список, запрашивает ввод в цикле
                до получения корректного ответа.

    Args:
        videos: отсортированный список Path-объектов mp4-файлов.

    Returns:
        Список Path-объектов, выбранных для обработки.
    """
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

    # Цикл повторяется до получения валидного ввода
    while True:
        raw = input('\nКакие файлы обработать? (all / номера через запятую: 1,2): ').strip()

        if raw.lower() == "all":
            logger.info(f"Пользователь выбрал: все {n} файлов")
            return videos

        # Парсим список номеров: "1,3" → [1, 3]
        try:
            indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            print(f"  Неверный ввод. Введите 'all' или номера через запятую (например: 1,2).")
            continue

        # Проверяем, что все номера в допустимом диапазоне [1, n]
        invalid = [i for i in indices if i < 1 or i > n]
        if invalid:
            print(f"  Неверные номера: {invalid}. Допустимый диапазон: 1–{n}.")
            continue

        if not indices:
            print(f"  Список пуст. Введите 'all' или хотя бы один номер.")
            continue

        # Убираем дубликаты, сохраняем порядок, переходим к 0-based индексам
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

def load_videos(
    input_dir: Union[str, Path],
    source: str,
    sample_rate: int = config.FRAME_SAMPLE_RATE,
    width: int = config.TARGET_WIDTH,
    height: int = config.TARGET_HEIGHT,
    fmt: str = config.FRAME_FORMAT,
) -> dict:
    """Извлекает кадры из выбранных mp4-файлов в указанной папке.

    При наличии нескольких видео предлагает интерактивный выбор:
    пользователь вводит "all" или номера через запятую (1,2).
    При одном файле обработка начинается без вопросов.

    Args:
        input_dir:   путь к папке с исходными mp4-файлами.
        source:      идентификатор источника — "real" или "airsim".
                     Определяет папку назначения и префикс имён файлов.
        sample_rate: сохранять каждый N-й кадр (по умолчанию из config).
        width:       ширина выходных кадров в пикселях.
        height:      высота выходных кадров в пикселях.
        fmt:         формат изображения — "jpg" или "png".

    Returns:
        {"videos": <кол-во обработанных файлов>, "frames": <кол-во кадров>}

    Raises:
        FileNotFoundError: если папка input_dir не существует.
        ValueError:        если source не входит в допустимые значения.
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Папка не найдена: {input_dir}")

    all_videos = sorted(input_dir.glob("*.mp4"))

    # --- Случай 1: видеофайлов нет ---
    if not all_videos:
        logger.warning(f"Видеофайлы не найдены в папке: {input_dir}")
        return {"videos": 0, "frames": 0}

    # --- Случаи 2 и 3: интерактивный выбор ---
    selected = _select_videos(all_videos)

    if not selected:
        logger.warning("Ни один файл не выбран — завершаем работу.")
        return {"videos": 0, "frames": 0}

    output_dir = _output_dir(source)
    total_frames = 0

    for video_path in selected:
        total_frames += _extract_frames(
            video_path, source, output_dir, sample_rate, width, height, fmt
        )

    logger.info("=" * 50)
    logger.info(
        f"DONE | source={source} | обработано={len(selected)} | "
        f"извлечено кадров={total_frames}"
    )
    logger.info(f"Результат: {output_dir}")
    logger.info("=" * 50)

    return {"videos": len(selected), "frames": total_frames}
