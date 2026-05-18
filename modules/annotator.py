# modules/annotator.py — модуль разметки кадров датасета.
#
# Поддерживает два режима работы:
#   "auto"   — автоматическая разметка с помощью предобученной модели YOLOv8.
#              Для каждого кадра создаётся txt-файл в формате YOLO.
#   "manual" — проверка состояния разметки: выводит список кадров,
#              для которых файл аннотации ещё не создан.
#
# Публичный интерфейс: функции annotate() и run_interactive().
# Обе принимают объект Project — пути берутся из него, а не из config.

from pathlib import Path

from modules.logger import get_logger
from modules.project import Project

logger = get_logger(__name__)

# Допустимые источники кадров (порядок фиксирован для меню выбора)
_VALID_SOURCES = ("real", "airsim")

# Допустимые расширения изображений при поиске кадров
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Вспомогательная утилита: путь к файлу аннотации
# ---------------------------------------------------------------------------

def _annotation_path(frame_path: Path, ann_dir: Path) -> Path:
    """Возвращает путь к txt-файлу аннотации для заданного кадра.

    Файл аннотации лежит в ann_dir и называется так же, как кадр, но с .txt.

    Args:
        frame_path: путь к файлу кадра (jpg/png).
        ann_dir:    директория для хранения аннотаций.

    Returns:
        Путь к соответствующему файлу аннотации.
    """
    return ann_dir / (frame_path.stem + ".txt")


# ---------------------------------------------------------------------------
# Вспомогательная функция: сбор кадров из источников проекта
# ---------------------------------------------------------------------------

def _collect_frames(project: Project, sources: list) -> list:
    """Собирает все кадры из указанных источников проекта.

    Для каждого кадра возвращает пару (путь_к_кадру, папка_аннотаций).
    Папка аннотаций формируется как project.annotations_dir / source.

    Args:
        project: объект Project — определяет пути к кадрам и аннотациям.
        sources: список имён источников ("real", "airsim").

    Returns:
        Список пар (frame_path, ann_dir).
    """
    result = []

    for source in sources:
        # Путь к кадрам берём из data_sources проекта — задаётся при --new-project или load
        frames_dir = project.get_source("frames", source)

        if frames_dir is None:
            print(f"Путь к кадрам не задан для источника '{source}' — пропускаем")
            logger.info(f"Путь к кадрам не задан для источника '{source}' — пропускаем")
            continue

        if not frames_dir.exists():
            logger.warning(
                f"Папка с кадрами не найдена: {frames_dir} — "
                f"пропускаем источник '{source}'"
            )
            continue

        # Создаём папку аннотаций для данного источника внутри проекта
        ann_dir = project.annotations_dir / source
        ann_dir.mkdir(parents=True, exist_ok=True)

        # Собираем изображения в алфавитном порядке для воспроизводимости
        frames = [
            p for p in sorted(frames_dir.iterdir())
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for frame_path in frames:
            result.append((frame_path, ann_dir))

        logger.info(f"Источник '{source}': найдено {len(frames)} кадров → {frames_dir}")

    return result


# ---------------------------------------------------------------------------
# Режим авторазметки
# ---------------------------------------------------------------------------

def _run_auto(frames: list, model_path: str, conf: float, overwrite: bool) -> dict:
    """Запускает автоматическую разметку кадров с помощью YOLOv8.

    Для каждого кадра выполняет инференс, конвертирует детекции в формат
    YOLO txt (нормализованные координаты) и сохраняет файл аннотации.

    Args:
        frames:     список пар (frame_path, ann_dir).
        model_path: путь к весам или название предобученной модели ("yolov8n.pt").
        conf:       порог уверенности — детекции ниже отбрасываются.
        overwrite:  если True, перезаписывать существующие аннотации.

    Returns:
        {"annotated": int, "skipped": int, "total_objects": int}
    """
    # Импорт откладываем до реального использования: в режиме "manual"
    # ultralytics не нужен, и модуль можно использовать без этой зависимости
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Библиотека ultralytics не установлена. "
            "Установите командой: pip install ultralytics"
        )

    logger.info(f"Загрузка модели YOLOv8: {model_path}")
    model = YOLO(model_path)

    annotated = 0      # кадров успешно размечено на этом запуске
    skipped = 0        # кадров пропущено (уже размечены, overwrite=False)
    total_objects = 0  # суммарное количество найденных объектов

    total = len(frames)

    for i, (frame_path, ann_dir) in enumerate(frames, start=1):
        ann_file = _annotation_path(frame_path, ann_dir)

        # Пропускаем кадры с готовой аннотацией, если перезапись не нужна
        if ann_file.exists() and not overwrite:
            skipped += 1
            logger.debug(f"[{i}/{total}] Пропускаем (уже размечен): {frame_path.name}")
            continue

        # verbose=False отключает внутренний вывод ultralytics в консоль
        results = model(str(frame_path), conf=conf, verbose=False)

        # results — список длиной 1 (одно изображение → один результат)
        result = results[0]

        lines = []

        # result.boxes содержит все найденные объекты.
        # .xywhn — нормализованный формат [x_center, y_center, w, h] ∈ [0, 1],
        # что напрямую соответствует формату YOLO txt — конвертации не нужно.
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())  # индекс класса
                x_c, y_c, w, h = box.xywhn[0].tolist()

                # Строка аннотации: "класс x_центр y_центр ширина высота"
                # 6 знаков после запятой — стандартная точность для YOLO датасетов
                lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        # Сохраняем файл (пустой txt = "негативный пример" — это допустимо в YOLO)
        ann_file.write_text("\n".join(lines), encoding="utf-8")

        n_objects = len(lines)
        total_objects += n_objects
        annotated += 1

        logger.info(f"[{i}/{total}] {frame_path.name} → найдено объектов: {n_objects}")

    return {
        "annotated":      annotated,
        "skipped":        skipped,
        "total_objects":  total_objects,
    }


# ---------------------------------------------------------------------------
# Режим ручной разметки
# ---------------------------------------------------------------------------

def _run_manual(frames: list) -> dict:
    """Проверяет состояние разметки и выводит список неразмеченных кадров.

    Не создаёт и не изменяет никакие файлы — только читает текущее состояние.

    Args:
        frames: список пар (frame_path, ann_dir).

    Returns:
        {"annotated": int, "unannotated": int, "unannotated_files": list[str]}
    """
    annotated = 0
    unannotated = []

    for frame_path, ann_dir in frames:
        ann_file = _annotation_path(frame_path, ann_dir)

        if ann_file.exists():
            annotated += 1
        else:
            unannotated.append(frame_path)

    # Выводим список кадров без разметки
    if unannotated:
        logger.info(f"Кадры без разметки ({len(unannotated)} шт.):")
        for p in unannotated:
            logger.info(f"  {p}")
    else:
        logger.info("Все кадры уже размечены.")

    return {
        "annotated":        annotated,
        "unannotated":      len(unannotated),
        "unannotated_files": [str(p) for p in unannotated],
    }


# ---------------------------------------------------------------------------
# Интерактивный режим: вспомогательные функции
# ---------------------------------------------------------------------------

def _count_stats(project: Project, sources_list: list) -> dict:
    """Подсчитывает кадры и аннотации для каждого источника проекта.

    Args:
        project:      объект Project — определяет пути к кадрам и аннотациям.
        sources_list: список имён источников для проверки.

    Returns:
        Словарь вида:
        {
          "real":   {"frames": 1000, "annotated": 600, "unannotated": 400},
          "airsim": {"frames": 250,  "annotated": 250, "unannotated": 0},
        }
    """
    frames_map = {
        "real":   project.frames_real_dir,
        "airsim": project.frames_airsim_dir,
    }

    result = {}
    for source in sources_list:
        frames_dir = frames_map.get(source)
        if not frames_dir or not frames_dir.exists():
            result[source] = {"frames": 0, "annotated": 0, "unannotated": 0}
            continue

        frames = [p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        ann_dir = project.annotations_dir / source
        annotated = sum(1 for f in frames if (ann_dir / (f.stem + ".txt")).exists())

        result[source] = {
            "frames":      len(frames),
            "annotated":   annotated,
            "unannotated": len(frames) - annotated,
        }
    return result


def _ask_source(stats: dict) -> list:
    """Интерактивно запрашивает у пользователя источник для разметки.

    Args:
        stats: словарь статистики от _count_stats().

    Returns:
        Список выбранных имён источников.
    """
    sources_with_frames = [s for s, v in stats.items() if v["frames"] > 0]

    print("\nКакой источник размечать?")
    for i, source in enumerate(sources_with_frames, start=1):
        n = stats[source]["frames"]
        print(f"  {i}. {source} ({n} кадров)")
    all_opt = len(sources_with_frames) + 1
    print(f"  {all_opt}. Все источники")

    opts = "/".join(str(i) for i in range(1, all_opt + 1))
    while True:
        raw = input(f"Выберите ({opts}): ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print(f"  Введите число от 1 до {all_opt}.")
            continue

        if choice == all_opt:
            logger.info(f"Пользователь выбрал: все источники {sources_with_frames}")
            return sources_with_frames

        if 1 <= choice <= len(sources_with_frames):
            selected = [sources_with_frames[choice - 1]]
            logger.info(f"Пользователь выбрал источник: {selected}")
            return selected

        print(f"  Введите число от 1 до {all_opt}.")


def _ask_mode() -> str:
    """Интерактивно запрашивает режим разметки.

    Returns:
        "auto" или "manual".
    """
    print("\nВыберите режим:")
    print("  1. auto   — авторазметка через YOLOv8")
    print("  2. manual — показать список неразмеченных кадров")
    while True:
        raw = input("Выберите (1/2): ").strip()
        if raw == "1":
            logger.info("Пользователь выбрал режим: auto")
            return "auto"
        if raw == "2":
            logger.info("Пользователь выбрал режим: manual")
            return "manual"
        print("  Введите 1 или 2.")


def _ask_overwrite(annotated_count: int, total_count: int) -> bool:
    """Спрашивает, перезаписывать ли уже размеченные кадры.

    Вызывается только в режиме auto при наличии существующих аннотаций.

    Args:
        annotated_count: количество уже размеченных кадров.
        total_count:     общее число кадров в выборке.

    Returns:
        True — перезаписать все, False — пропустить уже размеченные.
    """
    new_count = total_count - annotated_count
    print(f"\nНайдено {annotated_count} уже размеченных кадров.")
    print(f"  1. Пропустить (размечать только новые {new_count})")
    print(f"  2. Перезаписать все ({total_count} кадров)")
    while True:
        raw = input("Выберите (1/2): ").strip()
        if raw == "1":
            logger.info("Пользователь выбрал: пропустить размеченные (overwrite=False)")
            return False
        if raw == "2":
            logger.info("Пользователь выбрал: перезаписать все (overwrite=True)")
            return True
        print("  Введите 1 или 2.")


def run_interactive(
    project: Project,
    model_path: str = "yolov8n.pt",
    conf: float = 0.25,
) -> dict:
    """Интерактивный запуск аннотатора с выбором источника, режима и политики перезаписи.

    Последовательно задаёт пользователю вопросы и вызывает annotate()
    с собранными параметрами.

    Args:
        project:    объект Project — определяет пути к кадрам и аннотациям.
        model_path: модель YOLOv8 (используется только в режиме auto).
        conf:       порог уверенности (только для auto).

    Returns:
        Словарь со статистикой от annotate().
    """
    # Подключаем проектный лог до любых обращений к logger
    get_logger(__name__, project.logs_dir)

    stats = _count_stats(project, list(_VALID_SOURCES))

    # --- Шаг 1: общая статистика ---
    total_frames      = sum(v["frames"]      for v in stats.values())
    total_annotated   = sum(v["annotated"]   for v in stats.values())
    total_unannotated = sum(v["unannotated"] for v in stats.values())

    print(f"\nНайдено кадров: {total_frames}")
    print(f"Уже размечено:  {total_annotated}")
    print(f"Не размечено:   {total_unannotated}")
    logger.info(
        f"Статистика: всего={total_frames}, "
        f"размечено={total_annotated}, не размечено={total_unannotated}"
    )

    if total_frames == 0:
        logger.warning("Кадры не найдены ни в одном источнике.")
        return {}

    # --- Шаг 2: выбор источника ---
    selected_sources = _ask_source(stats)

    # --- Шаг 3: выбор режима ---
    mode = _ask_mode()

    # --- Шаг 4: политика перезаписи (только для auto) ---
    overwrite = False
    if mode == "auto":
        already    = sum(stats[s]["annotated"] for s in selected_sources)
        total_sel  = sum(stats[s]["frames"]    for s in selected_sources)
        if already > 0:
            overwrite = _ask_overwrite(already, total_sel)

    return annotate(project, mode=mode, model_path=model_path,
                    conf=conf, sources=selected_sources, overwrite=overwrite)


# ---------------------------------------------------------------------------
# Публичная функция модуля
# ---------------------------------------------------------------------------

def annotate(
    project: Project,
    mode: str = "auto",
    model_path: str = "yolov8n.pt",
    conf: float = 0.25,
    sources: list = None,
    overwrite: bool = False,
) -> dict:
    """Запускает разметку кадров проекта в выбранном режиме.

    Args:
        project:    объект Project — определяет пути к кадрам и аннотациям.
        mode:       режим работы — "auto" (авторазметка) или "manual" (проверка).
        model_path: путь к файлу весов YOLOv8 или имя предобученной модели
                    ("yolov8n.pt", "yolov8s.pt" и т.д.). Только для "auto".
        conf:       порог уверенности от 0.0 до 1.0. Только для режима "auto".
        sources:    список источников для обработки. По умолчанию — все
                    (["real", "airsim"]).
        overwrite:  если True, перезаписывать существующие аннотации.
                    Только для режима "auto".

    Returns:
        Словарь со статистикой. Набор ключей зависит от режима:
        - "auto":   {"annotated", "skipped", "total_objects"}
        - "manual": {"annotated", "unannotated", "unannotated_files"}

    Raises:
        ValueError:  если передан неизвестный режим.
        ImportError: если ultralytics не установлен (только в режиме "auto").
    """
    # Подключаем файловый лог проекта — все записи logger в этом модуле
    # (включая вызовы из _collect_frames, _run_auto, _run_manual) попадут
    # в project.logs_dir/annotator.log
    get_logger(__name__, project.logs_dir)

    if mode not in ("auto", "manual"):
        raise ValueError(
            f"Неизвестный режим '{mode}'. Допустимые значения: 'auto', 'manual'"
        )

    # По умолчанию обрабатываем все источники
    if sources is None:
        sources = list(_VALID_SOURCES)

    logger.info("=" * 50)
    logger.info(
        f"Annotator | project={project.name} | режим={mode} | источники={sources}"
    )
    if mode == "auto":
        logger.info(f"Модель: {model_path} | conf={conf} | перезапись={overwrite}")
    logger.info("=" * 50)

    # Собираем полный список кадров из всех выбранных источников
    frames = _collect_frames(project, sources)

    if not frames:
        logger.warning("Кадры для разметки не найдены. Проверьте папки источников.")
        return {}

    logger.info(f"Всего кадров для обработки: {len(frames)}")

    # Запускаем нужный режим
    if mode == "auto":
        result = _run_auto(frames, model_path, conf, overwrite)
    else:
        result = _run_manual(frames)

    # Итоговый отчёт
    logger.info("=" * 50)
    logger.info("ИТОГОВЫЙ ОТЧЁТ:")
    for key, value in result.items():
        # Список файлов уже был выведен построчно в _run_manual
        if key != "unannotated_files":
            logger.info(f"  {key}: {value}")
    logger.info(f"Папка аннотаций: {project.annotations_dir}")
    logger.info("=" * 50)

    # Фиксируем пути к аннотациям в data_sources проекта
    for source in sources:
        project.set_source("annotations", source, project.annotations_dir / source)

    # Обновляем метаданные проекта
    project.update_step("annotate")
    project.update_stats({"annotated": result.get("annotated", 0)})

    return result
