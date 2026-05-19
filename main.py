# main.py — единая точка входа dataset_builder с системой проектов.
#
# Управление проектами:
#   python main.py --new-project "название"
#   python main.py --list-projects
#   python main.py --delete-project "название"
#
# Запуск пайплайна (требуется --project):
#   python main.py --project "название" --all
#   python main.py --project "название" --all --from annotate
#   python main.py --project "название" --load
#   python main.py --project "название" --annotate
#   python main.py --project "название" --augment
#   python main.py --project "название" --balance
#   python main.py --project "название" --export --format coco
#
# Очистка данных проекта (требуется --project):
#   python main.py --project "название" --clean              # интерактивное меню
#   python main.py --project "название" --clean --frames     # удалить только кадры
#   python main.py --project "название" --clean --processed  # кадры + аннотации
#   python main.py --project "название" --clean --all-data   # всё кроме raw/

import argparse
import shutil
import sys
import time
from pathlib import Path

# Лог текущего запуска создаётся первым — до импорта остальных модулей,
# чтобы все их сообщения попали в run_*.log
from modules.logger import get_logger, setup_run_log

logger = get_logger(__name__)

from modules.loader    import load_videos
from modules.annotator import run_interactive as annotate_interactive
from modules.augmentor import augment_dataset
from modules.balancer  import build as balance_build
from modules.exporter  import export as do_export
from modules.project   import Project

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

BANNER = """\
================================
   dataset_builder v1.0
   Конструктор обучающей выборки
================================"""

# Порядок шагов пайплайна — фиксирован и используется в нескольких местах
PIPELINE_STEPS = ["load", "annotate", "augment", "balance", "export"]

STEP_NAMES = {
    "load":     "Загрузка видео",
    "annotate": "Разметка кадров",
    "augment":  "Аугментация",
    "balance":  "Балансировка",
    "export":   "Экспорт датасета",
}

# Соответствие пунктов меню «С чего начать?» первому шагу пайплайна
_START_FROM_MAP = {
    "1": "load",      # есть видео → начинаем с извлечения кадров
    "2": "annotate",  # есть кадры → начинаем с разметки
    "3": "balance",   # есть размеченный датасет → сразу балансировка
}

# Папки, которые затрагивает каждый уровень очистки.
# raw/, project.json и logs/ никогда не удаляются.
_CLEAN_DIRS = {
    "frames":    ["frames"],
    "processed": ["frames", "annotations"],
    "all_data":  ["frames", "annotations", "dataset", "export"],
}

# Ключи статистики, которые сбрасываются после каждого уровня.
# None означает «сбросить всю статистику».
_CLEAN_STATS_KEYS = {
    "frames":    ["load", "augmented_frames"],
    "processed": ["load", "augmented_frames", "annotated"],
    "all_data":  None,
}


# ---------------------------------------------------------------------------
# Вспомогательные утилиты
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """Форматирует секунды в читаемую строку вида «2 мин 15 сек»."""
    total = int(seconds)
    mins, secs = divmod(total, 60)
    if mins:
        return f"{mins} мин {secs} сек"
    return f"{secs} сек"


def _step_header(index: int, total: int, name: str) -> None:
    """Печатает заголовок шага пайплайна в консоль и лог."""
    print(f"\n[{index}/{total}] {name}...")
    logger.info(f"[{index}/{total}] Начало: {name}")


def _folder_size(path: Path) -> int:
    """Рекурсивно считает суммарный размер файлов в папке (байты)."""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _fmt_size(size_bytes: int) -> str:
    """Форматирует байты в строку с суффиксом (ГБ / МБ / КБ / Б)."""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / 1024 ** 3:.2f} ГБ"
    if size_bytes >= 1024 ** 2:
        return f"{size_bytes / 1024 ** 2:.1f} МБ"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f} КБ"
    return f"{size_bytes} Б"


def _clear_dir(path: Path) -> None:
    """Удаляет всё содержимое папки, сохраняя саму папку."""
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _ask_valid_path(prompt: str, optional: bool = False, max_attempts: int = 3):
    """Запрашивает путь к папке, повторяя до max_attempts раз при ошибке.

    Args:
        prompt:       строка приглашения ввода.
        optional:     если True, пустой ввод возвращает None без ошибки.
        max_attempts: максимальное число попыток (по умолчанию 3).

    Returns:
        Path если папка существует, None если optional и ввод пустой.
    """
    for attempt in range(1, max_attempts + 1):
        raw = input(prompt).strip()
        if not raw:
            if optional:
                return None
            print("  Путь не может быть пустым.")
        else:
            path = Path(raw)
            if path.exists():
                return path
            remaining = max_attempts - attempt
            if remaining:
                print(f"  Папка не найдена: {path}. Осталось попыток: {remaining}.")
            else:
                print(f"  Папка не найдена: {path}.")
        if attempt == max_attempts:
            print("  Превышено число попыток. Завершение.")
            sys.exit(1)
    return None


def _ask_sources(project: "Project", start_from: str) -> None:
    """Запрашивает пути к исходным данным в зависимости от стартового шага.

    Args:
        project:    объект Project для сохранения путей через set_source().
        start_from: стартовый шаг — "load", "annotate" или "balance".
    """
    print()
    if start_from == "load":
        p = _ask_valid_path("Укажите путь к папке с видео (real): ")
        project.set_source("videos", "real", p)
        p = _ask_valid_path(
            "Укажите путь к папке с видео airsim (Enter — пропустить): ",
            optional=True,
        )
        if p is not None:
            project.set_source("videos", "airsim", p)

    elif start_from == "annotate":
        p = _ask_valid_path("Укажите путь к папке с кадрами (real): ")
        project.set_source("frames", "real", p)
        p = _ask_valid_path(
            "Укажите путь к папке с кадрами airsim (Enter — пропустить): ",
            optional=True,
        )
        if p is not None:
            project.set_source("frames", "airsim", p)

    elif start_from == "balance":
        p = _ask_valid_path("Укажите путь к папке с изображениями: ")
        project.set_source("frames", "real", p)
        p = _ask_valid_path("Укажите путь к папке с разметкой: ")
        project.set_source("annotations", "real", p)


# ---------------------------------------------------------------------------
# Команды управления проектами
# ---------------------------------------------------------------------------

def _cmd_new_project(name: str) -> None:
    """Создаёт новый проект и спрашивает, с чего начать работу.

    Ответ пользователя сохраняется в project.json как «start_from» —
    это первый шаг, который будет выполнен при запуске --all без --from.

    Args:
        name: имя нового проекта.
    """
    project = Project.create(name)

    print("\nС чего начать работу?")
    print("  1. У меня есть видео (mp4)")
    print("  2. У меня есть кадры без разметки")
    print("  3. У меня есть размеченный датасет")

    while True:
        raw = input("Выберите (1/2/3): ").strip()
        if raw in _START_FROM_MAP:
            start_from = _START_FROM_MAP[raw]
            break
        print("  Введите 1, 2 или 3.")

    # Сохраняем start_from в project.json напрямую — это поле верхнего уровня,
    # а не статистика, поэтому update_stats() не подходит
    meta = project._read_meta()
    meta["start_from"] = start_from
    project._write_meta(meta)

    # Запрашиваем пути к исходным данным для выбранного сценария
    _ask_sources(project, start_from)

    logger.info(f"Проект '{name}' создан, start_from='{start_from}'")
    print(f"\nГотово! Начало работы: с шага «{STEP_NAMES[start_from]}».")
    print(f"Для запуска: python main.py --project \"{name}\" --all")


def _cmd_list_projects() -> None:
    """Выводит таблицу всех проектов с ключевой статистикой."""
    projects = Project.list_all()

    if not projects:
        print("Проектов не найдено.")
        print("Создайте первый: python main.py --new-project \"название\"")
        return

    print("\nПроекты:")
    for i, meta in enumerate(projects, start=1):
        name    = meta.get("name", "?")
        step    = meta.get("current_step") or "—"
        # Берём только дату из ISO-строки вида "2026-05-13T14:30:00"
        created = meta.get("created", "")[:10]
        stats   = meta.get("stats", {})

        # Количество кадров: сначала ищем после балансировки,
        # затем — из статистики загрузки (если баланс ещё не запускался)
        load_info = stats.get("load", {})
        frames = (
            stats.get("dataset_frames") or
            (load_info.get("frames") if isinstance(load_info, dict) else None) or
            "—"
        )

        print(
            f"  {i}. {name:<20} | "
            f"шаг: {step:<10} | "
            f"кадров: {str(frames):<6} | "
            f"создан: {created}"
        )


def _cmd_delete_project(name: str) -> None:
    """Удаляет проект после подтверждения пользователя.

    Args:
        name: имя проекта для удаления.
    """
    Project.delete(name)


def _cmd_clean_project(project: Project, level: str = None) -> None:
    """Удаляет обработанные данные проекта на указанном уровне.

    Никогда не удаляет: raw/ (исходные видео), project.json, logs/.
    После удаления сбрасывает связанные поля статистики в project.json.

    Args:
        project: объект Project — определяет пути к папкам.
        level:   "frames", "processed", "all_data" или None для интерактивного выбора.
    """
    # Корневые папки уровней очистки
    frames_dir      = project.frames_real_dir.parent    # frames/
    annotations_dir = project.annotations_dir            # annotations/
    dataset_dir     = project.dataset_images_dir.parent  # dataset/
    export_dir      = project.export_dir                 # export/

    dir_map = {
        "frames":      frames_dir,
        "annotations": annotations_dir,
        "dataset":     dataset_dir,
        "export":      export_dir,
    }

    # Интерактивный выбор уровня если флаг не передан
    if level is None:
        sz_frames      = _folder_size(frames_dir)
        sz_annotations = _folder_size(annotations_dir)
        sz_dataset     = _folder_size(dataset_dir)
        sz_export      = _folder_size(export_dir)

        sz1 = sz_frames
        sz2 = sz_frames + sz_annotations
        sz3 = sz_frames + sz_annotations + sz_dataset + sz_export

        print(f"\nЧто удалить в проекте '{project.name}'?")
        print(f"  1. Только кадры (frames/)"
              f"                              — {_fmt_size(sz1)}")
        print(f"  2. Кадры + аннотации (frames/ + annotations/)"
              f"           — {_fmt_size(sz2)}")
        print(f"  3. Все обработанные данные"
              f" (frames/ + annotations/ + dataset/ + export/) — {_fmt_size(sz3)}")
        print(f"  4. Отмена")

        while True:
            raw = input("Выберите (1/2/3/4): ").strip()
            if raw == "1":
                level = "frames"
                break
            if raw == "2":
                level = "processed"
                break
            if raw == "3":
                level = "all_data"
                break
            if raw == "4":
                print("Отменено.")
                return
            print("  Введите 1, 2, 3 или 4.")

    # Считаем итоговый объём удаляемых данных
    total_bytes = sum(_folder_size(dir_map[d]) for d in _CLEAN_DIRS[level])

    # Показываем что будет удалено и запрашиваем подтверждение
    folders_str = " + ".join(f"{d}/" for d in _CLEAN_DIRS[level])
    print(f"\nБудет удалено: {_fmt_size(total_bytes)}")
    print(f"Папки: {folders_str}")

    while True:
        raw = input("Вы уверены? (yes/no): ").strip().lower()
        if raw == "yes":
            break
        if raw == "no":
            print("Отменено.")
            return
        print("  Введите 'yes' или 'no'.")

    # Удаляем содержимое каждой папки (структуру папок сохраняем)
    for dir_name in _CLEAN_DIRS[level]:
        path = dir_map[dir_name]
        _clear_dir(path)
        logger.info(f"Очищено: {path}")

    # Обновляем project.json: сбрасываем связанные поля статистики
    meta       = project._read_meta()
    stats_keys = _CLEAN_STATS_KEYS[level]

    if stats_keys is None:
        # all_data — стираем всю статистику
        meta["stats"] = {}
    else:
        for key in stats_keys:
            meta["stats"].pop(key, None)

    meta["current_step"] = None
    project._write_meta(meta)

    logger.info(f"Очистка завершена: уровень='{level}', удалено={_fmt_size(total_bytes)}")
    print(f"Готово. Удалено: {_fmt_size(total_bytes)}")


# ---------------------------------------------------------------------------
# Интерактивный выбор формата экспорта
# ---------------------------------------------------------------------------

def _ask_export_format() -> str:
    """Спрашивает пользователя, в какой формат экспортировать датасет.

    Returns:
        "yolo" или "coco".
    """
    print("\nВыберите формат экспорта:")
    print("  1. YOLO (data.yaml)")
    print("  2. COCO (annotations.json)")
    while True:
        raw = input("Выберите (1/2): ").strip()
        if raw == "1":
            logger.info("Пользователь выбрал формат: yolo")
            return "yolo"
        if raw == "2":
            logger.info("Пользователь выбрал формат: coco")
            return "coco"
        print("  Введите 1 или 2.")


# ---------------------------------------------------------------------------
# Шаги пайплайна
# ---------------------------------------------------------------------------

def step_load(project: Project) -> dict:
    """Извлекает кадры из видео обоих источников проекта.

    Вызывает load_videos() для real и airsim поочерёдно.
    После обоих вызовов объединяет статистику и перезаписывает её в project.json,
    потому что каждый вызов load_videos() сохраняет только свои данные.

    Args:
        project: объект Project с путями к видео и кадрам.

    Returns:
        Суммарная статистика: {"videos": N, "frames": N}.
    """
    empty = {"videos": 0, "frames": 0}
    results = {}
    for source in ("real", "airsim"):
        if project.get_source("videos", source) is None:
            logger.info(f"Источник '{source}' пропущен — путь не задан")
            results[source] = empty
            continue
        results[source] = load_videos(project, source=source)

    combined = {
        "videos": results["real"]["videos"] + results["airsim"]["videos"],
        "frames": results["real"]["frames"] + results["airsim"]["frames"],
    }
    # Перезаписываем суммарной статистикой — каждый вызов load_videos()
    # сохранил только свой источник, нам нужна сумма по обоим
    project.update_stats({"load": combined})
    return combined


def step_annotate(project: Project) -> dict:
    """Интерактивная разметка кадров проекта через YOLOv8.

    Args:
        project: объект Project с путями к кадрам и аннотациям.

    Returns:
        Статистика разметки от run_interactive().
    """
    return annotate_interactive(project)


def step_augment(project: Project) -> dict:
    """Аугментация кадров проекта: туман, дождь, шум, размытие, яркость.

    Args:
        project: объект Project с путями к кадрам.

    Returns:
        Статистика аугментации от augment_dataset().
    """
    aug_types = ["fog", "rain", "noise", "blur", "brightness"]
    return augment_dataset(project, aug_types, intensity=0.5)


def step_balance(project: Project) -> dict:
    """Фильтрация, балансировка и сборка финального датасета.

    Args:
        project: объект Project с путями к кадрам, аннотациям и датасету.

    Returns:
        Полный отчёт со статистикой всех этапов от balance_build().
    """
    return balance_build(project, overwrite=True)


def step_export(project: Project, fmt: str) -> dict:
    """Экспортирует финальный датасет проекта в указанный формат.

    Args:
        project: объект Project с путями к датасету и папке экспорта.
        fmt:     "yolo" или "coco".

    Returns:
        Статистика экспорта от do_export().
    """
    return do_export(project, format=fmt)


# ---------------------------------------------------------------------------
# Итоговый отчёт
# ---------------------------------------------------------------------------

def _print_report(project: Project, results: dict, total_seconds: float) -> None:
    """Выводит финальный отчёт после выполнения --all.

    Args:
        project:       текущий проект (имя используется в заголовке).
        results:       словарь {шаг: статистика} по каждому выполненному шагу.
        total_seconds: общее время работы пайплайна.
    """
    print("\n================================")
    print(f"   ИТОГОВЫЙ ОТЧЁТ — {project.name}")

    if "load" in results:
        r = results["load"]
        print(f"   Загрузка:     видео={r.get('videos','—')}, "
              f"кадров={r.get('frames','—')}")

    if "annotate" in results:
        r = results["annotate"]
        print(f"   Разметка:     размечено={r.get('annotated','—')}, "
              f"объектов={r.get('total_objects','—')}")

    if "augment" in results:
        r = results["augment"]
        print(f"   Аугментация:  создано {r.get('created','—')} кадров")

    if "balance" in results:
        r = results["balance"]
        print(f"   Балансировка: финальных кадров {r.get('after_balance','—')} "
              f"(pos={r.get('positives','—')}, neg={r.get('negatives','—')})")

    if "export" in results:
        r = results["export"]
        fmt_label = "YOLO" if "yaml_path" in r else "COCO"
        imgs      = r.get("images", "—")
        anns      = (f", аннотаций={r.get('annotations','—')}"
                     if "annotations" in r else "")
        print(f"   Экспорт:      формат {fmt_label}, изображений={imgs}{anns}")

    print(f"   Время работы: {_fmt_duration(total_seconds)}")
    print("================================")

    logger.info(f"Итоговый отчёт: {results}")
    logger.info(f"Время работы: {_fmt_duration(total_seconds)}")


# ---------------------------------------------------------------------------
# Управление источниками данных
# ---------------------------------------------------------------------------

def _cmd_set_source(project: Project, category: str, key: str, path_str: str) -> None:
    """Задаёт путь к источнику данных проекта и сохраняет его в project.json.

    Args:
        project:   объект Project.
        category:  категория источника ("videos", "frames", "annotations", "dataset").
        key:       ключ внутри категории ("real", "airsim", "images", "labels").
        path_str:  путь к папке с данными.
    """
    path = Path(path_str)

    # Проверяем существование папки до обращения к project.set_source
    if not path.exists():
        print(f"Ошибка: папка не найдена: {path}")
        sys.exit(1)

    try:
        project.set_source(category, key, path)
    except KeyError as exc:
        # project.set_source выбрасывает KeyError при неверной категории или ключе
        print(f"Ошибка: {exc}")
        sys.exit(1)

    print(f"Источник обновлён: {category}.{key} = {path}")
    logger.info(f"set_source | project={project.name} | {category}.{key} → {path}")


# ---------------------------------------------------------------------------
# Разбор аргументов
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="dataset_builder — конструктор обучающей выборки для YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python main.py --new-project дрон\n"
            "  python main.py --list-projects\n"
            "  python main.py --project дрон --all\n"
            "  python main.py --project дрон --all --from annotate\n"
            "  python main.py --project дрон --export --format coco\n"
            "  python main.py --project дрон --set-source videos real C:/videos/\n"
            "  python main.py --project дрон --set-source frames airsim C:/airsim/\n"
            "  python main.py --project дрон --clean\n"
            "  python main.py --project дрон --clean --frames\n"
            "  python main.py --project дрон --clean --all-data\n"
        ),
    )

    # Группа управления проектами
    grp_proj = parser.add_argument_group("Управление проектами")
    grp_proj.add_argument(
        "--new-project", metavar="НАЗВАНИЕ", dest="new_project",
        help="Создать новый проект",
    )
    grp_proj.add_argument(
        "--list-projects", action="store_true", dest="list_projects",
        help="Список всех проектов",
    )
    grp_proj.add_argument(
        "--delete-project", metavar="НАЗВАНИЕ", dest="delete_project",
        help="Удалить проект",
    )

    # Группа пайплайна
    grp_run = parser.add_argument_group("Запуск пайплайна (требуется --project)")
    grp_run.add_argument(
        "--project", metavar="НАЗВАНИЕ",
        help="Имя проекта для запуска шагов пайплайна",
    )
    grp_run.add_argument(
        "--all", action="store_true",
        help="Запустить весь пайплайн (с шага, сохранённого в проекте)",
    )
    grp_run.add_argument(
        "--from", metavar="ШАГ", dest="from_step",
        help=f"Начать с указанного шага (с --all): {', '.join(PIPELINE_STEPS)}",
    )
    grp_run.add_argument("--load",     action="store_true", help="Загрузка видео")
    grp_run.add_argument("--annotate", action="store_true", help="Разметка кадров")
    grp_run.add_argument("--augment",  action="store_true", help="Аугментация кадров")
    grp_run.add_argument("--balance",  action="store_true", help="Балансировка датасета")
    grp_run.add_argument("--export",   action="store_true", help="Экспорт датасета")
    grp_run.add_argument(
        "--format", choices=["yolo", "coco"], default=None,
        help="Формат экспорта: yolo или coco (используется с --export)",
    )

    # Группа управления источниками данных
    grp_src = parser.add_argument_group("Управление источниками данных (требуется --project)")
    grp_src.add_argument(
        "--set-source", nargs=3, metavar=("КАТЕГОРИЯ", "КЛЮЧ", "ПУТЬ"),
        dest="set_source",
        help=(
            "Задать путь к источнику данных проекта. "
            "Категории: videos, frames, annotations, dataset. "
            "Ключи: real, airsim, images, labels. "
            "Пример: --set-source videos real C:/videos/"
        ),
    )

    # Группа очистки данных
    grp_clean = parser.add_argument_group("Очистка данных проекта (требуется --project)")
    grp_clean.add_argument(
        "--clean", action="store_true",
        help="Очистить данные проекта (интерактивный выбор или с --frames/--processed/--all-data)",
    )
    grp_clean.add_argument(
        "--frames", action="store_true",
        help="Удалить только кадры frames/ (с --clean)",
    )
    grp_clean.add_argument(
        "--processed", action="store_true",
        help="Удалить кадры + аннотации frames/ + annotations/ (с --clean)",
    )
    grp_clean.add_argument(
        "--all-data", action="store_true", dest="all_data",
        help="Удалить все данные кроме raw/ (с --clean)",
    )

    args = parser.parse_args()

    # Проверяем, что хотя бы что-то указано
    management_cmd = bool(args.new_project or args.list_projects or args.delete_project)
    # from_step и clean-модификаторы учитываются как pipeline-флаги,
    # чтобы при неверном сочетании аргументов пользователь получал ошибку, а не справку
    pipeline_flags = (
        any(getattr(args, s, False)
            for s in ["all", "load", "annotate", "augment", "balance", "export", "clean"])
        or bool(args.from_step)
        or bool(args.frames or args.processed or args.all_data)
        or bool(args.set_source)
    )

    if not management_cmd and not pipeline_flags:
        parser.print_help()
        sys.exit(0)

    # Шаги пайплайна и очистка требуют --project
    if pipeline_flags and not args.project:
        parser.error(
            "Укажите проект: --project \"название\"\n"
            "Список проектов: python main.py --list-projects"
        )

    # --from: проверяем допустимость значения и совместимость с --all
    if args.from_step is not None:
        if args.from_step not in PIPELINE_STEPS:
            valid = ", ".join(PIPELINE_STEPS)
            parser.error(f"Неверный шаг '{args.from_step}'. Допустимые: {valid}")
        if not args.all:
            parser.error("--from используется только вместе с --all")

    # --frames / --processed / --all-data имеют смысл только вместе с --clean
    if (args.frames or args.processed or args.all_data) and not args.clean:
        parser.error("--frames, --processed и --all-data используются только вместе с --clean")

    # Нельзя указывать несколько уровней очистки одновременно
    if args.clean and sum([args.frames, args.processed, args.all_data]) > 1:
        parser.error("Укажите только один из: --frames, --processed, --all-data")

    return args


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # -----------------------------------------------------------------------
    # Команды управления проектами (не требуют --project)
    # -----------------------------------------------------------------------

    if args.list_projects:
        _cmd_list_projects()
        return

    if args.new_project:
        try:
            _cmd_new_project(args.new_project)
        except ValueError as exc:
            print(f"Ошибка: {exc}")
        except KeyboardInterrupt:
            print("\nПрервано.")
        return

    if args.delete_project:
        try:
            _cmd_delete_project(args.delete_project)
        except FileNotFoundError as exc:
            print(f"Ошибка: {exc}")
        except KeyboardInterrupt:
            print("\nПрервано.")
        return

    # -----------------------------------------------------------------------
    # Пайплайн и очистка — загружаем проект
    # -----------------------------------------------------------------------

    try:
        project = Project.load(args.project)
    except FileNotFoundError as exc:
        print(f"Ошибка: {exc}")
        sys.exit(1)

    # Задание источника данных — лёгкая операция, лог запуска не нужен
    if args.set_source:
        category, key, path_str = args.set_source
        _cmd_set_source(project, category, key, path_str)
        return

    # Единый лог-файл запуска — все модули пишут сюда через корневой логгер
    setup_run_log(project.logs_dir)

    # -----------------------------------------------------------------------
    # Очистка данных проекта
    # -----------------------------------------------------------------------

    if args.clean:
        # Определяем уровень очистки из флагов; None → интерактивное меню
        level = None
        if args.frames:
            level = "frames"
        elif args.processed:
            level = "processed"
        elif args.all_data:
            level = "all_data"

        try:
            _cmd_clean_project(project, level)
        except KeyboardInterrupt:
            print("\nПрервано.")
        return

    # -----------------------------------------------------------------------
    # Запуск шагов пайплайна
    # -----------------------------------------------------------------------

    if args.all:
        if args.from_step:
            # Явно указан стартовый шаг через --from
            start_step = args.from_step
            logger.info(f"Старт с шага '{start_step}' (из --from)")
        else:
            # Читаем start_from из метаданных проекта (сохранён при --new-project)
            meta       = project._read_meta()
            start_step = meta.get("start_from", "load")
            logger.info(f"Старт с шага '{start_step}' (из project.json)")

        start_idx = PIPELINE_STEPS.index(start_step)
        steps     = PIPELINE_STEPS[start_idx:]
    else:
        # Только шаги, явно указанные флагами
        steps = [s for s in PIPELINE_STEPS if getattr(args, s, False)]

    total_steps    = len(steps)
    results        = {}
    pipeline_start = time.time()

    # Заголовок
    print(BANNER)
    logger.info(f"Проект: '{project.name}' | шаги: {steps}")

    for idx, step in enumerate(steps, start=1):
        _step_header(idx, total_steps, STEP_NAMES[step])
        t0 = time.time()

        try:
            if step == "load":
                results["load"] = step_load(project)

            elif step == "annotate":
                results["annotate"] = step_annotate(project)

            elif step == "augment":
                results["augment"] = step_augment(project)

            elif step == "balance":
                results["balance"] = step_balance(project)

            elif step == "export":
                # Формат: из --format или интерактивный запрос
                fmt = args.format if args.format else _ask_export_format()
                results["export"] = step_export(project, fmt)

        except KeyboardInterrupt:
            print("\nПрервано пользователем.")
            logger.warning(f"Пайплайн прерван на шаге '{step}' (KeyboardInterrupt)")
            break

        except Exception as exc:
            logger.error(f"Ошибка на шаге '{step}': {exc}", exc_info=True)
            print(f"\nОшибка на шаге «{STEP_NAMES[step]}»: {exc}")
            break

        elapsed = time.time() - t0
        print(f"   Готово за {_fmt_duration(elapsed)}")
        logger.info(f"Шаг '{step}' завершён за {_fmt_duration(elapsed)}")

    # Итоговый отчёт (только для --all)
    if args.all:
        _print_report(project, results, time.time() - pipeline_start)


if __name__ == "__main__":
    main()
