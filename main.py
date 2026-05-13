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

import argparse
import sys
import time

# Лог текущего запуска создаётся первым — до импорта остальных модулей,
# чтобы все их сообщения попали в run_*.log
from modules.logger import get_logger, setup_run_log

run_log = setup_run_log()
logger  = get_logger(__name__)

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

# Порядок шагов пайплайна — он фиксирован и используется в нескольких местах
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
    r_real   = load_videos(project, source="real")
    r_airsim = load_videos(project, source="airsim")

    combined = {
        "videos": r_real["videos"] + r_airsim["videos"],
        "frames": r_real["frames"] + r_airsim["frames"],
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
        choices=PIPELINE_STEPS,
        help=f"Начать с указанного шага (с --all): {PIPELINE_STEPS}",
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

    args = parser.parse_args()

    # Проверяем, что хотя бы что-то указано
    management_cmd = bool(args.new_project or args.list_projects or args.delete_project)
    pipeline_flags = any(getattr(args, s, False)
                         for s in ["all", "load", "annotate", "augment", "balance", "export"])

    if not management_cmd and not pipeline_flags:
        parser.print_help()
        sys.exit(0)

    # Шаги пайплайна требуют --project
    if pipeline_flags and not args.project:
        parser.error(
            "Укажите проект: --project \"название\"\n"
            "Список проектов: python main.py --list-projects"
        )

    # --from имеет смысл только с --all
    if args.from_step and not args.all:
        parser.error("--from используется только вместе с --all")

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
    # Пайплайн — загружаем проект
    # -----------------------------------------------------------------------

    try:
        project = Project.load(args.project)
    except FileNotFoundError as exc:
        print(f"Ошибка: {exc}")
        sys.exit(1)

    # Определяем список шагов для запуска
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

    # -----------------------------------------------------------------------
    # Выполнение шагов
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Итоговый отчёт (только для --all)
    # -----------------------------------------------------------------------

    if args.all:
        _print_report(project, results, time.time() - pipeline_start)


if __name__ == "__main__":
    main()
