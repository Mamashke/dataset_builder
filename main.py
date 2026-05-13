# main.py — единая точка входа пайплайна dataset_builder.
#
# Использование:
#   python main.py --all                    # весь пайплайн
#   python main.py --load                   # только загрузка видео
#   python main.py --annotate               # только разметка (интерактивно)
#   python main.py --augment                # только аугментация
#   python main.py --balance                # только балансировка
#   python main.py --export                 # экспорт (спросить формат)
#   python main.py --export --format coco   # экспорт в конкретный формат

import argparse
import sys
import time

# Лог текущего запуска создаётся первым — до импорта остальных модулей,
# чтобы все их сообщения попали в run_*.log
from modules.logger import get_logger, setup_run_log

run_log = setup_run_log()
logger  = get_logger(__name__)

import config
from modules.loader    import load_videos
from modules.annotator import run_interactive as annotate_interactive
from modules.augmentor import augment_dataset
from modules.balancer  import build as balance_build
from modules.exporter  import export as do_export

# ---------------------------------------------------------------------------
# Вспомогательные функции отображения
# ---------------------------------------------------------------------------

BANNER = """\
================================
   dataset_builder v1.0
   Конструктор обучающей выборки
================================"""

STEP_NAMES = {
    "load":     "Загрузка видео",
    "annotate": "Разметка кадров",
    "augment":  "Аугментация",
    "balance":  "Балансировка",
    "export":   "Экспорт датасета",
}


def _print(text: str = "") -> None:
    """Выводит строку в консоль и логирует её."""
    print(text)
    logger.info(text)


def _fmt_duration(seconds: float) -> str:
    """Форматирует количество секунд в читаемую строку.

    Args:
        seconds: время в секундах.

    Returns:
        Строка вида "14 минут 32 секунды" или "45 секунд".
    """
    total = int(seconds)
    mins, secs = divmod(total, 60)
    if mins:
        return f"{mins} мин {secs} сек"
    return f"{secs} сек"


def _step_header(index: int, total: int, name: str) -> None:
    """Печатает заголовок шага пайплайна."""
    print(f"\n[{index}/{total}] {name}...")
    logger.info(f"[{index}/{total}] {name}")


# ---------------------------------------------------------------------------
# Запрос экспортного формата (если не передан через --format)
# ---------------------------------------------------------------------------

def _ask_export_format() -> str:
    """Интерактивно запрашивает формат экспорта.

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

def step_load() -> dict:
    """Загрузка видео — интерактивный выбор файлов через load_videos()."""
    return load_videos(config.RAW_REAL_DIR, source="real")


def step_annotate() -> dict:
    """Разметка кадров — полностью интерактивная через run_interactive()."""
    return annotate_interactive()


def step_augment() -> dict:
    """Аугментация всех кадров из источника real с параметрами по умолчанию."""
    aug_types = ["fog", "rain", "noise", "blur", "brightness"]
    return augment_dataset(aug_types, intensity=0.5, sources=["real"])


def step_balance() -> dict:
    """Балансировка и сборка финального датасета."""
    return balance_build(sources=["real"], overwrite=True)


def step_export(fmt: str) -> dict:
    """Экспорт датасета в указанный формат."""
    return do_export(format=fmt)


# ---------------------------------------------------------------------------
# Итоговый отчёт
# ---------------------------------------------------------------------------

def _print_report(results: dict, total_seconds: float) -> None:
    """Выводит финальный отчёт по всем выполненным шагам.

    Args:
        results:       словарь {шаг: статистика} по каждому выполненному шагу.
        total_seconds: общее время работы пайплайна в секундах.
    """
    print("\n================================")
    print("   ИТОГОВЫЙ ОТЧЁТ")

    if "load" in results:
        r = results["load"]
        print(f"   Загрузка:     videos={r.get('videos', '—')}, frames={r.get('frames', '—')}")

    if "annotate" in results:
        r = results["annotate"]
        print(f"   Разметка:     размечено={r.get('annotated', '—')}, "
              f"объектов={r.get('total_objects', '—')}")

    if "augment" in results:
        r = results["augment"]
        print(f"   Аугментация:  создано {r.get('created', '—')} кадров")

    if "balance" in results:
        r = results["balance"]
        print(f"   Балансировка: финальных кадров {r.get('after_balance', '—')} "
              f"(pos={r.get('positives', '—')}, neg={r.get('negatives', '—')})")

    if "export" in results:
        r = results["export"]
        fmt_label = "YOLO" if "yaml_path" in r else "COCO"
        imgs = r.get("images", "—")
        anns = f", аннотаций={r.get('annotations', '—')}" if "annotations" in r else ""
        print(f"   Экспорт:      формат {fmt_label}, изображений={imgs}{anns}")

    print(f"   Время работы: {_fmt_duration(total_seconds)}")
    print("================================")

    # Дублируем в лог
    logger.info("ИТОГОВЫЙ ОТЧЁТ: " + str(results))
    logger.info(f"Время работы: {_fmt_duration(total_seconds)}")


# ---------------------------------------------------------------------------
# Разбор аргументов
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="dataset_builder — конструктор обучающей выборки для YOLO",
    )
    parser.add_argument("--all",      action="store_true", help="Запустить весь пайплайн")
    parser.add_argument("--load",     action="store_true", help="Загрузка видео")
    parser.add_argument("--annotate", action="store_true", help="Разметка кадров (интерактивно)")
    parser.add_argument("--augment",  action="store_true", help="Аугментация кадров")
    parser.add_argument("--balance",  action="store_true", help="Балансировка и сборка датасета")
    parser.add_argument("--export",   action="store_true", help="Экспорт датасета")
    parser.add_argument("--format",   choices=["yolo", "coco"], default=None,
                        help="Формат экспорта: yolo или coco (используется с --export)")

    args = parser.parse_args()

    # Если не передан ни один флаг — показываем справку и выходим
    if not any([args.all, args.load, args.annotate, args.augment, args.balance, args.export]):
        parser.print_help()
        sys.exit(0)

    return args


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    args     = _parse_args()
    results  = {}        # статистика по каждому шагу
    pipeline_start = time.time()

    # Определяем список шагов для текущего запуска
    run_all = args.all
    steps = (
        ["load", "annotate", "augment", "balance", "export"]
        if run_all else
        [s for s in ["load", "annotate", "augment", "balance", "export"]
         if getattr(args, s, False)]
    )
    total_steps = len(steps)

    # --- Заголовок ---
    if run_all:
        print(BANNER)
    logger.info(f"Запуск: шаги={steps}")

    # --- Выполнение шагов ---
    for idx, step in enumerate(steps, start=1):
        _step_header(idx, total_steps, STEP_NAMES[step])
        t0 = time.time()

        try:
            if step == "load":
                results["load"] = step_load()

            elif step == "annotate":
                results["annotate"] = step_annotate()

            elif step == "augment":
                results["augment"] = step_augment()

            elif step == "balance":
                results["balance"] = step_balance()

            elif step == "export":
                # Формат: из --format или интерактивно
                fmt = args.format if args.format else _ask_export_format()
                results["export"] = step_export(fmt)

        except KeyboardInterrupt:
            print("\nПрервано пользователем.")
            logger.warning("Пайплайн прерван пользователем (KeyboardInterrupt)")
            break
        except Exception as exc:
            logger.error(f"Ошибка на шаге '{step}': {exc}", exc_info=True)
            print(f"\nОшибка: {exc}")
            break

        elapsed = time.time() - t0
        logger.info(f"Шаг '{step}' завершён за {_fmt_duration(elapsed)}")

    # --- Итоговый отчёт ---
    _print_report(results, time.time() - pipeline_start)


if __name__ == "__main__":
    main()
