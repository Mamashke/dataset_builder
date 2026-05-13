# modules/logger.py — единая настройка логирования для всего проекта.
#
# При импорте этого модуля автоматически настраивается корневой логгер:
#   - вывод в консоль (stdout)
#   - вывод в постоянный файл logs/pipeline.log (пишется между запусками)
#
# Из main.py нужно дополнительно вызвать setup_run_log() — он добавляет
# файл logs/run_YYYYMMDD_HHMMSS.log, который охватывает только один запуск.
#
# Все остальные модули делают только:
#   from modules.logger import get_logger
#   logger = get_logger(__name__)

import logging
import sys
from datetime import datetime
from pathlib import Path

import config

# Переводим stdout в UTF-8, чтобы кириллица и Unicode-символы (→, ×, …)
# корректно отображались в консоли Windows независимо от системной кодировки.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Формат строки лога: "2026-05-09 14:05:32 [INFO] [loader] Текст сообщения"
# %(module)s — имя файла-источника вызова без расширения (loader, augmentor, …)
_FORMAT  = "%(asctime)s [%(levelname)s] [%(module)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Флаг, защищающий от повторной инициализации при множественном импорте
_initialized = False


def _setup_root_logger() -> None:
    """Настраивает корневой логгер один раз при первом импорте модуля.

    Добавляет два обработчика:
    - StreamHandler  → stdout (видно в консоли при любом запуске)
    - FileHandler    → logs/pipeline.log (сквозной журнал всех запусков)
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Создаём папку logs/ если её нет
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FORMAT, _DATEFMT)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # --- Консольный обработчик ---
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # --- Постоянный файловый обработчик ---
    # mode="a" — дописываем в конец, не затираем историю предыдущих запусков
    pipeline_log = config.LOGS_DIR / "pipeline.log"
    file_handler = logging.FileHandler(pipeline_log, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def setup_run_log() -> Path:
    """Добавляет лог-файл с временной меткой для текущего запуска main.py.

    Должна вызываться один раз в начале main.py.
    Создаёт файл вида: logs/run_20260509_140532.log

    Returns:
        Путь к созданному файлу.
    """
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = config.LOGS_DIR / f"run_{timestamp}.log"

    formatter = logging.Formatter(_FORMAT, _DATEFMT)
    run_handler = logging.FileHandler(run_log_path, mode="w", encoding="utf-8")
    run_handler.setFormatter(formatter)

    logging.getLogger().addHandler(run_handler)

    logging.getLogger(__name__).info(f"Лог текущего запуска: {run_log_path}")
    return run_log_path


def get_logger(name: str, logs_dir: Path = None) -> logging.Logger:
    """Возвращает логгер с указанным именем.

    Если передан logs_dir, дополнительно добавляет файловый обработчик,
    который пишет в logs_dir/{короткое_имя}.log (например, loader.log).
    Повторные вызовы с тем же файлом не создают дублирующих обработчиков.

    Использование в модулях:
        logger = get_logger(__name__)                         # только глобальный лог
        logger = get_logger("loader", project.logs_dir)      # + лог проекта

    Args:
        name:     имя логгера, обычно __name__ (например, "modules.loader").
        logs_dir: если передан — папка для проектного лог-файла.
                  Папка создаётся автоматически.

    Returns:
        Настроенный экземпляр logging.Logger.
    """
    log = logging.getLogger(name)

    if logs_dir is not None:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Имя файла — последняя часть имени логгера: "modules.loader" → "loader.log"
        short_name = name.split(".")[-1]
        log_file   = logs_dir / f"{short_name}.log"

        # Проверяем, что обработчик для этого файла ещё не добавлен,
        # чтобы не дублировать записи при повторных вызовах
        existing_paths = {
            Path(h.baseFilename).resolve()
            for h in log.handlers
            if isinstance(h, logging.FileHandler)
        }
        if log_file.resolve() not in existing_paths:
            formatter   = logging.Formatter(_FORMAT, _DATEFMT)
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            log.addHandler(file_handler)

    return log


# Инициализируем при импорте — до того как любой модуль создаст свой logger
_setup_root_logger()
