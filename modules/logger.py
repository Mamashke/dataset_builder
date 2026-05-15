# modules/logger.py — единая настройка логирования для всего проекта.
#
# При импорте настраивается корневой логгер с выводом в консоль.
# Файл для текущего запуска создаётся один раз через setup_run_log(logs_dir):
#   projects/{name}/logs/run_YYYYMMDD_HHMMSS.log
#
# Все модули делают только:
#   from modules.logger import get_logger
#   logger = get_logger(__name__)          # без logs_dir — только консоль
#   # или внутри функций:
#   get_logger(__name__, project.logs_dir) # logs_dir принимается, но игнорируется —
#                                          # файловый обработчик уже на корневом логгере

import logging
import sys
from datetime import datetime
from pathlib import Path

# Переводим stdout в UTF-8, чтобы кириллица и Unicode корректно отображались
# в консоли Windows независимо от системной кодировки.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Формат: "2026-05-13 14:00:01 [INFO] [loader] Текст сообщения"
_FORMAT  = "%(asctime)s [%(levelname)s] [%(module)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _setup_root_logger() -> None:
    """Настраивает корневой логгер один раз при первом импорте.

    Добавляет только консольный обработчик (StreamHandler → stdout).
    Файловый обработчик добавляется позже через setup_run_log().
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    formatter = logging.Formatter(_FORMAT, _DATEFMT)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)


def setup_run_log(logs_dir: Path) -> Path:
    """Добавляет единый файловый обработчик для текущего запуска пайплайна.

    Создаёт один файл run_YYYYMMDD_HHMMSS.log в папке logs_dir проекта.
    Все логгеры (через наследование от корневого) автоматически пишут туда.
    Вызывается один раз из main.py сразу после загрузки проекта.

    Args:
        logs_dir: папка для лог-файла (обычно project.logs_dir).

    Returns:
        Путь к созданному лог-файлу.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = logs_dir / f"run_{timestamp}.log"

    formatter    = logging.Formatter(_FORMAT, _DATEFMT)
    run_handler  = logging.FileHandler(run_log_path, mode="w", encoding="utf-8")
    run_handler.setFormatter(formatter)

    logging.getLogger().addHandler(run_handler)

    logging.getLogger(__name__).info(f"Лог текущего запуска: {run_log_path}")
    return run_log_path


def get_logger(name: str, logs_dir: Path = None) -> logging.Logger:
    """Возвращает логгер с указанным именем.

    logs_dir принимается для обратной совместимости с вызовами в модулях,
    но не создаёт дополнительных файлов — все записи попадают в единый
    run-файл, настроенный через setup_run_log().

    Args:
        name:     имя логгера (обычно __name__).
        logs_dir: игнорируется (принимается для совместимости).

    Returns:
        Экземпляр logging.Logger.
    """
    return logging.getLogger(name)


# Инициализируем при импорте — до того как любой модуль создаст свой logger
_setup_root_logger()
