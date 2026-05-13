# modules/project.py — система управления проектами dataset_builder.
#
# Каждый проект — изолированная папка со своей структурой данных:
#   projects/{name}/
#       raw/real/           — исходные видеозаписи реальной камеры
#       raw/airsim/         — видеозаписи из AirSim
#       frames/real/        — извлечённые кадры из реальных видео
#       frames/airsim/      — извлечённые кадры из AirSim
#       annotations/        — файлы разметки (YOLO txt)
#       dataset/images/     — финальный датасет, изображения
#       dataset/labels/     — финальный датасет, метки
#       export/             — экспортированные файлы (data.yaml, annotations.json)
#       logs/               — лог-файлы проекта
#       project.json        — метаданные проекта
#
# Публичный интерфейс:
#   Project.create(name)    — создать новый проект
#   Project.load(name)      — загрузить существующий
#   Project.list_all()      — список всех проектов
#   Project.delete(name)    — удалить проект
#   project.update_step()   — обновить текущий шаг в project.json
#   project.update_stats()  — обновить статистику в project.json

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import config
from modules.logger import get_logger

logger = get_logger(__name__)

# Допустимые символы в имени проекта:
# латиница, кириллица, цифры, дефис, подчёркивание — без пробелов и спецсимволов
_NAME_RE = re.compile(r'^[a-zA-Z0-9а-яА-ЯёЁ_\-]+$')

# Имя файла метаданных внутри папки проекта
_META_FILE = "project.json"


class Project:
    """Описывает один проект и предоставляет все его пути.

    Не создавайте экземпляр напрямую — используйте Project.create()
    или Project.load().
    """

    def __init__(self, name: str) -> None:
        """Инициализирует атрибуты путей для проекта с данным именем.

        Args:
            name: имя проекта (уже проверенное).
        """
        self.name = name
        self.dir  = config.PROJECTS_DIR / name

        # Исходные видеозаписи
        self.raw_real_dir   = self.dir / "raw"    / "real"
        self.raw_airsim_dir = self.dir / "raw"    / "airsim"

        # Извлечённые кадры
        self.frames_real_dir   = self.dir / "frames" / "real"
        self.frames_airsim_dir = self.dir / "frames" / "airsim"

        # Файлы разметки
        self.annotations_dir = self.dir / "annotations"

        # Финальный датасет
        self.dataset_images_dir = self.dir / "dataset" / "images"
        self.dataset_labels_dir = self.dir / "dataset" / "labels"

        # Экспорт и логи
        self.export_dir = self.dir / "export"
        self.logs_dir   = self.dir / "logs"

    # -----------------------------------------------------------------------
    # Внутренние утилиты
    # -----------------------------------------------------------------------

    @property
    def _meta_path(self) -> Path:
        """Путь к файлу метаданных проекта."""
        return self.dir / _META_FILE

    def _read_meta(self) -> dict:
        """Читает и возвращает содержимое project.json."""
        return json.loads(self._meta_path.read_text(encoding="utf-8"))

    def _write_meta(self, data: dict) -> None:
        """Записывает словарь в project.json с отступами."""
        self._meta_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _create_dirs(self) -> None:
        """Создаёт все папки структуры проекта."""
        dirs = [
            self.raw_real_dir,
            self.raw_airsim_dir,
            self.frames_real_dir,
            self.frames_airsim_dir,
            self.annotations_dir,
            self.dataset_images_dir,
            self.dataset_labels_dir,
            self.export_dir,
            self.logs_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Классовые методы: create / load / list_all / delete
    # -----------------------------------------------------------------------

    @classmethod
    def _validate_name(cls, name: str) -> None:
        """Проверяет имя проекта на допустимость.

        Args:
            name: предполагаемое имя проекта.

        Raises:
            ValueError: если имя пустое или содержит недопустимые символы.
        """
        if not name or not name.strip():
            raise ValueError("Имя проекта не может быть пустым.")
        if not _NAME_RE.match(name):
            raise ValueError(
                f"Недопустимое имя проекта: '{name}'.\n"
                "Разрешены только буквы (латиница/кириллица), цифры, "
                "дефис (-) и подчёркивание (_). Пробелы не допускаются."
            )

    @classmethod
    def create(cls, name: str) -> "Project":
        """Создаёт новый проект с указанным именем.

        Если проект с таким именем уже существует, спрашивает пользователя
        о перезаписи. При отказе возвращает существующий проект без изменений.

        Args:
            name: имя нового проекта.

        Returns:
            Объект Project с созданной структурой папок.

        Raises:
            ValueError: если имя содержит недопустимые символы.
        """
        cls._validate_name(name)
        project = cls(name)
        config.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

        # Проект уже существует — спрашиваем что делать
        if project.dir.exists():
            print(f"\nПроект '{name}' уже существует: {project.dir}")
            while True:
                raw = input("Перезаписать? (yes/no): ").strip().lower()
                if raw == "yes":
                    logger.info(f"Пользователь подтвердил перезапись проекта '{name}'")
                    shutil.rmtree(project.dir)
                    break
                if raw == "no":
                    logger.info(f"Перезапись отменена — загружаем существующий проект '{name}'")
                    return cls.load(name)
                print("  Введите 'yes' или 'no'.")

        # Создаём папки и сохраняем метаданные
        project._create_dirs()

        meta = {
            "name":         name,
            "created":      datetime.now().isoformat(timespec="seconds"),
            "current_step": None,
            "stats":        {},
        }
        project._write_meta(meta)

        logger.info(f"Проект '{name}' создан: {project.dir}")
        print(f"Проект '{name}' создан.")
        return project

    @classmethod
    def load(cls, name: str) -> "Project":
        """Загружает существующий проект по имени.

        Args:
            name: имя проекта.

        Returns:
            Объект Project.

        Raises:
            FileNotFoundError: если проект не найден, с подсказкой о доступных проектах.
        """
        project = cls(name)

        if not project.dir.exists():
            available = [p.name for p in config.PROJECTS_DIR.iterdir()
                         if p.is_dir()] if config.PROJECTS_DIR.exists() else []
            hint = (
                f"Доступные проекты: {', '.join(available)}"
                if available else
                "Существующих проектов не найдено. Создайте новый: Project.create(name)"
            )
            raise FileNotFoundError(
                f"Проект '{name}' не найден.\n{hint}"
            )

        if not project._meta_path.exists():
            raise FileNotFoundError(
                f"Файл метаданных не найден: {project._meta_path}\n"
                "Папка проекта повреждена."
            )

        logger.info(f"Проект '{name}' загружен из {project.dir}")
        return project

    @classmethod
    def list_all(cls) -> List[dict]:
        """Возвращает список всех проектов с их метаданными.

        Returns:
            Список словарей вида:
            [{"name": "proj1", "created": "2026-05-13T14:30:00",
              "current_step": "augment", "stats": {...}}, ...]
            Если проектов нет — пустой список.
        """
        if not config.PROJECTS_DIR.exists():
            return []

        projects = []
        for entry in sorted(config.PROJECTS_DIR.iterdir()):
            if not entry.is_dir():
                continue
            meta_file = entry / _META_FILE
            if not meta_file.exists():
                # Папка без project.json — пропускаем, но предупреждаем
                logger.warning(f"Папка без метаданных, пропускаем: {entry.name}")
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                projects.append(meta)
            except json.JSONDecodeError:
                logger.warning(f"Повреждённый project.json: {entry.name}")

        return projects

    @classmethod
    def delete(cls, name: str) -> None:
        """Удаляет проект после подтверждения пользователя.

        Args:
            name: имя проекта для удаления.

        Raises:
            FileNotFoundError: если проект не найден.
        """
        project = cls(name)

        if not project.dir.exists():
            raise FileNotFoundError(f"Проект '{name}' не найден.")

        print(f"\nУдаление проекта '{name}': {project.dir}")
        print("Это действие необратимо — все файлы проекта будут удалены.")
        while True:
            raw = input("Подтвердите удаление (yes/no): ").strip().lower()
            if raw == "yes":
                shutil.rmtree(project.dir)
                logger.info(f"Проект '{name}' удалён.")
                print(f"Проект '{name}' успешно удалён.")
                return
            if raw == "no":
                logger.info(f"Удаление проекта '{name}' отменено пользователем.")
                print("Удаление отменено.")
                return
            print("  Введите 'yes' или 'no'.")

    # -----------------------------------------------------------------------
    # Методы обновления метаданных
    # -----------------------------------------------------------------------

    def update_step(self, step: Optional[str]) -> None:
        """Обновляет поле current_step в project.json.

        Вызывается из main.py после завершения каждого шага пайплайна,
        чтобы можно было продолжить работу с того места, где остановились.

        Args:
            step: имя текущего шага ("load", "annotate", "augment",
                  "balance", "export") или None после завершения всего пайплайна.
        """
        meta = self._read_meta()
        meta["current_step"] = step
        self._write_meta(meta)
        logger.info(f"Проект '{self.name}': текущий шаг → {step}")

    def update_stats(self, stats: dict) -> None:
        """Обновляет (дополняет) поле stats в project.json.

        Новые данные объединяются с существующими, поэтому каждый вызов
        не затирает статистику предыдущих шагов.

        Args:
            stats: словарь с новыми данными статистики.
                   Например: {"augment": {"created": 5000}}.
        """
        meta = self._read_meta()
        meta["stats"].update(stats)
        self._write_meta(meta)
        logger.info(f"Проект '{self.name}': статистика обновлена → {list(stats)}")

    # -----------------------------------------------------------------------
    # Строковое представление
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        meta = self._read_meta() if self._meta_path.exists() else {}
        step = meta.get("current_step") or "—"
        return f"Project(name='{self.name}', step='{step}', dir='{self.dir}')"
