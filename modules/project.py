# modules/project.py — система управления проектами dataset_builder.
#
# Каждый проект — изолированная папка со своей структурой данных:
#   projects/{name}/
#       frames/real/        — извлечённые кадры из реальных видео
#       frames/airsim/      — извлечённые кадры из AirSim
#       annotations/        — файлы разметки (YOLO txt)
#       dataset/images/     — финальный датасет, изображения
#       dataset/labels/     — финальный датасет, метки
#       export/             — экспортированные файлы (data.yaml, annotations.json)
#       logs/               — лог-файлы проекта
#       project.json        — метаданные проекта
#
# Пути к входным данным (видео, кадры и т.д.) хранятся в data_sources и
# могут указывать на произвольные места файловой системы — не только внутри
# папки проекта. Управление через set_source() / get_source().
#
# Публичный интерфейс:
#   Project.create(name)          — создать новый проект
#   Project.load(name)            — загрузить существующий
#   Project.list_all()            — список всех проектов
#   Project.delete(name)          — удалить проект
#   project.set_source(cat, key, path) — задать путь в data_sources
#   project.get_source(cat, key)  — получить путь из data_sources
#   project.validate_sources()    — проверить доступность всех путей
#   project.update_step()         — обновить текущий шаг в project.json
#   project.update_stats()        — обновить статистику в project.json

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

# Пустая структура data_sources — используется при создании проекта
# и как эталон допустимых категорий / ключей в set_source / get_source
_DEFAULT_DATA_SOURCES: dict = {
    "videos":      {"real": None, "airsim": None},
    "frames":      {"real": None, "airsim": None},
    "annotations": {"real": None, "airsim": None},
    "dataset":     {"images": None, "labels": None},
}


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

        # Извлечённые кадры (внутри папки проекта)
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

        # GAN: веса обученных моделей и сэмплы промежуточных результатов
        self.gan_model_dir   = self.dir / "gan_model"
        self.gan_samples_dir = self.dir / "gan_samples"

        # Настраиваемые пути к источникам данных.
        # Хранятся в project.json и могут указывать на любое место файловой системы.
        # Инициализируем пустой структурой, затем заполняем из project.json (если он есть).
        import copy
        self.data_sources: dict = copy.deepcopy(_DEFAULT_DATA_SOURCES)
        if self._meta_path.exists():
            self._load_data_sources()

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

    def _load_data_sources(self) -> None:
        """Загружает data_sources из project.json в self.data_sources."""
        meta = self._read_meta()
        saved = meta.get("data_sources", {})
        for cat, keys in saved.items():
            if cat in self.data_sources and isinstance(keys, dict):
                for key, val in keys.items():
                    if key in self.data_sources[cat]:
                        self.data_sources[cat][key] = Path(val) if val is not None else None

    def _create_dirs(self) -> None:
        """Создаёт все рабочие папки структуры проекта.

        raw/ не создаётся — исходные видео могут лежать в произвольном месте
        и регистрируются через set_source("videos", ...).
        """
        dirs = [
            self.frames_real_dir,
            self.frames_airsim_dir,
            self.annotations_dir,
            self.dataset_images_dir,
            self.dataset_labels_dir,
            self.export_dir,
            self.logs_dir,
            self.gan_model_dir,
            self.gan_samples_dir,
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

        import copy
        meta = {
            "name":         name,
            "created":      datetime.now().isoformat(timespec="seconds"),
            "current_step": None,
            "stats":        {},
            "data_sources": copy.deepcopy(_DEFAULT_DATA_SOURCES),
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
    # Управление источниками данных
    # -----------------------------------------------------------------------

    def set_source(self, category: str, key: str, path) -> None:
        """Задаёт путь в data_sources и сохраняет его в project.json.

        Args:
            category: категория ("videos", "frames", "annotations", "dataset").
            key:      ключ внутри категории ("real", "airsim", "images", "labels").
            path:     путь к источнику данных (str или Path) или None для сброса.

        Raises:
            KeyError: если category или key не существуют в _DEFAULT_DATA_SOURCES.
        """
        if category not in _DEFAULT_DATA_SOURCES:
            raise KeyError(
                f"Неизвестная категория '{category}'. "
                f"Допустимые: {', '.join(_DEFAULT_DATA_SOURCES)}"
            )
        if key not in _DEFAULT_DATA_SOURCES[category]:
            raise KeyError(
                f"Неизвестный ключ '{key}' в категории '{category}'. "
                f"Допустимые: {', '.join(_DEFAULT_DATA_SOURCES[category])}"
            )

        self.data_sources[category][key] = Path(path) if path is not None else None

        meta = self._read_meta()
        if "data_sources" not in meta:
            import copy
            meta["data_sources"] = copy.deepcopy(_DEFAULT_DATA_SOURCES)
        if category not in meta["data_sources"]:
            meta["data_sources"][category] = {}
        meta["data_sources"][category][key] = str(path) if path is not None else None
        self._write_meta(meta)
        logger.info(f"Проект '{self.name}': data_sources[{category}][{key}] → {path}")

    def get_source(self, category: str, key: str) -> Optional[Path]:
        """Возвращает путь из data_sources или None, если не задан.

        Args:
            category: категория ("videos", "frames", "annotations", "dataset").
            key:      ключ внутри категории.

        Returns:
            Path или None.

        Raises:
            KeyError: если category или key не существуют в _DEFAULT_DATA_SOURCES.
        """
        if category not in _DEFAULT_DATA_SOURCES:
            raise KeyError(
                f"Неизвестная категория '{category}'. "
                f"Допустимые: {', '.join(_DEFAULT_DATA_SOURCES)}"
            )
        if key not in _DEFAULT_DATA_SOURCES[category]:
            raise KeyError(
                f"Неизвестный ключ '{key}' в категории '{category}'. "
                f"Допустимые: {', '.join(_DEFAULT_DATA_SOURCES[category])}"
            )
        return self.data_sources[category][key]

    def validate_sources(self) -> bool:
        """Проверяет существование всех непустых путей в data_sources.

        Returns:
            True, если все заданные пути существуют; False, если есть недоступные.
        """
        ok = True
        for category, keys in self.data_sources.items():
            for key, path in keys.items():
                if path is None:
                    continue
                if not Path(path).exists():
                    logger.warning(
                        f"Проект '{self.name}': путь недоступен — "
                        f"data_sources[{category}][{key}] = {path}"
                    )
                    ok = False
        return ok

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
