# gui.py — десктопное GUI приложение dataset_builder на PyQt6
#
# Запуск:   python gui.py
# Требует:  pip install PyQt6

import json
import logging
import queue
import shutil
import sys
from pathlib import Path

import modules.loader as _loader_module

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget, QCheckBox,
)

import config
from modules.annotator import annotate as run_annotate
from modules.augmentor import augment_dataset
from modules.balancer import build as balance_build
from modules.exporter import export as do_export
from modules.logger import get_logger, setup_run_log
from modules.project import Project
from main import step_load


# ─────────────────────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────────────────────

PIPELINE_STEPS = ["load", "annotate", "augment", "balance", "export"]

# Варианты пайплайна в зависимости от метода расширения датасета
_PIPELINE_STEPS_AUGMENT = ["load", "annotate", "augment",     "balance", "export"]
_PIPELINE_STEPS_GAN     = ["load", "annotate", "generate",    "balance", "export"]
_PIPELINE_STEPS_SD      = ["load", "annotate", "generate_sd", "compose", "balance", "export"]

STEP_NAMES = {
    "load":        "Загрузка видео",
    "annotate":    "Разметка кадров",
    "augment":     "Аугментация",
    "generate":    "Генерация (GAN)",
    "generate_sd": "Генерация SD фонов",
    "compose":     "Copy-Paste компоновка",
    "balance":     "Балансировка",
    "export":      "Экспорт датасета",
}

# Папки, очищаемые на каждом уровне
_CLEAN_DIRS = {
    "frames":    ["frames"],
    "processed": ["frames", "annotations"],
    "all_data":  ["frames", "annotations", "dataset", "export"],
}

# Ключи stats для сброса
_CLEAN_STATS_KEYS = {
    "frames":    ["load", "augmented_frames"],
    "processed": ["load", "augmented_frames", "annotated"],
    "all_data":  None,
}

# Пути data_sources для сброса
_CLEAN_SOURCES = {
    "frames":    [("frames", "real"), ("frames", "airsim")],
    "processed": [("frames", "real"), ("frames", "airsim"),
                  ("annotations", "real"), ("annotations", "airsim")],
    "all_data":  [("frames", "real"), ("frames", "airsim"),
                  ("annotations", "real"), ("annotations", "airsim"),
                  ("dataset", "images"), ("dataset", "labels")],
}

SETTINGS_FILE = config.BASE_DIR / "gui_settings.json"


# ─────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    if n >= 1024 ** 3: return f"{n / 1024 ** 3:.1f} ГБ"
    if n >= 1024 ** 2: return f"{n / 1024 ** 2:.0f} МБ"
    if n >= 1024:      return f"{n / 1024:.0f} КБ"
    return f"{n} Б"


def _dir_size(*paths: Path) -> int:
    total = 0
    for p in paths:
        if p.exists():
            for f in p.rglob("*"):
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except OSError:
                        pass
    return total


def _clear_dir(path: Path) -> None:
    """Удаляет содержимое папки, сохраняя саму папку."""
    if not path.exists():
        return
    for item in path.iterdir():
        shutil.rmtree(item) if item.is_dir() else item.unlink()


def _load_settings() -> dict:
    defaults = {
        "frame_sample_rate":  config.FRAME_SAMPLE_RATE,
        "pos_neg_ratio":      config.POS_NEG_RATIO,
        "gan_image_size":     config.GAN_IMAGE_SIZE,
        "gan_batch_size":     config.GAN_BATCH_SIZE,
        "augment_intensity":  0.5,
        "aug_types":          ["fog", "rain", "noise", "blur", "brightness"],
        "annotate_model":     "yolov8n.pt",
        "annotate_conf":      0.25,
        "annotate_mode":      "auto",
        "annotate_sources":   "all",
        "annotate_overwrite": False,
        "export_format":      "yolo",
        "expansion_method":   "augment",   # "augment", "gan" или "sd"
        "gan_count":          200,          # кадров при generate (GAN)
        "sd_count":           200,          # кадров при generate_sd (Stable Diffusion)
        "sd_bg_type":         "all",        # тип фонов: open / forest / all
    }
    if SETTINGS_FILE.exists():
        try:
            defaults.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return defaults


def _save_settings(settings: dict) -> None:
    SETTINGS_FILE.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _close_project_log_handlers() -> None:
    """Закрывает и снимает все файловые обработчики корневого логгера.

    На Windows открытый FileHandler удерживает лог-файл — папку проекта
    нельзя удалить (WinError 32). Вызывать перед shutil.rmtree(project.dir).
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root.removeHandler(handler)


# ─────────────────────────────────────────────────────────────
# Инфраструктура логирования
# ─────────────────────────────────────────────────────────────

class LogEmitter(QObject):
    """Переносит строки лога из любого потока в GUI через Qt-сигнал."""
    message = pyqtSignal(str)


_log_emitter = LogEmitter()


class QtLogHandler(logging.Handler):
    """Обработчик logging, направляющий записи в виджет лога через _log_emitter."""

    def __init__(self):
        super().__init__()
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _log_emitter.message.emit(self.format(record))
        except Exception:
            pass


# Подключаем к корневому логгеру один раз при импорте модуля
_qt_handler = QtLogHandler()
logging.getLogger().addHandler(_qt_handler)
logging.getLogger().setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────
# Рабочий поток пайплайна
# ─────────────────────────────────────────────────────────────

class PipelineWorker(QThread):
    """Выполняет шаги пайплайна в фоновом потоке, чтобы GUI не зависал."""

    finished             = pyqtSignal(dict)
    error                = pyqtSignal(str)
    stopped              = pyqtSignal()           # пользователь запросил остановку
    # (номер_шага, всего_шагов, название) — для обновления прогресс-бара
    step_started         = pyqtSignal(int, int, str)
    # Сигнал для GUI: воркер нашёл видеофайлы и ждёт выбора пользователя
    select_videos_signal = pyqtSignal(list)

    def __init__(
        self,
        project:            Project,
        steps:              list,
        augment_intensity:  float = 0.5,
        export_format:      str   = "yolo",
        annotate_model:     str   = "yolov8n.pt",
        annotate_conf:      float = 0.25,
        annotate_overwrite: bool  = False,
        annotate_mode:      str   = "auto",
        annotate_sources:   str   = "all",
        aug_types:          list  = None,
        gan_count:          int   = 200,
        sd_count:           int   = 200,
        sd_bg_type:         str   = "all",
    ):
        super().__init__()
        self.project            = project
        self.steps              = steps
        self.augment_intensity  = augment_intensity
        self.export_format      = export_format
        self.annotate_model     = annotate_model
        self.annotate_conf      = annotate_conf
        self.annotate_overwrite = annotate_overwrite
        self.annotate_mode      = annotate_mode
        self.annotate_sources   = annotate_sources
        self.aug_types          = aug_types or ["fog", "rain", "noise", "blur", "brightness"]
        self.gan_count          = gan_count
        self.sd_count           = sd_count
        self.sd_bg_type         = sd_bg_type
        # Очередь для получения ответа от GUI после показа диалога выбора видео
        self._video_queue: queue.Queue = queue.Queue()
        # Флаг остановки: GUI ставит True, воркер проверяет между шагами
        self._stop_requested: bool = False

    def request_stop(self) -> None:
        """Устанавливает флаг остановки — воркер остановится перед следующим шагом."""
        self._stop_requested = True

    def _gui_select_videos(self, videos: list) -> list:
        """Вызывается из рабочего потока вместо input() в loader._select_videos.

        Отправляет сигнал в главный поток GUI, затем блокируется на очереди
        и ждёт пока пользователь сделает выбор в диалоге (максимум 10 минут).
        """
        self.select_videos_signal.emit(videos)
        try:
            return self._video_queue.get(timeout=600)
        except queue.Empty:
            return []  # тайм-аут — продолжаем без файлов

    def run(self) -> None:
        # Устанавливаем перехватчик выбора видео на время работы воркера
        _loader_module._video_selector_override = self._gui_select_videos
        try:
            setup_run_log(self.project.logs_dir)
        except Exception:
            pass

        total   = len(self.steps)
        results = {}
        for step_num, step in enumerate(self.steps, start=1):
            # Проверяем флаг перед каждым шагом — останавливаемся между шагами
            if self._stop_requested:
                _loader_module._video_selector_override = None
                self.stopped.emit()
                return

            self.step_started.emit(step_num, total, STEP_NAMES[step])

            try:
                if step == "load":
                    results["load"] = step_load(self.project)

                elif step == "annotate":
                    _src_map = {"real": ["real"], "airsim": ["airsim"], "all": ["real", "airsim"]}
                    results["annotate"] = run_annotate(
                        self.project,
                        mode=self.annotate_mode,
                        model_path=self.annotate_model,
                        conf=self.annotate_conf,
                        overwrite=self.annotate_overwrite,
                        sources=_src_map.get(self.annotate_sources, ["real", "airsim"]),
                    )

                elif step == "augment":
                    results["augment"] = augment_dataset(
                        self.project,
                        self.aug_types,
                        intensity=self.augment_intensity,
                    )

                elif step == "generate":
                    from modules.generator import generate_images
                    results["generate"] = generate_images(
                        self.project, count=self.gan_count
                    )

                elif step == "generate_sd":
                    from modules.diffusion import generate_backgrounds
                    results["generate_sd"] = generate_backgrounds(
                        self.project,
                        count=self.sd_count,
                        background_type=self.sd_bg_type,
                    )

                elif step == "compose":
                    from modules.compositor import compose as do_compose
                    results["compose"] = do_compose(
                        self.project, count=self.sd_count
                    )

                elif step == "balance":
                    results["balance"] = balance_build(self.project, overwrite=True)

                elif step == "export":
                    results["export"] = do_export(
                        self.project, format=self.export_format
                    )

            except Exception as exc:
                _loader_module._video_selector_override = None
                self.error.emit(f"Ошибка на шаге '{step}': {exc}")
                return

        # Снимаем перехватчик после завершения всех шагов
        _loader_module._video_selector_override = None
        self.finished.emit(results)


# ─────────────────────────────────────────────────────────────
# Рабочий поток обучения GAN
# ─────────────────────────────────────────────────────────────

class GanTrainWorker(QThread):
    """Запускает train_gan() в фоновом потоке и транслирует прогресс в GUI."""

    # (текущая_эпоха, всего_эпох, loss_G, loss_D)
    epoch_done = pyqtSignal(int, int, float, float)
    finished   = pyqtSignal(dict)
    error      = pyqtSignal(str)

    def __init__(self, project: Project, epochs: int, image_size: int = 64, batch_size: int = 16):
        super().__init__()
        self.project          = project
        self.epochs           = epochs
        self.image_size       = image_size
        self.batch_size       = batch_size
        # Флаг досрочной остановки — GUI устанавливает через stop()
        self._stop_requested: bool = False

    def stop(self) -> None:
        """Запрашивает остановку после текущей эпохи. Веса будут сохранены."""
        self._stop_requested = True

    def run(self) -> None:
        from modules.generator import train_gan

        def _on_epoch(epoch: int, total: int, loss_g: float, loss_d: float) -> bool:
            self.epoch_done.emit(epoch, total, loss_g, loss_d)
            # True → train_gan прервёт цикл и сохранит веса
            return self._stop_requested

        try:
            result = train_gan(
                self.project, epochs=self.epochs,
                batch_size=self.batch_size,
                image_size=self.image_size, on_epoch=_on_epoch,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────
# Вспомогательный виджет: строка «метка + поле + Обзор»
# ─────────────────────────────────────────────────────────────

class PathRow(QWidget):
    """Строка для ввода пути к папке с кнопкой выбора."""

    def __init__(self, label: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._lbl = QLabel(label)
        self._lbl.setFixedWidth(150)
        self.edit = QLineEdit()
        self.edit.setMinimumWidth(400)
        self.edit.setPlaceholderText("Путь не задан")
        btn = QPushButton("Обзор…")
        btn.setFixedWidth(80)
        btn.clicked.connect(self._browse)
        layout.addWidget(self._lbl)
        layout.addWidget(self.edit)
        layout.addWidget(btn)

    def set_label(self, text: str) -> None:
        self._lbl.setText(text)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if path:
            self.edit.setText(path)
            self.edit.setToolTip(path)

    @property
    def path(self):
        t = self.edit.text().strip()
        return t if t else None

    @path.setter
    def path(self, val) -> None:
        text = str(val) if val else ""
        self.edit.setText(text)
        self.edit.setToolTip(text if text else "Путь не задан")


# ─────────────────────────────────────────────────────────────
# Диалог: Создать проект
# ─────────────────────────────────────────────────────────────

class CreateProjectDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Создать проект")
        self.setMinimumWidth(520)
        root = QVBoxLayout(self)

        # Имя проекта
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Имя проекта:"))
        self.name_edit = QLineEdit()
        name_row.addWidget(self.name_edit)
        root.addLayout(name_row)

        # Стартовый шаг
        grp_start = QGroupBox("С чего начать работу?")
        start_vbox = QVBoxLayout(grp_start)
        self.r_load     = QRadioButton("Есть видео (mp4) — начать с загрузки кадров")
        self.r_annotate = QRadioButton("Есть кадры — начать с разметки")
        self.r_balance  = QRadioButton("Есть размеченный датасет — сразу балансировка")
        self.r_load.setChecked(True)
        for r in (self.r_load, self.r_annotate, self.r_balance):
            start_vbox.addWidget(r)
        root.addWidget(grp_start)

        # Пути к данным
        grp_paths = QGroupBox("Пути к исходным данным")
        paths_vbox = QVBoxLayout(grp_paths)
        self.path_real   = PathRow()
        self.path_airsim = PathRow()
        paths_vbox.addWidget(self.path_real)
        paths_vbox.addWidget(self.path_airsim)
        root.addWidget(grp_paths)

        # Кнопки
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        for r in (self.r_load, self.r_annotate, self.r_balance):
            r.toggled.connect(self._update_labels)
        self._update_labels()

    def _update_labels(self) -> None:
        if self.r_load.isChecked():
            self.path_real.set_label("Видео real:")
            self.path_airsim.set_label("Видео airsim (необяз.):")
            self.path_airsim.setVisible(True)
        elif self.r_annotate.isChecked():
            self.path_real.set_label("Кадры real:")
            self.path_airsim.set_label("Кадры airsim (необяз.):")
            self.path_airsim.setVisible(True)
        else:
            self.path_real.set_label("Изображения:")
            self.path_airsim.setVisible(False)

    def _on_accept(self) -> None:
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Ошибка", "Введите имя проекта.")
            return
        if not self.path_real.path:
            QMessageBox.warning(self, "Ошибка", "Укажите путь к данным (real).")
            return
        self.accept()

    def get_data(self) -> dict:
        start_from = ("load"     if self.r_load.isChecked() else
                       "annotate" if self.r_annotate.isChecked() else "balance")
        return {
            "name":        self.name_edit.text().strip(),
            "start_from":  start_from,
            "real_path":   self.path_real.path,
            "airsim_path": self.path_airsim.path,
        }


# ─────────────────────────────────────────────────────────────
# Диалог: Подтверждение одного видеофайла
# ─────────────────────────────────────────────────────────────

class SingleVideoConfirmDialog(QDialog):
    """Показывается когда найден ровно один mp4 — запрашивает подтверждение."""

    def __init__(self, video_path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Найден видеофайл")
        self.setMinimumWidth(380)
        root = QVBoxLayout(self)

        size_mb = round(video_path.stat().st_size / (1024 * 1024))
        lbl = QLabel(
            f"Найден файл:\n\n"
            f"<b>{video_path.name}</b>\n\n"
            f"Размер: {size_mb} МБ\n\n"
            f"Начать обработку?"
        )
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setWordWrap(True)
        root.addWidget(lbl)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Yes |
            QDialogButtonBox.StandardButton.No
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)


# ─────────────────────────────────────────────────────────────
# Диалог: Выбор видеофайлов из списка
# ─────────────────────────────────────────────────────────────

class SelectVideosDialog(QDialog):
    """Показывается когда найдено несколько mp4 — пользователь выбирает чекбоксами."""

    def __init__(self, videos: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор видеофайлов")
        self.setMinimumWidth(480)
        self._videos   = videos
        self._selected = []

        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            f"Найдено {len(videos)} видеофайлов. Выберите файлы для обработки:"
        ))

        # Чекбокс на каждый файл
        self._checkboxes: list[QCheckBox] = []
        for v in videos:
            size_mb = round(v.stat().st_size / (1024 * 1024))
            cb = QCheckBox(f"{v.name}  ({size_mb} МБ)")
            cb.setChecked(True)
            self._checkboxes.append(cb)
            root.addWidget(cb)

        # Кнопка «Выбрать все»
        btn_all = QPushButton("Выбрать все")
        btn_all.clicked.connect(self._select_all)
        root.addWidget(btn_all)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _select_all(self) -> None:
        for cb in self._checkboxes:
            cb.setChecked(True)

    def _on_accept(self) -> None:
        self._selected = [
            v for v, cb in zip(self._videos, self._checkboxes) if cb.isChecked()
        ]
        if not self._selected:
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы один файл.")
            return
        self.accept()

    def get_selected(self) -> list:
        return self._selected


# ─────────────────────────────────────────────────────────────
# Диалог: Очистить данные проекта
# ─────────────────────────────────────────────────────────────

class CleanDataDialog(QDialog):

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Очистить данные — {project.name}")
        self.setMinimumWidth(440)
        root = QVBoxLayout(self)

        dir_map = {
            "frames":      project.frames_real_dir.parent,
            "annotations": project.annotations_dir,
            "dataset":     project.dataset_images_dir.parent,
            "export":      project.export_dir,
        }
        sz_f = _dir_size(dir_map["frames"])
        sz_a = _dir_size(dir_map["annotations"])
        sz_d = _dir_size(dir_map["dataset"])
        sz_e = _dir_size(dir_map["export"])

        grp = QGroupBox("Что удалить?")
        vbox = QVBoxLayout(grp)
        self.r_frames    = QRadioButton(
            f"Только кадры (frames/)  [{_fmt_bytes(sz_f)}]")
        self.r_processed = QRadioButton(
            f"Кадры + аннотации  [{_fmt_bytes(sz_f + sz_a)}]")
        self.r_all       = QRadioButton(
            f"Все обработанные данные  [{_fmt_bytes(sz_f + sz_a + sz_d + sz_e)}]")
        self.r_frames.setChecked(True)
        for r in (self.r_frames, self.r_processed, self.r_all):
            vbox.addWidget(r)
        root.addWidget(grp)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def get_level(self) -> str:
        if self.r_frames.isChecked():    return "frames"
        if self.r_processed.isChecked(): return "processed"
        return "all_data"


# ─────────────────────────────────────────────────────────────
# Вкладка 1: Проекты
# ─────────────────────────────────────────────────────────────

class ProjectsTab(QWidget):

    project_selected = pyqtSignal(object)   # Project или None

    def __init__(self, parent=None):
        super().__init__(parent)
        # Ссылка на PipelineTab устанавливается из MainWindow после создания обоих виджетов
        self._pipeline_tab = None
        root = QVBoxLayout(self)

        # Таблица
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Имя", "Текущий шаг", "Кадров", "Создан"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._on_selection)
        root.addWidget(self.table)

        # Кнопки
        btn_row = QHBoxLayout()
        self.btn_create = QPushButton("Создать проект")
        self.btn_delete = QPushButton("Удалить проект")
        self.btn_clean  = QPushButton("Очистить данные")
        self.btn_delete.setEnabled(False)
        self.btn_clean.setEnabled(False)
        for b in (self.btn_create, self.btn_delete, self.btn_clean):
            btn_row.addWidget(b)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self.btn_create.clicked.connect(self._on_create)
        self.btn_delete.clicked.connect(self._on_delete)
        self.btn_clean.clicked.connect(self._on_clean)

        self.refresh()

    # ── публичный метод ──────────────────────────────────────

    def refresh(self) -> None:
        """Перечитывает список проектов с диска и обновляет таблицу."""
        current_name = None
        row_sel = self.table.currentRow()
        if row_sel >= 0 and self.table.item(row_sel, 0):
            current_name = self.table.item(row_sel, 0).text()

        self.table.setRowCount(0)
        for meta in Project.list_all():
            row = self.table.rowCount()
            self.table.insertRow(row)
            stats    = meta.get("stats", {})
            load_inf = stats.get("load", {})
            frames   = (stats.get("dataset_frames") or
                        (load_inf.get("frames") if isinstance(load_inf, dict) else None) or
                        "—")
            self.table.setItem(row, 0, QTableWidgetItem(meta.get("name", "?")))
            self.table.setItem(row, 1, QTableWidgetItem(meta.get("current_step") or "—"))
            self.table.setItem(row, 2, QTableWidgetItem(str(frames)))
            self.table.setItem(row, 3, QTableWidgetItem(meta.get("created", "")[:10]))

        # Восстанавливаем выделение
        if current_name:
            for r in range(self.table.rowCount()):
                if self.table.item(r, 0).text() == current_name:
                    self.table.selectRow(r)
                    break

    # ── обработчики ─────────────────────────────────────────

    def _on_selection(self) -> None:
        has = self.table.currentRow() >= 0 and bool(self.table.selectedItems())
        self.btn_delete.setEnabled(has)
        self.btn_clean.setEnabled(has)
        if has:
            name = self.table.item(self.table.currentRow(), 0).text()
            try:
                self.project_selected.emit(Project.load(name))
            except Exception:
                pass
        else:
            self.project_selected.emit(None)

    def _on_create(self) -> None:
        dlg = CreateProjectDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        data = dlg.get_data()
        try:
            project = Project.create(data["name"])
        except ValueError as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

        # Сохраняем start_from в метаданные
        meta = project._read_meta()
        meta["start_from"] = data["start_from"]
        project._write_meta(meta)

        # Устанавливаем источники данных по сценарию
        cat_map = {"load": "videos", "annotate": "frames", "balance": "frames"}
        cat = cat_map[data["start_from"]]
        if data["real_path"]:
            project.set_source(cat, "real",   Path(data["real_path"]))
        if data["airsim_path"]:
            project.set_source(cat, "airsim", Path(data["airsim_path"]))

        self.refresh()
        # Выбираем созданный проект
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).text() == data["name"]:
                self.table.selectRow(r)
                break

    def _on_delete(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        name = self.table.item(row, 0).text()
        ans = QMessageBox.question(
            self, "Удалить проект",
            f"Удалить проект «{name}»?\nВсе файлы будут удалены безвозвратно.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        # Проверяем активные воркеры — удаление во время работы сломает файловую структуру
        if self._pipeline_tab is not None:
            if self._pipeline_tab.worker and self._pipeline_tab.worker.isRunning():
                QMessageBox.warning(self, "Ошибка",
                    "Остановите пайплайн перед удалением проекта.")
                return
            if self._pipeline_tab.gan_worker and self._pipeline_tab.gan_worker.isRunning():
                QMessageBox.warning(self, "Ошибка",
                    "Остановите обучение GAN перед удалением проекта.")
                return
        # Закрываем файловые обработчики логгера — иначе Windows блокирует папку
        _close_project_log_handlers()
        try:
            shutil.rmtree(Project.load(name).dir)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))
            return
        self.refresh()
        self.project_selected.emit(None)

    def _on_clean(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        name    = self.table.item(row, 0).text()
        project = Project.load(name)
        dlg     = CleanDataDialog(project, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        level = dlg.get_level()

        dir_map = {
            "frames":      project.frames_real_dir.parent,
            "annotations": project.annotations_dir,
            "dataset":     project.dataset_images_dir.parent,
            "export":      project.export_dir,
        }
        for d in _CLEAN_DIRS[level]:
            _clear_dir(dir_map[d])

        meta = project._read_meta()
        keys = _CLEAN_STATS_KEYS[level]
        if keys is None:
            meta["stats"] = {}
        else:
            for k in keys:
                meta["stats"].pop(k, None)
        meta["current_step"] = None
        project._write_meta(meta)

        for cat, key in _CLEAN_SOURCES[level]:
            project.set_source(cat, key, None)

        QMessageBox.information(self, "Готово", f"Данные проекта «{name}» очищены.")
        self.refresh()
        self.project_selected.emit(Project.load(name))


# ─────────────────────────────────────────────────────────────
# Вкладка 2: Пайплайн
# ─────────────────────────────────────────────────────────────

class PipelineTab(QWidget):

    pipeline_done = pyqtSignal()   # сигнал для обновления таблицы проектов

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.settings        = settings
        self.current_project = None
        self.worker          = None
        self.gan_worker      = None

        # Активный набор шагов пайплайна — меняется при смене метода расширения
        self._active_steps = list(_PIPELINE_STEPS_AUGMENT)

        root = QVBoxLayout(self)

        # Заголовок: активный проект и его статистика
        self.lbl_project = QLabel("Проект не выбран")
        self.lbl_project.setFont(QFont("", 11, QFont.Weight.Bold))
        root.addWidget(self.lbl_project)

        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet("color: grey;")
        root.addWidget(self.lbl_stats)

        # Формат экспорта
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Формат экспорта:"))
        self.combo_fmt = QComboBox()
        self.combo_fmt.addItems(["yolo", "coco"])
        self.combo_fmt.setCurrentText(settings.get("export_format", "yolo"))
        self.combo_fmt.setFixedWidth(100)
        fmt_row.addWidget(self.combo_fmt)
        fmt_row.addStretch()
        root.addLayout(fmt_row)

        # Отдельные шаги — кнопки для всех вариантов пайплайна
        grp_steps = QGroupBox("Отдельные шаги")
        steps_row = QHBoxLayout(grp_steps)
        self.step_btns: dict[str, QPushButton] = {}
        _all_step_keys = [
            "load", "annotate", "augment",
            "generate", "generate_sd", "compose",
            "balance", "export",
        ]
        for step in _all_step_keys:
            btn = QPushButton(STEP_NAMES[step])
            btn.setEnabled(False)
            btn.clicked.connect(lambda _checked, s=step: self._run_steps([s]))
            self.step_btns[step] = btn
            steps_row.addWidget(btn)
        # По умолчанию скрываем GAN- и SD-специфичные кнопки
        self.step_btns["generate"].setVisible(False)
        self.step_btns["generate_sd"].setVisible(False)
        self.step_btns["compose"].setVisible(False)
        root.addWidget(grp_steps)

        # Запуск всего пайплайна
        grp_all  = QGroupBox("Запустить весь пайплайн")
        all_vbox = QVBoxLayout(grp_all)

        from_row = QHBoxLayout()
        from_row.addWidget(QLabel("Начиная с шага:"))
        self.combo_from = QComboBox()
        self.combo_from.setFixedWidth(200)
        self._update_combo_from()   # заполняем по _active_steps
        from_row.addWidget(self.combo_from)
        from_row.addStretch()
        all_vbox.addLayout(from_row)

        self.btn_run_all = QPushButton("▶   Запустить всё")
        self.btn_run_all.setEnabled(False)
        self.btn_run_all.setMinimumHeight(42)
        f = self.btn_run_all.font()
        f.setPointSize(f.pointSize() + 1)
        self.btn_run_all.setFont(f)
        self.btn_run_all.clicked.connect(self._run_all)
        all_vbox.addWidget(self.btn_run_all)
        root.addWidget(grp_all)

        # ── Блок обучения GAN (показывается только в GAN-режиме) ────────
        self.grp_gan = QGroupBox("Обучение GAN")
        self.grp_gan.setVisible(False)
        gan_vbox = QVBoxLayout(self.grp_gan)

        gan_params_row = QHBoxLayout()
        gan_params_row.addWidget(QLabel("Эпох:"))
        self.spin_gan_epochs = QSpinBox()
        self.spin_gan_epochs.setRange(1, 1000)
        self.spin_gan_epochs.setValue(100)
        self.spin_gan_epochs.setFixedWidth(80)
        self.spin_gan_epochs.setToolTip("Количество эпох обучения GAN")
        gan_params_row.addWidget(self.spin_gan_epochs)
        gan_params_row.addStretch()
        gan_vbox.addLayout(gan_params_row)

        self.btn_train_gan = QPushButton("▶  Обучить GAN")
        self.btn_train_gan.setEnabled(False)
        self.btn_train_gan.setMinimumHeight(36)
        self.btn_train_gan.clicked.connect(self._start_gan_training)
        gan_vbox.addWidget(self.btn_train_gan)

        # Прогресс обучения GAN (по эпохам) + кнопка «Стоп» в одной строке
        gan_progress_row = QHBoxLayout()
        self.gan_progress = QProgressBar()
        self.gan_progress.setRange(0, 100)
        self.gan_progress.setVisible(False)
        self.btn_stop_gan = QPushButton("■  Стоп")
        self.btn_stop_gan.setFixedWidth(90)
        self.btn_stop_gan.setVisible(False)
        self.btn_stop_gan.clicked.connect(self._on_stop_gan_requested)
        gan_progress_row.addWidget(self.gan_progress)
        gan_progress_row.addWidget(self.btn_stop_gan)
        gan_vbox.addLayout(gan_progress_row)

        self.lbl_gan_status = QLabel("")
        self.lbl_gan_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gan_vbox.addWidget(self.lbl_gan_status)

        root.addWidget(self.grp_gan)

        # ── Блок генерации SD (показывается только в SD-режиме) ─────────────
        self.grp_sd = QGroupBox("Генерация SD фонов")
        self.grp_sd.setVisible(False)
        sd_vbox = QVBoxLayout(self.grp_sd)

        sd_params_row = QHBoxLayout()
        sd_params_row.addWidget(QLabel("Фонов:"))
        self.spin_sd_count = QSpinBox()
        self.spin_sd_count.setRange(1, 2000)
        self.spin_sd_count.setValue(settings.get("sd_count", 200))
        self.spin_sd_count.setFixedWidth(90)
        self.spin_sd_count.setToolTip(
            "Количество фоновых сцен для генерации через Stable Diffusion"
        )
        sd_params_row.addWidget(self.spin_sd_count)
        sd_params_row.addStretch()
        sd_vbox.addLayout(sd_params_row)

        # Тип фонов: открытые пространства / лес / оба
        bg_type_row = QHBoxLayout()
        bg_type_row.addWidget(QLabel("Тип фонов:"))
        self.r_sd_open   = QRadioButton("Открытые")
        self.r_sd_forest = QRadioButton("Лес")
        self.r_sd_all    = QRadioButton("Все")
        _saved_bg_type = settings.get("sd_bg_type", "all")
        if _saved_bg_type == "open":
            self.r_sd_open.setChecked(True)
        elif _saved_bg_type == "forest":
            self.r_sd_forest.setChecked(True)
        else:
            self.r_sd_all.setChecked(True)
        for _r in (self.r_sd_open, self.r_sd_forest, self.r_sd_all):
            bg_type_row.addWidget(_r)
        bg_type_row.addStretch()
        sd_vbox.addLayout(bg_type_row)

        self.btn_generate_sd = QPushButton("▶  Сгенерировать фоны")
        self.btn_generate_sd.setEnabled(False)
        self.btn_generate_sd.setMinimumHeight(36)
        self.btn_generate_sd.clicked.connect(self._start_sd_generation)
        sd_vbox.addWidget(self.btn_generate_sd)

        root.addWidget(self.grp_sd)

        # Прогресс-бар пайплайна (indeterminate) + кнопка Стоп
        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.btn_stop = QPushButton("■  Стоп")
        self.btn_stop.setFixedWidth(90)
        self.btn_stop.setVisible(False)
        self.btn_stop.clicked.connect(self._on_stop_requested)
        progress_row.addWidget(self.progress)
        progress_row.addWidget(self.btn_stop)
        root.addLayout(progress_row)

        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.lbl_status)

        root.addStretch()

    # ── публичный метод ──────────────────────────────────────

    def set_project(self, project) -> None:
        self.current_project = project
        has = project is not None
        for btn in self.step_btns.values():
            btn.setEnabled(has)
        self.btn_run_all.setEnabled(has)
        self.btn_train_gan.setEnabled(has)
        self.btn_generate_sd.setEnabled(has)

        if project:
            meta   = project._read_meta()
            step   = meta.get("current_step") or "—"
            stats  = meta.get("stats", {})
            frames = stats.get("dataset_frames", "—")
            pos    = stats.get("positive",       "—")
            neg    = stats.get("negative",       "—")
            self.lbl_project.setText(
                f"Проект: {project.name}  |  Текущий шаг: {step}")
            self.lbl_stats.setText(
                f"Кадров: {frames}   Позитивных: {pos}   Негативных: {neg}")
            # start_from из метаданных → устанавливаем combo_from
            start = meta.get("start_from", "load")
            if start in PIPELINE_STEPS:
                self.combo_from.setCurrentIndex(PIPELINE_STEPS.index(start))
        else:
            self.lbl_project.setText("Проект не выбран")
            self.lbl_stats.setText("")

    # ── запуск ──────────────────────────────────────────────

    def _worker_kwargs(self) -> dict:
        return {
            "augment_intensity":  self.settings.get("augment_intensity", 0.5),
            "export_format":      self.combo_fmt.currentText(),
            "annotate_model":     self.settings.get("annotate_model", "yolov8n.pt"),
            "annotate_conf":      self.settings.get("annotate_conf", 0.25),
            "annotate_overwrite": self.settings.get("annotate_overwrite", False),
            "annotate_mode":      self.settings.get("annotate_mode", "auto"),
            "annotate_sources":   self.settings.get("annotate_sources", "all"),
            "aug_types":          self.settings.get(
                "aug_types", ["fog", "rain", "noise", "blur", "brightness"]),
            "gan_count":  self.settings.get("gan_count", 200),
            "sd_count":   self.spin_sd_count.value(),
            "sd_bg_type": (
                "open"   if self.r_sd_open.isChecked()   else
                "forest" if self.r_sd_forest.isChecked() else
                "all"
            ),
        }

    def _run_steps(self, steps: list) -> None:
        if not self.current_project:
            return
        self._set_running(True, "Запускаем...")
        self.worker = PipelineWorker(
            self.current_project, steps, **self._worker_kwargs()
        )
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.stopped.connect(self._on_stopped)
        self.worker.step_started.connect(self._on_step_started)
        # Сигнал выбора видео: воркер ждёт пока GUI покажет диалог и вернёт результат
        self.worker.select_videos_signal.connect(self._on_select_videos)
        self.worker.start()

    def _run_all(self) -> None:
        if not self.current_project:
            return
        start = self.combo_from.currentData()
        # Ищем стартовый шаг в активном пайплайне
        try:
            idx = self._active_steps.index(start)
        except ValueError:
            idx = 0
        self._run_steps(self._active_steps[idx:])

    # ── вспомогательные методы ──────────────────────────────

    def _update_combo_from(self) -> None:
        """Перестраивает комбобокс 'Начиная с шага' по текущему _active_steps."""
        current = self.combo_from.currentData() if self.combo_from.count() else None
        self.combo_from.clear()
        for s in self._active_steps:
            self.combo_from.addItem(STEP_NAMES[s], s)
        # Восстанавливаем предыдущий выбор если шаг есть в новом списке
        for i in range(self.combo_from.count()):
            if self.combo_from.itemData(i) == current:
                self.combo_from.setCurrentIndex(i)
                break

    def _start_sd_generation(self) -> None:
        """Запускает генерацию SD фонов как отдельный шаг пайплайна."""
        if not self.current_project:
            return
        self._run_steps(["generate_sd"])

    def _on_expansion_method_changed(self, method: str) -> None:
        """Переключает пайплайн и UI между режимами 'augment', 'gan' и 'sd'."""
        is_gan = (method == "gan")
        is_sd  = (method == "sd")
        if is_sd:
            self._active_steps = list(_PIPELINE_STEPS_SD)
        elif is_gan:
            self._active_steps = list(_PIPELINE_STEPS_GAN)
        else:
            self._active_steps = list(_PIPELINE_STEPS_AUGMENT)
        self._update_combo_from()
        # Кнопки шагов — показываем только соответствующий шаг расширения
        self.step_btns["augment"].setVisible(not is_gan and not is_sd)
        self.step_btns["generate"].setVisible(is_gan)
        self.step_btns["generate_sd"].setVisible(is_sd)
        self.step_btns["compose"].setVisible(is_sd)
        # Блоки обучения GAN и генерации SD — взаимоисключающие
        self.grp_gan.setVisible(is_gan)
        self.grp_sd.setVisible(is_sd)

    # ── обучение GAN ────────────────────────────────────────

    def _start_gan_training(self) -> None:
        """Запускает обучение GAN в отдельном потоке (независимо от пайплайна)."""
        if not self.current_project:
            return
        self._set_gan_running(True)
        self.gan_worker = GanTrainWorker(
            self.current_project,
            self.spin_gan_epochs.value(),
            image_size=self.settings.get("gan_image_size", config.GAN_IMAGE_SIZE),
            batch_size=self.settings.get("gan_batch_size", config.GAN_BATCH_SIZE),
        )
        self.gan_worker.epoch_done.connect(self._on_gan_epoch_done)
        self.gan_worker.finished.connect(self._on_gan_finished)
        self.gan_worker.error.connect(self._on_gan_error)
        self.gan_worker.start()

    def _on_gan_epoch_done(self, epoch: int, total: int, loss_g: float, loss_d: float) -> None:
        """Обновляет прогресс GAN-обучения после каждой эпохи."""
        self.gan_progress.setMaximum(total)
        self.gan_progress.setValue(epoch)
        self.lbl_gan_status.setText(
            f"Эпоха {epoch}/{total} | loss_G={loss_g:.4f} | loss_D={loss_d:.4f}"
        )

    def _on_gan_finished(self, result: dict) -> None:
        """Вызывается по завершении обучения GAN."""
        self._set_gan_running(False)
        epochs = result.get("epochs", 0)
        loss_g = result.get("final_loss_g", 0.0)
        loss_d = result.get("final_loss_d", 0.0)
        self.lbl_gan_status.setText(
            f"Обучено {epochs} эпох | loss_G={loss_g:.4f} | loss_D={loss_d:.4f}"
        )

    def _on_gan_error(self, msg: str) -> None:
        """Обработчик ошибки обучения GAN."""
        self._set_gan_running(False)
        self.lbl_gan_status.setText("Ошибка обучения.")
        QMessageBox.critical(self, "Ошибка обучения GAN", msg)

    def _on_stop_gan_requested(self) -> None:
        """Запрашивает досрочную остановку GAN — веса сохранятся после эпохи."""
        if self.gan_worker:
            self.gan_worker.stop()
            # Блокируем кнопку чтобы не нажимали повторно
            self.btn_stop_gan.setEnabled(False)
            self.lbl_gan_status.setText("Останавливаем после текущей эпохи...")

    def _set_gan_running(self, running: bool) -> None:
        """Управляет состоянием UI во время обучения GAN."""
        has_proj = self.current_project is not None
        self.btn_train_gan.setEnabled(not running and has_proj)
        # Блокируем пайплайн — нельзя запустить оба процесса одновременно
        self.btn_run_all.setEnabled(not running and has_proj)
        self.btn_generate_sd.setEnabled(not running and has_proj)
        for btn in self.step_btns.values():
            btn.setEnabled(not running and has_proj)
        self.gan_progress.setVisible(running)
        # Кнопка Стоп: показываем при запуске, скрываем по завершении
        self.btn_stop_gan.setVisible(running)
        self.btn_stop_gan.setEnabled(running)   # сброс enabled после disable при нажатии
        if running:
            self.gan_progress.setMaximum(self.spin_gan_epochs.value())
            self.gan_progress.setValue(0)
            self.lbl_gan_status.setText("Обучение GAN...")

    # ── управление состоянием UI ────────────────────────────

    def _set_running(self, running: bool, status: str = "") -> None:
        self.progress.setVisible(running)
        self.btn_stop.setVisible(running)
        self.btn_stop.setEnabled(running)   # сброс после disable при нажатии
        self.lbl_status.setText(status)
        has_proj = self.current_project is not None
        for btn in self.step_btns.values():
            btn.setEnabled(not running and has_proj)
        self.btn_run_all.setEnabled(not running and has_proj)
        self.btn_generate_sd.setEnabled(not running and has_proj)

    def _on_step_started(self, step_num: int, total: int, name: str) -> None:
        """Обновляет строку статуса при старте каждого шага."""
        self.lbl_status.setText(f"Шаг {step_num}/{total}: {name}...")

    def _on_stop_requested(self) -> None:
        """Показывает диалог подтверждения и при согласии ставит флаг в воркере."""
        ans = QMessageBox.question(
            self, "Остановить пайплайн",
            "Остановить пайплайн?\nТекущий шаг будет прерван.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans == QMessageBox.StandardButton.Yes and self.worker:
            self.worker.request_stop()
            self.btn_stop.setEnabled(False)   # не даём нажать повторно
            self.lbl_status.setText("Останавливаем после текущего шага...")

    def _on_stopped(self) -> None:
        """Вызывается когда воркер остановился по флагу между шагами."""
        if self.current_project:
            try:
                self.current_project = Project.load(self.current_project.name)
            except Exception:
                pass
        self._set_running(False, "Остановлено.")
        self.set_project(self.current_project)
        self.pipeline_done.emit()

    def _on_select_videos(self, videos: list) -> None:
        """Вызывается в главном потоке когда воркер ждёт выбора видеофайлов.

        Показывает нужный диалог, затем кладёт результат в очередь воркера —
        тот разблокируется и продолжает загрузку с выбранными файлами.
        """
        if not self.worker:
            return

        if len(videos) == 1:
            dlg    = SingleVideoConfirmDialog(videos[0], self)
            result = videos if dlg.exec() == QDialog.DialogCode.Accepted else []
        else:
            dlg    = SelectVideosDialog(videos, self)
            result = dlg.get_selected() if dlg.exec() == QDialog.DialogCode.Accepted else []

        # Возвращаем выбор в рабочий поток через очередь
        self.worker._video_queue.put(result)

    def _on_finished(self, results: dict) -> None:
        # Перечитываем проект с диска — метаданные обновились в воркере
        if self.current_project:
            try:
                self.current_project = Project.load(self.current_project.name)
            except Exception:
                pass
        n = len(results)
        self._set_running(False, f"Готово. Выполнено шагов: {n}.")
        self.set_project(self.current_project)
        self.pipeline_done.emit()

    def _on_error(self, msg: str) -> None:
        self._set_running(False, "")
        self.pipeline_done.emit()
        QMessageBox.critical(self, "Ошибка пайплайна", msg)


# ─────────────────────────────────────────────────────────────
# Вкладка 3: Настройки
# ─────────────────────────────────────────────────────────────

class SettingsTab(QWidget):

    # Сигнал: метод расширения изменился ("augment" или "gan")
    expansion_method_changed = pyqtSignal(str)

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.settings        = settings
        self.current_project = None

        # Всё содержимое — в прокручиваемой области
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        root  = QVBoxLayout(inner)
        root.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # ── Источники данных ─────────────────────────────────
        self.grp_sources = QGroupBox("Источники данных активного проекта")
        src_vbox         = QVBoxLayout(self.grp_sources)

        self.source_rows: dict[tuple, PathRow] = {}
        source_labels = [
            (("videos",      "real"),   "videos.real:"),
            (("videos",      "airsim"), "videos.airsim:"),
            (("frames",      "real"),   "frames.real:"),
            (("frames",      "airsim"), "frames.airsim:"),
            (("annotations", "real"),   "annotations.real:"),
            (("annotations", "airsim"), "annotations.airsim:"),
            (("dataset",     "images"), "dataset.images:"),
            (("dataset",     "labels"), "dataset.labels:"),
        ]
        for (cat, key), label in source_labels:
            row = PathRow(label)
            self.source_rows[(cat, key)] = row
            src_vbox.addWidget(row)

        btn_save_src = QPushButton("Сохранить пути")
        btn_save_src.clicked.connect(self._save_sources)
        src_vbox.addWidget(btn_save_src)
        root.addWidget(self.grp_sources)

        # ── Метод расширения датасета ────────────────────────
        grp_method  = QGroupBox("Метод расширения датасета")
        method_vbox = QVBoxLayout(grp_method)
        self.r_expand_augment = QRadioButton("Аугментация")
        self.r_expand_gan     = QRadioButton("GAN генерация")
        self.r_expand_sd      = QRadioButton("Stable Diffusion")
        _saved_method = settings.get("expansion_method", "augment")
        if _saved_method == "gan":
            self.r_expand_gan.setChecked(True)
        elif _saved_method == "sd":
            self.r_expand_sd.setChecked(True)
        else:
            self.r_expand_augment.setChecked(True)
        method_vbox.addWidget(self.r_expand_augment)
        method_vbox.addWidget(self.r_expand_gan)
        method_vbox.addWidget(self.r_expand_sd)

        # Уведомляем PipelineTab немедленно при переключении (без сохранения)
        self.r_expand_augment.toggled.connect(
            lambda checked: self.expansion_method_changed.emit("augment") if checked else None
        )
        self.r_expand_gan.toggled.connect(
            lambda checked: self.expansion_method_changed.emit("gan") if checked else None
        )
        self.r_expand_sd.toggled.connect(
            lambda checked: self.expansion_method_changed.emit("sd") if checked else None
        )
        root.addWidget(grp_method)

        # ── Параметры пайплайна ──────────────────────────────
        grp_params  = QGroupBox("Параметры пайплайна")
        form_params = QFormLayout(grp_params)

        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(1, 60)
        self.spin_sample.setValue(settings.get("frame_sample_rate", config.FRAME_SAMPLE_RATE))
        self.spin_sample.setToolTip("Сохранять каждый N-й кадр из видео")
        form_params.addRow("Шаг выборки кадров (N):", self.spin_sample)

        self.spin_ratio = QSpinBox()
        self.spin_ratio.setRange(1, 20)
        self.spin_ratio.setValue(settings.get("pos_neg_ratio", config.POS_NEG_RATIO))
        self.spin_ratio.setToolTip("Максимальное число негативных примеров на один позитивный")
        form_params.addRow("Соотношение neg:pos:", self.spin_ratio)

        root.addWidget(grp_params)

        # ── Параметры разметки ───────────────────────────────
        grp_ann  = QGroupBox("Параметры разметки")
        form_ann = QFormLayout(grp_ann)

        # Режим разметки
        mode_widget = QWidget()
        mode_row    = QHBoxLayout(mode_widget)
        mode_row.setContentsMargins(0, 0, 0, 0)
        self.r_mode_auto   = QRadioButton("auto")
        self.r_mode_manual = QRadioButton("manual")
        if settings.get("annotate_mode", "auto") == "manual":
            self.r_mode_manual.setChecked(True)
        else:
            self.r_mode_auto.setChecked(True)
        mode_row.addWidget(self.r_mode_auto)
        mode_row.addWidget(self.r_mode_manual)
        mode_row.addStretch()
        form_ann.addRow("Режим разметки:", mode_widget)

        # Источник для разметки
        src_widget = QWidget()
        src_row    = QHBoxLayout(src_widget)
        src_row.setContentsMargins(0, 0, 0, 0)
        self.r_src_real   = QRadioButton("real")
        self.r_src_airsim = QRadioButton("airsim")
        self.r_src_all    = QRadioButton("все")
        _src_val = settings.get("annotate_sources", "all")
        if _src_val == "real":
            self.r_src_real.setChecked(True)
        elif _src_val == "airsim":
            self.r_src_airsim.setChecked(True)
        else:
            self.r_src_all.setChecked(True)
        src_row.addWidget(self.r_src_real)
        src_row.addWidget(self.r_src_airsim)
        src_row.addWidget(self.r_src_all)
        src_row.addStretch()
        form_ann.addRow("Источник для разметки:", src_widget)

        # Модель и порог
        self.edit_model = QLineEdit(settings.get("annotate_model", "yolov8n.pt"))
        self.edit_model.setPlaceholderText("yolov8n.pt")
        form_ann.addRow("YOLOv8 модель:", self.edit_model)

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setValue(settings.get("annotate_conf", 0.25))
        form_ann.addRow("YOLOv8 порог conf:", self.spin_conf)

        self.chk_overwrite = QCheckBox("Перезаписывать существующие аннотации")
        self.chk_overwrite.setChecked(settings.get("annotate_overwrite", False))
        form_ann.addRow("", self.chk_overwrite)

        root.addWidget(grp_ann)

        # ── Типы аугментации ──────────────────────────────────
        grp_aug  = QGroupBox("Типы аугментации")
        form_aug = QFormLayout(grp_aug)

        # Слайдер интенсивности
        self.slider_intensity = QSlider(Qt.Orientation.Horizontal)
        self.slider_intensity.setRange(0, 100)
        self.slider_intensity.setValue(
            int(settings.get("augment_intensity", 0.5) * 100))
        self.lbl_intensity = QLabel(
            f"{settings.get('augment_intensity', 0.5):.2f}")
        self.slider_intensity.valueChanged.connect(
            lambda v: self.lbl_intensity.setText(f"{v / 100:.2f}"))
        int_widget = QWidget()
        int_row    = QHBoxLayout(int_widget)
        int_row.setContentsMargins(0, 0, 0, 0)
        int_row.addWidget(self.slider_intensity)
        int_row.addWidget(self.lbl_intensity)
        form_aug.addRow("Интенсивность:", int_widget)

        # Чекбоксы типов
        _saved_types = set(settings.get("aug_types", ["fog", "rain", "noise", "blur", "brightness"]))
        self.chk_aug: dict[str, QCheckBox] = {}
        aug_labels = [
            ("fog",        "Туман (fog)"),
            ("rain",       "Дождь (rain)"),
            ("noise",      "Шум (noise)"),
            ("blur",       "Размытие (blur)"),
            ("brightness", "Яркость (brightness)"),
        ]
        for key, label in aug_labels:
            cb = QCheckBox(label)
            cb.setChecked(key in _saved_types)
            self.chk_aug[key] = cb
            form_aug.addRow("", cb)

        root.addWidget(grp_aug)

        # ── Параметры GAN ────────────────────────────────────
        grp_gan      = QGroupBox("Параметры GAN")
        gan_form     = QFormLayout(grp_gan)

        # Радио-кнопки выбора разрешения изображения
        size_widget = QWidget()
        size_row    = QHBoxLayout(size_widget)
        size_row.setContentsMargins(0, 0, 0, 0)
        self.r_gan_size_64  = QRadioButton("64×64")
        self.r_gan_size_128 = QRadioButton("128×128")
        self.r_gan_size_256 = QRadioButton("256×256")
        _saved_size = settings.get("gan_image_size", config.GAN_IMAGE_SIZE)
        if _saved_size == 128:
            self.r_gan_size_128.setChecked(True)
        elif _saved_size == 256:
            self.r_gan_size_256.setChecked(True)
        else:
            self.r_gan_size_64.setChecked(True)
        for r in (self.r_gan_size_64, self.r_gan_size_128, self.r_gan_size_256):
            size_row.addWidget(r)
        size_row.addStretch()
        gan_form.addRow("Разрешение изображения:", size_widget)

        # Размер батча
        self.spin_gan_batch = QSpinBox()
        self.spin_gan_batch.setRange(8, 128)
        self.spin_gan_batch.setSingleStep(8)
        self.spin_gan_batch.setValue(settings.get("gan_batch_size", config.GAN_BATCH_SIZE))
        self.spin_gan_batch.setFixedWidth(80)
        self.spin_gan_batch.setToolTip(
            "Больше = быстрее, но требует больше VRAM.\n"
            "RTX 3060 6 ГБ: максимум 16–32. RTX A6000 48 ГБ: до 128"
        )
        gan_form.addRow("Размер батча:", self.spin_gan_batch)

        root.addWidget(grp_gan)

        # Кнопка «Сохранить всё»
        btn_save = QPushButton("Сохранить настройки")
        btn_save.setMinimumHeight(36)
        btn_save.clicked.connect(self._save_all)
        root.addWidget(btn_save)
        root.addStretch()

    # ── публичный метод ──────────────────────────────────────

    def set_project(self, project) -> None:
        self.current_project = project
        for (cat, key), row in self.source_rows.items():
            row.path = project.get_source(cat, key) if project else None

    # ── сохранение ──────────────────────────────────────────

    def _save_sources(self) -> None:
        if not self.current_project:
            QMessageBox.warning(self, "Нет проекта", "Сначала выберите проект.")
            return
        for (cat, key), row in self.source_rows.items():
            p = row.path
            if p:
                path_obj = Path(p)
                if not path_obj.exists():
                    QMessageBox.warning(
                        self, "Ошибка", f"Папка не найдена:\n{path_obj}")
                    return
                self.current_project.set_source(cat, key, path_obj)
            else:
                self.current_project.set_source(cat, key, None)
        QMessageBox.information(self, "Сохранено", "Пути к источникам данных обновлены.")

    def _save_all(self) -> None:
        # Применяем к модулю config — действует до конца сессии
        config.FRAME_SAMPLE_RATE = self.spin_sample.value()
        config.POS_NEG_RATIO     = self.spin_ratio.value()

        if self.r_mode_manual.isChecked():
            ann_mode = "manual"
        else:
            ann_mode = "auto"

        if self.r_src_real.isChecked():
            ann_sources = "real"
        elif self.r_src_airsim.isChecked():
            ann_sources = "airsim"
        else:
            ann_sources = "all"

        if self.r_expand_sd.isChecked():
            expand_method = "sd"
        elif self.r_expand_gan.isChecked():
            expand_method = "gan"
        else:
            expand_method = "augment"

        if self.r_gan_size_256.isChecked():
            gan_image_size = 256
        elif self.r_gan_size_128.isChecked():
            gan_image_size = 128
        else:
            gan_image_size = 64

        self.settings.update({
            "frame_sample_rate":  self.spin_sample.value(),
            "pos_neg_ratio":      self.spin_ratio.value(),
            "augment_intensity":  self.slider_intensity.value() / 100,
            "aug_types":          [k for k, cb in self.chk_aug.items() if cb.isChecked()],
            "annotate_model":     self.edit_model.text().strip() or "yolov8n.pt",
            "annotate_conf":      self.spin_conf.value(),
            "annotate_mode":      ann_mode,
            "annotate_sources":   ann_sources,
            "annotate_overwrite": self.chk_overwrite.isChecked(),
            "expansion_method":   expand_method,
            "gan_image_size":     gan_image_size,
            "gan_batch_size":     self.spin_gan_batch.value(),
            "sd_count":           self.settings.get("sd_count", 200),
        })
        _save_settings(self.settings)
        # Сигнал уже мог уйти при переключении radio — посылаем ещё раз для надёжности
        self.expansion_method_changed.emit(expand_method)
        QMessageBox.information(self, "Сохранено", "Настройки сохранены в gui_settings.json.")


# ─────────────────────────────────────────────────────────────
# Вкладка 4: Логи
# ─────────────────────────────────────────────────────────────

class LogsTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Courier New", 9))
        self.text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        root.addWidget(self.text)

        btn_row   = QHBoxLayout()
        btn_clear = QPushButton("Очистить лог")
        btn_save  = QPushButton("Сохранить в файл…")
        btn_clear.clicked.connect(self.text.clear)
        btn_save.clicked.connect(self._save_log)
        btn_row.addWidget(btn_clear)
        btn_row.addWidget(btn_save)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Подключаем к глобальному эмиттеру — работает из любого потока
        _log_emitter.message.connect(self._append)

    def _append(self, msg: str) -> None:
        self.text.moveCursor(QTextCursor.MoveOperation.End)
        self.text.insertPlainText(msg + "\n")
        self.text.moveCursor(QTextCursor.MoveOperation.End)

    def _save_log(self) -> None:
        from datetime import datetime
        default = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить лог", default, "Text files (*.txt)")
        if path:
            Path(path).write_text(self.text.toPlainText(), encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# Главное окно
# ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("dataset_builder")
        self.setMinimumSize(860, 620)
        self.resize(980, 700)

        self.settings = _load_settings()

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        self.tab_projects = ProjectsTab()
        self.tab_pipeline = PipelineTab(self.settings)
        self.tab_settings = SettingsTab(self.settings)
        self.tab_logs     = LogsTab()
        # Связываем вкладки: ProjectsTab проверяет воркеры PipelineTab перед удалением
        self.tab_projects._pipeline_tab = self.tab_pipeline

        tabs.addTab(self.tab_projects, "  Проекты  ")
        tabs.addTab(self.tab_pipeline, "  Пайплайн  ")
        tabs.addTab(self.tab_settings, "  Настройки  ")
        tabs.addTab(self.tab_logs,     "  Логи  ")

        # Статусбар
        self.status_lbl = QLabel("Проект не выбран")
        self.statusBar().addWidget(self.status_lbl)

        # Выбор проекта → обновляем Пайплайн + Настройки + статусбар
        self.tab_projects.project_selected.connect(self._on_project_changed)

        # Завершение пайплайна → обновляем таблицу проектов
        self.tab_pipeline.pipeline_done.connect(self.tab_projects.refresh)

        # Смена метода расширения в Настройках → обновляем вкладку Пайплайн
        self.tab_settings.expansion_method_changed.connect(
            self.tab_pipeline._on_expansion_method_changed
        )

        # Применяем сохранённый метод расширения при старте
        self.tab_pipeline._on_expansion_method_changed(
            self.settings.get("expansion_method", "augment")
        )

    def _on_project_changed(self, project) -> None:
        self.tab_pipeline.set_project(project)
        self.tab_settings.set_project(project)
        self.status_lbl.setText(
            f"Активный проект: {project.name}" if project else "Проект не выбран")


# ─────────────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
