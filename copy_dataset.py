# copy_dataset.py — копирует готовый датасет в структуру проекта.
#
# Источники:
#   C:/dataset/public/images/  + C:/dataset/private/images/  → frames/real/
#   C:/dataset/public/labels/  + C:/dataset/private/labels/  → annotations/real/
#
# Запуск: python copy_dataset.py

import shutil
from pathlib import Path

import config

# ---------------------------------------------------------------------------
# Источники и назначения
# ---------------------------------------------------------------------------

IMAGE_SOURCES = [
    Path("C:/dataset/public/images"),
    Path("C:/dataset/private/images"),
]

LABEL_SOURCES = [
    Path("C:/dataset/public/labels"),
    Path("C:/dataset/private/labels"),
]

IMAGE_DST = config.FRAMES_REAL_DIR
LABEL_DST = config.ANNOTATIONS_DIR / "real"


def copy_files(src_dirs: list, dst_dir: Path, extensions: set) -> list:
    """Копирует файлы с заданными расширениями из нескольких папок в одну.

    Пропускает файлы, которые уже существуют в dst_dir (не перезаписывает),
    чтобы повторный запуск был безопасным.

    Args:
        src_dirs:   список исходных директорий.
        dst_dir:    папка назначения (создаётся если не существует).
        extensions: допустимые расширения в нижнем регистре (например {".jpg"}).

    Returns:
        Список путей к скопированным файлам.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    for src_dir in src_dirs:
        if not src_dir.exists():
            print(f"  [WARN] Папка не найдена, пропускаем: {src_dir}")
            continue

        files = sorted(f for f in src_dir.iterdir() if f.suffix.lower() in extensions)

        for src_file in files:
            dst_file = dst_dir / src_file.name

            # Пропускаем уже скопированные файлы
            if dst_file.exists():
                continue

            shutil.copy2(str(src_file), str(dst_file))
            copied.append(dst_file)

    return copied


def print_stats(copied_images: list, copied_labels: list) -> None:
    """Выводит статистику по скопированным файлам."""

    # Считаем непустые и пустые txt
    with_objects = 0
    empty = 0

    # Проверяем все txt в папке назначения (не только только что скопированные),
    # чтобы статистика отражала полное состояние папки после копирования.
    all_labels = list(LABEL_DST.glob("*.txt"))
    for lbl in all_labels:
        if lbl.read_text(encoding="utf-8").strip():
            with_objects += 1
        else:
            empty += 1

    all_images = len(list(IMAGE_DST.glob("*.jpg")) + list(IMAGE_DST.glob("*.jpeg")))

    print()
    print("=" * 50)
    print("СТАТИСТИКА:")
    print(f"  Скопировано изображений (сейчас)    : {len(copied_images)}")
    print(f"  Скопировано txt-файлов (сейчас)     : {len(copied_labels)}")
    print()
    print(f"  Всего изображений в frames/real/    : {all_images}")
    print(f"  Всего меток в annotations/real/     : {len(all_labels)}")
    print(f"    из них с объектами (позитивные)   : {with_objects}")
    print(f"    из них пустых     (негативные)    : {empty}")
    print("=" * 50)
    print(f"  Изображения : {IMAGE_DST}")
    print(f"  Аннотации   : {LABEL_DST}")
    print("=" * 50)


def main() -> None:
    print("Копирование изображений...")
    copied_images = copy_files(IMAGE_SOURCES, IMAGE_DST, {".jpg", ".jpeg"})
    print(f"  Скопировано: {len(copied_images)} файлов")

    print("Копирование аннотаций...")
    copied_labels = copy_files(LABEL_SOURCES, LABEL_DST, {".txt"})
    print(f"  Скопировано: {len(copied_labels)} файлов")

    print_stats(copied_images, copied_labels)


if __name__ == "__main__":
    main()
