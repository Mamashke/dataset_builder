"""
test_color_match.py — проверка цветового сопоставления compositor.

Шаги:
  1. Берёт первые 60 аннотированных кадров из frames/real → extract_persons
     (создаёт persons/metadata.json с bg_color для каждой фигуры).
  2. Из фонов (кадры с пустой аннотацией) выбирает два с наиболее
     контрастными средними цветами — условно «зимний» (холодный, светлый)
     и «летний» (тёплый, зелёный).
  3. Для каждого фона вызывает _pick_person_by_color и сохраняет итоговый
     скомпонованный кадр в корневую папку проекта.
  4. В консоль выводит bg_color выбранного человека и bg_color фона —
     можно визуально сверить насколько они совпадают.

Запуск:
  cd C:\diplom\dataset_builder
  python test_color_match.py
"""

import sys
import json
import random
import logging
from pathlib import Path

import cv2
import numpy as np

# Корень проекта в sys.path — чтобы импорты modules.* работали
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_color_match")

from modules.project import Project
from modules.compositor import (
    _read_image, _write_image, _bg_color_around_bbox,
    _pick_person_by_color, _load_metadata, _blend_person,
    _rotate_crop, _avg_person_height,
)

_IMG_EXTS   = {".jpg", ".jpeg", ".png"}
_SCALE_VAR  = 0.20
PROJECT_NAME = "bpla_dataset"
# Сколько аннотированных кадров обработать для генерации metadata
EXTRACT_LIMIT = 60
# В топ-N ближайших по цвету выбирается случайный
TOP_K = 5


def extract_persons_subset(project: Project, limit: int) -> int:
    """Запускает extract_persons на первых `limit` аннотированных кадрах.

    Создаёт persons/metadata.json с полем bg_color для каждой фигуры.
    Если metadata уже содержит bg_color — пропускает (не перегенерирует).
    Возвращает число вырезанных фигур.
    """
    frames_dir = project.frames_real_dir
    ann_dir    = project.annotations_dir / "real"
    persons_dir = project.persons_dir

    # Проверяем существующий metadata — если bg_color уже есть, ничего не делаем
    meta_path = persons_dir / "metadata.json"
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        with_color = [m for m in existing if "bg_color" in m]
        if with_color:
            logger.info(f"metadata.json уже содержит bg_color для {len(with_color)} фигур — пропускаем extract")
            return len(with_color)

    persons_dir.mkdir(parents=True, exist_ok=True)

    # Берём только изображения, у которых есть непустая аннотация
    image_files = sorted(
        p for p in frames_dir.iterdir() if p.suffix.lower() in _IMG_EXTS
    )
    metadata  = []
    extracted = 0
    processed = 0

    for img_path in image_files:
        if processed >= limit:
            break
        label_path = ann_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        content = label_path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        img = _read_image(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        processed += 1

        for obj_idx, line in enumerate(content.splitlines()):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                x_c, y_c, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue

            x1 = max(0, int((x_c - bw / 2) * w))
            y1 = max(0, int((y_c - bh / 2) * h))
            x2 = min(w, int((x_c + bw / 2) * w))
            y2 = min(h, int((y_c + bh / 2) * h))

            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_name = f"{img_path.stem}_obj{obj_idx}.jpg"
            out_path = persons_dir / out_name
            # Сохраняем только если файл уже существует или сохранение успешно
            if out_path.exists() or _write_image(out_path, crop):
                bg_color = _bg_color_around_bbox(img, x1, y1, x2, y2)
                metadata.append({
                    "filename":           out_name,
                    "original_height_px": y2 - y1,
                    "bg_color":           bg_color,
                })
                extracted += 1

    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Записано {len(metadata)} записей в metadata.json (кадров обработано: {processed})")
    return extracted


def find_winter_summer_backgrounds(project: Project):
    """Находит два фоновых кадра с максимально разными цветовыми профилями.

    «Зимний» — высокий средний по B+R, низкий G (холодный светлый).
    «Летний» — высокий G относительно R+B (тёплый зелёный).
    Ищет среди кадров с пустой аннотацией.
    """
    frames_dir = project.frames_real_dir
    ann_dir    = project.annotations_dir / "real"

    candidates = []
    for img_path in frames_dir.iterdir():
        if img_path.suffix.lower() not in _IMG_EXTS:
            continue
        # Исключаем GAN- и SD-сгенерированные кадры — тестируем только оригиналы
        stem = img_path.stem
        if stem.startswith("sd_") or stem.startswith("gan_") or stem.startswith("comp_"):
            continue
        label_path = ann_dir / (img_path.stem + ".txt")
        if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
            continue   # позитивный кадр — пропускаем
        candidates.append(img_path)

    if len(candidates) < 2:
        raise RuntimeError(f"Недостаточно фоновых кадров: найдено {len(candidates)}")

    logger.info(f"Кандидатов для фона: {len(candidates)}")

    # Считаем средний цвет для каждого кандидата (ресайзим для скорости)
    scored = []
    for p in candidates:
        img = _read_image(p)
        if img is None:
            continue
        small = cv2.resize(img, (64, 64))
        b = float(small[:, :, 0].mean())
        g = float(small[:, :, 1].mean())
        r = float(small[:, :, 2].mean())
        # Индекс «зимности»: светлость при низком зелёном = (b+r)/2 - g
        winter_score = (b + r) / 2 - g
        # Индекс «летности»: зелень = g - (b+r)/2
        summer_score = g - (b + r) / 2
        scored.append((p, b, g, r, winter_score, summer_score))

    # Берём самый «зимний» и самый «летний»
    winter_bg = max(scored, key=lambda x: x[4])
    summer_bg = max(scored, key=lambda x: x[5])

    logger.info(
        f"Зимний фон: {winter_bg[0].name} | B={winter_bg[1]:.0f} G={winter_bg[2]:.0f} R={winter_bg[3]:.0f}"
    )
    logger.info(
        f"Летний фон:  {summer_bg[0].name} | B={summer_bg[1]:.0f} G={summer_bg[2]:.0f} R={summer_bg[3]:.0f}"
    )

    return winter_bg[0], summer_bg[0]


def compose_one(bg_path: Path, project: Project, meta_by_name: dict,
                person_files: list, label: str) -> np.ndarray:
    """Компонует один кадр: вставляет 1–3 людей на фон с цветовым подбором."""
    bg = _read_image(bg_path)
    if bg is None:
        raise RuntimeError(f"Не удалось прочитать фон: {bg_path}")
    bg = bg.copy()
    if bg.shape[0] != 640 or bg.shape[1] != 640:
        bg = cv2.resize(bg, (640, 640), interpolation=cv2.INTER_AREA)

    bg_h, bg_w = bg.shape[:2]

    # Средний цвет фона для подбора человека
    bg_mean_color = [
        int(bg[:, :, 0].mean()),
        int(bg[:, :, 1].mean()),
        int(bg[:, :, 2].mean()),
    ]
    logger.info(f"[{label}] Средний цвет фона: B={bg_mean_color[0]} G={bg_mean_color[1]} R={bg_mean_color[2]}")

    n_persons = random.randint(1, 3)
    inserted  = 0

    for _ in range(n_persons):
        chosen = _pick_person_by_color(person_files, meta_by_name, bg_mean_color, top_k=TOP_K)
        person = _read_image(chosen)
        if person is None:
            continue

        p_h_orig, p_w_orig = person.shape[:2]
        if p_h_orig == 0 or p_w_orig == 0:
            continue

        # Логируем bg_color выбранного человека
        m = meta_by_name.get(chosen.name, {})
        person_color = m.get("bg_color", "н/д")
        logger.info(f"[{label}] Выбран: {chosen.name} | bg_color человека: {person_color}")

        # Берём оригинальный размер патча из metadata — точнее чем shape JPEG
        m_entry = meta_by_name.get(chosen.name, {})
        patch_h = m_entry.get("patch_h", p_h_orig)
        patch_w = m_entry.get("patch_w", p_w_orig)

        # Жёсткий целевой размер 15–35 px — как в compose()
        target_h = random.randint(15, 35)
        scale    = target_h / patch_h
        new_w    = max(1, int(patch_w * scale))
        person   = cv2.resize(person, (new_w, target_h), interpolation=cv2.INTER_AREA)

        # Поворот и размытие
        angle = random.uniform(-15, 15)
        if abs(angle) > 0.5:
            person = _rotate_crop(person, angle)

        ksize  = random.randint(1, 3) * 2 + 1
        person = cv2.GaussianBlur(person, (ksize, ksize), 0)

        # Яркость
        br = random.uniform(0.8, 1.2)
        person = np.clip(person.astype(np.float32) * br, 0, 255).astype(np.uint8)

        p_h, p_w = person.shape[:2]
        if p_w >= bg_w or p_h >= bg_h:
            continue

        x = random.randint(0, bg_w - p_w)
        y = random.randint(0, bg_h - p_h)
        _blend_person(bg, person, x, y)
        inserted += 1

    logger.info(f"[{label}] Вставлено людей: {inserted}")
    return bg


def main():
    project = Project.load(PROJECT_NAME)

    # 1. Генерируем/дополняем metadata.json с bg_color
    logger.info("=== Шаг 1: extract_persons (subset) ===")
    n_extracted = extract_persons_subset(project, limit=EXTRACT_LIMIT)
    logger.info(f"Готово: {n_extracted} фигур с bg_color")

    # 2. Загружаем metadata и список фигур
    metadata     = _load_metadata(project.persons_dir)
    meta_by_name = {m["filename"]: m for m in metadata}

    person_files = [
        p for p in project.persons_dir.iterdir()
        if p.suffix.lower() in _IMG_EXTS
    ]
    if not person_files:
        logger.error("Нет фигур людей в persons/")
        sys.exit(1)

    with_color = sum(1 for p in person_files if p.name in meta_by_name and "bg_color" in meta_by_name[p.name])
    logger.info(f"Фигур всего: {len(person_files)}, с bg_color: {with_color}")

    # 3. Находим зимний и летний фон
    logger.info("=== Шаг 2: поиск контрастных фонов ===")
    winter_path, summer_path = find_winter_summer_backgrounds(project)

    # 4. Компонуем
    logger.info("=== Шаг 3: компоновка ===")
    winter_out = compose_one(winter_path, project, meta_by_name, person_files, "ЗИМНИЙ")
    summer_out = compose_one(summer_path, project, meta_by_name, person_files, "ЛЕТНИЙ")

    # 5. Сохраняем в корень проекта
    out_winter = ROOT / "test_winter_composed.jpg"
    out_summer = ROOT / "test_summer_composed.jpg"
    _write_image(out_winter, winter_out)
    _write_image(out_summer, summer_out)

    logger.info(f"Сохранено: {out_winter}")
    logger.info(f"Сохранено: {out_summer}")

    # 6. Сводка по цветам
    print("\n=== Сводка ===")
    print(f"Зимний фон ({winter_path.name}):")
    bg = cv2.resize(_read_image(winter_path), (64, 64))
    print(f"  Средний BGR: B={bg[:,:,0].mean():.0f} G={bg[:,:,1].mean():.0f} R={bg[:,:,2].mean():.0f}")
    print(f"Летний фон ({summer_path.name}):")
    bg = cv2.resize(_read_image(summer_path), (64, 64))
    print(f"  Средний BGR: B={bg[:,:,0].mean():.0f} G={bg[:,:,1].mean():.0f} R={bg[:,:,2].mean():.0f}")
    print(f"\nОбраз с людьми сохранены:")
    print(f"  {out_winter}")
    print(f"  {out_summer}")


if __name__ == "__main__":
    main()
