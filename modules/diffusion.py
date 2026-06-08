# modules/diffusion.py — генерация фоновых сцен через Stable Diffusion.
#
# Генерирует синтетические фоны с видом БПЛА для последующей
# вставки людей методом Copy-Paste (compositor.py).
#
# Два семантических типа фонов:
#   sd_open_   — открытые пространства: луга, поля, просеки
#   sd_forest_ — лесные сцены: плотный лес разных сезонов
#
# Compositor использует только sd_open_ фоны — в открытых сценах
# силуэт человека хорошо различим, лесные сцены слишком шумные.
#
# Публичный интерфейс:
#   generate_backgrounds(project, count=200, background_type="all")

import random
from pathlib import Path

import cv2
import numpy as np

from modules.logger import get_logger
from modules.project import Project

logger = get_logger(__name__)

# Промпты открытых пространств — луга, поля, просеки (без плотного леса)
_PROMPTS_OPEN = [
    "aerial drone view open field meadow top down, summer green grass, no trees",
    "aerial drone view snowy open field top down, winter flat landscape",
    "aerial drone view forest clearing top down, grass open area, no dense trees",
    "aerial drone view grassland top down, open flat landscape, autumn",
    "aerial drone view dirt path open terrain top down, sparse vegetation",
]

# Промпты лесных сцен — плотный лес разных сезонов
_PROMPTS_FOREST = [
    "aerial drone view dense conifer forest top down, winter snow",
    "aerial drone view mixed forest top down, autumn leaves",
    "aerial drone view dense forest top down, spring green",
    "aerial drone view forest canopy top down, summer",
]

# Единый негативный промпт для обоих типов — явно исключаем городскую
# и спортивную инфраструктуру, технику и артефакты качества
_NEGATIVE_PROMPT = (
    "people, humans, person, cars, buildings, roads, stadium, "
    "urban, city, football field, sports, parking, artificial, "
    "construction, low quality, blurry, distorted"
)

# Префиксы имён файлов для каждого типа
_PREFIX_OPEN   = "sd_open_"
_PREFIX_FOREST = "sd_forest_"


def _generate_batch(
    pipe,
    device:      str,
    prompts:     list,
    prefix:      str,
    count:       int,
    out_img_dir: Path,
    out_ann_dir: Path,
    label:       str,
) -> int:
    """Генерирует один батч изображений заданного типа.

    Args:
        pipe:        загруженный StableDiffusionPipeline.
        device:      "cuda" или "cpu".
        prompts:     список промптов для данного типа.
        prefix:      префикс имён файлов (sd_open_ или sd_forest_).
        count:       количество изображений для генерации.
        out_img_dir: папка сохранения кадров.
        out_ann_dir: папка сохранения пустых аннотаций.
        label:       название типа для вывода ("открытые" / "лесные").

    Returns:
        Фактическое число сохранённых изображений.
    """
    import torch

    # Определяем стартовый номер — не перезаписываем существующие кадры данного типа
    existing: set = set()
    for f in out_img_dir.glob(f"{prefix}*.jpg"):
        try:
            existing.add(int(f.stem[len(prefix):]))
        except ValueError:
            pass
    start_num = max(existing, default=-1) + 1

    n_prompts = len(prompts)
    logger.info(
        f"_generate_batch | тип={label} | prefix={prefix} | "
        f"count={count} | нумерация с {prefix}{start_num:06d}"
    )
    print(f"  Генерация {count} сцен — {label}...")

    generated = 0

    for i in range(count):
        # Равномерное распределение по промптам (цикл по индексу)
        prompt = prompts[i % n_prompts]

        # Случайный seed для каждого изображения
        seed      = random.randint(0, 2 ** 32 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)

        try:
            result    = pipe(
                prompt              = prompt,
                negative_prompt     = _NEGATIVE_PROMPT,
                width               = 512,
                height              = 512,
                num_inference_steps = 20,
                guidance_scale      = 7.5,
                generator           = generator,
            )
            pil_image = result.images[0]
        except Exception as exc:
            logger.error(
                f"Ошибка генерации {prefix}{start_num + generated:06d}: {exc}"
            )
            continue

        # PIL → numpy → cv2 (RGB → BGR)
        img_np = np.array(pil_image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Ресайз до 640×640 для совместимости с пайплайном
        img_cv = cv2.resize(img_cv, (640, 640), interpolation=cv2.INTER_LANCZOS4)

        # Сохраняем через imencode + write_bytes (Unicode-совместимо на Windows)
        frame_num = start_num + generated
        img_path  = out_img_dir / f"{prefix}{frame_num:06d}.jpg"
        ann_path  = out_ann_dir / f"{prefix}{frame_num:06d}.txt"

        ok, buf = cv2.imencode(".jpg", img_cv, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            logger.warning(f"Не удалось закодировать: {img_path.name}")
            continue
        img_path.write_bytes(buf.tobytes())

        # Пустой txt — негативный пример (фон без людей)
        ann_path.write_text("", encoding="utf-8")

        generated += 1

        # Логируем прогресс каждые 10 изображений
        if generated % 10 == 0:
            logger.info(
                f"Сгенерировано {label}: {generated}/{count} "
                f"(промпт: {prompt[:50]}...)"
            )
            print(f"  Сгенерировано {label}: {generated}/{count}")

    logger.info(f"Батч '{label}' завершён: создано={generated} → {out_img_dir}")
    return generated


def generate_backgrounds(
    project:         Project,
    count:           int = 200,
    background_type: str = "all",
) -> dict:
    """Генерирует фоновые сцены с видом БПЛА через Stable Diffusion 1.5.

    Типы генерации:
      "open"   → только открытые пространства, файлы sd_open_*.jpg
      "forest" → только лесные сцены,          файлы sd_forest_*.jpg
      "all"    → оба типа, count/2 каждого (открытые получают остаток при нечётном)

    Изображения сохраняются как негативные примеры (пустые аннотации):
    - кадры:      project.frames_real_dir / {prefix}{n:06d}.jpg
    - аннотации:  project.annotations_dir / "real" / {prefix}{n:06d}.txt

    Args:
        project:         объект Project с путями к папкам проекта.
        count:           общее количество генерируемых изображений.
        background_type: "open", "forest" или "all" (по умолчанию "all").

    Returns:
        {"generated": N, "open": N_open, "forest": N_forest}

    Raises:
        ValueError: если background_type не входит в допустимые значения.
    """
    import torch

    if background_type not in ("open", "forest", "all"):
        raise ValueError(
            f"Недопустимый тип фонов: '{background_type}'. "
            "Допустимые значения: open, forest, all"
        )

    # Определяем устройство — GPU (float16) или CPU (float32) с предупреждением
    if torch.cuda.is_available():
        device = "cuda"
        dtype  = torch.float16
        logger.info("generate_backgrounds: используется GPU (float16)")
    else:
        device = "cpu"
        dtype  = torch.float32
        print(
            "Предупреждение: GPU недоступен, генерация на CPU. "
            "Ожидайте значительное замедление (несколько минут на изображение)."
        )
        logger.warning(
            "generate_backgrounds: GPU недоступен, используется CPU (float32)"
        )

    # Загружаем pipeline один раз — он используется для всех батчей
    print("Загрузка Stable Diffusion pipeline...")
    logger.info(
        f"Загрузка StableDiffusionPipeline: runwayml/stable-diffusion-v1-5 | "
        f"тип={background_type} | count={count}"
    )

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
    ).to(device)
    # Отключаем встроенный прогресс-бар diffusers — используем свой лог
    pipe.set_progress_bar_config(disable=True)

    # Подготавливаем выходные папки
    out_img_dir = project.frames_real_dir
    out_ann_dir = project.annotations_dir / "real"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    print(f"Генерация {count} фоновых сцен (тип: {background_type})...")
    logger.info(
        f"generate_backgrounds | проект={project.name} | "
        f"тип={background_type} | count={count}"
    )

    n_open   = 0
    n_forest = 0

    if background_type == "open":
        n_open = _generate_batch(
            pipe, device, _PROMPTS_OPEN, _PREFIX_OPEN,
            count, out_img_dir, out_ann_dir, "открытые",
        )

    elif background_type == "forest":
        n_forest = _generate_batch(
            pipe, device, _PROMPTS_FOREST, _PREFIX_FOREST,
            count, out_img_dir, out_ann_dir, "лесные",
        )

    else:
        # Открытые получают остаток при нечётном count — они важнее для compositor
        count_open   = count // 2 + count % 2
        count_forest = count // 2
        n_open = _generate_batch(
            pipe, device, _PROMPTS_OPEN, _PREFIX_OPEN,
            count_open, out_img_dir, out_ann_dir, "открытые",
        )
        n_forest = _generate_batch(
            pipe, device, _PROMPTS_FOREST, _PREFIX_FOREST,
            count_forest, out_img_dir, out_ann_dir, "лесные",
        )

    total = n_open + n_forest

    logger.info(
        f"generate_backgrounds завершён: всего={total} "
        f"(открытые={n_open}, лесные={n_forest}) → {out_img_dir}"
    )
    print(
        f"Готово: сгенерировано {total} фоновых сцен "
        f"(открытые={n_open}, лесные={n_forest}) → {out_img_dir}"
    )

    project.update_stats({
        "sd_generated": total,
        "sd_open":      n_open,
        "sd_forest":    n_forest,
    })
    return {"generated": total, "open": n_open, "forest": n_forest}
