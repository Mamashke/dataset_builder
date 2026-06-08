# modules/diffusion.py — генерация фоновых сцен через Stable Diffusion.
#
# Генерирует синтетические фоны с видом БПЛА для последующей
# вставки людей методом Copy-Paste (compositor.py).
#
# Публичный интерфейс:
#   generate_backgrounds(project, count=200) — генерация и сохранение фонов SD.

import random
from pathlib import Path

import cv2
import numpy as np

from modules.logger import get_logger
from modules.project import Project

logger = get_logger(__name__)

# Промпты аэросъёмки БПЛА — дикая природа без городских объектов,
# разные сезоны и условия погоды
_PROMPTS = [
    # Зима
    "top-down aerial drone photograph of wild forest, snow covered conifer trees, "
    "no people, wilderness, overcast sky, winter",
    "top-down aerial drone photograph of open snowy field, frozen ground, "
    "sparse dry grass, no people, remote wilderness, winter",
    # Весна / лето
    "top-down aerial drone photograph of dense deciduous forest, "
    "fresh green canopy, no people, wild nature, spring",
    "top-down aerial drone photograph of wild meadow and grassland, "
    "tall grass, wildflowers, no people, countryside, summer",
    "top-down aerial drone photograph of mixed forest and open clearings, "
    "green and brown tones, no people, remote woodland, summer",
    # Осень
    "top-down aerial drone photograph of autumn forest, "
    "orange red yellow foliage, fallen leaves on ground, no people, wilderness",
    "top-down aerial drone photograph of forest with dirt trail, "
    "autumn leaves, muddy path, no people, wild nature",
    # Сложные погодные условия
    "top-down aerial drone photograph of forest in fog and mist, "
    "diffuse lighting, no people, remote woodland, moody weather",
    "top-down aerial drone photograph of wet meadow after rain, "
    "puddles on ground, overcast, no people, wild field",
    "top-down aerial drone photograph of dense conifer forest, "
    "dark green canopy, shadows, no people, wilderness, cloudy",
    # Водоёмы и рельеф
    "top-down aerial drone photograph of river through wild forest, "
    "rocky riverbank, no people, remote nature",
    "top-down aerial drone photograph of forest edge next to open field, "
    "tree line, no people, wild landscape, daylight",
]

# Негативный промпт — явно исключаем городскую и спортивную инфраструктуру,
# технику и артефакты качества
_NEGATIVE_PROMPT = (
    "people, humans, person, crowd, "
    "cars, vehicles, trucks, "
    "buildings, houses, rooftops, "
    "stadium, arena, sports field, football field, soccer field, "
    "artificial grass, synthetic turf, "
    "parking lot, parking, "
    "urban, city, town, suburb, "
    "road, highway, street, sidewalk, "
    "construction, crane, fence, "
    "low quality, blurry, distorted, watermark, text"
)


def generate_backgrounds(project: Project, count: int = 200) -> dict:
    """Генерирует фоновые сцены с видом БПЛА через Stable Diffusion 1.5.

    Изображения сохраняются как негативные примеры (без людей):
    - кадры: project.frames_real_dir / sd_{n:06d}.jpg
    - пустые аннотации: project.annotations_dir / "real" / sd_{n:06d}.txt

    Args:
        project: объект Project с путями к папкам проекта.
        count:   количество генерируемых фоновых сцен (по умолчанию 200).

    Returns:
        {"generated": N} — фактическое число созданных изображений.
    """
    import torch

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

    # Загружаем Stable Diffusion pipeline (загрузка веса занимает 1–2 мин)
    print("Загрузка Stable Diffusion pipeline...")
    logger.info("Загрузка StableDiffusionPipeline: runwayml/stable-diffusion-v1-5")

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

    # Стартовый номер — не перезаписываем уже существующие sd-кадры
    existing_nums: set = set()
    for f in out_img_dir.glob("sd_*.jpg"):
        try:
            existing_nums.add(int(f.stem[3:]))
        except ValueError:
            pass
    start_num = max(existing_nums, default=-1) + 1

    n_prompts = len(_PROMPTS)
    logger.info(
        f"generate_backgrounds | проект={project.name} | "
        f"count={count} | нумерация с sd_{start_num:06d}"
    )
    print(f"Генерация {count} фоновых сцен...")

    generated = 0

    for i in range(count):
        # Равномерное распределение по промптам (цикл по индексу)
        prompt = _PROMPTS[i % n_prompts]

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
            logger.error(f"Ошибка генерации изображения {i}: {exc}")
            continue

        # PIL → numpy → cv2 (RGB → BGR)
        img_np = np.array(pil_image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Ресайз до 640×640 для совместимости с пайплайном
        img_cv = cv2.resize(img_cv, (640, 640), interpolation=cv2.INTER_LANCZOS4)

        # Сохраняем через imencode + write_bytes (Unicode-совместимо на Windows)
        frame_num = start_num + generated
        img_path  = out_img_dir / f"sd_{frame_num:06d}.jpg"
        ann_path  = out_ann_dir / f"sd_{frame_num:06d}.txt"

        ok, buf = cv2.imencode(".jpg", img_cv, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            logger.warning(f"Не удалось закодировать кадр: {img_path.name}")
            continue
        img_path.write_bytes(buf.tobytes())

        # Пустой txt — негативный пример (фон без людей)
        ann_path.write_text("", encoding="utf-8")

        generated += 1

        # Логируем прогресс каждые 10 изображений
        if generated % 10 == 0:
            short_prompt = prompt[:50]
            logger.info(
                f"Сгенерировано: {generated}/{count} "
                f"(промпт: {short_prompt}...)"
            )
            print(
                f"Сгенерировано: {generated}/{count} "
                f"(промпт: {short_prompt}...)"
            )

    logger.info(
        f"generate_backgrounds завершён: создано={generated} → {out_img_dir}"
    )
    print(f"Готово: сгенерировано {generated} фоновых сцен → {out_img_dir}")

    project.update_stats({"sd_generated": generated})
    return {"generated": generated}
