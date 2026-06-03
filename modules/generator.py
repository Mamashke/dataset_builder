# modules/generator.py — DCGAN для генерации синтетических кадров БПЛА.
#
# Архитектура: DCGAN (Deep Convolutional GAN), разрешение 64×64 / 128×128 / 256×256.
#
# Публичный интерфейс:
#   Generator(latent_dim, image_size)           — архитектура генератора (nn.Module)
#   Discriminator(image_size)                   — архитектура дискриминатора (nn.Module)
#   train_gan(project, epochs, batch_size, image_size)  — обучение DCGAN
#   generate_images(project, count)                     — генерация кадров
#
# Зависимости: torch, torchvision, cv2, numpy, Pillow

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config
from modules.logger import get_logger
from modules.project import Project

logger = get_logger(__name__)

# Размер вектора шума (латентного пространства)
LATENT_DIM = 100

# Суффиксы аугментированных кадров — исключаем их из обучающей выборки
_AUG_SUFFIXES = ("_fog", "_rain", "_noise", "_blur", "_brightness")


# ---------------------------------------------------------------------------
# Вспомогательный датасет
# ---------------------------------------------------------------------------

class _DroneFrameDataset(Dataset):
    """Загружает кадры из списка путей и применяет трансформации."""

    def __init__(self, paths: list, transform=None):
        self.paths     = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        # numpy+cv2 вместо PIL.open() — поддержка Unicode-путей на Windows
        buf = np.fromfile(str(self.paths[idx]), dtype=np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            # Повреждённый файл — возвращаем чёрный кадр
            img_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        if self.transform:
            img = self.transform(img)
        return img


# ---------------------------------------------------------------------------
# Стандартная инициализация весов DCGAN
# ---------------------------------------------------------------------------

def _weights_init(m: nn.Module) -> None:
    """Инициализирует Conv и BatchNorm по рекомендации авторов DCGAN."""
    cname = m.__class__.__name__
    if "Conv" in cname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ---------------------------------------------------------------------------
# Архитектура: Генератор
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """DCGAN-генератор с динамической архитектурой.

    Вход:  тензор (batch, latent_dim, 1, 1) — случайный шум z ~ N(0, 1).
    Выход: тензор (batch, 3, image_size, image_size) в диапазоне [-1, 1].

    image_size=64:  100 → 512 → 256 → 128 → 64 → 3
    image_size=128: 100 → 512 → 256 → 128 → 64 → 32 → 3
    image_size=256: 100 → 512 → 256 → 128 → 64 → 32 → 16 → 3

    Каждый шаг удвоения разрешения сверх базового 64px — один дополнительный блок.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, image_size: int = 64):
        super().__init__()
        # Количество дополнительных блоков апсэмплинга сверх базовых четырёх
        _extra = {64: 0, 128: 1, 256: 2}.get(image_size, 0)

        layers = [
            # Блок 1: (latent_dim, 1, 1) → (512, 4, 4)
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Блок 2: (512, 4, 4) → (256, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Блок 3: (256, 8, 8) → (128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Блок 4: (128, 16, 16) → (64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        ]

        # Дополнительные блоки для image_size > 64: каждый удваивает разрешение
        ch = 64
        for _ in range(_extra):
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch // 2),
                nn.ReLU(True),
            ]
            ch //= 2

        # Финальный слой → (3, image_size, image_size)
        layers += [
            nn.ConvTranspose2d(ch, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        ]

        self.main = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


# ---------------------------------------------------------------------------
# Архитектура: Дискриминатор
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """DCGAN-дискриминатор с динамической архитектурой.

    Вход:  тензор (batch, 3, image_size, image_size).
    Выход: тензор (batch,) — вероятность того, что изображение реальное.

    image_size=64:  3 → 64 → 128 → 256 → 512 → 1
    image_size=128: 3 → 32 → 64 → 128 → 256 → 512 → 1
    image_size=256: 3 → 16 → 32 → 64 → 128 → 256 → 512 → 1

    Первый блок всегда без BatchNorm — стандарт DCGAN для дискриминатора.
    """

    def __init__(self, image_size: int = 64):
        super().__init__()
        # Дополнительные блоки в начале: 64→0, 128→1, 256→2
        _extra = {64: 0, 128: 1, 256: 2}.get(image_size, 0)

        # Каналы первого блока: 64 для 64px, 32 для 128px, 16 для 256px
        first_ch = 64 >> _extra

        # Первый блок — без BatchNorm (стандарт DCGAN для дискриминатора)
        layers = [
            nn.Conv2d(3, first_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Дополнительные блоки для image_size > 64: удваиваем каналы
        ch = first_ch
        for _ in range(_extra):
            layers += [
                nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch *= 2

        # Основные блоки (одинаковы для всех image_size): → 128 → 256 → 512
        for out_ch in [128, 256, 512]:
            layers += [
                nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = out_ch

        # Финальный слой: (512, 4, 4) → (1, 1, 1)
        layers += [
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        ]

        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1)


# ---------------------------------------------------------------------------
# Публичная функция: обучение GAN
# ---------------------------------------------------------------------------

def train_gan(
    project: Project,
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = None,
    on_epoch=None,
) -> dict:
    """Обучает DCGAN на оригинальных кадрах проекта.

    Ищет кадры сначала через data_sources["frames"]["real"]; если не задан —
    берёт project.frames_real_dir. Исключает аугментированные кадры и уже
    сгенерированные (gan_*). Обучает генератор и дискриминатор с оптимизатором
    Adam (lr=0.0002, betas=(0.5, 0.999)).

    Каждые 10 эпох сохраняет сетку 4×4 из 16 сэмплов в project.gan_samples_dir.
    После обучения сохраняет generator.pth и discriminator.pth в project.gan_model_dir.

    Args:
        project:    объект Project — определяет пути к кадрам и результатам.
        epochs:     количество эпох (по умолчанию 100).
        batch_size: размер батча (по умолчанию 16).
        image_size: разрешение обучения: 64, 128 или 256. None → config.GAN_IMAGE_SIZE.
        on_epoch:   колбэк (epoch, total, loss_g, loss_d) → bool; True = досрочный стоп.

    Returns:
        {"epochs": N, "final_loss_g": float, "final_loss_d": float}

    Raises:
        FileNotFoundError: если папка с кадрами не найдена или пуста.
        ValueError:        если image_size не входит в {64, 128, 256}.
    """
    get_logger(__name__, project.logs_dir)

    # Берём image_size из config если не передан явно
    if image_size is None:
        image_size = config.GAN_IMAGE_SIZE

    if image_size not in (64, 128, 256):
        raise ValueError(f"image_size должен быть 64, 128 или 256, получено: {image_size}")

    # Сначала проверяем data_sources — путь может быть задан вне папки проекта
    frames_dir = project.get_source("frames", "real")

    # Если не задан — берём папку проекта
    if frames_dir is None:
        frames_dir = project.frames_real_dir

    # Папка не существует или пустая — ошибка с подсказкой по исправлению
    if not frames_dir.exists() or not any(frames_dir.iterdir()):
        raise FileNotFoundError(
            f"Кадры не найдены. Укажите путь: "
            f"python main.py --project '{project.name}' "
            f"--set-source frames real C:/path/"
        )

    # Фильтруем только оригинальные кадры — без суффиксов аугментации и gan_
    all_images = [
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and not any(p.stem.endswith(s) for s in _AUG_SUFFIXES)
        and not p.stem.startswith("gan_")
    ]

    if not all_images:
        raise FileNotFoundError(
            f"Оригинальные кадры не найдены в {frames_dir}.\n"
            f"Сначала запустите шаг load."
        )

    # Если кадров меньше batch_size — уменьшаем batch_size
    if len(all_images) < batch_size:
        logger.warning(
            f"Мало кадров ({len(all_images)}) — "
            f"batch_size уменьшен {batch_size} → {len(all_images)}"
        )
        batch_size = len(all_images)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        f"train_gan | проект={project.name} | кадров={len(all_images)} | "
        f"epochs={epochs} | batch_size={batch_size} | "
        f"image_size={image_size} | device={device}"
    )
    print(
        f"Устройство: {device}  |  кадров: {len(all_images)}  |  "
        f"разрешение: {image_size}×{image_size}"
    )

    # Трансформация: ресайз до image_size×image_size → тензор → нормализация в [-1, 1]
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    dataset    = _DroneFrameDataset(all_images, transform)
    # num_workers=0 — однопоточная загрузка: на Windows вызов из QThread с
    # num_workers>0 использует spawn и может зависнуть внутри Qt-приложения
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )

    # Инициализация моделей с нужным разрешением
    G = Generator(LATENT_DIM, image_size).to(device)
    D = Discriminator(image_size).to(device)
    G.apply(_weights_init)
    D.apply(_weights_init)

    criterion = nn.BCELoss()
    opt_G     = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D     = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Label smoothing: 0.9 вместо 1.0 помогает стабилизировать дискриминатор
    real_label = 0.9
    fake_label = 0.0

    # Фиксированный шум для наглядного контроля прогресса в сэмплах
    fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=device)

    project.gan_samples_dir.mkdir(parents=True, exist_ok=True)
    project.gan_model_dir.mkdir(parents=True, exist_ok=True)

    loss_g_last = 0.0
    loss_d_last = 0.0
    last_epoch  = 0   # фактически завершённых эпох (может быть < epochs при стопе)

    for epoch in range(1, epochs + 1):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        n_batches    = 0

        for real_imgs in dataloader:
            real_imgs = real_imgs.to(device)
            batch_n   = real_imgs.size(0)

            # ── Шаг дискриминатора ───────────────────────────
            D.zero_grad()

            # Реальные изображения
            labels_real = torch.full((batch_n,), real_label, device=device)
            loss_d_real = criterion(D(real_imgs), labels_real)

            # Фейковые изображения (генератор не обновляется)
            z         = torch.randn(batch_n, LATENT_DIM, 1, 1, device=device)
            fake_imgs = G(z)
            labels_fake = torch.full((batch_n,), fake_label, device=device)
            loss_d_fake = criterion(D(fake_imgs.detach()), labels_fake)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_D.step()

            # ── Шаг генератора ───────────────────────────────
            G.zero_grad()

            # Генератор хочет обмануть D — цель «реальный»
            labels_g = torch.full((batch_n,), real_label, device=device)
            loss_g   = criterion(D(fake_imgs), labels_g)
            loss_g.backward()
            opt_G.step()

            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item()
            n_batches    += 1

        loss_g_last = epoch_loss_g / max(n_batches, 1)
        loss_d_last = epoch_loss_d / max(n_batches, 1)

        # Каждые 10 эпох (и на первой) — сохраняем сэмплы и пишем в лог
        if epoch % 10 == 0 or epoch == 1:
            sample_path = project.gan_samples_dir / f"epoch_{epoch:04d}.jpg"
            G.eval()
            with torch.no_grad():
                grid    = vutils.make_grid(
                    G(fixed_noise), nrow=4, normalize=True, value_range=(-1, 1))
                img_np  = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", img_bgr)
                if ok:
                    sample_path.write_bytes(buf.tobytes())
            G.train()

            logger.info(
                f"Эпоха {epoch}/{epochs} | "
                f"loss_G={loss_g_last:.4f} | loss_D={loss_d_last:.4f}"
            )
            print(
                f"  Эпоха {epoch:4d}/{epochs} | "
                f"loss_G={loss_g_last:.4f} | loss_D={loss_d_last:.4f} | "
                f"сэмпл → {sample_path.name}"
            )

        last_epoch = epoch

        # Уведомляем внешний наблюдатель после каждой эпохи.
        # Если колбэк вернул True — запрошена досрочная остановка; выходим из цикла,
        # но веса всё равно сохраняем ниже — прогресс не теряется.
        if on_epoch and on_epoch(epoch, epochs, loss_g_last, loss_d_last):
            logger.info(f"Обучение GAN остановлено по запросу на эпохе {epoch}/{epochs}")
            print(f"Обучение остановлено на эпохе {epoch}/{epochs}.")
            break

    # Сохраняем веса обеих моделей (в том числе при досрочной остановке)
    torch.save(G.state_dict(), project.gan_model_dir / "generator.pth")
    torch.save(D.state_dict(), project.gan_model_dir / "discriminator.pth")
    logger.info(f"Веса сохранены в {project.gan_model_dir}")
    print(f"Обучение завершено. Веса → {project.gan_model_dir}")

    project.update_stats({
        "gan_epochs":     last_epoch,
        "gan_frames":     len(all_images),
        "gan_loss_g":     round(loss_g_last, 4),
        "gan_loss_d":     round(loss_d_last, 4),
        "gan_image_size": image_size,
    })

    return {
        "epochs":       last_epoch,   # фактическое число эпох (< epochs при стопе)
        "final_loss_g": round(loss_g_last, 4),
        "final_loss_d": round(loss_d_last, 4),
    }


# ---------------------------------------------------------------------------
# Публичная функция: генерация изображений
# ---------------------------------------------------------------------------

def generate_images(
    project: Project,
    count: int = 200,
) -> dict:
    """Генерирует синтетические кадры с помощью обученного генератора.

    Загружает generator.pth из project.gan_model_dir, генерирует count
    изображений батчами по 16, денормализует, масштабирует до 640×640
    (INTER_LANCZOS4) и сохраняет в project.frames_real_dir как gan_NNNNNN.jpg.

    Исходные (реальные) кадры могут храниться во внешнем пути через
    data_sources["frames"]["real"], но сгенерированные кадры всегда
    записываются в project.frames_real_dir — внутрь папки проекта.

    Args:
        project: объект Project — определяет пути к модели и кадрам.
        count:   количество изображений для генерации (по умолчанию 200).

    Returns:
        {"generated": count}

    Raises:
        FileNotFoundError: если generator.pth не найден в gan_model_dir.
    """
    get_logger(__name__, project.logs_dir)

    model_path = project.gan_model_dir / "generator.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Модель генератора не найдена: {model_path}\n"
            f"Сначала обучите модель: "
            f"python main.py --project '{project.name}' --train-gan"
        )

    # Читаем разрешение из статистики проекта — оно было сохранено при обучении
    meta       = project._read_meta()
    image_size = meta.get("stats", {}).get("gan_image_size", config.GAN_IMAGE_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"generate_images | проект={project.name} | "
        f"count={count} | image_size={image_size} | device={device}"
    )

    G = Generator(LATENT_DIM, image_size).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    G.eval()

    # Генерированные кадры всегда сохраняются внутри проекта,
    # независимо от того, где хранятся исходные (data_sources["frames"]["real"])
    output_dir = project.frames_real_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Определяем стартовый номер, чтобы не перезаписывать существующие gan-кадры
    existing_nums: set[int] = set()
    for f in output_dir.glob("gan_*.jpg"):
        try:
            existing_nums.add(int(f.stem[4:]))
        except ValueError:
            pass
    next_num = max(existing_nums, default=-1) + 1

    TARGET_SIZE  = 640   # целевое разрешение выходных кадров
    BATCH_SIZE   = 16
    n_batches    = math.ceil(count / BATCH_SIZE)
    generated    = 0

    print(f"Генерация {count} изображений ({device})...")
    logger.info(f"Генерация {count} кадров, нумерация с gan_{next_num:06d}")

    with torch.no_grad():
        for _ in range(n_batches):
            cur_batch = min(BATCH_SIZE, count - generated)
            z    = torch.randn(cur_batch, LATENT_DIM, 1, 1, device=device)
            imgs = G(z)

            for img_t in imgs:
                # Денормализуем из [-1, 1] → [0, 255]
                img_np = ((img_t.permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255.0)
                img_np = img_np.clip(0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # Масштабируем до целевого разрешения через Lanczos (высокое качество)
                img_bgr = cv2.resize(
                    img_bgr, (TARGET_SIZE, TARGET_SIZE),
                    interpolation=cv2.INTER_LANCZOS4,
                )

                out_path = output_dir / f"gan_{next_num:06d}.jpg"
                ok, buf = cv2.imencode(".jpg", img_bgr)
                if ok:
                    out_path.write_bytes(buf.tobytes())
                next_num  += 1
                generated += 1

    logger.info(f"Генерация завершена: {generated} кадров → {output_dir}")
    print(f"Создано {generated} кадров → {output_dir}")

    project.update_stats({"gan_generated": generated})
    return {"generated": generated}
