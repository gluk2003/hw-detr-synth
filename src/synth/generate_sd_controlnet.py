"""Генерация синтетических изображений редких классов
через Stable Diffusion + ControlNet (Canny).

Логика:
* для классов с чёткой геометрической формой (`stop_sign`, `traffic_light`)
  Canny-условие генерируется процедурно: рисуем октогон/прямоугольник
  на случайном месте холста — это «принуждает» SD ставить объект ровно
  туда, где нужно, что даёт нам бесплатные bbox-аннотации;
* для остальных (`motorcycle`) — берём Canny-edges с реальных тренировочных
  кропов: вариативность фона/освещения растёт, форма ground-truth сохраняется.

Каждой картинке сопутствует JSON с bbox-ом и классом —
готово к подмесу в датасет.

Запуск:
    python -m src.synth.generate_sd_controlnet --config configs/synth.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image
from tqdm import tqdm

from src.synth.prompts import NEGATIVE_PROMPT, PROMPTS
from src.utils.misc import ensure_dir, load_config, setup_logger
from src.utils.seed import set_seed


# ----------------------------- Canny priors ---------------------------------


def _octagon_polygon(cx: int, cy: int, r: int) -> list[tuple[int, int]]:
    pts = []
    for k in range(8):
        a = math.pi / 8 + k * math.pi / 4
        pts.append((int(cx + r * math.cos(a)), int(cy + r * math.sin(a))))
    return pts


def draw_stop_sign_canny(size: int = 512) -> tuple[np.ndarray, list[float]]:
    """Возвращает (canny[H,W,3], bbox=[x,y,w,h])."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    r = random.randint(int(size * 0.12), int(size * 0.22))
    cx = random.randint(r + 10, size - r - 10)
    cy = random.randint(r + 10, size - r - 10)
    pts = _octagon_polygon(cx, cy, r)
    cv2.polylines(canvas, [np.array(pts)], isClosed=True, color=(255, 255, 255), thickness=3)
    # надпись STOP — горизонтальная линия посередине
    cv2.line(canvas,
             (cx - int(r * 0.6), cy),
             (cx + int(r * 0.6), cy),
             (255, 255, 255), 2)
    bbox = [cx - r, cy - r, 2 * r, 2 * r]
    return canvas, bbox


def draw_traffic_light_canny(size: int = 512) -> tuple[np.ndarray, list[float]]:
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    w = random.randint(int(size * 0.06), int(size * 0.1))
    h = w * 3   # вертикальная стойка с 3 кругами
    x = random.randint(20, size - w - 20)
    y = random.randint(20, size - h - 20)
    # корпус
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), 2)
    # три круга
    cr = int(w * 0.35)
    for k in range(3):
        cy = y + int(w * 0.5) + k * w
        cx = x + w // 2
        cv2.circle(canvas, (cx, cy), cr, (255, 255, 255), 2)
    bbox = [x, y, w, h]
    return canvas, bbox


PRIOR_DRAWERS = {
    "stop_sign": draw_stop_sign_canny,
    "traffic_light": draw_traffic_light_canny,
}


def canny_from_image(img: Image.Image, low: int = 100, high: int = 200) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return np.stack([edges, edges, edges], axis=-1)


# ----------------------------- генерация ------------------------------------


def build_pipeline(cfg: dict) -> StableDiffusionControlNetPipeline:
    dtype = torch.float16 if cfg["generation"]["dtype"] == "fp16" else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        cfg["generation"]["controlnet"], torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg["generation"]["sd_model"],
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,        # отключаем NSFW-фильтр для воспроизводимости
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(cfg["generation"]["device"])
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe


def generate_for_class(
    pipe: StableDiffusionControlNetPipeline,
    cls_name: str,
    n: int,
    out_dir: Path,
    cfg: dict,
    real_canny_sources: list[Path] | None = None,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = PROMPTS[cls_name]
    size = cfg["generation"]["image_size"]
    records: list[dict] = []

    for i in tqdm(range(n), desc=f"gen {cls_name}"):
        # 1) Canny-условие
        bbox = None
        if cls_name in PRIOR_DRAWERS:
            cond_np, bbox = PRIOR_DRAWERS[cls_name](size)
            cond = Image.fromarray(cond_np)
        else:
            assert real_canny_sources, "Для класса без prior нужны реальные кропы"
            src = random.choice(real_canny_sources)
            ref = Image.open(src).convert("RGB").resize((size, size))
            cond = Image.fromarray(canny_from_image(ref))

        prompt = random.choice(prompts)
        seed = random.randint(0, 2**31 - 1)
        gen = torch.Generator(device=cfg["generation"]["device"]).manual_seed(seed)

        with torch.autocast(cfg["generation"]["device"]):
            out = pipe(
                prompt=prompt,
                image=cond,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=cfg["generation"]["steps"],
                guidance_scale=cfg["generation"]["guidance_scale"],
                controlnet_conditioning_scale=cfg["generation"][
                    "controlnet_conditioning_scale"
                ],
                generator=gen,
            )
        img = out.images[0]
        fname = f"{cls_name}_{i:05d}.png"
        img.save(out_dir / fname)

        rec = {
            "file_name": fname,
            "class": cls_name,
            "prompt": prompt,
            "seed": seed,
            "bbox": bbox,        # если есть процедурный
        }
        records.append(rec)

    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--classes", default=None,
                   help="Список классов через запятую (override cfg.rare_classes)")
    p.add_argument("--n_per_class", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = setup_logger("synth_gen")

    out_root = ensure_dir(cfg["generation"]["out_dir"])
    classes = (
        args.classes.split(",") if args.classes
        else cfg["rare_classes"]
    )
    n_per = args.n_per_class or cfg["generation"]["num_per_class"]

    logger.info(f"Будем генерировать: {classes}, по {n_per} картинок")
    pipe = build_pipeline(cfg)

    all_records: list[dict] = []
    for cls_name in classes:
        # для motorcycle/без prior'а — нужен список реальных кропов
        real_sources = None
        if cls_name not in PRIOR_DRAWERS:
            real_dir = Path("data/crops") / cls_name
            if real_dir.exists():
                real_sources = sorted(real_dir.glob("*.jpg"))
            if not real_sources:
                logger.warning(
                    f"Нет реальных кропов для {cls_name} в data/crops/{cls_name}/ — "
                    "генерация пропущена."
                )
                continue

        cls_out = out_root / cls_name
        recs = generate_for_class(
            pipe, cls_name, n_per, cls_out, cfg,
            real_canny_sources=real_sources,
        )
        all_records.extend(recs)

    with open(out_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    logger.info(f"Готово: {len(all_records)} картинок. Metadata: {out_root}/metadata.json")


if __name__ == "__main__":
    main()
