"""Отрисовка предсказанных и истинных боксов на картинке."""
from __future__ import annotations

import colorsys

from PIL import Image, ImageDraw, ImageFont


def _palette(n: int) -> list[tuple[int, int, int]]:
    out = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def _try_font(size: int = 14):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_boxes(
    img: Image.Image,
    pred_boxes=None,
    pred_labels=None,
    pred_scores=None,
    gt_boxes=None,
    gt_labels=None,
    class_names: list[str] | None = None,
) -> Image.Image:
    """Рисует GT (зелёные) + Pred (по цвету класса).

    pred_boxes / gt_boxes — np-массив (N, 4) в формате xyxy в пикселях.
    """
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = _try_font(16)
    palette = _palette(len(class_names)) if class_names else _palette(91)

    # GT
    if gt_boxes is not None and len(gt_boxes) > 0:
        for b, l in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = b
            draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=3)
            name = class_names[int(l)] if class_names else str(int(l))
            draw.text((x1 + 3, y1 + 3), f"GT:{name}", fill=(0, 200, 0), font=font)

    # PRED
    if pred_boxes is not None and len(pred_boxes) > 0:
        if pred_scores is None:
            pred_scores = [None] * len(pred_boxes)
        for b, l, s in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = b
            color = palette[int(l) % len(palette)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            name = class_names[int(l)] if class_names else str(int(l))
            label = f"{name}" + (f" {s:.2f}" if s is not None else "")
            draw.text((x1 + 3, y2 - 18), label, fill=color, font=font)

    return img
