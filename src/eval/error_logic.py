"""Чистая логика классификации ошибок детектора. Никаких torch / HF —
работает на numpy. Это позволяет тестировать без GPU/heavy-deps.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """IoU между двумя наборами боксов формата (x1,y1,x2,y2)."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    a = boxes1[:, None, :]
    b = boxes2[None, :, :]
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(
        inter_y2 - inter_y1, 0, None
    )
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter + 1e-9
    return inter / union


def classify_errors(
    preds: dict[str, np.ndarray],     # {boxes:[N,4] xyxy, scores:[N], labels:[N]}
    gts: dict[str, np.ndarray],       # {boxes:[M,4] xyxy, labels:[M]}
    score_thr: float = 0.3,
    iou_match: float = 0.5,
    iou_min: float = 0.1,
) -> dict[str, list[int]]:
    """Категории ошибок (TIDE-style).

    Возвращает индексы preds в каждой из категорий + индексы missed GT.

    Категории:
      TP    : IoU >= iou_match ∧ верный класс ∧ GT ещё свободен
      CLS   : IoU >= iou_match ∧ неверный класс
      LOC   : верный класс ∧ iou_min <= IoU < iou_match
      BOTH  : неверный класс ∧ iou_min <= IoU < iou_match
      DUPE  : верный класс ∧ IoU >= iou_match ∧ GT уже занят
      BKG   : IoU < iou_min со всеми
      MISS  : GT без TP-матча (false negative)
    """
    keep = preds["scores"] >= score_thr
    p_boxes = preds["boxes"][keep]
    p_labels = preds["labels"][keep]
    p_scores = preds["scores"][keep]

    g_boxes = gts["boxes"]
    g_labels = gts["labels"]

    if len(p_boxes) == 0:
        return {
            "tp": [], "cls": [], "loc": [], "both": [], "dupe": [], "bkg": [],
            "missed_gt": list(range(len(g_boxes))),
        }

    ious = box_iou(p_boxes, g_boxes)  # [P, G]

    tp, cls_err, loc_err, both_err, dupe, bkg = [], [], [], [], [], []
    matched_gt: set[int] = set()

    # обрабатываем pred-ы в порядке убывания score
    order = np.argsort(-p_scores)
    for i in order:
        if len(g_boxes) == 0:
            bkg.append(int(i))
            continue
        best_gt = int(np.argmax(ious[i]))
        best_iou = float(ious[i, best_gt])
        cls_match = int(p_labels[i]) == int(g_labels[best_gt])

        if best_iou >= iou_match and cls_match:
            if best_gt in matched_gt:
                dupe.append(int(i))
            else:
                tp.append(int(i))
                matched_gt.add(best_gt)
        elif best_iou >= iou_match and not cls_match:
            cls_err.append(int(i))
        elif iou_min <= best_iou < iou_match and cls_match:
            loc_err.append(int(i))
        elif iou_min <= best_iou < iou_match and not cls_match:
            both_err.append(int(i))
        else:
            bkg.append(int(i))

    missed_gt = [i for i in range(len(g_boxes)) if i not in matched_gt]
    return {
        "tp": tp, "cls": cls_err, "loc": loc_err, "both": both_err,
        "dupe": dupe, "bkg": bkg, "missed_gt": missed_gt,
    }
