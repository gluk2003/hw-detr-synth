"""Обёртка над HuggingFace Deformable-DETR.

* num_labels — переопределяем на 10;
* классификационная голова инициализируется заново
  (HF делает это автоматически, когда меняется num_labels);
* `freeze_backbone()` — для прогрева головы первые N эпох.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
)


def build_model_and_processor(
    model_name: str,
    num_classes: int,
    num_queries: int = 100,
    id2label: dict[int, str] | None = None,
) -> tuple[DeformableDetrForObjectDetection, AutoImageProcessor]:
    """Грузит pretrained Deformable-DETR и переинициализирует cls-голову."""

    label2id = {v: k for k, v in (id2label or {}).items()}

    config = DeformableDetrConfig.from_pretrained(
        model_name,
        num_labels=num_classes,
        num_queries=num_queries,
        id2label=id2label or {i: str(i) for i in range(num_classes)},
        label2id=label2id or {str(i): i for i in range(num_classes)},
    )

    # ignore_mismatched_sizes — чтобы голова на 91 класс не блокировала загрузку
    model = DeformableDetrForObjectDetection.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )

    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor


def freeze_backbone(model: DeformableDetrForObjectDetection) -> None:
    """Замораживает ResNet-backbone (для прогрева head)."""
    for name, p in model.named_parameters():
        if "backbone" in name:
            p.requires_grad = False


def unfreeze_backbone(model: DeformableDetrForObjectDetection) -> None:
    for p in model.parameters():
        p.requires_grad = True


def split_params_by_lr(
    model: nn.Module, lr_backbone: float, lr_head: float
) -> list[dict]:
    """Возвращает param_groups для AdamW с раздельным lr backbone/head.

    Это стандартная практика DETR: backbone уже хорошо обучен,
    хочется лр поменьше.
    """
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            other_params.append(p)
    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": other_params, "lr": lr_head},
    ]
