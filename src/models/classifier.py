"""Лёгкий классификатор для HW2.5 (baseline vs +synth).

Используем timm ViT-Tiny — он хорошо себя ведёт даже на маленьких датасетах
с предобучением на ImageNet, и быстро дообучается.
"""
from __future__ import annotations

import timm
import torch.nn as nn


def build_classifier(
    model_name: str = "vit_tiny_patch16_224",
    num_classes: int = 4,
    pretrained: bool = True,
    drop_rate: float = 0.1,
) -> nn.Module:
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model
