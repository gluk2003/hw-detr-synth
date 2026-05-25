"""Фиксация всех источников рандома для воспроизводимости."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Фиксирует seed для random/numpy/torch + cudnn-детерминизм.

    Note: cudnn.deterministic=True замедляет обучение на ~10–20%, но
    даёт точную воспроизводимость mAP между запусками.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Передаём seed в каждый DataLoader-worker."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
