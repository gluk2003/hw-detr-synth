"""Один проход обучения и валидация."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.eval.metrics import CocoMapEvaluator


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "pixel_mask": batch["pixel_mask"].to(device, non_blocking=True),
        "labels": [
            {k: v.to(device, non_blocking=True) for k, v in lbl.items()}
            for lbl in batch["labels"]
        ],
    }
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    scaler: GradScaler | None,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    log_every: int = 20,
    grad_clip: float = 0.1,
    profiler: Any = None,
    global_step_start: int = 0,
) -> tuple[dict[str, float], int]:
    """Один эпохов тренировки. Возвращает усреднённые loss-ы и обновлённый
    global_step."""
    model.train()

    loss_meters: dict[str, list[float]] = {}
    global_step = global_step_start

    pbar = tqdm(loader, desc=f"epoch {epoch} [train]", leave=False)
    for it, batch in enumerate(pbar):
        batch = _to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=scaler is not None):
            outputs = model(**batch)
            loss = outputs.loss
            loss_dict = outputs.loss_dict  # отдельные компоненты

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if profiler is not None:
            profiler.step()

        # копим loss-ы
        for k, v in loss_dict.items():
            loss_meters.setdefault(k, []).append(v.detach().item())
        loss_meters.setdefault("total_loss", []).append(loss.item())

        # лог в TB
        if global_step % log_every == 0:
            for k, vals in loss_meters.items():
                writer.add_scalar(f"train/{k}", vals[-1], global_step)
            writer.add_scalar(
                "train/lr_backbone", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar(
                "train/lr_head", optimizer.param_groups[1]["lr"], global_step
            )
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        global_step += 1

    avg = {k: float(sum(v) / len(v)) for k, v in loss_meters.items()}
    return avg, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    processor,
    device: torch.device,
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Считает mAP/mAP@50 через pycocotools (см. src.eval.metrics)."""
    model.eval()
    evaluator = CocoMapEvaluator(
        ann_file=loader.dataset.coco.coco.dataset,  # COCO dict (in-memory)
    )

    for batch in tqdm(loader, desc="val", leave=False):
        batch_d = _to_device(batch, device)
        outputs = model(
            pixel_values=batch_d["pixel_values"],
            pixel_mask=batch_d["pixel_mask"],
        )

        # processor.post_process_object_detection делает softmax + sigmoid выбор + bbox decode
        target_sizes = torch.stack(
            [lbl["orig_size"] for lbl in batch_d["labels"]]
        )
        results = processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target_sizes
        )

        image_ids = [int(lbl["image_id"].item()) for lbl in batch_d["labels"]]
        evaluator.update(results, image_ids)

    metrics = evaluator.summarize()
    return metrics
