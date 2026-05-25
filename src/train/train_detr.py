"""Главный train-скрипт для Deformable-DETR на COCO-10.

Запуск:
    python -m src.train.train_detr --config configs/detr.yaml

Что делает:
* набирает данные через CocoDetectionAlb + HF image-processor;
* инициализирует модель (HF, 10 классов);
* запускает train-loop с TB-логированием:
  - loss-компоненты (cls, bbox L1, GIoU) per step и per epoch,
  - lr, grad norm,
  - mAP/mAP@50 каждые eval_every эпох;
* запускает torch.profiler на эпохах из cfg.profiler.epochs;
* сохраняет ckpt каждые save_every и при улучшении mAP@50.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from src.data.datamodule import build_dataloaders
from src.models.detr_wrapper import (
    build_model_and_processor,
    freeze_backbone,
    split_params_by_lr,
    unfreeze_backbone,
)
from src.train.engine import evaluate, train_one_epoch
from src.utils.misc import count_parameters, ensure_dir, load_config, setup_logger
from src.utils.seed import set_seed, worker_init_fn


def build_optimizer_and_scheduler(model, cfg):
    param_groups = split_params_by_lr(
        model, cfg["train"]["lr_backbone"], cfg["train"]["lr_head"]
    )
    optimizer = AdamW(param_groups, weight_decay=cfg["train"]["weight_decay"])
    scheduler = StepLR(optimizer, step_size=cfg["train"]["lr_drop_epoch"], gamma=0.1)
    return optimizer, scheduler


def build_profiler(cfg, epoch: int):
    """Возвращает контекстный профилировщик, если эпоха в списке."""
    if not cfg["profiler"]["enable"] or epoch not in cfg["profiler"]["epochs"]:
        return None

    out_dir = ensure_dir(Path(cfg["profiler"]["out_dir"]) / f"epoch_{epoch}")

    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=cfg["profiler"]["wait"],
            warmup=cfg["profiler"]["warmup"],
            active=cfg["profiler"]["active"],
            repeat=cfg["profiler"]["repeat"],
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out_dir)),
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None, help="Путь к checkpoint")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    exp_dir = ensure_dir(f"runs/{cfg['experiment_name']}")
    logger = setup_logger("train_detr", log_file=exp_dir / "train.log")
    logger.info(f"Config: {json.dumps(cfg, ensure_ascii=False, indent=2)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- model + processor ---
    id2label = {i: name for i, name in enumerate(cfg["data"]["class_names"])}
    model, processor = build_model_and_processor(
        cfg["model"]["name"],
        num_classes=cfg["data"]["num_classes"],
        num_queries=cfg["model"]["num_queries"],
        id2label=id2label,
    )
    model.to(device)
    total, train_p = count_parameters(model)
    logger.info(f"Параметров: total={total:,}  trainable={train_p:,}")

    # --- data ---
    train_loader, val_loader = build_dataloaders(
        cfg, processor, worker_init_fn=worker_init_fn
    )
    logger.info(f"Train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # --- optim ---
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    scaler = GradScaler(enabled=cfg["train"]["mixed_precision"] == "fp16")

    writer = SummaryWriter(cfg["logging"]["tb_dir"])
    ckpt_dir = ensure_dir(cfg["ckpt"]["dir"])
    start_epoch = 0
    best_metric = -1.0
    global_step = 0

    # --- resume ---
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_metric = ck["metrics"].get("map_50", -1.0)
        logger.info(f"Resumed from {args.resume} (epoch {start_epoch})")

    # --- backbone freeze для warmup ---
    if cfg["model"]["freeze_backbone_epochs"] > 0:
        freeze_backbone(model)
        logger.info(
            f"Backbone заморожен на первые {cfg['model']['freeze_backbone_epochs']} эпох"
        )

    # === train loop ===
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        if epoch == cfg["model"]["freeze_backbone_epochs"]:
            unfreeze_backbone(model)
            logger.info(f"Эпоха {epoch}: backbone разморожен.")

        profiler = build_profiler(cfg, epoch)
        if profiler is not None:
            profiler.__enter__()
            logger.info(f"Профайлер активен на эпохе {epoch}")

        try:
            train_loss, global_step = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=None,           # step-уровневый scheduler не нужен (StepLR — per-epoch)
                scaler=scaler if cfg["train"]["mixed_precision"] == "fp16" else None,
                device=device,
                epoch=epoch,
                writer=writer,
                log_every=cfg["logging"]["log_every_n_steps"],
                grad_clip=cfg["train"]["grad_clip"],
                profiler=profiler,
                global_step_start=global_step,
            )
        finally:
            if profiler is not None:
                profiler.__exit__(None, None, None)

        scheduler.step()

        # эпоховые лог-средние
        for k, v in train_loss.items():
            writer.add_scalar(f"train_epoch/{k}", v, epoch)
        logger.info(
            f"Epoch {epoch} | loss {train_loss['total_loss']:.3f} | "
            + " ".join(
                f"{k}={v:.3f}"
                for k, v in train_loss.items()
                if k != "total_loss"
            )
        )

        # --- eval ---
        if (epoch + 1) % cfg["eval"]["eval_every"] == 0 or epoch == cfg["train"]["epochs"] - 1:
            metrics = evaluate(
                model, val_loader, processor, device,
                iou_thresholds=cfg["eval"]["iou_thresholds"],
            )
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"val/{k}", v, epoch)
            logger.info(
                f"Epoch {epoch} | mAP {metrics['map']:.4f} | "
                f"mAP@50 {metrics['map_50']:.4f} | mAP@75 {metrics['map_75']:.4f}"
            )

            # save best
            if metrics["map_50"] > best_metric:
                best_metric = metrics["map_50"]
                save_checkpoint(
                    model, optimizer, scheduler, epoch, metrics,
                    Path(ckpt_dir) / "best.pt",
                )
                logger.info(f"NEW BEST mAP@50 = {best_metric:.4f}; ckpt сохранён")

        # save periodic
        if (epoch + 1) % cfg["ckpt"]["save_every"] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_loss,
                Path(ckpt_dir) / f"epoch_{epoch:03d}.pt",
            )

    writer.close()
    logger.info("Training done.")


if __name__ == "__main__":
    main()
