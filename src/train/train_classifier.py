"""Тренировка классификатора для HW2.5: baseline (real) vs +synth.

Тренируем ViT-Tiny различать 3 редких класса COCO-10 + background.
Источник «настоящих» данных — кропы по bbox-ам из train-сабсета.
Источник синтетики — папки `data/synth/<class>/`.

`synth_ratio` управляет долей синтетики (0.0 = baseline, 1.0 = +synth максимально).
Каждый эксперимент пишется в свою TB-папку, итог — таблица в reports/.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from src.models.classifier import build_classifier
from src.utils.misc import ensure_dir, load_config, setup_logger
from src.utils.seed import set_seed, worker_init_fn


# --------------------------- датасет ----------------------------------------

CLASS_TO_IDX = {
    "background": 0,
    "stop_sign": 1,
    "traffic_light": 2,
    "motorcycle": 3,
}


class CropsDataset(Dataset):
    """Картинка + метка из плоской папочной структуры.

    Принимает список (Path, label_idx) — позволяет смешивать real и synth.
    """

    def __init__(self, items: list[tuple[Path, int]], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), y


def collect_items(
    real_dir: Path,
    synth_dir: Path | None,
    synth_ratio: float,
    classes: list[str],
) -> list[tuple[Path, int]]:
    """Собирает список (file, label).

    `synth_ratio` — сколько синтетики на 1 единицу real в этом классе:
      0.0 -> 0 synth, 1.0 -> столько же synth, сколько real.
    """
    items: list[tuple[Path, int]] = []
    for cls_name in classes:
        idx = CLASS_TO_IDX[cls_name]
        real_files = sorted((real_dir / cls_name).glob("*.*"))
        items.extend([(p, idx) for p in real_files])
        if synth_dir and synth_ratio > 0:
            synth_files = sorted((synth_dir / cls_name).glob("*.png"))
            n_use = int(len(real_files) * synth_ratio)
            n_use = min(n_use, len(synth_files))
            items.extend([(p, idx) for p in synth_files[:n_use]])
    return items


def build_transforms(is_train: bool):
    if is_train:
        return T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def make_balanced_sampler(items: list[tuple[Path, int]]) -> WeightedRandomSampler:
    counts = Counter(y for _, y in items)
    weights = [1.0 / counts[y] for _, y in items]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# --------------------------- тренировка -------------------------------------


def run_one_experiment(
    cfg: dict,
    exp_cfg: dict,
    classes: list[str],
    real_train_dir: Path,
    real_val_dir: Path,
    synth_dir: Path,
) -> dict:
    """Один прогон baseline | synth_xx. Возвращает финальные метрики."""
    set_seed(cfg["seed"])

    name = exp_cfg["name"]
    out_dir = ensure_dir(Path("runs/synth_ablation") / name)
    writer = SummaryWriter(out_dir / "tb")
    logger = setup_logger(f"cls_{name}", log_file=out_dir / "train.log")

    train_items = collect_items(
        real_train_dir, synth_dir, exp_cfg["synth_ratio"], classes
    )
    val_items = collect_items(real_val_dir, None, 0.0, classes)
    logger.info(f"[{name}] train={len(train_items)} val={len(val_items)}")
    logger.info(
        f"  per-class train: {Counter(y for _, y in train_items)}"
    )

    train_ds = CropsDataset(train_items, build_transforms(True))
    val_ds = CropsDataset(val_items, build_transforms(False))

    sampler = make_balanced_sampler(train_items)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["classifier"]["batch_size"],
        sampler=sampler,
        num_workers=cfg["classifier"]["num_workers"],
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["classifier"]["batch_size"],
        shuffle=False,
        num_workers=cfg["classifier"]["num_workers"],
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(
        cfg["classifier"]["model"],
        num_classes=cfg["classifier"]["num_classes"],
        pretrained=cfg["classifier"]["pretrained"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["classifier"]["lr"],
        weight_decay=cfg["classifier"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["classifier"]["epochs"])
    scaler = GradScaler(enabled=cfg["classifier"]["mixed_precision"] == "fp16")
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_per_class = {}
    global_step = 0

    for epoch in range(cfg["classifier"]["epochs"]):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"[{name}] ep{epoch} tr", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            if global_step % cfg["logging"]["log_every_n_steps"] == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

        scheduler.step()
        writer.add_scalar("train_epoch/loss", float(np.mean(losses)), epoch)

        # eval
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                p = logits.argmax(dim=1).cpu().numpy()
                all_p.append(p)
                all_y.append(y.numpy())
        all_p = np.concatenate(all_p)
        all_y = np.concatenate(all_y)
        acc = float((all_p == all_y).mean())
        writer.add_scalar("val/acc", acc, epoch)

        # per-class recall
        per_class = {}
        for c_idx, c_name in {v: k for k, v in CLASS_TO_IDX.items()}.items():
            mask = all_y == c_idx
            if mask.sum() == 0:
                continue
            per_class[c_name] = float((all_p[mask] == c_idx).mean())
            writer.add_scalar(f"val/recall_{c_name}", per_class[c_name], epoch)

        logger.info(
            f"[{name}] ep{epoch} | loss {np.mean(losses):.4f} | "
            f"acc {acc:.4f} | per-cls {per_class}"
        )

        if acc > best_acc:
            best_acc = acc
            best_per_class = per_class
            torch.save(
                {"model": model.state_dict(), "acc": acc, "per_class": per_class},
                out_dir / "best.pt",
            )

    writer.close()
    return {"name": name, "best_acc": best_acc, "per_class": best_per_class,
            "synth_ratio": exp_cfg["synth_ratio"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/synth.yaml")
    p.add_argument("--real_train", default="data/crops/train")
    p.add_argument("--real_val", default="data/crops/val")
    p.add_argument("--synth_dir", default="data/synth")
    args = p.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("synth_ablation")

    classes = ["background"] + cfg["rare_classes"]
    results = []
    for exp_cfg in cfg["experiments"]:
        res = run_one_experiment(
            cfg, exp_cfg, classes,
            Path(args.real_train),
            Path(args.real_val),
            Path(args.synth_dir),
        )
        results.append(res)
        logger.info(f"DONE {res['name']}: acc={res['best_acc']:.4f}")

    out_path = Path("reports/synth_ablation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results dumped: {out_path}")

    # ASCII-табличка
    print("\n=== ABLATION ===")
    print(f"{'experiment':<20} {'synth_ratio':<12} {'acc':<8} per-class")
    for r in results:
        per_class_str = ", ".join(f"{k}={v:.3f}" for k, v in r["per_class"].items())
        print(f"{r['name']:<20} {r['synth_ratio']:<12} {r['best_acc']:<8.4f} {per_class_str}")


if __name__ == "__main__":
    main()
