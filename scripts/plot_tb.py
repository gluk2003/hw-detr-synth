#!/usr/bin/env python3
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("pip install tensorboard")

TB_DIR = "runs/detr_coco10/tb"
OUT_DIR = "reports"

ea = EventAccumulator(TB_DIR)
ea.Reload()
tags = ea.Tags().get("scalars", [])
print("Available tags:", tags)

def get_scalar(tag):
    if tag not in tags:
        return [], []
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]

# --- Loss curves ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Training Loss Curves — DETR COCO-10", fontsize=14)
loss_tags = [
    ("train/total_loss", "Total Loss"),
    ("train/loss_ce",    "CE Loss"),
    ("train/loss_bbox",  "BBox L1 Loss"),
    ("train/loss_giou",  "GIoU Loss"),
]
for ax, (tag, label) in zip(axes.flat, loss_tags):
    steps, vals = get_scalar(tag)
    if steps:
        ax.plot(steps, vals, linewidth=1.5)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    else:
        ax.text(0.5, 0.5, "tag not found", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_title(label); ax.grid(True, alpha=0.3)
plt.tight_layout()
loss_path = os.path.join(OUT_DIR, "loss_curves.png")
plt.savefig(loss_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved {loss_path}")

# --- mAP curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Validation mAP — DETR COCO-10", fontsize=14)
map_tags = [
    ("val/map",    "mAP @[.5:.95]"),
    ("val/map_50", "mAP@50"),
]
for ax, (tag, label) in zip(axes, map_tags):
    steps, vals = get_scalar(tag)
    if steps:
        ax.plot(steps, vals, marker="o", linewidth=1.5, markersize=4)
        ax.set_xlabel("Epoch"); ax.set_ylabel("mAP")
    else:
        ax.text(0.5, 0.5, "tag not found", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_title(label); ax.grid(True, alpha=0.3)
plt.tight_layout()
map_path = os.path.join(OUT_DIR, "map_curves.png")
plt.savefig(map_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved {map_path}")
