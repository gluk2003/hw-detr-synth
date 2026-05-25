"""Анализ ошибок DETR в духе TIDE (Bolya et al., ECCV-2020).

Категории ошибок (для каждого pred с уверенностью > thr):
* CLS — IoU(pred, GT) >= 0.5, но pred.cls != GT.cls
* LOC — pred.cls == GT.cls, 0.1 <= IoU(pred, лучший GT этого же класса) < 0.5
* BOTH — pred.cls != GT.cls и 0.1 <= IoU < 0.5
* DUPE — pred.cls == GT.cls, IoU >= 0.5, но этот GT уже отдан другому pred
* BKG — pred ни с одним GT не имеет IoU >= 0.1 (false positive)
* MISS — GT без матча с pred (false negative; считается отдельно по GT)

На выходе:
* bar-plot с распределением ошибок;
* таблица per-class;
* dump топ-N картинок с наиболее «плохими» ошибками и их визуализацией.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datamodule import CocoDetectionAlb, collate_fn
from src.eval.error_logic import box_iou, classify_errors  # noqa: F401
from src.eval.visualize import draw_boxes
from src.models.detr_wrapper import build_model_and_processor
from src.utils.misc import ensure_dir, load_config, setup_logger


def aggregate_and_plot(stats: list[dict], out_dir: Path, class_names: list[str]) -> dict:
    totals = Counter()
    per_class_cls = Counter()
    per_class_loc = Counter()
    per_class_miss = Counter()

    for s in stats:
        for k in ("tp", "cls", "loc", "both", "dupe", "bkg"):
            totals[k] += len(s["per_image"][k])
        for cls_id, n in s["per_class_cls"].items():
            per_class_cls[cls_id] += n
        for cls_id, n in s["per_class_loc"].items():
            per_class_loc[cls_id] += n
        for cls_id, n in s["per_class_miss"].items():
            per_class_miss[cls_id] += n

    # bar plot
    labels = ["TP", "Cls", "Loc", "Both", "Dupe", "Bkg"]
    values = [totals["tp"], totals["cls"], totals["loc"],
              totals["both"], totals["dupe"], totals["bkg"]]
    colors = ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#3498db", "#7f8c8d"]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=colors)
    for i, v in enumerate(values):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.title("Распределение ошибок DETR (TIDE-style)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "error_types.png", dpi=120)
    plt.close()

    # per-class
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, counter, title in zip(
        axes,
        [per_class_cls, per_class_loc, per_class_miss],
        ["Cls errors", "Loc errors", "Missed GT"],
    ):
        x = [class_names[i] for i in range(len(class_names))]
        y = [counter.get(i, 0) for i in range(len(class_names))]
        ax.bar(x, y, color="#e67e22")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "errors_per_class.png", dpi=120)
    plt.close(fig)

    summary = {
        "totals": dict(totals),
        "per_class_cls": {class_names[k]: v for k, v in per_class_cls.items()},
        "per_class_loc": {class_names[k]: v for k, v in per_class_loc.items()},
        "per_class_miss": {class_names[k]: v for k, v in per_class_miss.items()},
    }
    return summary


@torch.no_grad()
def run_error_analysis(
    ckpt_path: str,
    config_path: str,
    out_dir: str,
    max_visualize: int = 20,
    score_thr: float = 0.3,
) -> None:
    cfg = load_config(config_path)
    out_dir = ensure_dir(out_dir)
    logger = setup_logger("error_analysis")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id2label = {i: n for i, n in enumerate(cfg["data"]["class_names"])}

    model, processor = build_model_and_processor(
        cfg["model"]["name"],
        num_classes=cfg["data"]["num_classes"],
        num_queries=cfg["model"]["num_queries"],
        id2label=id2label,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device).eval()
    logger.info(f"Loaded ckpt: {ckpt_path}")

    val_ds = CocoDetectionAlb(
        cfg["data"]["val_img_dir"],
        cfg["data"]["val_ann"],
        processor,
        is_train=False,
    )
    loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn)

    all_stats = []
    vis_dir = ensure_dir(out_dir / "vis")
    n_vis = 0

    for idx, batch in enumerate(tqdm(loader, desc="error analysis")):
        pv = batch["pixel_values"].to(device)
        pm = batch["pixel_mask"].to(device)
        outputs = model(pixel_values=pv, pixel_mask=pm)

        target_sizes = torch.stack([lbl["orig_size"] for lbl in batch["labels"]]).to(device)
        results = processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target_sizes
        )[0]
        preds = {
            "boxes": results["boxes"].cpu(),
            "scores": results["scores"].cpu(),
            "labels": results["labels"].cpu(),
        }
        gts = batch["labels"][0]
        # HF: gts["boxes"] в cxcywh нормализованной шкале; конвертируем в xyxy * orig_size
        H, W = gts["orig_size"].tolist()
        cxcywh = gts["boxes"].clone()
        cxcywh[:, 0] *= W
        cxcywh[:, 1] *= H
        cxcywh[:, 2] *= W
        cxcywh[:, 3] *= H
        gt_xyxy = torch.stack(
            [
                cxcywh[:, 0] - cxcywh[:, 2] / 2,
                cxcywh[:, 1] - cxcywh[:, 3] / 2,
                cxcywh[:, 0] + cxcywh[:, 2] / 2,
                cxcywh[:, 1] + cxcywh[:, 3] / 2,
            ],
            dim=1,
        )
        gt_dict = {"boxes": gt_xyxy.numpy(), "labels": gts["class_labels"].numpy()}
        # numpyify preds for IoU
        preds_np = {
            "boxes": preds["boxes"].numpy(),
            "scores": preds["scores"].numpy(),
            "labels": preds["labels"].numpy(),
        }

        cats = classify_errors(preds_np, gt_dict, score_thr=score_thr)

        # per-class
        per_class_cls = Counter(int(preds_np["labels"][i]) for i in cats["cls"])
        per_class_loc = Counter(int(preds_np["labels"][i]) for i in cats["loc"])
        per_class_miss = Counter(int(gt_dict["labels"][i]) for i in cats["missed_gt"])

        all_stats.append(
            dict(
                per_image=cats,
                per_class_cls=per_class_cls,
                per_class_loc=per_class_loc,
                per_class_miss=per_class_miss,
            )
        )

        # visualization
        if n_vis < max_visualize and (cats["cls"] or cats["loc"] or cats["missed_gt"]):
            img_id = int(gts["image_id"].item())
            img_info = val_ds.coco.coco.loadImgs([img_id])[0]
            img_path = Path(cfg["data"]["val_img_dir"]) / img_info["file_name"]
            if img_path.exists():
                pil = Image.open(img_path).convert("RGB")
                vis = draw_boxes(
                    pil,
                    pred_boxes=preds_np["boxes"][preds_np["scores"] >= score_thr],
                    pred_labels=preds_np["labels"][preds_np["scores"] >= score_thr],
                    pred_scores=preds_np["scores"][preds_np["scores"] >= score_thr],
                    gt_boxes=gt_dict["boxes"],
                    gt_labels=gt_dict["labels"],
                    class_names=cfg["data"]["class_names"],
                )
                vis.save(vis_dir / f"img_{img_id:08d}.png")
                n_vis += 1

    summary = aggregate_and_plot(all_stats, out_dir, cfg["data"]["class_names"])
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Summary: {summary['totals']}")
    logger.info(f"Saved to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="reports/error_analysis")
    p.add_argument("--score_thr", type=float, default=0.3)
    p.add_argument("--max_vis", type=int, default=20)
    args = p.parse_args()

    run_error_analysis(args.ckpt, args.config, args.out, args.max_vis, args.score_thr)


if __name__ == "__main__":
    main()
