"""Метрики через pycocotools.

Этап:
* preds в HF-формате конвертируются в COCO-формат
  (image_id, category_id, bbox=[x,y,w,h], score);
* `COCOeval(gt, dt, "bbox")` отрабатывает -> печатает 12 чисел;
* мы достаём mAP, mAP@50, mAP@75, mAP_small/medium/large.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoMapEvaluator:
    def __init__(self, ann_file: dict | str | Path) -> None:
        """ann_file может быть путём к JSON, либо уже распарсенным dict."""
        if isinstance(ann_file, (str, Path)):
            self.coco_gt = COCO(str(ann_file))
        else:
            # передан dict — загружаем "вручную"
            self.coco_gt = COCO()
            self.coco_gt.dataset = ann_file
            with contextlib.redirect_stdout(io.StringIO()):
                self.coco_gt.createIndex()

        self.predictions: list[dict] = []

    def update(
        self,
        batch_results: list[dict[str, torch.Tensor]],
        image_ids: list[int],
    ) -> None:
        """batch_results — выход processor.post_process_object_detection."""
        for img_id, res in zip(image_ids, batch_results):
            scores = res["scores"].detach().cpu().numpy()
            labels = res["labels"].detach().cpu().numpy()
            boxes = res["boxes"].detach().cpu().numpy()  # x1,y1,x2,y2

            for s, l, b in zip(scores, labels, boxes):
                x1, y1, x2, y2 = b
                self.predictions.append(
                    {
                        "image_id": int(img_id),
                        "category_id": int(l),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s),
                    }
                )

    def summarize(self) -> dict[str, float]:
        if not self.predictions:
            return {f"map": 0.0, "map_50": 0.0}
        # COCOeval хочет JSON-файл или список через loadRes
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = self.coco_gt.loadRes(self.predictions)
            coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                coco_eval.summarize()

        s = coco_eval.stats  # 12-вектор
        return {
            "map": float(s[0]),
            "map_50": float(s[1]),
            "map_75": float(s[2]),
            "map_small": float(s[3]),
            "map_medium": float(s[4]),
            "map_large": float(s[5]),
            "ar_1": float(s[6]),
            "ar_10": float(s[7]),
            "ar_100": float(s[8]),
            "summary_text": buf.getvalue(),
        }
