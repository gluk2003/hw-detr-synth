#!/usr/bin/env bash
# Шаг 1 для HW2.5: нарезаем кропы из COCO-10 (real source) +
# Canny-источники для motorcycle. Шаг 2: запускаем SD+ControlNet генерацию.
set -euo pipefail

mkdir -p data/crops/train data/crops/val data/synth

# 1) кропы из train (для real-части ablation + Canny-edges для motorcycle)
python -m src.data.extract_crops \
    --ann data/coco10/train/annotations.json \
    --img_dir data/coco10/train/images \
    --out data/crops/train \
    --classes stop_sign,traffic_light,motorcycle

# 2) кропы из val (только real, для оценки)
python -m src.data.extract_crops \
    --ann data/coco10/val/annotations.json \
    --img_dir data/coco10/val/images \
    --out data/crops/val \
    --classes stop_sign,traffic_light,motorcycle

# 3) синтетика
python -m src.synth.generate_sd_controlnet --config configs/synth.yaml

echo ">>> Синтетика: data/synth/<class>/"
