#!/usr/bin/env bash
# Запуск fine-tuning Deformable-DETR на COCO-10.
set -euo pipefail

CONFIG="${1:-configs/detr.yaml}"

mkdir -p runs

python -m src.train.train_detr --config "${CONFIG}"

echo ">>> Тренировка завершена."
echo "    TensorBoard:  tensorboard --logdir runs/"
echo "    Profiler:     runs/<exp>/profiler/"
echo "    Checkpoints:  runs/<exp>/ckpt/"
