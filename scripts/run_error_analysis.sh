#!/usr/bin/env bash
# Error analysis на лучшем чекпойнте.
set -euo pipefail

CKPT="${1:-runs/detr_coco10/ckpt/best.pt}"
CONFIG="${2:-configs/detr.yaml}"

python -m src.eval.error_analysis \
    --ckpt "${CKPT}" \
    --config "${CONFIG}" \
    --out reports/error_analysis \
    --score_thr 0.3 \
    --max_vis 30

echo ">>> Анализ ошибок: reports/error_analysis/"
