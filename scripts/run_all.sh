#!/usr/bin/env bash
# E2E: подготовка данных -> DETR -> error analysis -> synth -> classifier ablation.
# Использование: bash scripts/run_all.sh /path/to/coco_root
set -euo pipefail

COCO_ROOT="${1:?Передай путь к корню COCO}"

echo "=== [1/5] Подготовка COCO-10 ==="
bash scripts/prepare_coco_subset.sh "${COCO_ROOT}"

echo "=== [2/5] Fine-tuning Deformable-DETR ==="
bash scripts/run_detr.sh

echo "=== [3/5] Error analysis ==="
bash scripts/run_error_analysis.sh

echo "=== [4/5] Генерация синтетики (SD + ControlNet) ==="
bash scripts/generate_synth.sh

echo "=== [5/5] Classifier ablation: baseline vs +synth ==="
bash scripts/run_classifier_ablation.sh

echo ""
echo "=========================================="
echo "  Пайплайн отработал. Артефакты:"
echo "    runs/detr_coco10/             # TB-логи + чекпойнты + profiler trace"
echo "    runs/synth_ablation/          # TB-логи ablation"
echo "    reports/error_analysis/       # бары ошибок + визуализации"
echo "    reports/synth_ablation_results.json"
echo "    data/synth/                   # синтетические картинки"
echo "=========================================="
