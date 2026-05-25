#!/usr/bin/env bash
# Запуск всех 4 экспериментов ablation: baseline + 3 уровня synth.
set -euo pipefail

python -m src.train.train_classifier \
    --config configs/synth.yaml \
    --real_train data/crops/train \
    --real_val   data/crops/val \
    --synth_dir  data/synth

echo ">>> Готово."
echo "    Таблица:     reports/synth_ablation_results.json"
echo "    TensorBoard: tensorboard --logdir runs/synth_ablation"
