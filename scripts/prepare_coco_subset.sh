#!/usr/bin/env bash
# Скачивает COCO 2017 (если нет) и нарезает 10-классовый сабсет.
# Использование: bash scripts/prepare_coco_subset.sh /path/to/coco_root
set -euo pipefail

COCO_ROOT="${1:?Передай путь к корню COCO}"

# Скачиваем, если не лежит
mkdir -p "${COCO_ROOT}"
cd "${COCO_ROOT}"

if [[ ! -d annotations ]]; then
    echo ">>> качаю annotations..."
    wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
fi
if [[ ! -d train2017 ]]; then
    echo ">>> качаю train2017 (≈19 ГБ)..."
    wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip
    unzip -q train2017.zip
    rm train2017.zip
fi
if [[ ! -d val2017 ]]; then
    echo ">>> качаю val2017 (≈1 ГБ)..."
    wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip
    unzip -q val2017.zip
    rm val2017.zip
fi

cd - > /dev/null

echo ">>> Нарезаю 10-классовый сабсет..."
python -m src.data.coco_subset --coco_root "${COCO_ROOT}" --out_dir data/coco10
echo ">>> Готово: data/coco10/{train,val}/"
