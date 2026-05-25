"""Unit-тесты чистой логики error_logic. Запускается без GPU/HF/torch.

Запуск:
    python -m pytest tests/test_error_logic.py -v
или просто:
    python tests/test_error_logic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.error_logic import box_iou, classify_errors  # noqa: E402


# ---------- box_iou ----------------------------------------------------------

def test_box_iou_identical():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    iou = box_iou(a, a)
    assert iou.shape == (1, 1)
    assert abs(float(iou[0, 0]) - 1.0) < 1e-5


def test_box_iou_disjoint():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[20, 20, 30, 30]], dtype=np.float32)
    iou = box_iou(a, b)
    assert float(iou[0, 0]) == 0.0


def test_box_iou_half_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[5, 0, 15, 10]], dtype=np.float32)
    # intersection 5*10=50, union 100+100-50=150, iou=1/3
    iou = float(box_iou(a, b)[0, 0])
    assert abs(iou - 1 / 3) < 1e-5


def test_box_iou_empty():
    a = np.zeros((0, 4), dtype=np.float32)
    b = np.array([[0, 0, 1, 1]], dtype=np.float32)
    iou = box_iou(a, b)
    assert iou.shape == (0, 1)


# ---------- classify_errors --------------------------------------------------

def _preds(boxes, labels, scores):
    return {
        "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
        "labels": np.array(labels, dtype=np.int64),
        "scores": np.array(scores, dtype=np.float32),
    }


def _gts(boxes, labels):
    return {
        "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
        "labels": np.array(labels, dtype=np.int64),
    }


def test_pure_true_positive():
    preds = _preds([[0, 0, 10, 10]], [3], [0.9])
    gts = _gts([[0, 0, 10, 10]], [3])
    r = classify_errors(preds, gts)
    assert r["tp"] == [0]
    assert r["missed_gt"] == []
    assert all(len(r[k]) == 0 for k in ["cls", "loc", "both", "dupe", "bkg"])


def test_cls_error():
    # хороший IoU, но класс не тот
    preds = _preds([[0, 0, 10, 10]], [5], [0.9])
    gts = _gts([[0, 0, 10, 10]], [3])
    r = classify_errors(preds, gts)
    assert r["cls"] == [0]
    assert r["missed_gt"] == [0]  # GT не получил TP


def test_loc_error():
    # верный класс, но IoU между iou_min и iou_match
    preds = _preds([[0, 0, 10, 10]], [3], [0.9])
    gts = _gts([[6, 0, 16, 10]], [3])
    # inter=4*10=40, union=100+100-40=160, iou=0.25 → в [0.1, 0.5)
    r = classify_errors(preds, gts)
    assert r["loc"] == [0]


def test_both_error():
    # неверный класс, IoU в [iou_min, iou_match)
    preds = _preds([[0, 0, 10, 10]], [7], [0.9])
    gts = _gts([[6, 0, 16, 10]], [3])
    r = classify_errors(preds, gts)
    assert r["both"] == [0]


def test_bkg_hallucination():
    # IoU ниже iou_min со всеми GT — фоновая галлюцинация
    preds = _preds([[100, 100, 110, 110]], [3], [0.9])
    gts = _gts([[0, 0, 10, 10]], [3])
    r = classify_errors(preds, gts)
    assert r["bkg"] == [0]
    assert r["missed_gt"] == [0]


def test_dupe_detection():
    # два хороших предсказания одного и того же GT
    preds = _preds(
        [[0, 0, 10, 10], [0, 0, 10, 10]],
        [3, 3],
        [0.9, 0.8],
    )
    gts = _gts([[0, 0, 10, 10]], [3])
    r = classify_errors(preds, gts)
    assert r["tp"] == [0]
    assert r["dupe"] == [1]


def test_score_threshold_filters():
    preds = _preds([[0, 0, 10, 10]], [3], [0.1])
    gts = _gts([[0, 0, 10, 10]], [3])
    r = classify_errors(preds, gts, score_thr=0.3)
    # отфильтровано по score — GT остаётся непокрытым
    assert r["tp"] == []
    assert r["missed_gt"] == [0]


def test_no_predictions():
    preds = _preds([], [], [])
    gts = _gts([[0, 0, 10, 10]], [3])
    r = classify_errors(preds, gts)
    assert r["missed_gt"] == [0]
    assert all(r[k] == [] for k in ["tp", "cls", "loc", "both", "dupe", "bkg"])


def test_no_gts():
    preds = _preds([[0, 0, 10, 10]], [3], [0.9])
    gts = _gts([], [])
    r = classify_errors(preds, gts)
    assert r["bkg"] == [0]
    assert r["missed_gt"] == []


def test_score_order_matters_for_dupe():
    # лучший по score должен стать TP, второй — DUPE
    preds = _preds(
        [[0, 0, 10, 10], [0, 0, 10, 10]],
        [3, 3],
        [0.5, 0.99],
    )
    gts = _gts([[0, 0, 10, 10]], [3])
    r = classify_errors(preds, gts)
    # индекс 1 имел больший score → он TP, индекс 0 → DUPE
    assert r["tp"] == [1]
    assert r["dupe"] == [0]


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    n_pass = n_fail = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            n_fail += 1
        except Exception as e:
            print(f"  ERR   {fn.__name__}: {type(e).__name__}: {e}")
            n_fail += 1
    print(f"\n{n_pass}/{n_pass + n_fail} passed")
    sys.exit(0 if n_fail == 0 else 1)
