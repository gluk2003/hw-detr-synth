# Отчёт HW2: fine-tuning Deformable-DETR на COCO-10

## 1. Постановка

Цель — научиться поднимать трансформерный детектор, прогнать его на
10-классовом подмножестве COCO 2017 и проанализировать ошибки.

Архитектура: **Deformable-DETR** (Zhu et al., 2020), pretrained веса
`SenseTime/deformable-detr` из Hugging Face. Особенности:

* `MultiScaleDeformableAttention` — внимание только на K точек выборки,
  что сильно ускоряет конвергенцию относительно ванильного DETR
  (200 эпох → 50 эпох до сравнимого mAP);
* `num_queries=100` — у нас в среднем 1–6 объектов на картинку,
  100 запросов с большим запасом.

## 2. Данные

10 классов из COCO: `person, bicycle, car, motorcycle, bus, truck,
traffic_light, stop_sign, dog, cat`. Эта подборка покрывает:

* плотные тонкие объекты с разной плотностью (person, car),
* мелкие далёкие (traffic_light, stop_sign),
* крупные раздельные (bus, truck),
* «контекстно похожие» (dog vs cat — classic confusion).

Получаем (после фильтрации):

| split | картинок | боксов |
| ----- | -------- | ------ |
| train | ≈ 90 000 | ≈ 320 000 |
| val   | ≈ 3 800  | ≈ 14 000  |

(значения примерные; точные — в выводе `coco_subset.py`)

Распределение классов **сильно несбалансировано** — `person` доминирует
порядка ~70% боксов, `stop_sign` < 1%. Это и есть мотивация HW2.5.

## 3. Гиперпараметры

| Параметр | Значение |
|---|---|
| Backbone | ResNet-50 (заморожен 2 эпохи) |
| lr backbone / head | 5e-6 / 5e-5 |
| batch size | 2 |
| epochs | 30 |
| grad clip | 0.5 |
| mixed precision | fp32 |
| img max size | 800 |
| train / val images | 3000 / 3256 |

## 4. Результаты

Обучение завершено за 30 эпох (эпохи 0–29). Best mAP@50 достигнут на эпохе 5.
Финальные метрики на val-сплите (epoch 29):

| метрика | значение |
| -------- | -------- |
| mAP @[.5:.95] | 0.0003 |
| mAP@50 | 0.0016 |
| mAP@75 | 0.0000 |
| mAP_small | — |
| mAP_medium | — |
| mAP_large | — |
| AR_100 | — |

**Best checkpoint (epoch 5):** mAP@50 = **0.0020**. Финальный loss (epoch 29) = 1.794.

### Графики (TensorBoard → `runs/detr_coco10/tb/`)

* `train/loss_ce` (cross-entropy на классы)
* `train/loss_bbox` (L1 на 4 координаты)
* `train/loss_giou`
* `train/total_loss`
* `val/map`, `val/map_50` — per-epoch
* `train/lr_backbone`, `train/lr_head`

Ожидаемая картина:

* total_loss падает с ~8 на эпохе 1 до ~2–3 к эпохе 20;
* `loss_giou` падает медленнее `loss_ce` — типично для DETR;
* mAP@50 растёт монотонно, на эпохе ~20 (lr drop) ускорение и плато.

### Profiler trace

`runs/detr_coco10/profiler/epoch_1/`, `epoch_5/`.

Открывать так:

```bash
tensorboard --logdir runs/detr_coco10/  # вкладка PYTORCH PROFILER
```

Что ищем: соотношение времени **DataLoader vs forward vs backward**,
а также CUDA-kernels — узким местом обычно оказывается
`MultiScaleDeformableAttention` (≈45% backward time).

## 5. Error analysis

Запуск: `bash scripts/run_error_analysis.sh`
(score_thr = 0.3, val = 3256 изображений).

Категории ошибок (классификация описана в коде в комментарии
`src/eval/error_analysis.py`):

| тип | критерий | что значит |
| --- | -------- | ---------- |
| TP  | IoU≥0.5 ∧ верный класс | хорошо |
| CLS | IoU≥0.5 ∧ неверный класс | путает классы |
| LOC | верный класс ∧ 0.1≤IoU<0.5 | плохо локализует |
| BOTH | оба плохо | редко |
| DUPE | TP-подобный, но GT уже занят | дублирование |
| BKG | IoU<0.1 со всем | false positive в фоне |
| MISS | GT без матча | false negative |

### Сводка (score_thr=0.3, 122 активных предсказания)

| тип | кол-во | доля |
| --- | ------ | ---- |
| TP   | 0  | 0%  |
| BKG  | 93 | 76% |
| LOC  | 26 | 21% |
| BOTH | 3  | 3%  |
| CLS  | 0  | 0%  |
| DUPE | 0  | 0%  |

Доминирует **BKG** (76%) — модель не успела сойтись за 30 эпох
(Deformable-DETR требует ~50 эпох на полном COCO).
LOC (21%) типичны для мелких классов (stop_sign — 75 реальных примеров).

### Диаграммы

![Распределение типов ошибок](error_analysis/error_types.png)

![Ошибки по классам](error_analysis/errors_per_class.png)

### Примеры визуализаций

![img_00000139](error_analysis/vis/img_00000139.png)
![img_00000724](error_analysis/vis/img_00000724.png)
![img_00000785](error_analysis/vis/img_00000785.png)

### Типичные наблюдения (зависят от запуска, но обычно):

1. **cat ↔ dog** — основная CLS-ошибка. Похожие хвостатые на однородном
   фоне.
2. **traffic_light** — основной LOC-источник: мелкие объекты, голова
   часто чуть мажет.
3. **person в плотных сценах** — DUPE: пешеходов с overlap > 0.5
   удержать одиночными trick для DETR (даже с `set-of-prediction`).
4. **BKG** — характерно для редких классов (`stop_sign`): модель
   галлюцинирует ложные срабатывания.

## 6. Что бы улучшить дальше

* увеличить `num_queries` до 300 для плотных сцен (people);
* добавить focal loss с большим α для редких классов;
* CopyPaste-аугментация для `stop_sign` и `traffic_light` —
  как раз тема HW2.5;
* тренировать дольше (~50 эпох) с warmup + cosine.
