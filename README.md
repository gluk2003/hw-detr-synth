# HW2 + HW2.5: DETR-детектор и синтетика через Stable Diffusion + ControlNet

Репозиторий объединяет два домашних задания:

1. **HW2** — fine-tuning Deformable-DETR на подмножестве COCO (10 классов),
   полный train-loop с TensorBoard, чекпойнтами, PyTorch profiler,
   расчёт mAP / mAP@50, error analysis (классификация vs локализация).
2. **HW2.5** — генерация синтетических изображений редких классов через
   Stable Diffusion + ControlNet, добавление их в тренировку классификатора
   (ViT) и ablation: baseline vs +synth.

> Замечание о воспроизводимости: реальное обучение DETR на 10 классах COCO
> занимает ~6–8 часов на одной A100. Все скрипты ниже **детерминированы**
> (фиксированный seed, версии в `requirements.txt`), но физический запуск
> требует GPU. В `reports/` лежат ожидаемые шаблоны таблиц с пояснениями,
> куда после запуска подставятся реальные числа.

---

## Результаты

### HW2 — Deformable-DETR (COCO-10, 30 эпох)

| метрика | значение |
| -------- | -------- |
| mAP @[.5:.95] (epoch 29) | 0.0003 |
| mAP@50 — best (epoch 5) | **0.0020** |
| mAP@75 (epoch 29) | 0.0000 |
| loss (epoch 29) | 1.794 |

Error analysis (score_thr=0.3, 3256 val images): BKG 76% · LOC 21% · BOTH 3%.
Подробнее: [reports/HW2_report.md](reports/HW2_report.md).

### HW2.5 — Синтетика SD + ControlNet (ablation, ViT-Tiny)

| эксперимент | overall acc | stop_sign | traffic_light | motorcycle |
| ----------- | ----------- | --------- | ------------- | ---------- |
| baseline_real_only | 0.9832 | 0.897 | 0.489 | 0.832 |
| synth_25pct        | 0.9834 | 0.897 | 0.422 | 0.856 |
| synth_50pct        | 0.9839 | 0.872 | 0.467 | 0.844 |
| synth_100pct       | 0.9841 | 0.897 | 0.422 | 0.848 |

Вывод: доменный разрыв нивелирует прирост объёма данных.
Подробнее: [reports/HW2.5_report.md](reports/HW2.5_report.md).

Чекпойнт `best.pt` (478 MB): [Releases v1.0](https://github.com/gluk2003/hw-detr-synth/releases/tag/v1.0)

---

## Структура

```
hw-detr-synth/
├── configs/
│   ├── detr.yaml             # гиперпараметры HW2
│   └── synth.yaml            # гиперпараметры HW2.5
├── src/
│   ├── data/
│   │   ├── coco_subset.py    # выделение 10-классового сабсета из COCO
│   │   ├── transforms.py     # аугментации (HF image-processor + albumentations)
│   │   └── datamodule.py     # CocoDetection + DataLoader
│   ├── models/
│   │   ├── detr_wrapper.py   # обёртка над HF DeformableDetrForObjectDetection
│   │   └── classifier.py     # ViT/ResNet-классификатор для HW2.5
│   ├── train/
│   │   ├── train_detr.py     # главный train-loop: TB + profiler + ckpt
│   │   ├── train_classifier.py
│   │   └── engine.py         # train_one_epoch / evaluate
│   ├── synth/
│   │   ├── generate_sd_controlnet.py
│   │   └── prompts.py        # промпты по классам
│   ├── eval/
│   │   ├── metrics.py        # mAP / mAP@50 (pycocotools)
│   │   ├── error_analysis.py # cls-error vs loc-error в духе TIDE
│   │   └── visualize.py      # отрисовка предсказанных боксов
│   └── utils/
│       ├── logger.py
│       ├── misc.py
│       └── seed.py
├── scripts/
│   ├── prepare_coco_subset.sh
│   ├── run_detr.sh
│   ├── run_error_analysis.sh
│   ├── generate_synth.sh
│   ├── run_classifier_ablation.sh
│   └── run_all.sh
├── notebooks/
│   ├── 01_hw2_detr_exploration.ipynb
│   └── 02_hw25_synth_ablation.ipynb
├── tests/
│   └── test_error_logic.py     # unit-тесты логики ошибок (numpy-only)
├── reports/
│   ├── HW2_report.md         # таблица mAP, наблюдения
│   └── HW2.5_report.md       # ablation +synth
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Установка

```bash
git clone <repo> hw-detr-synth && cd hw-detr-synth
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Требуется PyTorch ≥ 2.2 с CUDA, ≥ 16 ГБ VRAM для DETR (batch=2 fits в 12 ГБ
если включить `mixed_precision: fp16`).

---

## HW2 — Deformable-DETR

### 1. Подготовка датасета

```bash
bash scripts/prepare_coco_subset.sh /path/to/coco
```

Скрипт скачивает COCO 2017 (если не лежит) и через `src/data/coco_subset.py`
делает подмножество из 10 классов:

```
person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign, dog, cat
```

(тематика — «улица + домашние животные», даёт разнообразие по размеру боксов
и плотности, что хорошо для error analysis). Подмножество сохраняется в
`data/coco10/{train,val}/{images, annotations.json}` в формате COCO,
с переиндексацией `category_id` в `[0, 9]`.

### 2. Тренировка

```bash
bash scripts/run_detr.sh
# = python -m src.train.train_detr --config configs/detr.yaml
```

Что произойдёт:

* модель `SenseTime/deformable-detr` грузится из HF с `num_labels=10`
  (классификационная голова переинициализируется);
* train-loop логирует в TensorBoard:
  * total / classification / bbox L1 / GIoU losses (per-step и per-epoch),
  * learning rate, grad norm,
  * mAP, mAP@50, mAP_small/medium/large каждую N-ю эпоху на val;
* PyTorch profiler пишет trace на эпохах 1 и 5 в `runs/<exp>/profiler/`
  (открывать в Chrome через `chrome://tracing` или TB Profiler plugin);
* чекпойнты сохраняются каждые `save_every` эпох и при улучшении mAP@50
  в `runs/<exp>/ckpt/`.

```bash
tensorboard --logdir runs/
```

### 3. Оценка и error analysis

```bash
python -m src.eval.error_analysis \
    --ckpt runs/detr_coco10/ckpt/best.pt \
    --val  data/coco10/val \
    --out  reports/error_analysis/
```

Делит ошибки на:

* **classification** — IoU(pred, GT) ≥ 0.5, но класс другой,
* **localization** — класс верный, 0.1 ≤ IoU < 0.5,
* **duplicate** — верный класс, IoU ≥ 0.5, но GT уже занят другим detection,
* **background** — pred без матча с GT (false positive),
* **missed** — GT без матча с pred (false negative).

Результаты — таблица в `reports/HW2_report.md` + бары и
визуализации в `reports/error_analysis/`.

---

## HW2.5 — синтетика через SD + ControlNet

### 1. Выбор редких классов

`src/data/coco_subset.py --stats` печатает таблицу частот.
Редкими в нашем сабсете обычно оказываются `stop sign`, `traffic light`,
`motorcycle` — для них и генерируем синтетику.

### 2. Генерация

```bash
bash scripts/generate_synth.sh
# = python -m src.synth.generate_sd_controlnet \
#       --classes "stop sign,traffic light,motorcycle" \
#       --n_per_class 500 \
#       --out data/synth/
```

Используется `stable-diffusion-v1-5` + `lllyasviel/sd-controlnet-canny`.
Canny-условия — это:

* для `stop sign`, `traffic light` — нарисованные процедурно
  октогон/прямоугольник со светофорными кругами (см. `src/synth/canny_priors.py`),
  расположенные в случайных позициях на канвасе. ControlNet удерживает
  форму, SD дорисовывает фон / освещение / погоду.
* для `motorcycle` — Canny edges, извлечённые из реальных мото-фото
  из тренировочного набора (стиль аугментации).

Промпты — в `src/synth/prompts.py` (positive + negative).

### 3. Обучение классификатора (baseline vs +synth)

Чтобы изолировать эффект синтетики, тренируем ViT-Tiny на бинарной
задаче «есть ли объект редкого класса на кропе» — это быстро (~10 минут
на A100) и даёт чистый ablation.

```bash
bash scripts/run_baseline_cls.sh    # только real
bash scripts/run_synth_cls.sh       # real + synth
```

Сравнение метрик пишется в `reports/HW2.5_report.md`.

---

## Воспроизводимость

* `src/utils/seed.py` — фиксация seed для `random`, `numpy`, `torch`,
  `torch.cuda`, `cudnn.deterministic=True`;
* версии всех библиотек жёстко зафиксированы в `requirements.txt`;
* конфиги в YAML — гиперпараметры не в коде.

---

## Лицензия

Учебный проект, MIT.
