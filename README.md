# Chord recognition in audio — final model

Code and trained weights of the compact chord recogniser (2.46M parameters) from the paper
*Why Do Chord Recognisers Change Their Answer When the Key or the Dataset Changes?* (A. Nurym, Zh. Basheyeva, G. Abdoldinova)
and the master's thesis *Development of Machine Learning Models for Musical Chord Recognition in Digital Audio*
(Astana IT University).

- **Model:** CQT input → CNN stem → 4 × (Transformer layer with rotary positions + convolution module), outputs for
  major/minor, the set of sounding pitch classes, bass, root and (training only) notes; a template Viterbi decoder over
  180 chords; five inference rules (averaging over three transpositions, change penalty 96, level clamping, key-to-function
  decoding, overlapping 10 s windows).
- **Weights:** 15 models of the frozen recipe (5 cross-validation folds × seeds 42/43/44) in `weights/`.
- **Held-out results** (evaluated once after freezing; mean of 15 models with the five rules; weighted chord symbol recall, %):

  | Set | root | majmin | triads | sevenths | tetrads | MIREX |
  |---|---|---|---|---|---|---|
  | BPSD-final, 80 recordings (Beethoven sonatas) | 70.71 | 75.74 | 69.10 | 61.70 | 56.83 | 73.31 |
  | JazzSAMBA-test, 14 recordings | 68.67 | 62.86 | 58.87 | 41.31 | 38.57 | 70.09 |

**Quick start:** `pip install -r requirements.txt`, then `python -m chordrec.recognize recording.wav` (one model, all five rules,
CPU). The notebook `notebooks/overview.ipynb` (or `overview.html`) walks through the code, the training steps and all numbers.

**Licences:** code — MIT (`LICENSE`); trained weights — CC BY-NC-SA 4.0 (`weights/LICENSE.md`), because the training data
include non-commercial piano datasets (MAPS, SMD) and MAESTRO. Datasets are not redistributed.

Below: details in Russian.

---

## Распознавание аккордов в аудио — итоговая модель

Магистерская диссертация «Development of Machine Learning Models for Musical Chord Recognition in Digital Audio»,
Нұрым Абзал, Astana IT University.

Чистая копия проекта: только код итогового (замороженного) рецепта CV-Rmnds, веса 15 обученных моделей и сохранённые числа. Исследовательский
проект (`chord-research-lab`) со всеми опытами, планами и отчётами остаётся как есть: числа диссертации получены на нём.
Этот код проверен против него: те же веса дают те же ответы (раздел «Проверка»).

**Начать с ноутбука:** [`notebooks/overview.ipynb`](notebooks/overview.ipynb) (или `notebooks/overview.html` в браузере, без Jupyter).

## Что внутри

| Папка или файл | Что там |
|---|---|
| `chordrec/config.py` | константы рецепта (CQT, сдвиги, фолды) и пути к данным |
| `chordrec/features.py` | признаки CQT, окно бинов, нормировка входа, правило громкости (iii) |
| `chordrec/model.py` | сеть: свёрточная часть → 4 × (Transformer с RoPE + свёрточный модуль) → 5 выходов; 2,46 млн параметров |
| `chordrec/decode.py` | Витерби; декодер «мажор/минор»; декодер полных меток по 180 шаблонам |
| `chordrec/rules.py` | пять правил распознавания: K = 3, штраф 96, громкость, тональность → функция, перекрывающиеся окна |
| `chordrec/recognize.py` | распознать аудиофайл из командной строки |
| `chordrec/labels.py` | цели обучения: мажор/минор и корень, «набор высот» и бас, ноты, бас из нот, сдвиг целей |
| `chordrec/losses.py` | потери: сглаживание по консонансу, согласие транспозиций, дистилляция |
| `chordrec/data.py` | данные: GuitarSet по фолду, синтетика AAM, пулы MusicNet, рояля и джаза, ноты MAESTRO |
| `chordrec/train.py` | обучение одной модели |
| `chordrec/cv.py` | 15 моделей: 5 фолдов × 3 seeds |
| `chordrec/evaluate.py` | уровни `mir_eval` (WCSR) |
| `weights/` | 15 моделей `fold{k}/s{seed}.pt`, нормировка фолдов `stats_fold{k}.npz`, штраф декодера каждой модели `models.json` |
| `results/` | сохранённые числа для таблиц ноутбука; источник каждого файла — `results/SOURCES.json` |
| `notebooks/` | ноутбук и его сборщик `build_overview.py` |
| `tests/check_against_research.py` | проверка против исследовательского кода |
| `tools/` | упаковка весов и чисел из исследовательского проекта |

## Запуск

Окружение исследовательского проекта уже подходит (`chord-research-lab\.venv`); для нового — `pip install -r requirements.txt`.

```
python -m chordrec.recognize запись.wav              # одна модель, все пять правил, процессор
python -m chordrec.recognize запись.wav --models 5   # среднее 5 моделей
python -m chordrec.recognize запись.wav --no-rules   # модель как есть, без правил
python -m chordrec.train --fold 0 --seed 42 --out runs/final   # обучение одной модели (видеокарта, ≈ 35 мин + учитель)
python -m chordrec.cv --out runs/final                          # все 15 моделей
python notebooks/build_overview.py                              # пересобрать ноутбук
```

## Данные

Данные в проект не копируются: признаки, манифесты и ноты читаются из исследовательского проекта, лежащего рядом
(другой путь — переменная окружения `CHORD_RESEARCH_ROOT`). Часть наборов (MAPS, SMD, RWC) по лицензии распространять нельзя.
Подготовка данных — в исследовательском проекте: подписи самообучения MusicNet (`tools/cycles/st_make.py`), уточнённые метки
рояля (`results/side-data/scripts/labels-refine/piano_refine.py`), ноты MAESTRO (`tools/data/maestro_notes.py`), манифесты пулов. Здесь используются их готовые манифесты.

## Проверка

```
python tests/check_against_research.py          # быстро: нормировка, модель, 8 записей
python tests/check_against_research.py --full   # + всё BPSD-dev для одной модели против сохранённых чисел
python tests/check_training.py runs/check       # 20 шагов обучения исследовательским и чистым кодом
```

Распознавание и правила итоговой модели совпадают с исследовательским кодом побитно (`tests/last_check.txt`, 23.11.2026).
Код обучения повторяет рецепт итоговой модели (доли пулов, ноты MAESTRO, бас из нот, дистилляция, согласие транспозиций);
короткая проверка обучения — `tests/check_training.py`. Полное переобучение 15 моделей чистым кодом не запускалось.
