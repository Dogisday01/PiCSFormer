"""Сборка ноутбука notebooks/overview.ipynb (с выполненными ячейками) и его копии для браузера overview.html.

python notebooks/build_overview.py
Ячейки выполняются здесь же (Jupyter не нужен); в Jupyter ноутбук можно перезапустить по порядку из папки проекта.
"""
import contextlib
import html
import io
import json
import re
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CELLS = []


def md(text):
    CELLS.append(("md", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


# ================================================================ содержание
md(r"""
# PiCSFormer: распознавание аккордов в аудио

**Магистерская диссертация** «Development of Machine Learning Models for Musical Chord Recognition in Digital Audio».
Нұрым Абзал, Astana IT University. Состояние на 24.11.2026.

Это чистая копия проекта: только код итоговой модели PiCSFormer (название — от выхода «набор высот», pitch-class set, и Transformer; пакет `chordrec`), веса 15 обученных моделей и сохранённые числа.
Ноутбук показывает, где в коде каждый механизм и каждое правило распознавания, сколько каждый дал, итоговые результаты
и распознаёт аккорды в аудиозаписи прямо здесь. Код проверен против исследовательского кода, на котором получены числа
диссертации: те же веса дают те же ответы (`tests/check_against_research.py`, раздел 8).

## Как читать числа
- Мера — взвешенная полнота символов аккордов (WCSR, `mir_eval`): доля времени записи с верным ответом на данном уровне, %.
- Уровни: **root** — основной тон; **majmin** — мажор или минор (C, Cm); **triads** — трезвучие целиком (C, Cm, Cdim, Caug, Csus4);
  **sevenths** — мажор, минор и септаккорды C7, Cmaj7, Cm7; **tetrads** — все четырёхзвучия (Cdim7, Cm7b5 и др.);
  **MIREX** — совпадают хотя бы три звука аккорда.
- «15 моделей» — 5 фолдов кросс-валидации по группам GuitarSet × 3 seeds. Разница рецептов — среднее 15 пар;
  «фолдов > 0» — в скольких фолдах из 5 она положительна.
- Наборы: **GuitarSet DEV** — 24 записи гитары; **BPSD-dev** без Op. 53 — 44 записи фортепиано (сонаты Бетховена);
  **JazzSAMBA-val** — 12 записей джазовых ансамблей; **RWC-P** — 100 записей поп-музыки.
""")

code(r'''
import importlib, inspect, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from IPython.display import HTML, display

ROOT = Path.cwd()
while not (ROOT / "chordrec").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import chordrec

LEVELS = ["root", "majmin", "triads", "sevenths", "tetrads", "mirex"]
SEEDS, FOLDS = (42, 43, 44), range(5)


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def f2(v, sign=False, nd=2):
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def cube(get):
    """[5 фолдов, 3 seeds, 6 уровней]"""
    return np.array([[[get(k, s)[q] for q in LEVELS] for s in SEEDS] for k in FOLDS])


def paired(a, b):
    """b − a: среднее 15 пар и число фолдов из 5 с плюсом."""
    d = (b - a).mean(1)
    return d.mean(0), (d > 0).sum(0)


def table(head, rows, note=None):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    out = f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"
    if note:
        out += f"<p style='font-size:90%;opacity:.75'>{note}</p>"
    display(HTML(out))


def where(obj):
    """Файл и строка начала функции или класса в пакете chordrec."""
    mod = inspect.getmodule(obj)
    return f"<code>chordrec/{mod.__name__.split('.')[-1]}.py:{inspect.getsourcelines(obj)[1]} {obj.__name__}</code>"


def line_of(module, marker):
    """Строка модуля, где стоит поясняющий комментарий marker."""
    for i, line in enumerate(inspect.getsource(module).splitlines(), 1):
        if marker in line:
            return f"<code>chordrec/{module.__name__.split('.')[-1]}.py:{i}</code>"
    raise KeyError(marker)


print("Пакет chordrec найден, PyTorch", torch.__version__)
''')

md(r"""
## 1. Структура проекта
| Папка | Что там |
|---|---|
| `chordrec/` | весь код итогового рецепта (таблица ниже строится из самих файлов) |
| `weights/` | 15 моделей (`fold{k}/s{seed}.pt`), нормировка входа каждого фолда, штраф декодера каждой модели (`models.json`) |
| `results/` | сохранённые числа исследовательского проекта, на которых строятся таблицы; источник каждого файла — `results/SOURCES.json` |
| `notebooks/` | этот ноутбук и его сборщик |
| `tests/` | проверка против исследовательского кода |
| `tools/` | упаковка весов и чисел из исследовательского проекта |
""")

code(r'''
MODULES = ["config", "features", "model", "decode", "rules", "recognize", "labels", "losses", "data", "train", "cv", "evaluate"]
rows = []
for name in MODULES:
    mod = importlib.import_module(f"chordrec.{name}")
    src = inspect.getsource(mod).splitlines()
    rows.append([f"<code>chordrec/{name}.py</code>", len(src), (mod.__doc__ or "").strip().splitlines()[0]])
rows.append(["<b>всего</b>", sum(r[1] for r in rows), ""])
table(["Файл", "строк", "Что делает"], rows)
''')

md(r"""
## 2. Как устроено распознавание
1. **Признаки.** CQT: 22 050 Гц, шаг 512 (23 мс), 36 бинов на октаву, 288 бинов от F#0, log(1 + 100·|X|); сеть читает 252 бина,
   запас ±6 полутонов нужен для транспозиций.
2. **Сеть.** Свёрточная часть, затем 4 блока «Transformer с вращательными позициями (RoPE) + свёрточный модуль», ширина 192.
3. **Пять выходов по кадрам:** мажор/минор (25), «набор высот» (12 сигмоид: звучит ли класс высоты как звук аккорда), бас (13),
   корень (13), ноты (88 клавиш; только при обучении).
4. **Декодер.** Витерби по 180 шаблонам (12 корней × 15 качеств) и «нет аккорда»; обращение — если бас звук аккорда, но не корень.
5. **Пять правил распознавания** поверх модели (раздел 5).
""")

code(r'''
from chordrec import decode, features, model as M, rules
d = load("results/models/fold0_s42.json")
table(["Часть", "Где в коде"], [
    ["Признаки CQT", where(features.cqt_from_signal)],
    ["Окно бинов и нормировка входа", where(features.crop_bins) + " · " + where(features.norm_stats)],
    ["Сеть целиком", where(M.ChordNet)],
    ["Блок Transformer с RoPE", where(M.Block)],
    ["Свёрточный модуль", where(M.ConvModule)],
    ["Витерби", where(decode.viterbi)],
    ["Декодер «мажор/минор»", where(decode.decode_flat)],
    ["Оценка 180 шаблонов", where(decode.pcs_emissions)],
    ["Декодер полных меток", where(decode.decode_pcs)],
], note=f"Параметров в одной модели: {d['params']:,}".replace(",", " "))
''')

md(r"""
## 3. Механизмы обучения: где в коде и что дали
Сначала — основа рецепта (ранние шаги; числа из отчётов того времени, тогда распознавание шло записью целиком). Затем — шаги
итогового рецепта, пересчитанные здесь из сохранённых файлов: 15 моделей шага против 15 моделей предыдущего, BPSD-dev и
JazzSAMBA-val окнами 10 с, без правил распознавания.

Каждый шаг принимался по правилу, записанному до запуска: выигрыш — BPSD sevenths или majmin ≥ +1,0 (4 из 5 фолдов), устойчивая
прибавка септим BPSD ≥ +0,3 (4/5) или гитара ≥ +0,3; без вреда — majmin и sevenths нигде не падают больше чем на 0,5.
Решения — только по валидации; тестовые наборы для решений не используются.
""")

code(r'''
from chordrec import data, labels, losses, train as T
early = [
    ("Транспозиционная аугментация: окно CQT сдвигается на −5…+6 полутонов, метки — так же",
     line_of(T, "# транспозиционная аугментация") + " · " + where(labels.shift_targets) + " · " + where(labels.shift_pcs), "GuitarSet majmin +12,9 — главный механизм"),
    ("Синтетика AAM: 600 синтезированных треков, доля 0,10", where(data.aam) + " · " + where(data.aam_features), "MusicNet majmin +11 на негитарных инструментах"),
    ("Пул MusicNet: классика, аккорды и септимы из нот, доля 0,15", where(data.pool), "BPSD majmin 43,1 → 56,3 (+13,3, 5/5)"),
    ("Пул рояля Disklavier (MAPS, SMD, KRAISLER), доля 0,10", where(data.pool), "BPSD majmin +5,25 (5/5)"),
    ("Выход «набор высот» и декодер по 180 шаблонам", where(labels.pcs_targets) + " · " + where(decode.pcs_emissions), "BPSD sevenths +4,93 (5/5)"),
    ("Джазовые записи JazzSAMBA, доля 0,15", where(data.pool), "джаз sevenths +11; BPSD majmin +1,0"),
    ("Уточнённые метки рояля", "манифест данных (подготовка — в исследовательском проекте)", "BPSD около +2 по всем уровням (sevenths +2,0, 5/5)"),
    ("Голова корня рядом с «набором высот» (вес 2 в декодере)", where(decode.pcs_emissions), "гитара root +2,49 (5/5)"),
]
table(["Механизм", "Где в коде", "Что дал (по отчёту)"], [list(r) for r in early])

STEPS = [
    ("Исходный рецепт", "00_base", ""),
    ("Вращательные позиции (RoPE)", "01_rope", where(M.Block)),
    ("Последняя эпоха вместо выбора", "02_last_epoch", line_of(T, "# веса последней эпохи")),
    ("Голова нот (88 клавиш)", "03_note_head", where(labels.note_targets) + " · " + line_of(T, "# ноты сдвигаются с окном")),
    ("Самообучение, круг 1", "04_self_training", "манифест MusicNet с подписями учителя"),
    ("Круг 2 + согласие транспозиций", "05_round2_consistency", where(losses.consistency_loss) + " · " + line_of(T, "# вторая копия отрезка")),
    ("Сглаживание по консонансу", "06_consonance", where(losses.consonance_ce)),
    ("Дистилляция ансамбля", "07_distillation", where(losses.teacher_targets) + " · " + where(losses.distillation_loss)),
    ("Бас из нот", "08_bass_from_notes", where(labels.bass_from_notes)),
    ("Ноты MAESTRO в голове нот", "09_maestro_notes", where(data.maestro_notes)),
]
LAST = STEPS[-1][0]  # итоговый (замороженный) рецепт
win = {}
for name, f, _ in STEPS:
    w = load(f"results/path/{f}.json")
    win[name] = {ds: cube(lambda k, s: w[f"fold{k}|s{s}|{ds}|win10s"]) for ds in ("bpsd", "jazz")}
Mi, Si = LEVELS.index("majmin"), LEVELS.index("sevenths")
rows = []
for i, (name, _, code_ref) in enumerate(STEPS):
    b, j = win[name]["bpsd"], win[name]["jazz"]
    if i == 0:
        rows.append([name, "", f2(b[..., Mi].mean()), f2(b[..., Si].mean()), f2(j[..., Si].mean()), f2(b[..., Si].mean())])
        continue
    pb, nb = paired(win[STEPS[i - 1][0]]["bpsd"], b)
    pj, _ = paired(win[STEPS[i - 1][0]]["jazz"], j)
    rows.append([name, code_ref, f"{f2(pb[Mi], True)} ({nb[Mi]})", f"{f2(pb[Si], True)} ({nb[Si]})", f2(pj[Si], True), f2(b[..., Si].mean())])
table(["Шаг", "Где в коде", "BPSD majmin", "BPSD sevenths", "Джаз sevenths", "BPSD sevenths, итого"], rows,
      note="Разница с предыдущей строкой, в скобках — фолдов > 0 из 5; первая строка — абсолютные значения. "
           "Круг 2 + согласие — ниже правила, принят решением автора (все шесть уровней BPSD выросли в 5/5); консонанс и дистилляция — "
           "по дополнению о септимах; бас из нот — по гитаре (balanced +0,59, 5/5); ноты MAESTRO (с дистилляцией) — по правилу.")
''')

md(r"""
## 4. Обучение
Одна модель — `python -m chordrec.train --fold 0 --seed 42 --out runs/final`; все 15 — `python -m chordrec.cv --out runs/final`.
Рецепт: 60 эпох × 100 шагов, батч 16 отрезков по 10 с, AdamW (lr 0,001, weight decay 0,01), OneCycle, веса последней эпохи.
Доли в батче: GuitarSet 0,45 · синтетика AAM 0,10 · MusicNet 0,15 · рояль 0,10 · JazzSAMBA 0,15 · ноты MAESTRO 0,05 (только голова нот).
Обучение одной модели — около 35 минут на RTX 4060 Laptop плюс модели-учителя; данные (десятки гигабайт) читаются из
исследовательского проекта, часть наборов по лицензии распространять нельзя.
""")

code(r'''
table(["Слагаемое потери", "Вес", "Где в коде"], [
    ["«Мажор/минор»: CE", "1", line_of(T, 'ce(o["flat"]')],
    ["Корень и бас: CE со сглаживанием по консонансу, α = 0,1", "1 + 1", where(losses.consonance_ce)],
    ["«Набор высот»: BCE по 12 классам", "12", line_of(T, 'o["pc"].float()[m]')],
    ["Ноты: BCE по 88 клавишам", "12", line_of(T, 'o["note"].float()[m]')],
    ["Дистилляция: 3 модели учителя × 3 сдвига", "1", where(losses.distillation_loss)],
    ["Согласие двух транспозиций отрезка", "0,5", where(losses.consistency_loss)],
])
''')

md(r"""
## 5. Пять правил распознавания (без переобучения)
""")

code(r'''
table(["Правило", "Что делает", "Где в коде"], [
    ["(i) K = 3", "вход сдвигается на −1, 0, +1 полутон, ответы возвращаются к исходной тональности и усредняются", where(rules.unshift) + " · " + where(rules.recognize)],
    ["(ii) штраф 96", "Витерби реже меняет аккорд на коротких всплесках", "<code>rules.PENALTY</code> · " + where(decode.viterbi)],
    ["(iii) громкость", "окно 10 с громче или тише [−12,2; +8,5] дБ подтягивается к границе", where(features.clamp_level)],
    ["(iv) тональность → функция", "первый проход → тональность по своим аккордам → второй проход с прибавкой β·W", where(rules.frame_keys) + " · " + where(rules.decode_with_key)],
    ["(v) перекрывающиеся окна", "окна 10 с с шагом 2 с; кадру — окно, где он ближе к центру", where(rules.center_probs)],
])
ladder, summ = load("results/rules/ladder.json"), load("results/rules/final_rows_dev_summary.json")
SETS = [("bpsd", "BPSD"), ("rwcp", "RWC-P"), ("jazz", "Джаз"), ("guitar", "Гитара")]
LADDER = [("Одна модель, окна 10 с", "single"), ("+ (i) K = 3", "i_K3"), ("+ (ii) штраф 96", "ii_pen96"),
          ("+ (iii) громкость", "iii_clamp"), ("+ (iv) тональность → функция", "iv_key"), ("+ (v) перекрывающиеся окна", "v_overlap")]
rows = []
for i, (n, key) in enumerate(LADDER):
    cells = [n]
    for ds, _ in SETS:
        v = ladder[f"{ds}|{key}"]["sevenths"]
        cells.append(f2(v) if i == 0 else f"{f2(v)} ({f2(ladder[f'{ds}|{key}|delta']['sevenths'], True)})")
    rows.append(cells)
table(["Распознавание", "BPSD", "RWC-P", "Джаз", "Гитара"], rows,
      note="sevenths, среднее 15 моделей итогового рецепта; правила добавляются по одному; в скобках — прибавка к предыдущей строке. "
           "Гитара — отложенные группы кросс-валидации. Цена: правило (i) — распознавание × 3, правило (v) — ещё × 5.")
''')

md(r"""
## 6. Итог и сравнение с открытыми системами
Открытые системы — опубликованные веса (MIT) и их собственный код, без дообучения. Все три обучались на RWC-Pop, поэтому
на RWC-P не сравниваются. Обучающие части GuitarSet и JazzSAMBA входят в наши данные: DEV и джаз для нас свой домен,
для открытых систем — нет. BPSD-final и JazzSAMBA-test открыты один раз после заморозки рецепта (раздел 10);
95 % бутстреп-интервалы разниц — в `results/final/report_*.json` и в статье.
""")

code(r'''
OPEN = [("Chord-CNN-LSTM (2019)", "chord_cnn_lstm"), ("BTC (2019)", "btc"), ("ChordMini (2026)", "chordmini")]
openv = {key: load(f"results/open_systems/{key}.json") for _, key in OPEN}


def dev_ours():
    vals = {q: [] for q in LEVELS}
    for k in FOLDS:
        for s in SEEDS:
            lv = load(f"results/models/fold{k}_s{s}.json")["levels"]
            for q in LEVELS:
                src = lv[f"CV-Rmnds-s{s}/flat/viterbi" if q == "majmin" else f"CV-Rmnds-s{s}/struct/viterbi"]["comp"][q]
                vals[q].append(src["wcsr"] if isinstance(src, dict) else src)
    return [float(np.mean(vals[q])) for q in LEVELS]


def five_rules(ds):
    return [summ[f"{ds}|rules"]["mean"][q] for q in LEVELS]


rows = []
for title, key, ours, ours_r in [("GuitarSet DEV (24)", "dev", dev_ours(), None),
                                 ("BPSD-dev без Op. 53 (44)", "bpsd", list(win[LAST]["bpsd"].mean((0, 1))), five_rules("bpsd")),
                                 ("JazzSAMBA-val (12)", "jazz", list(win[LAST]["jazz"].mean((0, 1))), five_rules("jazz"))]:
    rows.append([f"<b>{title}</b>"] + [""] * 6)
    cand = [[openv[k][key]["levels"][q] for q in LEVELS] for _, k in OPEN] + [ours] + ([ours_r] if ours_r else [])
    names = [n for n, _ in OPEN] + ["PiCSFormer, одна модель"] + (["PiCSFormer + 5 правил"] if ours_r else [])
    best = np.max(np.array(cand), 0)
    for n, v in zip(names, cand):
        rows.append([n] + [f"<b>{f2(x)}</b>" if abs(x - b) < 1e-9 else f2(x) for x, b in zip(v, best)])
rows.append(["<b>Только PiCSFormer</b>"] + [""] * 6)
rows.append(["RWC-P (100), PiCSFormer + 5 правил"] + [f2(x) for x in five_rules("rwcp")])
rows.append(["GuitarSet, отложенные группы, PiCSFormer + 5 правил"] + [f2(x) for x in five_rules("guitar")])
table(["Набор и система"] + LEVELS, rows, note="Жирным — лучшее в столбце внутри набора. PiCSFormer — среднее 15 моделей.")
''')

md(r"""
## 7. Согласованность по тональностям (главный вывод статьи)
Звук 11 записей BPSD-dev (по одной на сонату) сдвигается на −4, −1, +1, +4 полутона; ответ сдвигается обратно и сравнивается
с ответом без сдвига. PiCSFormer — без правил, пять моделей (по одной на фолд).
""")

code(r'''
shift, vote, singles = load("results/consistency/audio_shift.json"), load("results/consistency/vote.json"), load("results/consistency/singles.json")


def dis(key):
    a = shift[key]["audio"]
    return [float(np.mean([a[s][i] for s in a])) for i in (0, 1)]


rows = []
for name, key in [("BTC", "btc_teacher"), ("ChordMini, ученик BTC", "btc_cl"), ("ChordMini, ChordNet", "chordnet")]:
    c, e = dis(key)
    rows.append([name, 1, f2(c, nd=1), f2(e, nd=1), f2(vote[key]["K1"]["sevenths"]), f2(vote[key]["K5"]["sevenths"]),
                 f2(vote[key]["K5"]["sevenths"] - vote[key]["K1"]["sevenths"], True)])
c, e = singles["cnnlstm_single_mean"]
rows.append(["Chord-CNN-LSTM, одна модель", 1, f2(c, nd=1), f2(e, nd=1), f2(singles["levels_s0"]["cnnlstm_single_mean"]["sevenths"]), "—", "—"])
c, e = dis("cnnlstm")
rows.append(["Chord-CNN-LSTM, ансамбль 5 моделей", 5, f2(c, nd=1), f2(e, nd=1), f2(vote["cnnlstm"]["K1"]["sevenths"]),
             f2(vote["cnnlstm"]["K5"]["sevenths"]), f2(vote["cnnlstm"]["K5"]["sevenths"] - vote["cnnlstm"]["K1"]["sevenths"], True)])
ours = load("results/consistency/ours_audio_shift.json")  # итоговый рецепт CV-Rmnds
c, e = ours["single_D"]
k1, k5 = ours["single_levels"]["K1"]["sevenths"], ours["single_levels"]["K5"]["sevenths"]
rows.append(["PiCSFormer, одна модель", 1, f2(c, nd=1), f2(e, nd=1), f2(k1), f2(k5), f2(k5 - k1, True)])
c, e = ours["ensemble_D"]
rows.append(["PiCSFormer, ансамбль 5 моделей", 5, f2(c, nd=1), f2(e, nd=1), f2(ours["ensemble_levels"]["K1"]["sevenths"]), "—", "—"])
table(["Система", "моделей", "меняет ответ, грубо, %", "точно, %", "sevenths", "голосование 5 сдвигов", "прирост"], rows,
      note="«Грубо» — основной тон и род аккорда, «точно» — полная метка; среднее по четырём сдвигам. Выборка мала (11 записей), числа ориентировочные.")
''')

md(r"""
## 8. Проверка против исследовательского кода
`tests/check_against_research.py` загружает те же веса в чистый и в исследовательский код. Нормировка входа совпадает побитно,
выходы сети — с расхождением 0, а распознавание с пятью правилами и без них даёт те же аккорды на записях всех четырёх
наборов. С ключом `--full` чистый код пересчитывает всё BPSD-dev для одной модели и сравнивает уровни с сохранёнными.
`tests/check_training.py` обучает модель 20 шагов исследовательским и чистым кодом с одним seed и сравнивает результат.
""")

code(r'''
for name in ("last_check.txt", "last_training_check.txt"):
    p = ROOT / "tests" / name
    print(p.read_text(encoding="utf-8") if p.exists() else f"{name}: отчёт ещё не сохранён")
''')

md(r"""
## 9. Распознавание аккордов прямо здесь
Запись GuitarSet DEV (гитара, джаз, Bb, 35 с), одна модель (фолд 0, seed 42), все пять правил, процессор. Признаки считаются
из аудиофайла заново. Из командной строки то же самое: `python -m chordrec.recognize запись.wav`.
""")

code(r'''
import librosa
from chordrec.config import MANIFESTS, SR
from chordrec.evaluate import levels as eval_levels
from chordrec.features import cqt_from_signal
from chordrec.recognize import load_models

rec = next(json.loads(l) for l in open(MANIFESTS["guitarset"], encoding="utf-8") if '"00_Jazz2-110-Bb_comp"' in l)
y, _ = librosa.load(rec["audio_path"], sr=SR, mono=True)
cpu = torch.device("cpu")
t0 = time.time()
x = cqt_from_signal(y)
res = rules.recognize(load_models(1, cpu), x, len(y) / SR, cpu, rules=True)
print(f"Запись {len(y) / SR:.1f} с, распознана за {time.time() - t0:.1f} с на процессоре")


def ref_at(t):
    return next((s["label"] for s in rec["chord_intervals"] if s["start"] <= t < s["end"]), "N")


import mir_eval

merged = []  # соседние отрезки одного аккорда (различаются только басом) склеиваются для показа
for s in res["chords"]:
    lab = s["label"].split("/")[0]
    if merged and merged[-1]["label"] == lab:
        merged[-1]["end"] = s["end"]
    else:
        merged.append({"start": s["start"], "end": s["end"], "label": lab})
rows = []
for s in merged[:14]:
    r = ref_at((s["start"] + s["end"]) / 2)
    sc = mir_eval.chord.sevenths([r], [s["label"]])[0]
    rows.append([f"{s['start']:.1f}–{s['end']:.1f}", r, s["label"], "✓" if sc == 1 else ("вне словаря" if sc < 0 else "✗")])
table(["Время, с", "Эталон (в середине отрезка)", "Распознано", "верно на уровне sevenths"], rows,
      note="Первые 14 аккордов. Обозначения Harte: C:maj7 — большой мажорный септаккорд, D:min7 — малый минорный, C:7 — доминантсептаккорд; "
           "/1, /5 — бас. «Вне словаря» — эталонный аккорд не входит в словарь уровня sevenths (например, sus2 с септимой) и в оценке не участвует.")
lv = eval_levels([rec], {rec["id"]: res["majmin"]}, {rec["id"]: res["chords"]})
table(LEVELS, [[f2(lv[q]) for q in LEVELS]], note="Уровни на этой записи (одна модель, пять правил).")
''')

md(r"""
## 10. Отложенные наборы
**BPSD-final** (80 записей, 20 сонат) и **JazzSAMBA-test** (14 записей, 7 пьес) открыты один раз после заморозки рецепта:
протокол с SHA-256 весов и скриптов — `results/final/protocol_frozen.json`. Те же 15 моделей и те же три открытые системы;
95 % интервалы — бутстреп по сонатам (пьесам), 10 000 повторов. GuitarSet TEST не используется.
""")

code(r'''
fin = load("results/final/report_final.json")
NAMES = [("cnnlstm", "Chord-CNN-LSTM"), ("btc_teacher", "BTC"), ("btc_cl", "ChordMini"), ("ours_single", "PiCSFormer, одна модель"),
         ("ours_rules", "PiCSFormer + 5 правил")]
rows = []
for ds, title in [("bpsd_final", "BPSD-final (80)"), ("jazz_test", "JazzSAMBA-test (14)")]:
    rows.append([f"<b>{title}</b>"] + [""] * 6)
    lv = fin[f"{ds}|levels"]
    best = np.max(np.array([[lv[k][q] for q in LEVELS] for k, _ in NAMES]), 0)
    for k, n in NAMES:
        rows.append([n] + [f"<b>{f2(lv[k][q])}</b>" if abs(lv[k][q] - b) < 1e-9 else f2(lv[k][q]) for q, b in zip(LEVELS, best)])
table(["Набор и система"] + LEVELS, rows, note="Жирным — лучшее в столбце внутри набора.")
rows = []
for ds, title in [("bpsd_final", "BPSD-final"), ("jazz_test", "JazzSAMBA-test")]:
    for k, n in [("btc_teacher", "BTC"), ("cnnlstm", "Chord-CNN-LSTM")]:
        d = fin[f"{ds}|ours_rules-{k}"]
        rows.append([title, n] + [f"{f2(d[q]['diff'], True)} [{f2(d[q]['ci95'][0])}; {f2(d[q]['ci95'][1])}]" for q in ("majmin", "sevenths", "tetrads")])
table(["Набор", "против", "majmin", "sevenths", "tetrads"], rows, note="PiCSFormer + 5 правил минус открытая система, 95 % интервал.")
''')


# ================================================================ выполнение и запись
class _HTML:
    def __init__(self, data):
        self.data = data


class _Events(io.TextIOBase):
    def __init__(self):
        self.events = []

    def write(self, s):
        if s:
            if self.events and self.events[-1][0] == "text":
                self.events[-1] = ("text", self.events[-1][1] + s)
            else:
                self.events.append(("text", s))
        return len(s)


_CURRENT = {"sink": None}


def _install_fake_ipython():
    disp = types.ModuleType("IPython.display")
    disp.HTML = _HTML
    disp.display = lambda obj: _CURRENT["sink"].events.append(("html", obj.data if isinstance(obj, _HTML) else str(obj)))
    pkg = types.ModuleType("IPython")
    pkg.display = disp
    sys.modules["IPython"], sys.modules["IPython.display"] = pkg, disp


def _lines(text):
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def run_cells():
    import os

    os.chdir(PROJECT)
    ns, results, n = {"__name__": "__main__"}, [], 0
    _install_fake_ipython()
    for kind, src in CELLS:
        if kind == "md":
            results.append((kind, src, None, None))
            continue
        n += 1
        sink = _Events()
        _CURRENT["sink"] = sink
        with contextlib.redirect_stdout(sink):
            exec(compile(src, f"<ячейка {n}>", "exec"), ns)
        results.append((kind, src, sink.events, n))
    return results


def write_ipynb(results, path):
    cells = []
    for i, (kind, src, events, n) in enumerate(results):
        if kind == "md":
            cells.append({"cell_type": "markdown", "id": f"c{i:02d}", "metadata": {}, "source": _lines(src)})
            continue
        outs = [{"output_type": "stream", "name": "stdout", "text": _lines(d)} if t == "text" else
                {"output_type": "display_data", "metadata": {}, "data": {"text/html": _lines(d), "text/plain": ["<HTML>"]}}
                for t, d in events]
        cells.append({"cell_type": "code", "id": f"c{i:02d}", "metadata": {}, "execution_count": n, "source": _lines(src), "outputs": outs})
    nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}}
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def _inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return s


def md_to_html(text):
    out, lines, i = [], text.split("\n"), 0
    item = re.compile(r"^(- |\d+\. )")
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            lv = len(line) - len(line.lstrip("#"))
            out.append(f"<h{lv}>{_inline(line[lv:].strip())}</h{lv}>")
            i += 1
        elif line.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in lines[i:j]]
            t = "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>"
            t += "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows[2:])
            out.append(t + "</tbody></table></div>")
            i = j
        elif item.match(line):
            ordered, items, j = bool(re.match(r"^\d+\. ", line)), [], i
            while j < len(lines) and (item.match(lines[j]) or (lines[j].startswith("  ") and items)):
                if item.match(lines[j]):
                    items.append(item.sub("", lines[j]))
                else:
                    items[-1] += " " + lines[j].strip()
                j += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + f"</{tag}>")
            i = j
        elif not line.strip():
            i += 1
        else:
            j = i
            while j < len(lines) and lines[j].strip() and not re.match(r"^(#|\||- |\d+\. )", lines[j]):
                j += 1
            out.append("<p>" + _inline(" ".join(x.strip() for x in lines[i:j])) + "</p>")
            i = j
    return "\n".join(out)


CSS = """
:root { --bg:#f6f7f9; --fg:#18212b; --muted:#5b6674; --line:#d8dee6; --card:#ffffff; --accent:#2b55c4; --code:#eef1f6; }
@media (prefers-color-scheme: dark) { :root { --bg:#10151c; --fg:#e4e9ef; --muted:#9aa5b2; --line:#2b3542; --card:#161d26;
  --accent:#88a7ff; --code:#1d2530; color-scheme: dark; } }
body { background:var(--bg); color:var(--fg); font:15px/1.55 "Segoe UI", system-ui, sans-serif; margin:0; padding:32px 16px 56px; }
main { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 28px; line-height: 1.2; } h2 { font-size: 21px; margin-top: 36px; border-top: 1px solid var(--line); padding-top: 20px; }
p, li { max-width: 80ch; }
code { background: var(--code); border-radius: 4px; padding: 1px 4px; font: 12.5px Consolas, "Cascadia Mono", monospace; }
pre { background: var(--code); border-radius: 8px; padding: 12px; overflow-x: auto; font: 12.5px/1.45 Consolas, "Cascadia Mono", monospace; }
.scroll, .out-html { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13.5px; background: var(--card); margin: 8px 0; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 9px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: 12.5px; }
details { margin: 10px 0 4px; } summary { cursor: pointer; color: var(--accent); font-size: 13px; }
pre.out { background: transparent; border-left: 3px solid var(--line); border-radius: 0; padding: 4px 10px; }
"""


def write_html(results, path):
    parts = []
    for kind, src, events, n in results:
        if kind == "md":
            parts.append(md_to_html(src))
            continue
        parts.append(f"<details><summary>Код ячейки {n} (Python)</summary><pre>{html.escape(src)}</pre></details>")
        for t, d in events:
            parts.append(f"<pre class='out'>{html.escape(d.rstrip())}</pre>" if t == "text" else f"<div class='out-html'>{d}</div>")
    path.write_text("<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
                    f"<title>Итоговая модель аккордов</title><style>{CSS}</style></head><body><main>" + "\n".join(parts) + "</main></body></html>",
                    encoding="utf-8")


if __name__ == "__main__":
    res = run_cells()
    write_ipynb(res, HERE / "overview.ipynb")
    write_html(res, HERE / "overview.html")
    for kind, src, events, n in res:
        if kind == "code":
            for t, d in events:
                sys.stdout.write(f"--- ячейка {n} ({t})\n" + (d if t == "text" else re.sub(r"<[^>]+>", " ", d)[:1500]) + "\n")
    print("готово:", HERE / "overview.ipynb")
