"""Константы итогового рецепта и пути к данным.

Данные (аудио, признаки, манифесты) в этот проект не копируются: по умолчанию они читаются из исследовательского
проекта `chord-research-lab`, лежащего рядом. Другой путь задаётся переменной окружения CHORD_RESEARCH_ROOT.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- признаки (CQT)
SR = 22050                 # частота дискретизации, Гц
HOP = 512                  # шаг кадра, отсчётов
FRAME = HOP / SR           # длительность кадра, с (≈ 23 мс)
BINS_PER_OCT = 36          # 3 бина на полутон
N_BINS_FULL = 288          # полный CQT: 8 октав от F#0
N_BINS = 252               # сколько бинов читает сеть
SHIFT_BINS = 3             # бинов на полутон
MARGIN = 18                # запас по краям: ±6 полутонов для транспозиций
SHIFTS = list(range(-5, 7))  # сдвиги окна при обучении, полутоны
WIN = 431                  # окно 10 с в кадрах (обучение и распознавание)

# ---------------------------------------------------------------- метки
ROOTS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT = [f"{r}:maj" for r in ROOTS] + [f"{r}:min" for r in ROOTS] + ["N"]  # выход «мажор/минор»: 24 класса + «нет аккорда»
IGNORE = -100              # метка неизвестна: кадр не участвует в потере
LEVELS = ("root", "majmin", "triads", "sevenths", "tetrads", "mirex")  # уровни mir_eval в таблицах

# ---------------------------------------------------------------- кросс-валидация: 5 фолдов по 2 группы GuitarSet
FOLDS = [
    ("guitarset:Jazz3", "guitarset:Funk2"),
    ("guitarset:BN1", "guitarset:Rock2"),
    ("guitarset:BN2", "guitarset:SS1"),
    ("guitarset:BN3", "guitarset:Rock3"),
    ("guitarset:Funk1", "guitarset:Jazz1"),
]
SEEDS = (42, 43, 44)

# ---------------------------------------------------------------- пути
PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = Path(os.environ.get("CHORD_RESEARCH_ROOT", PROJECT.parent / "chord-research-lab"))
CACHE = RESEARCH / "cache/phase_b"                       # готовые CQT-признаки (.npy)
WEIGHTS = PROJECT / "weights"                            # веса 15 моделей итогового рецепта и статистики нормировки
GS_ANNOTATION = Path(os.environ.get(                     # разметка GuitarSet (JAMS): ноты для головы нот
    "GUITARSET_ANNOTATION", str(RESEARCH / "materials/guitarset/annotation")))
MN_LABELS = RESEARCH / "materials/research/archives/musicnet_labels/train_labels"  # ноты MusicNet (csv)

MANIFESTS = {  # манифесты данных: одна запись — одна строка JSON
    "guitarset": RESEARCH / "materials/research/manifests/guitarset.jsonl",
    "aam": RESEARCH / "materials/research/manifests/aam.jsonl",
    "musicnet": RESEARCH / "results/cycle-st3/musicnet_train_st3.jsonl",          # MusicNet + подписи самообучения (круг 2)
    "piano": RESEARCH / "results/side-data/labels-refine-20261029/manifests/piano_c7r.jsonl",  # рояль Disklavier
    "jazz": RESEARCH / "materials/research/manifests/jazzsamba_train_c6.jsonl",   # JazzSAMBA, обучающая часть
    "maestro_notes": RESEARCH / "results/cycle-maestro-notes/maestro_notes.jsonl",  # ноты MAESTRO для головы нот
    "bpsd_dev": RESEARCH / "results/side-bpsd/no-op053/dev11_minus_op053.jsonl",  # BPSD-dev без Op. 53
    "jazz_val": RESEARCH / "materials/research/manifests/jazzsamba_val_c6_clean.jsonl",
    "rwcp": RESEARCH / "materials/research/manifests/rwc_p_c4.jsonl",
}
NOTE_GOOD = RESEARCH / "results/phase0-next/notes_all.json"  # записи MusicNet, у которых ноты прошли проверку
