"""Распознать аккорды в аудиофайле итоговой моделью со всеми пятью правилами.

python -m chordrec.recognize песня.wav                 # одна модель (фолд 0, seed 42), процессор
python -m chordrec.recognize песня.wav --models 5      # среднее 5 моделей (по одной на фолд)
python -m chordrec.recognize песня.wav --no-rules      # одна модель как есть, без правил распознавания
"""
import argparse
import json

import librosa
import numpy as np
import torch

from .config import FOLDS, SEEDS, SR, WEIGHTS
from .features import cqt_from_signal
from .model import load_model
from .rules import recognize


def load_models(count: int = 1, device="cpu") -> list:
    """Модели итогового рецепта из weights/: count = 1 — фолд 0 seed 42; 5 — по одной на фолд (seed 42); 15 — все.
    Каждая модель — с нормировкой своего фолда и штрафом декодера, выбранным при её обучении."""
    pairs = [(k, s) for s in SEEDS for k in range(len(FOLDS))][:count] if count > 1 else [(0, 42)]
    meta = json.loads((WEIGHTS / "models.json").read_text(encoding="utf-8"))
    out = []
    for k, s in pairs:
        st = np.load(WEIGHTS / f"stats_fold{k}.npz")
        out.append((load_model(WEIGHTS / f"fold{k}" / f"s{s}.pt", device), st["mean"], st["std"], meta[f"fold{k}/s{s}"]["chosen_penalty"]))
    return out


def format_segments(segs: list) -> str:
    return "\n".join(f"{s['start']:7.2f}–{s['end']:7.2f} с  {s['label']}" for s in segs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--models", type=int, default=1, choices=(1, 5, 15))
    ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", help="сохранить отрезки в файл JSON")
    a = ap.parse_args()
    device = torch.device(a.device)
    y, _ = librosa.load(a.audio, sr=SR, mono=True)
    x, duration = cqt_from_signal(y), len(y) / SR
    res = recognize(load_models(a.models, device), x, duration, device, rules=not a.no_rules)
    print(format_segments(res["chords"]))
    if a.json:
        json.dump(res, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
