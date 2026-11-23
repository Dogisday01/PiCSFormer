"""Распознавание для сайта: те же шаги, что chordrec.rules.recognize (аккорды совпадают побитно, см. tests/check_web.py),
плюс ход работы по шагам, уверенность каждого отрезка и тональность, которую оценивает правило (iv)."""
import mir_eval
import numpy as np

from ..config import FRAME, ROOTS, SHIFT_BINS
from ..decode import decode_flat, decode_pcs
from ..features import clamp_level
from ..rules import PENALTY, center_probs, decode_with_key, frame_keys, to_logits, unshift, windowed_probs

KEYS = [(r, "major") for r in ROOTS] + [(r, "minor") for r in ROOTS]  # номера тональностей правила (iv): 0–11 мажор, 12–23 минор


def flat_index(label: str) -> int:
    """Класс выхода «мажор/минор» для метки: 0–11 мажор, 12–23 минор, 24 — нет аккорда."""
    if label in ("N", "X"):
        return 24
    root, bits, _ = mir_eval.chord.encode(label)
    return int(root) + (12 if bits[3] and not bits[4] else 0)


def key_segments(frame_key: np.ndarray, duration: float) -> list:
    out = []
    start = 0
    for i in range(1, len(frame_key) + 1):
        if i == len(frame_key) or frame_key[i] != frame_key[start]:
            tonic, mode = KEYS[int(frame_key[start])]
            out.append({"start": round(start * FRAME, 3), "end": round(min(i * FRAME, duration), 3), "tonic": tonic, "mode": mode})
            start = i
    return out


def analyze(models: list, x: np.ndarray, duration: float, device, rules: bool = True, progress=None) -> dict:
    """models — как у chordrec.recognize.load_models. progress(stage, done, total) — ход работы."""
    progress = progress or (lambda *a: None)
    total = len(models) * (3 if rules else 1)
    done = 0
    acc = []
    for model, mean, std, _ in models:
        if rules:
            xc = clamp_level(np.asarray(x))                                          # (iii)
            per = {}
            for s in (-1, 0, 1):                                                     # (i) и (v)
                per[s] = unshift(center_probs(model, np.roll(xc, -SHIFT_BINS * s, axis=1), mean, std, device), s)
                done += 1
                progress("model", done, total)
            acc.append({h: np.mean([per[s][h] for s in (-1, 0, 1)], 0) for h in per[0]})
        else:
            acc.append(windowed_probs(model, np.asarray(x), mean, std, device))
            done += 1
            progress("model", done, total)
    p = acc[0] if len(acc) == 1 else {h: np.mean([a[h] for a in acc], 0) for h in acc[0]}
    o = to_logits(p)
    progress("decode", 0, 1)
    flat_pen = float(np.mean([m[3] for m in models]))
    majmin = decode_flat(o["flat"], flat_pen, duration)
    chords = decode_with_key(o, PENALTY, duration) if rules else decode_pcs(o, flat_pen, duration)  # (ii), (iv)
    keys = key_segments(frame_keys(decode_pcs(o, PENALTY, duration), len(o["pc"])), duration) if rules else []
    pf = p["flat"]
    for seg in chords:  # уверенность: средняя вероятность выхода «мажор/минор» для рода этого аккорда на его кадрах
        a = int(seg["start"] / FRAME)
        b = max(a + 1, int(seg["end"] / FRAME))
        seg["confidence"] = round(float(pf[a:b, flat_index(seg["label"])].mean()), 4) if a < len(pf) else None
    progress("decode", 1, 1)
    return {"chords": chords, "majmin": majmin, "keys": keys}
