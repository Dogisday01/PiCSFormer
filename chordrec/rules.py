"""Пять правил распознавания поверх обученной модели (без переобучения).

(i)   K = 3: вход сдвигается на −1, 0, +1 полутон, ответы возвращаются к исходной тональности и усредняются;
(ii)  штраф смены аккорда 96 в декодере «набора высот»;
(iii) громкость каждого окна 10 с подтягивается к обучающему диапазону (features.clamp_level);
(iv)  «тональность → функция»: первый проход → тональность по своим аккордам → второй проход с прибавкой β·W;
(v)   перекрывающиеся окна 10 с с шагом 2 с; каждому кадру — то окно, где он ближе всего к центру.
Выход «мажор/минор» декодируется со штрафом, выбранным при обучении; полные метки — из «набора высот» с правилами.
"""
import mir_eval
import numpy as np
import torch

from .config import FRAME, SHIFT_BINS, WIN
from .decode import N_TEMPLATES, PCS_Q, decode_flat, decode_pcs
from .features import clamp_level, crop_bins

PENALTY = 96.0   # правило (ii)
BETA = 1.0       # правило (iv)
HOP_W = 86       # правило (v): шаг окон 2 с в кадрах
CENTER = 172     # правило (v): кадру отдаётся окно, у которого впереди от CENTER до CENTER + HOP_W кадров


def probs(model, x: np.ndarray, mean, std, device) -> dict:
    """Вероятности выходов по кадрам: «набор высот» — сигмоида, остальные — softmax. Голова нот не нужна."""
    xt = torch.as_tensor((crop_bins(x, 0) - mean) / std, device=device)[None]
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        o = model(xt)
    return {k: (torch.sigmoid(v[0].float()) if k == "pc" else torch.softmax(v[0].float(), -1)).cpu().numpy() for k, v in o.items()}


def windowed_probs(model, x, mean, std, device, n: int = WIN) -> dict:
    """Без правила (v): окна 10 с без перекрытия, вероятности склеиваются."""
    parts = [probs(model, x[i:i + n], mean, std, device) for i in range(0, len(x), n)]
    return {h: np.concatenate([p[h] for p in parts]) for h in parts[0] if h != "note"}


def center_probs(model, x, mean, std, device) -> dict:
    """Правило (v): окна [e − 431, e) с шагом 86 кадров; кадр получает ответ окна, где он ближе к центру."""
    n = len(x)
    wins = []
    for e in range(HOP_W, n + WIN, HOP_W):
        a, b = max(0, e - WIN), min(n, e)
        if b > a:
            p = probs(model, x[a:b], mean, std, device)
            wins.append((e, a, {h: v for h, v in p.items() if h != "note"}))
    res = {h: np.zeros((n,) + v.shape[1:], np.float32) for h, v in wins[0][2].items()}
    for e, a, p in wins:
        lo, hi = max(0, e - CENTER - HOP_W), min(n, e - CENTER)
        if hi > lo:
            for h in res:
                res[h][lo:hi] = p[h][lo - a:hi - a]
    return res


def unshift(p: dict, k: int) -> dict:
    """Ответы для входа, сдвинутого на k полутонов, → классы исходной записи (класс c был классом c − k)."""
    q = dict(p)
    q["pc"] = np.roll(p["pc"], k, axis=1)
    for h in ("bass", "root"):
        q[h] = np.concatenate([np.roll(p[h][:, :12], k, axis=1), p[h][:, 12:]], 1)
    f = p["flat"]
    q["flat"] = np.concatenate([np.roll(f[:, :12], k, axis=1), np.roll(f[:, 12:24], k, axis=1), f[:, 24:]], 1)
    return q


def to_logits(p: dict) -> dict:
    """Вероятности → вход декодеров: логиты для «набора высот», лог-вероятности для остальных."""
    return {h: (np.log(v + 1e-9) - np.log(1 - v + 1e-9)) if h == "pc" else np.log(v + 1e-9) for h, v in p.items()}


# ---------------------------------------------------------------- правило (iv): тональность → функция
NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# оценка тональности по своим аккордам: вес аккорда (ступень корня от тоники, качество) из учебника гармонии
FUNC_MAJ = {(0, "maj"): 1.0, (7, "maj"): 0.8, (7, "7"): 1.0, (5, "maj"): 0.6, (2, "min"): 0.5, (9, "min"): 0.4,
            (4, "min"): 0.3, (11, "dim"): 0.6, (11, "hdim7"): 0.6, (2, "min7"): 0.5, (0, "maj7"): 0.8, (5, "maj7"): 0.5}
FUNC_MIN = {(0, "min"): 1.0, (7, "maj"): 0.8, (7, "7"): 1.0, (5, "min"): 0.6, (2, "dim"): 0.5, (2, "hdim7"): 0.5,
            (8, "maj"): 0.4, (3, "maj"): 0.3, (11, "dim"): 0.6, (11, "dim7"): 1.0, (0, "min7"): 0.6}
OFF = -0.3       # аккорд вне тональности
SWITCH = 6.0     # штраф смены тональности (вес × секунда)
# прибавка W(тональность, ступень корня, качество) к оценке шаблона во втором проходе — диатонические функции
W_MAJ = {0: {"maj": 1.0, "maj7": 1.0, "maj6": 1.0, "sus4": 0.6, "sus2": 0.6, "7": 0.4},
         2: {"min": 0.5, "min7": 0.5, "7": 0.4, "dim7": 0.0},
         4: {"min": 0.3, "min7": 0.3, "7": 0.4},
         5: {"maj": 0.6, "maj7": 0.6, "maj6": 0.6, "min": 0.2, "min6": 0.2},
         7: {"maj": 0.9, "7": 0.9, "sus4": 0.5, "sus4(b7)": 0.5},
         9: {"min": 0.4, "min7": 0.4, "7": 0.4},
         11: {"dim": 0.6, "hdim7": 0.6, "dim7": 0.6, "7": 0.4},
         1: {"maj": 0.2, "dim7": 0.4}, 6: {"dim7": 0.4, "hdim7": 0.3}, 8: {"dim7": 0.4, "maj": 0.2}}
W_MIN = {0: {"min": 1.0, "min7": 0.6, "minmaj7": 0.6, "min6": 0.6, "7": 0.4},
         2: {"dim": 0.5, "hdim7": 0.5, "7": 0.4},
         3: {"maj": 0.3, "maj7": 0.3},
         5: {"min": 0.6, "min7": 0.6, "min6": 0.6, "maj": 0.2},
         7: {"maj": 0.9, "7": 0.9, "sus4": 0.5, "sus4(b7)": 0.5},
         8: {"maj": 0.4, "maj7": 0.4},
         10: {"maj": 0.3, "7": 0.3},
         11: {"dim": 0.8, "dim7": 0.8, "hdim7": 0.3},
         1: {"maj": 0.2}, 6: {"dim7": 0.4}}
WTAB = np.zeros((24, N_TEMPLATES + 1))  # тональности 0–11 мажор, 12–23 минор
for _k in range(24):
    _t, _tab = _k % 12, (W_MIN if _k >= 12 else W_MAJ)
    for _r in range(12):
        for _qi, _q in enumerate(PCS_Q):
            WTAB[_k, _r * len(PCS_Q) + _qi] = _tab.get((_r - _t) % 12, {}).get(_q, 0.0)


def _chord_parts(label):
    """(корень, качество) метки или None для «нет аккорда»."""
    if label in ("N", "X", "", None) or label[0] not in NAMES:
        return None
    try:
        rt, _, _ = mir_eval.chord.encode(label, reduce_extended_chords=True)
    except Exception:  # noqa: BLE001
        return None
    q = label.split(":")[1].split("/")[0].split("(")[0] if ":" in label else "maj"
    return int(rt), q


def key_scores(segs: list) -> np.ndarray:
    """Очки 24 тональностей для каждого отрезка: длительность × вес функции аккорда в тональности."""
    out = np.zeros((len(segs), 24))
    for i, s in enumerate(segs):
        p = _chord_parts(s["label"])
        if p is None:
            continue
        rt, q = p
        for k in range(24):
            table = FUNC_MIN if k >= 12 else FUNC_MAJ
            qq = {"maj6": "maj", "min6": "min", "9": "7", "sus4": "maj", "sus2": "maj"}.get(q, q)
            out[i, k] = (s["end"] - s["start"]) * table.get(((rt - k % 12) % 12, qq), OFF)
    return out


def viterbi_keys(sc: np.ndarray) -> list:
    """Тональность каждого отрезка: Витерби по 24 тональностям со штрафом смены SWITCH."""
    n = len(sc)
    if n == 0:
        return []
    acc, back = sc[0].copy(), np.zeros((n, 24), int)
    for i in range(1, n):
        best, arg = acc.max(), int(acc.argmax())
        keep = acc >= best - SWITCH
        back[i] = np.where(keep, np.arange(24), arg)
        acc = np.where(keep, acc, best - SWITCH) + sc[i]
    path = [int(acc.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return path[::-1]


def frame_keys(segs: list, n: int) -> np.ndarray:
    """Тональность по кадрам из отрезков первого прохода."""
    k = np.zeros(n, int)
    for s, kk in zip(segs, viterbi_keys(key_scores(segs))):
        k[int(s["start"] / FRAME):int(s["end"] / FRAME) + 1] = kk
    return k


def decode_with_key(o: dict, penalty: float, duration: float, beta: float = BETA) -> list:
    """Правило (iv): два прохода декодера «набора высот»."""
    first = decode_pcs(o, penalty, duration)
    if not beta:
        return first
    keys = frame_keys(first, len(o["pc"]))
    return decode_pcs(o, penalty, duration, bonus=beta * WTAB[keys])


# ---------------------------------------------------------------- распознавание целиком
def recognize(models: list, x: np.ndarray, duration: float, device, rules: bool = True) -> dict:
    """models — список (модель, mean, std, штраф при обучении). Несколько моделей — вероятности усредняются.
    rules=True — все пять правил; rules=False — одна модель как есть: окна 10 с без перекрытия, K = 1, штраф при обучении.
    Возвращает {"majmin": отрезки мажор/минор, "chords": отрезки полных аккордов}."""
    acc = []
    for model, mean, std, _ in models:
        if rules:
            xc = clamp_level(np.asarray(x))                                          # (iii)
            per = {s: unshift(center_probs(model, np.roll(xc, -SHIFT_BINS * s, axis=1), mean, std, device), s)
                   for s in (-1, 0, 1)}                                              # (i) и (v)
            acc.append({h: np.mean([per[s][h] for s in (-1, 0, 1)], 0) for h in per[0]})
        else:
            acc.append(windowed_probs(model, np.asarray(x), mean, std, device))
    p = acc[0] if len(acc) == 1 else {h: np.mean([a[h] for a in acc], 0) for h in acc[0]}
    o = to_logits(p)
    flat_pen = float(np.mean([m[3] for m in models]))
    majmin = decode_flat(o["flat"], flat_pen, duration)
    chords = decode_with_key(o, PENALTY, duration) if rules else decode_pcs(o, flat_pen, duration)  # (ii), (iv)
    return {"majmin": majmin, "chords": chords}
