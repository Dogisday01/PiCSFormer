"""Декодеры: Витерби со штрафом смены аккорда; выход «мажор/минор»; «набор высот» по 180 шаблонам (12 корней × 15 качеств)."""
import numpy as np

from .config import FLAT, FRAME, ROOTS

# ---------------------------------------------------------------- шаблоны аккордов: интервалы от корня в полутонах
PCS_VOCAB = {"maj": [0, 4, 7], "min": [0, 3, 7], "dim": [0, 3, 6], "aug": [0, 4, 8], "sus4": [0, 5, 7], "sus2": [0, 2, 7],
             "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10], "hdim7": [0, 3, 6, 10], "dim7": [0, 3, 6, 9],
             "maj6": [0, 4, 7, 9], "min6": [0, 3, 7, 9], "sus4(b7)": [0, 5, 7, 10], "minmaj7": [0, 3, 7, 11]}
PCS_Q = list(PCS_VOCAB)
N_TEMPLATES = 12 * len(PCS_Q)                      # 180; номер шаблона = корень · 15 + качество, последний столбец — «нет аккорда»
PCS_T = np.zeros((N_TEMPLATES, 12), np.float32)    # PCS_T[шаблон, класс высоты] = 1, если звук входит в аккорд
for _r in range(12):
    for _qi, _q in enumerate(PCS_Q):
        PCS_T[_r * len(PCS_Q) + _qi, [(_r + i) % 12 for i in PCS_VOCAB[_q]]] = 1
IVN = {0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4", 6: "b5", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7"}  # обозначение баса
ROOT_GAMMA = 2.0  # вес головы корня в оценке шаблона (выбран заранее, не подбирается)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def viterbi(emission: np.ndarray, penalty: float) -> np.ndarray:
    """Лучший путь по кадрам: сумма оценок минус penalty за каждую смену класса."""
    count, classes = emission.shape
    scores = emission[0].copy()
    back = np.empty((count, classes), dtype=np.int32)
    for t in range(1, count):
        best = scores.argmax()
        jump = scores[best] - penalty
        choose = scores >= jump
        back[t] = np.where(choose, np.arange(classes), best)
        scores = np.where(choose, scores, jump) + emission[t]
    path = np.empty(count, dtype=np.int64)
    path[-1] = scores.argmax()
    for t in range(count - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def to_segments(labels: list, duration: float) -> list:
    """Метки по кадрам → отрезки [{start, end, label}]."""
    segs, start = [], 0.0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[i - 1]:
            end = min(i * FRAME, duration) if i < len(labels) else duration
            if end > start:
                segs.append({"start": start, "end": end, "label": labels[i - 1]})
            start = end
    return segs


def decode_flat(flat: np.ndarray, penalty: float, duration: float) -> list:
    """Выход «мажор/минор» (логиты или лог-вероятности, кадры × 25) → отрезки C:maj … B:min, N."""
    em = log_softmax(flat)
    path = viterbi(em, penalty) if penalty > 0 else em.argmax(1)
    return to_segments([FLAT[i] for i in path], duration)


def pcs_emissions(o: dict) -> np.ndarray:
    """Оценка каждого шаблона в каждом кадре (кадры × 181).

    s(r, q) = Σ_c [T(c)·log p(c) + (1 − T(c))·log(1 − p(c))] + log Σ_{c ∈ аккорд} P(бас = c) + 0,1·log P(бас = r) + γ·log P(корень = r);
    s(N)    = Σ_c log(1 − p(c)) + log P(бас = нет) + γ·log P(корень = нет).
    p(c) — «звучит ли класс высоты c как звук аккорда» (выход pc), T — шаблон.
    """
    z = np.asarray(o["pc"], np.float64)
    lp, lq = -np.logaddexp(0, -z), -np.logaddexp(0, z)  # log σ(z), log(1 − σ(z))
    lb = log_softmax(np.asarray(o["bass"], np.float64))
    pb = np.exp(lb[:, :12])
    bass_in = np.log(np.maximum(pb @ PCS_T.T, 1e-12))   # бас — любой звук аккорда (обращения допустимы)
    roots = np.repeat(np.arange(12), len(PCS_Q))
    chord = lp @ PCS_T.T + lq @ (1 - PCS_T).T + bass_in + 0.1 * lb[:, roots]  # 0,1·log P(бас = корень) разрешает ничьи dim7, aug
    em = np.concatenate([chord, lq.sum(1, keepdims=True) + lb[:, 12:13]], 1)
    lr = log_softmax(np.asarray(o["root"], np.float64))
    return em + ROOT_GAMMA * np.concatenate([np.repeat(lr[:, :12], len(PCS_Q), 1), lr[:, 12:13]], 1)


def decode_pcs(o: dict, penalty: float, duration: float, bonus: np.ndarray = None) -> list:
    """«Набор высот» + бас + корень → полные метки аккордов (C:maj7, A:min/b3 …). bonus — прибавка к оценкам
    шаблонов той же формы (правило «тональность → функция»). Обращение — если самый вероятный бас — звук аккорда, отличный от корня."""
    em = pcs_emissions(o)
    if bonus is not None:
        em = em + bonus
    path = viterbi(em, penalty) if penalty > 0 else em.argmax(1)
    bass = log_softmax(np.asarray(o["bass"], np.float64))[:, :12].argmax(1)
    labels = []
    for t, k in enumerate(path):
        if k == N_TEMPLATES:
            labels.append("N")
            continue
        r, q = divmod(int(k), len(PCS_Q))
        lab = f"{ROOTS[r]}:{PCS_Q[q]}"
        if bass[t] != r and PCS_T[k, bass[t]] > 0:
            lab += "/" + IVN[(bass[t] - r) % 12]
        labels.append(lab)
    return to_segments(labels, duration)
