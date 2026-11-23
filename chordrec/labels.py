"""Цели обучения по кадрам: «мажор/минор» и корень, «набор высот» и бас, ноты; их сдвиг вместе с окном CQT."""
import json

import mir_eval
import numpy as np

from .config import FRAME, GS_ANNOTATION, IGNORE, MN_LABELS


def parse_label(label: str):
    """Метка Harte → (корень 0–11 или 12 = нет аккорда | None, класс «мажор/минор» 0–24 | None).
    None — метка не разбирается или аккорд не сводится к мажору/минору (кадр не учит этот выход)."""
    if label == "N":
        return 12, 24
    try:
        root, b, _ = mir_eval.chord.encode(label)
    except Exception:  # noqa: BLE001
        return None, None
    if root < 0:
        return 12, 24
    third = b[3] or b[4]
    if b[4] and b[7]:
        q = 0                                   # мажор
    elif b[3] and b[7]:
        q = 1                                   # минор
    elif (b[3] and b[6]) or (b[4] and b[8]) or (b[2] and b[7] and not third) or (b[5] and b[7] and not third) \
            or (b[7] and not third and not b[2] and not b[5]):
        q = None                                # dim, aug, sus2, sus4, квинта — не мажор и не минор
    elif b[4]:
        q = 0
    elif b[3]:
        q = 1
    else:
        q = None
    flat = root if q == 0 else (12 + root if q == 1 else None)
    return root, flat


def frame_targets(intervals: list, n: int) -> np.ndarray:
    """(кадры, 2): столбец 0 — корень (12 = нет аккорда), столбец 1 — класс «мажор/минор»; IGNORE — неизвестно."""
    t = (np.arange(n) + 0.5) * FRAME
    out = np.full((n, 2), IGNORE, dtype=np.int64)
    for item in intervals:
        root, flat = parse_label(item["label"])
        if root is None:
            continue
        sel = (t >= item["start"]) & (t < item["end"])
        out[sel, 0] = root
        out[sel, 1] = flat if flat is not None else IGNORE
    return out


def shift_targets(y: np.ndarray, k: int) -> np.ndarray:
    """Окно CQT выше на k полутонов = звук ниже на k: корень и класс «мажор/минор» сдвигаются на −k."""
    y = y.copy()
    root, flat = y[:, 0], y[:, 1]
    ok = (root >= 0) & (root < 12)
    root[ok] = (root[ok] - k) % 12
    maj, mnr = (flat >= 0) & (flat < 12), (flat >= 12) & (flat < 24)
    flat[maj] = (flat[maj] - k) % 12
    flat[mnr] = 12 + (flat[mnr] - 12 - k) % 12
    return y


def pcs_targets(intervals: list, n: int, triad_only: bool = False) -> np.ndarray:
    """(кадры, 13) int8: столбцы 0–11 — класс высоты звучит как звук аккорда (1 / 0, −1 — неизвестно), столбец 12 — бас
    (0–11, 12 = нет аккорда, −1 — неизвестно). X → неизвестно. triad_only (синтетика AAM: размечены только трезвучия) —
    классы вне трезвучия неизвестны."""
    t = (np.arange(n) + 0.5) * FRAME
    out = np.full((n, 13), -1, np.int8)
    for it in intervals:
        lab = {"N.C.": "N"}.get(it["label"], it["label"])
        if lab == "X":
            continue
        try:
            root, bits, bass = mir_eval.chord.encode(lab)
        except Exception:  # noqa: BLE001
            continue
        sel = (t >= it["start"]) & (t < it["end"])
        if root < 0:
            out[sel, :12] = 0
            out[sel, 12] = 12
            continue
        pcs = np.roll(np.asarray(bits, int), root)
        out[sel, :12] = np.where(pcs > 0, 1, -1 if triad_only else 0).astype(np.int8)
        out[sel, 12] = (root + int(bass)) % 12
    return out


def shift_pcs(p: np.ndarray, k: int) -> np.ndarray:
    """«Набор высот» и бас при окне выше на k полутонов: класс c → (c − k) mod 12."""
    q = p.copy()
    q[:, :12] = np.roll(p[:, :12], -k, axis=1)
    b = q[:, 12]
    ok = (b >= 0) & (b < 12)
    b[ok] = (b[ok] - k) % 12
    return q


def aam_intervals(intervals: list) -> list:
    """Синтетика AAM: N.C. → N, X выбрасывается; остаётся задуманный генератором аккорд."""
    out = []
    for i in intervals:
        if i["label"] in ("N.C.", "N"):
            out.append({**i, "label": "N"})
        elif i["label"] != "X":
            out.append(i)
    return out


def note_targets(record_id: str, n: int):
    """(кадры, 88) int8 — звучащие ноты A0…C8. GuitarSet: JAMS (note_midi), MusicNet: csv (отсчёты 44,1 кГц). None — нот нет."""
    if record_id.startswith("musicnet:"):
        path = MN_LABELS / f"{record_id.split(':')[1]}.csv"
        if not path.exists():
            return None
        lines = path.read_text(encoding="utf-8").splitlines()[1:]
        notes = [(int(c[0]) / 44100, int(c[1]) / 44100, int(c[3])) for c in (line.split(",") for line in lines)]
    else:
        path = GS_ANNOTATION / f"{record_id.split(':')[-1]}.jams"
        if not path.exists():
            return None
        notes = []
        for a in json.loads(path.read_text(encoding="utf-8"))["annotations"]:
            if a["namespace"] == "note_midi":
                notes += [(e["time"], e["time"] + e["duration"], int(round(e["value"]))) for e in a["data"]]
    out = np.zeros((n, 88), np.int8)
    for start, end, pitch in notes:
        if 21 <= pitch <= 108:
            out[int(start / FRAME):max(int(end / FRAME), int(start / FRAME) + 1), pitch - 21] = 1
    return out


def bass_from_notes(p: np.ndarray, notes: np.ndarray, intervals: list) -> float:
    """Бас из нот: бас метки заменяется самой частой нижней звучащей нотой отрезка аккорда, если она — звук этого аккорда.
    Меняет p на месте; возвращает долю кадров с изменённым басом. Только для обучения: оценка — по исходным меткам."""
    if p is None or notes is None:
        return 0.0
    t = (np.arange(len(p)) + 0.5) * FRAME
    act = np.asarray(notes[:len(p)]) > 0
    low_pc = np.where(act.any(1), (act.argmax(1) + 21) % 12, -1)  # индекс 0 = A0 (MIDI 21)
    changed = 0
    for it in intervals:
        sel = (t >= it["start"]) & (t < it["end"]) & (p[:, 12] >= 0) & (p[:, 12] < 12)
        if not sel.any():
            continue
        lp = low_pc[sel]
        lp = lp[lp >= 0]
        if not len(lp):
            continue
        mode = int(np.bincount(lp, minlength=12).argmax())
        rows = np.where(sel)[0]
        if (p[rows, mode] == 1).all() and (p[rows, 12] != mode).any():
            changed += int((p[rows, 12] != mode).sum())
            p[rows, 12] = mode
    return changed / max(int(((p[:, 12] >= 0) & (p[:, 12] < 12)).sum()), 1)
