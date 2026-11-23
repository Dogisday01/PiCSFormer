"""Оценка: взвешенная полнота символов аккордов (WCSR, mir_eval) по уровням — доля времени с верным ответом, %."""
import mir_eval
import numpy as np

METRICS = ("root", "majmin", "thirds", "triads", "sevenths", "tetrads", "mirex")


def evaluate(rows: list, predictions: dict) -> dict:
    """rows — записи с эталоном (id, chord_intervals), predictions — {id: отрезки}.
    Итог по группам: суффикс id (у GuitarSet «comp» — аккомпанемент, «solo» — соло) и «pooled» — все записи вместе."""
    acc = {}
    for row in rows:
        group = row["id"].rsplit("_", 1)[-1]
        ri = np.array([[s["start"], s["end"]] for s in row["chord_intervals"]])
        rl = [s["label"] for s in row["chord_intervals"]]
        segs = predictions[row["id"]]
        ei = np.array([[s["start"], s["end"]] for s in segs])
        el = [s["label"] for s in segs]
        ei, el = mir_eval.util.adjust_intervals(ei, el, ri.min(), ri.max(), "N", "N")
        iv, r, e = mir_eval.util.merge_labeled_intervals(ri, rl, ei, el)
        dur = mir_eval.util.intervals_to_durations(iv)
        for m in METRICS:
            sc = getattr(mir_eval.chord, m)(r, e)
            ok = sc >= 0  # отрицательная оценка — эталон вне словаря уровня, время не считается
            for key in (group, "pooled"):
                a = acc.setdefault(key, {}).setdefault(m, [0.0, 0.0])
                a[0] += float(np.sum(dur[ok] * sc[ok]))
                a[1] += float(np.sum(dur[ok]))
    return {k: {m: 100 * v[0] / v[1] if v[1] else None for m, v in d.items()} for k, d in acc.items()}


def levels(rows: list, majmin_preds: dict, chord_preds: dict, group: str = "comp") -> dict:
    """Шесть уровней таблиц: majmin — по выходу «мажор/минор», остальные — по полным меткам."""
    a, b = evaluate(rows, majmin_preds)[group], evaluate(rows, chord_preds)[group]
    return {q: (a if q == "majmin" else b)[q] for q in ("root", "majmin", "triads", "sevenths", "tetrads", "mirex")}
