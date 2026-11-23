"""Проверка: чистый код даёт те же ответы, что исследовательский код, на котором получены числа диссертации.

python tests/check_against_research.py [--full]
1. Нормировка фолдов совпадает побитно.
2. Модель: те же веса → те же выходы на одном входе.
3. Распознавание с пятью правилами и без правил: те же отрезки аккордов на записях BPSD, джаза, попа и гитары.
4. --full: известный ответ — уровни модели фолда 0, seed 42 на всём BPSD-dev совпадают с сохранёнными в исследовательском проекте.
Нужен исследовательский проект рядом (chord-research-lab) и его окружение.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from chordrec import rules  # noqa: E402
from chordrec.config import FOLDS, MANIFESTS, RESEARCH, WEIGHTS  # noqa: E402
from chordrec.evaluate import levels as clean_levels  # noqa: E402
from chordrec.model import load_model  # noqa: E402

for p in ("tools/phase_b", "tools/cycles"):
    sys.path.insert(0, str(RESEARCH / p))
import eval_test as T  # noqa: E402
import phase_f as F  # noqa: E402
from decoder_rows_lvl import apply, per_shift  # noqa: E402
from key_rows import decode_key, lv  # noqa: E402
from overlap_rows import per_shift_center  # noqa: E402
from phase0_rope import windowed  # noqa: E402
import bpsd_eval as E  # noqa: E402
from r_gamma import CACHE, GS, ext_rows  # noqa: E402

OLD = RESEARCH / "results/cycle-mnds/CV-Rmnds/fold0/CV-Rmnds-s42"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def old_model():
    meta = json.loads((OLD / "dev_levels.json").read_text(encoding="utf-8"))
    m = F.build(meta["args"]["heads"], meta["args"]).to(dev)
    m.load_state_dict(torch.load(OLD / "best.pt", map_location=dev, weights_only=True))
    m.eval()
    return m, meta["args"]["heads"], float(meta["chosen_penalty"])


def old_rules(m, heads, pen, r, mean, std):
    x = apply(np.asarray(r["x"]), "clamp")
    ps = per_shift_center(m, r, x, mean, std, dev)
    o3 = E.to_logits({h: np.mean([ps[s][h] for s in (-1, 0, 1)], 0) for h in ps[0]})
    return F.decode_all(o3, heads, pen, r["duration"])["flat"], decode_key(o3, 96.0, r["duration"], 1.0)


def old_plain(m, heads, pen, r, mean, std):
    p = windowed(m, [r], mean, std, dev, 431)[r["id"]]
    d = F.decode_all(E.to_logits(p), heads, pen, r["duration"])
    return d["flat"], d["struct"]


def main(full: bool):
    ok = True
    # 1. нормировка
    st = np.load(WEIGHTS / "stats_fold0.npz")
    mean, std = T.norm_stats(GS, CACHE, None, FOLDS[0])
    same = np.array_equal(st["mean"], mean) and np.array_equal(st["std"], std)
    print("1. нормировка фолда 0 совпадает побитно:", same)
    ok &= same
    # 2. модель
    m_old, heads, pen = old_model()
    m_new = load_model(WEIGHTS / "fold0" / "s42.pt", dev)
    x = torch.randn(1, 431, 252, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    with torch.no_grad():
        a, b = m_old(x), m_new(x)
    diff = max(float((a[h] - b[h]).abs().max()) for h in a)
    print("2. модель: наибольшее расхождение выходов", diff)
    ok &= diff == 0.0
    models = [(m_new, st["mean"], st["std"], json.loads((WEIGHTS / "models.json").read_text(encoding="utf-8"))["fold0/s42"]["chosen_penalty"])]
    # 3. распознавание: по 2 записи каждого набора
    inner = [r for r in T.load_rows(GS, "train", CACHE, None, groups=set(FOLDS[0])) if r["id"].endswith("_comp")]
    sets = {"bpsd": ext_rows(MANIFESTS["bpsd_dev"]), "jazz": ext_rows(MANIFESTS["jazz_val"]), "rwcp": ext_rows(MANIFESTS["rwcp"]), "guitar": inner}
    for name, rows in sets.items():
        for r in rows[:2]:
            fo, so = old_rules(m_old, heads, pen, r, mean, std)
            new = rules.recognize(models, np.asarray(r["x"]), r["duration"], dev, rules=True)
            fp, sp = old_plain(m_old, heads, pen, r, mean, std)
            newp = rules.recognize(models, np.asarray(r["x"]), r["duration"], dev, rules=False)
            eq = fo == new["majmin"] and so == new["chords"] and fp == newp["majmin"] and sp == newp["chords"]
            print(f"3. {name} {r['id']}: правила {len(so)} отрезков, без правил {len(sp)} — совпадают: {eq}", flush=True)
            ok &= eq
    # 4. известный ответ на всём BPSD-dev
    if full:
        bp = sets["bpsd"]
        res = {}
        for mode, flag in (("overlap", True), ("plain", False)):
            fl, ch = {}, {}
            for r in bp:
                out = rules.recognize(models, np.asarray(r["x"]), r["duration"], dev, rules=flag)
                fl[r["id"]], ch[r["id"]] = out["majmin"], out["chords"]
            res[mode] = clean_levels(bp, fl, ch)
        saved = json.loads((RESEARCH / "results/cycle-la/overlap_rows.json").read_text(encoding="utf-8"))["fold0|s42|bpsd|overlap"]
        win = json.loads((RESEARCH / "results/cycle-mnds/CV-Rmnds_windows.json").read_text(encoding="utf-8"))["fold0|s42|bpsd|win10s"]
        for mode, ref in (("overlap", saved), ("plain", win)):
            d = max(abs(res[mode][q] - ref[q]) for q in ref)
            print(f"4. BPSD-dev, {mode}: чистый код {[round(res[mode][q], 2) for q in ref]} · сохранено {[round(ref[q], 2) for q in ref]} · расхождение {d:.2e}")
            ok &= d < 1e-9
    print("ИТОГ:", "всё совпадает" if ok else "ЕСТЬ РАСХОЖДЕНИЯ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main("--full" in sys.argv))
