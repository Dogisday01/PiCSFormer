"""Упаковка весов: 15 моделей итогового рецепта из исследовательского проекта → weights/, плюс статистики нормировки фолдов.

python tools/make_weights.py
Статистики считаются этим проектом (features.norm_stats по обучающим записям GuitarSet фолда); проверка их совпадения
с исследовательским кодом — tests/check_against_research.py.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chordrec.config import CACHE, FOLDS, MANIFESTS, RESEARCH, SEEDS, WEIGHTS  # noqa: E402
from chordrec.features import norm_stats  # noqa: E402

SOURCE = RESEARCH / "results/cycle-mnds/CV-Rmnds"  # итоговый (замороженный) рецепт: ноты MAESTRO + бас из нот + дистилляция


def main():
    WEIGHTS.mkdir(exist_ok=True)
    models = {}
    for k in range(len(FOLDS)):
        (WEIGHTS / f"fold{k}").mkdir(exist_ok=True)
        for s in SEEDS:
            d = SOURCE / f"fold{k}" / f"CV-Rmnds-s{s}"
            shutil.copy2(d / "best.pt", WEIGHTS / f"fold{k}" / f"s{s}.pt")
            meta = json.loads((d / "dev_levels.json").read_text(encoding="utf-8"))
            models[f"fold{k}/s{s}"] = {"chosen_penalty": float(meta["chosen_penalty"]), "params": meta["params"],
                                       "source": f"results/cycle-mnds/CV-Rmnds/fold{k}/CV-Rmnds-s{s}"}
        rows = []
        for line in open(MANIFESTS["guitarset"], encoding="utf-8"):
            r = json.loads(line)
            if r["split"] == "train" and r["group_id"] not in set(FOLDS[k]):
                rows.append({"x": np.load(CACHE / f"{r['audio_sha256'][:24]}-cqt288-h512.npy")})
        mean, std = norm_stats(rows)
        np.savez(WEIGHTS / f"stats_fold{k}.npz", mean=mean, std=std)
        print(f"fold{k}: 3 модели, нормировка по {len(rows)} записям", flush=True)
    (WEIGHTS / "models.json").write_text(json.dumps(models, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
