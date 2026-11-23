"""Проверка обучения: короткий прогон (1 эпоха × 20 шагов, фолд 0, seed 42) исследовательским и чистым кодом.

python tests/check_training.py OUT_DIR
Сравниваются потеря эпохи, выбранный штраф декодера, уровни на DEV и веса после обучения. Побитного совпадения весов
ждать не стоит: обратный проход на видеокарте недетерминирован, поэтому сравнение — с допуском.
"""
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from chordrec.config import FOLDS, RESEARCH  # noqa: E402
from chordrec.train import train  # noqa: E402

TEACHERS = "results/cycle-maestro-notes/CV-Rmen/{fold}/CV-Rmen-s{seed}/best.pt"
OLD_ARGS = ["--aam", "600", "--aam-frac", "0.10", "--select", "balanced", "--heads", "pcsrn", "--rope", "1",
            "--mn-manifest", "results/cycle-st3/musicnet_train_st3.jsonl", "--mn-frac", "0.15",
            "--rw-manifest", "results/side-data/labels-refine-20261029/manifests/piano_c7r.jsonl", "--rw-frac", "0.10",
            "--extra-pool", "materials/research/manifests/jazzsamba_train_c6.jsonl:0.15", "--note-pools", "guitarset,musicnet,maestro",
            "--note-good", "results/phase0-next/notes_all.json", "--last-epoch", "1", "--consist", "0.5", "--cons", "0.1",
            "--bass-notes", "1", "--distill", TEACHERS, "--distill-w", "1.0",
            "--extra-pool", "results/cycle-maestro-notes/maestro_notes.jsonl:0.05"]


def main(out):
    out = Path(out)
    old_dir = out / "research" / "fold0"
    cmd = [sys.executable, str(RESEARCH / "tools/phase_b/phase_f.py"), "--tag", "CHK", "--out", str(old_dir), "--seeds", "42",
           "--inner-groups", ",".join(FOLDS[0]), "--epochs", "1", "--steps", "20"] + OLD_ARGS
    subprocess.run(cmd, check=True, cwd=RESEARCH)
    old = json.loads((old_dir / "CHK-s42" / "dev_levels.json").read_text(encoding="utf-8"))
    new = train(0, 42, out / "clean", TEACHERS, "cuda", epochs=1, steps=20, resume=False)
    print("потеря эпохи:     исследовательский", round(old["history"][-1]["loss"], 6), "· чистый", round(new["history"][-1]["loss"], 6))
    print("мера на отложенных группах:", round(old["history"][-1]["inner_score"], 4), "·", round(new["history"][-1]["inner_score"], 4))
    print("выбранный штраф:  ", old["chosen_penalty"], "·", new["chosen_penalty"])
    for kind in ("flat", "struct"):
        a = old["levels"][f"CHK-s42/{kind}/viterbi"]["comp"]
        b = new["dev_levels"][f"{kind}/viterbi"]["comp"]
        print(f"DEV {kind}:", {q: (round(a[q], 2), round(b[q], 2)) for q in ("root", "majmin", "sevenths")})
    wa = torch.load(old_dir / "CHK-s42" / "best.pt", map_location="cpu", weights_only=True)
    wb = torch.load(out / "clean" / "fold0" / "s42" / "best.pt", map_location="cpu", weights_only=True)
    diff = max(float((wa[k].float() - wb[k].float()).abs().max()) for k in wa)
    print("веса после обучения: наибольшее расхождение", diff, "· ключи совпадают:", set(wa) == set(wb))


if __name__ == "__main__":
    main(sys.argv[1])
