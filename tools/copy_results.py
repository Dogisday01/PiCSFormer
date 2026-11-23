"""Копия сохранённых чисел исследовательского проекта, на которых строятся таблицы ноутбука (results/), с указанием источника.

python tools/copy_results.py
Числа не пересчитываются: это те же файлы, что лежат в основе таблиц статьи. Источник каждого — results/SOURCES.json.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chordrec.config import PROJECT, RESEARCH, SEEDS  # noqa: E402

R = RESEARCH / "results"
FILES = {  # файл здесь ← файл исследовательского проекта
    "path/00_base.json": "phase0-arch/rope.json",
    "path/01_rope.json": "cycle-arch/CV-Rrope_windows.json",
    "path/02_last_epoch.json": "cycle-next/CV-RL_windows.json",
    "path/03_note_head.json": "cycle-notes2/CV-Rn2_windows.json",
    "path/04_self_training.json": "cycle-st/CV-Rst_windows.json",
    "path/05_round2_consistency.json": "cycle-combo/CV-Rc_windows.json",
    "path/06_consonance.json": "cycle-cons/CV-Rcons_windows.json",
    "path/07_distillation.json": "cycle-dist/CV-Rds_windows.json",
    "path/08_bass_from_notes.json": "cycle-bnds/CV-Rbds_windows.json",
    "path/09_maestro_notes.json": "cycle-mnds/CV-Rmnds_windows.json",
    "rules/final_rows_dev.json": "final-freeze/dev_CV-Rmnds/levels.json",
    "rules/final_rows_dev_summary.json": "final-freeze/dev_CV-Rmnds/summary.json",
    "rules/ladder.json": "final-freeze/ladder_CV-Rmnds_summary.json",
    "rules/single_whole_records.json": "cycle-mnds/inv_CV-Rmnds.json",
    "final/report_dev.json": "final-freeze/dev_CV-Rmnds/report.json",
    "final/report_final.json": "final-freeze/final_CV-Rmnds/report.json",
    "final/protocol_frozen.json": "final-protocol/protocol_frozen.json",
    "open_systems/btc.json": "open-baselines/levels_btc_teacher.json",
    "open_systems/chordmini.json": "open-baselines/levels_btc_cl.json",
    "open_systems/chord_cnn_lstm.json": "open-baselines/levels_cnnlstm.json",
    "consistency/audio_shift.json": "open-consistency/audio_shift_summary.json",
    "consistency/vote.json": "open-consistency/audio_vote_summary.json",
    "consistency/singles.json": "open-consistency/singles_summary.json",
    "consistency/ours_audio_shift.json": "final-freeze/audio_CV-Rmnds_summary.json",
    "consistency/ours_keys.json": "final-freeze/symk_CV-Rmnds_summary.json",
}


def main():
    out = PROJECT / "results"
    sources = {}
    for dst, src in FILES.items():
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(R / src, out / dst)
        sources[dst] = f"results/{src}"
    for k in range(5):  # уровни каждой из 15 моделей на GuitarSet (DEV и отложенные группы фолда)
        for s in SEEDS:
            src = f"cycle-mnds/CV-Rmnds/fold{k}/CV-Rmnds-s{s}/dev_levels.json"
            dst = f"models/fold{k}_s{s}.json"
            (out / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(R / src, out / dst)
            sources[dst] = f"results/{src}"
    (out / "SOURCES.json").write_text(json.dumps(sources, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"скопировано файлов: {len(sources)}")


if __name__ == "__main__":
    main()
