"""Данные обучения итогового рецепта: GuitarSet (разбиение по фолду), синтетика AAM, три пула реального звука и ноты MAESTRO.

Пулы и доли в батче: GuitarSet 0,45 · синтетика AAM 0,10 · MusicNet 0,15 · рояль Disklavier 0,10 · JazzSAMBA 0,15 ·
Признаки пулов читаются через mmap (в память целиком не загружаются).
"""
import hashlib
import io
import json
import zipfile

import numpy as np

from .config import BINS_PER_OCT, CACHE, FOLDS, HOP, IGNORE, MANIFESTS, N_BINS_FULL, NOTE_GOOD, SR
from .features import cqt_cached, norm_stats
from .labels import aam_intervals, bass_from_notes, frame_targets, note_targets, pcs_targets

AAM_COUNT = 600      # синтетических треков AAM
FRACTIONS = {"aam": 0.10, "musicnet": 0.15, "piano": 0.10, "jazz": 0.15, "maestro": 0.05}  # GuitarSet — остаток (0,45)


def _rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def aam_features(row: dict) -> np.ndarray:
    """Признаки трека AAM из кеша (mmap); если кеша нет — считаются из zip-архива синтетики."""
    ref = row["audio_archive"]
    key = hashlib.sha256(f"{ref['archive']}::{ref['member']}".encode()).hexdigest()
    target = CACHE / f"aam-{key[:24]}-cqt288-h512.npy"
    if not target.exists():
        import librosa
        import soundfile as sf

        with zipfile.ZipFile(ref["archive"]) as zf:
            audio, sr = sf.read(io.BytesIO(zf.read(ref["member"])), dtype="float32", always_2d=True)
        y = librosa.resample(audio.mean(1), orig_sr=sr, target_sr=SR)
        c = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("F#0"),
                               n_bins=N_BINS_FULL, bins_per_octave=BINS_PER_OCT)).T.astype(np.float32)
        np.save(target, np.log1p(100.0 * c))
    return np.load(target, mmap_mode="r")


def guitarset(fold: int) -> dict:
    """GuitarSet: train — обучение; inner — 2 отложенные группы фолда (выбор штрафа, оценка «гитара»); dev — DEV (только отчёт).
    Тестовая часть не читается."""
    inner_groups = set(FOLDS[fold])
    out = {"train": [], "inner": [], "dev": []}
    for r in _rows(MANIFESTS["guitarset"]):
        if r["split"] == "validation":
            part = "dev"
        elif r["split"] == "train":
            part = "inner" if r["group_id"] in inner_groups else "train"
        else:
            continue
        x = cqt_cached(r["audio_path"], r["audio_sha256"])
        row = {"id": r["id"], "duration": r["duration"], "chord_intervals": r["chord_intervals"], "x": x,
               "y": frame_targets(r["chord_intervals"], len(x))}
        if part == "train":  # цели «набора высот» и ноты нужны только для обучения
            row["p"] = pcs_targets(r["chord_intervals"], len(x))
            row["n"] = note_targets(r["id"], len(x))
            bass_from_notes(row["p"], row["n"], r["chord_intervals"])
        out[part].append(row)
    return out


def aam() -> list:
    """Синтетика AAM: первые 600 обучающих треков; размечены только трезвучия (прочие классы высоты — неизвестно)."""
    rows = [r for r in _rows(MANIFESTS["aam"]) if r["split"] == "train"][:AAM_COUNT]
    out = []
    for r in rows:
        x = aam_features(r)
        out.append({"id": r["id"], "x": x, "y": frame_targets(aam_intervals(r["chord_intervals"]), len(x)),
                    "p": pcs_targets(r["chord_intervals"], len(x), triad_only=True)})
    return out


def pool(name: str) -> list:
    """Пул реального звука (MusicNet, рояль, джаз). Метка X (пробел разметки) — неизвестно, а не «нет аккорда».
    У MusicNet с проверенными нотами — ещё ноты и бас из нот."""
    good = set(json.load(open(NOTE_GOOD, encoding="utf-8"))["musicnet"]["good"]) if name == "musicnet" else set()
    out = []
    for r in _rows(MANIFESTS[name]):
        x = np.load(r["feat"], mmap_mode="r")
        known = [i for i in r["chord_intervals"] if i["label"] != "X"]
        row = {"id": r["id"], "x": x, "y": frame_targets(known, len(x)), "p": pcs_targets(known, len(x))}
        if name == "musicnet" and r["id"] in good:
            row["n"] = note_targets(r["id"], len(x))
            bass_from_notes(row["p"], row["n"], known)
        out.append(row)
    return out


def maestro_notes() -> list:
    """Ноты MAESTRO (с педалью) для головы нот: 223 записи без сонат Бетховена; цели аккордов неизвестны (IGNORE / −1)."""
    out = []
    for r in _rows(MANIFESTS["maestro_notes"]):
        x = np.load(r["feat"], mmap_mode="r")
        out.append({"id": r["id"], "x": x, "y": np.full((len(x), 2), IGNORE, np.int16),
                    "p": np.full((len(x), 13), -1, np.int8), "n": np.load(r["notes"], mmap_mode="r")})
    return out


def load_all(fold: int) -> dict:
    """Все данные фолда и статистики нормировки входа (по обучающим записям GuitarSet)."""
    data = guitarset(fold)
    data["aam"] = aam()
    for name in ("musicnet", "piano", "jazz"):
        data[name] = pool(name)
    data["maestro"] = maestro_notes()
    data["stats"] = norm_stats(data["train"])
    return data
