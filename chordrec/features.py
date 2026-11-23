"""Признаки: CQT-спектрограмма, окно бинов для сети, нормировка и правило громкости (iii)."""
import os
from pathlib import Path

import numpy as np

from .config import BINS_PER_OCT, CACHE, HOP, MARGIN, N_BINS, N_BINS_FULL, SHIFT_BINS, SR, WIN


def cqt_from_signal(y: np.ndarray) -> np.ndarray:
    """Моно-сигнал 22 050 Гц → признаки (кадры × 288): log(1 + 100·|CQT|), 36 бинов на октаву от F#0, шаг 512."""
    import librosa

    c = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("F#0"),
                           n_bins=N_BINS_FULL, bins_per_octave=BINS_PER_OCT)).T.astype(np.float32)
    return np.log1p(100.0 * c)


def cqt_from_file(audio_path: str) -> np.ndarray:
    """Аудиофайл любого формата, который читает librosa → признаки (кадры × 288)."""
    import librosa

    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    return cqt_from_signal(y)


def cqt_cached(audio_path: str, sha256: str, cache: Path = CACHE) -> np.ndarray:
    """Как cqt_from_file, но с кешем по SHA-256 аудио (так хранятся признаки исследовательского проекта)."""
    target = Path(cache) / f"{sha256[:24]}-cqt288-h512.npy"
    if target.exists():
        return np.load(target)
    x = cqt_from_file(audio_path)
    tmp = target.with_suffix(".tmp.npy")
    np.save(tmp, x)
    os.replace(tmp, target)
    return x


def crop_bins(x: np.ndarray, k: int = 0) -> np.ndarray:
    """252 бина, которые читает сеть; k > 0 — окно выше на k полутонов (звук «опускается» на k полутонов)."""
    start = MARGIN + SHIFT_BINS * k
    return x[:, start:start + N_BINS]


def norm_stats(rows: list) -> tuple:
    """Среднее и разброс каждого бина по обучающим записям GuitarSet (вход сети нормируется ими)."""
    allx = np.concatenate([crop_bins(r["x"], 0) for r in rows])
    return allx.mean(0).astype(np.float32), (allx.std(0) + 1e-5).astype(np.float32)


# ---------------------------------------------------------------- правило (iii): громкость окна в обучающий диапазон
LEVEL_REF = 0.15847              # медиана RMS |X| окон 10 с обучающей гитары (0 дБ шкалы громкости)
LEVEL_RANGE = (-12.21, 8.51)     # 5–95 % громкости окон реальных обучающих записей, дБ


def magnitude(f: np.ndarray) -> np.ndarray:
    """Признаки → |CQT| (обратное к log(1 + 100·|X|))."""
    return np.expm1(np.asarray(f, np.float32)) / 100.0


def from_magnitude(a: np.ndarray) -> np.ndarray:
    return np.log1p(100.0 * a).astype(np.float32)


def level_db(a: np.ndarray) -> float:
    """Громкость окна: RMS |X| в дБ относительно LEVEL_REF."""
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-12)
    return 20 * np.log10(rms / LEVEL_REF)


def clamp_level(x: np.ndarray) -> np.ndarray:
    """Каждое окно 10 с громче или тише обучающего диапазона подтягивается к его границе (усиление ≤ 40 дБ);
    окна внутри диапазона не меняются. Сдвиг CQT по бинам громкость не меняет, поэтому правило и транспозиции перестановочны."""
    lo, hi = LEVEL_RANGE
    a = magnitude(x) * (10 ** (0.0 / 20))
    out = np.empty_like(a)
    for i in range(0, len(a), WIN):
        d = level_db(a[i:i + WIN])
        out[i:i + WIN] = a[i:i + WIN] * min(100.0, 10 ** ((min(max(d, lo), hi) - d) / 20))
    return from_magnitude(out)
