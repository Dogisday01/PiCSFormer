"""Очередь распознавания сайта: один фоновый поток, модели загружаются один раз и держатся в памяти."""
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import librosa
import torch

from ..config import SR
from ..features import cqt_from_signal
from ..recognize import load_models
from .analysis import analyze

MODES = {"single": 1, "ensemble": 5}  # одна модель (фолд 0, seed 42) или среднее пяти моделей (по одной на фолд)
# доли общего хода работы: чтение звука, CQT, модели, декодер
SHARE = {"audio": (0.0, 0.08), "features": (0.08, 0.22), "model": (0.22, 0.92), "decode": (0.92, 1.0)}


class Jobs:
    def __init__(self, store, device: str = "cpu"):
        self.store = store
        self.device = torch.device(device)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chordrec")
        self.models = {}
        self.lock = threading.Lock()

    def model_info(self) -> dict:
        return {"name": "CV-Rmnds", "parameters": 2_460_000, "device": str(self.device), "modes": list(MODES)}

    def _models(self, mode: str) -> list:
        with self.lock:
            if mode not in self.models:
                self.models[mode] = load_models(MODES[mode], self.device)
            return self.models[mode]

    def submit(self, run_id: str):
        self.pool.submit(self._run, run_id)

    def _run(self, run_id: str):
        store = self.store
        run = store.update_run(run_id, status="running")

        def step(stage, done=0, total=1):
            a, b = SHARE[stage]
            store.event(run_id, stage, a + (b - a) * (done / max(total, 1)), f"{done}/{total}")

        try:
            step("audio")
            y, _ = librosa.load(run["audio"]["path"], sr=SR, mono=True)
            duration = len(y) / SR
            step("features")
            x = cqt_from_signal(y)
            models = self._models(run["mode"])
            step("model", 0, 1)
            result = analyze(models, x, duration, self.device, rules=run["rules"], progress=step)
            rep = store.create_report(run, result, {**self.model_info(), "models": MODES[run["mode"]]})
            store.event(run_id, "done", 1.0)
            store.update_run(run_id, status="completed", report_id=rep["id"], chords=len(result["chords"]))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            store.update_run(run_id, status="failed", error=type(exc).__name__)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
