"""Хранилище сайта: аудио, запуски и отчёты — файлы JSON в одной папке (по умолчанию ./chordrec-data). Потокобезопасно."""
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

TERMINAL = {"completed", "failed"}


class ConflictError(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.audio_root = self.root / "audio"
        for d in ("audio", "runs", "reports"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    # ------------------------------------------------------------ файлы
    def _path(self, kind: str, oid: str) -> Path:
        if not oid or any(c in oid for c in "/\\.:"):
            raise KeyError(oid)
        return self.root / kind / f"{oid}.json"

    def _read(self, kind: str, oid: str) -> dict:
        p = self._path(kind, oid)
        if not p.is_file():
            raise KeyError(oid)
        return json.loads(p.read_text(encoding="utf-8"))

    def _write(self, kind: str, obj: dict) -> dict:
        p = self._path(kind, obj["id"])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)
        return obj

    def _list(self, kind: str) -> list:
        items = [json.loads(p.read_text(encoding="utf-8")) for p in (self.root / kind).glob("*.json")]
        return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)

    # ------------------------------------------------------------ аудио
    def add_audio(self, data: bytes, name: str, info: dict) -> dict:
        oid = uuid.uuid4().hex
        suffix = Path(name).suffix.lower()
        path = self.audio_root / f"{oid}{suffix}"
        path.write_bytes(data)
        meta = {"id": oid, "name": Path(name).name[:200], "path": str(path), "created_at": now(), **info}
        (self.audio_root / f"{oid}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return meta

    def get_audio(self, oid: str) -> dict:
        p = self.audio_root / f"{oid}.json"
        if any(c in oid for c in "/\\.:") or not p.is_file():
            raise KeyError(oid)
        return json.loads(p.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ запуски
    def create_run(self, audio: dict, mode: str, rules: bool) -> dict:
        with self.lock:
            run = {"id": uuid.uuid4().hex, "status": "queued", "mode": mode, "rules": rules, "audio": audio, "created_at": now(),
                   "progress": 0.0, "stage": "queued", "events": [], "report_id": None, "error": None}
            return self._write("runs", run)

    def get_run(self, oid: str) -> dict:
        with self.lock:
            return self._read("runs", oid)

    def list_runs(self) -> list:
        with self.lock:
            return [{k: v for k, v in r.items() if k != "events"} for r in self._list("runs")]

    def update_run(self, oid: str, **fields) -> dict:
        with self.lock:
            run = self._read("runs", oid)
            run.update(fields)
            return self._write("runs", run)

    def event(self, oid: str, stage: str, progress: float, detail: str = "") -> dict:
        with self.lock:
            run = self._read("runs", oid)
            item = {"id": len(run["events"]) + 1, "time": now(), "stage": stage, "progress": round(progress, 4), "detail": detail}
            run["events"].append(item)
            run["stage"], run["progress"] = stage, round(progress, 4)
            self._write("runs", run)
            return item

    def events(self, oid: str, after: int = 0) -> list:
        return [e for e in self.get_run(oid)["events"] if e["id"] > after]

    def interrupt_active(self):
        """После перезапуска сервера незавершённые запуски считаются прерванными."""
        with self.lock:
            for r in self._list("runs"):
                if r["status"] not in TERMINAL:
                    r.update(status="failed", error="interrupted")
                    self._write("runs", r)

    # ------------------------------------------------------------ отчёты
    def create_report(self, run: dict, result: dict, model: dict) -> dict:
        with self.lock:
            rep = {"id": uuid.uuid4().hex, "run_id": run["id"], "revision": 1, "created_at": now(), "updated_at": now(),
                   "audio": {k: v for k, v in run["audio"].items() if k != "path"}, "mode": run["mode"], "rules": run["rules"],
                   "model": model, "segments": result["chords"], "majmin": result["majmin"], "keys": result["keys"],
                   "original_segments": result["chords"], "corrections": [], "evaluation": None}
            self._write("reports", rep)
            return rep

    def get_report(self, oid: str) -> dict:
        with self.lock:
            return self._read("reports", oid)

    def list_reports(self) -> list:
        with self.lock:
            keep = ("id", "run_id", "revision", "created_at", "audio", "mode", "rules", "keys")
            return [{**{k: r.get(k) for k in keep}, "chords": len(r["segments"])} for r in self._list("reports")]

    def correct_report(self, oid: str, expected_revision: int, segments: list, note: str = "") -> dict:
        with self.lock:
            rep = self._read("reports", oid)
            if rep["revision"] != expected_revision:
                raise ConflictError("revision")
            rep["corrections"].append({"revision": rep["revision"] + 1, "time": now(), "note": note[:500]})
            rep.update(segments=segments, revision=rep["revision"] + 1, updated_at=now(), evaluation=None)
            return self._write("reports", rep)

    def set_evaluation(self, oid: str, evaluation: dict) -> dict:
        with self.lock:
            rep = self._read("reports", oid)
            rep["evaluation"] = evaluation
            return self._write("reports", rep)

    def delete_run(self, oid: str):
        with self.lock:
            run = self._read("runs", oid)
            if run["status"] not in TERMINAL:
                raise ConflictError("active")
            if run.get("report_id"):
                self._path("reports", run["report_id"]).unlink(missing_ok=True)
            self._path("runs", oid).unlink(missing_ok=True)
