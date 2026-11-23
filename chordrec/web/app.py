"""Локальный сайт распознавания аккордов: FastAPI + собранный интерфейс React (web/dist).

Только для своего компьютера: сервер слушает 127.0.0.1, запросы с изменениями требуют токен сессии (заголовок X-Chord-Token)
и проверяются по Origin. Запуск — python -m chordrec.web.
"""
import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import APIRouter, FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import PROJECT
from .exports import FORMATS, evaluate_report, export_report
from .jobs import MODES, Jobs
from .store import TERMINAL, ConflictError, Store

MAX_UPLOAD = 60 * 1024 * 1024
MAX_DURATION = 15 * 60
FORMATS_IN = (".wav", ".flac", ".ogg", ".mp3")
DIST = PROJECT / "web" / "dist"


def ok(data):
    return {"success": True, "data": data}


def fail(status: int, code: str):
    return JSONResponse({"success": False, "error": {"code": code}}, status_code=status)


class StartRun(BaseModel):
    audio_id: str = Field(min_length=8, max_length=64)
    mode: str = Field(pattern="^(single|ensemble)$")
    rules: bool = True


class Segment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    label: str = Field(min_length=1, max_length=40)
    confidence: float | None = None


class Correction(BaseModel):
    expected_revision: int
    segments: list[Segment] = Field(min_length=1, max_length=20000)
    note: str = Field(default="", max_length=500)


class LocalOnly(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "DELETE", "PATCH") and request.url.path.startswith("/api"):
            if not secrets.compare_digest(request.headers.get("x-chord-token", ""), request.app.state.token):
                return fail(403, "forbidden")
            origin = request.headers.get("origin")
            if origin and origin.split("://", 1)[-1] != request.headers.get("host", ""):
                return fail(403, "forbidden")
        return await call_next(request)


def audio_info(data: bytes, name: str) -> dict:
    import io

    import librosa

    suffix = Path(name).suffix.lower()
    if suffix not in FORMATS_IN:
        raise ValueError("format")
    try:
        info = sf.info(io.BytesIO(data))
        duration, sr, ch = info.frames / info.samplerate, info.samplerate, info.channels
    except Exception:  # noqa: BLE001 — например, mp3 в старом libsndfile
        y, sr = librosa.load(io.BytesIO(data), sr=None, mono=False)
        duration, ch = y.shape[-1] / sr, (1 if y.ndim == 1 else y.shape[0])
    if not 0.5 <= duration <= MAX_DURATION:
        raise ValueError("duration")
    return {"duration": round(duration, 3), "sample_rate": int(sr), "channels": int(ch)}


def build_router() -> APIRouter:
    api = APIRouter(prefix="/api")

    @api.get("/bootstrap")
    def bootstrap(request: Request):
        s = request.app.state
        return ok({"app": "chordrec/web/1", "token": s.token, "model": s.jobs.model_info(), "modes": list(MODES),
                   "limits": {"max_upload_bytes": MAX_UPLOAD, "max_duration_seconds": MAX_DURATION, "formats": [f[1:] for f in FORMATS_IN]}})

    @api.post("/audio")
    async def upload(request: Request, file: UploadFile = File(...)):
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise ValueError("size")
        name = file.filename or "audio.wav"
        info = await run_in_threadpool(audio_info, data, name)
        meta = request.app.state.store.add_audio(data, name, info)
        return ok({k: v for k, v in meta.items() if k != "path"})

    @api.post("/runs")
    def start(request: Request, body: StartRun):
        s = request.app.state
        run = s.store.create_run(s.store.get_audio(body.audio_id), body.mode, body.rules)
        s.jobs.submit(run["id"])
        return ok(public_run(run))

    @api.get("/runs")
    def runs(request: Request):
        return ok([public_run(r) for r in request.app.state.store.list_runs()])

    @api.get("/runs/{run_id}")
    def run(request: Request, run_id: str):
        return ok(public_run(request.app.state.store.get_run(run_id)))

    @api.delete("/runs/{run_id}")
    def delete(request: Request, run_id: str):
        request.app.state.store.delete_run(run_id)
        return ok({"deleted": run_id})

    @api.get("/runs/{run_id}/audio")
    def audio(request: Request, run_id: str):
        store = request.app.state.store
        path = Path(store.get_run(run_id)["audio"]["path"]).resolve()
        if path.parent != store.audio_root or not path.is_file():
            raise KeyError(run_id)
        return FileResponse(path)

    @api.get("/runs/{run_id}/events")
    async def events(request: Request, run_id: str, after: int = 0):
        store = request.app.state.store
        await run_in_threadpool(store.get_run, run_id)

        async def stream():
            cursor = after
            while not await request.is_disconnected():
                snap = await run_in_threadpool(store.get_run, run_id)
                for e in [e for e in snap["events"] if e["id"] > cursor]:
                    cursor = e["id"]
                    yield f"id: {cursor}\nevent: progress\ndata: {json.dumps(e)}\n\n"
                if snap["status"] in TERMINAL:
                    yield f"event: done\ndata: {json.dumps({'status': snap['status'], 'report_id': snap.get('report_id')})}\n\n"
                    return
                await asyncio.sleep(0.3)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @api.get("/reports")
    def reports(request: Request):
        return ok(request.app.state.store.list_reports())

    @api.get("/reports/{report_id}")
    def report(request: Request, report_id: str):
        return ok(request.app.state.store.get_report(report_id))

    @api.post("/reports/{report_id}/corrections")
    def correct(request: Request, report_id: str, body: Correction):
        segs = sorted((s.model_dump() for s in body.segments), key=lambda s: s["start"])
        import mir_eval

        for s in segs:
            if s["end"] <= s["start"]:
                raise ValueError("interval")
            try:
                mir_eval.chord.validate_chord_label(s["label"])
            except Exception as exc:  # noqa: BLE001
                raise ValueError("label") from exc
        return ok(request.app.state.store.correct_report(report_id, body.expected_revision, segs, body.note))

    @api.post("/reports/{report_id}/evaluate")
    async def evaluate(request: Request, report_id: str, file: UploadFile = File(...)):
        store = request.app.state.store
        rep = await run_in_threadpool(store.get_report, report_id)
        result = await run_in_threadpool(evaluate_report, rep, await file.read(2 * 1024 * 1024))
        return ok(store.set_evaluation(report_id, result))

    @api.get("/reports/{report_id}/export")
    def export(request: Request, report_id: str, format: str = "json"):
        if format not in FORMATS:
            raise ValueError(format)
        rep = request.app.state.store.get_report(report_id)
        payload, mime = export_report(rep, format)
        stem = Path(rep["audio"]["name"]).stem or "chords"
        return Response(payload, media_type=mime,
                        headers={"Content-Disposition": f'attachment; filename="{stem}.chords.r{rep["revision"]}.{format}"'})

    return api


def public_run(run: dict) -> dict:
    return {**{k: v for k, v in run.items() if k not in ("audio", "events")},
            "audio": {k: v for k, v in run["audio"].items() if k != "path"}}


def create_app(data_root: Path = None, device: str = "cpu") -> FastAPI:
    store = Store(Path(data_root or Path.cwd() / "chordrec-data"))
    jobs = Jobs(store, device)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.interrupt_active()
        yield
        jobs.close()

    app = FastAPI(title="Chord Recognition", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.store, app.state.jobs, app.state.token = store, jobs, secrets.token_urlsafe(32)
    app.add_middleware(LocalOnly)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return fail(404, "not_found")

    @app.exception_handler(ConflictError)
    async def conflict(request, exc):
        return fail(409, f"conflict_{exc}")

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return fail(422, f"invalid_{exc}" if str(exc) in ("format", "duration", "size", "empty", "interval", "label") else "invalid_input")

    app.include_router(build_router())

    @app.get("/{path:path}")
    def frontend(path: str):
        if path.startswith("api"):
            return fail(404, "not_found")
        candidate = (DIST / path).resolve()
        if candidate.is_file() and candidate.is_relative_to(DIST.resolve()):
            return FileResponse(candidate)
        index = DIST / "index.html"
        return FileResponse(index) if index.is_file() else fail(503, "frontend_not_built")

    return app
