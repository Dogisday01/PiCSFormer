"""Проверка сайта: (1) распознавание сайта (chordrec.web.analysis) даёт те же аккорды, что chordrec.rules.recognize;
(2) весь путь API: загрузка записи → запуск → ход работы → отчёт → правка → оценка по эталону → выгрузка.

python tests/check_web.py [запись.wav [эталон.lab]]
Без аргументов берётся запись GuitarSet DEV из исследовательского проекта и её эталон.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import librosa
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from chordrec.config import MANIFESTS, SR  # noqa: E402
from chordrec.features import cqt_from_signal  # noqa: E402
from chordrec.recognize import load_models  # noqa: E402
from chordrec.rules import recognize  # noqa: E402
from chordrec.web.analysis import analyze  # noqa: E402


def reference_record():
    rec = next(json.loads(line) for line in open(MANIFESTS["guitarset"], encoding="utf-8") if '"00_Jazz2-110-Bb_comp"' in line)
    lab = "".join(f"{s['start']}\t{s['end']}\t{s['label']}\n" for s in rec["chord_intervals"])
    return rec["audio_path"], lab.encode("utf-8")


def main(audio=None, lab=None):
    if audio is None:
        audio, ref = reference_record()
    else:
        ref = Path(lab).read_bytes() if lab else None
    cpu = torch.device("cpu")
    y, _ = librosa.load(audio, sr=SR, mono=True)
    x, dur = cqt_from_signal(y), len(y) / SR
    models = load_models(1, cpu)
    for rules in (True, False):
        a = analyze(models, x, dur, cpu, rules=rules)["chords"]
        b = recognize(models, x, dur, cpu, rules=rules)["chords"]
        same = [(s["start"], s["end"], s["label"]) for s in a] == [(s["start"], s["end"], s["label"]) for s in b]
        print(f"1. правила={rules}: {len(a)} отрезков, совпадает с chordrec.rules.recognize: {same}")
        assert same

    from fastapi.testclient import TestClient

    from chordrec.web.app import create_app

    with tempfile.TemporaryDirectory() as tmp, TestClient(create_app(Path(tmp))) as client:
        boot = client.get("/api/bootstrap").json()["data"]
        hdr = {"X-Chord-Token": boot["token"]}
        assert client.post("/api/runs", json={"audio_id": "x" * 10, "mode": "single"}).status_code == 403  # без токена
        up = client.post("/api/audio", files={"file": (Path(audio).name, Path(audio).read_bytes(), "audio/wav")}, headers=hdr).json()
        assert up["success"], up
        run = client.post("/api/runs", json={"audio_id": up["data"]["id"], "mode": "single", "rules": True}, headers=hdr).json()["data"]
        t0 = time.time()
        while True:
            snap = client.get(f"/api/runs/{run['id']}").json()["data"]
            if snap["status"] in ("completed", "failed"):
                break
            time.sleep(0.5)
        print(f"2. запуск: {snap['status']} за {time.time() - t0:.1f} с, аккордов {snap.get('chords')}")
        assert snap["status"] == "completed"
        rep = client.get(f"/api/reports/{snap['report_id']}").json()["data"]
        same = [s["label"] for s in rep["segments"]] == [s["label"] for s in analyze(models, x, dur, cpu)["chords"]]
        print(f"3. отчёт: {len(rep['segments'])} аккордов, тональностей {len(rep['keys'])}, совпадает с распознаванием: {same}")
        assert same
        segs = rep["segments"]
        segs[0] = {**segs[0], "label": "C:maj"}
        cor = client.post(f"/api/reports/{rep['id']}/corrections", json={"expected_revision": 1, "segments": segs}, headers=hdr).json()
        assert cor["success"] and cor["data"]["revision"] == 2
        stale = client.post(f"/api/reports/{rep['id']}/corrections", json={"expected_revision": 1, "segments": segs}, headers=hdr)
        print(f"4. правка: ревизия {cor['data']['revision']}, устаревшая правка отклонена: {stale.status_code == 409}")
        if ref:
            ev = client.post(f"/api/reports/{rep['id']}/evaluate", files={"file": ("ref.lab", ref, "text/plain")}, headers=hdr).json()
            print("5. оценка по эталону:", ev["data"]["evaluation"]["levels"])
        for fmt in ("json", "lab", "csv"):
            r = client.get(f"/api/reports/{rep['id']}/export?format={fmt}")
            assert r.status_code == 200 and len(r.content) > 10
        print("6. выгрузка json / lab / csv: ок")
    print("ИТОГ: всё совпадает")


if __name__ == "__main__":
    main(*sys.argv[1:3])
