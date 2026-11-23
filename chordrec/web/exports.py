"""Выгрузка отчёта (JSON, LAB, CSV) и оценка по эталонной разметке .lab."""
import csv
import io
import json

from ..evaluate import levels

FORMATS = {"json": "application/json", "lab": "text/plain; charset=utf-8", "csv": "text/csv; charset=utf-8"}


def export_report(rep: dict, fmt: str) -> tuple:
    if fmt not in FORMATS:
        raise ValueError(fmt)
    if fmt == "json":
        keep = ("id", "revision", "created_at", "audio", "mode", "rules", "model", "segments", "majmin", "keys", "evaluation")
        return json.dumps({k: rep.get(k) for k in keep}, ensure_ascii=False, indent=1).encode("utf-8"), FORMATS[fmt]
    if fmt == "lab":
        text = "".join(f"{s['start']:.3f}\t{s['end']:.3f}\t{s['label']}\n" for s in rep["segments"])
        return text.encode("utf-8"), FORMATS[fmt]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["start", "end", "label", "confidence"])
    for s in rep["segments"]:
        w.writerow([f"{s['start']:.3f}", f"{s['end']:.3f}", s["label"], s.get("confidence", "")])
    return buf.getvalue().encode("utf-8"), FORMATS[fmt]


def parse_lab(data: bytes) -> list:
    """Эталон: строки «начало конец метка» (пробелы или табуляция), метки Harte."""
    import mir_eval

    out = []
    for line in data.decode("utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) < 3 or line.lstrip().startswith("#"):
            continue
        start, end, label = float(parts[0]), float(parts[1]), parts[2]
        try:
            mir_eval.chord.validate_chord_label(label)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("label") from exc
        if end > start:
            out.append({"start": start, "end": end, "label": label})
    if not out:
        raise ValueError("empty")
    return out


def evaluate_report(rep: dict, reference: bytes) -> dict:
    ref = parse_lab(reference)
    row = {"id": "upload_comp", "chord_intervals": ref, "duration": ref[-1]["end"]}
    # majmin — по выходу «мажор/минор», как в статье; после ручной правки — по исправленной разметке
    majmin = rep["majmin"] if rep["revision"] == 1 else rep["segments"]
    lv = levels([row], {row["id"]: majmin}, {row["id"]: rep["segments"]})
    return {"levels": {k: (round(v, 2) if v is not None else None) for k, v in lv.items()},
            "reference_segments": len(ref), "revision": rep["revision"]}
