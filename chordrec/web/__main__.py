"""Локальный сайт распознавания аккордов.

python -m chordrec.web                       # http://127.0.0.1:8765, браузер откроется сам
python -m chordrec.web --device cuda         # распознавание на видеокарте
python -m chordrec.web --data D:/chords      # где хранить загруженные записи и отчёты (по умолчанию ./chordrec-data)
"""
import argparse
import threading
import webbrowser


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data", default=None)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    import uvicorn

    from .app import create_app

    app = create_app(a.data, a.device)
    url = f"http://127.0.0.1:{a.port}"
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Сайт: {url}  (остановка — Ctrl+C)")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
