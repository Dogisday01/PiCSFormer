"""Кросс-валидация итогового рецепта: 5 фолдов по группам GuitarSet × 3 seeds = 15 моделей.

python -m chordrec.cv --out runs/final [--folds 0,1,2,3,4] [--seeds 42,43,44]
Данные фолда загружаются один раз на все seeds. Готовые модели (есть meta.json) пропускаются.
"""
import argparse
from pathlib import Path

from .data import load_all
from .train import train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--teachers", default="results/cycle-maestro-notes/CV-Rmen/{fold}/CV-Rmen-s{seed}/best.pt")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    for k in (int(f) for f in a.folds.split(",")):
        data = None
        for s in (int(v) for v in a.seeds.split(",")):
            if (Path(a.out) / f"fold{k}" / f"s{s}" / "meta.json").exists():
                print(f"fold{k} s{s} уже готово", flush=True)
                continue
            data = data or load_all(k)
            train(k, s, Path(a.out), a.teachers, a.device, data=data)


if __name__ == "__main__":
    main()
