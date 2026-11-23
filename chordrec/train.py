"""Обучение одной модели итогового рецепта (один фолд, один seed).

python -m chordrec.train --fold 0 --seed 42 --out runs/final --teachers "<папка учителя>/{fold}/CV-Rmen-s{seed}/best.pt"

Рецепт: 60 эпох × 100 шагов, батч 16 отрезков по 10 с, AdamW (lr 1e-3, weight decay 0,01), OneCycle, веса последней эпохи.
Каждый отрезок: случайный сдвиг окна CQT на −5…+6 полутонов (метки сдвигаются так же) — транспозиционная аугментация.
Потери: «мажор/минор» (CE) + корень и бас (CE со сглаживанием по консонансу, α = 0,1) + 12 · «набор высот» (BCE)
+ 12 · ноты (BCE) + дистилляция ансамбля учителя (вес 1) + 0,5 · согласие двух транспозиций отрезка.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import FRAME, IGNORE, MARGIN, RESEARCH, SHIFT_BINS, SHIFTS
from .data import FRACTIONS, load_all
from .decode import decode_flat, decode_pcs
from .evaluate import evaluate
from .features import crop_bins
from .labels import shift_pcs, shift_targets
from .losses import consistency_loss, consonance_ce, distillation_loss, load_teachers, teacher_targets
from .model import ChordNet

RECIPE = {"epochs": 60, "steps": 100, "batch": 16, "lr": 1e-3, "crop_s": 10.0, "eval_every": 3,
          "cons_alpha": 0.1, "consist_w": 0.5, "distill_w": 1.0}
PENALTIES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)  # штраф декодера выбирается по отложенным группам GuitarSet


def select_score(lv: dict) -> float:
    """Мера выбора штрафа: (majmin по «мажор/минор» + sevenths по полным меткам) / 2 на аккомпанементе отложенных групп."""
    comp = lambda d: d.get("comp", d["pooled"])  # noqa: E731
    return (comp(lv["flat"])["majmin"] + comp(lv["struct"])["sevenths"]) / 2


def train(fold: int, seed: int, out_dir: Path, teachers_template: str, device: str = "cuda", data: dict = None,
          epochs: int = RECIPE["epochs"], steps: int = RECIPE["steps"], resume: bool = True) -> dict:
    out = Path(out_dir) / f"fold{fold}" / f"s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    data = data or load_all(fold)
    mean, std = data["stats"]
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    device = torch.device(device)
    model = ChordNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=RECIPE["lr"], total_steps=epochs * steps, pct_start=0.1)
    crop = int(round(RECIPE["crop_s"] / FRAME))  # 431 кадр
    gs_frac = 1.0 - sum(FRACTIONS.values())  # 0,45
    pools = [("gs", data["train"], gs_frac), ("aam", data["aam"], FRACTIONS["aam"]), ("musicnet", data["musicnet"], FRACTIONS["musicnet"]),
             ("piano", data["piano"], FRACTIONS["piano"]), ("jazz", data["jazz"], FRACTIONS["jazz"]),
             ("maestro", data["maestro"], FRACTIONS["maestro"])]
    pool_w = [np.array([len(r["x"]) for r in rows], dtype=np.float64) for _, rows, _ in pools]
    pool_w = [w / w.sum() for w in pool_w]  # запись выбирается с вероятностью, пропорциональной длине
    rng = np.random.default_rng(seed)
    ce = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    t_path = Path(teachers_template)
    teachers = load_teachers(str(t_path if t_path.is_absolute() else RESEARCH / t_path), fold, device)
    history, started, start_epoch = [], time.time(), 1

    def infer(rows):
        """Логиты выходов для записей целиком (для выбора штрафа и отчёта)."""
        model.eval()
        res = {}
        with torch.no_grad():
            for row in rows:
                x = torch.as_tensor((crop_bins(row["x"], 0) - mean) / std, device=device)[None]
                with torch.autocast(device.type, enabled=amp):
                    o = model(x)
                res[row["id"]] = {k: v[0].float().cpu().numpy() for k, v in o.items()}
        model.train()
        return res

    def levels_for(rows, outs, penalty):
        per = {"flat": {}, "struct": {}}
        for r in rows:
            per["flat"][r["id"]] = decode_flat(outs[r["id"]]["flat"], penalty, r["duration"])
            per["struct"][r["id"]] = decode_pcs(outs[r["id"]], penalty, r["duration"])
        return {kind: evaluate(rows, preds) for kind, preds in per.items()}, per

    ck = out / "resume.pt"  # продолжение после остановки: полное состояние после каждой эпохи
    if resume and ck.exists():
        st = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]), opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        history = st["history"]
        random.setstate(st["py"]), np.random.set_state(st["np"])
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch"].cpu())
        if device.type == "cuda" and st.get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])
        started, start_epoch = time.time() - st["elapsed"], st["epoch"] + 1

    for epoch in range(start_epoch, epochs + 1):
        losses = []
        for _ in range(steps):
            xb, yb, pb, nb, xb2, sb2 = [], [], [], [], [], []
            tx, td = [[], [], []], [[], [], []]  # окна для учителя и их сдвиг относительно окна ученика
            which = rng.choice(len(pools), size=RECIPE["batch"], p=[p[2] for p in pools])
            for w in which:
                rows = pools[w][1]
                row = rows[int(rng.choice(len(rows), p=pool_w[w]))]
                n = len(row["x"])
                start = int(rng.integers(0, max(1, n - crop)))
                k = int(rng.choice(SHIFTS))                      # транспозиционная аугментация
                xs = row["x"][start:start + crop]                 # полный CQT отрезка (288 бинов)
                x = crop_bins(xs, k)
                y = row["y"][start:start + crop]
                y = shift_targets(y, k) if k else y
                p = row["p"][start:start + crop]
                p = shift_pcs(p, k) if k else p
                if len(p) < crop:
                    p = np.pad(p, ((0, crop - len(p)), (0, 0)), constant_values=-1)
                pb.append(p)
                q = np.full((crop, 88), -1, np.int8)              # ноты сдвигаются с окном; нет нот — неизвестно
                if row.get("n") is not None:
                    seg = row["n"][start:start + crop]
                    q[:len(seg)] = 0
                    lo, hi = max(0, k), 88 + min(0, k)
                    q[:len(seg), lo - k:hi - k] = seg[:, lo:hi]   # нота p в окне, сдвинутом на k, — это p − k
                nb.append(q)
                if len(x) < crop:
                    x = np.pad(x, ((0, crop - len(x)), (0, 0)))
                    y = np.pad(y, ((0, crop - len(y)), (0, 0)), constant_values=IGNORE)
                xb.append((x - mean) / std)
                yb.append(y)
                for j, dlt in enumerate((-2, -1, 0) if k + 1 > MARGIN // SHIFT_BINS else (-1, 0, 1)):  # учитель: 3 соседних сдвига
                    xt = crop_bins(row["x"][start:start + crop], k + dlt)
                    if len(xt) < crop:
                        xt = np.pad(xt, ((0, crop - len(xt)), (0, 0)))
                    tx[j].append((xt - mean) / std)
                    td[j].append(dlt)
                k2 = int(rng.choice([v for v in SHIFTS if v != k]))  # вторая копия отрезка в другой тональности
                x2 = crop_bins(xs, k2)
                if len(x2) < crop:
                    x2 = np.pad(x2, ((0, crop - len(x2)), (0, 0)))
                xb2.append((x2 - mean) / std)
                sb2.append(k2 - k)
            x = torch.as_tensor(np.stack(xb), device=device, dtype=torch.float32)
            y = torch.as_tensor(np.stack(yb), device=device)
            with torch.autocast(device.type, enabled=amp):
                o = model(x)
                loss = 0.0
                loss = loss + ce(o["flat"].flatten(0, 1).float(), y[..., 1].flatten())
                loss = loss + consonance_ce(o["root"].flatten(0, 1), y[..., 0].flatten(), RECIPE["cons_alpha"], IGNORE)
                pt = torch.as_tensor(np.stack(pb), device=device).long()
                m = pt[..., :12] >= 0
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    o["pc"].float()[m], pt[..., :12][m].float(), reduction="mean") if m.any() else 0.0
                bt = pt[..., 12].clone()
                bt[bt < 0] = IGNORE
                loss = loss + 12 * bce + consonance_ce(o["bass"].flatten(0, 1), bt.flatten(), RECIPE["cons_alpha"], IGNORE)
                nt = torch.as_tensor(np.stack(nb), device=device)
                m = nt >= 0
                if m.any():
                    loss = loss + 12 * torch.nn.functional.binary_cross_entropy_with_logits(o["note"].float()[m], nt[m].float(), reduction="mean")
                tt = teacher_targets(teachers, [torch.as_tensor(np.stack(v), device=device, dtype=torch.float32) for v in tx], td, device)
                loss = loss + RECIPE["distill_w"] * distillation_loss(o, tt)
                o2 = model(torch.as_tensor(np.stack(xb2), device=device, dtype=torch.float32))
                loss = loss + RECIPE["consist_w"] * consistency_loss(o, o2, torch.as_tensor(sb2, device=device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
        if epoch % RECIPE["eval_every"] == 0 or epoch == epochs:  # контроль по отложенным группам (на выбор весов не влияет)
            lv, _ = levels_for(data["inner"], infer(data["inner"]), 0.0)
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "inner_score": select_score(lv), "seconds": time.time() - started})
            print(f"fold{fold} s{seed} эпоха {epoch}/{epochs} loss {np.mean(losses):.3f} inner {history[-1]['inner_score']:.2f}", flush=True)
        if resume:
            st = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                  "scaler": scaler.state_dict(), "history": history, "py": random.getstate(), "np": np.random.get_state(),
                  "rng": rng.bit_generator.state, "torch": torch.get_rng_state(),
                  "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None, "elapsed": time.time() - started}
            torch.save(st, out / "resume.tmp")
            os.replace(out / "resume.tmp", ck)

    torch.save(model.state_dict(), out / "best.pt")  # веса последней эпохи
    inner_out = infer(data["inner"])
    scores = {p: select_score(levels_for(data["inner"], inner_out, p)[0]) for p in PENALTIES}
    chosen = max(scores, key=scores.get)
    dev_out = infer(data["dev"])
    meta = {"fold": fold, "seed": seed, "recipe": RECIPE, "params": sum(p.numel() for p in model.parameters()),
            "chosen_penalty": chosen, "penalty_inner": scores,
            "inner_levels": levels_for(data["inner"], inner_out, chosen)[0],
            "dev_levels": {f"{kind}/{name}": lv for name, pen in (("raw", 0.0), ("viterbi", chosen))
                           for kind, lv in levels_for(data["dev"], dev_out, pen)[0].items()},
            "history": history, "train_seconds": time.time() - started}
    json.dump(meta, open(out / "meta.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if ck.exists():
        ck.unlink()
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teachers", default="results/cycle-maestro-notes/CV-Rmen/{fold}/CV-Rmen-s{seed}/best.pt",
                    help="веса учителя дистилляции (относительно исследовательского проекта или абсолютный путь)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=RECIPE["epochs"])
    ap.add_argument("--steps", type=int, default=RECIPE["steps"])
    a = ap.parse_args()
    train(a.fold, a.seed, Path(a.out), a.teachers, a.device, epochs=a.epochs, steps=a.steps)


if __name__ == "__main__":
    main()
