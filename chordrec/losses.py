"""Потери итогового рецепта: сглаживание по консонансу, согласие транспозиций, дистилляция ансамбля."""
import numpy as np
import torch
import torch.nn.functional as Fn

from .config import SEEDS

# Консонанс интервалов 0–11 полутонов (меньше — консонантнее), Poltronieri и др., ISMIR 2025
CONSONANCE = np.array([0, 7, 5, 1, 1, 2, 3, 1, 2, 2, 4, 6], dtype=np.float64)


def consonance_ce(logits, idx, alpha: float, ignore: int):
    """CE со сглаживанием меток по консонансу для выходов «12 классов + нет» (корень, бас): верный класс получает 1 − α,
    остальные классы высоты — α пропорционально консонантности интервала до верного; «нет» и неизвестные — как в обычной CE."""
    sim = 1.0 - CONSONANCE / CONSONANCE.max()
    sim[0] = 0.0
    k = np.stack([np.roll(sim, t) for t in range(12)])  # k[t, i] — сходство интервала (i − t) mod 12
    soft = np.zeros((13, 13))
    soft[:12, :12] = alpha * k / k.sum(1, keepdims=True)
    soft[np.arange(12), np.arange(12)] = 1.0 - alpha
    soft[12, 12] = 1.0
    table = torch.as_tensor(soft, dtype=torch.float32, device=logits.device)
    m = idx != ignore
    if not m.any():
        return logits.sum() * 0.0
    return -(table[idx[m]] * torch.log_softmax(logits[m].float(), -1)).sum(-1).mean()


def _roll_classes(v, idx):
    """Переставить первые 12 классов по индексам idx (батч × 12)."""
    return torch.gather(v, 2, idx[:, None, :].expand(-1, v.shape[1], -1))


def consistency_loss(o1: dict, o2: dict, s) -> torch.Tensor:
    """Согласие двух копий одного отрезка в разных тональностях (s = сдвиг второй минус сдвиг первой):
    класс c первой копии — это класс c − s второй. «Набор высот» — взаимная BCE, корень и «мажор/минор» — симметричная KL."""
    idx = (torch.arange(12, device=s.device)[None] - s[:, None]) % 12

    def kl(a, b):
        la, lb = Fn.log_softmax(a, -1), Fn.log_softmax(b, -1)
        return ((lb.exp().detach() * (lb.detach() - la)).sum(-1).mean() + (la.exp().detach() * (la.detach() - lb)).sum(-1).mean()) / 2

    a, b = o1["pc"].float(), _roll_classes(o2["pc"].float(), idx)
    pc = (Fn.binary_cross_entropy_with_logits(a, torch.sigmoid(b).detach()) + Fn.binary_cross_entropy_with_logits(b, torch.sigmoid(a).detach())) / 2
    r2, f2 = o2["root"].float(), o2["flat"].float()
    r2 = torch.cat([_roll_classes(r2[..., :12], idx), r2[..., 12:]], -1)
    f2 = torch.cat([_roll_classes(f2[..., :12], idx), _roll_classes(f2[..., 12:24], idx), f2[..., 24:]], -1)
    return 12 * pc + kl(o1["root"].float(), r2) + kl(o1["flat"].float(), f2)


def load_teachers(template: str, fold: int, device) -> list:
    """Учитель дистилляции: 3 замороженные модели (seeds 42/43/44) того же фолда; template с {fold} и {seed}."""
    from .model import load_model

    out = []
    for s in SEEDS:
        m = load_model(template.format(fold=f"fold{fold}", seed=s), device)
        for q in m.parameters():
            q.requires_grad_(False)
        out.append(m)
    return out


def teacher_targets(teachers: list, xs: list, deltas: list, device) -> dict:
    """Ответы учителя на трёх соседних транспозициях окна, сведённые к классам окна ученика и усреднённые
    по 3 моделям × 3 сдвигам. Окно, сдвинутое на δ дальше, видит класс c ученика как c − δ."""
    acc = {h: 0.0 for h in ("pc", "bass", "root", "flat")}
    n = 0
    with torch.no_grad():
        for x, d in zip(xs, deltas):
            d = torch.as_tensor(d, device=device)
            idx = (torch.arange(12, device=device)[None] - d[:, None]) % 12
            for m in teachers:
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    o = m(x)
                acc["pc"] = acc["pc"] + _roll_classes(torch.sigmoid(o["pc"].float()), idx)
                for h in ("bass", "root"):
                    p = torch.softmax(o[h].float(), -1)
                    acc[h] = acc[h] + torch.cat([_roll_classes(p[..., :12], idx), p[..., 12:]], -1)
                p = torch.softmax(o["flat"].float(), -1)
                acc["flat"] = acc["flat"] + torch.cat([_roll_classes(p[..., :12], idx), _roll_classes(p[..., 12:24], idx), p[..., 24:]], -1)
                n += 1
    return {h: v / n for h, v in acc.items()}


def distillation_loss(o: dict, t: dict) -> torch.Tensor:
    """Ученик повторяет мягкие ответы учителя: BCE для «набора высот» (×12, как основная потеря), KL для баса, корня и «мажор/минор»."""
    loss = 12 * Fn.binary_cross_entropy_with_logits(o["pc"].float(), t["pc"])
    for h in ("bass", "root", "flat"):
        lp = Fn.log_softmax(o[h].float(), -1)
        loss = loss + (t[h] * (t[h].clamp_min(1e-8).log() - lp)).sum(-1).mean()
    return loss
