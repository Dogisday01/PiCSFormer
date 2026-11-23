"""Сеть итогового рецепта: свёрточная часть → 4 × (Transformer с RoPE + свёрточный модуль) → пять выходов.

Порядок создания слоёв и их имена совпадают с исследовательским кодом: веса обученных моделей загружаются без изменений,
а при одинаковом seed начальные веса получаются те же. 2 455 447 параметров.
"""
import torch
from torch import nn

from .config import N_BINS

D = 192         # ширина
LAYERS = 4      # блоков
HEADS = 4       # голов внимания
DROPOUT = 0.1

# выход → число классов; «note» нужен только при обучении (голова нот)
OUTPUTS = {"flat": 25, "pc": 12, "bass": 13, "root": 13, "note": 88}


def _ffn(width: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(D), nn.Linear(D, width), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(width, D), nn.Dropout(DROPOUT))


class ConvModule(nn.Module):
    """Свёрточный модуль после каждого блока внимания: GLU → свёртка по времени (ядро 15) → SiLU, с остаточной связью."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.pw1 = nn.Linear(D, 2 * D)
        self.dw = nn.Conv1d(D, D, 15, padding=7, groups=D)
        self.bn = nn.BatchNorm1d(D)
        self.pw2 = nn.Linear(D, D)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        y = nn.functional.glu(self.pw1(self.norm(x)), dim=-1).transpose(1, 2)
        y = nn.functional.silu(self.bn(self.dw(y))).transpose(1, 2)
        return x + self.drop(self.pw2(y))


class Block(nn.Module):
    """Блок Transformer (нормировка до внимания, ReLU) с вращательными позициями RoPE в запросах и ключах."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(D, D)
        self.drop = nn.Dropout(DROPOUT)
        self.ff = nn.ModuleList([_ffn(4 * D)])

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.norm(x)).view(b, t, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        half = D // HEADS // 2  # RoPE: поворот пар координат на угол, растущий с номером кадра
        freq = 10000.0 ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
        ang = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freq
        cos, sin = ang.cos().to(q.dtype), ang.sin().to(q.dtype)

        def rot(z):
            return torch.cat([z[..., :half] * cos - z[..., half:] * sin, z[..., :half] * sin + z[..., half:] * cos], -1)

        q, k = rot(q), rot(k)
        a = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=DROPOUT if self.training else 0.0)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(b, t, D)))
        return x + self.ff[-1](x)


class ChordNet(nn.Module):
    """Вход (батч, кадры, 252 бина CQT) → словарь выходов по кадрам (логиты)."""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(  # по частоте: 252 → 84 → 28 бинов, по времени разрешение не меняется
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d((1, 3)),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d((1, 3)),
            nn.Dropout(DROPOUT),
        )
        self.proj = nn.Linear(32 * (N_BINS // 9), D)
        self.attn = nn.ModuleList(Block() for _ in range(LAYERS))
        self.conv = nn.ModuleList(ConvModule() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(D)
        self.heads = nn.ModuleDict({"flat": nn.Linear(D, OUTPUTS["flat"])})
        for name in ("pc", "bass", "root", "note"):  # порядок как при обучении (влияет на начальные веса)
            self.heads[name] = nn.Linear(D, OUTPUTS[name])

    def forward(self, x):
        b, t, _ = x.shape
        h = self.cnn(x.unsqueeze(1))                                  # (B, 32, T, 28)
        h = self.proj(h.permute(0, 2, 1, 3).reshape(b, t, -1))       # (B, T, 192)
        for attn, conv in zip(self.attn, self.conv):
            h = conv(attn(h))
        h = self.norm(h)
        return {name: head(h) for name, head in self.heads.items()}


def load_model(path, device="cpu") -> ChordNet:
    """Модель с весами из файла .pt; режим распознавания (без dropout)."""
    m = ChordNet().to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m
