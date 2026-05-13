from __future__ import annotations
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F



def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)



class BinaryConfusionMatrix:
    

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds   = preds.bool()
        targets = targets.bool()
        self.tp += int(( preds &  targets).sum().item())
        self.fp += int(( preds & ~targets).sum().item())
        self.fn += int((~preds &  targets).sum().item())
        self.tn += int((~preds & ~targets).sum().item())

    def compute(self) -> Dict[str, float]:
        eps = 1e-9
        precision = self.tp / (self.tp + self.fp + eps)
        recall    = self.tp / (self.tp + self.fn + eps)
        iou       = self.tp / (self.tp + self.fp + self.fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        return {
            "iou":       iou,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
        }



@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    x: torch.Tensor,           
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
    use_amp: bool = True,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:             
 
    model.eval()
    _, C, H, W = x.shape
    stride = tile_size - overlap

    pad_h = (stride - (H - tile_size) % stride) % stride if H > tile_size else max(0, tile_size - H)
    pad_w = (stride - (W - tile_size) % stride) % stride if W > tile_size else max(0, tile_size - W)
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, Hp, Wp = x.shape

    win_1d = torch.hann_window(tile_size, periodic=False, device=device)
    win_2d = (win_1d[:, None] * win_1d[None, :]).clamp(min=1e-3)

    probs   = torch.zeros((1, 1, Hp, Wp), device=device)
    weights = torch.zeros((1, 1, Hp, Wp), device=device)

    coords = []
    for y in range(0, Hp - tile_size + 1, stride):
        for xpos in range(0, Wp - tile_size + 1, stride):
            coords.append((y, xpos))

    x = x.to(device)
    autocast_ctx = torch.amp.autocast('cuda', enabled=use_amp)

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i + batch_size]
        tiles = torch.stack(
            [x[0, :, y:y + tile_size, xpos:xpos + tile_size] for (y, xpos) in batch_coords]
        )
        with autocast_ctx:
            logits = model(tiles)                       
        p = torch.sigmoid(logits).float()
        for k, (y, xpos) in enumerate(batch_coords):
            probs  [0, 0, y:y+tile_size, xpos:xpos+tile_size] += p[k, 0] * win_2d
            weights[0, 0, y:y+tile_size, xpos:xpos+tile_size] += win_2d

    probs = probs / weights.clamp(min=1e-6)
    if pad_h or pad_w:
        probs = probs[..., :H, :W]
    return probs



def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")