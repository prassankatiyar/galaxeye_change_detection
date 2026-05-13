from __future__ import annotations
import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import EOSARChangeDataset
from src.model import build_model
from src.utils import (
    seed_everything, BinaryConfusionMatrix, sliding_window_predict, get_device,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True,
                   help="Root directory containing the split folder (e.g. D:/galaxeye_data)")
    p.add_argument("--weights",   type=str, required=True,
                   help="Path to checkpoint .pth file")
    p.add_argument("--split",     type=str, default="test",
                   help="Sub-folder under data_path: train / val / test")
    p.add_argument("--config",    type=str, default=None,
                   help="(Optional) override config; otherwise read from checkpoint")
    p.add_argument("--threshold", type=float, default=None,
                   help="(Optional) override sigmoid threshold")
    p.add_argument("--out",       type=str, default=None,
                   help="(Optional) write metrics JSON here")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device()


    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    cfg = yaml.safe_load(open(args.config, "r")) if args.config else ckpt["config"]
    seed_everything(cfg["experiment"]["seed"])

    threshold = args.threshold if args.threshold is not None else cfg["eval"]["threshold"]

    dataset = EOSARChangeDataset(args.data_path, args.split, cfg, mode="eval")
    print(f"[info] {args.split} images: {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

   
    cm = BinaryConfusionMatrix()
    use_amp = cfg["train"]["amp"]

    for batch in tqdm(loader, desc=f"eval/{args.split}"):
        x   = batch["image"]
        tgt = batch["mask"].to(device)

        probs = sliding_window_predict(
            model, x,
            tile_size=cfg["eval"]["tile_size"],
            overlap=cfg["eval"]["tile_overlap"],
            batch_size=cfg["eval"]["batch_size"],
            use_amp=use_amp, device=device,
        )
        preds = (probs > threshold).long().squeeze(1)
        cm.update(preds, tgt)

    metrics = cm.compute()

 
    print("\n" + "=" * 50)
    print(f"  Results on `{args.split}` split (change class)")
    print("=" * 50)
    print(f"  IoU       : {metrics['iou']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print()
    print("  Confusion matrix (pixels)")
    print("                Pred 0 (no-change)   Pred 1 (change)")
    print(f"  GT 0          {metrics['tn']:>14d}     {metrics['fp']:>14d}")
    print(f"  GT 1          {metrics['fn']:>14d}     {metrics['tp']:>14d}")
    print("=" * 50 + "\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({k: (float(v) if isinstance(v, float) else int(v)) for k, v in metrics.items()},
                  open(args.out, "w"), indent=2)
        print(f"[info] metrics written to {args.out}")


if __name__ == "__main__":
    main()
