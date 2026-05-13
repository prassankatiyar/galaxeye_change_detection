from __future__ import annotations
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import EOSARChangeDataset
from src.model import build_model
from src.utils import (
    seed_everything, sliding_window_predict, get_device, BinaryConfusionMatrix,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--weights",   required=True)
    p.add_argument("--split",     default="val")
    p.add_argument("--num",       type=int, default=12, help="number of samples to save")
    p.add_argument("--out_dir",   default=None)
    return p.parse_args()


def to_uint8(arr, low_p=2, high_p=98):
    lo = np.percentile(arr, low_p)
    hi = np.percentile(arr, high_p)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device()

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    seed_everything(cfg["experiment"]["seed"])

    dataset = EOSARChangeDataset(args.data_path, args.split, cfg, mode="eval")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    out_dir = Path(args.out_dir) if args.out_dir else \
              Path(args.weights).parent / "viz" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    use_amp = cfg["train"]["amp"]
    threshold = cfg["eval"]["threshold"]

    sample_results = []

    for i, batch in enumerate(tqdm(loader, desc="visualizing")):
        x   = batch["image"]
        tgt = batch["mask"].to(device)
        fname = batch["filename"][0]

        probs = sliding_window_predict(
            model, x,
            tile_size=cfg["eval"]["tile_size"],
            overlap=cfg["eval"]["tile_overlap"],
            batch_size=cfg["eval"]["batch_size"],
            use_amp=use_amp, device=device,
        )
        pred = (probs > threshold).long().squeeze().cpu().numpy()
        gt   = tgt.squeeze().cpu().numpy()
        prob = probs.squeeze().cpu().numpy()

        cm = BinaryConfusionMatrix()
        cm.update(torch.from_numpy(pred)[None], torch.from_numpy(gt)[None])
        m = cm.compute()
        sample_results.append((m["iou"], i, fname, x[0].numpy(), gt, pred, prob))

    sample_results.sort(key=lambda t: t[0])
    n = min(args.num, len(sample_results))
    half = n // 2
    chosen = sample_results[:half] + sample_results[-(n - half):]   

    for rank, (iou, i, fname, x_np, gt, pred, prob) in enumerate(chosen):
        eo  = to_uint8(np.transpose(x_np[:3], (1, 2, 0)))
        sar = to_uint8(x_np[3])

        err = np.zeros((*gt.shape, 3), dtype=np.uint8)
        err[(pred == 1) & (gt == 1)] = (0, 200, 0)        
        err[(pred == 1) & (gt == 0)] = (220, 0, 0)        
        err[(pred == 0) & (gt == 1)] = (0, 0, 220)        

        fig, axes = plt.subplots(1, 5, figsize=(22, 5))
        axes[0].imshow(eo);                       axes[0].set_title("EO (pre)")
        axes[1].imshow(sar, cmap="gray");         axes[1].set_title("SAR (post)")
        axes[2].imshow(gt, cmap="gray", vmin=0, vmax=1);   axes[2].set_title("Ground truth")
        axes[3].imshow(pred, cmap="gray", vmin=0, vmax=1); axes[3].set_title("Prediction")
        axes[4].imshow(err);                      axes[4].set_title("Error (TP/FP/FN)")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{fname}  |  IoU = {iou:.3f}", fontsize=12)
        plt.tight_layout()
        kind = "fail" if rank < half else "success"
        plt.savefig(out_dir / f"{kind}_{rank:02d}_iou{iou:.3f}_{fname.replace('.tif','')}.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"[done] saved {n} visualisations to {out_dir}")


if __name__ == "__main__":
    main()
