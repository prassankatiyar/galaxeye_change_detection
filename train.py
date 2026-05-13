from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import EOSARChangeDataset
from src.model import build_model, count_parameters
from src.losses import build_loss
from src.utils import (
    seed_everything, BinaryConfusionMatrix, sliding_window_predict, get_device,
)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    return p.parse_args()



def make_lr_lambda(epochs: int, warmup: int, min_lr_ratio: float):
    def fn(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return fn




def evaluate(model, loader, device, threshold, tile_size, overlap, batch_size, use_amp):
    model.eval()
    cm = BinaryConfusionMatrix()
    pbar = tqdm(loader, desc="val", leave=False)
    for batch in pbar:
        x   = batch["image"]                  
        tgt = batch["mask"].to(device)        

        probs = sliding_window_predict(
            model, x, tile_size=tile_size, overlap=overlap,
            batch_size=batch_size, use_amp=use_amp, device=device,
        )
        preds = (probs > threshold).long().squeeze(1)   
        cm.update(preds, tgt)
    return cm.compute()




def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    seed_everything(cfg["experiment"]["seed"])
    device = get_device()
    print(f"[info] device = {device}")
    if device.type == "cuda":
        print(f"[info] gpu     = {torch.cuda.get_device_name(0)}")
        print(f"[info] vram    = {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

  
    root = cfg["data"]["root"]
    train_set = EOSARChangeDataset(root, cfg["data"]["train_split"], cfg, mode="train")
    val_set   = EOSARChangeDataset(root, cfg["data"]["val_split"],   cfg, mode="eval")
    print(f"[info] train images: {len(train_set)}  |  val images: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=cfg["data"]["pin_memory"],
    )


    model = build_model(cfg).to(device)
    print(f"[info] model params (trainable): {count_parameters(model):,}")

    criterion = build_loss(cfg).to(device)

    optim_cfg = cfg["train"]["optimizer"]
    optimizer = AdamW(
        model.parameters(),
        lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"],
    )

    epochs = cfg["train"]["epochs"]
    sched_cfg = cfg["train"]["scheduler"]
    min_lr_ratio = sched_cfg["min_lr"] / optim_cfg["lr"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(epochs, sched_cfg["warmup_epochs"], min_lr_ratio),
    )

    use_amp = cfg["train"]["amp"]
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)


    out_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out_dir / "config_used.yaml", "w"), sort_keys=False)
    writer = SummaryWriter(out_dir / "tb")

    best_iou = -1.0
    grad_clip = cfg["train"]["grad_clip"]


    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x   = batch["image"].to(device, non_blocking=True)
            tgt = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, tgt)

            if not torch.isfinite(loss):
                print(f"\n  [warn] non-finite loss={loss.item():.4f} — skipping batch")
                continue

            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()
        train_loss = epoch_loss / max(1, n_batches)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr",   optimizer.param_groups[0]["lr"], epoch)


        metrics = evaluate(
            model, val_loader, device,
            threshold=cfg["eval"]["threshold"],
            tile_size=cfg["eval"]["tile_size"],
            overlap=cfg["eval"]["tile_overlap"],
            batch_size=cfg["eval"]["batch_size"],
            use_amp=use_amp,
        )
        for k, v in metrics.items():
            if isinstance(v, float):
                writer.add_scalar(f"val/{k}", v, epoch)

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch+1:03d}] "
            f"train_loss={train_loss:.4f}  "
            f"val IoU={metrics['iou']:.4f}  P={metrics['precision']:.4f}  "
            f"R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
            f"({elapsed:.0f}s)"
        )


        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "config": cfg,
            "metrics": metrics,
        }
        torch.save(ckpt, out_dir / "last.pth")
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            torch.save(ckpt, out_dir / "best.pth")
            print(f" best IoU: {best_iou:.4f} — saved best.pth")

    writer.close()
    print(f"[done] best val IoU = {best_iou:.4f}")
    print(f"[done] checkpoints in {out_dir}")


if __name__ == "__main__":
    main()