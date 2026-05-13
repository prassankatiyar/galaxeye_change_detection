from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
import albumentations as A



def _read_tif(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:                       
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4) and arr.shape[0] < arr.shape[-1]:
        
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32)


def _normalize_eo(eo: np.ndarray) -> np.ndarray:
    if eo.max() > 1.5:                       
        eo = eo / 255.0
    return np.clip(eo, 0.0, 1.0)


def _normalize_sar(sar: np.ndarray) -> np.ndarray:

    if sar.max() > 1.5 and sar.max() <= 255.0 and sar.dtype != np.float32:
        return sar.astype(np.float32) / 255.0
    if sar.max() <= 1.5:
        return np.clip(sar, 0.0, 1.0)
    sar = np.log1p(np.maximum(sar, 0.0))
    lo, hi = np.percentile(sar, 1), np.percentile(sar, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(sar)
    return np.clip((sar - lo) / (hi - lo), 0.0, 1.0)


def _remap_labels(mask: np.ndarray) -> np.ndarray:
   
    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask >= 2] = 1
    return out




class EOSARChangeDataset(Dataset):


    def __init__(self, root: str, split: str, cfg: Dict, mode: str = "train"):
        assert mode in ("train", "eval")
        self.cfg = cfg
        self.mode = mode

        data_cfg = cfg["data"]
        split_dir = Path(root) / split
        self.pre_dir    = split_dir / data_cfg["pre_dir"]
        self.post_dir   = split_dir / data_cfg["post_dir"]
        self.target_dir = split_dir / data_cfg["target_dir"]

        if not self.pre_dir.is_dir():
            raise FileNotFoundError(f"Missing folder: {self.pre_dir}")

        self.filenames: List[str] = sorted(
            p.name for p in self.pre_dir.glob("*.tif")
            if (self.post_dir / p.name).exists() and (self.target_dir / p.name).exists()
        )
        if len(self.filenames) == 0:
            raise RuntimeError(f"No matched (pre/post/target) triplets found in {split_dir}")

        self.patch_size = cfg["train"]["patch_size"]
        self.pos_p = cfg["train"]["positive_oversample_p"]
        aug_cfg = cfg["augment"]

        self.geometric = A.Compose(
            [
                A.HorizontalFlip(p=aug_cfg["hflip_p"]),
                A.VerticalFlip(p=aug_cfg["vflip_p"]),
                A.RandomRotate90(p=aug_cfg["rot90_p"]),
            ],
            additional_targets={"image2": "image", "mask": "mask"},
        )
        self.eo_photometric = A.RandomBrightnessContrast(
            p=aug_cfg["eo_brightness_contrast_p"], brightness_limit=0.15, contrast_limit=0.15,
        )


    def __len__(self) -> int:
        return len(self.filenames)


    def _load_triplet(self, idx: int):
        fname = self.filenames[idx]
        eo  = _read_tif(self.pre_dir   / fname)            
        sar = _read_tif(self.post_dir  / fname)           
        tgt = _read_tif(self.target_dir / fname)[..., 0]   

        eo  = eo[..., :3]  if eo.shape[-1]  >= 3 else np.repeat(eo[..., :1],  3, axis=-1)
        sar = sar[..., :1] if sar.shape[-1] >= 1 else sar[..., None]

        eo  = _normalize_eo(eo)
        sar = _normalize_sar(sar)
        tgt = _remap_labels(tgt)
        return eo, sar, tgt, fname


    def _random_crop_biased(self, eo, sar, tgt):
        H, W = tgt.shape
        ps = self.patch_size
        if H < ps or W < ps:
            pad_h, pad_w = max(0, ps - H), max(0, ps - W)
            eo  = np.pad(eo,  ((0, pad_h), (0, pad_w), (0, 0)))
            sar = np.pad(sar, ((0, pad_h), (0, pad_w), (0, 0)))
            tgt = np.pad(tgt, ((0, pad_h), (0, pad_w)))
            H, W = tgt.shape

        want_positive = (np.random.rand() < self.pos_p) and (tgt.sum() > 0)
        for _ in range(20):                                  
            y = np.random.randint(0, H - ps + 1)
            x = np.random.randint(0, W - ps + 1)
            crop_tgt = tgt[y:y + ps, x:x + ps]
            if (not want_positive) or (crop_tgt.mean() > 0.01):
                break
        return eo[y:y+ps, x:x+ps], sar[y:y+ps, x:x+ps], tgt[y:y+ps, x:x+ps]


    def __getitem__(self, idx: int):
        eo, sar, tgt, fname = self._load_triplet(idx)

        if self.mode == "train":
            eo, sar, tgt = self._random_crop_biased(eo, sar, tgt)

            out = self.geometric(image=eo, image2=sar, mask=tgt)
            eo, sar, tgt = out["image"], out["image2"], out["mask"]

            eo = self.eo_photometric(image=eo)["image"]

        eo_t  = torch.from_numpy(np.ascontiguousarray(eo.transpose(2, 0, 1))).float()
        sar_t = torch.from_numpy(np.ascontiguousarray(sar.transpose(2, 0, 1))).float()
        tgt_t = torch.from_numpy(np.ascontiguousarray(tgt)).long()           

        x = torch.cat([eo_t, sar_t], dim=0)        
        return {"image": x, "mask": tgt_t, "filename": fname}
