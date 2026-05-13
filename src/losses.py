import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
 

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs  = torch.sigmoid(logits.float()).squeeze(1)  
        target = target.float()

        intersection = (probs * target).sum(dim=(1, 2))
        denom = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_w = bce_weight
        self.dice_w = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        bce  = self.bce(logits_f.squeeze(1), target.float())
        dice = self.dice(logits_f, target)
        return self.bce_w * bce + self.dice_w * dice


def build_loss(cfg: dict) -> nn.Module:
    L = cfg["train"]["loss"]
    return BCEDiceLoss(bce_weight=L["bce_weight"], dice_weight=L["dice_weight"])