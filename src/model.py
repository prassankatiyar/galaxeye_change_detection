import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(cfg: dict) -> nn.Module:
    m_cfg = cfg["model"]

    
    model = smp.Unet(
        encoder_name=m_cfg["encoder"],
        encoder_weights=m_cfg["encoder_weights"],
        in_channels=m_cfg["in_channels"],
        classes=m_cfg["classes"],
    )

    if m_cfg["in_channels"] == 4 and m_cfg["encoder_weights"] == "imagenet":
        _adapt_first_conv_for_sar(model)

    return model


def _adapt_first_conv_for_sar(model: nn.Module) -> None:

    first_conv = None
    if hasattr(model, "encoder") and hasattr(model.encoder, "conv1"):
        first_conv = model.encoder.conv1
    else:
        for module in model.encoder.modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 4:
                first_conv = module
                break

    if first_conv is None or first_conv.in_channels != 4:
        return

    with torch.no_grad():
        w = first_conv.weight                      
        w[:, 3:4, :, :] = w[:, :3, :, :].mean(dim=1, keepdim=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
