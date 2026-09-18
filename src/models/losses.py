import torch
import torch.nn as nn
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure

class L1ReconstructionLoss(nn.Module):
    """
    Mean absolute error between restored and clean images.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(prediction, target)

class CharbonnierLoss(nn.Module):
    """
    A robust alternative to L1/L2 loss that handles outliers better.
    Often used in image restoration tasks instead of plain L1.
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

class CombinedLoss(nn.Module):
    """
    Combined loss function supporting 4 experimental configurations:
    A: L1 + LPIPS (Baseline)
    B: Charbonnier + LPIPS
    C: L1 + SSIM
    D: L1 + SSIM + LPIPS
    """
    def __init__(self, mode="A", lpips_weight=0.5, ssim_weight=0.1, device="cpu"):
        super().__init__()
        self.mode = mode.upper()
        self.lpips_weight = lpips_weight
        self.ssim_weight = ssim_weight
        
        self.l1_loss = nn.L1Loss()
        self.charbonnier_loss = CharbonnierLoss()
        
        if self.mode in ["A", "B", "D"]:
            self.perceptual_loss = lpips.LPIPS(net='vgg').to(device).eval()
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False
                
        if self.mode in ["C", "D"]:
            # Note: SSIM measures similarity (1 is perfect). 
            # We minimize (1 - SSIM).
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        
        # 1. Base Loss (L1 or Charbonnier)
        if self.mode in ["A", "C", "D"]:
            loss += self.l1_loss(prediction, target)
        elif self.mode == "B":
            loss += self.charbonnier_loss(prediction, target)
        else:
            raise ValueError(f"Unknown loss mode: {self.mode}")
            
        # 2. LPIPS Loss
        if self.mode in ["A", "B", "D"]:
            pred_scaled = prediction * 2.0 - 1.0
            target_scaled = target * 2.0 - 1.0
            p_loss = self.perceptual_loss(pred_scaled, target_scaled).mean()
            loss += self.lpips_weight * p_loss
            
        # 3. SSIM Loss
        if self.mode in ["C", "D"]:
            # SSIM can occasionally complain about grads if not careful, 
            # but torchmetrics SSIM handles differentiation.
            ssim_loss = 1.0 - self.ssim_metric(prediction, target)
            loss += self.ssim_weight * ssim_loss
            
        return loss