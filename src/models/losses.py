import torch
import torch.nn as nn
import lpips


class L1ReconstructionLoss(nn.Module):
    """
    Mean absolute error between restored and clean images.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss(prediction, target)


class CombinedLoss(nn.Module):
    """
    Combined L1 and Perceptual (LPIPS) loss.
    L1 provides overall structure while LPIPS ensures the image 
    looks sharp and perceptually realistic to the human eye.
    """
    def __init__(self, lpips_weight: float = 0.5):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        
        # VGG-based LPIPS is the standard for image restoration
        # Using eval() to prevent gradients from updating the VGG network
        self.perceptual_loss = lpips.LPIPS(net='vgg').eval()
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False
            
        self.lpips_weight = lpips_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # LPIPS expects inputs in range [-1, 1], but our images are [0, 1]
        # We need to scale them for LPIPS to work correctly
        pred_scaled = prediction * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        
        l1 = self.l1_loss(prediction, target)
        
        # LPIPS returns a tensor of shape [B, 1, 1, 1], we mean() it to get a scalar
        p_loss = self.perceptual_loss(pred_scaled, target_scaled).mean()
        
        return l1 + (self.lpips_weight * p_loss)