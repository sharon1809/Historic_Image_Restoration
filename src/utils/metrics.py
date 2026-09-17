import torch
import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

class RestorationMetrics:
    def __init__(self, device):
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.mae = MeanAbsoluteError().to(device)
        self.mse = MeanSquaredError().to(device)
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(device).eval()
        
        # Don't update LPIPS weights
        for param in self.lpips_vgg.parameters():
            param.requires_grad = False
            
    def compute(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds and targets should be [B, C, H, W] in range [0, 1]
        """
        with torch.no_grad():
            psnr_val = self.psnr(preds, targets).item()
            ssim_val = self.ssim(preds, targets).item()
            
            # MAE/MSE expect flattened or matching shapes
            mae_val = self.mae(preds.flatten(), targets.flatten()).item()
            mse_val = self.mse(preds.flatten(), targets.flatten()).item()
            
            # LPIPS expects [-1, 1]
            preds_scaled = preds * 2.0 - 1.0
            targets_scaled = targets * 2.0 - 1.0
            lpips_val = self.lpips_vgg(preds_scaled, targets_scaled).mean().item()
            
        return {
            "PSNR": psnr_val,
            "SSIM": ssim_val,
            "MAE": mae_val,
            "MSE": mse_val,
            "LPIPS": lpips_val
        }
