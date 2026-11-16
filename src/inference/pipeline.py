"""
Historic Image Restoration Pipeline
====================================
A production-grade restoration engine for degraded, scratched, faded, and
noisy historic photographs.

Capabilities:
- Smart Defect Detection: local brightness-anomaly + connected-component
  filtering so only real tears/scratches are flagged (not texture noise).
- High-Performance Defect Inpainting (Telea fast-marching, Navier-Stokes)
- Photochemical Degradation Correction (CLAHE, Gray-World white balance)
- Film Grain & Sensor Denoising (edge-preserving bilateral filtering)
- Optical Detail & Lens Sharpness Recovery (Unsharp masking)
- Neural U-Net Restoration Integration (optional, with checkpoint)
- Cloud AI Face Restoration (CodeFormer API, graceful offline fallback)
- Real-Time Quantitative Quality Metrics (PSNR, SSIM, Sharpness, etc.)
"""

import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

# Prevent Matplotlib cache delays
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import cv2
import numpy as np
from PIL import Image

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from src.models.unet import RestorationUNet


@dataclass
class RestorationConfig:
    """Configuration for the Historic Image Restoration Pipeline."""

    # Profile preset
    profile: str = "balanced"   # balanced | scratch_only | color_only | clarity_only | custom

    # ── 1. Defect & Scratch Detection ────────────────────────────────────────
    detect_bright_scratches: bool = True
    detect_dark_cracks: bool = False      # OFF by default – causes false positives on textured dark images
    scratch_sensitivity: int = 60         # 1-100; higher = more sensitive
    scratch_kernel_size: int = 7          # morphological kernel (used for dark-crack path only)
    scratch_dilation_radius: int = 2      # dilation after detection to cover fringe pixels

    # ── 2. Defect Inpainting ─────────────────────────────────────────────────
    enable_inpainting: bool = True
    inpaint_method: str = "telea"         # 'telea' (Fast Marching) | 'ns' (Navier-Stokes)
    inpaint_radius: int = 5
    inpaint_passes: int = 1               # single pass prevents smearing

    # ── 3. Color & Tone Correction ───────────────────────────────────────────
    enable_color_correction: bool = True
    clahe_clip_limit: float = 1.5         # 0.0 to 5.0 – lower = subtler contrast lift
    clahe_grid_size: int = 8
    gray_world_weight: float = 0.3        # 0.0 to 1.0 – white-balance strength
    saturation_multiplier: float = 1.05

    # ── 4. Denoising ─────────────────────────────────────────────────────────
    enable_denoising: bool = True
    denoise_strength: int = 4             # 1-15 bilateral sigma

    # ── 5. Sharpness Recovery ────────────────────────────────────────────────
    enable_sharpening: bool = True
    sharpen_amount: float = 0.9           # 0.0 to 3.0
    sharpen_radius: float = 1.0

    # ── 6. Neural U-Net (Optional) ───────────────────────────────────────────
    use_unet: bool = False
    unet_checkpoint: Optional[str] = None
    unet_passes: int = 3

    # ── 7. CodeFormer AI (Optional Remote API) ───────────────────────────────
    use_codeformer: bool = False
    hf_token: Optional[str] = None
    codeformer_fidelity: float = 0.7
    codeformer_upscale: int = 1


class HistoricImageRestorationPipeline:
    """
    Unified restoration engine: multi-stage classical CV + optional deep learning.
    """

    def __init__(self, default_checkpoint: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet_model: Optional[RestorationUNet] = None
        self.loaded_checkpoint_path: Optional[str] = None

        if default_checkpoint and Path(default_checkpoint).exists():
            self.load_unet(default_checkpoint)
        else:
            self._search_default_checkpoints()

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _search_default_checkpoints(self):
        candidates = [
            PROJECT_ROOT / "checkpoints" / "best_model.pth",
            PROJECT_ROOT / "checkpoints" / "latest_model.pth",
            PROJECT_ROOT / "best_model.pth",
        ]
        for c in candidates:
            if c.exists():
                try:
                    self.load_unet(str(c))
                    break
                except Exception as e:
                    print(f"Notice: checkpoint at {c} failed to load: {e}")

    def load_unet(self, checkpoint_path: str) -> bool:
        p = Path(checkpoint_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            print(f"Warning: Checkpoint not found at {p}")
            return False
        try:
            model = RestorationUNet(in_channels=3, out_channels=3, base_channels=32)
            ckpt = torch.load(str(p), map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self.unet_model = model
            self.loaded_checkpoint_path = str(p)
            print(f"Loaded U-Net checkpoint from {p}")
            return True
        except Exception as e:
            print(f"Error loading U-Net: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Histogram matching (pure NumPy)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def match_histograms_np(source: np.ndarray, template: np.ndarray) -> np.ndarray:
        matched = np.zeros_like(source)
        for c in range(source.shape[-1]):
            s_ch = source[..., c]
            t_ch = template[..., c]
            s_vals, bin_idx, s_counts = np.unique(s_ch.ravel(), return_inverse=True, return_counts=True)
            t_vals, t_counts = np.unique(t_ch.ravel(), return_counts=True)
            s_q = np.cumsum(s_counts).astype(np.float64) / s_ch.size
            t_q = np.cumsum(t_counts).astype(np.float64) / t_ch.size
            matched[..., c] = np.interp(s_q, t_q, t_vals)[bin_idx].reshape(s_ch.shape)
        return matched.astype(np.uint8)

    # ─────────────────────────────────────────────────────────────────────────
    # SMART Defect Detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_defects(
        self,
        img_bgr: np.ndarray,
        detect_bright: bool = True,
        detect_dark: bool = False,
        sensitivity: int = 60,
        kernel_size: int = 7,
        dilation_radius: int = 2,
    ) -> np.ndarray:
        """
        Smart defect detection that isolates real tears/scratches without
        flagging natural image texture.

        Strategy for BRIGHT scratches (white tears, paper damage):
          - Local brightness anomaly: pixel must be significantly BRIGHTER
            than its local neighbourhood (21×21 Gaussian blur reference)
            AND above an absolute brightness floor.
          - Connected-component filtering: tiny isolated blobs (< min_area)
            are discarded as noise.

        Strategy for DARK cracks (optional, off by default):
          - Morphological Black-Hat to find locally dark elongated regions.
          - Same CC filter applied.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # ── Sensitivity → thresholds ──────────────────────────────────────
        # sensitivity 1  → very conservative (only extreme tears)
        # sensitivity 100 → aggressive (most bright regions)
        #
        # absolute floor: pixels below this cannot be a "white tear"
        abs_floor = max(130, int(255 - sensitivity * 1.3))   # e.g. s=60 → 177
        # local delta: how much brighter than neighbourhood must the pixel be
        local_delta = max(10, int(55 - sensitivity * 0.4))   # e.g. s=60 → 31
        # minimum connected-component area (fraction of total pixels)
        min_area = max(4, int(gray.size * 0.0003))           # ~15 px for 228×230

        # ── Bright scratch detection ──────────────────────────────────────
        if detect_bright:
            # Reference: smooth local background
            blur = cv2.GaussianBlur(gray, (21, 21), 0).astype(np.float32)
            diff = gray.astype(np.float32) - blur

            bright = ((diff > local_delta) & (gray.astype(np.float32) > abs_floor)).astype(np.uint8) * 255

            # CC filter: drop tiny specks
            bright = self._filter_components(bright, min_area)
            mask = cv2.bitwise_or(mask, bright)

        # ── Dark crack detection (optional) ──────────────────────────────
        if detect_dark:
            k_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
            k_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_rect)
            dark_thresh = max(5, int(55 - sensitivity * 0.4))
            _, dark_mask = cv2.threshold(blackhat, dark_thresh, 255, cv2.THRESH_BINARY)
            dark_mask = self._filter_components(dark_mask, min_area)
            mask = cv2.bitwise_or(mask, dark_mask)

        # ── Dilate to cover fringe gradient around scratch edges ──────────
        if dilation_radius > 0 and np.any(mask > 0):
            d = dilation_radius * 2 + 1
            dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
            mask = cv2.dilate(mask, dk, iterations=1)

        return mask

    @staticmethod
    def _filter_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
        """Remove connected components smaller than min_area pixels."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        filtered = np.zeros_like(binary_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == i] = 255
        return filtered

    # ─────────────────────────────────────────────────────────────────────────
    # Inpainting
    # ─────────────────────────────────────────────────────────────────────────

    def inpaint_defects(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        method: str = "telea",
        radius: int = 5,
        passes: int = 1,
    ) -> np.ndarray:
        """Seamlessly inpaints detected defects (Telea or Navier-Stokes)."""
        if np.sum(mask > 0) == 0:
            return img_bgr.copy()

        flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
        current = img_bgr.copy()
        for _ in range(max(1, passes)):
            current = cv2.inpaint(current, mask, max(1, radius), flag)
        return current

    # ─────────────────────────────────────────────────────────────────────────
    # Color & Tone Enhancement
    # ─────────────────────────────────────────────────────────────────────────

    def enhance_tones_and_color(
        self,
        img_bgr: np.ndarray,
        clahe_clip: float = 1.5,
        grid_size: int = 8,
        gray_world_weight: float = 0.3,
        saturation_multiplier: float = 1.05,
    ) -> np.ndarray:
        """
        Recovers dynamic range and neutralizes aged colour casts.
        Uses CLAHE on the LAB L-channel (preserves hue) and Gray-World
        white balance (blended so it doesn't over-shift).
        """
        # 1. CLAHE on luminance channel
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        if clahe_clip > 0:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(grid_size, grid_size))
            l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # 2. Blended Gray-World white balance
        if gray_world_weight > 0:
            b_m = np.mean(enhanced[:, :, 0])
            g_m = np.mean(enhanced[:, :, 1])
            r_m = np.mean(enhanced[:, :, 2])
            avg = (b_m + g_m + r_m) / 3.0
            bal = enhanced.astype(np.float32)
            bal[:, :, 0] = np.clip(bal[:, :, 0] * (avg / (b_m + 1e-5)), 0, 255)
            bal[:, :, 1] = np.clip(bal[:, :, 1] * (avg / (g_m + 1e-5)), 0, 255)
            bal[:, :, 2] = np.clip(bal[:, :, 2] * (avg / (r_m + 1e-5)), 0, 255)
            enhanced = cv2.addWeighted(
                enhanced, 1.0 - gray_world_weight,
                bal.astype(np.uint8), gray_world_weight, 0
            )

        # 3. Subtle saturation bump
        if abs(saturation_multiplier - 1.0) > 0.01:
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_multiplier, 0, 255)
            enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return enhanced

    # ─────────────────────────────────────────────────────────────────────────
    # Denoising & Sharpening
    # ─────────────────────────────────────────────────────────────────────────

    def denoise(self, img_bgr: np.ndarray, strength: int = 4) -> np.ndarray:
        """Edge-preserving bilateral filter to suppress film grain."""
        if strength <= 0:
            return img_bgr.copy()
        sigma = strength * 7
        return cv2.bilateralFilter(img_bgr, 7, sigma, sigma // 2)

    def sharpen(self, img_bgr: np.ndarray, amount: float = 0.9, radius: float = 1.0) -> np.ndarray:
        """Unsharp masking to restore lens/optical softness."""
        if amount <= 0:
            return img_bgr.copy()
        k = int(radius * 4) | 1
        blurred = cv2.GaussianBlur(img_bgr, (k, k), radius)
        return np.clip(
            cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0),
            0, 255
        ).astype(np.uint8)

    # ─────────────────────────────────────────────────────────────────────────
    # Neural U-Net pass
    # ─────────────────────────────────────────────────────────────────────────

    def run_unet(self, img_bgr: np.ndarray, passes: int = 3) -> np.ndarray:
        if self.unet_model is None:
            raise RuntimeError("U-Net not loaded.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        _, _, h, w = t.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        padded = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
        with torch.no_grad():
            curr = padded
            for _ in range(max(1, passes)):
                curr = self.unet_model(curr)
        unpadded = curr[:, :, :h, :w].squeeze(0).clamp(0, 1)
        out_rgb = (unpadded.permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        matched = self.match_histograms_np(out_rgb, img_rgb)
        return cv2.cvtColor(matched, cv2.COLOR_RGB2BGR)

    # ─────────────────────────────────────────────────────────────────────────
    # CodeFormer (remote API)
    # ─────────────────────────────────────────────────────────────────────────

    def run_codeformer(
        self,
        img_bgr: np.ndarray,
        token: Optional[str] = None,
        fidelity: float = 0.7,
        upscale: int = 1,
    ) -> Tuple[np.ndarray, str]:
        try:
            import tempfile
            from gradio_client import Client, handle_file
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            cv2.imwrite(tmp.name, img_bgr)
            tmp.close()
            client = Client("sczhou/CodeFormer", token=token)
            result = client.predict(
                image=handle_file(tmp.name),
                face_align=True,
                background_enhance=True,
                face_upsample=True,
                upscale=upscale,
                codeformer_fidelity=fidelity,
                api_name="/inference",
            )
            restored = cv2.imread(result[0])
            try:
                os.remove(tmp.name)
            except OSError:
                pass
            return restored, "CodeFormer AI enhancement successful!"
        except Exception as e:
            return img_bgr.copy(), f"CodeFormer unavailable ({e}). Using local restoration."

    # ─────────────────────────────────────────────────────────────────────────
    # Quality Metrics
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_sharpness_score(img_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def calculate_contrast_ratio(img_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        c_max, c_min = np.max(gray), np.min(gray)
        denom = c_max + c_min
        return float((c_max - c_min) / denom) if denom > 0 else 0.0

    @staticmethod
    def calculate_metrics(restored: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
        if restored.shape != reference.shape:
            restored = cv2.resize(restored, (reference.shape[1], reference.shape[0]))
        t = restored.astype(np.float64)
        r = reference.astype(np.float64)
        mse = float(np.mean((t - r) ** 2))
        mae = float(np.mean(np.abs(t - r)))
        psnr = float("inf") if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        mu1 = cv2.GaussianBlur(t, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(r, (11, 11), 1.5)
        s1 = cv2.GaussianBlur(t ** 2, (11, 11), 1.5) - mu1 ** 2
        s2 = cv2.GaussianBlur(r ** 2, (11, 11), 1.5) - mu2 ** 2
        s12 = cv2.GaussianBlur(t * r, (11, 11), 1.5) - mu1 * mu2
        ssim = float(np.mean(
            ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) /
            ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
        ))
        return {"PSNR": psnr, "SSIM": ssim, "MAE": mae, "MSE": mse}

    # ─────────────────────────────────────────────────────────────────────────
    # Main restore() entry point
    # ─────────────────────────────────────────────────────────────────────────

    def restore(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        config: Optional[RestorationConfig] = None,
        reference_image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
    ) -> Dict[str, Any]:
        """
        Full restoration workflow. Returns restored image + intermediates + metrics.
        """
        start = time.time()
        cfg = config or RestorationConfig()

        # ── Load input ────────────────────────────────────────────────────
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                raise FileNotFoundError(f"Could not load image: {image_input}")
        elif isinstance(image_input, Image.Image):
            img_bgr = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

        original_bgr = img_bgr.copy()
        current_bgr = img_bgr.copy()
        intermediates: Dict[str, np.ndarray] = {"original": original_bgr}

        # ── Step 1: Optional Neural U-Net ─────────────────────────────────
        if cfg.use_unet and self.unet_model is not None:
            try:
                current_bgr = self.run_unet(current_bgr, passes=cfg.unet_passes)
                intermediates["unet"] = current_bgr.copy()
            except Exception as e:
                print(f"U-Net pass skipped: {e}")

        # ── Step 2: Smart Defect Detection ────────────────────────────────
        defect_mask = self.detect_defects(
            current_bgr,
            detect_bright=cfg.detect_bright_scratches,
            detect_dark=cfg.detect_dark_cracks,
            sensitivity=cfg.scratch_sensitivity,
            kernel_size=cfg.scratch_kernel_size,
            dilation_radius=cfg.scratch_dilation_radius,
        )
        intermediates["defect_mask"] = defect_mask.copy()

        defect_px = int(np.sum(defect_mask > 0))
        defect_pct = float(defect_px / defect_mask.size * 100)

        # Safety valve: if more than 20% of the image is flagged, the
        # detector has gone haywire – skip inpainting to avoid making
        # things worse.
        if defect_pct > 20.0:
            print(
                f"[Pipeline] WARNING: {defect_pct:.1f}% of pixels flagged "
                f"({defect_px} px). Defect mask looks wrong – inpainting "
                f"skipped to protect image quality. Try lowering sensitivity."
            )
            defect_mask_for_inpaint = np.zeros_like(defect_mask)
        else:
            defect_mask_for_inpaint = defect_mask

        # ── Step 3: Defect Inpainting ─────────────────────────────────────
        if cfg.enable_inpainting and np.any(defect_mask_for_inpaint > 0):
            current_bgr = self.inpaint_defects(
                current_bgr,
                defect_mask_for_inpaint,
                method=cfg.inpaint_method,
                radius=cfg.inpaint_radius,
                passes=cfg.inpaint_passes,
            )
            intermediates["inpainted"] = current_bgr.copy()

        # ── Step 4: Color & Tone Correction ──────────────────────────────
        if cfg.enable_color_correction:
            current_bgr = self.enhance_tones_and_color(
                current_bgr,
                clahe_clip=cfg.clahe_clip_limit,
                grid_size=cfg.clahe_grid_size,
                gray_world_weight=cfg.gray_world_weight,
                saturation_multiplier=cfg.saturation_multiplier,
            )
            intermediates["color_restored"] = current_bgr.copy()

        # ── Step 5: Denoising ─────────────────────────────────────────────
        if cfg.enable_denoising:
            current_bgr = self.denoise(current_bgr, strength=cfg.denoise_strength)
            intermediates["denoised"] = current_bgr.copy()

        # ── Step 6: Sharpness Recovery ────────────────────────────────────
        if cfg.enable_sharpening:
            current_bgr = self.sharpen(
                current_bgr, amount=cfg.sharpen_amount, radius=cfg.sharpen_radius
            )
            intermediates["sharpened"] = current_bgr.copy()

        # ── Step 7: Optional CodeFormer ───────────────────────────────────
        codeformer_status = "Skipped"
        if cfg.use_codeformer:
            current_bgr, codeformer_status = self.run_codeformer(
                current_bgr,
                token=cfg.hf_token,
                fidelity=cfg.codeformer_fidelity,
                upscale=cfg.codeformer_upscale,
            )
            intermediates["codeformer"] = current_bgr.copy()

        # ── Metrics ───────────────────────────────────────────────────────
        duration = time.time() - start
        intermediates["difference"] = cv2.absdiff(current_bgr, original_bgr)

        metrics: Dict[str, Any] = {
            "processing_time_s": duration,
            "defect_pixel_count": defect_px,
            "defect_percentage": defect_pct,
            "sharpness_original": self.calculate_sharpness_score(original_bgr),
            "sharpness_restored": self.calculate_sharpness_score(current_bgr),
            "contrast_original": self.calculate_contrast_ratio(original_bgr),
            "contrast_restored": self.calculate_contrast_ratio(current_bgr),
            "codeformer_status": codeformer_status,
        }
        metrics["sharpness_gain_pct"] = (
            (metrics["sharpness_restored"] - metrics["sharpness_original"])
            / max(1e-5, metrics["sharpness_original"])
        ) * 100

        if reference_image is not None:
            if isinstance(reference_image, (str, Path)):
                ref_bgr = cv2.imread(str(reference_image))
            elif isinstance(reference_image, Image.Image):
                ref_bgr = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
            else:
                ref_bgr = reference_image
            if ref_bgr is not None:
                metrics.update(self.calculate_metrics(current_bgr, ref_bgr))

        return {
            "restored_bgr": current_bgr,
            "restored_rgb": cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB),
            "restored_pil": Image.fromarray(cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)),
            "intermediates": intermediates,
            "metrics": metrics,
            "config": cfg,
        }
