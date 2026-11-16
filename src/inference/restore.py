import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.inference.pipeline import HistoricImageRestorationPipeline, RestorationConfig

def main():
    parser = argparse.ArgumentParser(description="Restore a single historical image")
    parser.add_argument("--image", type=str, required=True, help="Path to the damaged input image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the trained .pth checkpoint (optional)")
    parser.add_argument("--output", type=str, default="outputs/restored.jpg", help="Path to save the restored image")
    parser.add_argument("--passes", type=int, default=3, help="Number of times to pass the image through the model recursively to erase thick scratches")
    parser.add_argument("--pipeline", action="store_true", default=True, help="Run comprehensive restoration pipeline (defects, inpainting, color, sharpness)")
    parser.add_argument("--sensitivity", type=int, default=25, help="Scratch detection sensitivity (1-100)")
    parser.add_argument("--clahe", type=float, default=2.0, help="CLAHE contrast clip limit")
    parser.add_argument("--gray-world", type=float, default=0.4, help="Gray-world color balance weight (0.0 - 1.0)")
    parser.add_argument("--sharpen", type=float, default=1.0, help="Unsharp masking sharpness factor")
    
    args = parser.parse_args()

    p_img = Path(args.image)
    if p_img.exists():
        image_path = p_img.resolve()
    elif (PROJECT_ROOT / p_img).exists():
        image_path = (PROJECT_ROOT / p_img).resolve()
    else:
        raise FileNotFoundError(f"Input image not found: {args.image}")

    p_out = Path(args.output)
    if p_out.is_absolute():
        output_path = p_out
    elif p_out.parts and p_out.parts[0] == PROJECT_ROOT.name:
        output_path = (PROJECT_ROOT.parent / p_out).resolve()
    else:
        output_path = (PROJECT_ROOT / p_out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading restoration engine...")
    pipeline = HistoricImageRestorationPipeline(default_checkpoint=args.checkpoint)

    cfg = RestorationConfig(
        use_unet=(args.checkpoint is not None and pipeline.unet_model is not None),
        unet_passes=args.passes,
        scratch_sensitivity=args.sensitivity,
        clahe_clip_limit=args.clahe,
        gray_world_weight=args.gray_world,
        sharpen_amount=args.sharpen,
    )

    print(f"Restoring image from {image_path}...")
    result = pipeline.restore(image_path, config=cfg)

    print(f"Saving restored image to {output_path}...")
    result["restored_pil"].save(output_path)

    m = result["metrics"]
    print(f"\nRestoration Complete!")
    print(f"  - Processing Time: {m['processing_time_s']:.3f} s")
    print(f"  - Defects Repaired: {m['defect_percentage']:.2f}% of image")
    print(f"  - Sharpness Gain: {m['sharpness_gain_pct']:.1f}%")
    print(f"  - Output saved to: {output_path}")

if __name__ == "__main__":
    main()
