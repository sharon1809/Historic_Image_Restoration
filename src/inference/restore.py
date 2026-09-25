import argparse
import os
import sys
from pathlib import Path

# Ensure the project root is in the python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.models import RestorationUNet

def load_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    tensor = transform(image).unsqueeze(0) # Add batch dimension [1, 3, H, W]
    return tensor

def pad_image(tensor: torch.Tensor, multiple: int = 16):
    """
    Pad the image to ensure dimensions are divisible by `multiple` (required for U-Net).
    Returns the padded tensor and the original padding applied.
    """
    _, _, h, w = tensor.shape
    
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    # padding format for F.pad is (left, right, top, bottom)
    padding = (0, pad_w, 0, pad_h)
    
    padded_tensor = F.pad(tensor, padding, mode='reflect')
    return padded_tensor, padding

def unpad_image(tensor: torch.Tensor, padding: tuple) -> torch.Tensor:
    """Remove the padding applied previously."""
    _, right_pad, _, bottom_pad = padding
    _, _, h, w = tensor.shape
    
    unpadded = tensor[:, :, :h - bottom_pad, :w - right_pad]
    return unpadded

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.squeeze(0).cpu().clamp(0, 1)
    transform = transforms.ToPILImage()
    return transform(tensor)

def main():
    parser = argparse.ArgumentParser(description="Restore a single historical image")
    parser.add_argument("--image", type=str, required=True, help="Path to the damaged input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .pth checkpoint")
    parser.add_argument("--output", type=str, default="outputs/restored.jpg", help="Path to save the restored image")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure output dir exists
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = RestorationUNet(in_channels=3, out_channels=3, base_channels=32)
    
    # Load checkpoint
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loading image from {args.image}...")
    input_tensor = load_image(PROJECT_ROOT / args.image).to(device)
    
    # Pad image for U-Net downsampling dimensions
    padded_tensor, padding = pad_image(input_tensor, multiple=16)

    print("Restoring image...")
    with torch.no_grad():
        output_tensor = model(padded_tensor)

    # Unpad back to original size
    output_tensor = unpad_image(output_tensor, padding)

    print(f"Saving restored image to {output_path}...")
    output_image = tensor_to_image(output_tensor)
    
    # --- POST-PROCESSING ---
    # The U-Net tends to shift colors/brightness (darkening). We can perfectly fix this 
    # by matching the color histogram of the output to the original input image.
    try:
        import numpy as np
        from skimage import exposure
        
        # Convert both to numpy arrays
        original_np = np.array(Image.open(PROJECT_ROOT / args.image).convert("RGB"))
        restored_np = np.array(output_image)
        
        # Match histograms
        matched_np = exposure.match_histograms(restored_np, original_np, channel_axis=-1)
        
        # Convert back to PIL Image
        output_image = Image.fromarray(matched_np.astype(np.uint8))
        print("Successfully corrected color and brightness using histogram matching!")
    except Exception as e:
        print(f"Warning: Histogram matching failed ({e}). Saving raw output instead.")
    # -----------------------

    output_image.save(output_path)
    
    print("Done!")

if __name__ == "__main__":
    main()
