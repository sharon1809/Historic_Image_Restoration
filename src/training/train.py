import argparse
import os
import time
from pathlib import Path
import sys

# Ensure the project root is in the python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.models import RestorationUNet, CombinedLoss
from src.datasets.openphoto_dataset import OpenPhotoRestoreDataset

def main():
    parser = argparse.ArgumentParser(description="Train Historical Image Restoration Model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches for quick testing")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint dir
    checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from HuggingFace...")
    hf_dataset = load_dataset("joshuachin/openphoto-restore-dataset")

    train_dataset = OpenPhotoRestoreDataset(
        hf_dataset["train"],
        crop_size=256,
        training=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    print("Initializing model...")
    model = RestorationUNet(in_channels=3, out_channels=3, base_channels=32).to(device)
    criterion = CombinedLoss(lpips_weight=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 1

    if args.resume:
        resume_path = PROJECT_ROOT / args.resume
        if resume_path.exists():
            print(f"Loading checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Successfully resumed! Will start training at Epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint {resume_path} not found. Starting from scratch.")

    print(f"Starting training for {args.epochs} epoch(s)...")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break

            damaged = batch["damaged"].to(device)
            clean = batch["clean"].to(device)

            optimizer.zero_grad()
            prediction = model(damaged)
            loss = criterion(prediction, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        epoch_time = time.time() - start_time
        num_batches = args.max_batches if args.max_batches else len(train_loader)
        avg_loss = running_loss / num_batches
        
        print(f"==== Epoch {epoch} Summary ====")
        print(f"Average Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"unet_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}\n")

    print("Training complete!")

if __name__ == "__main__":
    main()
