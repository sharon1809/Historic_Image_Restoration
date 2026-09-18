import argparse
import os
import time
import csv
import random
import numpy as np
from pathlib import Path
import sys

# Ensure the project root is in the python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets import load_dataset

from src.models import RestorationUNet, CombinedLoss
from src.datasets.openphoto_dataset import OpenPhotoRestoreDataset
from src.utils.metrics import RestorationMetrics

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Train Historical Image Restoration Model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches for quick testing")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--loss-mode", type=str, default="A", choices=["A", "B", "C", "D"], help="Loss configuration mode (A, B, C, D)")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint and outputs dirs
    checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    val_output_dir = PROJECT_ROOT / "outputs" / "val_images"
    val_output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from HuggingFace...")
    hf_dataset = load_dataset("joshuachin/openphoto-restore-dataset")
    
    # Create deterministic split
    splits = hf_dataset["train"].train_test_split(test_size=0.05, seed=42)

    train_dataset = OpenPhotoRestoreDataset(
        splits["train"],
        crop_size=256,
        training=True
    )
    
    val_dataset = OpenPhotoRestoreDataset(
        splits["test"],
        crop_size=256,
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print("Initializing model...")
    model = RestorationUNet(in_channels=3, out_channels=3, base_channels=32).to(device)
    criterion = CombinedLoss(mode=args.loss_mode, lpips_weight=0.5, ssim_weight=0.1, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    metrics_calculator = RestorationMetrics(device)

    start_epoch = 1
    best_val_loss = float('inf')

    early_stopper = EarlyStopping(patience=args.patience)

    if args.resume:
        resume_path = PROJECT_ROOT / args.resume
        if resume_path.exists():
            print(f"Loading checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
                early_stopper.best_loss = best_val_loss
            print(f"Successfully resumed! Will start training at Epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint {resume_path} not found. Starting from scratch.")

    print(f"Starting training for {args.epochs} epoch(s)...")

    # CSV for logging
    csv_file = PROJECT_ROOT / "metrics_log.csv"
    write_header = not csv_file.exists()
    
    with open(csv_file, mode='a', newline='') as f:
        csv_writer = csv.writer(f)
        if write_header:
            csv_writer.writerow(["Epoch", "Train Loss", "Val Loss", "PSNR", "SSIM", "MAE", "MSE", "LPIPS", "Time(s)"])

        # Fetch a fixed batch for visual validation
        fixed_val_batch = next(iter(val_loader))
        fixed_val_damaged = fixed_val_batch["damaged"].to(device)
        fixed_val_clean = fixed_val_batch["clean"].to(device)

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
                    print(f"Epoch [{epoch}/{args.epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Train Loss: {loss.item():.4f}")

            epoch_time = time.time() - start_time
            num_batches = args.max_batches if args.max_batches else len(train_loader)
            avg_train_loss = running_loss / num_batches
            
            # Validation Loop
            model.eval()
            val_loss = 0.0
            val_metrics = {"PSNR": 0.0, "SSIM": 0.0, "MAE": 0.0, "MSE": 0.0, "LPIPS": 0.0}
            val_batches = args.max_batches if args.max_batches else len(val_loader)
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if args.max_batches and batch_idx >= args.max_batches:
                        break
                        
                    damaged = batch["damaged"].to(device)
                    clean = batch["clean"].to(device)
                    
                    prediction = model(damaged)
                    loss = criterion(prediction, clean)
                    val_loss += loss.item()
                    
                    # Compute metrics
                    batch_metrics = metrics_calculator.compute(prediction, clean)
                    for k in val_metrics:
                        val_metrics[k] += batch_metrics[k]
                
                # Save fixed validation image comparisons
                val_pred = model(fixed_val_damaged)
                # Concatenate along width: [Damaged, Prediction, Clean]
                comparison = torch.cat([fixed_val_damaged, val_pred, fixed_val_clean], dim=3)
                save_image(comparison, val_output_dir / f"val_epoch_{epoch}.png", nrow=1)
                
            avg_val_loss = val_loss / val_batches
            for k in val_metrics:
                val_metrics[k] /= val_batches
                
            print(f"==== Epoch {epoch} Summary ====")
            print(f"Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"PSNR: {val_metrics['PSNR']:.2f} | SSIM: {val_metrics['SSIM']:.4f} | MAE: {val_metrics['MAE']:.4f} | MSE: {val_metrics['MSE']:.4f} | LPIPS: {val_metrics['LPIPS']:.4f}")
            
            # Log to CSV
            csv_writer.writerow([epoch, avg_train_loss, avg_val_loss, val_metrics['PSNR'], val_metrics['SSIM'], val_metrics['MAE'], val_metrics['MSE'], val_metrics['LPIPS'], epoch_time])
            f.flush()

            # Save checkpoint
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'best_val_loss': best_val_loss
            }
            
            checkpoint_path = checkpoint_dir / f"unet_epoch_{epoch}.pth"
            torch.save(checkpoint_state, checkpoint_path)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = checkpoint_dir / "best_model.pth"
                torch.save(checkpoint_state, best_path)
                print(f"Saved NEW Best Checkpoint to {best_path}")
            
            print(f"Saved checkpoint to {checkpoint_path}\n")
            
            early_stopper(avg_val_loss)
            if early_stopper.early_stop:
                print(f"Early stopping triggered at Epoch {epoch}! Validation loss hasn't improved for {args.patience} epochs.")
                break

    print("Training complete!")

if __name__ == "__main__":
    main()
