"""
Training script for CenterNet.

Usage:
    python train.py --img_dir /path/to/train_images \
                     --ann_file /path/to/train_annotations.json \
                     --epochs 50 --batch_size 16 --lr 1.25e-4
"""

import argparse
import torch
from torch.utils.data import DataLoader

from model import CenterNet
from dataset import CenterNetDataset, collate_fn
from utils import CenterNetLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--ann_file", type=str, required=True)
    p.add_argument("--input_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1.25e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CenterNetDataset(args.img_dir, args.ann_file, input_size=args.input_size, augment=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
    )
    print(f"Dataset: {len(dataset)} images, {dataset.num_classes} classes")

    model = CenterNet(num_classes=dataset.num_classes, backbone_name=args.backbone, pretrained=True)
    model.to(device)

    criterion = CenterNetLoss(lambda_size=0.1, lambda_offset=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.6), int(args.epochs * 0.9)], gamma=0.1
    )

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    import os
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {"total_loss": 0.0, "heatmap_loss": 0.0, "size_loss": 0.0, "offset_loss": 0.0}

        for step, batch in enumerate(loader):
            images = batch["image"].to(device)
            targets = {
                "heatmap": batch["heatmap"].to(device),
                "offset": batch["offset"].to(device),
                "size": batch["size"].to(device),
                "mask": batch["mask"].to(device),
            }

            preds = model(images)
            losses = criterion(preds, targets)

            optimizer.zero_grad()
            losses["total_loss"].backward()
            # Gradient clipping stabilizes early training (heatmap focal
            # loss gradients can spike when predictions are very wrong).
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            optimizer.step()

            for k in running:
                running[k] += losses[k].item()

            if step % 20 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(loader)}] "
                    f"total={losses['total_loss'].item():.4f} "
                    f"hm={losses['heatmap_loss'].item():.4f} "
                    f"size={losses['size_loss'].item():.4f} "
                    f"off={losses['offset_loss'].item():.4f}"
                )

        scheduler.step()
        n = len(loader)
        print(f"== Epoch {epoch} avg: " + ", ".join(f"{k}={v/n:.4f}" for k, v in running.items()))

        ckpt_path = os.path.join(args.checkpoint_dir, f"centernet_epoch{epoch}.pth")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "num_classes": dataset.num_classes,
            "backbone": args.backbone,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
