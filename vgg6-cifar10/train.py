"""Train one VGG6 / CIFAR-10 configuration, log to W&B, save best-val checkpoint.

Single run:  python train.py --activation gelu --optimizer nadam --lr 0.001
Sweep:       wandb sweep sweeps/sweep_activation.yaml  ->  wandb agent <id>
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.data import get_loaders
from src.engine import evaluate, train_one_epoch
from src.model import ACTIVATIONS, ARCHS, build_vgg6
from src.optimizers import OPTIMIZERS, build_optimizer
from src.utils import count_params, set_seed, str2bool


def parse_args():
    p = argparse.ArgumentParser(description="VGG6 on CIFAR-10")
    p.add_argument("--activation", default="relu", choices=list(ACTIVATIONS))
    p.add_argument("--arch", default="template", choices=list(ARCHS))
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--optimizer", default="adam", choices=OPTIMIZERS)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_norm", type=str2bool, default=True)
    p.add_argument("--cutout", type=str2bool, default=True)
    p.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_size", type=int, default=5000)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--save_dir", default="./checkpoints")
    p.add_argument("--wandb_project", default="vgg6-cifar10")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def run_name(a) -> str:
    return (f"act_{a.activation}_bs_{a.batch_size}_opt_{a.optimizer}_lr_{a.lr}"
            f"_wd_{a.weight_decay}_bn_{a.batch_norm}_ep_{a.epochs}_arch_{a.arch}_ls_{a.label_smoothing}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = run_name(args)

    wandb = None
    if not args.no_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=args.wandb_project, name=name, config=vars(args))
        # val_acc column in parallel-coords / scatter = best value over the run
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
        wandb.define_metric("val_acc", summary="max")
        wandb.define_metric("val_loss", summary="min")

    train_loader, val_loader, test_loader = get_loaders(
        args.batch_size, args.val_size, args.seed, args.data_dir, args.cutout, device)

    model = build_vgg6(args.activation, args.batch_norm, arch=args.arch).to(device)
    # label smoothing only affects the training loss; val/test loss stay plain CE
    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr,
                                args.weight_decay, args.momentum)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.scheduler == "cosine" else None)
    print(f"{name} | params={count_params(model):,} | device={device}")

    ckpt_dir = os.path.join(args.save_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    best_val, best_epoch = -1.0, -1

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, train_criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()

        print(f"[{epoch+1:3d}/{args.epochs}] train_loss {tr_loss:.4f} train_acc {tr_acc:.2f} "
              f"| val_loss {va_loss:.4f} val_acc {va_acc:.2f}")
        if wandb:
            wandb.log({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                       "val_loss": va_loss, "val_acc": va_acc,
                       "lr": optimizer.param_groups[0]["lr"]})

        if va_acc > best_val:
            best_val, best_epoch = va_acc, epoch
            torch.save({"model": model.state_dict(), "config": vars(args),
                        "val_acc": va_acc, "epoch": epoch}, ckpt_path)
        if not math.isfinite(tr_loss):
            print("Loss diverged (NaN/inf) - stopping this run early.")
            break

    # Final report: reload best-validation weights and score on the test set once.
    test_acc = None
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
        _, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Best val_acc {best_val:.2f} (epoch {best_epoch+1}) -> test_acc {test_acc:.2f}")

    with open(os.path.join(ckpt_dir, "result.json"), "w") as f:
        json.dump({"config": vars(args), "best_val_acc": best_val,
                   "best_epoch": best_epoch + 1, "test_acc": test_acc}, f, indent=2)
    if wandb:
        wandb.summary["best_val_acc"] = best_val
        wandb.summary["best_epoch"] = best_epoch + 1
        if test_acc is not None:
            wandb.summary["test_acc"] = test_acc
        wandb.finish()


if __name__ == "__main__":
    main()
