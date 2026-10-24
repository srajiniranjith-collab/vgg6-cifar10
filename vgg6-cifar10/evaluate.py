"""Evaluate a saved checkpoint on the CIFAR-10 test set.

python evaluate.py --ckpt models/best_model.pt
"""
import argparse

import torch
import torch.nn as nn

from src.data import get_loaders
from src.engine import evaluate
from src.model import build_vgg6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/best_model.pt")
    p.add_argument("--data_dir", default="./data")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.ckpt, map_location=device)
    cfg = ckpt["config"]
    model = build_vgg6(cfg["activation"], cfg["batch_norm"], arch=cfg.get("arch", "template")).to(device)
    model.load_state_dict(ckpt["model"])

    _, val_loader, test_loader = get_loaders(512, cfg["val_size"], cfg["seed"],
                                             a.data_dir, cfg["cutout"], device)
    crit = nn.CrossEntropyLoss()
    _, val_acc = evaluate(model, val_loader, crit, device)
    test_loss, test_acc = evaluate(model, test_loader, crit, device)
    print(f"val_acc {val_acc:.2f} | test_loss {test_loss:.4f} | test_acc {test_acc:.2f}")


if __name__ == "__main__":
    main()
