"""Small helpers: seeding, argparse bool type, parameter counting."""
import argparse
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU+CUDA).

    deterministic=True also forces cuDNN to pick deterministic kernels
    (slightly slower, but makes re-runs of the "best config" match).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn so numpy/random inside workers are seeded."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def str2bool(v) -> bool:
    """Lets W&B sweeps pass --batch_norm=True / false / 1 / 0."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "1"):
        return True
    if v.lower() in ("false", "f", "no", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
