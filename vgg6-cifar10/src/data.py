"""CIFAR-10 pipeline that runs entirely on the GPU.

The whole dataset (150 MB as uint8) is kept on the device and the augmentation
is applied per batch with tensor ops, so there is no CPU/PIL bottleneck.
Recipe is identical to the assignment template:
    RandomCrop(32, padding=4) -> RandomHorizontalFlip(0.5) -> normalise
    -> Cutout(1 hole, 16x16)           (train only)
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

# Per-channel statistics of the CIFAR-10 training set.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def _stats(device):
    mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
    return mean, std


def normalize(x: torch.Tensor) -> torch.Tensor:
    """x: float in [0,1], (B,3,H,W)."""
    mean, std = _stats(x.device)
    return (x - mean) / std


def augment_batch(x: torch.Tensor, gen: torch.Generator, cutout: bool = True,
                  pad: int = 4, cut_len: int = 16) -> torch.Tensor:
    """Augment + normalise a batch. x: float in [0,1], (B,C,H,W); `gen` lives on x.device.

    Order matches the template: pad with black (0) -> random crop -> flip ->
    normalise -> Cutout (so a cut-out patch equals the dataset mean colour).
    """
    B, C, H, W = x.shape
    dev = x.device
    xp = F.pad(x, (pad, pad, pad, pad))                       # zero padding
    i = torch.randint(0, 2 * pad + 1, (B,), generator=gen, device=dev)
    j = torch.randint(0, 2 * pad + 1, (B,), generator=gen, device=dev)
    rows = i[:, None] + torch.arange(H, device=dev)           # (B,H)
    cols = j[:, None] + torch.arange(W, device=dev)           # (B,W)
    b_idx = torch.arange(B, device=dev)[:, None, None, None]
    c_idx = torch.arange(C, device=dev)[None, :, None, None]
    x = xp[b_idx, c_idx, rows[:, None, :, None], cols[:, None, None, :]]  # per-sample crop

    flip = torch.rand(B, generator=gen, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    x = normalize(x)

    if cutout:
        cy = torch.randint(0, H, (B,), generator=gen, device=dev)[:, None, None]
        cx = torch.randint(0, W, (B,), generator=gen, device=dev)[:, None, None]
        ys = torch.arange(H, device=dev)[None, :, None]
        xs = torch.arange(W, device=dev)[None, None, :]
        half = cut_len // 2
        hole = (ys >= cy - half) & (ys < cy + half) & (xs >= cx - half) & (xs < cx + half)
        x = x * (~hole)[:, None].to(x.dtype)
    return x


class GPUBatches:
    """Iterable of (images, labels) batches; data already lives on the device."""

    def __init__(self, x_u8, y, batch_size, train, gen=None, cutout=True):
        self.x, self.y = x_u8, y
        self.batch_size, self.train, self.gen, self.cutout = batch_size, train, gen, cutout

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def __iter__(self):
        n, dev = len(self.y), self.x.device
        order = (torch.randperm(n, generator=self.gen, device=dev) if self.train
                 else torch.arange(n, device=dev))
        for s in range(0, n, self.batch_size):
            idx = order[s:s + self.batch_size]
            xb = self.x[idx].float().div_(255.0)
            xb = augment_batch(xb, self.gen, self.cutout) if self.train else normalize(xb)
            yield xb, self.y[idx]


def _load(data_dir, train):
    ds = datasets.CIFAR10(data_dir, train=train, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()   # uint8, (N,3,32,32)
    return x, torch.tensor(ds.targets, dtype=torch.long)


def get_loaders(batch_size: int, val_size: int = 5000, seed: int = 42,
                data_dir: str = "./data", cutout: bool = True, device="cuda"):
    """Build train / val / test iterables.

    The 50k training images are split ONCE (fixed by `seed`) into
    (50k - val_size) train and `val_size` validation images; validation uses no
    augmentation. The official 10k test set is only used for the final report.
    """
    device = torch.device(device)
    x, y = _load(data_dir, train=True)
    xt, yt = _load(data_dir, train=False)

    perm = torch.from_numpy(np.random.RandomState(seed).permutation(len(y)))
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)

    gen = torch.Generator(device=device).manual_seed(seed)   # shuffle + augmentation RNG
    train = GPUBatches(x[train_idx.to(device)], y[train_idx.to(device)], batch_size, True, gen, cutout)
    val = GPUBatches(x[val_idx.to(device)], y[val_idx.to(device)], 512, False)
    test = GPUBatches(xt, yt, 512, False)
    return train, val, test
