"""Offline smoke test (no dataset download): every activation x optimizer builds,
takes a training step on random data, and the loss is finite."""
import itertools
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import augment_batch  # noqa: E402
from src.engine import evaluate, train_one_epoch  # noqa: E402
from src.model import ACTIVATIONS, ARCHS, build_vgg6  # noqa: E402
from src.optimizers import OPTIMIZERS, build_optimizer  # noqa: E402

x = torch.randn(64, 3, 32, 32)
y = torch.randint(0, 10, (64,))
loader = DataLoader(TensorDataset(x, y), batch_size=32)
crit = nn.CrossEntropyLoss()

img = torch.rand(8, 3, 32, 32)
a1 = augment_batch(img, torch.Generator().manual_seed(0))
a2 = augment_batch(img, torch.Generator().manual_seed(0))
assert a1.shape == img.shape and torch.equal(a1, a2), "augmentation must be seeded/deterministic"
assert (a1 == 0).any(), "cutout should zero some pixels"
assert not torch.equal(a1, augment_batch(img, torch.Generator().manual_seed(1)))
for act, opt in itertools.product(ACTIVATIONS, OPTIMIZERS):
    model = build_vgg6(act, True)
    assert model(x[:2]).shape == (2, 10)
    o = build_optimizer(opt, model.parameters(), 1e-3, 5e-5)
    loss, _ = train_one_epoch(model, loader, crit, o, "cpu")
    assert math.isfinite(loss), (act, opt)
    evaluate(model, loader, crit, "cpu")
    assert not model.training  # evaluate() leaves eval mode; the next epoch must re-enable train()
    train_one_epoch(model, loader, crit, o, "cpu")
    assert model.training, "train_one_epoch must switch back to train mode"
for arch in ARCHS:
    assert build_vgg6("relu", True, arch=arch)(x[:2]).shape == (2, 10), arch
print("smoke test passed:", len(ACTIVATIONS) * len(OPTIMIZERS), "combinations")
