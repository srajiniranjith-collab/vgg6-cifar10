# VGG6 on CIFAR-10: activations, optimizers and hyper-parameters (CS6886W Assignment 1)

**Author:** Rajalakshmi S (CS25M528)  
**GitHub:** https://github.com/srajiniranjith-collab/vgg6-cifar10  
**W&B project:** https://wandb.ai/cs25m528-iit-madras/vgg6-cifar10  
**Report (PDF):** `report/CS6886W_Assignment1_Report.pdf`

## Headline result (Q4)
Best configuration (highest validation accuracy over all runs): `--arch six_layer --label_smoothing 0.1`,
ReLU, Adam, lr 0.001, batch size 128, 30 epochs, weight decay 5e-5, BatchNorm, Cutout, seed 42.

| best val acc | test top-1 (at best-val epoch) |
|---|---|
| **87.22 %** (epoch 30) | **86.29 %** |

W&B run: https://wandb.ai/cs25m528-iit-madras/vgg6-cifar10/runs/3himf0gw. Reproduce it with `./reproduce_q4.sh` (see "Reproducing Q4" below).

## Repository layout
```
train.py              one training run (CLI args) -> W&B logging + best-val checkpoint
evaluate.py           re-score a saved checkpoint on the CIFAR-10 test set
reproduce_q4.sh       re-runs the Q4 configuration and verifies models/best_model.pt
src/data.py           GPU data pipeline: crop, flip, normalise, Cutout, train/val/test batches
src/model.py          VGG6 with selectable activation, BatchNorm switch and --arch variant
src/optimizers.py     sgd, nesterov, adam, adagrad, rmsprop, nadam
src/engine.py         train_one_epoch / evaluate
src/utils.py          seeding helpers
sweeps/*.yaml         W&B sweep configs (activation, optimizer, hparams, epochs, arch)
tests/smoke_test.py   offline sanity check (no download)
models/best_model.pt  final trained model (the Q4 configuration)
report/               PDF report with plots and analysis
```

## Environment
Developed and run on Google Colab (T4 GPU, Python 3). Install with:
```bash
pip install -r requirements.txt
```
Exact versions used for the reported results (Colab, T4 GPU, CUDA 13.0 builds of torch):
```
matplotlib==3.10.0
numpy==2.1.3
torch==2.11.0+cu130
torchvision==0.26.0+cu130
wandb==0.28.1
```
`requirements.txt` pins the same versions (without the `+cu130` build tag, so the default PyTorch build for your machine is installed).
Print your own with: `pip freeze | grep -iE "^(torch|torchvision|numpy|wandb|matplotlib)=="`

## Seeds / determinism
Default `--seed 42`. It seeds `random`, `numpy`, `torch` (CPU+CUDA), a dedicated GPU generator that drives
shuffling and all augmentation randomness, and the fixed 45k/5k train/val split.
cuDNN deterministic mode is on and global average pooling uses `mean` (deterministic backward).
Identical configurations gave identical per-epoch numbers across separate runs and sweeps (e.g. the 30-epoch
template run was repeated 7 times with the same best val acc, 81.46 %). Different GPU models or library versions can
still change results by a few tenths of a point.

## Architectures (`--arch`)
- `template` (default): conv64-conv64-pool-conv128-conv128-pool-GAP-FC (4 conv + 1 FC), as in the assignment template (262,218 parameters)
- `six_layer`: adds one conv256 block and a third pooling stage (5 conv + 1 FC = 6 weight layers, 559,178 parameters)
- `wide`: template with 2x channels (1,040,522 parameters)

`--label_smoothing` (default 0) applies to the training loss only; validation/test use plain cross-entropy.

## Data (augmentation runs on the GPU, same recipe as the template)
- Train: 45,000 images (random crop 32 pad 4, horizontal flip, normalise with
  mean (0.4914, 0.4822, 0.4465) / std (0.2023, 0.1994, 0.2010), Cutout 1 hole of 16x16)
- Validation: 5,000 held-out train images (normalise only). Used for model selection / sweeps.
- Test: official 10,000 images (normalise only). Scored once, from the best-val checkpoint.

## Running on Colab
```bash
!git clone https://github.com/srajiniranjith-collab/vgg6-cifar10
%cd vgg6-cifar10
!pip install -r requirements.txt
!wandb login
!python tests/smoke_test.py                       # optional sanity check
```
Single run (add `--no_wandb` to skip W&B logging):
```bash
python train.py --activation relu --optimizer adam --lr 0.001 --batch_size 128 --epochs 30
```
Sweeps (each prints a sweep id; `--count` lets you stop and resume across Colab sessions). The sweep configs
do not set `project`, so create them with `--project` so all runs land in one project and one parallel-coordinates plot:
```bash
wandb sweep --project vgg6-cifar10 sweeps/sweep_activation.yaml   # 5 runs
wandb sweep --project vgg6-cifar10 sweeps/sweep_optimizer.yaml    # 18 runs
wandb sweep --project vgg6-cifar10 sweeps/sweep_hparams.yaml      # 12 runs
wandb sweep --project vgg6-cifar10 sweeps/sweep_epochs.yaml       # 3 runs
wandb sweep --project vgg6-cifar10 sweeps/sweep_arch.yaml         # 6 runs (architecture x label smoothing)
wandb agent <entity>/vgg6-cifar10/<sweep_id> --count 6
```
Sweeps used for the reported results:

| Sweep | W&B link |
|---|---|
| activation | https://wandb.ai/cs25m528-iit-madras/vgg6-cifar10/sweeps/qhgtddfc |
| optimizer x lr | https://wandb.ai/cs25m528-iit-madras/vgg6-cifar10/sweeps/eo3eytmb |
| batch size x lr | https://wandb.ai/cs25m528-iit-madras/vgg6-cifar10/sweeps/cs5i18sl |
| epochs (run twice) | https://wandb.ai/cs25m528-iit-madras/vgg6-cifar10/sweeps/inrr1jfj , https://wandb.ai/cs25m528-iit-madras/vgg6-cifar10/sweeps/0az70lwc |
| architecture x label smoothing | https://wandb.ai/cs25m528-iit-madras/vgg6-cifar10/sweeps/xa20d8e2 |

## Reproducing Q4
One command (re-trains from scratch without W&B, then scores the checkpoint and the committed model):
```bash
./reproduce_q4.sh
```
which runs:
```bash
python train.py --arch six_layer --label_smoothing 0.1 --activation relu --optimizer adam \
  --lr 0.001 --batch_size 128 --epochs 30 --weight_decay 0.00005 --batch_norm true --seed 42 --no_wandb
python evaluate.py --ckpt checkpoints/act_relu_bs_128_opt_adam_lr_0.001_wd_5e-05_bn_True_ep_30_arch_six_layer_ls_0.1/best.pt
cp checkpoints/act_relu_bs_128_opt_adam_lr_0.001_wd_5e-05_bn_True_ep_30_arch_six_layer_ls_0.1/best.pt models/best_model.pt
python evaluate.py        # scores models/best_model.pt
```
Expected output (same GPU type): epoch 1 `train_loss 1.6532 train_acc 47.20 | val_loss 1.2334 val_acc 56.06`;
epoch 30 `train_loss 0.9032 train_acc 85.42 | val_loss 0.4088 val_acc 87.22`;
final line `Best val_acc 87.22 (epoch 30) -> test_acc 86.29`; and `evaluate.py` prints `test_acc 86.29`.
(The run name includes the label-smoothing value, so runs with different smoothing never overwrite each other's checkpoint.)

## Results and analysis
All numbers are best validation accuracy in % over epochs (test accuracy at that epoch in brackets or noted). One seed per cell;
differences below about 1 point are within noise (5,000 validation images, noisy epoch-to-epoch validation accuracy). Unless stated,
runs use ReLU, Adam, lr 0.001, batch 128, weight decay 5e-5, BatchNorm, 30 epochs.

### Activation function
| Activation | Best val | Test | Best epoch |
|---|---|---|---|
| SiLU | 82.14 | 80.74 | 28 |
| GELU | 81.82 | 80.61 | 30 |
| ReLU | 81.46 | 80.59 | 26 |
| Tanh | 70.46 | 70.07 | 30 |
| Sigmoid | 46.70 | 46.55 | 12 |

SiLU, GELU and ReLU are tied (0.7 points apart on validation, 0.15 on test). Tanh and Sigmoid saturate, so gradients shrink
and the network underfits (final train loss 0.80 and 0.95 vs. about 0.6 for the others). Sigmoid also shows a large train/validation gap,
which suggests a BatchNorm train/eval mismatch rather than ordinary overfitting; this was not isolated.

### Optimizer x learning rate (batch 128, 30 epochs; momentum 0.9 for SGD, Nesterov, RMSprop)
| Optimizer | lr 0.001 | lr 0.01 | lr 0.1 |
|---|---|---|---|
| SGD (momentum 0.9) | 64.88 (64.75) | 77.62 (76.51) | **82.04 (81.94)** |
| Nesterov-SGD | 65.44 (65.03) | 79.14 (79.04) | **82.26 (81.65)** |
| Adam | **81.46 (80.59)** | 76.54 (74.73) | 47.50 (47.53) |
| Adagrad | 60.04 (59.55) | 75.50 (73.55) | **80.92 (79.40)** |
| RMSprop | **77.66 (76.84)** | 49.28 (49.46) | 13.56 (13.97) |
| Nadam | **80.90 (79.33)** | 77.92 (76.51) | 52.94 (52.84) |

The ranking of optimizers flips with the learning rate: Adam and Nadam are best at 0.001 and fail at 0.1; SGD, Nesterov and Adagrad are slow at 0.001 and
best at 0.1 (the largest value tried). Adagrad's accumulated squared gradients shrink its step, so it needs a large initial lr. RMSprop was the least stable
(noisy at 0.001, stalls at 0.01, no learning at 0.1). At each optimizer's own best lr, five of six land within 1.4 points (80.9 to 82.3 %), so the
differences in final accuracy are not significant; Nesterov had a slightly lower final training loss than plain momentum at every lr. Adam reached 70 % validation accuracy
fastest (epoch 7), while the SGD-type methods at lr 0.1 ended with the lowest training loss.

### Batch size x learning rate (Adam, 30 epochs, best val)
| Batch size | lr 3e-4 | lr 1e-3 | lr 3e-3 |
|---|---|---|---|
| 64 | 79.28 | 83.20 | 81.96 |
| 128 | 76.08 | 81.46 | 81.46 |
| 256 | 72.84 | 78.58 | 81.96 |
| 512 | 70.32 | 75.20 | 77.94 |

At a fixed number of epochs smaller batches win, because they make more updates (21,120 vs. 2,640 for batch 64 vs. 512). The best learning rate grows with the batch size
(1e-3 for batch 64, 3e-3 for batch 256 and 512), in line with the usual lr-scaling heuristic.

### Number of epochs (batch 128, lr 0.001)
| Epochs | Best val | Best epoch | Test |
|---|---|---|---|
| 10 | 73.84 | 10 | 74.46 |
| 20 | 77.82 | 18 | 77.86 |
| 30 | 81.46 | 26 | 80.59 |
| 50 | 84.66 | 50 | 83.48 |

More epochs helped at every step with diminishing gain per epoch; the training loss was still falling at 50 epochs and no overfitting was visible. Because the learning rate is constant,
the first k epochs of a longer run are identical to a k-epoch run (confirmed by the logs).

### Architecture x label smoothing (best val / test)
| Architecture | Parameters | Label smoothing 0 | Label smoothing 0.1 |
|---|---|---|---|
| template | 262,218 | 81.46 / 80.59 | 80.58 / 79.10 |
| wide | 1,040,522 | 83.68 / 83.44 | 83.38 / 82.04 |
| six_layer | 559,178 | 86.44 / 85.00 | **87.22 / 86.29** |

Depth helped most (+5.0 points for one extra conv block), more than width (+2.2 points with 4x the parameters). Label smoothing helped only the six_layer network
(+0.78 val, +1.29 test) and slightly hurt the other two; with one seed per cell this pattern may be partly noise.

### Caveats
Single seed per configuration; validation accuracy fluctuates by several points between epochs (constant learning rate), so "best over epochs" is slightly optimistic
(test accuracy at the selected epoch was on average 0.8 points below validation accuracy in the batch-size grid). Longer training or a learning-rate decay would likely improve all results.
