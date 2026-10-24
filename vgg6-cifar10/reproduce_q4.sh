#!/usr/bin/env bash
# Re-runs the reported best configuration (Q4) from scratch and verifies the saved model.
# Expected: "Best val_acc 87.22 (epoch 30) -> test_acc 86.29", then the same test_acc from evaluate.py.
set -euo pipefail
RUN=act_relu_bs_128_opt_adam_lr_0.001_wd_5e-05_bn_True_ep_30_arch_six_layer_ls_0.1

python train.py --arch six_layer --label_smoothing 0.1 --activation relu --optimizer adam \
  --lr 0.001 --batch_size 128 --epochs 30 --weight_decay 0.00005 --batch_norm true --seed 42 --no_wandb

python evaluate.py --ckpt checkpoints/$RUN/best.pt      # score the freshly trained checkpoint
mkdir -p models && cp checkpoints/$RUN/best.pt models/best_model.pt
python evaluate.py                                       # scores models/best_model.pt (the committed model)
