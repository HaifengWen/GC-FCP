#!/usr/bin/env bash
set -euo pipefail

# This file is a command index. Most full paper runs are expensive, so commands
# are grouped and commented. Uncomment the block you want to reproduce, or use
# the README for details and expected outputs.

mkdir -p figures results

# Quick synthetic smoke test.
python realdata_rebuttal_v4.py smoke --out_dir results/smoke

# Synthetic regression figure used in the main paper.
# python synthetic_main_patched.py

# CIFAR-10 and PathMNIST alpha curves.
# python realdata_alpha_exp.py --exp alpha --dataset cifar10 --delta 0.1 --times 50 --overlap
# python realdata_alpha_exp.py --exp alpha --dataset pathmnist --delta 0.1 --times 50 --overlap --model_path checkpoints/path_cnn.pt

# CIFAR-10 table.
# python realdata_rebuttal_v4.py run \
#   --dataset cifar10 \
#   --clients 5 \
#   --scores thr \
#   --x_feature pred_label \
#   --group_mode pred_window \
#   --client_split label \
#   --n_cal 5000 \
#   --n_test 5000 \
#   --times 50 \
#   --deltas 25,250,2500 \
#   --methods cp,fedcp,condcp,naive_gcfcp,gcfcp \
#   --lp_max_iter 50 \
#   --n_jobs 8 \
#   --out_dir results/paper_tables

# PathMNIST table.
# python realdata_rebuttal_v4.py run \
#   --dataset pathmnist \
#   --clients 5 \
#   --scores thr \
#   --x_feature pred_label \
#   --group_mode pred_window \
#   --client_split label \
#   --n_cal 8592 \
#   --n_test 8592 \
#   --times 50 \
#   --deltas 25,250,2500 \
#   --methods cp,fedcp,gcfcp \
#   --pathmnist_model_path checkpoints/path_cnn.pt \
#   --lp_max_iter 50 \
#   --n_jobs 8 \
#   --out_dir results/paper_tables

# ImageNet cache, main run, and ablations are documented in README.md because
# they require a local ImageNet validation set or precomputed probability cache.
