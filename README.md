# GC-FCP Experiments

This repository contains the code needed to reproduce the experiments for **Efficient Federated Conformal Prediction with Group-Conditional Guarantee**. 

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the requirements.

```powershell
pip install -r requirements.txt
```

The CIFAR-10 model is loaded through `torch.hub` from `chenyaofo/pytorch-cifar-models` on first use. CIFAR-10 and PathMNIST are
downloaded automatically by `torchvision` / `medmnist`. ImageNet-1K is not downloaded automatically; prepare the validation set locally or provide a cached probability file.

## Quick Smoke Test

Run a small synthetic check before launching the expensive jobs:

```bash
python realdata_rebuttal_v4.py smoke --out_dir results/smoke
```

## Main Reproduction Commands

Synthetic regression figure:

```bash
python synthetic_main_patched.py
```

CIFAR-10 and PathMNIST coverage/set-size curves:

```bash
python realdata_alpha_exp.py --exp alpha --dataset cifar10 --delta 0.1 --times 50 --overlap
python realdata_alpha_exp.py --exp alpha --dataset pathmnist --delta 0.1 --times 50 --overlap --model_path checkpoints/path_cnn.pt
```

CIFAR-10 table:

```bash
python realdata_rebuttal_v4.py run \
  --dataset cifar10 \
  --clients 5 \
  --scores thr \
  --x_feature pred_label \
  --group_mode pred_window \
  --client_split label \
  --n_cal 5000 \
  --n_test 5000 \
  --times 50 \
  --deltas 25,250,2500 \
  --methods cp,fedcp,condcp,naive_gcfcp,gcfcp \
  --lp_max_iter 50 \
  --n_jobs 8 \
  --out_dir results/paper_tables
```

Then run `make_paper_tables_with_stderr.py` on the generated `results/paper_tables/rebuttal_cifar10_*/raw_results.csv`.

PathMNIST table:

```bash
python realdata_rebuttal_v4.py run \
  --dataset pathmnist \
  --clients 5 \
  --scores thr \
  --x_feature pred_label \
  --group_mode pred_window \
  --client_split label \
  --n_cal 8592 \
  --n_test 8592 \
  --times 50 \
  --deltas 25,250,2500 \
  --methods cp,fedcp,gcfcp \
  --pathmnist_model_path checkpoints/path_cnn.pt \
  --lp_max_iter 50 \
  --n_jobs 8 \
  --out_dir results/paper_tables
```

If `checkpoints/path_cnn.pt` is unavailable, train it first:

```bash
python medmnist_train.py --epochs 40 --out checkpoints/path_cnn.pt
```

ImageNet probability cache:

```bash
python realdata_rebuttal.py cache_imagenet \
  --data_root data/imagenet \
  --imagenet_val_dir data/imagenet/val_by_synset \
  --imagenet_class_index_json data/imagenet/imagenet_class_index.json \
  --imagenet_model resnet50 \
  --batch_size 256 \
  --num_workers 8 \
  --pin_memory \
  --cache_path data/imagenet_resnet50_val_probs.npz
```

ImageNet main RAPS experiment and all-group plots:

```bash
python realdata_rebuttal.py run \
  --dataset imagenet \
  --imagenet_cache data/imagenet_resnet50_val_probs.npz \
  --clients 50 \
  --client_split dirichlet \
  --dirichlet_beta 0.3 \
  --min_client_count 200 \
  --scores raps \
  --group_rule min \
  --n_cal 40000 \
  --n_test 10000 \
  --times 100 \
  --group_mode imagenet_semantic_ambiguity \
  --ambiguity_score margin \
  --ambiguity_bins 5 \
  --ambiguity_binning quantile \
  --deltas 50,250,500 \
  --n_jobs 8 \
  --out_dir results/rebuttal

python plot_group_boxplots.py \
  --csv results/rebuttal/rebuttal_imagenet_YYYYMMDD_HH/raw_results.csv \
  --dataset imagenet \
  --score raps \
  --big-all-groups \
  --big-ncols 5 \
  --suffix .pdf \
  --big-coverage-out figures/figure_r1.pdf \
  --big-size-out figures/figure_r2.pdf
```

ImageNet ablation and compression figure:

```bash
python realdata_ablation.py run \
  --ablation all \
  --dataset imagenet \
  --imagenet_cache data/imagenet_resnet50_val_probs.npz \
  --clients 50 \
  --client_split dirichlet \
  --dirichlet_beta 0.3 \
  --min_client_count 200 \
  --scores thr,raps,aps \
  --n_cal 40000 \
  --n_test 10000 \
  --times 10 \
  --ambiguity_score margin \
  --ambiguity_bins 5 \
  --ambiguity_binning quantile \
  --group_designs semantic,ambiguity,semantic_ambiguity \
  --group_delta 250 \
  --delta_values 25,50,250,500,1000,2500 \
  --delta_group_design semantic_ambiguity \
  --k_values 1,10,20,30,40,50 \
  --k_delta 250 \
  --k_group_design semantic_ambiguity \
  --n_jobs 8 \
  --out_dir results/rebuttal_ablation

python realdata_ablation.py report \
  --results_csv results/rebuttal_ablation/gcfcp_ablation_YYYYMMDD_HHMMSS/ablation_raw_results.csv \
  --out_dir results/rebuttal_ablation/gcfcp_ablation_YYYYMMDD_HHMMSS
```

## Citation

```
@article{wen2026efficient,
  title={Efficient Federated Conformal Prediction with Group-Conditional Guarantees},
  author={Wen, Haifeng and Simeone, Osvaldo and Xing, Hong},
  journal={arXiv preprint arXiv:2603.14198},
  year={2026}
}
```

