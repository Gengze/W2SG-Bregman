# Weak-to-Strong Generalization via Bregman Bias-Variance Decomposition

This repository contains the code and data for the ICML 2026 paper
**"Weak-to-Strong Generalization via Bregman Bias-Variance Decomposition"**.

The code implements reward-modeling experiments for weak-to-strong
generalization (W2SG), including CE/BCE, reverse CE/BCE, CACE, SL, and AUX
training objectives.

## Installation

We use Python 3.10 and PyTorch 2.1.2. The required packages are listed in
`requirements.txt`.

The GPT-2 checkpoints are expected under `./Models`:

```text
Models/
  gpt2/
  gpt2-medium/
  gpt2-large/
  gpt2-xl/
```

If your checkpoints are stored elsewhere, update `MODEL_CONFIGS` in
`train_w2s.py`.

## Data

The released data is placed under `data/`:

```text
data/
  cai/
  helpful/
```

The main dataset names used by `train_w2s.py` are `cai` and `helpful`.

## Experiments

Run commands from the repository root. For example, the following command trains
a GPT2-Medium student on CAI-Harmless using reverse BCE with weak labels produced
by a GPT2 teacher:

```bash
python train_w2s.py \
  --ds_name=cai \
  --weak_model_size=gpt2 \
  --model_size=gpt2-medium \
  --loss=bce \
  --w2s_loss=reverse_bce \
  --n_docs=4000 \
  --n_w2s_docs=4000 \
  --n_test_docs=4000 \
  --epochs=2 \
  --batch_size=32 \
  --minibatch_size_per_device=1 \
  --sweep_subfolder=cai_gpt2
```

Available W2SG losses include:

```text
bce
reverse_bce
aux
CACE_0.1
Quan_CACE_5
SL_a1_b0.1
kl
reverse_kl
```

Outputs are saved to:

```text
results/<sweep_subfolder>/<config_name>/
```

## Citation

If you find this repository helpful, please cite our paper:

```bibtex
@inproceedings{
anonymous2026weaktostrong,
title={Weak-to-Strong Generalization via Bregman Bias{\textendash}Variance Decomposition},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=nWxUhxv4d8}
}
```

## Acknowledgement

This code is based on and adapted from:

- [keven980716/weak-to-strong-deception](https://github.com/keven980716/weak-to-strong-deception)
- [openai/weak-to-strong](https://github.com/openai/weak-to-strong)
