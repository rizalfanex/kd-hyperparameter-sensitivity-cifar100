# Hyperparameter Sensitivity of Vanilla Knowledge Distillation for Compact CNNs on CIFAR-100

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--100-lightgrey)
![Task](https://img.shields.io/badge/Task-Knowledge%20Distillation-purple)
![License](https://img.shields.io/badge/License-MIT-green)

This repository provides the code and experimental outputs for the paper:

> **Hyperparameter Sensitivity of Vanilla Knowledge Distillation for Compact CNNs on CIFAR-100**  
> Mochamad Rizal Fauzan, Raden Muhammad Rafi Rachman, Shifa Rangga Saputra, and Daffa Irsyad Nugraha  
> *Journal of Computer Networks, Architecture and High Performance Computing*, Vol. 8, No. 2, April 2026  
> DOI: `10.47709/cnahpc.v8i2.8239`

The study systematically evaluates how two core vanilla knowledge distillation hyperparameters, namely **temperature scaling** (`T`) and **loss balancing** (`alpha`), affect compact convolutional neural networks on CIFAR-100. A **ResNet-50** teacher is used to distill knowledge into two lightweight student models: **MobileNetV2** and **ShuffleNetV2 x1.0**.

---

## Overview

Knowledge distillation is commonly used to improve compact neural networks, but many studies still rely on inherited or default hyperparameter settings. This repository supports a controlled experimental study that re-examines vanilla KD under a unified CIFAR-100 setup.

The main goals are:

- compare standard supervised training and vanilla knowledge distillation for compact CNNs;
- evaluate MobileNetV2 and ShuffleNetV2 x1.0 as lightweight student architectures;
- analyze the sensitivity of temperature scaling (`T`) and loss balancing (`alpha`);
- report not only accuracy, but also parameter count, inference latency, and training time.

---

## Key Results

| Model | Training Type | Params (M) | Top-1 (%) | Top-5 (%) | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | Teacher | 23.71 | 81.24 | 96.05 | 4.72 |
| MobileNetV2 | Standard | 2.35 | 79.18 | 95.77 | 3.98 |
| MobileNetV2 | KD | 2.35 | 80.83 | 96.40 | 3.44 |
| ShuffleNetV2 x1.0 | Standard | 1.36 | 77.00 | 94.81 | 4.23 |
| ShuffleNetV2 x1.0 | KD | 1.36 | 78.36 | 95.45 | 4.29 |

### KD Gains

| Student Model | Top-1 Gain | Top-5 Gain | Latency Difference |
|---|---:|---:|---:|
| MobileNetV2 | +1.65 | +0.63 | -0.54 ms |
| ShuffleNetV2 x1.0 | +1.36 | +0.64 | +0.05 ms |

### Hyperparameter Ablation on MobileNetV2

| Ablation Type | Setting | Best Val Top-1 (%) | Top-1 (%) | Top-5 (%) |
|---|---|---:|---:|---:|
| Temperature | `T = 2, alpha = 0.5` | 80.26 | 79.90 | 95.92 |
| Temperature | `T = 4, alpha = 0.5` | 81.08 | 80.87 | 96.31 |
| Temperature | `T = 6, alpha = 0.5` | 81.60 | 80.82 | 96.27 |
| Loss balancing | `alpha = 0.3, T = 4` | 81.36 | 80.88 | 96.51 |
| Loss balancing | `alpha = 0.5, T = 4` | 81.08 | 80.87 | 96.31 |
| Loss balancing | `alpha = 0.7, T = 4` | 81.02 | 80.78 | 96.19 |

The best ablation configuration was obtained using:

```text
T = 4
alpha = 0.3
Top-1 = 80.88%
Top-5 = 96.51%
```

---

## Figures

### Overall Top-1 Accuracy Comparison

![Top-1 Comparison](outputs_kd_paper_ready/figures/bar_top1_comparison.png)

### KD Gain per Student Model

![KD Gain](outputs_kd_paper_ready/figures/bar_kd_gain.png)

### Accuracy-Latency Trade-off

![Accuracy vs Latency](outputs_kd_paper_ready/figures/scatter_accuracy_vs_latency.png)

### Hyperparameter Sensitivity

![Temperature Ablation](outputs_kd_paper_ready/figures/ablation_top1_vs_temperature.png)

![Alpha Ablation](outputs_kd_paper_ready/figures/ablation_top1_vs_alpha.png)

---

## Repository Structure

```text
.
├── train_kd_cifar100_paper_ready.py
├── requirements.txt
├── CITATION.bib
├── LICENSE
├── README.md
└── outputs_kd_paper_ready/
    ├── results_summary.csv
    ├── results_summary.json
    ├── kd_gain_summary.csv
    ├── kd_gain_summary.json
    ├── ablation_results.csv
    ├── ablation_results.json
    ├── figures/
    │   ├── bar_top1_comparison.png
    │   ├── bar_kd_gain.png
    │   ├── scatter_accuracy_vs_latency.png
    │   ├── scatter_accuracy_vs_params.png
    │   ├── curve_mobilenet_v2.png
    │   ├── curve_shufflenet_v2_x1_0.png
    │   ├── ablation_top1_vs_temperature.png
    │   └── ablation_top1_vs_alpha.png
    └── logs/
        └── *_history.json
```

Large files such as raw CIFAR-100 files and model checkpoints are intentionally excluded from the repository to keep the GitHub repository lightweight and compliant with file-size limits.

---

## Environment

The reported experiments were conducted using:

| Component | Configuration |
|---|---|
| Python | 3.11.14 |
| Framework | PyTorch 2.11.0 + cu130 |
| Torchvision | 0.26.0 + cu130 |
| GPU | NVIDIA GeForce RTX 5060 |
| Precision | Mixed precision training with AMP |
| Dataset | CIFAR-100 |
| Input size | 128 x 128 |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Random seed | 42 |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rizalfanex/kd-hyperparameter-sensitivity-cifar100.git
cd kd-hyperparameter-sensitivity-cifar100
```

Create and activate a Python environment:

```bash
python -m venv .venv
```

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the appropriate PyTorch and torchvision build for your CUDA version. For the reported Windows CUDA 13.0 environment:

```bash
python -m pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 -f https://download.pytorch.org/whl/cu130/torch_stable.html
```

For a different CUDA version or CPU-only environment, install PyTorch using the official command generator from the PyTorch website.

---

## Dataset

The experiments use **CIFAR-100**. The dataset is automatically handled through `torchvision.datasets.CIFAR100` when running the training script.

Dataset configuration used in the paper:

| Item | Configuration |
|---|---|
| Number of classes | 100 |
| Original training images | 50,000 |
| Training split | 45,000 |
| Validation split | 5,000 |
| Test images | 10,000 |
| Split seed | 42 |
| Original image size | 32 x 32 |
| Resized image size | 128 x 128 |
| Normalization | ImageNet mean and standard deviation |

---

## Usage

Run the full training and evaluation pipeline:

```bash
python train_kd_cifar100_paper_ready.py
```

Run the ablation pipeline only:

```bash
python train_kd_cifar100_paper_ready.py --ablation-only
```

The script saves outputs to:

```text
outputs_kd_paper_ready/
```

Expected outputs include:

- result summaries in `.csv` and `.json` format;
- training history logs;
- publication-ready figures;
- model checkpoints generated locally during training.

---

## Knowledge Distillation Objective

The vanilla KD loss combines hard-label supervision and soft-target supervision:

```text
L_KD = alpha * L_CE(z_s, y)
       + (1 - alpha) * T^2 * D_KL(softmax(z_s / T), softmax(z_t / T))
```

where:

- `z_s` denotes the student logits;
- `z_t` denotes the teacher logits;
- `y` denotes the ground-truth label;
- `T` is the temperature scaling parameter;
- `alpha` is the hard-label loss balancing coefficient;
- `L_CE` is the cross-entropy loss;
- `D_KL` is the Kullback-Leibler divergence.

---

## Reproducibility Notes

To reproduce the reported results as closely as possible:

1. use the same random seed (`42`);
2. keep the same CIFAR-100 train-validation split;
3. use the same input resolution (`128 x 128`);
4. use ImageNet-pretrained backbones;
5. select checkpoints based on best validation top-1 accuracy;
6. measure latency after warm-up with GPU synchronization;
7. do not compare latency across different hardware without reporting the device and measurement protocol.

Small deviations may occur because of GPU type, CUDA/cuDNN behavior, PyTorch version, and hardware-level timing variation.

---

## What Is Included and Excluded

Included:

- main training and ablation script;
- result summaries;
- figure files;
- training logs;
- citation file;
- license file.

Excluded:

- CIFAR-100 raw dataset files;
- large `.pt` model checkpoints;
- local cache files;
- Python bytecode files;
- compressed archives.

This keeps the repository clean and avoids GitHub's 100 MB file-size limitation.

---

## Citation

Please cite the paper if this repository is useful for your research:

```bibtex
@article{fauzan2026kdhyperparameter,
  title   = {Hyperparameter Sensitivity of Vanilla Knowledge Distillation for Compact CNNs on CIFAR-100},
  author  = {Fauzan, Mochamad Rizal and Rachman, Raden Muhammad Rafi and Saputra, Shifa Rangga and Nugraha, Daffa Irsyad},
  journal = {Journal of Computer Networks, Architecture and High Performance Computing},
  volume  = {8},
  number  = {2},
  pages   = {235--246},
  year    = {2026},
  doi     = {10.47709/cnahpc.v8i2.8239}
}
```

A BibTeX entry is also provided in [`CITATION.bib`](CITATION.bib).

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## Contact

For questions, issues, or reproducibility discussions, please use the GitHub Issues page of this repository.
