# Score-Guided Manifold Projection (SGMP)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20A100%20Tested-76b900.svg)](https://www.nvidia.com/en-us/data-center/a100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official PyTorch implementation of **Score-Guided Manifold Projection (SGMP)**, a high-performance generative pre-filtering framework designed to protect visual pattern classifiers against distribution shifts, severe corruptions, and adversarial attacks.

SGMP guides reverse generative trajectories back to the clean data manifold by unifying continuous score-based diffusion priors with discriminative likelihood gradients and geometric manifold contraction.

---

## ⚡ Key Features

- **Generative Purification:** Pre-filters inputs without altering or retraining downstream classifier weights.
- **Directional Alignment & Contraction:** Eliminates destructive orthogonal drift between generative score priors and discriminative gradients.
- **Fast Solver:** Employs an amortized $K=3$ predictor-corrector solver achieving sub-millisecond inference latency ($0.24\text{ ms}$ on NVIDIA A100).
- **Pretrained Checkpoints:** Ready-to-use pretrained weights for the score network, classifiers, and baseline purifiers in `checkpoints/`.
- **Comprehensive Benchmark Suite:** Built-in evaluation covering clean accuracy, 12 standardized corruptions (noise, blur, weather, digital, occlusions), and white-box adversarial PGD attacks.

---

## 📊 Benchmark Summary (NVIDIA A100 GPU)

| Method | Clean Acc (%) | Corrupt Acc (%) | mCE ($\downarrow$) | RRS ($\uparrow$) | PSNR (dB) | SSIM | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla ERM** | 91.63 | 46.24 | 100.00 | 0.505 | — | — | 3.11 |
| **PGD-AT** | 83.95 | 64.33 | 77.24 | 0.766 | — | — | **0.008** |
| **DAE Purifier** | 83.96 | 66.24 | 78.05 | 0.789 | 17.73 | 0.829 | 1.04 |
| **DiffPure** | 69.16 | 49.42 | 117.61 | 0.715 | 15.10 | 0.701 | 4.84 |
| **SGMP (Ours, $K=3$)** | **87.95** | **60.14** | **80.59** | **0.684** | **17.98** | **0.844** | **0.24** |

---

## 🛠️ Repository Structure

```
SGMP/
├── assets/                       # Benchmark & model figures
│   ├── fig1_framework.png
│   ├── fig2_phase_plane.png
│   ├── fig3_robustness_radar.png
│   ├── fig4_qualitative_gallery.png
│   └── fig5_ablations.png
├── checkpoints/                  # Pretrained model weights (~3 MB total)
│   ├── classifier.pt             # Standard classifier
│   ├── classifier_adv.pt         # Adversarially trained classifier
│   ├── dae.pt                    # Denoising autoencoder
│   └── score_net.pt              # Denoising continuous score network
├── data/                         # Corruption suite & data pipeline
│   ├── __init__.py
│   └── corruptions.py            # 12 corruption families + PGD generator
├── models/                       # Core neural architectures
│   ├── __init__.py
│   ├── baselines.py              # DAE and DiffPure baselines
│   ├── classifier.py             # Feature extractor & classifier
│   ├── score_net.py              # Continuous-time U-Net Score Network
│   └── sgmp.py                   # SGMP Purifier core module
├── results/                      # Precomputed benchmark logs & metrics
│   ├── ablation_results.json
│   ├── benchmark_results.json
│   └── benchmark_summary.csv
├── config.py                     # Global configuration & hyperparameters
├── evaluate.py                   # Full evaluation harness (clean, corrupt & adv)
├── run_a100.py                   # High-throughput A100 orchestrator script
├── run_all.py                    # End-to-end execution pipeline
├── train.py                      # Training pipeline for all models
├── visualize.py                  # Generates benchmark plots & qualitative visualizer
├── SGMP_A100_Colab.ipynb         # Interactive Google Colab notebook
├── requirements.txt              # Python package dependencies
├── LICENSE                       # MIT License
└── README.md                     # Documentation
```

---

## 🚀 Installation & Environment Setup

```bash
git clone https://github.com/SuttA-MNNIT/SGMP.git
cd SGMP
pip install -r requirements.txt
```

Verify CUDA GPU availability:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 💻 Quickstart Guide

### 1. Robustness Evaluation
Evaluate all models against clean inputs, 12 corruption categories, and white-box adversarial PGD attacks using the included checkpoints:

```bash
python evaluate.py
```

To run a rapid test on a subset of samples:
```bash
python -c "from evaluate import BenchmarkEvaluator; BenchmarkEvaluator().run_full_benchmark(num_samples=500)"
```

### 2. Training Models from Scratch
Train the classifier, continuous score network, DAE, and adversarial baseline:

```bash
python train.py
```

### 3. High-Throughput NVIDIA A100 Pipeline
Run the hardware-accelerated training, evaluation, and ablation pipeline:

```bash
python run_a100.py
```

### 4. Interactive Google Colab Notebook
Open `SGMP_A100_Colab.ipynb` directly in Google Colab to run SGMP interactively with GPU acceleration.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
