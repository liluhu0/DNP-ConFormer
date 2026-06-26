# DNP-ConFormer: Diverse Normal Prototypes-Guided Contrastive Reconstruction for Medical Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026-blue.svg)](https://conferences.miccai.org/2026/)

## News

- **[2026/06]** Our paper has been **provisionally accepted to MICCAI 2026**.

**DNP-ConFormer** is a medical anomaly detection framework based on diverse normal prototypes-guided contrastive reconstruction. It learns compact normal representations with prototype-guided reconstruction and contrastive supervision for image-level anomaly detection in medical datasets.

## Highlights

- **Diverse Normal Prototypes**: Represents normal patterns with multiple learnable prototype tokens to improve coverage of normal appearance variation.
- **Prototype-guided Reconstruction**: Reconstructs encoder features through INP-guided decoder blocks for anomaly-sensitive feature comparison.
- **Contrastive Optimization**: Uses global cosine-based contrastive objectives and optional EMA encoder guidance to stabilize training.
- **Medical Dataset Support**: Provides dataset preparation scripts for APTOS2019, Br35H, and ISIC2018, with a generic normal/abnormal folder loader.
- **Transformer Backbones**: Supports DINOv2, DINOv1, and BEiT-style vision transformer encoders.

## Method

DNP-ConFormer consists of three core components:

### 1. Vision Transformer Encoder

The framework extracts hierarchical visual tokens from a frozen or partially trainable transformer encoder. The default encoder is `dinov2reg_vit_small_14`, with support for base and large variants.

### 2. Diverse Normal Prototype Modeling

Learnable prototype tokens aggregate normal feature patterns through attention blocks. These prototypes guide the decoder and provide an auxiliary prototype compactness objective.

### 3. Contrastive Feature Reconstruction

The decoder reconstructs encoder features under prototype guidance. Training minimizes cosine-based reconstruction discrepancies, with optional EMA encoder contrast for more stable feature targets.

## Project Structure

```text
DNP-ConFormer/
|-- main.py                  # Training and testing entry point
|-- dataset.py               # Dataset loader and image transforms
|-- aug_funcs.py             # Image augmentation utilities
|-- utils.py                 # Evaluation, logging, schedulers, and metrics
|-- inference_onnx.py        # ONNX inference utility
|-- requirements.txt         # Python dependencies
|-- LICENSE                  # MIT License
|-- backbones/               # Backbone resources
|-- beit/                    # BEiT vision transformer implementation
|-- dinov1/                  # DINOv1 vision transformer implementation
|-- dinov2/                  # DINOv2 backbone and related modules
|-- flops_profiler/          # FLOPs and parameter profiling utilities
|-- models/                  # DNP-ConFormer model and transformer blocks
|   |-- DNP_ConFormer.py     # INP_Former architecture
|   |-- vit_encoder.py       # Encoder loading utilities
|   `-- vision_transformer.py # Decoder and aggregation blocks
|-- optimizers/              # Optimizer implementations
|-- prepare_dataset/         # Dataset preparation scripts
|-- dataset/                 # Place prepared datasets here
`-- assets/                  # Optional figures, logs, and release assets
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU is recommended for training
- PyTorch with a CUDA version matching your local driver

### Install Dependencies

```bash
git clone https://github.com/liluhu0/DNP-ConFormer.git
cd DNP-ConFormer
pip install -r requirements.txt
```

If your CUDA version is different from CUDA 11.8, install the matching PyTorch build from the official PyTorch index before installing the remaining dependencies.

## Usage

### 1. Prepare Data

Prepare the dataset with the following structure:

```text
dataset/APTOS2019/
|-- train/
|   `-- NORMAL/
|       `-- image files
`-- test/
    |-- NORMAL/
    |   `-- image files
    `-- ABNORMAL/
        `-- image files
```

Dataset-specific preprocessing scripts are available in `prepare_dataset/`:

```bash
python prepare_dataset/prepare_aptos.py
python prepare_dataset/prepare_br35h.py
python prepare_dataset/prepare_isic2018.py
```

### 2. Configure Parameters

Core parameters can be configured from the command line:

```bash
python main.py \
  --dataset APTOS2019 \
  --data_path ./dataset/APTOS2019 \
  --encoder dinov2reg_vit_small_14 \
  --input_size 252 \
  --INP_num 6 \
  --total_iters 5000 \
  --batch_size 32 \
  --phase train
```

Important options:

- `--dataset`: Dataset name, such as `APTOS2019`, `Br35H`, `ISIC2018`, or `OCT2017`.
- `--data_path`: Path to the prepared dataset root.
- `--encoder`: Backbone encoder name, such as `dinov2reg_vit_small_14`, `dinov2reg_vit_base_14`, or `dinov2reg_vit_large_14`.
- `--phase`: Use `train` for training and `test` for evaluation.
- `--save_dir`: Root directory for checkpoints, logs, and evaluation outputs.

### 3. Train

```bash
python main.py --phase train --data_path ./dataset/APTOS2019
```

### 4. Test

```bash
python main.py --phase test --data_path ./dataset/APTOS2019
```

### 5. Output

Results are saved under `saved_results/`, including:

- `model.pth`: Final model checkpoint.
- `model_best.pth`: Best model checkpoint during validation.
- `ema_encoder.pth`: EMA encoder checkpoint when EMA contrast is enabled.
- `image_test/`: Test-time visual outputs when `save_root` is enabled.
- Log files containing AUROC, AP, F1, accuracy, recall, specificity, and threshold statistics.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{dnpconformer2026,
  title={DNP-ConFormer: Diverse Normal Prototypes-Guided Contrastive Reconstruction for Medical Anomaly Detection},
  author={DNP-ConFormer Authors},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

This repository follows the open-source release style of [SeedPro](https://github.com/Haitao-Lee/SeedPro). We also acknowledge the DINO, DINOv2, BEiT, PyTorch, and timm communities for open-source components used by this project.
