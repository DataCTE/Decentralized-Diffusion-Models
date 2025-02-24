# Decentralized Diffusion Model (DDM) DiT Training Replication

**Author:** Datacte (GitHub: [Datavoid](https://github.com/Datacte))  
**Date:** February 24, 2025  
**License:** MIT License  

## Overview

This repository contains a single-file PyTorch implementation replicating the Decentralized Diffusion Models (DDM) training methodology as described in the paper *"Decentralized Diffusion Models"* by David McAllister, Matthew Tancik, Jiaming Song, and Angjoo Kanazawa (arXiv:2501.05450v2, published January 9, 2025). The implementation focuses on training a Diffusion Transformer (DiT) model in a decentralized manner, leveraging isolated expert models and a lightweight router for inference, as outlined in the DDM framework.

The goal of this project is to provide a simplified, yet functional, replication of DDM training that eliminates the need for high-bandwidth centralized networking, making diffusion model training more accessible on distributed, heterogeneous hardware. This code adapts the DiT architecture from the original paper and integrates key DDM concepts such as data clustering, expert model training, and router-based ensembling.

## Features

- **Decentralized Training:** Trains multiple DiT expert models on disjoint data clusters without cross-communication, following the DFM (Decentralized Flow Matching) objective.
- **Lightweight Router:** Implements a separate DiT-based router for test-time ensembling of expert predictions.
- **Single-File Design:** All functionality (model, dataset, training, and inference) is contained in one Python script for simplicity and portability.
- **Reusability:** Builds on standard PyTorch diffusion training components, making it easy to adapt to existing workflows.
- **Efficiency:** Supports top-1 expert selection at inference time for sparse computation, reducing FLOPs while maintaining quality.

## Prerequisites

- **Python:** 3.8 or higher
- **PyTorch:** 1.13 or higher (with CUDA support for GPU acceleration)
- **Dependencies:**
  - `torchvision`
  - `numpy`
  - `scikit-learn` (for clustering)
  - `tqdm` (for progress bars)
  - `transformers` (for CLIP integration, optional for text conditioning)
  - `diffusers` (for VAE integration)

Install dependencies via pip:
```bash
pip install torch torchvision numpy scikit-learn tqdm transformers diffusers
```
Hardware: Multi-GPU support is recommended but not required. The code can run on a single GPU or CPU with adjustments.

## Usage
File Structure
train_model.py: The single-file implementation containing all code for DDM training and inference.

## Running the Code
Prepare Your Dataset:
  - Place your image dataset in a directory (e.g., /path/to/dataset).
  - Images should be paired with optional text captions (e.g., image1.jpg with image1.txt).

Configure the Script:
  - Edit the DATA_DIR variable in the script to point to your dataset directory.
  - Adjust hyperparameters in the ModelConfig class (e.g., hidden_dim, num_experts, etc.) as needed.

## Run Training
```python
torchrun --nproc-per-node=8 --nnodes=1 train_model.py
```
## The script will:
Cluster the dataset using DINOv2 features and MiniBatchKMeans.
Train expert DiT models on each cluster in isolation.
Train a router model to predict expert weights.
Save checkpoints to checkpoints/ and logs to runs/main/.

## Inference:
After training, use the generate() method to sample images:
```python
trainer.generate("a photo of a mountain", size=(256, 256), num_steps=50)
```
Outputs are generated using the trained ensemble with top-1 expert selection.

## Key Parameters
  - NUM_EXPERTS: Number of expert models (default: 8, based on paper's findings).
  - BATCH_SIZE: Per-expert batch size (default: 1, adjust based on GPU memory).
  - NUM_STEPS: Total training steps (default: 1,000,000).
  - buckets: Image resolution buckets for multi-resolution training (default matches paper).

## Implementation Details
  - Model Architecture: Uses a simplified DiT with adaLN-Zero conditioning, MoE-inspired expert layers, and sinusoidal timestep embeddings.
  - Clustering: Employs DINOv2 for feature extraction and MiniBatchKMeans for efficient data partitioning (1024 fine clusters consolidated to NUM_EXPERTS coarse clusters).
  - Training Objective: Implements Decentralized Flow Matching (DFM) with a cosine noise schedule.
  - Router: A smaller DiT model predicts expert weights via cross-entropy loss over cluster labels.
  - Inference: Supports top-1 expert selection for efficiency, with optional full ensembling.

## Limitations
  - Simplified Scope: This replication omits some advanced features from the original code (e.g., FSDP, distributed sampling) for simplicity.
  - Hardware Constraints: Tested on limited resources; scaling to billions of parameters (e.g., 24B as in the paper) requires significant compute not replicated here.
  - Text Conditioning: Basic CLIP integration is included but not fully optimized for large-scale text-to-image tasks.

## Acknowledgments
  - Inspired by "Decentralized Diffusion Models" by McAllister et al. (2025).
  - Builds on the DiT implementation from Peebles and Xie (2023).
  - Thanks to the PyTorch and Hugging Face communities for open-source tools.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
