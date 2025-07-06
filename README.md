# Neural BRDF Quality Metric (BRDF-NQM)

A perceptually informed neural quality metric for evaluating BRDF models directly in BRDF space—no rendering required.

## 🧠 Overview

This project implements a neural quality metric (BRDF-NQM) that predicts perceptual differences between reference and approximated BRDFs. Trained on measured datasets with perceptual pseudo-labels, it enables efficient, rendering-free evaluation of BRDF model quality.

## 🔍 Features

- Operates directly in BRDF space
- Compact multi-layer perceptron (MLP)
- Predicts perceptual JOD (Just-Objectionable-Difference) scores
- Outperforms traditional BRDF-space metrics in human correlation
- Fast inference (~0.04 ms per sample)

## 🚀 Getting Started

### Requirements

Install the required packages:

```bash
pip install torch numpy scipy matplotlib
```

### Running the Notebook 

Launch the notebook in Jupyter:

```bash
jupyter notebook nbrdfq.ipynb
```
The notebook demonstrates how to:

- Load BRDF samples
- Apply preprocessing
- Run the trained MLP model
- Predict perceptual JOD scores

## Input Format
Each BRDF is a 500 × 3 array of RGB reflectance samples in Rusinkiewicz coordinates. Preprocessing includes:

- Cube root transformation
- Log compression: log(rho^(1/3) + 1)
- Whitening (per-channel mean/std normalization)

## Output
The model outputs a scalar JOD score (between 0 and 10) representing the perceptual difference between the reference and distorted BRDFs.

### 🗃️ Dataset

The dataset used for training and evaluation can be found here:  
📦 [Google Drive – BRDF Dataset](https://drive.google.com/drive/folders/1ujdKAe-s0DlhciXcBv0iU2EZZrnet8Cx?usp=sharing)

The repository includes supporting data in `./data/`, containing:
- Precomputed JOD labels used for training, validation, and testing
- Metadata used during augmentation and preprocessing


## 📌 Citation

If you use this work, please cite:

```bibtex
@inproceedings{Kavoosighafi2025NBRDFQ,
  title     = {A Neural Quality Metric for BRDF Models},
  author    = {Behnaz Kavoosighafi and Rafał K. Mantiuk and Saghi Hajisharif and Ehsan Miandji and Jonas Unger},
  booktitle = {Proceedings of the London Imaging Meeting (LIM)},
  year      = {2025},
  note      = {To appear},
}
