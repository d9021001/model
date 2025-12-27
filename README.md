# TPX Pipeline for Smartphone Addiction Risk Detection

This repository contains the code and scripts for the TPX pipeline, a machine learning framework for identifying college students at high risk of problematic smartphone use (PSU) using raw app-usage telemetry.

## Overview

The TPX pipeline integrates:
- **Triplet-based Convolutional Neural Network (Triplet-CNN):** Learns discriminative temporal embeddings from raw per-app session sequences.
- **Principal Component Analysis (PCA):** Compresses the learned embeddings.
- **Extreme Gradient Boosting (XGBoost):** Classifies weekly risk scores.

The approach is designed to overcome the limitations of self-report questionnaires by leveraging high-dimensional, real-time behavioral data.

## Dataset

- **Subjects:** 114 undergraduates (38 addicted, 76 non-addicted; labeled by the Smartphone Addiction Scale)
- **Observation period:** 24 weeks
- **Data:** Passive logging via Android’s native “App Usage” service

## Main Files

- `tpx24w_apptime.py` — Main evaluation script (5-fold cross-validation, TPX pipeline); Example of app usage time feature extraction
- `ts0.xlsx`, `ts1.xlsx` — Test set (class 0: negative, class 1: positive)
- `tr0.xlsx`, `tr1.xlsx` — Training set (class 0: negative, class 1: positive)
- `triplet.h5` — Pretrained Triplet-CNN model
- `xgb_model.model` — Pretrained XGBoost model (not included due to file size limits)

## Usage

1. **Install dependencies:**
   - Python 3.95+
   - TensorFlow 2.x
   - scikit-learn
   - xgboost
   - pandas, numpy, matplotlib

2. **Prepare data:**
   - Place your `tr0.xlsx`, `tr1.xlsx`, `ts0.xlsx`, `ts1.xlsx` in the `model/` directory.

3. **Run cross-validation:**
   ```bash
   python tpx24w_apptime.py
   python tpx24w_apptime_roc.py
   ```

4. **Model files:**
   - Large model files (e.g., `xgb_model.model`) are not tracked by git due to GitHub's 100MB file size limit.
   - Please use the provided download link (https://asiaedutw-my.sharepoint.com/:u:/g/personal/ren_live_asia_edu_tw/ETK5Qkw91RhGlksZbC7zM4QBYAadDYo5Xl0pUeUF14703w?e=TVO61c) to obtain these files if needed.

## Results

- **5-fold cross-validation (with all test samples included in each fold):**
  - Precision: 0.9397 ± 0.0085
  - Accuracy: 0.9597 ± 0.0082
  - Recall (TPR): 0.8097 ± 0.0467
  - AUC: 0.96

## Citation

If you use this code or pipeline in your research, please cite:

> [Detecting High-Risk Smartphone Addiction among College Students via AI-Based App Usage Analysis During COVID-19]  
> [Zhi-Ren Tsai, Jeffrey J.P. Tsai, and Huei-Chen Ko]  
> [Conference/Journal], 2025

## Contact

For questions or access to large model files, please contact:  
[ren@asia.edu.tw]
