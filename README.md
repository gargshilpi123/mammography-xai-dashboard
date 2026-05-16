# Mammography XAI Dashboard

Hybrid Vision Transformer (ViT) + DenseNet121 based mammography classification system with Explainable AI.

## Overview

This project combines Vision Transformers and DenseNet for breast cancer detection from mammography images. The framework integrates multiple Explainable AI (XAI) techniques to improve interpretability and trustworthiness in medical predictions.

## Features

* Hybrid ViT + DenseNet121 architecture
* GradCAM++ visualization
* Attention Rollout for Vision Transformer
* LIME explanations
* ROI extraction from mammograms
* Gradio interactive dashboard
* Mixed precision training
* Class imbalance handling

## Tech Stack

* Python
* PyTorch
* timm
* OpenCV
* Gradio
* scikit-learn
* LIME

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## Screenshots

### Dashboard

![Dashboard](screenshots/dashboard.png)

### Prediction & XAI Output

![Prediction](screenshots/prediction.png)

## Live Demo

https://huggingface.co/spaces/shilpigarghere/XAI_Mammo

## Research Status

Research manuscript currently in preparation / under review.

