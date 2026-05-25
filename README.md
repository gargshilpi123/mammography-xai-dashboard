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



## Explainable AI (XAI) Techniques

This project uses multiple Explainable AI (XAI) methods to visualize and interpret the model’s decision-making process for mammography classification.

### 1. GradCAM++

GradCAM++ is applied on the DenseNet121 feature maps to identify the most influential regions responsible for the prediction.

* Uses second-order gradients for improved localization.
* Highlights suspicious tumor regions spatially.
* Produces heatmaps showing where the CNN focuses most.
* Better than traditional GradCAM for multiple or small lesions.

In the visualization:

* Red/Yellow regions → high importance
* Blue regions → low importance

This helps radiologists understand which breast regions contributed most to the malignant prediction.

---

### 2. Attention Rollout (Vision Transformer)

Attention Rollout visualizes how the Vision Transformer (ViT) attends to different image patches across all transformer layers.

* Combines attention maps from all 12 transformer layers.
* Tracks cumulative patch importance using residual connections.
* Shows global contextual understanding of the mammogram.

Unlike CNNs that focus locally, ViT attention captures long-range dependencies and overall breast structure.

In the visualization:

* Bright regions → highly attended image patches
* Dark regions → less important patches

This provides insight into how the transformer interprets the mammogram globally.

---

### 3. LIME (Local Interpretable Model-Agnostic Explanations)

LIME explains predictions by perturbing image superpixels and observing how the prediction changes.

* Divides the image into interpretable superpixels.
* Identifies which regions positively or negatively influence prediction.
* Model-agnostic explanation technique.

In the visualization:

* Red regions → support malignant prediction
* Green regions → support benign prediction

LIME provides human-understandable local explanations and increases transparency of the AI system.

---

## Why XAI is Important in Medical Imaging

Explainability is critical in healthcare applications because doctors must understand why an AI system makes a prediction.

The combination of:

* GradCAM++ (CNN spatial attention)
* Attention Rollout (Transformer attention)
* LIME (local interpretable explanations)

improves:

* Model transparency
* Clinical trust
* Decision interpretability
* Reliability in breast cancer diagnosis

This hybrid XAI framework helps bridge the gap between deep learning predictions and medical understanding.

