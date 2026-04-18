# Time Series Modeling with Deep Learning
### Classification using Transformer and Forecasting using LSTM
*CMPE 401 - Instructor-defined Project 2*

---

## Overview

This project benchmarks two deep learning approaches for time-series modeling using official Keras examples as a starting point:

- **Transformer** for time-series **classification** on the FordA dataset
- **LSTM** for time-series **forecasting** on the Jena Climate dataset

The goal is to reproduce baseline results, run controlled improvement experiments, and analyze what changes actually move the needle.

---

## Repository Structure

```
Classification-Transformer-Forecasting-LSTM/
│
├── classification_transformer/
│   ├── baseline.ipynb              
│   └── experiments.ipynb           
│
├── forecasting_lstm/
│   └── baseline.ipynb              
│
├── results/
│   ├── baseline/
│   │   ├── transformer_metrics.txt
│   │   └── lstm_metrics.txt
│   └── experiments/
│       ├── exp1_transformer_blocks/
│       ├── exp2_attention_heads/
│       └── exp3_dropout/
│
├── requirements.txt
└── README.md
```

---

## Datasets

### FordA - Classification
- **Task:** Binary classification of motor sensor time-series
- **Input:** Sequences of length 500, single channel
- **Classes:** 2 (normal vs. fault)
- **Split:** 3,601 train / 1,320 test
- **Source:** UCR Time Series Archive

### Jena Climate — Forecasting
- **Task:** Predict temperature 24 hours ahead
- **Input:** 14 meteorological features, 10-minute intervals
- **Sequence length:** 120 timesteps (20 hours of lookback)
- **Source:** Max Planck Institute for Biogeochemistry

---

## Models

### Transformer Classifier
The model uses multi-head self-attention to capture global dependencies across the time series, followed by a feedforward classification head.

| Component | Value |
|---|---|
| Transformer blocks | 2 |
| Attention heads | 4 |
| Key dimension | 256 |
| Dropout | 0.25 |
| MLP units | [128, 64] |
| Epochs | 200 |

### LSTM Forecaster
A single-layer LSTM that processes sequential weather data and predicts temperature at a future timestep.

| Component | Value |
|---|---|
| LSTM units | 32 |
| Sequence length | 120 |
| Batch size | 256 |
| Epochs | 10 |

---

## Baseline Results

> Results recorded after running official Keras notebooks unmodified.

### Transformer - FordA Classification

| Metric | Value |
|---|---|
| Test Accuracy | *to be filled* |
| Test Loss | *to be filled* |

### LSTM - Jena Climate Forecasting

| Metric | Value |
|---|---|
| Val MAE | *to be filled* |
| Val Loss | *to be filled* |

---

## Experiments (Transformer Classifier)

Three controlled modifications were applied to the Transformer baseline, one variable changed at a time.

### Experiment 1 - Number of Transformer Blocks

| Config | Test Accuracy | Test Loss |
|---|---|---|
| 1 block | *TBF* | *TBF* |
| 2 blocks (baseline) | *TBF* | *TBF* |
| 4 blocks | *TBF* | *TBF* |

**Finding:** *to be filled after running*

---

### Experiment 2 - Number of Attention Heads

| Config | Test Accuracy | Test Loss |
|---|---|---|
| 4 heads (baseline) | *TBF* | *TBF* |
| 8 heads | *TBF* | *TBF* |

**Finding:** *to be filled after running*

---

### Experiment 3 - Dropout Rate

| Config | Test Accuracy | Test Loss |
|---|---|---|
| 0.1 | *TBF* | *TBF* |
| 0.25 (baseline) | *TBF* | *TBF* |
| 0.4 | *TBF* | *TBF* |

**Finding:** *to be filled after running*

---

## Loss Curve Analysis

> Plots saved to `results/experiments/` after each run.

*Analysis of training vs. validation loss curves - convergence behaviour, overfitting signals, and what each experiment's curves reveal - to be written after experiments complete.*

---

## Reflection

### Which model was easier to understand and why?
*To be written after completing both baselines.*

### What improvement did you try, and what did you learn?
*To be written after completing experiments.*

---

## How to Reproduce

### Option A - Google Colab (recommended)
1. Open the notebook links below
2. Runtime → Run All
3. Results save automatically

| Notebook | Colab Link |
|---|---|
| Transformer Baseline | [Open in Colab](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/timeseries/ipynb/timeseries_classification_transformer.ipynb) |
| LSTM Baseline | [Open in Colab](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/timeseries/ipynb/timeseries_weather_forecasting.ipynb) |

### Option B - Local / Jupyter
```bash
git clone https://github.com/thndlovu/Classification-Transformer-Forecasting-LSTM.git
cd Classification-Transformer-Forecasting-LSTM
pip install -r requirements.txt
jupyter notebook
```

---

## Dependencies

See `requirements.txt`. Core packages:

- TensorFlow / Keras ≥ 2.12
- NumPy
- Pandas
- Matplotlib

---

*CMPE 401 · UBC Okanagan · 2026*