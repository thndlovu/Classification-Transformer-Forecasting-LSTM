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
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── classification_transformer/
│   └── baseline.ipynb
├── forecasting_lstm/
│   ├── baseline.ipynb
│   └── experiments.ipynb
└── results/
    ├── baseline/
    │   ├── lstm_metrics.txt
    │   └── transformer_metrics.txt
    └── experiments/
        └── lstm_experiments_results.md
```

---

## Datasets

### FordA - Classification
- **Task:** Binary classification of motor sensor time-series
- **Input:** Sequences of length 500, single channel
- **Classes:** 2 (normal vs. fault)
- **Split:** 3,601 train / 1,320 test
- **Source:** UCR Time Series Archive

### Jena Climate - Forecasting
- **Task:** Predict temperature 72 hours ahead
- **Input:** 14 meteorological features, sampled hourly
- **Sequence length:** 720 timesteps (720 hours lookback)
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
| FF dimension | 4 |
| Dropout | 0.25 |
| MLP units | [128] |
| Epochs | 150 (early stopping) |

### LSTM Forecaster
A single-layer LSTM that processes sequential weather data and predicts temperature at a future timestep.

| Component | Value |
|---|---|
| LSTM units | 32 |
| Sequence length | 720 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Epochs | 10 |

---

## Baseline Results

### Transformer - FordA Classification

> ⚠️ **Compatibility Note:** The Transformer example was authored for Keras 2. On Colab's current environment (Keras 3.13.2 / TF 2.19.0), and even on TF 2.12.0 with GPU (tested on vast.ai RTX 4090, Quebec CA), the model consistently failed to learn, producing ~51% accuracy (majority class prediction). The issue stems from degenerate initialization in the attention layers when operating on single-channel inputs (shape 500×1) with `ff_dim=4`, which is too narrow to propagate useful gradients. Multiple fixes were attempted including TF version pinning, learning rate tuning, seed fixing, and input projection layers. All improvement experiments were therefore conducted on the LSTM model.

| Metric | Value |
|---|---|
| Test Accuracy | ~51% (degenerate - majority class) |
| Test Loss | ~0.693 |
| Environment | Keras 3.13.2 / TF 2.19.0 (Colab) |

### LSTM - Jena Climate Forecasting

| Metric | Value |
|---|---|
| Val Loss (MSE) | 0.1196 |
| Train Loss | 0.0994 |
| Best Val Loss | 0.1196 (epoch 10, still improving) |
| Epoch duration | ~57-64s on Colab T4 |

---

## Experiments (LSTM Forecaster)

Three controlled modifications applied to the LSTM baseline, one variable changed at a time.

### Experiment 1 - LSTM Units: 32 → 64

**Change:** `keras.layers.LSTM(64)`

| Metric | Baseline | Exp 1 |
|---|---|---|
| Val Loss | 0.1196 | 0.1271 |
| Train Loss | 0.0994 | 0.0985 |
| Change | — | +6.3% worse |

**Finding:** Doubling LSTM units increased capacity but hurt generalization. The larger model overfit faster within 10 epochs.

---

### Experiment 2 - Stacked LSTM (32 + 32)

**Change:** Added second LSTM layer with `return_sequences=True`

| Metric | Baseline | Exp 2 |
|---|---|---|
| Val Loss | 0.1196 | 0.1293 |
| Train Loss | 0.0994 | 0.1007 |
| Change | — | +8.1% worse |

**Finding:** Stacking two LSTM layers added depth but made results worse. 10 epochs is insufficient for a deeper architecture to converge.

---

### Experiment 3 - Extended Training: 10 → 20 Epochs

**Change:** `epochs = 20` (architecture unchanged)

| Metric | Baseline | Exp 3 |
|---|---|---|
| Val Loss | 0.1196 | 0.1098 |
| Train Loss | 0.0994 | 0.0945 |
| Change | — | **-8.2% better** |

**Finding:** Clear winner. The baseline was undertrained, not undercapacitated. Val loss improved consistently through all 20 epochs and was still declining at epoch 20.

---

## Benchmark Summary

| Config | Val Loss | Train Loss | vs Baseline |
|---|---|---|---|
| Baseline (32 units, 10 epochs) | 0.1196 | 0.0994 | — |
| Exp 1 (64 units, 10 epochs) | 0.1271 | 0.0985 | +6.3% worse |
| Exp 2 (Stacked 32+32, 10 epochs) | 0.1293 | 0.1007 | +8.1% worse |
| **Exp 3 (32 units, 20 epochs)** | **0.1098** | **0.0945** | **-8.2% better** |

**Key insight:** Training duration matters more than model capacity for this dataset and model size. The original 10-epoch limit in the Keras example is a practical demonstration constraint, not an optimal training budget.

---

## Reflection

### Which model was easier to understand and why?

The LSTM was significantly easier to understand. Its sequential nature reading one timestep at a time and maintaining a hidden state maps intuitively to how humans think about time series. The input, hidden state, and output all have clear physical meaning in the context of weather forecasting.

The Transformer's self-attention mechanism is more abstract. Every timestep attending to every other timestep simultaneously is powerful but harder to reason about for a 1D signal. The multi-head architecture and positional encoding add complexity that requires deeper background knowledge to interpret.

### What improvement did you try, and what did you learn?

The most informative experiment was Experiment 3. Extending training from 10 to 20 epochs (val_loss 0.1196 → 0.1098, -8.2%). The result taught a clear lesson: before adding model complexity, verify that the baseline is actually converging. Experiments 1 and 2 both added parameters and made results worse, while simply training longer produced the best outcome. Reaching for architectural changes when the real constraint is training budget is a common trap in deep learning.

---

## How to Reproduce

### Option A - Google Colab (recommended)
1. Open the notebook links below
2. Runtime → Change runtime type → T4 GPU
3. File → Save a copy in Drive
4. Runtime → Run All

| Notebook | Link |
|---|---|
| LSTM Baseline | [Open in Colab](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/timeseries/ipynb/timeseries_weather_forecasting.ipynb) |
| Transformer Baseline | [Open in Colab](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/timeseries/ipynb/timeseries_classification_transformer.ipynb) |

### Option B — Local
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