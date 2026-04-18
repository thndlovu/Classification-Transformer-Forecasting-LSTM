# LSTM Experiment Results

All experiments use the Jena Climate dataset.  
**Baseline config:** LSTM(32), 10 epochs, batch_size=256, lr=0.001

---

## Experiment 1 — LSTM Units: 32 → 64

**Change:** `keras.layers.LSTM(64)`

| Metric | Value |
|---|---|
| Val Loss (final) | 0.1271 |
| Train Loss (final) | 0.0985 |
| vs Baseline | +6.3% worse |

**Finding:** More capacity caused faster overfitting within 10 epochs. Val loss diverged from train loss earlier than the baseline.

---

## Experiment 2 — Stacked LSTM (32 + 32)

**Change:** Added second LSTM layer with `return_sequences=True`

| Metric | Value |
|---|---|
| Val Loss (final) | 0.1293 |
| Train Loss (final) | 0.1007 |
| vs Baseline | +8.1% worse |

**Finding:** Stacking layers added parameters but hurt generalization. 10 epochs is insufficient for a deeper architecture to converge.

---

## Experiment 3 — Extended Training: 10 → 20 Epochs

**Change:** `epochs = 20` (architecture unchanged from baseline)

| Metric | Value |
|---|---|
| Val Loss (final) | 0.1098 |
| Train Loss (final) | 0.0945 |
| vs Baseline | **-8.2% better** |

**Finding:** Clear winner. The baseline was undertrained, not undercapacitated. Val loss improved consistently through all 20 epochs and was still declining at epoch 20, suggesting 25-30 epochs would improve further.

---

## Summary

| Config | Val Loss | Train Loss | vs Baseline |
|---|---|---|---|
| Baseline (32 units, 10 epochs) | 0.1196 | 0.0994 | — |
| Exp 1 (64 units, 10 epochs) | 0.1271 | 0.0985 | +6.3% worse |
| Exp 2 (Stacked 32+32, 10 epochs) | 0.1293 | 0.1007 | +8.1% worse |
| **Exp 3 (32 units, 20 epochs)** | **0.1098** | **0.0945** | **-8.2% better** |

**Key insight:** Training duration matters more than model capacity for this dataset and model size. The original 10-epoch limit in the Keras example is a practical demonstration constraint, not an optimal training budget.