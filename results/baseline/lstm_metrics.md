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