# Comparative Analysis of RNN and LSTM for Cybersecurity Attack Detection
### Vanishing-Gradient Analysis and Explainable AI on a Synthetic Cybersecurity Dataset

A research-style Jupyter notebook that goes beyond "train a model, print accuracy." It treats
cybersecurity events as **sequences**, trains a vanilla RNN and an LSTM side by side, and
empirically measures the **vanishing-gradient problem**, **layer-by-layer behavior**, and
**feature/time-step explainability** (via Captum Integrated Gradients) — all executed
end-to-end with real, non-fabricated results.

> ⚠️ **Dataset is synthetic.** The results below describe *this specific synthetic dataset*
> and are a methodology demonstration, not evidence of real-world IDS performance. See
> [Limitations](#limitations).

---

## Contents

- [`cybersecurity_rnn_lstm_analysis.ipynb`](./cybersecurity_rnn_lstm_analysis.ipynb) — the full notebook (128 cells: 66 markdown explanations, 62 code cells)
- `outputs/models/` — trained model checkpoints (`best_mlp.pt`, `best_rnn.pt`, `best_lstm.pt`)
- `outputs/plots/` — all generated figures (26 PNGs)
- `outputs/tables/` — CSV result tables
- `outputs/predictions/` — error-analysis CSV

## Dataset

**Cybersecurity Attacks Dataset** (synthetic) — 40,000 records, 25 raw features, including
network metadata (ports, protocol, packet length), security flags (malware indicators,
firewall/IDS logs), and context fields (user, device, geo-location).

Key finding from inspection (Section 15 of the notebook): the `Attack Type` column contains
**only three values — `DDoS`, `Malware`, `Intrusion` — with no Normal/Benign class**. This
notebook therefore treats the task as **3-class attack-type classification**, not binary
intrusion detection, and states that explicitly rather than assuming otherwise.

## Methodology

| Stage | What happens |
|---|---|
| Data quality & leakage checks | Missingness, duplicates, invalid ports, and a mutual-information leakage check on security-label fields (none found to leak the target) |
| Chronological split | 70% / 15% / 15% train/val/test **sorted by timestamp**, not random — avoids training on "future" events |
| Sequence construction | Sliding windows of 20 consecutive events per split, final-event labeling, no window crosses a split boundary |
| Models | MLP baseline (temporal-order-agnostic) vs. vanilla RNN vs. LSTM, same hidden size/training budget |
| Gradient analysis | Per-layer gradient norms logged every epoch; a direct `‖∂L/∂h_t‖` vs. time-step experiment (not assumed, measured) |
| Explainability | Captum Integrated Gradients → global feature importance + time-step × feature attribution heatmap |
| Evaluation | Accuracy / macro & weighted Precision / Recall / F1 / ROC-AUC / PR-AUC / specificity / FPR / FNR on held-out test data |

**Note on training budget:** `epochs` was set to 6 (not 20) in `CONFIG` to keep the notebook
runnable end-to-end on a CPU-only machine in reasonable time. This is documented and fully
configurable in the notebook.

---

## Results

All numbers below are copied directly from the notebook's own executed output — nothing is
hand-typed or estimated.

### Model comparison (held-out test set)

| Model | Parameters | Train time (s) | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC | PR-AUC | Specificity | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP  | 40,643 | 1.9 | 0.3342 | 0.3329 | 0.3338 | 0.3288 | 0.5008 | 0.3324 | 0.6669 | 0.3331 | 0.6662 |
| RNN  | 6,339  | 3.5 | 0.3339 | 0.3326 | 0.3333 | 0.3301 | 0.4941 | 0.3307 | 0.6667 | 0.3333 | 0.6667 |
| LSTM | 24,771 | 5.8 | 0.3249 | 0.3251 | 0.3255 | 0.3176 | 0.4931 | 0.3285 | 0.6627 | 0.3373 | 0.6745 |

**Best model by macro-F1: RNN (0.3301)** — though all three models, including the
non-sequential MLP, land within noise of each other and of the 3-class random-guessing
baseline (0.333 accuracy / F1).

### Computational comparison

| Model | Parameters | Training time (s) | Inference (ms/batch) |
|---|---|---|---|
| MLP  | 40,643 | 1.9 | 0.461 |
| RNN  | 6,339  | 3.5 | 1.416 |
| LSTM | 24,771 | 5.8 | 2.282 |

LSTM has ~4× the recurrent parameters of the RNN (four gates vs. one recurrence) and the
highest training/inference cost, without a corresponding accuracy gain on this dataset.

### Vanishing-gradient experiment (measured, not assumed)

Direct `‖∂L/∂h_t‖` measurement across the 20 time steps of one test sequence:

- **RNN** — first-step gradient norm ≈ 0.000000, last-step norm ≈ 0.311835
- **LSTM** — first-step gradient norm ≈ 0.000000, last-step norm ≈ 0.410381

Both architectures show gradient magnitude collapsing to near-zero at the earliest time step
relative to the most recent one — a directly observed vanishing-gradient effect (see the
notebook's `outputs/plots/vanishing_gradient_comparison.png`). The LSTM retains a slightly
larger raw gradient norm at the final step, consistent with (though not dramatically
demonstrating, at this hidden size / sequence length) the theoretical expectation that gated
architectures preserve gradient flow better than a vanilla RNN.

### Explainability (Captum Integrated Gradients)

- **Top RNN features:** `Source Port`, `Packet Length`, `Destination Port`, `Attack Signature_Known Pattern A`, `Action Taken_Blocked`
- **Top LSTM features:** `Source Port`, `Packet Type_Control`, `Packet Length`, `Protocol_TCP`, `Packet Type_Data`
- **Most influential time step (LSTM):** t = 20 of 20 — i.e. the model relies most on the final (labeled) event, as expected under the final-event labeling strategy.

### Conclusion (from the notebook's own generated summary)

> On this synthetic dataset, the RNN achieved the higher macro-F1 (0.3301 vs 0.3176 for
> LSTM). Both models perform close to the 3-class random-guessing baseline, and the ROC-AUC
> values (~0.49–0.50) indicate the models found **little to no genuinely predictive signal**
> in these features for this label. This is consistent with the dataset's synthetic,
> template-generated construction (Section 8/12 of the notebook show attack type is
> essentially independent of protocol/traffic type/timestamp). The experiment nonetheless
> successfully demonstrates the full methodology — sequence construction, gradient-flow
> measurement, and explainability — which is the actual purpose of this notebook.

---

## Key takeaways

1. **RQ1 — Can events be sequenced for RNN/LSTM input?** Yes — the pipeline builds valid
   sliding-window sequences with no leakage across splits.
2. **RQ2 — Does LSTM outperform vanilla RNN?** Not on this dataset — performance is
   statistically indistinguishable from random guessing for all three models, so no
   meaningful architecture advantage could be observed here.
3. **RQ3/RQ4 — Vanishing gradients / LSTM preservation?** Both architectures show clear
   gradient shrinkage across the 20-step window; LSTM retains a modestly larger gradient norm
   at the final step.
4. **RQ5 — Which features/time-steps matter?** `Source Port` and `Packet Length` dominate
   attribution for both models; the final (most recent) event in the window dominates
   temporal attribution, matching the labeling strategy.
5. **RQ6 — Which layer produces the final prediction?** The RNN/LSTM layer only produces a
   *representation*; the subsequent Linear → Softmax → Argmax pipeline produces the actual
   class prediction (explicitly traced in the notebook).

## Limitations

- Dataset is **synthetic** — timestamps, IPs, and payloads appear randomly/template-generated,
  not sampled from real traffic, and near-chance ROC-AUC (~0.49–0.50) suggests attack type is
  largely independent of the available features in this data source.
- No real session/flow structure — sequences are built purely from row order.
- Feature-leakage check found no leakage, but was only run on this dataset.
- Training budget (6 epochs, hidden size 64, single split) was reduced for CPU tractability;
  results may differ with a larger budget or GPU-scale training.
- Explainability results (Integrated Gradients) are baseline-dependent and should be read as
  relative importance, not causal explanation.

See the notebook's own **Section 57 (Limitations)** and **Section 58 (Future Work)** for the
full discussion, including suggested next steps (GRU, BiLSTM, Transformers, real-world
datasets like CIC-IDS2017/UNSW-NB15, cross-dataset evaluation).

## Reproducing

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn captum jupyter
jupyter nbconvert --to notebook --execute cybersecurity_rnn_lstm_analysis.ipynb
```

Set the dataset path via the `CYBERSEC_DATA_PATH` environment variable (defaults to
`cybersecurity_attacks.csv` in the working directory):

```bash
CYBERSEC_DATA_PATH=/path/to/cybersecurity_attacks.csv jupyter nbconvert --to notebook --execute cybersecurity_rnn_lstm_analysis.ipynb
```

All hyperparameters (sequence length, hidden size, epochs, etc.) live in the `CONFIG` dict at
the top of the notebook.

## License

Add your preferred license here (e.g. MIT).
