# BERT Full Fine-Tuning vs. LoRA (Experimental Comparison)

A controlled experimental comparison of **full-parameter fine-tuning** vs. **LoRA
(Low-Rank Adaptation)** fine-tuning of `bert-base-uncased` for binary sentiment
classification on the **IMDb Movie Reviews** dataset.

Both approaches are trained and evaluated under an identical protocol — same data
split, tokenizer, sequence length, batch size, learning rate, epochs, optimizer,
scheduler, and checkpoint-selection criterion — so that any measured difference in
performance, memory, time, or storage can be attributed to the *method*, not to
inconsistent experimental conditions.

> **Notebook:** [`bert_full_vs_lora_experiment.ipynb`](./bert_full_vs_lora_experiment.ipynb)
> **Runtime used:** Google Colab, Tesla T4 GPU (14.56 GB)
> **Status:** Executed end-to-end; all results below are measured, not projected.

---

## Table of Contents

- [Research Question & Hypotheses](#research-question--hypotheses)
- [Experimental Setup](#experimental-setup)
- [How to Run](#how-to-run)
- [Results](#results)
  - [Predictive Performance](#predictive-performance)
  - [Parameter Efficiency](#parameter-efficiency)
  - [GPU Memory](#gpu-memory)
  - [Training & Inference Time](#training--inference-time)
  - [Model Storage](#model-storage)
  - [Prediction Agreement & Error Analysis](#prediction-agreement--error-analysis)
  - [Efficiency Summary](#efficiency-summary)
- [Hypothesis Evaluation](#hypothesis-evaluation)
- [Key Finding & Likely Cause](#key-finding--likely-cause)
- [Explainability](#explainability)
- [Limitations](#limitations)
- [Repository / Output Structure](#repository--output-structure)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Research Question & Hypotheses

**Research question:** How does parameter-efficient LoRA fine-tuning compare with
full-parameter BERT fine-tuning for sentiment classification, when both approaches use
identical data, preprocessing, hyperparameters, and evaluation protocol?

| # | Hypothesis |
|---|---|
| H1 | LoRA should achieve competitive performance while substantially reducing trainable parameters. |
| H2 | LoRA should require far fewer trainable parameters and produce a much smaller checkpoint. |
| H3 | LoRA may reduce peak GPU memory, but by less than the parameter-reduction percentage (activations and the frozen backbone still consume memory). |
| H4 | Full fine-tuning may reach different predictive performance since all parameters can adapt. |

Results for each hypothesis are in [Hypothesis Evaluation](#hypothesis-evaluation) below.

---

## Experimental Setup

| Setting | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Dataset | IMDb (Hugging Face `imdb`) |
| Train / Val / Test | 3,200 / 800 / 2,000 (stratified subsample of the full 25k/25k IMDb split — see [Limitations](#limitations)) |
| Max sequence length | 256 tokens |
| Batch size | 16 (train), 32 (eval) |
| Epochs | 3 |
| Learning rate | 2e-5 (**identical for both experiments**) |
| Weight decay | 0.01 |
| Warmup ratio | 10% |
| Optimizer / Scheduler | AdamW / linear warmup + decay |
| Checkpoint selection | Best **validation F1** (test set untouched until final evaluation) |
| LoRA config | `r=8`, `alpha=16`, `dropout=0.1`, `target_modules=["query","value"]`, `bias="none"` |
| Seed | 42 |
| Hardware | Tesla T4, 14.56 GB |

Both models start from the same pretrained `bert-base-uncased` checkpoint, see the
same tokenized batches (same tokenizer, same `max_length`), and are trained/evaluated
by the **same shared training and evaluation functions** — the only structural
difference is that LoRA freezes the BERT backbone and trains only its low-rank
adapters plus the classification head.

---

## How to Run

1. Open `bert_full_vs_lora_experiment.ipynb` in Google Colab.
2. Runtime → Change runtime type → **GPU** (T4 or better).
3. Run all cells top to bottom (Runtime → Run all).
4. If you hit an import error immediately after the install cell, **Runtime → Restart
   session**, then run all cells again — this is required once per fresh Colab
   session because pip-installed packages don't replace what's already loaded in the
   running Python process.

All hyperparameters live in a single `CONFIG` dictionary near the top of the notebook.
`train_subsample` / `test_subsample` are set to `4000` / `2000` by default purely to
keep the notebook runnable in one Colab session; set both to `None` for a full-scale
run on the complete 25k/25k IMDb data.

---

## Results

### Predictive Performance

Measured once on the held-out, untouched 2,000-example test set:

| Metric | Full BERT | LoRA |
|---|---:|---:|
| Accuracy | **0.9150** | 0.6745 |
| Precision | **0.9014** | 0.6386 |
| Recall | 0.9320 | 0.8040 |
| F1 | **0.9164** | 0.7118 |
| ROC-AUC | **0.9705** | 0.7539 |
| PR-AUC | **0.9694** | 0.7422 |
| Specificity | **0.8980** | 0.5450 |
| FPR | **0.1020** | 0.4550 |
| FNR | 0.0680 | **0.1960** |

Full fine-tuning outperformed LoRA on every metric in this run (F1 gap: **+0.2046**
in favor of Full BERT). See [Key Finding & Likely Cause](#key-finding--likely-cause)
for why, and how to close this gap.

### Parameter Efficiency

| Metric | Full BERT | LoRA |
|---|---:|---:|
| Total parameters | 109,483,778 | 109,780,228 |
| Trainable parameters | 109,483,778 | **296,450** |
| Frozen parameters | 0 | 109,483,778 |
| Trainable % | 100.00% | **0.2700%** |
| **Trainable-parameter reduction** | — | **99.73%** |

LoRA trains **369x fewer** parameters than full fine-tuning (296,450 vs. 109.48M),
of which 294,912 are LoRA adapter weights and 1,538 are the classification head.

### GPU Memory

Peak memory recorded via `torch.cuda.max_memory_allocated/reserved` under identical
measurement conditions for both runs:

| Metric | Full BERT | LoRA |
|---|---:|---:|
| Peak allocated | 4,242.5 MB | **3,307.0 MB** |
| Peak reserved | 4,368.0 MB | **3,494.0 MB** |
| **Memory reduction** | — | **935.5 MB (22.05%)** |

As anticipated in H3, the memory reduction (22.05%) is far smaller than the
parameter-count reduction (99.73%) — the frozen backbone's forward-pass activations
and the identical batch size dominate memory use in both experiments; only optimizer
state and gradient buffers shrink meaningfully with LoRA.

### Training & Inference Time

| Metric | Full BERT | LoRA |
|---|---:|---:|
| Total training time | 448.6 s | **351.5 s** |
| Avg. epoch time | 149.5 s | **117.1 s** |
| Avg. step time | 0.748 s | **0.586 s** |
| Inference time (2,000 samples) | 29.71 s | 30.40 s |
| Inference throughput | **67.3 samples/s** | 65.8 samples/s |

LoRA trained **21.65% faster** overall (fewer gradients to compute/update), but
inference speed was essentially the same for both (LoRA's adapters add a small amount
of extra compute on top of the still-fully-executed frozen backbone).

### Model Storage

| Metric | Full BERT | LoRA |
|---|---:|---:|
| Checkpoint size | 417.7 MB | **1.1 MB** |
| **Storage reduction** | — | **99.73%** |

Note: the 1.1 MB LoRA artifact is *adapter-only* — using it for inference still
requires the frozen `bert-base-uncased` base checkpoint alongside it.

### Prediction Agreement & Error Analysis

Computed over the 2,000-example test set:

- **Agreement:** 70.85% | **Disagreement:** 29.15%
- Both models correct: 1,298 (64.9%)
- Both models wrong: 119 (6.0%)
- Full BERT correct, LoRA wrong: 532 (26.6%)
- LoRA correct, Full BERT wrong: 51 (2.5%)

|  | LoRA: Neg | LoRA: Pos |
|---|---:|---:|
| **Full: Neg** | 562 | 404 |
| **Full: Pos** | 179 | 855 |

### Efficiency Summary

| Model | F1 | Trainable Params (M) | F1 per Million Trainable Params |
|---|---:|---:|---:|
| Full BERT | 0.9164 | 109.484 | 0.0084 |
| LoRA BERT | 0.7118 | 0.296 | **2.4011** |

Despite its lower absolute F1, LoRA is **~287x more parameter-efficient** by this
measure — it extracts far more predictive performance per trainable parameter, even
though the raw performance ceiling it reached in this run was lower.

---

## Hypothesis Evaluation

| Hypothesis | Verdict | Basis |
|---|---|---|
| **H1** — LoRA competitive with full fine-tuning | ❌ **Not supported** | F1 gap of +0.2046 (LoRA 0.7118 vs. Full BERT 0.9164) exceeds the 0.03 threshold for "competitive" |
| **H2** — Far fewer trainable params & smaller checkpoint | ✅ **Supported** | 99.73% parameter reduction; 417.7 MB → 1.1 MB checkpoint |
| **H3** — Memory reduction smaller than parameter reduction | ✅ **Supported** | 22.05% memory reduction vs. 99.73% parameter reduction |
| **H4** — Full fine-tuning may differ in predictive performance | ✅ **Observed** | A substantial difference (+0.2046 F1) was measured |

---

## Key Finding & Likely Cause

**In this run, LoRA did not reach competitive accuracy with full fine-tuning** —
LoRA's validation F1 was still climbing at epoch 3 (0.596 → 0.671 → 0.697) and had
clearly not converged, whereas Full BERT converged within 1–2 epochs. The most likely
cause, based on the training curves, is **hyperparameter mismatch, not a fundamental
limitation of LoRA**:

- **Learning rate:** `2e-5` is tuned for updating ~110M pretrained weights with small
  per-step changes. LoRA's newly-initialized, randomly-initialized `A`/`B` matrices
  (296K parameters) typically need a **substantially higher learning rate**
  (commonly `1e-4` to `5e-4` in the LoRA literature) to converge in the same number of
  steps, since they start from scratch rather than from a pretrained state.
- **Epoch budget:** 3 epochs was sufficient for full fine-tuning but likely too few
  for LoRA's slower-starting optimization trajectory at this learning rate.
- **Training set size:** the 3,200-example training subsample (used for Colab runtime
  feasibility — see [Limitations](#limitations)) gives LoRA fewer gradient steps to
  close the gap than the full 20,000-example IMDb training split would.

This is intentionally reported here rather than adjusted after the fact, per the
notebook's rule against tuning based on outcomes — but it is worth rerunning with a
higher LoRA-specific learning rate (leaving Full BERT's LR untouched) or the full
dataset/epoch budget to test whether the gap closes, which is consistent with results
generally reported in the LoRA literature.

---

## Explainability

Integrated Gradients (Captum) was run on representative correct/incorrect examples for
both models. Both models attributed high (negative-direction) importance to `[SEP]`
and similar sentiment-bearing tokens (e.g., `love`, `bad`, `grief`) for correctly
classified examples, suggesting LoRA's frozen-backbone representations still carry
usable sentiment signal even though its classifier head had not fully learned to
exploit them within 3 epochs. Full attribution tables and attention-map comparisons are
in the notebook (Sections 39–40); a mean absolute attention-weight difference of
**0.0056** was observed between the two models on one example/layer/head — a
descriptive observation only, not a general claim about model behavior.

---

## Limitations

- **Subsampled data:** Results use a stratified 3,200/800/2,000 train/val/test
  subsample of full IMDb (25k/25k) for Colab runtime feasibility. Both experiments
  always used the *identical* subsample, so the comparison remains fair, but absolute
  performance (especially LoRA's, given the convergence issue above) may differ on the
  full dataset.
- **Single learning rate for both methods:** matching the learning rate across
  experiments was a deliberate fairness control (Rule: no secretly different
  hyperparameters), but it disadvantages LoRA, whose adapters are randomly initialized
  and typically benefit from a higher LR than full fine-tuning.
- **Single seed:** all results are from `seed=42`. Multi-seed replication (42, 123,
  456) is implemented in the notebook but disabled by default due to compute cost.
- **LoRA target modules:** only `query` and `value` projections were adapted (the
  standard configuration from the original LoRA paper); other configurations (e.g.,
  adding `key`, feed-forward layers, or a higher rank `r`) were not tested.
- **Explainability metrics:** Integrated Gradients results are reported qualitatively
  (top attributed tokens per example) rather than as a single aggregate agreement
  score between models.

---

## Repository / Output Structure

Running the notebook populates:

```
outputs/
├── models/
│   ├── full_bert/          # Full BERT checkpoint (.bin)
│   └── lora/adapter/        # LoRA adapter (PEFT save_pretrained)
├── plots/                   # EDA, training curves, confusion matrices, ROC/PR,
│                             # parameter/memory/time comparison charts
├── predictions/
├── explainability/
├── errors/                  # error_analysis_full.csv
├── tables/                  # final_comparison.csv, efficiency_analysis.csv
├── logs/
└── reports/                 # final_conclusion.md (auto-generated)
```

---

## Reproducibility

- Seed `42` fixed across Python, NumPy, PyTorch, and CUDA; deterministic cuDNN mode
  enabled.
- Both experiments share one tokenizer instance, one set of DataLoaders, one
  `CONFIG` dictionary, and one training/evaluation function — eliminating an entire
  class of "silently different settings" bugs.
- Checkpoint selection is by validation F1 only; the test set is touched exactly once
  per model, at the very end.
- Pinned package versions used for this run: `transformers==4.44.2`,
  `datasets==2.21.0`, `peft==0.12.0`, `accelerate==0.34.2`, `torch==2.11.0+cu128`.

---

## License

This notebook and README are provided for research/educational use. IMDb dataset
usage is subject to its own [Hugging Face dataset card](https://huggingface.co/datasets/imdb)
terms; `bert-base-uncased` usage is subject to Google's original BERT license terms
as distributed via Hugging Face.
