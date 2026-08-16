# BERT-Based Sentiment Analysis on IMDb Movie Reviews

A research-quality, fully reproducible Jupyter notebook that fine-tunes a pretrained **BERT-base-uncased** model for binary sentiment classification on the **IMDb Movie Reviews** dataset, with in-depth architectural inspection, attention visualization, gradient-based explainability, and a classical baseline for comparison.

**Notebook:** `bert_imdb_sentiment_analysis.ipynb`

---

## Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Environment](#environment)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Error Analysis](#error-analysis)
- [Explainability](#explainability)
- [Computational Cost](#computational-cost)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Overview

This project implements and evaluates a complete transfer-learning pipeline for sentiment analysis:

```
Text -> WordPiece Tokenizer -> BERT (12 encoder layers) -> Classification Head -> Sentiment
```

Rather than treating BERT as a black box, the notebook walks through every stage of the pipeline — tokenization, embeddings, layer-by-layer hidden-state evolution, multi-head attention, fine-tuning, evaluation, and token-level attribution — with a markdown explanation preceding every code cell and a "what happened" summary following key outputs. It also trains a TF-IDF + Logistic Regression baseline on the identical data split to quantify what pretraining and contextual representations add over classical bag-of-words methods.

## Research Question

> *How effectively can a pretrained BERT model be fine-tuned to classify sentiment, and what linguistic features and attention patterns contribute to its predictions?*

## Environment

The notebook was executed end-to-end on Google Colab with the following environment:

| Component | Version |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.15.0 |
| Datasets | 5.0.1 |
| scikit-learn | 1.9.0 |
| GPU | Tesla T4 (15.64 GB) |
| CUDA available | Yes |

## Dataset

- **Source:** IMDb Movie Reviews (Hugging Face `stanfordnlp/imdb`, with fallback to legacy `imdb` identifier)
- **Full dataset:** 25,000 training reviews / 25,000 test reviews, perfectly balanced (12,500 positive / 12,500 negative)
- **Data quality:** 0 missing values, 0 empty reviews, 96 exact duplicate texts (retained — no automatic removal), review length ranging from 52 to 13,704 characters (mean 1,325 characters)

**Splits used for this run** (stratified subsampling applied for tractable Colab runtime — configurable in `CONFIG`):

| Split | Size | Class balance |
|---|---|---|
| Training | 3,200 | 50 / 50 |
| Validation | 800 | 50 / 50 |
| Test | 2,000 | 50 / 50 |

> The full 50,000-review dataset can be used by setting `train_subset_size` / `test_subset_size` to `None` in the configuration cell.

## Methodology

| Stage | Detail |
|---|---|
| Tokenizer | `bert-base-uncased` WordPiece tokenizer |
| Max sequence length | 256 tokens (justified via percentile analysis: 50th=235, 75th=380, 90th=609, 95th=813, 99th=1,160 tokens; **44.15%** of reviews are truncated at this length) |
| Model | `BertForSequenceClassification` (`bert-base-uncased`), 109,483,778 parameters, all trainable |
| Optimizer | AdamW, learning rate 2e-5, weight decay 0.01 |
| Schedule | Linear warmup (10% of steps) then linear decay |
| Batch size | 16 (train), 32 (eval) |
| Epochs | 3 (early stopping patience = 2, based on validation F1) |
| Model selection | Best **validation F1** checkpoint — the test set was never used for tuning or model selection |
| Baseline | TF-IDF (unigrams + bigrams, 20,000 features) + Logistic Regression, trained on the identical split |

### Training Curve

| Epoch | Train Loss | Train F1 | Val Loss | Val F1 |
|---|---|---|---|---|
| 1 | 0.4407 | 0.7842 | 0.3097 | 0.8952 |
| 2 | 0.2232 | 0.9183 | 0.2713 | **0.9043** ← best checkpoint |
| 3 | 0.1128 | 0.9677 | 0.3447 | 0.9040 |

Training loss decreases monotonically while validation loss begins to rise slightly after epoch 2, alongside a widening train/validation F1 gap — a mild but expected sign of overfitting on a relatively small (3,200-example) training subset over 3 epochs. The checkpoint from epoch 2 (highest validation F1) was restored and used for all downstream evaluation, per the project's rule that model selection must not use test performance.

Total training time: **7.88 minutes** on a Tesla T4 GPU.

## Results

### Final Test Set Performance (BERT, evaluated once, held-out test set)

| Metric | Value |
|---|---|
| Accuracy | **0.8975** |
| Precision | 0.9111 |
| Recall | 0.8810 |
| F1 | **0.8958** |
| ROC-AUC | 0.9655 |
| PR-AUC | 0.9649 |
| Specificity | 0.9140 |
| False Positive Rate | 0.0860 |
| False Negative Rate | 0.1190 |

### Confusion Matrix (Test Set, n = 2,000)

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 914 (TN) | 86 (FP) |
| **Actual Positive** | 119 (FN) | 881 (TP) |

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.8190 | 0.8109 | 0.8320 | 0.8213 | 0.9108 | 0.9091 |
| **BERT (fine-tuned)** | **0.8975** | **0.9111** | **0.8810** | **0.8958** | **0.9655** | **0.9649** |

Fine-tuned BERT outperforms the TF-IDF + Logistic Regression baseline by **+7.85 points of accuracy** and **+7.45 points of F1** on this split, consistent with the expectation that contextual, pretrained representations capture nuances (negation, long-range dependency, word sense) that bag-of-words features cannot.

### Threshold Analysis (Validation Set Only)

The optimal classification threshold on `P(positive)` for maximizing validation F1 was **0.22** (validation F1 = 0.9106), versus the default 0.5. All test-set metrics reported above use the default 0.5 threshold; the alternative threshold is reported for diagnostic purposes only and was **not** used to affect the reported test results, in keeping with the rule against threshold-tuning on test data.

## Error Analysis

- **86 false positives** (true negative, predicted positive) and **119 false negatives** (true positive, predicted negative) out of 2,000 test examples.
- **59 high-confidence errors** (model confidence > 0.95 while incorrect) — these are the most concerning failure mode, since the model gives no signal (via confidence) that it might be wrong.
- Qualitative review of misclassified examples suggests plausible contributors including **mixed sentiment** within a single review, **backhanded or hedged praise/criticism**, and **long reviews subject to truncation** at 256 tokens, consistent with the 44% truncation rate reported above. These are illustrative observations from the sampled examples, not a statistically validated taxonomy of error causes.

## Explainability

Two complementary interpretability methods are used and explicitly distinguished:

1. **Attention visualization** (Layers 1, 6, 12 × Heads 1, 6, 12) shows *where* the model's self-attention mechanism looked when building token representations. This is a mechanism, not a causal explanation of a specific prediction.
2. **Integrated Gradients token attribution** shows *how much* each input token's presence moved the model's output logit toward "positive" or "negative" — a closer approximation of a causal explanation.

Example (`explain_prediction`):

> **Text:** *"The acting was wooden and the plot made no sense, but the soundtrack was fantastic."*
> **Prediction:** POSITIVE (P = 0.5956)
> **Top attributions:** `fantastic` (+1.00), `sense` (−0.61), `and` (+0.33), `was` (+0.30), `but` (+0.28)

The model correctly identifies "fantastic" as the dominant positive signal despite two preceding negative clauses, illustrating sensitivity to a sentence-final positive pivot — though the prediction confidence (0.60) is comparatively low, reflecting the genuinely mixed sentiment in the input.

## Computational Cost

| Metric | Value |
|---|---|
| Total parameters | 109,483,778 |
| Trainable parameters | 109,483,778 (full fine-tuning) |
| Training time (3 epochs, 3,200 examples) | 7.88 minutes |
| Test-set inference time (2,000 examples) | 31.80 seconds |
| Average inference time per sample | 15.90 ms |
| Peak GPU memory used | 5.38 GB |
| Baseline (TF-IDF + LR) training time | 0.51 seconds |

BERT requires roughly **900x** the training time of the classical baseline for a **+7.45 point F1** improvement — a trade-off to weigh explicitly against deployment constraints.

## How to Run

1. Open `bert_imdb_sentiment_analysis.ipynb` in Google Colab.
2. Set the runtime to a GPU instance (**Runtime → Change runtime type → T4 GPU** or better).
3. Run all cells top to bottom (**Runtime → Run all**).
4. To reproduce results on the full 50,000-review IMDb dataset instead of the subsampled split above, set `train_subset_size` and `test_subset_size` to `None` in the **Configuration** cell before running.
5. Outputs (plots, model checkpoint, tokenizer, predictions, tables) are written to the `outputs/` directory created by the notebook.

## Project Structure

```
outputs/
├── model/            # Fine-tuned BERT checkpoint + label mapping
├── tokenizer/         # Saved tokenizer
├── plots/             # All generated figures (EDA, training curves, confusion matrix, ROC/PR, attention heatmaps, etc.)
├── predictions/        # test_predictions.csv — per-example predictions and probabilities
├── explainability/      # Token attribution artifacts
├── errors/            # Error analysis artifacts
├── tables/            # model_comparison.csv, computational_analysis.csv
└── reports/           # config.json (full run configuration)
```

## Limitations

- **Domain-specific:** fine-tuned only on movie reviews; may not transfer to other domains without further fine-tuning.
- **English-only:** `bert-base-uncased` was pretrained on English text.
- **Sequence truncation:** 256-token limit truncates ~44% of reviews, potentially discarding sentiment-relevant content in longer reviews.
- **Sarcasm and mixed sentiment:** remain difficult, as illustrated in the error analysis above.
- **Subsampled training run:** this run used 3,200/800/2,000 train/val/test examples for tractable Colab runtime rather than the full 50,000-review dataset; results may shift (likely upward, given more training data) on a full-dataset run.
- **Attention ≠ explanation:** attention weights indicate where the model attended, not why it reached a decision; only the Integrated Gradients attributions approximate causal influence.

## Future Work

- Full-dataset fine-tuning (50,000 reviews) and comparison against the current subsampled results.
- Larger or more recent pretrained encoders (e.g. RoBERTa, DeBERTa) under the same evaluation protocol.
- Systematic, statistically-grounded attention-head probing rather than illustrative spot-checks.
- Dedicated evaluation subsets for sarcasm and mixed-sentiment reviews.
