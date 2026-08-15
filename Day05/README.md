# Attention Is All You Need — Transformer Reproduction

A transparent, from-scratch PyTorch reproduction of the Transformer architecture and training methodology introduced in:

> Vaswani et al., **“Attention Is All You Need”**, NeurIPS 2017.

This project focuses on faithfully implementing and inspecting the Transformer rather than treating it as a black-box library component. The implementation builds the major architectural components directly in PyTorch, including embeddings, sinusoidal positional encoding, scaled dot-product attention, multi-head attention, masking, feed-forward networks, residual connections, LayerNorm, encoder/decoder stacks, the Noam learning-rate schedule, label smoothing, greedy decoding, and beam search.

The notebook also includes parameter verification, activation tracing, training visualization, BLEU evaluation, attention visualization, attention-head statistics, gradient-flow analysis, computational-complexity analysis, and Base-vs-Big parameter comparisons.

---

## 1. Project Status

**Status:** Completed architectural reproduction and demo-scale end-to-end experiment

**Primary task:** WMT 2014 English → German (EN-DE)

**Implementation:** PyTorch, built from primitives

**Default execution mode:** Demo-scale WMT14 subset

**Execution environment recorded in the notebook:**

- Python: **3.12.13**
- PyTorch: **2.11.0+cpu**
- CUDA: **Unavailable**
- Device: **CPU**
- Random seed: **42**

The notebook explicitly distinguishes between:

1. **Architectural reproduction** — implemented and verified.
2. **Demo-scale training** — actually executed.
3. **Full paper reproduction** — not claimed, because the original paper used substantially more data, compute, training steps, and checkpoint averaging.

---

## 2. Important Reproducibility Note

This project is intentionally honest about the difference between reproducing the **architecture** and reproducing the **original training result**.

The original Transformer paper trained on approximately **4.5 million WMT14 EN-DE sentence pairs** and reported **27.3 BLEU for the Base model** and **28.4 BLEU for the Big model** on the reported EN-DE evaluation.

The notebook does **not** claim to reproduce those scores. The executed experiment used:

- `FULL_DATASET = False`
- **2,000** training sentence pairs
- **200** validation sentence pairs
- **200** test sentence pairs
- **300 development training steps**
- CPU execution
- A demo BPE vocabulary of **2,000 tokens**

The resulting BLEU score was **0.23**. This low score is expected under the intentionally restricted training setup and is treated as a pipeline/training sanity check rather than as a failure of the Transformer architecture.

---

# 3. Research Objective

The main objective is to reproduce the Transformer described in *Attention Is All You Need* and verify the implementation experimentally.

The project investigates:

- How self-attention replaces recurrence.
- How multi-head attention allows multiple learned projections/subspaces.
- How positional information is injected without recurrence.
- How encoder and decoder stacks are assembled.
- How causal masking prevents the decoder from seeing future tokens.
- How residual connections and LayerNorm support deep stacks.
- How the Noam learning-rate schedule is implemented.
- How label smoothing affects the training objective.
- How greedy and beam-search decoding operate.
- How the implementation's parameter count compares with the paper.
- How the model behaves under a small training budget.
- How attention heads and gradient norms can be inspected.
- Why self-attention is highly parallelizable despite its quadratic sequence-length cost.

---

# 4. Transformer Architecture

The reproduced architecture is an encoder-decoder Transformer.

### Encoder

Each encoder layer contains:

1. Multi-head self-attention
2. Residual connection + LayerNorm
3. Position-wise feed-forward network
4. Residual connection + LayerNorm

There are **6 encoder layers** in the Base configuration.

### Decoder

Each decoder layer contains:

1. Masked multi-head self-attention
2. Residual connection + LayerNorm
3. Encoder-decoder / cross-attention
4. Residual connection + LayerNorm
5. Position-wise feed-forward network
6. Residual connection + LayerNorm

There are **6 decoder layers** in the Base configuration.

### Final prediction

The decoder representation is projected through the final generator layer to produce vocabulary logits. The implementation uses a shared/tied embedding-output projection.

---

# 5. Paper Specification vs. Implementation

| Component | Original Paper — Base | This Project |
|---|---:|---:|
| Architecture | Transformer encoder-decoder | Same |
| Encoder layers | 6 | 6 |
| Decoder layers | 6 | 6 |
| `d_model` | 512 | 512 |
| `d_ff` | 2048 | 2048 |
| Attention heads | 8 | 8 |
| `d_k` | 64 | 64 |
| `d_v` | 64 | 64 |
| Dropout | 0.1 | 0.1 |
| Label smoothing | 0.1 | 0.1 |
| Optimizer | Adam | Adam |
| β1 | 0.9 | 0.9 |
| β2 | 0.98 | 0.98 |
| ε | 1e-9 | 1e-9 |
| Warmup steps | 4,000 | 4,000 |
| EN-DE vocabulary | ~37K shared BPE | 2K demo vocabulary; 37K architectural verification |
| Paper training steps | 100,000 | 300 demo steps |
| Beam size | 4 | 4 |
| Length penalty | 0.6 | 0.6 |

The project keeps these settings centralized in a configuration dictionary so that the executed setup can be identified without relying on hidden or hard-coded values.

---

# 6. Dataset

## WMT 2014 English → German

The primary dataset is WMT14 EN-DE.

The original paper used approximately **4.5 million sentence pairs**.

For the executed development experiment, the notebook loaded a genuine WMT14 subset:

| Split | Sentence Pairs |
|---|---:|
| Training | 2,000 |
| Validation | 200 |
| Test | 200 |
| **Total** | **2,400** |

The notebook does not replace WMT14 with a synthetic task. The demo subset comes from the WMT14 EN-DE distribution.

### Dataset statistics from the executed run

- Number of training sentence pairs: **2,000**
- Average English sentence length: **26.6 words**
- Average German sentence length: **23.6 words**
- Maximum English/German length: **130 / 130**
- Minimum English/German length: **1 / 1**

Example:

```text
EN: Resumption of the session
DE: Wiederaufnahme der Sitzungsperiode
```

---

# 7. Tokenization

The project uses Byte Pair Encoding (BPE) through the `tokenizers` package.

Two vocabulary settings are supported:

- **Demo vocabulary:** 2,000 tokens
- **Paper-scale vocabulary:** 37,000 tokens

The demo vocabulary is deliberately smaller so that tokenizer training and model execution remain practical in a CPU/Colab development environment.

The paper-scale 37K vocabulary is also instantiated for parameter-count verification.

---

# 8. Core Components Implemented

The Transformer is implemented without calling `torch.nn.Transformer` as a black box.

## 8.1 Token Embeddings

Input token IDs are converted into dense vectors using learned embeddings.

The Base configuration uses:

```text
d_model = 512
```

---

## 8.2 Sinusoidal Positional Encoding

Because the Transformer contains no recurrence or convolution, token order must be explicitly represented.

The project implements the sinusoidal positional encoding described in the paper and adds it to token embeddings.

The encoding provides position-dependent sine and cosine signals with different wavelengths.

---

## 8.3 Scaled Dot-Product Attention

The implementation follows the paper's central attention equation:

```text
Attention(Q, K, V)
    = softmax(QKᵀ / √d_k)V
```

The scaling term `√d_k` prevents excessively large dot products from producing poorly behaved softmax distributions.

---

## 8.4 Multi-Head Attention

The Base model uses:

```text
8 attention heads
d_model = 512
d_k = 64
d_v = 64
```

Each head operates in a separate learned projection space, after which the outputs are concatenated and projected back into the model dimension.

---

## 8.5 Attention Masking

Two major masks are implemented:

### Padding mask

Prevents attention from using padding tokens.

### Subsequent/causal mask

Used in decoder self-attention so that position `i` cannot attend to future target positions.

This preserves the autoregressive nature of translation.

---

## 8.6 Position-Wise Feed-Forward Network

Each Transformer layer contains:

```text
512 → 2048 → 512
```

with the paper's position-wise ReLU feed-forward formulation.

---

## 8.7 Residual Connections and LayerNorm

Residual connections surround the sublayers, followed by LayerNorm in the implementation.

These connections provide direct paths through the network and help maintain stable information and gradient flow across the six-layer encoder and decoder stacks.

---

# 9. Optimizer and Learning-Rate Schedule

The project uses Adam with the paper's reported hyperparameters:

```text
β1 = 0.9
β2 = 0.98
ε  = 1e-9
```

The Noam learning-rate schedule is implemented according to the Transformer paper:

```text
lr = d_model^(-0.5)
     × min(step^(-0.5), step × warmup_steps^(-1.5))
```

with:

```text
d_model = 512
warmup_steps = 4000
```

The notebook visualizes the learning-rate schedule, including the warm-up phase.

---

# 10. Label Smoothing

Label smoothing is implemented with:

```text
ε_ls = 0.1
```

The notebook demonstrates the difference between ordinary negative log-likelihood and label-smoothed loss on an example.

Recorded example:

| Objective | Loss |
|---|---:|
| Standard NLL | 7.8142 |
| Label-smoothed loss | 6.7583 |

The primary experiment uses label smoothing of **0.1**, matching the paper.

---

# 11. Training Configuration

The executed development run used:

```text
FULL_DATASET        = False
VOCAB_SIZE_DEMO     = 2000
MODEL_SIZE          = base

d_model             = 512
N                   = 6
h                   = 8
d_ff                = 2048
dropout             = 0.1

label_smoothing     = 0.1
Adam betas          = (0.9, 0.98)
Adam epsilon        = 1e-9
warmup_steps        = 4000

DEVELOPMENT_STEPS   = 300
PAPER_STEPS         = 100000

beam_size           = 4
length_penalty      = 0.6

RUN_ABLATIONS       = False
```

---

# 12. Training Results

The development run completed **300 training steps** on CPU.

### Training loss progression

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 1 | 6.6634 | 6.7029 |
| 20 | 6.5385 | 6.5054 |
| 40 | 6.2240 | 6.2201 |
| 60 | 6.0358 | 6.0491 |
| 80 | 5.9217 | 5.9297 |
| 100 | 5.8701 | 5.8513 |
| 120 | 5.7458 | 5.7381 |
| 140 | 5.6452 | 5.6507 |
| 160 | 5.5262 | 5.5888 |
| 180 | 5.5238 | 5.4796 |
| 200 | 5.5278 | 5.4490 |
| 220 | 5.3171 | 5.3815 |
| 240 | 5.3497 | 5.3527 |
| 260 | 5.0883 | 5.3212 |
| 280 | 5.2555 | 5.3118 |
| 300 | 5.2951 | 5.3250 |

### Training observations

The training loss decreased substantially:

```text
6.6634 → 5.2951
```

The validation loss also decreased:

```text
6.7029 → 5.3250
```

This confirms that the model learned from the development subset during the executed run.

The validation loss reached its lowest logged value around step 280:

```text
5.3118
```

Because only 300 steps were used, the model was nowhere near the training regime required for high-quality translation.

---

# 13. Training Performance

The notebook recorded:

```text
Training steps: 300
Training device: CPU
Training time: approximately 3866.4 seconds
```

This corresponds to approximately:

```text
64.4 minutes
```

for the development run.

Logged token throughput varied considerably during training, roughly from **90 to 174 tokens/sec** in the reported checkpoints.

This CPU result also demonstrates why the original paper required substantial GPU resources for full-scale training.

---

# 14. Parameter Verification

Parameter-count verification was performed using both the demo vocabulary and the paper-scale 37K vocabulary.

## Demo configuration

With the 2,000-token demo vocabulary:

```text
Total parameters      : 45,166,544
Trainable parameters  : 45,166,544
Encoder parameters    : 18,915,328
Decoder parameters    : 25,225,216
Embedding parameters  : 1,024,000
```

The embedding parameters are shared across the relevant embedding/output projections.

## Paper-scale Base configuration

With the 37,000-token vocabulary:

```text
Our implementation : 63,121,544 parameters
Paper              : approximately 65,000,000 parameters
Difference         : approximately -2.9%
```

This is a strong architectural consistency check. The remaining difference is attributed in the notebook to implementation/tokenizer bookkeeping and details such as bias/norm parameter accounting that are not fully specified by the paper.

---

# 15. Base vs. Big Model Verification

The project also instantiates the Base and Big configurations for parameter-count comparison.

| Model | Parameters in Project | Paper Report |
|---|---:|---:|
| Transformer Base | 63.1M | ~65M |
| Transformer Big | 214.3M | ~213M |

### Base

```text
N       = 6
d_model = 512
d_ff    = 2048
heads   = 8
dropout = 0.1
```

### Big

```text
N       = 6
d_model = 1024
d_ff    = 4096
heads   = 16
dropout = 0.3
```

The Big model was **not trained** in the executed notebook. It was instantiated only for architectural and parameter verification.

---

# 16. Inference Results

Both decoding mechanisms were implemented:

- Greedy decoding
- Beam search

The configured beam-search parameters are:

```text
Beam size       = 4
Length penalty  = 0.6
```

An example from the executed run:

```text
Source:
Gutach: Increased safety for pedestrians

Reference:
Gutach: Noch mehr Sicherheit für Fußgänger
```

The development-trained model generated repetitive output such as:

```text
Die, daß, daß, daß, ...
```

and beam search similarly produced repetitive sequences.

This is an expected result given:

- only 300 training steps,
- only 2,000 training sentence pairs,
- a 2K demo vocabulary,
- CPU-limited development training.

The decoding implementation itself is therefore being validated rather than presented as a high-quality translation system.

---

# 17. BLEU Evaluation

The notebook uses SacreBLEU for evaluation.

### Recorded result

```text
Our BLEU:
0.23
```

### Original paper comparison

```text
Paper Transformer Base EN-DE : 27.3 BLEU
Our demo run                : 0.23 BLEU
Difference                  : -27.07 BLEU
```

| Experiment | BLEU | Status |
|---|---:|---|
| Our Base EN-DE demo | **0.23** | Executed |
| Paper Base EN-DE | **27.3** | Reference |
| Paper Big EN-DE | **28.4** | Reference only |
| Paper Big EN-FR | **41.0** | Reference only |

### Interpretation

The **0.23 BLEU** result must not be interpreted as the performance of the Transformer architecture itself.

The experiment differs dramatically from the paper's training setup:

```text
Data:
2,000 pairs vs. ~4.5M pairs

Training:
300 steps vs. 100,000 steps

Hardware:
CPU vs. 8×P100 GPUs

Vocabulary:
2K demo BPE vs. ~37K shared BPE

Checkpoint averaging:
Not performed
```

Therefore, the BLEU score is primarily a **pipeline sanity check**.

---

# 18. Attention Explainability

The notebook visualizes:

1. Encoder self-attention
2. Decoder self-attention
3. Encoder-decoder cross-attention

Attention matrices from the last layer and all eight heads are inspected.

The project deliberately avoids claiming meaningful syntactic specialization from this short training run.

The notebook's observed output demonstrates that attention matrices can be extracted and visualized, but the model is too under-trained to support strong conclusions about learned linguistic behavior.

---

# 19. Attention Head Analysis

For each of the eight encoder attention heads, the notebook calculates:

- Attention entropy
- Average attention distance
- Maximum attention weight / concentration

Recorded values:

| Head | Entropy | Avg. Distance | Max Weight |
|---:|---:|---:|---:|
| 1 | 2.796 | 5.737 | 0.090 |
| 2 | 2.807 | 5.554 | 0.086 |
| 3 | 2.812 | 5.591 | 0.080 |
| 4 | 2.763 | 5.747 | 0.136 |
| 5 | 2.789 | 5.820 | 0.099 |
| 6 | 2.758 | 5.343 | 0.108 |
| 7 | 2.785 | 5.747 | 0.102 |
| 8 | 2.803 | 5.527 | 0.086 |

### Interpretation

The notebook explicitly concludes that, with such limited training, observed differences between heads should primarily be treated as effects of initialization and limited learning.

A fully trained WMT14 experiment would be required before making strong claims about specialized syntactic or semantic attention heads.

---

# 20. Gradient-Flow Analysis

The project performs a gradient-norm analysis across:

- Source embedding
- Encoder layers 1–6
- Decoder layers 1–6
- Final generator

A gradient visualization is generated from one training batch.

The purpose is not to claim that the experiment proves the absence of vanishing gradients, but to inspect how gradients propagate through the residual/LayerNorm Transformer stack.

Residual connections provide additive paths through the network, while LayerNorm helps maintain stable activation and gradient scales.

---

# 21. Complexity Analysis

The notebook reproduces the main computational comparison from the paper.

| Layer Type | Complexity / Layer | Sequential Operations | Maximum Path Length |
|---|---|---|---|
| Self-Attention | `O(n² · d)` | `O(1)` | `O(1)` |
| Recurrent | `O(n · d²)` | `O(n)` | `O(n)` |
| Convolutional | `O(k · n · d²)` | `O(1)` | `O(log_k(n))` |
| Restricted Self-Attention | `O(r · n · d)` | `O(1)` | `O(n/r)` |

### Main implication

Self-attention has quadratic sequence-length computation, but it reduces the number of sequential operations required to connect distant tokens.

For a sequence of length `n`:

```text
RNN:
Sequential operations = O(n)

Self-attention:
Sequential operations = O(1)
```

This is a central reason the Transformer can be trained much more efficiently in parallel across sequence positions.

The trade-off is the `O(n² · d)` attention computation, which becomes increasingly expensive for very long sequences.

---

# 22. Research Dashboard

The notebook generates a final dashboard containing:

- Dataset sentence-length distribution
- Base vs. Big parameter counts
- Training/validation loss
- Demo BLEU vs. paper BLEU
- Sinusoidal positional encoding visualization
- Learning-rate schedule
- Gradient norms by layer
- Token throughput
- Encoder attention visualization

The dashboard provides a compact summary of the implementation and experiment.

---

# 23. Reproducibility

A random seed of:

```text
42
```

is applied to:

- Python
- NumPy
- PyTorch CPU
- PyTorch CUDA, when available

The notebook also enables deterministic cuDNN settings when CUDA is available.

Important: deterministic settings do not guarantee bit-for-bit identical results across every hardware/software combination.

---

# 24. Project Structure

The notebook is organized into the following major sections:

```text
1.  Paper Overview
2.  Paper Specification
3.  Environment
4.  Reproducibility
5.  Configuration
6.  Dataset Download
7.  Dataset Inspection
8.  BPE Tokenization
9.  Length-Aware Batching
10. Embeddings
11. Positional Encoding
12. Scaled Dot-Product Attention
13. Masking
14. Multi-Head Attention
15. Feed-Forward Network
16. Residual + LayerNorm
17. Encoder
18. Decoder
19. Complete Transformer
20. Parameter Verification
21. Activation Trace
22. Final Prediction Layer
23. Optimizer & LR Schedule
24. Label Smoothing
25. Training Loop
26. Training Visualization
27. Greedy & Beam Search
28. Beam Search Behaviour
29. BLEU Evaluation
30. Paper Comparison
31. Attention Explainability
32. Attention Head Analysis
33. Gradient-Flow Analysis
34. Complexity Analysis
35. Base vs. Big
36. Optional Ablations
37. Final Dashboard
38. Final Report & Limitations
```

---

# 25. Dependencies

The notebook uses the following main Python packages:

```text
Python
PyTorch
NumPy
Matplotlib
Hugging Face Datasets
Hugging Face Tokenizers
SacreBLEU
```

Typical installation command:

```bash
pip install torch numpy matplotlib datasets tokenizers sacrebleu
```

For GPU execution, install the PyTorch build appropriate for the target CUDA environment.

---

# 26. How to Run

## Option A — Demo / Development Run

The default configuration is designed for architectural verification and end-to-end execution.

Keep:

```python
CONFIG["FULL_DATASET"] = False
CONFIG["DEVELOPMENT_STEPS"] = 300
```

Then execute the notebook from top to bottom.

This mode:

- Uses a small WMT14 subset.
- Uses a 2K vocabulary.
- Trains for 300 steps.
- Can operate without a GPU.
- Produces the architecture, training curves, inference outputs, attention visualizations, and BLEU evaluation.

---

## Option B — Full Paper-Scale Experiment

To move toward a genuine reproduction of the paper's training regime:

```python
CONFIG["FULL_DATASET"] = True
CONFIG["DEVELOPMENT_STEPS"] = CONFIG["PAPER_STEPS"]
```

For the Base model:

```text
PAPER_STEPS = 100,000
```

A capable GPU environment and substantial disk/storage are required.

The original paper used:

```text
~4.5M EN-DE sentence pairs
100,000 training steps
8 × P100 GPUs
approximately 12 hours
```

A modern GPU may have different performance characteristics, so runtime should not be assumed to match the original paper.

For a closer reproduction, also use the paper-scale vocabulary configuration and reproduce checkpoint averaging.

---

# 27. What Was Successfully Reproduced

The following components were implemented and exercised:

- Transformer encoder-decoder architecture
- Token embeddings
- Sinusoidal positional encoding
- Scaled dot-product attention
- Multi-head attention
- Padding masks
- Causal decoder masks
- Position-wise feed-forward networks
- Residual connections
- Layer normalization
- Six-layer encoder
- Six-layer decoder
- Final vocabulary projection
- Tied embedding/output projection
- Adam optimizer configuration
- Noam learning-rate schedule
- Label smoothing
- Length-aware token batching
- Greedy decoding
- Beam search
- Attention extraction/visualization
- Parameter-count verification
- Gradient-flow inspection
- Complexity analysis
- Base-vs-Big parameter comparison
- BLEU evaluation

The implementation does **not** rely on `torch.nn.Transformer` as a black-box Transformer implementation.

---

# 28. What Was Not Fully Reproduced

A complete reproduction of the original paper's reported performance was **not** performed.

Specifically:

- Full 4.5M-pair WMT14 training was not used in the executed run.
- 100,000 Base-model training steps were not performed.
- The original 8×P100 hardware setup was not available.
- The 37K vocabulary was not used for the demo training run.
- The Big model was not trained.
- WMT14 EN-FR was not trained.
- Original checkpoint averaging was not reproduced.
- The paper's reported BLEU scores were therefore not expected or claimed.

These are limitations of the experimental environment and training budget, not hidden omissions.

---

# 29. Key Findings

### Finding 1 — Architecture is consistent with the paper

Using the paper-scale hyperparameters and vocabulary size, the implementation produced:

```text
63.1M parameters
```

versus the paper's approximately:

```text
65M parameters
```

This is within approximately **2.9%**.

---

### Finding 2 — The training pipeline learns on the development subset

Training loss:

```text
6.6634 → 5.2951
```

Validation loss:

```text
6.7029 → 5.3250
```

The decreasing losses demonstrate that the model and optimization pipeline are functioning on the selected WMT14 subset.

---

### Finding 3 — Demo BLEU is intentionally low

The final reported demo BLEU was:

```text
0.23
```

This is far below the paper's:

```text
27.3 BLEU — Transformer Base EN-DE
```

The gap is expected because the development run uses only 2,000 training examples and 300 steps rather than approximately 4.5M examples and 100,000 steps.

---

### Finding 4 — Beam search mechanics execute successfully

Both greedy and beam decoding run successfully, but the generated translations are highly repetitive.

This demonstrates functioning autoregressive decoding while also exposing the consequences of severe under-training.

---

### Finding 5 — Base and Big parameter counts are consistent

The implementation reports:

```text
Base: 63.1M
Big : 214.3M
```

which closely matches the paper's approximate:

```text
Base: 65M
Big : 213M
```

---

### Finding 6 — Attention-head specialization cannot be claimed from this run

The head-level statistics show measurable differences in entropy, distance, and concentration, but the notebook correctly avoids interpreting these as learned linguistic specialization because the model was only trained for a few hundred steps.

---

# 30. Limitations

This project should be cited as a **faithful implementation and experimental reproduction of the Transformer architecture**, not as an exact reproduction of the original paper's final translation benchmark.

The main limitations are:

1. **Limited data**
   - 2,000 training pairs were used in the executed run.

2. **Limited training**
   - Only 300 steps were performed.

3. **CPU execution**
   - No CUDA GPU was available.

4. **Small vocabulary**
   - The demo uses 2,000 BPE tokens instead of the paper-scale ~37K vocabulary.

5. **No checkpoint averaging**
   - The original training methodology included checkpoint averaging.

6. **Big model not trained**
   - Only parameter count was verified.

7. **EN-FR not trained**
   - The English-French experiment is outside the executed experiment.

8. **Attention interpretation**
   - The short training run is insufficient for strong linguistic claims.

9. **Exact reproducibility**
   - Different PyTorch, CUDA, GPU, tokenizer, and hardware environments can produce small numerical differences.

---

# 31. Recommended Next Experiment

For a stronger research-grade reproduction, the next run should:

1. Use a GPU runtime.
2. Set `FULL_DATASET=True`.
3. Use the paper-scale ~37K shared BPE vocabulary.
4. Train the Base model for 100,000 steps.
5. Use token-budget batching comparable to the paper.
6. Preserve the Adam and Noam hyperparameters.
7. Reproduce checkpoint averaging.
8. Evaluate using SacreBLEU on the appropriate WMT14 test set.
9. Repeat attention-head analysis after full training.
10. Compare greedy and beam-search decoding.
11. Record GPU type, memory, runtime, throughput, and checkpoint information.
12. Only then compare the resulting BLEU directly against the paper's reported benchmark.

---

# 32. Final Conclusion

This project provides a transparent and inspectable reproduction of the Transformer introduced in *Attention Is All You Need*.

The strongest evidence of successful reproduction is architectural:

- The encoder-decoder structure is implemented directly.
- The core attention equations are implemented explicitly.
- The six-layer Base configuration is reproduced.
- The paper's optimization settings are implemented.
- The implementation reaches approximately **63.1M parameters** with the paper-scale vocabulary, close to the reported **~65M**.
- The Big configuration reaches approximately **214.3M parameters**, close to the reported **~213M**.
- The complete training/inference/evaluation pipeline executes end-to-end.

The development experiment also demonstrates learning behavior, with training loss decreasing from **6.6634** to **5.2951** and validation loss decreasing from **6.7029** to **5.3250**.

However, the **0.23 BLEU** result is not a reproduction of the paper's benchmark. It is the expected outcome of a deliberately small development experiment using 2,000 training pairs and 300 CPU training steps. The notebook explicitly documents this distinction rather than presenting an artificially favorable result.

Overall, the project is best characterized as:

> **A faithful, from-scratch Transformer architecture reproduction with a transparent demo-scale WMT14 training experiment and explicit quantitative comparison against the original paper.**

---

## 33. Reference

**Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.**

**Attention Is All You Need.**

Advances in Neural Information Processing Systems (NeurIPS), 2017.

