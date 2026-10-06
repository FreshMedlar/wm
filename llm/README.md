# Hierarchical LeJEPA Language Model

Implementation of a Two-Level Hierarchical Language Model where the **LeJEPA Transformer acts as a continuous prior/planner** and a lightweight **decoder acts as an autoregressive generator conditioned on that plan**:

$$P(x_{t+1} \mid x_{\le t}, \hat{z}_{t+1})$$

---

## 1. Theoretical Architecture

### 1.1 How Conditioning on Past Tokens Resolves the "Conditional Mean" Collapse
In pure continuous latent language modeling, predicting future latents $\hat{z}_{t+1}$ often collapses to the blurry mean of all plausible futures (e.g., an ungrounded centroid of "red", "blue", and "green"). Conditioning the decoder on both $x_{\le t}$ and $\hat{z}_{t+1}$ resolves this:
1. **Information Routing:** The past discrete tokens $x_{\le t}$ carry the exact syntactic history, grammatical constraints, and entity names.
2. **Disambiguation:** The continuous latent $\hat{z}_{t+1}$ acts as a semantic bias or soft guidance vector. The decoder's cross-entropy loss over the discrete vocabulary allows it to form a sharp multimodal distribution over actual words.
3. **Exact Syntax Handling:** Capitalization, punctuation, and local grammar rules do not need to be represented precisely in the continuous LeJEPA latent space; the autoregressive decoder handles them natively.

### 1.2 Parameter Budget Split (~90% LeJEPA / ~10% Decoder)
To address the information bypass dilemma (posterior collapse) without complicated kl-divergence annealing:
- **LeJEPA Planner (Encoder + Predictor):** Carries ~90–95% of model capacity (e.g., 7 causal encoder layers + 2 latent predictor layers).
- **Conditional Decoder:** Kept deliberately shallow (1 layer, ~5–10% of total parameters). Because the decoder is constrained in capacity, it must rely on the continuous representations provided by the planner.

### 1.3 Training Scheme A: Two-Stage (Strict JEPA)
- **Stage 1 (LeJEPA Pretraining):** The LeJEPA predictor is trained strictly with $L_2$ prediction loss and the Sketch Isotropic Gaussian Regularizer (SIGReg).
  $$\mathcal{L}_{\text{Stage 1}} = \|\hat{z}_{t+1} - \text{stop\_grad}(z_{t+1})\|_2^2 + \lambda \cdot \text{SIGReg}(z)$$
  This guarantees that the continuous latent space remains isotropic and non-degenerate without mode collapse.
- **Stage 2 (Conditional Decoder Training):** The LeJEPA planner is frozen (or stop-gradient on $\hat{z}_{t+1}$). The decoder is trained with next-token Cross-Entropy loss:
  $$\mathcal{L}_{\text{Stage 2}} = \text{CrossEntropy}(\text{Decoder}(x_{\le t}, \hat{z}_{t+1}), x_{t+1})$$

### 1.4 Per-Token Latents
Rather than multi-token or chunk latents, every token step $t$ produces a continuous latent $z_t$, and predicts the next token's continuous latent $\hat{z}_{t+1}$.

---

## 2. Directory Structure

```
llm/
├── modules.py      # Transformer building blocks, RoPE positional embedding, SIGReg regularizer, AdaLN conditioning
├── models.py       # LeJEPAEncoder, LeJEPAPredictor, ConditionalDecoder, LeJEPALanguageModel, StandardTransformerLM
├── dataset.py      # MemmapTextDataset for zero-RAM streaming of wiki_train.bin and wiki_val.bin
├── train.py        # Comprehensive trainer with continuous JSON metrics logging and auto-plot generation
├── compare.py      # Direct comparative evaluation benchmark with side-by-side curves
├── plot_metrics.py # Standalone plotting utility for custom smoothing, titles, and formats
└── README.md       # Documentation and guide
```

---

## 3. Quick Start & CLI Usage

Always run using the project virtual environment:
```bash
source /home/medlar/Projects/.venv/bin/activate
```

### 3.1 Lightning Fast Sanity Check (10 steps)
Verify the complete pipeline in under 5 seconds:
```bash
/home/medlar/Projects/.venv/bin/python -m llm.train --quick_test
```

### 3.2 Scheme A Two-Stage Training (Custom Hyperparameters for Test Runs)
Run a custom test run before the big run:
```bash
/home/medlar/Projects/.venv/bin/python -m llm.train \
    --stage all \
    --batch_size 16 \
    --seq_len 256 \
    --stage1_steps 100 \
    --stage2_steps 100 \
    --d_model 256 \
    --d_latent 128 \
    --sigreg_weight 0.1 \
    --save_dir checkpoints_llm
```

### 3.3 Standalone Stages
**Stage 1 Only (Train Continuous Planner):**
```bash
/home/medlar/Projects/.venv/bin/python -m llm.train \
    --stage 1 \
    --max_steps 500 \
    --save_dir checkpoints_llm
```

**Stage 2 Only (Train Conditional Decoder with Pretrained Planner):**
```bash
/home/medlar/Projects/.venv/bin/python -m llm.train \
    --stage 2 \
    --checkpoint_lejepa checkpoints_llm/lejepa_stage1.pt \
    --max_steps 500
```

### 3.4 Baseline Standard Transformer Comparison
Train the standard autoregressive baseline transformer (matching total layers):
```bash
/home/medlar/Projects/.venv/bin/python -m llm.train \
    --model baseline \
    --baseline_layers 10 \
    --max_steps 200
```

### 3.5 Head-to-Head Comparative Benchmark
Compare both models side-by-side on WikiText-103:
```bash
/home/medlar/Projects/.venv/bin/python -m llm.compare \
    --steps 50 \
    --batch_size 8 \
    --seq_len 128 \
    --prompt "The history of science demonstrates that"
```

---

## 4. Key Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `wiki_train.bin` | Path to binary tokenized training set |
| `--val_data_path` | `wiki_val.bin` | Path to binary tokenized validation set |
| `--model` | `lejepa` | Model type: `lejepa` or `baseline` |
| `--stage` | `all` | Training stage: `all`, `1`, `2`, or `baseline` |
| `--d_model` | `256` | Transformer hidden dimension |
| `--d_latent` | `128` | LeJEPA continuous latent space dimension |
| `--lejepa_encoder_layers` | `7` | Causal transformer layers in planner encoder |
| `--lejepa_predictor_layers`| `2` | Causal transformer layers in planner predictor |
| `--decoder_layers` | `1` | Layers in conditional decoder (~5–10% budget) |
| `--baseline_layers` | `10` | Layers in standard baseline transformer |
| `--sigreg_weight` | `0.1` | Weight $\lambda$ for isotropic Gaussian regularizer |
| `--seq_len` | `256` | Token context length |
| `--batch_size` | `16` | Batch size |
| `--max_steps` | `0` | Max steps to train (`0` = full epoch) |
| `--quick_test` | `False` | 10-step instant test run flag |
| `--save_dir` | `checkpoints_llm` | Directory where checkpoints, JSON metrics, and plots are stored |

---

## 5. Metrics Logging & Plotting

During any training run (short or multi-hour):
1. **Continuous Real-Time Logging:** Step losses, validation scores, and perplexities are automatically logged and flushed to `<save_dir>/training_metrics.json` at every step and evaluation point. If you stop or restart a run, your data is preserved.
2. **Auto-Generated Curves:** At the end of the run, a 4-panel publication-ready chart is automatically generated and saved to `<save_dir>/training_curves.png`:
   - Stage 1: $L_2$ prediction loss curve
   - Stage 1: SIGReg isotropic regularizer convergence curve
   - Stage 2: Training vs. Validation Cross-Entropy Loss
   - Stage 2: Validation Perplexity (PPL)
3. **Re-Plot Anytime:** You can re-render the curves with custom smoothing or styling at any time without retraining:
   ```bash
   /home/medlar/Projects/.venv/bin/python -m llm.plot_metrics \
       --log_file checkpoints_llm/training_metrics.json \
       --smooth 0.7 \
       --save_path my_custom_curves.png
   ```

