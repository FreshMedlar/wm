# Guided Latent Video Evolution Engine (LeV-JEPA + Upsampler + Flow DiT)

An action/dynamics-conditioned video generation and frame prediction architecture. Rather than generating video frames from pure Gaussian noise or hallucinating content unconditionally, this model frames video synthesis as an **evolution operator** over time:

$$\hat{x}_t = \mathcal{G}\Big(x_{t-1:t-k}, \; \hat{z}_t^{\text{jepa}}\Big)$$

* **Previous frames ($x_{t-1:t-k}$)** anchor physical scene identity: textures, lighting, static geometry, and persistent background.
* **LeV-JEPA semantic representations ($\hat{z}_t^{\text{jepa}}$)** provide the semantic vector field of what changed: object velocities, interactions, and disocclusions.
* **The Guided Latent Upsampler** bridges the spatial resolution gap ($14 \times 14 \to 32 \times 32$) to provide sharp structural guidance.
* **The Flow-Matching Evolution DiT ($v_\theta$)** solves the optimal-transport bridge to render crisp, physically coherent video rollouts.

---

## 1. System Architecture Diagram

```
 ┌───────────────────────────┐                ┌───────────────────────────┐
 │ Historical Frames (t-2,t-1)│                │ Target Video Frame Stream │
 └─────────────┬─────────────┘                └─────────────┬─────────────┘
               │                                            │
               ▼ (Frozen SD-VAE Encoder)                    ▼ (Frozen LeV-JEPA ViT-L/16)
 ┌───────────────────────────┐                ┌───────────────────────────┐
 │ Latents [z_{t-2}, z_{t-1}]│                │ Target Tokens ẑ_t^{jepa}  │
 │      [4, 32, 32] each     │                │     [196, 1024] (14x14)   │
 └─────────────┬─────────────┘                └─────────────┬─────────────┘
               │                                            │
               │               STAGE 1                      │
               │   ┌────────────────────────────────────────┴───┐
               └──►│       JEPA Latent Upsampler (3.6M)         │
                   │  - 2D Coordinate Fourier Meshgrid Injection│
                   │  - Spatial Transposed Convolution (14->32) │
                   │  - Texture Advection & Inpainting Decoder  │
                   └────────────────────┬───────────────────────┘
                                        │
                                        ▼ High-Resolution Guide
                                ┌───────────────────────────┐
                                │ z_t^{guide} [4, 32, 32]   │
                                └───────┬───────────────────┘
                                        │
                               STAGE 2  │
               ┌────────────────────────┴───────────────────────┐
               │                                                │
               ▼                                                ▼
 ┌───────────────────────────┐                ┌───────────────────────────┐
 │ Channel Concatenation In  │                │ Spatial Cross-Attention   │
 │ [z_s, z_{t-1}, z_{t-2},   │                │ Keys/Values: Guide Tokens │
 │  z_t^{guide}] (16 chans)  │                │ Queries: DiT Latent Grid  │
 └─────────────┬─────────────┘                └─────────────┬─────────────┘
               │                                            │
               └───────────────────────┬────────────────────┘
                                       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                   Latent Evolution DiT (v_θ, 7.2M)                     │
 │  - Flow-Matching Bridge: dz/ds = v_θ(z_s, s | z_ctx, z_guide, jepa)    │
 │  - 1-to-2 Step Heun / Euler ODE Integration                            │
 └─────────────────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼ Next Latent
                               ┌───────────────┐
                               │     ẑ_t       │
                               │  [4, 32, 32]  │
                               └───────┬───────┘
                                       │
                                       ▼ (Frozen SD-VAE Decoder)
                               ┌───────────────┐
                               │  Output Frame │
                               │     x̂_t       │
                               │ [3, 256, 256] │
                               └───────────────┘
```

---

## 2. Mathematical Formulation

### A. Stage 1: Guided High-Resolution JEPA Upsampler
LeV-JEPA compresses frames into a $14 \times 14$ grid of 1024-dimensional semantic patch tokens:
$$z_t^{\text{jepa}} \in \mathbb{R}^{196 \times 1024}$$
The Upsampler learns a deterministic mapping from the coarse semantic grid to the $32 \times 32$ latent geometry using $z_{t-1}$ as a texture bank:
$$\hat{z}_t^{\text{guide}} = z_{t-1} + \Delta z\Big(z_{t-1}, \; \text{CoordGrid}(x, y), \; z_t^{\text{jepa}}\Big)$$

**Objective:** Trained with composite $L_1$ and spatial gradient (finite-difference Sobel edge) loss to force crisp boundaries instead of blur:
$$\mathcal{L}_{\text{upsampler}} = \Vert \hat{z}_t^{\text{guide}} - z_t \Vert_1 + \lambda_{\text{edge}} \, \Vert \nabla \hat{z}_t^{\text{guide}} - \nabla z_t \Vert_1$$

### B. Stage 2: Flow-Matching Bridge ($v_\theta$)
Frame evolution from $s=0$ ($z_{t-1}$) to $s=1$ ($z_t$) is modeled via an optimal-transport velocity vector field:
$$z_s = (1 - s) \, z_{t-1} + s \, z_t, \quad s \in [0, 1]$$
$$u_t(z_s \mid z_{t-1}, z_t) = \frac{d z_s}{ds} = z_t - z_{t-1}$$

The Evolution DiT predicts this vector field:
$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{s \sim \mathcal{U}[0, 1]} \left[ \Vert v_\theta(z_s, s, [z_{t-1}, z_{t-2}], z_t^{\text{guide}}, z_t^{\text{jepa}}) - u_t \Vert_2^2 \right]$$

### C. Direct 1-Step Terminal Supervision
To anchor high-contrast edges and prevent the model from settling into trivial stationary solutions:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{guide}} \, \Vert (z_{t-1} + v_\theta(z_{t-1}, s=0, \dots)) - z_t \Vert_1$$

---

## 3. Repository Structure

```
video model/
├── dataset.py              # Synthetic bouncing shapes video generator (Option A)
├── cache_features.py       # Offline precomputation: caches VAE & LeV-JEPA latents to disk
├── jepa_upsampler.py       # Stage 1: Guided JEPA Latent Upsampler model definition
├── train_upsampler.py      # Stage 1: Training script with W&B logging & rollout GIFs
├── model.py                # Stage 2: LatentEvolutionDiT with Guide Cross-Attention & AdaLN
├── flow_solver.py          # Few-step Euler / Heun ODE bridge numerical integrator
├── fast_train_guided.py    # Stage 2: Main guided training loop using pretrained upsampler
└── outputs_guided/         # Checkpoints & evaluation side-by-side rollout comparison GIFs
```

---

## 4. Pipeline Execution & Commands

All scripts use the project virtual environment with ROCm/AMD GPU acceleration (`HSA_OVERRIDE_GFX_VERSION=10.3.0`).

### Step 1: Precompute Dataset Latents (~15 min)
Caches 500 clips of 16 frames to eliminate frozen ViT/VAE memory usage during DiT training:
```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 /home/medlar/Projects/.venv/bin/python "video model/cache_features.py" \
  --num-samples 500 \
  --seq-len 16
```

### Step 2: Train the JEPA Upsampler (~5-7 min)
Trains the 3.6M parameter upsampler to translate $(z_{t-1}, z_t^{\text{jepa}}) \to z_t^{\text{guide}}$:
```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 /home/medlar/Projects/.venv/bin/python "video model/train_upsampler.py" \
  --epochs 50 \
  --batch-size 16 \
  --lr 5e-4 \
  --hidden-dim 128 \
  --edge-weight 0.5 \
  --eval-every 10 \
  --project "video-evolution-model" \
  --run-name "jepa-upsampler-run-1"
```

### Step 3: Train the Guided Video Evolution DiT
Trains the Flow-Matching DiT guided by the frozen upsampler:
```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 /home/medlar/Projects/.venv/bin/python "video model/fast_train_guided.py" \
  --epochs 50 \
  --batch-size 8 \
  --lr 3e-4 \
  --hidden-size 256 \
  --depth 4 \
  --num-heads 4 \
  --guide-weight 0.2 \
  --eval-every 10 \
  --project "video-evolution-model" \
  --run-name "guided-dit-evolution-run-1"
```

---

## 5. Performance & Convergence Highlights

* **Throughput:** ~30–40 batches/sec during training due to offline feature caching.
* **Loss Reduction:** The guided architecture achieves **>50% lower flow MSE** compared to the unguided baseline, converging in 8 epochs to levels that required ~80+ epochs previously.
* **Sharpness:** Direct $L_1$ + spatial gradient loss eliminates Gaussian motion blur on moving objects while leaving the background untouched.
