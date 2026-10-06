# First-Principles Active Inference Agent (Single Shared Encoder with Target Detach)

A goal-agnostic and language-conditioned, first-principles active inference agent trained directly on **raw pixel observations** in continuous-control Gymnasium and 3D MiniWorld environments.

```
                  +----------------------------------------------+
                  |  Gymnasium / 3D MiniWorld (Pixels + Text)    |
                  +----------------------+-----------------------+
                                         | Observation o_t (Pixels), Action a_t,
                                         | Context Prompt c_t ("reach red object")
                                         v
                  +----------------------------------------------+
                  |      Raw FIFO Sequence Replay Buffer         |
                  +--------+----------------+-----------+--------+
                           |                |           |
             Sample Sub-seq|                | Next Obs  | Text Prompt
                           v                v           v
                  +------------------+          +------------------------+
                  |  Pixel Encoder   |          | Language Context Enc.  |
                  |     E_θ(·)       |          | MiniLM Transformer + W |
                  +--------+---------+          +-----------+------------+
                           |   Target Latent                | Context c_emb
            Active Latent  |   z_{t+1} = E_θ(o_{t+1})       |
            z_t = E_θ(o_t) |                |               |
                           |                v               |
                           |      +--------------------+    |
                           |      | stop_gradient (·)  |    |
                           |      +---------+----------+    |
                           |                | Target z̄_{t+1}|
                           v                v               |
+----------------------------------+   +--------------------+-------------+
|   Causal Ensemble World Model    |-->| Prediction Loss (Dynamics)       |
|   {P_ϕ1, ..., P_ϕM}              |   | L_pred = ||ẑ_{t+1}^m - z̄_{t+1}||²|
|   ẑ_{t+1}^m = P_ϕm(z_t, a_t, c)  |   +----------------------------------+
+-----------------+----------------+
                  | Predictions {ẑ^m} & Uncertainty σ²
                  v
+-------------------------------------------------------------------------+
|                  Active Inference Controller / MPC Planner              |
|                                                                         |
| Action Selection via CEM / MPPI minimizing Expected Free Energy (EFE):  |
|   G(π) = - w_epi · Epistemic(σ²)                                        |
|          - w_emp · Empowerment(Var_candidates(ẑ_{t+H}))                 |
|          + w_viab · ViabilityBoundary(ẑ, c)                             |
|          - w_rew · PragmaticReward(ẑ, a, c)                             |
+-------------------------------------------------------------------------+
```

---

## 1. Key Principles & Mathematical Formulation

### A. Single Perceptual Encoder & True `SIGReg` Anti-Collapse Regularizer
The architecture avoids momentum/EMA target networks completely. Both current and future frames are projected through the exact same encoder $E_\theta$, preventing representational coordinate lag:

$$z_t = E_\theta(o_t), \quad z_{t+1} = E_\theta(o_{t+1}), \quad z_t, z_{t+1} \in \mathbb{R}^D$$

To prevent representation and dimensional collapse without an EMA teacher, **SIGReg** (Sketch Isotropic Gaussian Regularizer, as formulated in LeWorldModel / LeJEPA) matches the empirical distribution of latent projections to an isotropic standard normal distribution $\mathcal{N}(0, I)$ via the **Epps-Pulley characteristic function test**:

$$\phi(t) = e^{-t^2 / 2}, \quad x_t = (z \cdot A) t$$

$$\mathcal{L}_{\text{SIGReg}} = \frac{1}{K \cdot P} \sum_{p=1}^P \sum_{k=1}^K w_k \left| \mathbb{E}[\cos(x_{t_k})] - \phi(t_k) \right|^2 + \left| \mathbb{E}[\sin(x_{t_k})] \right|^2$$

evaluated over $P=1024$ random 1D projections on the unit sphere and $K=17$ numerical quadrature knots.

### B. Stop-Gradient Dynamics Loss (LeJEPA Forward Loss)
An ensemble of $M$ independent transition networks $\{P_{\phi_1}, \dots, P_{\phi_M}\}$ predicts forward dynamics in latent space, optionally conditioned on language context $c$:

$$\bar{z}_{t+1} = \text{stop\_gradient}(z_{t+1}) = z_{t+1}.\text{detach}()$$

$$\mathcal{L}_{\text{dyn}}(\theta, \phi) = \frac{1}{M \cdot B} \sum_{m=1}^M \sum_{i=1}^B \left\Vert{} P_{\phi_m}(z_t^{(i)}, a_t^{(i)}, c^{(i)}) - \bar{z}_{t+1}^{(i)} \right\Vert{}_2^2$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{dyn}}(\theta, \phi) + \gamma \, \mathcal{L}_{\text{SIGReg}}(z_t)$$

Gradients from $\mathcal{L}_{\text{dyn}}$ update predictor heads $\phi$ and flow through $z_t$ into the encoder $\theta$. Gradients from $\mathcal{L}_{\text{SIGReg}}$ directly shape the encoder manifold.

### C. Active Inference Controller: Expected Free Energy (EFE)
Action selection optimizes continuous trajectories $a_{t:t+H-1}$ via the Cross-Entropy Method (CEM) or MPPI, minimizing Expected Free Energy:

$$G(a_{t:t+H-1}) = \sum_{\tau=t}^{t+H-1} \mathcal{G}(\hat{z}_\tau, a_\tau, c)$$

1. **Pragmatic Value (Task Reward / Prior Preferences):**
   Predicts scalar task rewards using auxiliary network $R_\omega(z, a, c) \to \mathbb{R}$:
   $$\mathcal{G}_{\text{pragmatic}} = - R_\omega(\hat{z}_\tau, a_\tau, c)$$

2. **Epistemic Value (Model Uncertainty / Curiosity):**
   $$\bar{z}_{\tau+1} = \frac{1}{M} \sum_{m=1}^M P_{\phi_m}(\hat{z}_\tau, a_\tau, c)$$
   $$\mathcal{U}(\hat{z}_\tau, a_\tau) = \frac{1}{M} \sum_{m=1}^M \left\Vert{} P_{\phi_m}(\hat{z}_\tau, a_\tau, c) - \bar{z}_{\tau+1} \right\Vert{}_2^2$$
   $$\mathcal{G}_{\text{epistemic}} = - \mathcal{U}(\hat{z}_\tau, a_\tau)$$

3. **Viability Boundary (Homeostatic Prior):**
   Penalizes states predicting terminal failure using auxiliary network $V_\psi(z, c) \in [0, 1]$:
   $$\mathcal{G}_{\text{viab}} = -\log(1 - V_\psi(\hat{z}_\tau, c) + \epsilon)$$

4. **Empowerment (Preserving Downstream Control Options):**
   Evaluated over the population of candidate trajectories rolled out at horizon $H$:
   $$\mathcal{G}_{\text{emp}} = - \text{Var}_{n \in \{1,\dots,N\}}(\hat{z}_{t+H}^{(n)})$$

---

## 2. Project Directory Structure

```
firstPAgent/
├── __init__.py           # Package exports (Agent, Buffer, Envs, Losses, Networks, Planner)
├── agent.py              # FirstPrinciplesActiveInferenceAgent implementation
├── buffer.py             # Memory-efficient raw pixel FIFO replay buffer (uint8, rewards, context)
├── env.py                # GymPixelWrapper & make_pixel_env (Gymnasium continuous environments)
├── miniworld_env.py      # MiniWorldLanguageWrapper (3D continuous navigation + procedural text)
├── language.py           # LanguageContextEncoder (Frozen MiniLM Transformer + adapter projection)
├── evaluate.py           # Evaluation & diagnostic rollout script
├── losses.py             # True SIGReg (Epps-Pulley) & compute_latent_metrics
├── networks.py           # PixelEncoder, TransitionPredictor, PredictorEnsemble, ViabilityClassifier, RewardPredictor
├── planner.py            # ActiveInferencePlanner (CEM & MPPI with context support)
├── train.py              # Full 3-phase training lifecycle CLI (supports Gymnasium & MiniWorld 3D)
└── tests/                # Comprehensive unit and integration test suite
    ├── test_agent_and_integration.py
    ├── test_buffer.py
    ├── test_detach_validation.py
    ├── test_losses.py
    ├── test_networks.py
    └── test_planner.py
```

---

## 3. Training Lifecycle

The training lifecycle strictly follows three distinct phases:

### Phase 0: Initialization
- Wraps the continuous environment to emit $(C, H, W)$ normalized pixel frames (e.g. $64 \times 64$).
- Raw FIFO replay buffer storing tuples $(o_t, a_t, r_t, o_{t+1}, d_t, [c_t])$. Observations are packed as `uint8` to minimize RAM. Latents are **never** cached.
- Optional Language Encoder initialized with a frozen Transformer backbone (`sentence-transformers/all-MiniLM-L6-v2`).

### Phase 1: Dynamics Warm-up (Random Exploration)
- Collect $N_{\text{warmup}}$ steps under uniform random exploration $a_t \sim \text{Uniform}(\mathcal{A})$.
- Perform gradient updates on random mini-batches ($B \ge 64$).
- Verify that `SIGReg` converges and latent std across dimensions reaches $\approx 1.0$, with off-diagonal covariance $< 0.1$.

### Phase 2: Active Inference Online Control Loop
- For each episode:
  1. Encode current frame $z_t = E_\theta(o_t)$ and optional language context $c_t = \text{LangEncoder}(\text{instruction})$.
  2. Plan action sequences via CEM/MPPI minimizing Expected Free Energy (EFE).
  3. Execute $a_t = \mu_0$, observe next frame $o_{t+1}$ and done flag.
  4. Append raw transition to replay buffer.
  5. Continuous online learning: sample mini-batch of size $B=64$, execute `train_step`.

---

## 4. Usage Guide

### Running Unit Tests
```bash
/home/medlar/Projects/.venv/bin/python -m unittest discover -s firstPAgent/tests -p "test_*.py" -v
```

### Training on 3D MiniWorld with Language Instructions
```bash
/home/medlar/Projects/.venv/bin/python firstPAgent/train.py \
    --env_id MiniWorld-PickupObjects-v0 \
    --warmup_steps 2000 \
    --warmup_train_steps 1500 \
    --episodes 50 \
    --max_steps_per_episode 150 \
    --horizon 10 \
    --num_candidates 128 \
    --num_elites 24 \
    --num_iterations 3 \
    --batch_size 64 \
    --save_dir ./checkpoints/miniworld_run \
    --device cuda
```

### Training on Continuous LunarLander (Raw Pixels)
```bash
/home/medlar/Projects/.venv/bin/python firstPAgent/train.py \
    --env_id LunarLander-v3 \
    --continuous \
    --img_size 64 \
    --latent_dim 64 \
    --ensemble_size 5 \
    --warmup_steps 2000 \
    --warmup_train_steps 1500 \
    --episodes 50 \
    --horizon 12 \
    --num_candidates 256 \
    --num_elites 32 \
    --w_rew 1.0 \
    --w_viab 10.0 \
    --save_dir ./checkpoints/lunar_run \
    --device cuda
```

### Training on Pendulum (Goal-Agnostic Active Inference)
```bash
/home/medlar/Projects/.venv/bin/python firstPAgent/train.py \
    --env_id Pendulum-v1 \
    --continuous \
    --w_rew 0.0 \
    --w_epi 1.0 \
    --w_viab 5.0 \
    --w_emp 0.5 \
    --save_dir ./checkpoints/pendulum_curiosity \
    --device cuda
```
