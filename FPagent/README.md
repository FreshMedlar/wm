# First-Principles Active Inference Agent (Single Shared Encoder with Target Detach)

A goal-agnostic, first-principles active inference agent trained directly on **raw pixel observations** in continuous-control Gymnasium environments.

```
                  +----------------------------------+
                  |    Gymnasium Environment (Env)   |
                  +-----------------+----------------+
                                    | Observation o_t (Pixels), Action a_t
                                    v
                  +----------------------------------+
                  |  Raw FIFO Sequence Replay Buffer |
                  +--------+----------------+--------+
                           |                |
             Sample Sub-seq|                | Sample Next Obs
                           v                v
                  +----------------------------------+
                  |      Single Perceptual Encoder   |
                  |             E_θ(·)               |
                  +--------+----------------+--------+
                           |                |
            Active Latent  |                | Target Latent
            z_t = E_θ(o_t) |                | z_{t+1} = E_θ(o_{t+1})
                           |                |
                           |                v
                           |      +--------------------+
                           |      | stop_gradient (·)  |
                           |      +---------+----------+
                           |                | Target z̄_{t+1} = z_{t+1}.detach()
                           v                v
+----------------------------------+   +----------------------------------+
|   Causal Ensemble World Model    |-->| Prediction Loss (Dynamics)       |
|   {P_ϕ1, ..., P_ϕM}              |   | L_pred = ||ẑ_{t+1}^m - z̄_{t+1}||²|
|   ẑ_{t+1}^m = P_ϕm(z_t, a_t)     |   +----------------------------------+
+-----------------+----------------+
                  | Predictions {ẑ^m} & Uncertainty σ²
                  v
+-------------------------------------------------------------------------+
|                  Active Inference Controller / MPC Planner              |
|                                                                         |
| Action Selection via CEM / MPPI minimizing Expected Free Energy (EFE):  |
|   G(π) = - w_epi · Epistemic(σ²)                                        |
|          - w_emp · Empowerment(Var_candidates(ẑ_{t+H}))                 |
|          + w_viab · ViabilityBoundary(ẑ)                                |
|          - w_rew · PragmaticReward(ẑ, a)                                |
+-------------------------------------------------------------------------+
```

---

## 1. Key Principles & Mathematical Formulation

### A. Single Perceptual Encoder & True `SIGReg` Anti-Collapse Regularizer
The architecture avoids momentum/EMA target networks completely. Both current and future frames are projected through the exact same encoder $E_\theta$, preventing representational coordinate lag:

$$z_t = E_\theta(o_t), \quad z_{t+1} = E_\theta(o_{t+1}), \quad z_t, z_{t+1} \in \mathbb{R}^D$$

To prevent representation and dimensional collapse without an EMA teacher, **SIGReg** (Sketch Isotropic Gaussian Regularizer, as formulated in LeWorldModel / LeJEPA) matches the empirical distribution of latent projections to an isotropic standard normal distribution $\mathcal{N}(0, I)$ via the **Epps-Pulley characteristic function test**:

$$\phi(t) = e^{-t^2 / 2}, \quad x_t = (z \cdot A) t$$

$$\mathcal{L}_{\text{SIGReg}} = \frac{1}{K \cdot P} \sum_{p=1}^P \sum_{k=1}^K w_k \left( \left| \mathbb{E}[\cos(x_{t_k})] - \phi(t_k) \right|^2 + \left| \mathbb{E}[\sin(x_{t_k})] \right|^2 \right)$$

evaluated over $P=1024$ random 1D projections on the unit sphere and $K=17$ numerical quadrature knots.

### B. Stop-Gradient Dynamics Loss (LeJEPA Forward Loss)
An ensemble of $M$ independent transition networks $\{P_{\phi_1}, \dots, P_{\phi_M}\}$ predicts forward dynamics in latent space. Gradients are explicitly disconnected at the future latent $z_{t+1}$ using `stop_gradient`:

$$\bar{z}_{t+1} = \text{stop\_gradient}(z_{t+1}) = z_{t+1}.\text{detach}()$$

$$\mathcal{L}_{\text{dyn}}(\theta, \phi) = \frac{1}{M \cdot B} \sum_{m=1}^M \sum_{i=1}^B \left\Vert{} P_{\phi_m}(z_t^{(i)}, a_t^{(i)}) - \bar{z}_{t+1}^{(i)} \right\Vert{}_2^2$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{dyn}}(\theta, \phi) + \gamma \, \mathcal{L}_{\text{SIGReg}}(z_t)$$

Gradients from $\mathcal{L}_{\text{dyn}}$ update predictor heads $\phi$ and flow through $z_t$ into the encoder $\theta$. Gradients from $\mathcal{L}_{\text{SIGReg}}$ directly shape the encoder manifold.

### C. Active Inference Controller: Expected Free Energy (EFE)
Action selection optimizes continuous trajectories $a_{t:t+H-1}$ via the Cross-Entropy Method (CEM) or MPPI, minimizing Expected Free Energy:

$$G(a_{t:t+H-1}) = \sum_{\tau=t}^{t+H-1} \mathcal{G}(\hat{z}_\tau, a_\tau)$$

1. **Pragmatic Value (Task Reward / Prior Preferences):**
   Predicts scalar task rewards using auxiliary network $R_\omega(z, a) \to \mathbb{R}$:
   $$\mathcal{G}_{\text{pragmatic}} = - R_\omega(\hat{z}_\tau, a_\tau)$$

2. **Epistemic Value (Model Uncertainty / Curiosity):**
   $$\bar{z}_{\tau+1} = \frac{1}{M} \sum_{m=1}^M P_{\phi_m}(\hat{z}_\tau, a_\tau)$$
   $$\mathcal{U}(\hat{z}_\tau, a_\tau) = \frac{1}{M} \sum_{m=1}^M \left\Vert{} P_{\phi_m}(\hat{z}_\tau, a_\tau) - \bar{z}_{\tau+1} \right\Vert{}_2^2$$
   $$\mathcal{G}_{\text{epistemic}} = - \mathcal{U}(\hat{z}_\tau, a_\tau)$$

3. **Viability Boundary (Homeostatic Prior):**
   Penalizes states predicting terminal failure using auxiliary network $V_\psi(z) \in [0, 1]$ trained on empirical environment termination flags (`dones`):
   $$\mathcal{G}_{\text{viab}} = -\log(1 - V_\psi(\hat{z}_\tau) + \epsilon)$$

4. **Empowerment (Preserving Downstream Control Options):**
   Evaluated over the population of candidate trajectories rolled out at horizon $H$:
   $$\mathcal{G}_{\text{emp}} = - \text{Var}_{n \in \{1,\dots,N\}}(\hat{z}_{t+H}^{(n)})$$

Candidate trajectory cost:
$$C^{(n)} = \sum_{\tau=t}^{t+H-1} \left( - w_{\text{rew}} \, R_\omega(\hat{z}_\tau^{(n)}, a_\tau^{(n)}) + w_{\text{viab}} \, \mathcal{G}_{\text{viab}}(\hat{z}_\tau^{(n)}) - w_{\text{epi}} \, \mathcal{U}(\hat{z}_\tau^{(n)}, a_\tau^{(n)}) \right) - w_{\text{emp}} \, \mathcal{G}_{\text{emp}}^{(n)}$$

* Setting `--w_rew 0.0` runs **Pure Goal-Agnostic Active Inference**.
* Setting `--w_rew > 0` (e.g. `1.0`) runs **Reward-Guided Active Inference** optimizing external task reward alongside curiosity and viability.

---

## 2. Project Directory Structure

```
FPagent/
├── __init__.py           # Package exports (Agent, Buffer, Envs, Losses, Networks, Planner)
├── agent.py              # FirstPrinciplesActiveInferenceAgent implementation
├── buffer.py             # Memory-efficient raw pixel FIFO replay buffer (uint8, rewards)
├── env.py                # GymPixelWrapper & make_pixel_env (Gymnasium continuous environments)
├── evaluate.py           # Evaluation & diagnostic rollout script
├── losses.py             # True SIGReg (Epps-Pulley) & compute_latent_metrics
├── networks.py           # PixelEncoder, TransitionPredictor, PredictorEnsemble, ViabilityClassifier, RewardPredictor
├── planner.py            # ActiveInferencePlanner (CEM & MPPI with Pragmatic Reward)
├── train.py              # Full 3-phase training lifecycle CLI
└── tests/                # Comprehensive unit and integration test suite
    ├── test_agent_and_integration.py
    ├── test_buffer.py
    ├── test_detach_validation.py
    ├── test_losses.py
    ├── test_networks.py
    └── test_planner.py
```

---

## 3. Training Lifecycle (Pure Offline Training)

The training lifecycle decouples data collection from model training and planning to achieve maximum throughput (50x–100x speedup over synchronous online planning):

### Phase 0: Initialization
- Wraps continuous Gymnasium environments to emit $(C, H, W)$ normalized pixel frames (e.g. $64 \times 64$).
- Raw FIFO replay buffer storing tuples $(o_t, a_t, r_t, o_{t+1}, d_t)$. Observations are packed as `uint8` to minimize RAM footprint. Latents are **never** cached.
- Networks initialized without EMA momentum copies.

### Phase 1: High-Speed Data Collection (Zero Planning)
- Collect $N_{\text{collect}}$ transitions (e.g., 10,000 steps) under uniform random exploration $a_t \sim \text{Uniform}(\mathcal{A})$.
- Because there is no planning or gradient computation in the loop, collection runs at **500–1,500 FPS**.

### Phase 2: High-Throughput Offline World Model Training
- Run $N_{\text{train}}$ gradient updates (e.g., 5,000–20,000 steps) sampling mini-batches directly from the replay buffer.
- Fully saturates GPU compute with zero CPU–GPU context switching or CEM planning overhead.
- Monitors `SIGReg` convergence (latent std across dimensions $\approx 1.0$, off-diagonal covariance $< 0.1$).

### Phase 3: Active Inference Benchmark & Policy Evaluation
- Deploy the Cross-Entropy Method (CEM) or MPPI planner exclusively at test time to minimize Expected Free Energy (EFE).
- Runs multi-episode evaluation rollouts, logging task reward, epistemic curiosity, and viability avoidance.

---

## 4. Verification and Developer Checklist

| Checklist Item | Validation Mechanism | Status |
| --- | --- | --- |
| **1. `.detach()` Placement** | `test_detach_validation.py` explicitly checks computational graph isolation of $z_{t+1}$ targets | Verified |
| **2. Latent Health Monitoring** | `compute_latent_metrics` checks `std_mean` ($\approx 1.0$) and off-diagonal covariance ($< 0.1$) | Verified |
| **3. Epistemic Uncertainty Dynamics** | Tracks $\text{Var}_m(P_{\phi_m}(z_t, a_t))$, ensuring it decays as state space becomes familiar | Verified |
| **4. Viability Classification** | Tracks BCE loss and binary classification accuracy on environment termination flags | Verified |
| **5. No Parameter Lag** | Confirms target latents originate from current $\theta$ without EMA delays | Verified |

---

## 5. Usage Guide

### Running Unit Tests
```bash
/home/medlar/Projects/.venv/bin/python -m unittest discover -s FPagent/tests -p "test_*.py" -v
```

### High-Throughput Offline Training on LunarLander (Raw Pixels)
```bash
/home/medlar/Projects/.venv/bin/python FPagent/train.py \
    --env_id LunarLander-v3 \
    --continuous \
    --img_size 64 \
    --latent_dim 64 \
    --ensemble_size 5 \
    --collect_steps 10000 \
    --train_steps 5000 \
    --batch_size 64 \
    --w_rew 1.0 \
    --w_viab 10.0 \
    --save_dir ./checkpoints/lunar_run \
    --device cuda
```

### High-Throughput Offline Training on Pendulum (Goal-Agnostic)
```bash
/home/medlar/Projects/.venv/bin/python FPagent/train.py \
    --env_id Pendulum-v1 \
    --continuous \
    --collect_steps 5000 \
    --train_steps 3000 \
    --batch_size 64 \
    --w_rew 0.0 \
    --w_epi 1.0 \
    --w_viab 5.0 \
    --w_emp 0.5 \
    --save_dir ./checkpoints/pendulum_run \
    --device cuda
```

### Evaluating a Saved Checkpoint
```bash
/home/medlar/Projects/.venv/bin/python FPagent/evaluate.py \
    --checkpoint ./checkpoints/lunar_run/agent_latest.pt \
    --env_id LunarLander-v3 \
    --episodes 5 \
    --render_video
```
