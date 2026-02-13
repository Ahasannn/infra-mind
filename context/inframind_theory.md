# InfraMind: Infrastructure-Aware Multi-Agent Orchestration

## 1. Problem Formulation

Existing multi-agent LLM routing approaches select collaboration topologies, roles, and models based solely on task semantics, ignoring the infrastructure state of the serving layer. This infrastructure-oblivious routing causes: (1) **load imbalance** — static model preferences create deep queues on preferred models while others sit idle; (2) **avoidable queuing latency** — requests wait on congested models when idle alternatives exist; (3) **wasted compute** — workflows assigned infeasible topologies exceed their latency budget, discarding all generated tokens; and (4) **missed quality opportunities** — idle capacity at low load is never invested in richer reasoning.

InfraMind addresses all four problems by making infrastructure state observable at every decision level. Given a query $q$, a model pool $\mathcal{M}$, and a per-query latency budget $\beta$, InfraMind learns to:
1. **Select a collaboration topology** $\tau \in \mathcal{T}$ and assign roles $\mathcal{R}$ — budget-aware structural planning.
2. **Route each role to an LLM** $m \in \mathcal{M}$ with a prompting strategy $\sigma \in \Sigma$ — infrastructure-aware resource allocation.
3. **Schedule requests** via Earliest-Deadline-First (EDF) priority at the serving layer — deadline-aware scheduling.

We formulate this as a Constrained Markov Decision Process (CMDP):

$$\pi^* = \arg\max_{\pi} \; \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right] \quad \text{s.t.} \quad \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t C(s_t, a_t) \right] \leq \beta$$

where $R$ is the task quality reward and $C$ is the latency cost.

---

## 2. Architecture Overview

<!--
ARCHITECTURE DIAGRAM (Figure X)

                    ┌─────────────────────────────────────────┐
                    │            PLANNER (once, t=0)           │
  Query q ────▶ e_q │                                         │
                    │  ┌──────┐    ┌───────────────────────┐  │
  Budget β ────────▶│  │ BCFM │──▶│  MAS Planner Pipeline  │  │
                    │  │(FiLM)│    │      (black box)       │  │
                    │  └──────┘    └───────────┬───────────┘  │
                    │   ★ ours                 │              │
                    └──────────────────────────┼──────────────┘
                                               │
                              Topology τ, Roles (r₁...rₖ)
                                               │
                    ┌──────────────────────────▼──────────────┐
                    │      EXECUTOR (per role rₖ) ★ ours      │
                    │                                         │
                    │  State: [e_q ‖ e_rₖ ‖ b_rem ‖ z_sys]  │
                    │                                         │
                    │  z_sys = predicted latency for every    │
   Metrics ────────▶│  (model, strategy) + queue depths       │
   Watcher          │                                         │
                    │  ┌─────────┐ ┌─────────┐ ┌──────────┐  │
                    │  │Model Hd │ │Strat Hd │ │Value Hd  │  │
                    │  └────┬────┘ └────┬────┘ └──────────┘  │
                    └───────┼───────────┼─────────────────────┘
                            │           │
                     (model m, strategy σ)
                            │
                    ┌───────▼─────────────────────────────────┐
                    │      vLLM SERVING LAYER ★ ours          │
                    │   EDF priority = ⌊t_arrival + β⌋        │
                    │                                         │
                    │  ┌───────┐ ┌───────┐      ┌───────┐    │
                    │  │  m₁   │ │  m₂   │ ···  │  mₘ   │    │
                    │  └───────┘ └───────┘      └───────┘    │
                    └─────────────────────────────────────────┘

★ = InfraMind contributions
-->

InfraMind uses a **two-level hierarchy** separating structural decisions from resource allocation:

| Level | Observes | Decides | Operates |
|-------|----------|---------|----------|
| **Planner** | Query embedding + budget $\beta$ | Topology $\tau$, agent count $K$, roles $\mathcal{R}$ | Once at $t=0$ |
| **Executor** | Query + role + remaining budget + system state | Model $m$, strategy $\sigma$ per role | Per agent node |
| **EDF Scheduler** | Deadline $D_i = t_i^{\text{arr}} + \beta_i$ | Queue ordering | Per request at serving layer |

### 2.1 Planner with Budget-Conditioned Feature Modulation

The Planner uses an existing MAS routing pipeline \cite{masrouter} to select topology $\tau$, agent count $K$, and roles $\mathcal{R}$ from a query embedding. We treat this pipeline as a black box — our contribution is making it **budget-aware**.

Existing MAS planners operate on task semantics alone and cannot distinguish a 10-second budget from a 120-second budget for the same query, leading to structurally infeasible plans (e.g., assigning a multi-round debate when only seconds remain). We introduce **Budget-Conditioned Feature Modulation (BCFM)**, a FiLM layer that modulates the query embedding *before* it enters the planner:

$$\tilde{\mathbf{e}}_q = \gamma(\beta) \odot \mathbf{e}_q + \beta(\beta)$$

where $\gamma, \beta \in \mathbb{R}^{d_q}$ are scale and shift parameters produced by a budget-encoding MLP, with $\gamma$ initialized near identity ($\gamma \approx \mathbf{1}$). This design preserves the embedding dimensionality, enabling direct weight transfer from a pretrained MAS checkpoint without any architectural change to the downstream modules. Under tight budgets, BCFM shifts the embedding toward latent regions associated with simpler topologies; under generous budgets, it enables richer collaboration.

### 2.2 Executor

For each role $r_k$, the Executor selects a model and prompting strategy based on a state that fuses task semantics with real-time infrastructure conditions:

$$s_{\text{exec}}^{(k)} = \left[ \mathbf{e}_q \,\|\, \mathbf{e}_{r_k} \,\|\, b_{\text{rem}} \,\|\, \mathbf{z}_{\text{sys}} \right]$$

where $b_{\text{rem}}$ is the remaining budget and $\mathbf{z}_{\text{sys}}$ is the system state vector.

**Action-Conditional Latency Map.** Rather than providing a single latency estimate per model, the system state enumerates the predicted latency for every (model, strategy) pair under current conditions:

$$\mathbf{z}_{\text{sys}} = \bigoplus_{m \in \mathcal{M}} \left[ \hat{L}_{m,\sigma_1} \,\|\, \cdots \,\|\, \hat{L}_{m,\sigma_{|\Sigma|}} \,\|\, n_m^{\text{run}} \,\|\, n_m^{\text{wait}} \right]$$

This gives the executor a complete latency "price list" of all possible actions, enabling real-time quality-latency tradeoffs: routing to underutilized models under high load, and investing in richer reasoning when headroom exists.

**Architecture:** A shared MLP backbone with factorized heads for model selection, strategy selection, and value estimation:

$$\pi_m, \; \pi_\sigma = \text{Softmax}(\text{Head}_m(\mathbf{h})), \; \text{Softmax}(\text{Head}_\sigma(\mathbf{h})), \quad V(s) = \text{Head}_V(\mathbf{h})$$

### 2.3 EDF Scheduling at the Serving Layer

Each query's deadline $D_i = t_i^{\text{arr}} + \beta_i$ is propagated to vLLM as an EDF priority. Requests closer to their deadline are served first, reducing tail-latency violations. This creates a cooperative two-level scheduling hierarchy: the CMDP executor decides *which* model to use; EDF scheduling decides *when* the request is processed.

---

## 3. Latency Estimation

The executor's action-conditional latency map is constructed from three neural predictors:

$$\hat{L}_{m,\sigma} = \hat{T}_{m,\sigma}^{\text{TTFT}} + \hat{T}_{m,\sigma}^{\text{TPOT}} \cdot \hat{N}_{m,\sigma}^{\text{out}}$$

| Predictor | Input Features | Output |
|-----------|---------------|--------|
| **TTFT** (time-to-first-token) | Input token count, $n_m^{\text{run}}$, $n_m^{\text{wait}}$ | Prefill + queuing delay |
| **TPOT** (time-per-output-token) | $n_m^{\text{run}}$, $n_m^{\text{wait}}$ | Decode-phase speed |
| **Output Length** | Query embedding $\mathbf{e}_q$, strategy embedding $\mathbf{e}_\sigma$, model embedding $\mathbf{e}_m$ | Predicted output tokens |

All predictors use MLPs with Softplus output activations to ensure non-negative predictions. Strategy conditioning is critical because prompting strategies (Flash, Concise, DeepThink) produce dramatically different output lengths, directly determining generation-phase latency.

---

## 4. Training

### 4.1 Reward and Cost

**Quality reward:** Task-specific binary correctness (e.g., $\mathbb{1}[\text{answer correct}]$ for math, $\mathbb{1}[\text{pass all tests}]$ for code).

**Proportional latency cost:** Normalized by budget to provide continuous gradient signal:

$$C_{\text{plan}} = \frac{C_{\text{workflow}}}{\beta}, \qquad C_{\text{exec}}^{(k)} = \frac{L_k^{\text{observed}}}{b_{\text{rem}}^{(k)}}$$

The proportional formulation ensures gradient signal whether under or over budget, unlike a one-sided penalty that produces zero gradient when the constraint is satisfied.

### 4.2 Lagrangian Relaxation

We convert the constrained optimization to an unconstrained problem:

$$\tilde{R} = R_{\text{quality}} - \lambda \cdot \frac{C}{\beta}$$

The Lagrange multiplier $\lambda$ is updated via **signed dual ascent**:

$$\lambda \leftarrow \max\left(0, \; \lambda + \eta_\lambda \left( \frac{\bar{C}_{\text{workflow}}}{\bar{\beta}} - 1 \right) \right)$$

The signed update is essential: when average latency is below budget, $\lambda$ *decreases*, relaxing latency pressure and allowing the policy to explore higher-quality configurations. An unsigned (monotonic) update causes $\lambda$ to grow without bound, collapsing the policy to the cheapest action regardless of quality.

### 4.3 Policy Gradient

**Planner:** REINFORCE with EMA baseline for variance reduction:

$$\nabla_{\theta} J_{\text{plan}} = \frac{1}{N} \sum_{i} \nabla_{\theta} \log \pi_{\text{plan}}(a^{(i)} | s^{(i)}) \cdot \left(\tilde{R}^{(i)} - \bar{G}\right) + \alpha_H \nabla \mathcal{H}[\pi_{\text{plan}}]$$

**Executor:** Actor-critic with learned value baseline:

$$\mathcal{L}_{\text{exec}} = \underbrace{-\frac{1}{K}\sum_k \log \pi_{\text{exec}} \cdot \hat{A}^{(k)}}_{\text{actor}} + \underbrace{c_V \cdot \frac{1}{K}\sum_k \left(V(s^{(k)}) - \tilde{R}^{(k)}\right)^2}_{\text{critic}} - \underbrace{\alpha_H \cdot \mathcal{H}[\pi_{\text{exec}}]}_{\text{entropy}}$$

Both policies use entropy regularization to maintain exploration and gradient clipping ($\|\nabla\|_2 \leq 1.0$).

### 4.4 Transfer Learning

The planner is initialized from a pretrained MAS checkpoint, transferring existing task-routing knowledge. Because BCFM preserves the embedding dimensionality and initializes near-identity, the planner behaves identically to the baseline at initialization. End-to-end fine-tuning then adapts the combined system to InfraMind's budget-aware reward.

### 4.5 Workflow Latency

The workflow executes in topological waves. Nodes within a wave run in parallel; waves execute sequentially:

$$C_{\text{workflow}} = \sum_{w=1}^{W} \max_{r_k \in \text{wave}_w} L_k$$

---

## 5. Notation Reference

| Symbol | Description |
|--------|-------------|
| $q$, $\mathbf{e}_q$ | Input query, query embedding |
| $\tilde{\mathbf{e}}_q$ | Budget-conditioned query embedding (BCFM output) |
| $\mathcal{M}$, $m$ | Model pool, selected model |
| $\mathcal{T}$, $\tau$ | Topology set, selected topology |
| $\mathcal{R}$, $r_k$ | Role set, role at position $k$ |
| $\Sigma$, $\sigma$ | Strategy set $\{\text{Flash, Concise, DeepThink}\}$, selected strategy |
| $\beta$, $b_{\text{rem}}$ | Total latency budget, remaining budget |
| $\lambda$ | Lagrange multiplier for latency constraint |
| $\hat{L}_{m,\sigma}$ | Predicted latency for model $m$ with strategy $\sigma$ |
| $n_m^{\text{run}}$, $n_m^{\text{wait}}$ | Running / waiting requests on model $m$ |
| $\pi_{\text{plan}}$, $\pi_{\text{exec}}$ | Planner policy, executor policy |
| $V(s)$ | Learned value function (executor critic) |
| $D_i$ | Deadline of query $i$: $t_i^{\text{arr}} + \beta_i$ |
