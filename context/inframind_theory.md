# InfraMind: Infrastructure-Aware Multi-Agent Orchestration

## 1. Problem Formulation

Existing multi-agent LLM routing approaches select collaboration topologies, roles, and models based solely on task semantics, ignoring the infrastructure state of the serving layer. This infrastructure-oblivious routing causes: (1) **load imbalance** — static model preferences create deep queues on preferred models while others sit idle; (2) **avoidable queuing latency** — requests wait on congested models when idle alternatives exist; (3) **wasted compute** — workflows assigned infeasible topologies exceed their latency budget, discarding all generated tokens; and (4) **missed quality opportunities** — idle capacity at low load is never invested in richer reasoning.

InfraMind addresses all four problems by making infrastructure state observable at every decision level. Given a query $q$, a model pool $\mathcal{M}$, and a per-query latency budget $\beta$, InfraMind learns to:
1. **Select a collaboration topology** $\tau \in \mathcal{T}$ and assign roles $\mathcal{R}$ — quality-driven structural planning.
2. **Route each role to an LLM** $m \in \mathcal{M}$ with a prompting strategy $\sigma \in \Sigma$ — infrastructure-aware resource allocation.
3. **Schedule requests** via Earliest-Deadline-First (EDF) priority at the serving layer — deadline-aware scheduling.

We formulate this as a Constrained Markov Decision Process (CMDP):

$$\pi^* = \arg\max_{\pi} \; \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right] \quad \text{s.t.} \quad \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t C(s_t, a_t) \right] \leq \beta$$

where $R$ is the task quality reward and $C$ is the latency cost.

---

## 2. Architecture Overview

InfraMind uses a **two-level hierarchy** separating structural decisions from resource allocation:

| Level | Observes | Decides | Operates |
|-------|----------|---------|----------|
| **Planner** | Query embedding $\mathbf{e}_q$ | Topology $\tau$, agent count $K$, roles $\mathcal{R}$ | Once at $t=0$ |
| **Executor** | Query + role + remaining budget + raw system metrics | Model $m$, strategy $\sigma$ per role | Per agent node |
| **EDF Scheduler** | Deadline $D_i = t_i^{\text{arr}} + \beta_i$ | Queue ordering | Per request at serving layer |

### 2.1 Planner (Quality-Driven, No Budget Awareness)

The Planner uses the MAS routing pipeline (VAE + GFusion cross-attention) to select topology $\tau$, agent count $K$, and roles $\mathcal{R}$ from a query embedding. The planner operates on **task semantics only** — it is purely quality-driven with no budget conditioning.

**Design rationale:** The planner decides *what reasoning structure* to use (e.g., debate, chain-of-thought, reflection). This is a semantic decision that should depend on task difficulty and domain, not on infrastructure constraints. Budget awareness is delegated entirely to the executor, which decides *how* to execute each step cheaply or richly.

The planner is initialized from a pretrained MAS Router checkpoint (transferring task_classifier, collab_mode, num_agents, role_selector weights). The LLM router from MAS is discarded — model/strategy selection is handled by the executor.

**Available topologies:** IO (single agent), CoT (chain-of-thought), Chain, Debate, Reflection, FullConnected.

### 2.2 Executor (Infrastructure-Aware, Dual-Pathway)

For each role $r_k$, the Executor selects a model and prompting strategy based on a state that fuses task semantics with real-time infrastructure conditions:

$$s_{\text{exec}}^{(k)} = \left[ \mathbf{e}_q \,\|\, \mathbf{e}_{r_k} \,\|\, b_{\text{rem}} \,\|\, \mathbf{d}_{\text{queue}} \,\|\, \mathbf{d}_{\text{e2e}} \right]$$

where:
- $\mathbf{e}_q \in \mathbb{R}^{384}$ = query embedding (SentenceTransformer)
- $\mathbf{e}_{r_k} \in \mathbb{R}^{384}$ = role embedding
- $b_{\text{rem}} \in \mathbb{R}^1$ = remaining budget, log-scaled: $\log(\max(b, 1)) / \log(300)$
- $\mathbf{d}_{\text{queue}} \in \mathbb{R}^{|\mathcal{M}|}$ = per-model queue depth: $(n_{\text{run}} + n_{\text{wait}}) / 16$
- $\mathbf{d}_{\text{e2e}} \in \mathbb{R}^{|\mathcal{M}|}$ = per-model average end-to-end latency: $\log(1 + \text{e2e}) / \log(300)$

**Raw Metrics (No Predictors).** The executor observes **raw vLLM metrics directly** — queue depths and running-average E2E latencies polled from each model's `/metrics` endpoint. No latency predictors (TTFT, TPOT) or output length estimators are used. The RL policy learns to interpret these raw signals and map them to action choices, avoiding compounding prediction errors.

**Dual-Pathway Architecture.** To prevent the budget and system signals from being drowned by the 768-dimensional semantic embeddings, the executor uses separate encoding pathways:

- **Pathway 1 — Semantic**: $[\mathbf{e}_q \| \mathbf{e}_{r_k}] \in \mathbb{R}^{768} \xrightarrow{\text{Linear+ReLU+LN}} \mathbb{R}^{128}$
- **Pathway 2 — Resource**:
  - Budget: $b_{\text{rem}} \in \mathbb{R}^{1} \xrightarrow{\text{Linear+ReLU}} \mathbb{R}^{16}$
  - System: $[\mathbf{d}_{\text{queue}} \| \mathbf{d}_{\text{e2e}}] \in \mathbb{R}^{10} \xrightarrow{\text{Linear+ReLU}} \mathbb{R}^{16}$
  - Concatenate → $\mathbb{R}^{32} \xrightarrow{\text{Linear+ReLU+LN}} \mathbb{R}^{64}$
- **Merge**: $[\text{sem}_{128} \| \text{res}_{64}] \in \mathbb{R}^{192} \xrightarrow{\text{Linear+ReLU}} \mathbb{R}^{128}$

**Joint Action Head.** A single head outputs $|\mathcal{M}| \times |\Sigma| = 15$ logits (one per model-strategy combination):

$$\pi_{m,\sigma} = \text{Softmax}(\text{Head}_{\text{joint}}(\mathbf{h})) \in \mathbb{R}^{15}, \quad V(s) = \text{Head}_V(\mathbf{h})$$

The selected joint action index $a$ decodes as: $m = a \,//\, |\Sigma|$, $\sigma = a \bmod |\Sigma|$. This captures model-strategy correlations (e.g., DeepThink is effective with large models but wasteful with small ones) that a factored design with independent heads would miss.

**Model pool** (5 models on vLLM):
- DeepSeek-R1-Distill-Qwen-32B (strongest, slowest)
- Mistral-Small-24B-Instruct-2501
- Qwen2.5-Coder-14B-Instruct
- Llama-3.1-8B-Instruct
- Llama-3.2-3B-Instruct (weakest, fastest)

**Prompt strategies** (3 strategies):
- Flash: "Go straight to the answer. No reasoning."
- Concise: "Brief 2-3 key points, then answer."
- DeepThink: "Reason thoroughly step-by-step, verify result."

### 2.3 EDF Scheduling at the Serving Layer

Each query's deadline $D_i = t_i^{\text{arr}} + \beta_i$ is propagated to vLLM as an EDF priority. Requests closer to their deadline are served first, reducing tail-latency violations.

---

## 3. System Metrics (Raw vLLM Telemetry)

The executor observes raw infrastructure metrics collected from each vLLM model server's `/metrics` endpoint by a background polling thread (5-second interval):

| Metric | Description | Normalization |
|--------|-------------|---------------|
| Queue depth | Running + waiting requests per model | $(n_{\text{run}} + n_{\text{wait}}) / 16$ |
| E2E latency | Delta-averaged end-to-end request latency | $\log(1 + \text{e2e}) / \log(300)$ |

These two metrics per model (10 total for 5 models) are fed directly to the executor's resource pathway. The metrics watcher also collects TTFT, ITL, KV cache usage, queue time, and inference time from vLLM's Prometheus endpoint, but only queue depth and E2E average are used by the executor state.

**Design rationale.** Earlier versions used neural predictors (TTFT predictor, TPOT predictor, output length estimator) to construct a predicted latency "price list" $\hat{L}_{m,\sigma}$ for every (model, strategy) pair. This approach was abandoned because: (1) prediction errors compounded across three models (TTFT × TPOT × length); (2) the predictors added training complexity and data requirements; (3) raw metrics provide a more direct signal — the RL policy can learn to interpret queue depth and E2E latency patterns that the predictors may miss.

---

## 4. Training: PPO-Lagrangian

### 4.1 Constrained Optimization via Lagrangian Relaxation

The CMDP constraint $\mathbb{E}[C] \leq \beta$ is enforced via a shared Lagrange multiplier $\lambda$ that converts the constrained problem into an unconstrained saddle-point problem:

$$\max_{\pi} \min_{\lambda \geq 0} \; \mathbb{E}_{\pi}[R] - \lambda \cdot (\mathbb{E}_{\pi}[C] - 1)$$

where cost $C = \text{latency} / \beta$ so the constraint threshold is 1 (budget-normalized). A single shared $\lambda$ serves both planner and executor — the $1/\beta$ factor in the cost naturally creates different penalty magnitudes across budget tiers (tight budgets amplify cost, loose budgets attenuate it).

**Dual update** (after each executor batch):
$$\lambda \leftarrow \text{clip}\left(\lambda + \eta_\lambda \cdot (\bar{C} - 1),\; 0,\; \lambda_{\max}\right)$$

where $\bar{C}$ is the batch-mean episode cost ratio, $\eta_\lambda = 0.001$, and $\lambda_{\max} = 1.0$.

When the policy violates budgets ($\bar{C} > 1$), $\lambda$ increases → both levels prioritize cost reduction. When the policy is within budget ($\bar{C} < 1$), $\lambda$ decreases → both levels focus on quality.

### 4.2 Planner Training (REINFORCE + Shared $\lambda$)

$$\text{utility}_i = \underbrace{q_i}_{\text{quality}} - \underbrace{\lambda \cdot C_i}_{\text{cost penalty}}$$

where $q_i = 1$ if solved, $0$ otherwise, and $C_i = L_{\text{workflow}} / \beta$.

$$A_i = \frac{\text{utility}_i - \bar{u}}{\sigma_u} \qquad \text{(normalized advantage)}$$

$$\mathcal{L}_{\text{planner}} = -\log \pi_{\text{plan}} \cdot A + \mathcal{L}_{\text{task}} + 0.001 \cdot \mathcal{L}_{\text{VAE}}$$

The planner uses REINFORCE (no critic) because it makes a single decision per episode. Task classification loss and VAE regularization are inherited from the MAS Router pretraining.

### 4.3 Executor Training (PPO + Shared $\lambda$)

**Reward construction (per step):**
$$r_i = \underbrace{q_i}_{\text{quality (binary)}} - \underbrace{\lambda \cdot \frac{l_i}{\beta}}_{\text{step cost}}$$

where $q_i$ is the episode-level solve outcome (shared across all steps) and $l_i$ is the step-level latency.

**PPO clipped surrogate** (K=3 mini-epochs per batch):
$$\hat{A}_i = r_i - V(s_i) \qquad \text{(value-baselined, then normalized)}$$

$$\rho_i = \frac{\pi_\theta(a_i | s_i)}{\pi_{\theta_{\text{old}}}(a_i | s_i)}$$

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}\left[\min\left(\rho \hat{A},\; \text{clip}(\rho, 1-\epsilon, 1+\epsilon) \hat{A}\right)\right]$$

$$\mathcal{L}_{\text{executor}} = \mathcal{L}_{\text{actor}} + 0.5 \cdot \text{MSE}(V(s), r) - 0.10 \cdot \mathcal{H}[\pi]$$

where $\epsilon = 0.2$ (clipping), $0.5$ is the value coefficient, and $0.10$ is the entropy bonus coefficient.

### 4.4 Budget Randomization (Fixed Tiers)

Each training item draws a budget from a fixed set of 8 tiers with equal probability:

$$\beta \in \{10, 30, 50, 100, 200, 300, 600, 1000\} \text{ seconds}$$

| Tier | Regime | Expected behavior |
|------|--------|-------------------|
| 10s | Extreme constraint | Only Llama-3B + Flash feasible |
| 30s | Tight | Cheap models only, multi-step barely fits |
| 50s | Moderate | Cheap/mid models comfortable |
| 100s | Transition | Qwen comfortable, DeepSeek infeasible |
| 200s | System-awareness boundary | DeepSeek viable at low load only |
| 300s | DeepSeek viable | 2–3 steps possible with cheap filling |
| 600s | Comfortable | DeepSeek comfortable at all loads |
| 1000s | Generous | Unconstrained quality optimization |

**Design rationale.** Fixed tiers replaced earlier LogUniform(5, 300) sampling because: (1) each tier gets equal representation; (2) cleaner learning signal at specific budget breakpoints; (3) easier analysis of budget-conditional behavior.

### 4.5 Arrival Rate Sweeps

Each training epoch consists of 6 arrival rate sweeps: $\{10, 30, 50, 100, 150, 200\}$ req/min (shuffled order). For each sweep:
1. All training items are processed with Poisson arrivals at the given rate
2. Every 64 episodes, a training batch update is performed (both planner and executor)
3. Checkpoints are saved periodically

After each full epoch (6 sweeps), validation is run deterministically on held-out data at the median budget tier (200s) and rate=100 req/min. LR scheduling (ReduceLROnPlateau) and early stopping (patience=12) are based on validation solve rate.

### 4.6 Warmup Strategy

$\lambda$ is active from epoch 0 (no warmup). Earlier experiments showed that warmup (holding $\lambda = 0$ for several epochs) was counterproductive: it allows the policy to develop quality-biased habits that $\lambda$ must then undo, creating instability. Initializing $\lambda = 0.2$ from the start provides gentle cost pressure that shapes budget-conditional behavior alongside quality learning.

### 4.7 Workflow Latency

The workflow executes in topological waves. Nodes within a wave run in parallel; waves execute sequentially:

$$C_{\text{workflow}} = \sum_{w=1}^{W} \max_{r_k \in \text{wave}_w} L_k$$

---

## 5. How Load-Awareness Emerges

The executor does not receive an explicit "load level" signal. Instead, load-awareness emerges **indirectly through cost pressure**:

1. High arrival rates → deeper queues → longer E2E latencies → budget depletes faster
2. Faster budget depletion → higher cost ratio $L/\beta$
3. Higher cost ratio → $\lambda$ dual update increases → executor shifts to cheaper (model, strategy) pairs
4. Additionally, the executor sees raw queue depths in its state, learning that high queues predict long waits

This indirect mechanism is sufficient: experiments show that when cost ratio > 1.0, DeepSeek usage drops from 25.9% to 8.5% while Llama-3B surges from ~15% to 40.2%.

---

## 6. Notation Reference

| Symbol | Description |
|--------|-------------|
| $q$, $\mathbf{e}_q$ | Input query, query embedding |
| $\mathcal{M}$, $m$ | Model pool (5 models), selected model |
| $\mathcal{T}$, $\tau$ | Topology set (6 topologies), selected topology |
| $\mathcal{R}$, $r_k$ | Role set, role at position $k$ |
| $\Sigma$, $\sigma$ | Strategy set $\{\text{Flash, Concise, DeepThink}\}$, selected strategy |
| $\beta$, $b_{\text{rem}}$ | Total latency budget, remaining budget |
| $\lambda$ | Shared Lagrange multiplier (quality-cost tradeoff) |
| $C$ | Cost: $\text{latency} / \beta$ (budget-normalized) |
| $\mathbf{d}_{\text{queue}}$ | Per-model queue depth (running + waiting) / 16 |
| $\mathbf{d}_{\text{e2e}}$ | Per-model log-scaled average E2E latency |
| $\pi_{\text{plan}}$, $\pi_{\text{exec}}$ | Planner policy, executor policy |
| $V(s)$ | Learned value function (executor critic) |
| $D_i$ | Deadline of query $i$: $t_i^{\text{arr}} + \beta_i$ |
| $\eta_\lambda$ | Lagrange multiplier learning rate (0.001) |
| $\lambda_{\max}$ | Maximum $\lambda$ cap (1.0) |
| $\epsilon$ | PPO clipping parameter (0.2) |
