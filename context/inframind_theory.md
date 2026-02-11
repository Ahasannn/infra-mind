# InfraMind: Theoretical Foundations

This document provides the formal mathematical foundations for InfraMind, a hierarchical Constrained Markov Decision Process (CMDP) framework for infrastructure-aware LLM routing in Multi-Agent Systems with Earliest-Deadline-First (EDF) serving-layer scheduling.

---

## 1. Problem Formulation

### 1.1 Motivation: Resource Underutilization in Infrastructure-Oblivious Routing

Existing multi-agent LLM routing approaches (e.g., MasRouter) select collaboration topologies, roles, and models based solely on **task characteristics** — the semantic content of the query. They are oblivious to the **infrastructure state**: queue depths, GPU memory pressure, and real-time latency dynamics of the serving layer. This obliviousness leads to three compounding forms of resource underutilization:

**Problem 1: Load Imbalance Across the Model Pool.**
Infrastructure-oblivious routers develop static preferences for certain models based on training-time quality correlations. Under concurrent load, these preferences create severe imbalance: preferred models accumulate deep queues while other models sit idle. Empirically, we observe queue depth disparities exceeding 10$\times$ between small and large model groups at high arrival rates (Figure 1a). The idle capacity of underutilized models represents wasted GPU resources that could serve requests faster.

**Problem 2: Avoidable Latency From Queue Congestion.**
As requests pile up on preferred models, queuing delay dominates end-to-end latency. Meanwhile, alternative models with empty queues could serve the same request with lower latency. The gap between the latency a request *experiences* on a congested model and the latency it *would have experienced* on an idle model is entirely avoidable (Figure 1b, shaded region). An infrastructure-aware router that senses queue depths can redirect requests to underutilized models, eliminating this queuing waste.

**Problem 3: Wasted Compute From Budget Violations.**
Every multi-agent workflow that exceeds its latency budget represents wasted compute: all tokens generated across all agents — prompt processing, generation, inter-agent communication — are discarded because the response arrived too late. An infrastructure-oblivious planner may assign a three-round debate topology to a query with a 10-second budget. Even if the executor picks the fastest models, the topology's inherent sequential structure (its **latency floor**) guarantees a budget violation. All compute spent on partial rounds is lost.

**Problem 4: Missed Quality Opportunities at Low Load.**
Conversely, when system load is low and budgets are generous, infrastructure-oblivious routers miss opportunities to *invest* idle capacity in richer reasoning. Large models that are idle could serve extended DeepThink prompts to boost quality, but a load-unaware system defaults to the same routing policy regardless of available headroom. This represents quality underutilization — the system could deliver better answers with the resources it already has.

**The Core Insight:** These four problems share a single root cause — the routing policy is **decoupled from infrastructure reality**. InfraMind addresses this by making infrastructure state observable at every decision level:

| Problem | Root Cause | InfraMind Solution |
|---------|------------|-------------------|
| Load imbalance | Static model preferences | Executor observes per-model queue depths, routes to underutilised models |
| Avoidable latency | Queue-blind routing | Action-conditional latency map predicts queuing delay for each (model, strategy) pair |
| Wasted compute | Budget-blind topology | Planner observes budget, selects structurally feasible topologies |
| Missed quality | Load-blind strategy | Executor shifts to DeepThink / larger models when system has headroom |
| FCFS queuing waste | Deadline-blind serving | EDF scheduling prioritises urgent requests, reducing budget violations at the serving layer |

### 1.2 Overview

Given a query $q$ and a pool of $M$ heterogeneous LLMs $\mathcal{M} = \{m_1, m_2, \ldots, m_M\}$ with varying capabilities and latencies, InfraMind learns to:

1. **Select a collaboration topology** $\tau \in \mathcal{T}$ (e.g., single-agent, debate, ensemble) — budget-aware structural adaptation
2. **Assign roles** $\mathcal{R} = \{r_1, r_2, \ldots, r_K\}$ based on the topology
3. **Route each role to an LLM** $m \in \mathcal{M}$ with a prompting strategy $\sigma \in \Sigma$ — infrastructure-aware resource adaptation
4. **Schedule LLM requests** with Earliest-Deadline-First (EDF) priority at the serving layer — deadline-aware scheduling adaptation

The objective is to **maximize task quality** while **satisfying per-query latency deadlines** under dynamic system conditions, with the serving infrastructure co-operating via deadline-aware request scheduling. By coupling routing decisions to infrastructure state, InfraMind converts underutilised resources into quality improvements and eliminates wasted compute from budget violations.

### 1.3 Constrained Markov Decision Process (CMDP)

We formulate the routing problem as a CMDP:

$$\mathcal{C} = \langle \mathcal{S}, \mathcal{A}, P, R, C, d_0, \gamma, \beta \rangle$$

where:
- $\mathcal{S}$: State space
- $\mathcal{A}$: Action space
- $P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0,1]$: Transition probability
- $R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$: Reward function
- $C: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}^+$: Cost function (latency)
- $d_0$: Initial state distribution
- $\gamma \in [0,1]$: Discount factor
- $\beta$: Latency budget constraint

### 1.4 Optimization Objective

The goal is to find a policy $\pi^*$ that maximizes expected cumulative reward subject to expected cumulative cost constraints:

$$\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right]$$

$$\text{subject to: } \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t C(s_t, a_t) \right] \leq \beta$$

---

## 2. Hierarchical Architecture

InfraMind employs a **three-level budget-aware hierarchy**, where each level directly addresses the resource underutilization problems identified in Section 1.1:

| Level | Observes | Decides | Adapts to | Solves |
|-------|----------|---------|-----------|--------|
| **Planner** | Query + total budget $\beta$ | Collaboration *structure* (topology, roles) | Task complexity under time constraint | Problem 3 (wasted compute from infeasible topologies) |
| **Executor** | Query + role + $b_{\text{rem}}$ + system state | *Resource* allocation per node (model, strategy) | Real-time infrastructure load | Problems 1, 2, 4 (load imbalance, queuing waste, missed quality) |
| **vLLM EDF** | Deadline $D_i = t_i^{\text{arr}} + \beta_i$ | Queue ordering | Request urgency | Problem 2 (avoidable latency from FCFS queuing) |

Budget awareness permeates every decision layer, from macro-structural choices down to queue scheduling. The hierarchy is designed so that each level handles the underutilisation problem at the appropriate granularity: the planner makes coarse structural decisions that determine the latency floor, the executor makes fine-grained per-node decisions that adapt to real-time system load, and the serving layer resolves contention among concurrent requests via deadline-aware scheduling.

### 2.1 Planner (High-Level Policy)

At timestep $t=0$, the Planner observes the query **and** the total latency budget to select the workflow configuration:

$$\pi_{\text{plan}}: \mathcal{S}_{\text{plan}} \rightarrow \Delta(\mathcal{T} \times \mathcal{R})$$

**Planner Action Space:**
$$a_{\text{plan}} = (\tau, \mathcal{R}) \in \mathcal{A}_{\text{plan}}$$

where:
- $\tau \in \mathcal{T} = \{\text{IO}, \text{CoT}, \text{Chain}, \text{FullConnected}, \text{Debate}, \text{Reflection}\}$
- $\mathcal{R} \subseteq \mathcal{R}_{\text{all}}$ is the selected role set

**Key insight:** The topology determines the **latency floor** — the minimum achievable latency regardless of model or strategy choices. A three-round debate has a fundamentally higher latency floor than a single-agent call. Without budget visibility, the planner may select topologies whose latency floor exceeds the budget, leaving the executor unable to compensate. By conditioning on the budget, the planner learns to select structurally appropriate topologies: simpler topologies under tight budgets, richer collaboration when time permits.

### 2.2 Domain-Specific Role Sets

The planner selects from **domain-specific role sets** that define meaningful multi-agent collaborations. Each role set specifies the exact agents that will participate in the workflow. Role sets vary by domain to reflect domain expertise:

**Math Domain:**

| Role Set | Agents | Rationale |
|----------|--------|-----------|
| MathSolver | 1 agent | Fast single-expert for straightforward problems |
| MathSolver + Mathematician | 2 agents | Solver + verifier for moderate difficulty |
| MathSolver + Mathematician + MathTeacher | 3 agents | Multi-perspective for complex problems |
| MathSolver + Inspector | 2 agents | Solution + error checking |
| MathAnalyst + Mathematician + Inspector | 3 agents | Analysis + proof + verification |

**Code Domain:**

| Role Set | Agents | Rationale |
|----------|--------|-----------|
| ProgrammingExpert | 1 agent | Direct coding |
| ProgrammingExpert + TestAnalyst | 2 agents | Code + test review |
| ProjectManager + ProgrammingExpert + TestAnalyst | 3 agents | Plan + code + test |
| AlgorithmDesigner + ProgrammingExpert | 2 agents | Algorithm design + implementation |
| BugFixer + ProgrammingExpert | 2 agents | Debug + fix cycle |

The interplay between topology and role set determines the workflow structure: a Debate topology with a 3-agent role set produces 3 agents debating across 2 rounds, while Chain with 2 agents produces a sequential A→B pipeline. This combinatorial space is what the planner learns to navigate.

### 2.3 Executor (Low-Level Policy)

For each role $r_k$ at step $t$, the Executor selects the LLM and strategy:

$$\pi_{\text{exec}}: \mathcal{S}_{\text{exec}} \rightarrow \Delta(\mathcal{M} \times \Sigma)$$

**Executor Action Space:**
$$a_{\text{exec}}^{(k)} = (m, \sigma) \in \mathcal{A}_{\text{exec}}$$

where:
- $m \in \mathcal{M}$: Selected LLM
- $\sigma \in \Sigma = \{\text{Flash}, \text{Concise}, \text{DeepThink}\}$: Prompting strategy

---

## 3. State Space Definitions

### 3.1 Planner State

The Planner observes the query embedding concatenated with the per-query latency budget:

$$s_{\text{plan}} = \left[ \mathbf{e}_q \,\|\, \beta_i \right] \in \mathbb{R}^{d_q + 1}$$

where:
- $\mathbf{e}_q = \text{Encoder}(q)$ is the query embedding from a pre-trained encoder
- $\beta_i$ is the per-query latency budget in seconds

The planner backbone includes LayerNorm, which handles the raw-seconds scale internally.

**Why budget, not system state?** The planner's job is *structural*: deciding how much collaboration a query needs. This depends on query difficulty and available time, not on per-model queue depths. System state changes between the planner's decision and agent execution; the executor, which runs per-node in real time, is the right level for infrastructure adaptation. Keeping the planner state lean ($d_q + 1$ dimensions) ensures the budget signal is not diluted by system noise and trains efficiently.

### 3.2 Executor State

The Executor observes a rich state combining query, role, budget, and system information:

$$s_{\text{exec}}^{(k)} = \left[ \mathbf{e}_q \,\|\, \mathbf{e}_{r_k} \,\|\, b_{\text{rem}} \,\|\, \mathbf{z}_{\text{sys}} \right] \in \mathbb{R}^{d_{\text{exec}}}$$

where:
- $\mathbf{e}_q \in \mathbb{R}^{d_q}$: Query embedding
- $\mathbf{e}_{r_k} \in \mathbb{R}^{d_r}$: Role embedding (learnable)
- $b_{\text{rem}} \in \mathbb{R}$: Remaining latency budget
- $\mathbf{z}_{\text{sys}} \in \mathbb{R}^{d_{\text{sys}}}$: System state vector

### 3.3 System State Vector (Action-Conditional Latency Map)

The system state captures the current load and **action-conditional estimated latencies** for each LLM. Rather than providing a single latency estimate per model (which would require knowing the strategy before it is selected), we enumerate the predicted latency for every candidate strategy, giving the executor a complete latency map of all possible actions under current infrastructure conditions.

$$\mathbf{z}_{\text{sys}} = \bigoplus_{m \in \mathcal{M}} \mathbf{z}_m$$

For each model $m$, the state vector includes the predicted latency for **each** strategy $\sigma \in \Sigma$, plus the live queue metrics:

$$\mathbf{z}_m = \left[ \hat{L}_{m,\sigma_1} \,\|\, \hat{L}_{m,\sigma_2} \,\|\, \cdots \,\|\, \hat{L}_{m,\sigma_{|\Sigma|}} \,\|\, n_m^{\text{run}} \,\|\, n_m^{\text{wait}} \right]$$

where:
- $\hat{L}_{m,\sigma}$: **Action-conditional estimated latency** for model $m$ with strategy $\sigma$ (see Section 4)
- $n_m^{\text{run}}$: Number of requests currently running
- $n_m^{\text{wait}}$: Number of requests waiting in queue

**Dimensionality:** For $M$ models and $|\Sigma|$ strategies, $\mathbf{z}_{\text{sys}} \in \mathbb{R}^{M(|\Sigma|+2)}$

---

## 4. Latency Estimation

The latency estimation module predicts end-to-end request latency using neural network predictors for TTFT, TPOT, and output length.

### 4.1 End-to-End Latency Model

The estimated latency for a request to model $m$ under strategy $\sigma$ is:

$$\hat{L}_{m,\sigma} = \hat{T}_{m,\sigma}^{\text{TTFT}} + \hat{T}_{m,\sigma}^{\text{TPOT}} \cdot \hat{N}_{m,\sigma}^{\text{out}}$$

where:
- $\hat{T}_{m,\sigma}^{\text{TTFT}}$: Predicted Time-To-First-Token (prefill phase), conditioned on strategy $\sigma$
- $\hat{T}_{m,\sigma}^{\text{TPOT}}$: Predicted Time-Per-Output-Token (decode phase), conditioned on strategy $\sigma$
- $\hat{N}_{m,\sigma}^{\text{out}}$: Predicted output sequence length, conditioned on strategy $\sigma$

Strategy conditioning is critical because different prompting strategies (Flash, Concise, DeepThink) produce dramatically different output lengths (often 5-10x variation), which directly determines the generation-phase latency via the $\hat{T}^{\text{TPOT}} \cdot \hat{N}^{\text{out}}$ term.

### 4.2 TTFT Predictor

The Time-To-First-Token depends on input length and current system load. TTFT captures the time to process the input prompt (prefill) plus any queuing delay.

**Input Features:**
$$\mathbf{x}_{\text{TTFT}} = \left[ N^{\text{in}} \,\|\, n_m^{\text{run}} \,\|\, n_m^{\text{wait}} \right] \in \mathbb{R}^3$$

**MLP Architecture:**

$$\hat{T}_m^{\text{TTFT}} = \text{Softplus}\left( \text{MLP}_{\text{TTFT}}(\mathbf{x}_{\text{TTFT}}) \right)$$

where Softplus ensures non-negative output: $\text{Softplus}(x) = \log(1 + e^x)$.

### 4.3 TPOT Predictor

$$\hat{T}_m^{\text{TPOT}} = \text{Softplus}\left( \text{MLP}_{\text{TPOT}}([n_m^{\text{run}} \,\|\, n_m^{\text{wait}}]) \right)$$

### 4.4 Output Length Predictor

$$\hat{N}_m^{\text{out}} = \text{Softplus}\left( \text{MLP}_{\text{len}}([\mathbf{e}_q \,\|\, \mathbf{e}_\sigma \,\|\, \mathbf{e}_m]) \right)$$

### 4.5 Complete Latency Estimation Pipeline (Action-Conditional)

```
Input: query q, model pool M, strategy set Sigma, system metrics {n_m^run, n_m^wait}_{m in M}

1. Encode query: e_q = Encoder(q)
2. Get input length: N_in = TokenCount(q)
3. For each model m in M:
     For each strategy sigma in Sigma:
       a. L_hat(m,sigma) = TTFT_hat(m,sigma) + TPOT_hat(m,sigma) * N_hat_out(m,sigma)

Output: {L_hat(m,sigma)}_{m in M, sigma in Sigma} (complete latency map)
```

---

## 5. Policy Networks

### 5.1 Planner Network Architecture

$$\pi_{\text{plan}}(a_{\text{plan}} | s_{\text{plan}}; \theta_{\text{plan}}) = \text{Softmax}\left( \text{MLP}_{\text{plan}}([\mathbf{e}_q \,\|\, \beta_i]) \right)$$

The planner uses separate heads for topology and role set selection with a shared backbone:

$$\mathbf{h}_{\text{plan}} = \text{Backbone}_{\text{plan}}([\mathbf{e}_q \,\|\, \beta_i])$$
$$\pi_{\tau}(\cdot) = \text{Softmax}(\text{Head}_{\tau}(\mathbf{h}_{\text{plan}}))$$
$$\pi_{\mathcal{R}}(\cdot) = \text{Softmax}(\text{Head}_{\mathcal{R}}(\mathbf{h}_{\text{plan}}))$$

### 5.2 Executor Network Architecture

The executor uses factorized action selection with a shared backbone and separate heads for model and strategy:

$$\mathbf{h}_{\text{exec}} = \text{Backbone}_{\text{exec}}(s_{\text{exec}})$$
$$\pi_m(\cdot | s_{\text{exec}}) = \text{Softmax}(\text{Head}_m(\mathbf{h}_{\text{exec}}))$$
$$\pi_\sigma(\cdot | s_{\text{exec}}) = \text{Softmax}(\text{Head}_\sigma(\mathbf{h}_{\text{exec}}))$$
$$V(s_{\text{exec}}) = \text{Head}_V(\mathbf{h}_{\text{exec}})$$

The value head $V(s)$ provides the advantage baseline for the executor policy gradient.

---

## 6. Reward and Cost Functions

### 6.1 Quality Reward

The quality reward is computed based on task-specific evaluation:

$$R_{\text{quality}}(s, a) = Q(\text{response}, \text{ground\_truth})$$

For different tasks:
- **Code Generation:** $Q = \mathbb{1}[\text{pass\_all\_tests}]$
- **Math:** $Q = \mathbb{1}[\text{answer\_correct}]$
- **General QA:** $Q = \text{similarity}(\text{pred}, \text{gold})$

### 6.2 Latency Cost (Proportional)

The latency cost is the **normalized** observed latency, always providing a gradient signal regardless of whether the budget is met:

**Planner cost (episode-level):**
$$C_{\text{plan}} = \frac{C_{\text{workflow}}}{\beta}$$

**Executor cost (step-level):**
$$C_{\text{exec}}^{(k)} = \frac{L_k^{\text{observed}}}{b_{\text{rem}}^{(k)}}$$

where $b_{\text{rem}}^{(k)}$ is the remaining budget at step $k$.

**Why proportional and not one-sided?** A one-sided penalty $\max(0, C_{\text{workflow}} - \beta) / \beta$ produces zero gradient when under budget, removing all incentive to differentiate between actions of different latencies. The proportional cost ensures:

- Under budget ($C < \beta$): cost is in $[0, 1)$, providing gradient to prefer faster options
- At budget ($C = \beta$): cost is exactly $1.0$
- Over budget ($C > \beta$): cost grows linearly, penalizing violations

This continuous gradient is essential for learning meaningful topology and model preferences. Without it, the policy collapses to whichever action first achieves zero penalty (typically the cheapest single-agent option) regardless of quality.

### 6.3 Budget Constraint

The remaining budget after step $k$ is:

$$b_{\text{rem}}^{(k+1)} = b_{\text{rem}}^{(k)} - L_k^{\text{observed}}$$

The constraint is satisfied when:

$$C_{\text{workflow}} \leq \beta$$

---

## 7. Lagrangian Relaxation

### 7.1 Lagrangian Formulation

We convert the constrained optimization to an unconstrained problem using the Lagrangian method:

$$\mathcal{L}(\pi, \lambda) = \mathbb{E}_{\pi} \left[ R_{\text{quality}} \right] - \lambda \left( \mathbb{E}_{\pi} \left[ \frac{C_{\text{workflow}}}{\beta} \right] - 1 \right)$$

The budget-normalized constraint $\mathbb{E}[C/\beta] \leq 1$ ensures the Lagrange multiplier operates on a consistent scale regardless of the absolute budget value.

### 7.2 Dual Problem

The optimal policy solves the min-max problem:

$$\pi^* = \arg\max_{\pi} \min_{\lambda \geq 0} \mathcal{L}(\pi, \lambda)$$

### 7.3 Combined Reward

The effective reward used for policy gradient becomes:

$$\tilde{R}_{\text{plan}} = R_{\text{quality}} - \lambda \cdot \frac{C_{\text{workflow}}}{\beta}$$

$$\tilde{R}_{\text{exec}}^{(k)} = R_{\text{quality}} - \lambda \cdot \frac{L_k}{b_{\text{rem}}^{(k)}}$$

### 7.4 Lagrange Multiplier Update (Signed Dual Ascent)

The multiplier $\lambda$ is updated via **signed** gradient ascent on the dual. This is critical for correct CMDP behavior:

$$\lambda \leftarrow \max\left(0, \; \lambda + \eta_\lambda \left( \frac{\bar{C}_{\text{workflow}}}{\bar{\beta}} - 1 \right) \right)$$

where $\bar{C}_{\text{workflow}}$ and $\bar{\beta}$ are the batch means of workflow latency and budget respectively.

**The signed update is essential.** When the average latency is below budget ($\bar{C}/\bar{\beta} < 1$), the constraint gap is negative, and $\lambda$ **decreases**. This relaxes the latency pressure, allowing the policy to explore more expensive but potentially higher-quality topologies (multi-agent debate, larger models, DeepThink strategies). When the average latency exceeds budget ($\bar{C}/\bar{\beta} > 1$), $\lambda$ increases, tightening the constraint.

**Failure mode of unsigned update:** If the update uses $\text{relu}(C - \beta)$ instead of the signed gap, $\lambda$ can only increase (monotonically). The policy becomes increasingly latency-averse over training, eventually collapsing to the cheapest possible action (single-agent, smallest model, Flash strategy) regardless of quality. This is a degenerate equilibrium that satisfies the constraint but maximizes neither quality nor the Lagrangian objective.

With Softplus parameterization (ensuring $\lambda \geq 0$):

$$\tilde{\lambda} \leftarrow \tilde{\lambda} + \eta_\lambda \cdot \left( \frac{\bar{C}_{\text{workflow}}}{\bar{\beta}} - 1 \right)$$

$$\lambda = \text{softplus}(\tilde{\lambda})$$

---

## 8. Policy Gradient Training

### 8.1 Entropy Regularization

Both the planner and executor policies include entropy regularization to prevent premature convergence and maintain exploration:

$$\mathcal{H}[\pi] = -\sum_a \pi(a|s) \log \pi(a|s)$$

The entropy bonus is added to the policy gradient objective:

$$J(\theta) = \mathbb{E}_\pi[\tilde{R} \cdot \log \pi(a|s)] + \alpha_H \cdot \mathbb{E}[\mathcal{H}[\pi]]$$

**Why strong entropy matters for InfraMind:** Unlike the baseline MasRouter which maintains exploration through stochastic temperature-based sampling over learned VAE embeddings, InfraMind's REINFORCE policy gradient can collapse to a deterministic mode when the reward signal is noisy or sparse. The entropy coefficient $\alpha_H$ must be large enough to counteract this collapse:

- **Planner**: $\alpha_H^{\text{plan}} = 0.05$. The planner has 6 topologies and up to 12 role sets. Without strong entropy, it converges to a single topology within the first few gradient updates.
- **Executor**: $\alpha_H^{\text{exec}} = 0.02$. The executor has 5 models and 3 strategies. Moderate entropy ensures it explores model/strategy combinations rather than fixating on the empirically lowest-latency option.

Entropy should be **annealed** over training: start high to explore, reduce as the policy matures.

### 8.2 Planner Policy Gradient

The planner uses REINFORCE with an **exponential moving average (EMA) baseline** for variance reduction:

$$\nabla_{\theta_{\text{plan}}} J = \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta_{\text{plan}}} \log \pi_{\text{plan}}(a_{\text{plan}}^{(i)} | s_{\text{plan}}^{(i)}) \cdot \hat{A}_{\text{plan}}^{(i)}$$

where the advantage estimate uses an EMA baseline $\bar{G}$:

$$\hat{A}_{\text{plan}}^{(i)} = \tilde{R}_{\text{plan}}^{(i)} - \bar{G}_{\text{plan}}$$

$$\bar{G}_{\text{plan}} \leftarrow (1 - \rho) \cdot \bar{G}_{\text{plan}} + \rho \cdot \text{mean}(\tilde{R}_{\text{plan}})$$

with $\rho = 0.05$ (slow-moving baseline that spans multiple batches).

**Why EMA and not batch mean?** A batch-mean baseline subtracts the current batch average, producing advantage estimates centered at zero for each batch independently. When the policy is near-converged (e.g., all episodes use Chain topology), all rewards are similar and advantages are near-zero noise. An EMA baseline retains memory of past reward levels, producing meaningful advantages when the current batch's rewards differ from the historical average. This provides learning signal even during periods of low diversity.

**Full planner loss:**

$$\mathcal{L}_{\text{plan}} = -\frac{1}{N}\sum_{i} \log \pi_{\text{plan}}(a^{(i)}|s^{(i)}) \cdot \hat{A}^{(i)} - \alpha_H^{\text{plan}} \cdot \mathcal{H}[\pi_{\text{plan}}]$$

### 8.3 Executor Policy Gradient (Actor-Critic)

The executor uses an **actor-critic** architecture with a learned value baseline:

$$\hat{A}_{\text{exec}}^{(k)} = \tilde{R}_{\text{exec}}^{(k)} - V_\phi(s_{\text{exec}}^{(k)})$$

**Full executor loss:**

$$\mathcal{L}_{\text{exec}} = \underbrace{-\frac{1}{K}\sum_k \log \pi_{\text{exec}}(a^{(k)}|s^{(k)}) \cdot \hat{A}^{(k)}}_{\text{actor loss}} + \underbrace{c_V \cdot \frac{1}{K}\sum_k (V_\phi(s^{(k)}) - \tilde{R}^{(k)})^2}_{\text{critic loss}} - \underbrace{\alpha_H^{\text{exec}} \cdot \mathcal{H}[\pi_{\text{exec}}]}_{\text{entropy bonus}}$$

with $c_V = 0.5$.

### 8.4 Mini-Batch Training

Rather than performing a single gradient update per epoch (which wastes the collected experience), transitions are shuffled and split into mini-batches:

- **Mini-batch size**: 32-64 transitions
- **Gradient updates per epoch**: $\lceil N / B \rceil$ where $N$ is the number of collected transitions and $B$ is the batch size
- **Gradient clipping**: $\|\nabla\|_2 \leq 1.0$ for both planner and executor

This ensures the policy receives sufficient gradient updates to learn from the collected experience, especially during early training when episodes are diverse.

### 8.5 Complete Training Algorithm

```
Initialize: theta_plan, theta_exec, lambda_tilde = 0.0
            EMA baseline: G_bar_plan = 0.0, rho = 0.05
            Entropy coefficients: alpha_H_plan = 0.05, alpha_H_exec = 0.02

For each sweep config (arrival_rate, pattern, budget):
  For each epoch:
    Collect N episodes:
      For each episode i:
        1. Encode query: e_q = Encoder(q_i)
        2. PLANNER: s_plan = [e_q || beta_i]
           Sample (topology, role_set) ~ pi_plan(.|s_plan)
        3. EXECUTOR: For each role r_k in role_set:
           Build system state z_sys (action-conditional latency map)
           s_exec = [e_q || e_r_k || b_rem || z_sys]
           Sample (model, strategy) ~ pi_exec(.|s_exec)
           Execute LLM call with EDF priority
        4. Evaluate quality R_quality
        5. Compute rewards:
           R_plan = R_quality - lambda * C_workflow / beta
           R_exec^(k) = R_quality - lambda * L_k / b_rem^(k)

    Update policies (mini-batch SGD):
      Shuffle planner transitions into mini-batches of size B
      For each mini-batch:
        Compute advantages: A = R - G_bar_plan
        L_plan = -mean(log_prob * A) - alpha_H * entropy
        Backprop with gradient clipping (max_norm=1.0)

      Update EMA baseline: G_bar_plan = (1-rho)*G_bar_plan + rho*mean(R_plan)

      Shuffle executor transitions into mini-batches of size B
      For each mini-batch:
        Compute advantages: A = R - V(s)
        L_exec = actor_loss + 0.5*value_loss - alpha_H*entropy
        Backprop with gradient clipping (max_norm=1.0)

    Update Lagrange multiplier (SIGNED dual ascent):
      constraint_gap = mean(C_workflow) / mean(beta) - 1.0
      lambda_tilde += eta_lambda * constraint_gap
      lambda = softplus(lambda_tilde)
```

---

## 9. Workflow Latency Computation

### 9.1 Topological Execution

The workflow executes in waves based on the topology's DAG structure:

$$\text{wave}_1 = \{r : \text{in\_degree}(r) = 0\}$$
$$\text{wave}_{w+1} = \{r : \text{all predecessors in } \bigcup_{j \leq w} \text{wave}_j\}$$

### 9.2 Critical Path Latency

Since nodes within a wave execute in parallel, the workflow latency is:

$$C_{\text{workflow}} = \sum_{w=1}^{W} \max_{r_k \in \text{wave}_w} L_k$$

---

## 10. System Metrics Collection

### 10.1 Real-Time Metrics

The router continuously monitors each LLM server to capture system load:

| Metric | Symbol | Description |
|--------|--------|-------------|
| Running Requests | $n_m^{\text{run}}$ | Active inference requests (prefill + decode) |
| Waiting Requests | $n_m^{\text{wait}}$ | Queued requests awaiting processing |

These metrics are collected via the vLLM `/metrics` endpoint at regular intervals.

---

## 11. Infrastructure-Level EDF Scheduling

### 11.1 Deadline Computation

Each query $q_i$ arrives at time $t_i^{\text{arr}}$ and is assigned a latency budget $\beta_i$. The **deadline** is:

$$D_i = t_i^{\text{arr}} + \beta_i$$

All LLM calls within the same multi-agent workflow inherit the same deadline $D_i$.

### 11.2 Priority Mapping

vLLM's priority scheduler uses a min-heap where lower values are served first:

$$\text{priority}(q_i) = \lfloor D_i \rfloor$$

### 11.3 Interaction with the CMDP

The EDF priority creates a two-level scheduling hierarchy:

| Level | Decision Maker | Mechanism | Controls |
|-------|---------------|-----------|----------|
| **Routing** | CMDP Executor | Policy $\pi_{\text{exec}}$ | *Which* model and strategy to use |
| **Serving** | vLLM Scheduler | EDF priority queue | *When* a request is processed |

---

## 12. Comparison with Baseline MasRouter

Understanding why the baseline MasRouter produces diverse, stable policies while InfraMind can collapse informs the design decisions above:

| Aspect | Baseline MasRouter | InfraMind CMDP |
|--------|-------------------|----------------|
| **Architecture** | VAE + cross-attention (GFusion) | MLP + softmax heads |
| **Exploration** | Temperature-based stochastic sampling ($\tau=0.5-1.0$) | Entropy regularization ($\alpha_H$) |
| **Topology selection** | Learned via cumulative softmax CDF sampling | Learned via categorical softmax + REINFORCE |
| **Model selection** | Sequential per-agent with history accumulation | Independent per-agent via executor policy |
| **Agent count** | Learned scalar (sigmoid + scaling) | Determined by role set selection |
| **Reward** | `utility = is_solved - cost * cost_rate` (always-on cost) | `quality - lambda * latency/budget` (Lagrangian) |
| **Training** | Per-batch SGD, multiple epochs | Per-epoch batch then mini-batch SGD |
| **Infrastructure awareness** | None | Full (queue depth, latency predictions) |

**Key stability insight:** The baseline's `cost * cost_rate` term is **always active** — every model selection incurs a proportional cost penalty that shapes the policy even when quality is high. This prevents collapse to a single expensive model. InfraMind's Lagrangian achieves the same effect through the proportional latency cost $C/\beta$ and the signed dual update, which together ensure the latency signal is always present and the multiplier reaches an equilibrium rather than growing without bound.

---

## 13. Summary of Key Equations

| Component | Equation |
|-----------|----------|
| **Action-Conditional Latency** | $\hat{L}_{m,\sigma} = \hat{T}_{m,\sigma}^{\text{TTFT}} + \hat{T}_{m,\sigma}^{\text{TPOT}} \cdot \hat{N}_{m,\sigma}^{\text{out}}$ |
| **Planner State** | $s_{\text{plan}} = [\mathbf{e}_q \| \beta_i]$ |
| **Planner CMDP Reward** | $\tilde{R}_{\text{plan}} = R_{\text{quality}} - \lambda \cdot C_{\text{workflow}} / \beta$ |
| **Executor State** | $s_{\text{exec}} = [\mathbf{e}_q \| \mathbf{e}_r \| b_{\text{rem}} \| \mathbf{z}_{\text{sys}}]$ |
| **Executor CMDP Reward** | $\tilde{R}_{\text{exec}} = R_{\text{quality}} - \lambda \cdot L_k / b_{\text{rem}}$ |
| **System State (per model)** | $\mathbf{z}_m = [\hat{L}_{m,\sigma_1} \| \cdots \| \hat{L}_{m,\sigma_{|\Sigma|}} \| n_m^{\text{run}} \| n_m^{\text{wait}}]$ |
| **Lagrangian Reward** | $\tilde{R} = R_{\text{quality}} - \lambda \cdot C / \beta$ |
| **Constraint** | $C_{\text{workflow}} \leq \beta$ |
| **Multiplier Update (signed)** | $\lambda \leftarrow \max(0, \lambda + \eta_\lambda (\bar{C}/\bar{\beta} - 1))$ |
| **Policy Gradient** | $\nabla J = \mathbb{E}[\nabla \log \pi(a|s) \cdot \hat{A}] + \alpha_H \nabla \mathcal{H}[\pi]$ |
| **EMA Baseline** | $\bar{G} \leftarrow (1-\rho)\bar{G} + \rho \cdot \text{mean}(G)$ |
| **Deadline (EDF)** | $D_i = t_i^{\text{arr}} + \beta_i$ |
| **EDF Priority** | $\text{priority}(q_i) = \lfloor D_i \rfloor$ |

---

## 14. Notation Reference

| Symbol | Description |
|--------|-------------|
| $q$ | Input query |
| $\mathbf{e}_q$ | Query embedding |
| $\mathcal{M}$ | Set of available LLMs |
| $\mathcal{T}$ | Set of topologies |
| $\mathcal{R}$ | Set of roles |
| $\Sigma$ | Set of prompting strategies |
| $\tau$ | Selected topology |
| $m$ | Selected LLM |
| $\sigma$ | Selected strategy |
| $\beta$ | Latency budget |
| $b_{\text{rem}}$ | Remaining budget |
| $\lambda$ | Lagrange multiplier |
| $\tilde{\lambda}$ | Raw Lagrange parameter (before softplus) |
| $\alpha_H$ | Entropy regularization coefficient |
| $\rho$ | EMA baseline smoothing factor |
| $L$ | Observed latency |
| $\hat{L}_{m,\sigma}$ | Action-conditional predicted latency |
| $n^{\text{run}}$ | Running requests |
| $n^{\text{wait}}$ | Waiting requests |
| $\pi_{\text{plan}}$ | Planner policy |
| $\pi_{\text{exec}}$ | Executor policy |
| $V(s)$ | Learned value function (executor baseline) |
| $\bar{G}$ | EMA reward baseline (planner) |
| $D_i$ | Deadline of query $q_i$ |

---

*This document serves as a theoretical reference for the InfraMind implementation.*
