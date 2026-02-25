# Alternative RL Approaches for InfraMind: Research Survey

## Problem Recap

InfraMind's executor must select (model, strategy) pairs for each role in a multi-agent LLM workflow, maximizing correctness (binary is_solved) while staying within a latency budget sampled from LogUniform(5, 300). Current REINFORCE/Actor-Critic with hand-crafted reward shaping collapses to cheap configs (Llama-3B + Flash). This document surveys four alternative directions.

---

## 1. Reward-Conditioned Policies / Upside-Down RL / Decision Transformer

### Core Idea

Instead of maximizing reward via policy gradients, train a policy conditioned on the *desired* return. At test time, condition on "high quality + within budget" to get the corresponding actions.

### Key Papers

**Decision Transformer** (Chen et al., NeurIPS 2021)
- Casts RL as sequence modeling: input is (return-to-go, state, action) tuples
- At test time, set return-to-go to desired value; the model generates actions to achieve it
- arxiv.org/abs/2106.01345

**Reward-Conditioned Policies (RCP)** (Kumar et al., 2019)
- Non-expert trajectories are optimal supervision *for matching the reward of that trajectory*
- Train: p(a|s, desired_reward) via supervised learning on collected data
- At test time: condition on high reward
- arxiv.org/abs/1912.13465

**Upside-Down RL** (Schmidhuber, 2019)
- Maps (desired_reward, time_horizon) -> actions via supervised learning
- Commands: "obtain X total reward in Y steps"
- arxiv.org/abs/1912.02875

**Constrained Decision Transformer (CDT)** (Liu et al., ICML 2023)
- Directly relevant: extends DT to handle cost constraints
- Input tokens: (reward-return-to-go, **cost-return-to-go**, state, action)
- At test time, specify desired reward AND cost budget -- zero-shot adaptation to new thresholds
- Pareto frontier-oriented data augmentation: relabels trajectories with infeasible (reward, cost) pairs by finding nearest safe Pareto-optimal trajectory
- Stochastic policy with entropy regularization
- Loss: negative log-likelihood + entropy bonus
- **Key result**: Can generalize to different cost thresholds *without retraining*
- arxiv.org/abs/2302.07351

### How It Maps to InfraMind

The executor could be trained as a return-conditioned policy:
- **State**: (query_embedding, role_embedding, system_metrics)
- **Conditioning variables**: (desired_quality ∈ {0, 1}, budget_remaining)
- **Action**: (model, strategy) selection
- **Training data**: All collected (state, action, outcome, latency) tuples from exploration
- At test time: condition on (quality=1, budget=budget_remaining)

The CDT variant is most directly applicable:
- Condition on both reward-return (quality) and cost-return (latency budget)
- Train on offline data from exploration episodes
- At deployment, set reward-target=1.0, cost-target=budget_remaining

### Specific Reward Formulation

No reward engineering needed -- this is the key advantage. Instead:
- Collect trajectories: (state, action, is_solved, latency_used)
- Train: p(action | state, desired_is_solved, budget_remaining) via supervised learning
- The model learns the *mapping* from desired outcomes to actions

### Failure Modes and Limitations

1. **Cannot stitch trajectories**: RCSL/DT cannot combine sub-trajectories from different episodes to create better-than-seen behavior. If the training data never shows "correct answer within tight budget using model X", the policy cannot discover this combination. This is the fundamental limitation vs. dynamic programming methods (Q-learning).
   - Mitigation: CDT's data augmentation (Pareto relabeling) partially addresses this
   - For InfraMind: each episode is SHORT (1-6 steps), so stitching is less critical

2. **Conditioning on out-of-distribution returns**: If you condition on quality=1 but most training data has quality=0, the policy may hallucinate actions. Need sufficient coverage of high-quality trajectories.
   - For InfraMind: with 5 models x 3 strategies, the best configs achieve ~73% accuracy, so there's decent positive data

3. **Scalar return conditioning is a bottleneck**: A single scalar doesn't capture the full complexity of what makes a trajectory good.
   - CDT addresses this with multi-dimensional conditioning (reward + cost)

4. **Requires offline data collection first**: Need a diverse dataset of (state, action, outcome) tuples, typically from random or epsilon-greedy exploration.

### Verdict for InfraMind

**HIGH RELEVANCE**. CDT is the most directly applicable. The key insight: instead of trying to craft a reward that balances quality and cost (which keeps failing), condition the policy on BOTH desired quality AND budget, and let supervised learning figure out the mapping. The short episode length (1-6 steps per workflow) means the stitching limitation is minimal. Main risk: need sufficient positive examples in training data.

---

## 2. Hindsight Experience Replay (HER) for Budget Constraints

### Core Idea

When a trajectory fails to meet its goal, relabel it as if the goal was what was actually achieved. Applied to budget constraints: when an episode exceeds its budget but gets the correct answer, relabel it with a higher budget where it would have succeeded.

### Key Papers

**Hindsight Experience Replay** (Andrychowicz et al., NeurIPS 2017)
- For goal-conditioned RL with sparse rewards
- After episode: replace desired goal with achieved goal in replay buffer
- Turns every episode into a "success" for some goal
- papers.nips.cc/paper/7090-hindsight-experience-replay.pdf

**Generalized Hindsight** (Li et al., NeurIPS 2020)
- Extends HER beyond goal-conditioned settings
- Relabels not just goals but also reward functions
- proceedings.neurips.cc/paper/2020/file/57e5cb96e22546001f1d6520ff11d9ba-Paper.pdf

**Hindsight Goal-conditioned Regularization (GCHR)** (2025)
- Generates action regularization priors from hindsight goals
- Enables off-policy algorithms to maximize experience utilization
- arxiv.org/abs/2508.06108

### How It Maps to InfraMind

The budget is naturally a "goal" that can be relabeled:

**Scenario 1: Correct but over-budget**
- Episode: budget=30s, used DeepSeek-32B+DeepThink, latency=45s, is_solved=1
- Relabel: budget=50s -> this is now a success trajectory for budget=50s
- Dense signal: "For loose budgets, big model + deep thinking works"

**Scenario 2: Wrong but under-budget**
- Episode: budget=60s, used Llama-3B+Flash, latency=3s, is_solved=0
- Relabel: budget=5s -> "Even with tight budget, this was the right call for cost"
- BUT: quality=0, so this is still a failure. Relabeling budget doesn't help.

**Scenario 3: Correct and under-budget**
- Episode: budget=60s, used Mistral-24B+Concise, latency=8s, is_solved=1
- Relabel: budget=10s -> "Even with tight budget of 10s, this config succeeds"
- Very valuable: shows good configs work across budget ranges

### Proposed Relabeling Strategy

```
For each episode (state, actions, budget, latency, is_solved):
    # Original experience
    store(state, actions, budget, is_solved, latency)

    # Hindsight relabeling
    if is_solved:
        # Relabel with actual latency as budget (tighter)
        # Shows: "this config works even with budget = actual_latency"
        new_budget = latency * (1 + small_margin)
        store(state, actions, new_budget, is_solved=1, latency)

        # Also relabel with several intermediate budgets
        for b in linspace(latency, budget, num_relabels):
            store(state, actions, b, is_solved=1, latency)

    if not is_solved:
        # Cannot relabel quality -- is_solved is binary ground truth
        # But CAN relabel budget to show "even with infinite budget, this failed"
        # This teaches: "this (model, strategy) is BAD for this query"
        store(state, actions, budget=300, is_solved=0, latency)
```

### Failure Modes and Limitations

1. **Only works for goal-conditioned policies**: The policy must take budget as an input (which InfraMind's executor already does -- budget_remaining is part of the state).

2. **Cannot relabel quality**: HER relabels goals, not outcomes. If a trajectory gets the wrong answer, you cannot relabel it as correct. Quality is ground truth, not a goal.
   - This limits HER's usefulness: it only helps learn budget sensitivity, not quality improvement

3. **Bias in relabeled data**: Over-representing easy relabeled budgets may bias the policy toward thinking most budgets are easy.
   - Mitigation: importance sampling (USHER approach)

4. **Off-policy requirement**: HER requires off-policy learning (DQN, SAC, etc.). REINFORCE is on-policy.
   - For InfraMind: would need to switch executor to off-policy (DQN or SAC)

### Combination with Approach 1 (CDT)

HER naturally combines with reward-conditioned policies / CDT:
- Collect episodes with various (budget, outcome) pairs
- Relabel budgets in hindsight for correct episodes
- Train CDT on the augmented dataset
- This is essentially what CDT's "Pareto frontier data augmentation" already does!

### Verdict for InfraMind

**MEDIUM RELEVANCE**. HER provides a principled way to augment training data with hindsight-relabeled budgets, but it cannot help with the core problem (quality improvement). Best used as a data augmentation technique alongside CDT. The relabeling of correct episodes with tighter budgets is directly analogous to CDT's Pareto frontier augmentation. The limitation is that wrong answers stay wrong -- HER cannot fix the quality signal.

---

## 3. Contextual Bandits for LLM Routing

### Core Idea

Treat each (model, strategy) selection as a contextual bandit problem: observe context (query features + system state), choose an arm (model+strategy), observe reward. No sequential dependency between steps.

### Key Papers and Systems

**FrugalGPT** (Chen et al., Stanford, 2023)
- LLM cascade: query cheap model first, escalate if confidence is low
- Scoring function g(query, answer) -> [0,1] trained via DistilBERT regression
- Optimization: maximize quality subject to avg_cost < budget (mixed-integer program)
- Key result: matches GPT-4 with 98% cost reduction
- arxiv.org/abs/2305.05176

**RouteLLM** (Ong et al., LMSYS, 2024)
- Binary routing: strong model vs. weak model
- Trained on human preference data from Chatbot Arena
- Training data format: {query, strong_response, weak_response, preference_label}
- Data augmentation: 120K queries + GPT-4-as-judge synthetic labels
- Four router architectures (MF, SW, BERT, Causal LM)
- Cost reduction: 85% on MT Bench, 45% on MMLU, 35% on GSM8K at 95% of GPT-4 quality
- arxiv.org/abs/2406.18665

**AutoMix** (Aggarwal et al., NeurIPS 2024)
- Self-verification: smaller model generates answer + self-checks via few-shot prompting
- POMDP-based meta-verifier decides whether to escalate to larger model
- No verifier training needed -- few-shot prompting for verification
- 50%+ cost reduction at comparable performance
- arxiv.org/abs/2310.12963

**Hybrid LLM** (Ding et al., ICLR 2024)
- Quality-aware query routing with tunable quality levels
- Router predicts query difficulty, routes to small/large model
- 40% fewer calls to large models, no quality drop
- arxiv.org/abs/2404.14618

**LLM Bandit** (Li et al., 2025)
- Multi-armed bandit with preference-conditioned routing
- Model identity vectors via Item Response Theory (IRT)
- Bandit reward: r(x,k) = [s(x,k), -c_k] (vector-valued: quality + negative cost)
- User preference vector omega scalarizes: r_omega = omega_1 * quality - omega_2 * cost
- Generalizes to unseen models with only 20-50 evaluations
- 27% improvement in cost-efficiency vs. existing routing
- arxiv.org/abs/2502.02743

**BaRP: Bandit-feedback Routing with Preferences** (Wang et al., 2025)
- Most relevant to InfraMind's problem
- Reward: r_t = w_q * quality - w_c * normalized_cost, where w = (w_q, w_c) on 1-simplex
- Policy: pi(a|x,w) = softmax(g_theta([h(x); phi(w)]))
  - h(x): frozen prompt encoder
  - phi(w): trainable preference MLP mapping 2D preference to higher-dim embedding
  - g_theta: decision head producing logits over K models
- Training: REINFORCE with baseline + entropy regularization
  - Loss: -(r_t - b_t) * log pi(a|x,w) - beta * H(pi)
  - Adam, lr=1e-4, batch=32, 100 epochs
- **Key feature**: preference vector tunable at inference without retraining
- 16.84% improvement over GraphRouter, 50% cost reduction
- arxiv.org/abs/2510.07429

**MixLLM** (2025, NAACL)
- Dynamic contextual bandit routing with per-LLM quality/cost predictors
- Lightweight prediction models per LLM (no system-wide retraining for new models)
- Meta-decision maker optimizes quality-cost-latency tradeoff
- 97.25% of GPT-4 quality at 24.18% cost
- arxiv.org/abs/2502.18482

**xRouter** (Salesforce, 2025)
- RL-trained router (Qwen2.5-7B) that decides: answer locally OR offload to external model
- **Reward: R = R_binary * (K - lambda * C)**
  - R_binary in {0,1}: task success (binary gate)
  - K: fixed success bonus
  - C: total cost of model invocations
  - lambda: cost penalty coefficient (lambda=2 works best)
- **Success-gated**: wrong answer -> zero reward regardless of cost
- Training: DAPO (variant of GRPO) in Verl framework
- Results: near GPT-5 accuracy, 60-80% cost reduction
- **Failure mode**: converges to "disappointingly simple patterns" rather than complex multi-step strategies; some model architectures (Qwen3-4B) are "remarkably resistant to router training"
- arxiv.org/abs/2510.08439

**Router-R1** (Zhang et al., 2025)
- Router as LLM: interleaves "think" (deliberation) and "route" (model invocation)
- Reward = format_reward + outcome_reward + cost_reward
- Trained via GRPO
- Conditions on model descriptors (pricing, latency, example performance)
- Generalizes to unseen model pools
- arxiv.org/abs/2506.09033

### Reward Formulations That Work (Summary)

| System | Reward Formula | Key Feature |
|--------|---------------|-------------|
| FrugalGPT | Maximize quality s.t. avg_cost < budget | Constrained optimization |
| BaRP | w_q * quality - w_c * norm_cost | Preference-conditioned, tunable at inference |
| LLM Bandit | omega_1 * quality - omega_2 * cost | Vector-valued, preference-conditioned |
| xRouter | R_binary * (K - lambda * C) | Success-gated: wrong = 0 reward |
| Router-R1 | format + outcome + cost_penalty | Multi-component |
| MixLLM | Contextual bandit meta-decision | Per-model quality/cost predictors |

### Key Insight: Success-Gated Rewards

The most successful systems (xRouter, BaRP) use a **success-gated** approach: quality is the primary gate, and cost optimization happens *only among correct answers*. This prevents the collapse InfraMind is experiencing, where the policy learns that "cheap + wrong" beats "expensive + right" in the reward landscape.

**xRouter's formulation is closest to what InfraMind needs:**
```
reward = is_solved * (K - lambda * latency/budget)
```
- Correct + cheap: high reward (K - small_penalty)
- Correct + expensive: moderate reward (K - large_penalty, but still positive)
- Wrong (any cost): ZERO reward

This eliminates the failure mode where wrong+cheap beats correct+expensive.

### How It Maps to InfraMind

InfraMind's executor is already essentially a contextual bandit at each step:
- Context: (query_embedding, role_embedding, budget_remaining, system_metrics)
- Arms: 15 (model, strategy) pairs
- Reward: is_solved (with latency consideration)

**Recommended adaptation of BaRP for InfraMind:**
```python
# Per-step reward for executor
quality = is_solved  # 0 or 1
latency_ratio = actual_latency / budget_total
cost_penalty = min(latency_ratio, 1.0)

# Preference-conditioned (tunable at inference)
reward = w_q * quality - w_c * cost_penalty

# Or simpler, xRouter-style success-gated:
reward = quality * (1.0 - lambda * cost_penalty)
```

### Failure Modes

1. **Bandit assumes single-step**: Most bandit approaches make one routing decision per query. InfraMind makes sequential decisions (one per role in workflow). Need to handle credit assignment across steps.
   - Mitigation: Per-step bandits with shared parameters, or treat full workflow as single bandit action

2. **Preference vector requires tuning**: BaRP's preference vector must be set appropriately. Wrong settings lead to either quality collapse (too much cost weight) or cost explosion (too much quality weight).

3. **Cold start**: Bandit methods need exploration data. Initial performance may be poor.

4. **xRouter limitation**: Converges to simple patterns, not complex multi-step strategies. This is concerning for multi-agent workflows.

### Verdict for InfraMind

**HIGHEST RELEVANCE**. The contextual bandit literature on LLM routing directly addresses InfraMind's problem. The success-gated reward formulation (xRouter) and preference-conditioned routing (BaRP/LLM Bandit) are immediately applicable. The key takeaway: **gate the reward on correctness first, then optimize cost among correct solutions**. This is a simple change to InfraMind's reward that could prevent collapse.

---

## 4. Curriculum Learning for Budget Parameters

### Core Idea

Instead of training on random budgets from the start, schedule budget difficulty during training: start with generous budgets (easy -- just get the answer right), then progressively tighten (harder -- get it right AND be fast).

### Key Papers

**ACPO: Adversarial Constrained Policy Optimization** (2024)
- Key insight: relaxing constraints during early training leads to better final performance
- Two alternating stages: (1) maximize reward under current budget, (2) minimize cost while maintaining reward threshold
- Outperforms fixed-budget baselines
- arxiv.org/abs/2410.20786

**Self-Paced Deep Reinforcement Learning** (Klink et al., 2020)
- Curriculum as inference: progressively learned distributions over task parameters
- Agent controls its own pace based on current competence
- arxiv.org/abs/2004.11812

**Distributionally Robust Self-Paced Curriculum RL (DR-SPCRL)** (2025)
- Treats robustness budget as continuous curriculum
- Adapts constraint difficulty based on agent progress
- arxiv.org/abs/2511.05694

**Curriculum Learning for Job Shop Scheduling** (2023)
- Easy-to-hard curriculum for constrained optimization
- Probabilistic scheduler shifts focus from easy to hard tasks
- arxiv.org/abs/2305.10192

**Curriculum RL from Easy to Hard** (2025)
- Gradual difficulty scaling improves generalization
- arxiv.org/abs/2506.06632

### Proposed Curriculum for InfraMind

**Phase 1: Quality Only (epochs 1-N/3)**
- Budget = infinity (or 300s -- always achievable)
- Reward = is_solved (pure quality signal)
- Goal: learn which (model, strategy) configs produce correct answers for which queries
- This prevents early collapse to cheap configs

**Phase 2: Gradual Budget Introduction (epochs N/3 - 2N/3)**
- Budget ~ LogUniform(50, 300) -- generous but bounded
- Reward = is_solved * (1.0 - 0.3 * max(0, latency/budget - 1.0))
- Goal: learn to respect budget while maintaining quality

**Phase 3: Full Budget Range (epochs 2N/3 - N)**
- Budget ~ LogUniform(5, 300) -- full range including tight budgets
- Full reward function with quality gate + cost shaping
- Goal: learn adaptive behavior across all budget regimes

**Alternative: ACPO-style Adversarial Curriculum**
```
for epoch in range(num_epochs):
    # Stage 1: Maximize quality under relaxed budget
    budget_threshold = initial_budget - epoch * tightening_rate
    train_maximize_quality(budget=budget_threshold)

    # Stage 2: Minimize latency while maintaining quality
    quality_threshold = current_quality * 0.95  # allow 5% drop
    train_minimize_latency(quality_floor=quality_threshold)
```

**Alternative: Self-Paced Curriculum**
- Start with LogUniform(100, 300) -- generous
- After each validation epoch, compute success rate per budget quartile
- If success rate > 80% in current easiest quartile, expand range downward
- Automatically adapts to agent's learning pace

### Specific Schedule Recommendations

| Approach | Pros | Cons |
|----------|------|------|
| Fixed easy-to-hard | Simple, reliable | Requires manual scheduling |
| ACPO adversarial | Adapts automatically | More complex, two-stage |
| Self-paced | Agent controls pace | May get stuck on easy |
| Random (current) | No schedule needed | Causes collapse (observed!) |

### Failure Modes

1. **Catastrophic forgetting**: When tightening budget, agent may forget quality-oriented behavior learned in Phase 1.
   - Mitigation: replay buffer with Phase 1 experiences, or regularization toward Phase 1 policy (KL penalty)

2. **Self-paced gets stuck**: Agent may never voluntarily increase difficulty.
   - Mitigation: minimum progression schedule as fallback

3. **Phase transition shock**: Sudden budget tightening can destabilize training.
   - Mitigation: smooth transitions, overlap between phases

4. **Over-reliance on easy budgets**: If most training is on generous budgets, policy may not learn tight-budget behavior.
   - Mitigation: ensure Phase 3 has enough epochs for tight budget learning

### Combination with Other Approaches

Curriculum learning combines naturally with all other approaches:
- **CDT + curriculum**: Collect data with curriculum, train CDT on full dataset
- **Bandit + curriculum**: Start bandit exploration with generous budgets
- **HER + curriculum**: Hindsight relabel from Phase 1 to create tight-budget successes

### Verdict for InfraMind

**HIGH RELEVANCE**. The current LogUniform(5, 300) random budget sampling is likely contributing to collapse: the agent sees tight budgets early and learns "cheap is the only way to not get penalized." Starting with generous budgets to establish quality-first behavior, then gradually tightening, directly addresses this. This is a LOW-COST change (just modify the budget sampling schedule) with potentially high impact.

---

## Synthesis: Recommended Approach for InfraMind

### Priority 1: Success-Gated Reward (from Contextual Bandits literature)
**Immediate change, minimal code modification.**

Replace the current reward with:
```python
if is_solved:
    reward = 1.0 - alpha * max(0, latency/budget - 1.0)  # [0.7, 1.0]
else:
    reward = 0.0  # or small negative: -0.1
```

This matches xRouter's proven formulation. Wrong answers get ZERO (or near-zero), not a graded penalty that can exceed the reward for being correct. The key property: **worst correct (0.7) >> best wrong (0.0)**. No possibility of collapse to cheap+wrong.

### Priority 2: Curriculum on Budget (from Curriculum Learning literature)
**Low-cost change, just modify budget sampling.**

```python
if epoch < num_epochs // 3:
    budget = 300.0  # generous -- learn quality first
elif epoch < 2 * num_epochs // 3:
    budget = log_uniform(50, 300)  # introduce budget pressure
else:
    budget = log_uniform(5, 300)  # full range
```

### Priority 3: Preference-Conditioned Policy (from BaRP / LLM Bandit)
**Medium effort, requires architecture change.**

Add a preference vector w = (w_quality, w_cost) to the executor input. Train with varying w values. At inference, set w to desired quality-cost tradeoff. This gives a single policy that can serve the entire Pareto frontier.

Architecture change for executor:
```python
# Current: executor_input = [query_emb, role_emb, budget_remaining, system_metrics]
# New: executor_input = [query_emb, role_emb, budget_remaining, system_metrics, pref_emb]
# Where pref_emb = MLP(preference_vector)  # 2D -> 32D
```

### Priority 4: CDT-style Return Conditioning (from Decision Transformer literature)
**Higher effort, architecture overhaul.**

Reframe the executor as a CDT:
- Input: (desired_quality, budget_remaining, query_emb, role_emb, system_metrics)
- Output: (model, strategy) distribution
- Training: supervised learning on collected (state, action, outcome, cost) tuples
- At test time: condition on quality=1.0, budget=budget_remaining

This eliminates reward engineering entirely but requires collecting a diverse offline dataset first and fundamentally changes the training paradigm from online RL to offline supervised learning.

### What NOT to Do

1. **Do not use HER alone** -- it cannot fix quality (wrong stays wrong), only budget sensitivity
2. **Do not use vanilla Decision Transformer** -- the stitching limitation means it cannot discover novel (model, strategy) combos not well-represented in data
3. **Do not set preference vectors to extreme values** -- BaRP shows this leads to either quality collapse or cost explosion
4. **Do not skip the quality gate in rewards** -- every successful LLM routing system gates on correctness first

---

## Key References

1. Chen et al., "Decision Transformer: RL via Sequence Modeling," NeurIPS 2021 -- https://arxiv.org/abs/2106.01345
2. Kumar et al., "Reward-Conditioned Policies," 2019 -- https://arxiv.org/abs/1912.13465
3. Schmidhuber, "RL Upside Down," 2019 -- https://arxiv.org/abs/1912.02875
4. Liu et al., "Constrained Decision Transformer for Offline Safe RL," ICML 2023 -- https://arxiv.org/abs/2302.07351
5. Andrychowicz et al., "Hindsight Experience Replay," NeurIPS 2017 -- https://papers.nips.cc/paper/7090-hindsight-experience-replay.pdf
6. Chen et al., "FrugalGPT," 2023 -- https://arxiv.org/abs/2305.05176
7. Ong et al., "RouteLLM," 2024 -- https://arxiv.org/abs/2406.18665
8. Aggarwal et al., "AutoMix," NeurIPS 2024 -- https://arxiv.org/abs/2310.12963
9. Ding et al., "Hybrid LLM," ICLR 2024 -- https://arxiv.org/abs/2404.14618
10. Li et al., "LLM Bandit," 2025 -- https://arxiv.org/abs/2502.02743
11. Wang et al., "BaRP: Learning to Route LLMs from Bandit Feedback," 2025 -- https://arxiv.org/abs/2510.07429
12. MixLLM, NAACL 2025 -- https://arxiv.org/abs/2502.18482
13. Qian et al., "xRouter," Salesforce, 2025 -- https://arxiv.org/abs/2510.08439
14. Zhang et al., "Router-R1," 2025 -- https://arxiv.org/abs/2506.09033
15. ACPO, 2024 -- https://arxiv.org/abs/2410.20786
16. Panda et al., "Adaptive LLM Routing Under Budget Constraints," EMNLP 2025 -- https://arxiv.org/abs/2508.21141
17. Klink et al., "Self-Paced Deep RL," 2020 -- https://arxiv.org/abs/2004.11812
