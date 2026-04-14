# Research Companion: InstructGPT — Training Language Models to Follow Instructions with Human Feedback
**Paper:** Ouyang et al. (2022) | OpenAI  
**Full Title:** Training Language Models to Follow Instructions with Human Feedback

---

## 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Problem Domain** | AI Alignment — making LLMs follow user intent safely and helpfully |
| **Paper Type** | Experimental ML / Empirical + Algorithmic |
| **Core Contribution** | A three-stage RLHF pipeline (SFT → RM → PPO) that aligns GPT-3 with human preferences |
| **Key Idea** | Fine-tune a large LM using human preference rankings as a reward signal, not just next-token prediction loss |
| **Required Background** | Transformers, GPT-3, Supervised Fine-Tuning, Reinforcement Learning basics, PPO algorithm |
| **Primary Baseline** | GPT-3 (175B) and few-shot prompted GPT-3 |
| **Main Innovation Type** | Training methodology / Alignment technique |
| **Difficulty Level** | Intermediate — concepts are accessible, but training pipeline is complex in practice |
| **Reproducibility Level** | Low — requires proprietary human labeler data, OpenAI API prompts, and large compute |

---

## 1. Research Context & Core Problem

### Exact Problem Formulation

- Large language models (LLMs) are trained to predict the next word on internet text.
- The goal of "predict next token" is fundamentally different from "do what the user wants."
- This gap is called **misalignment**: the model learns to sound plausible, not to be helpful, honest, or harmless.
- Even a very large, capable model (GPT-3, 175B parameters) will make up facts, produce harmful content, or simply ignore instructions.

### Why the Problem Exists

- The pre-training objective (next-token prediction) is a **proxy objective** — optimizing it does not directly optimize for human intent.
- There is no signal in pre-training that tells the model: "this response is better than that response for the user."
- Scaling the model size alone does not fix this — a bigger model is still optimizing the same misaligned objective.

### Historical and Theoretical Gap

- Previous work on RLHF was confined to narrow tasks: summarization (Stiennon et al., 2020) and stylistic continuation (Ziegler et al., 2019).
- No one had applied RLHF at scale across a broad, real-world distribution of tasks (open generation, QA, code, brainstorming, etc.).
- Publicly available instruction-following datasets (FLAN, T0) covered primarily structured NLP tasks, not the diverse open-ended prompts real users submit.

### Limitations of Previous Approaches

| Approach | Limitation |
|---|---|
| Pre-training only (GPT-3) | Misaligned with user intent; hallucination; toxicity |
| Few-shot prompting | Brittle; requires careful prompt design; no safety guarantee |
| Supervised fine-tuning (SFT) on demonstrations only | Better, but still misses nuanced human preference; does not rank outputs |
| FLAN / T0 instruction tuning | Only covers structured tasks; low diversity; does not generalize to open-ended use |

### Contribution Category

- **Algorithmic** — new training pipeline (SFT + RM + PPO)
- **Empirical** — large-scale human evaluation across diverse real-world prompts
- **System Design** — data collection pipeline, labeler management, prompt distribution design

### Why This Paper Matters

InstructGPT demonstrates that a model with **100x fewer parameters** can be more useful than a much larger one when trained with the right feedback signal. It establishes RLHF as the dominant alignment technique and became the direct predecessor of ChatGPT, GPT-4, and all modern aligned LLMs.

### Remaining Open Problems

- The reward model reflects the preferences of a **specific group** of labelers — not universal human values.
- Aligning to one group can misalign to another (cultural, demographic, linguistic variation).
- Bias reduction does not improve with RLHF — the model becomes more certain, not less biased.
- RLHF requires expensive human annotation — hard to scale or democratize.
- The "alignment tax" (performance drop on NLP benchmarks) is not fully solved.
- Reward hacking: the model can exploit the reward model without genuinely being better.

---

## 2. Minimum Background Concepts

### 2.1 Language Model Pre-training

- **Plain definition:** Train a neural network (GPT-3) on massive internet text to predict the next word.
- **Role in paper:** The starting point — InstructGPT starts from a pre-trained GPT-3 and refines it.
- **Why authors needed it:** Pre-training gives the model knowledge of language, facts, and reasoning; alignment training then steers its behavior.

### 2.2 Supervised Fine-Tuning (SFT)

- **Plain definition:** Take the pre-trained model and train it further on curated (input, desired output) pairs, using standard cross-entropy loss.
- **Role in paper:** Step 1 of the RLHF pipeline — creates a well-behaved starting policy.
- **Why authors needed it:** SFT produces a model that already follows instructions reasonably; it is the base for subsequent RL fine-tuning.

### 2.3 Reward Model (RM)

- **Plain definition:** A neural network that takes a prompt and a model response, and outputs a scalar score indicating how good that response is according to human preferences.
- **Role in paper:** Step 2 — trained on human preference rankings; acts as the optimization target for PPO.
- **Why authors needed it:** Human feedback cannot be used directly in RL because humans cannot score millions of samples. The RM generalizes human preferences to new examples.

### 2.4 Proximal Policy Optimization (PPO)

- **Plain definition:** A reinforcement learning algorithm that updates a model (the "policy") to maximize a reward signal while not changing the model too drastically in any one step.
- **Role in paper:** Step 3 — fine-tunes the SFT model using the RM's score as the reward signal.
- **Why authors needed it:** Standard RL algorithms are unstable on large language models; PPO's clipping mechanism provides stable, conservative updates.

### 2.5 KL Divergence Penalty

- **Plain definition:** A mathematical measure of how different two probability distributions are.
- **Role in paper:** Added to the PPO objective to prevent the model from drifting too far from the original SFT model — prevents reward hacking.
- **Why authors needed it:** Without it, the model would learn to produce high-scoring outputs that exploit the RM's weaknesses rather than genuinely improving.

### 2.6 Alignment Tax

- **Plain definition:** The drop in performance on standard NLP benchmarks (question answering, reading comprehension, etc.) that occurs after RLHF fine-tuning.
- **Role in paper:** An identified problem that the PPO-ptx variant addresses.
- **Why authors needed it:** Shows that alignment and capability can conflict; motivates the pretraining data mixing strategy.

---

## 3. Mathematical / Theoretical Understanding Layer

### 3.1 Reward Model Training Objective

**Intuition:** Given two model outputs for the same prompt, where a human said "output A is better than output B," train the RM to assign higher score to A than B.

**Formulation:**

```
Loss(RM) = -E [ log( σ( r(x, y_w) - r(x, y_l) ) ) ]
```

| Symbol | Meaning |
|---|---|
| `r(x, y)` | RM score for prompt x and response y |
| `y_w` | The "winner" (preferred) response in a comparison pair |
| `y_l` | The "loser" (less preferred) response |
| `σ` | Sigmoid function — converts score difference to probability |
| `E[...]` | Expectation over all comparison pairs in the dataset |

**Practical interpretation:** The RM learns to rank responses the way human labelers would. It is a binary classifier extended to a scalar scorer via Bradley-Terry model.

**Assumptions:**
- Human preferences are consistent and transitive (if A > B and B > C, then A > C).
- A single scalar score captures the full complexity of response quality — this is a strong simplification.

**Limitation:** Human preferences are noisy; different labelers may rank the same pair differently, introducing noise into the RM training signal.

### 3.2 PPO Fine-Tuning Objective

**Intuition:** Maximize the reward from the RM while staying close to the original SFT model (via KL penalty) and preserving pre-training capabilities (via pretraining mix).

**Formulation:**

```
Objective(θ) = E_x [ r_θ(x, y) - β * KL( π_θ(y|x) || π_SFT(y|x) ) ] + γ * E_x_pretrain [ log π_θ(x) ]
```

| Symbol | Meaning |
|---|---|
| `θ` | Parameters of the model being trained (policy) |
| `r_θ(x, y)` | Reward score from the RM for prompt x, response y |
| `β` | Weight on the KL penalty — controls how far the model can stray from SFT |
| `KL(...)` | Kullback-Leibler divergence — measures distribution shift from SFT baseline |
| `π_θ` | Current policy (model being updated) |
| `π_SFT` | Fixed SFT model used as reference |
| `γ` | Weight on the pretraining log-likelihood term |
| `x_pretrain` | Samples from the original GPT-3 pre-training distribution |

**Practical interpretation:**
- The KL term prevents the model from becoming an entirely different model that exploits RM weaknesses.
- The pretraining term (γ) prevents the "alignment tax" — it forces the model to retain its original language modeling ability on standard tasks.
- The model trained WITHOUT the pretraining term is called **PPO**; the model trained WITH it is called **PPO-ptx**.

### Mathematical Insight Box

> **Key researcher insight:** The RM is a compressed, differentiable proxy of human judgment. The entire alignment quality depends on how well the RM captures true preferences. A reward model that rewards verbosity or sycophancy will produce a sycophantic model — the RM is both the solution and the bottleneck.

---

## 4. Proposed Method / Framework

### Overall Pipeline (3 Steps)

```
[Pre-trained GPT-3]
        ↓
[Step 1: SFT — Supervised Fine-Tuning]
        ↓
[SFT Model — follows instructions but not ranked]
        ↓
[Step 2: RM Training — human preference comparisons]
        ↓
[Reward Model — scalar scorer of response quality]
        ↓
[Step 3: PPO Fine-Tuning using RM as reward]
        ↓
[InstructGPT — aligned policy]
```

---

### Step 1: Supervised Fine-Tuning (SFT)

**What happens:**
- Human labelers write demonstrations of ideal responses to diverse prompts.
- Prompts include: generation tasks, open QA, brainstorming, summarization, rewriting, classification, chat, coding.
- GPT-3 is fine-tuned on these (prompt, ideal response) pairs using standard supervised learning.
- Dataset: ~13,000 training prompts.

**Why authors did this:**
- SFT provides a strong behavioral baseline — the model already knows how to follow instructions.
- It creates a stable starting policy for the subsequent RL stage.

**Weakness of this step:**
- SFT only teaches the model to imitate demonstrations; it cannot distinguish between "good" and "better" responses.
- Labeler demonstrations may be inconsistent or biased.
- Coverage is limited — only prompts where humans wrote ideal responses.

**Research opportunity:**
- Replace human demonstrations with AI-generated demonstrations filtered by quality — reduces annotation cost.
- Use active learning to select maximally informative prompts for demonstration collection.

---

### Step 2: Reward Model (RM) Training

**What happens:**
- The SFT model generates multiple responses for the same prompt (typically 4–9 outputs).
- Human labelers rank these outputs from best to worst.
- The RM is initialized from the SFT model (same architecture, with a linear head replacing the language model head to output a scalar).
- The RM is trained on the ranking data using the pairwise comparison loss (Section 3.1).
- Dataset: ~33,000 training prompts with comparison rankings.

**Why authors did this:**
- Ranking is much easier for humans than writing ideal responses — labelers can judge quality without generating it.
- Rankings generalize the preference signal across many (prompt, response) pairs.

**Weakness of this step:**
- RM can be exploited ("reward hacking") — the policy finds outputs that score high without being genuinely good.
- RM is trained on a fixed snapshot of labeler preferences — does not update as the policy improves.
- A single scalar score conflates many dimensions (helpfulness, harmlessness, honesty).

**Research opportunity:**
- Train a multi-dimensional reward model that separately scores helpfulness, truthfulness, and harmlessness.
- Use iterative RM updating (constitutional AI, debate) to reduce reward hacking.

---

### Step 3: PPO Fine-Tuning

**What happens:**
- The SFT model is treated as the initial policy.
- For each prompt, the policy generates a response; the RM scores it.
- PPO updates the policy parameters to maximize the RM score, constrained by a KL penalty from the SFT baseline.
- Optionally, pre-training data is mixed in (PPO-ptx) to prevent the alignment tax.
- The loop can iterate: collect more comparison data → retrain RM → retrain policy.
- Dataset: ~31,000 prompts (no human labels needed at this stage, just prompts).

**Why authors did this:**
- RL allows the model to explore and find better responses beyond what was in the SFT demonstrations.
- PPO's stability properties make it suitable for large models.

**Weakness of this step:**
- PPO is computationally expensive — requires running multiple forward and backward passes.
- KL coefficient β is a hyperparameter that requires careful tuning.
- If the RM is imperfect, PPO will optimize toward RM flaws.

**Research opportunity:**
- Replace PPO with more sample-efficient RL algorithms (e.g., DPO — Direct Preference Optimization, which eliminates the RM entirely).
- Study the effect of varying the KL coefficient schedule rather than a fixed β.

---

### Pseudocode-Style Algorithm Summary

```
INPUT: Pre-trained GPT-3 (π_base)

// STEP 1: SFT
SFT_data = collect {(prompt, human_demonstration)} pairs   [~13k]
π_SFT = fine_tune(π_base, SFT_data, loss=cross_entropy)

// STEP 2: RM Training
comparison_data = collect {(prompt, ranked_outputs)} from labelers  [~33k]
RM = initialize from π_SFT with scalar head
RM = train(RM, comparison_data, loss=pairwise_ranking_loss)

// STEP 3: PPO
ppo_prompts = collect API prompts  [~31k]
π_InstructGPT = π_SFT   // initialize from SFT
for each training step:
    sample prompt x from ppo_prompts
    generate response y ~ π_InstructGPT(x)
    reward = RM(x, y) - β * KL(π_InstructGPT(y|x) || π_SFT(y|x))
    optionally: add γ * log π_InstructGPT(x_pretrain)  // PPO-ptx
    update π_InstructGPT using PPO

OUTPUT: InstructGPT (aligned policy)
```

---

## 5. Experimental Setup / Evaluation Design

### Dataset Characteristics

| Dataset | Size | Purpose |
|---|---|---|
| SFT demonstrations | ~13,000 prompts | Train supervised baseline |
| RM comparison data | ~33,000 prompts | Train reward model |
| PPO prompts | ~31,000 prompts | RLHF fine-tuning (no labels) |

- **Prompt sources:** OpenAI API Playground (real user prompts) + labeler-written prompts.
- **Language:** >96% English.
- **Task distribution:** Generation (45.6%), Open QA (12.4%), Brainstorming (11.2%), Chat (8.4%), Rewrite (6.6%), Summarization (4.2%), Classification (3.5%), Others.
- **PII filtering:** All training prompts filtered for personally identifiable information.

### Human Labeler Setup

- ~40 contractors hired via Upwork and Scale AI.
- Selected via screening test measuring sensitivity to harmful content and demographic awareness.
- Separate "held-out" labelers used for evaluation (not involved in training data collection).

### Model Sizes Trained

- 1.3B, 6B, and 175B parameter models — all using GPT-3 architecture.

### Metrics Used and Why

| Metric | Purpose |
|---|---|
| Labeler preference rate (win rate vs. 175B SFT baseline) | Primary metric — captures holistic output quality |
| 1–7 Likert scale rating | Absolute quality score per output |
| TruthfulQA accuracy | Measures truthfulness / hallucination reduction |
| RealToxicityPrompts (Perspective API + human eval) | Measures toxicity reduction |
| Winogender / CrowS-Pairs entropy | Measures gender/demographic bias |
| Zero-shot NLP benchmark performance | Measures alignment tax (capability regression) |

### Baseline Selection Logic

- **GPT-3 (raw):** Shows the starting point without any alignment.
- **GPT-3 (few-shot prompted):** Tests whether prompting alone can close the gap.
- **SFT model:** Shows the gain from supervised demonstration tuning alone.
- **FLAN / T0:** Tests whether public instruction-following datasets achieve similar results.

### Experimental Reliability Analysis

**What is trustworthy:**
- Human preference evaluations use held-out labelers — reduces overfitting to training labeler preferences.
- Cross-validation of RM (5-fold) confirms generalization (72.4% train vs. 69.6% held-out accuracy).
- Multiple model sizes tested — trends are consistent across sizes.

**What is questionable:**
- Evaluation distribution is the same as training distribution — unclear how models perform on truly out-of-distribution prompts.
- Human labelers (40 people) are not a representative sample of global users.
- Toxicity evaluation uses Perspective API which has known demographic biases.
- Bias evaluation results are ambiguous — more confident outputs are not necessarily less biased.

---

## 6. Results & Findings Interpretation

### Main Outcomes

1. **Preference over GPT-3:** 1.3B InstructGPT outputs preferred over 175B GPT-3 outputs — a 100x parameter efficiency gain from alignment training.
2. **Quantitative win rates:** 175B InstructGPT preferred over 175B GPT-3 in 85±3% of comparisons; preferred over few-shot GPT-3 in 71±4% of comparisons.
3. **Truthfulness:** ~2x improvement on TruthfulQA; significantly less hallucination on closed-domain tasks.
4. **Toxicity:** Reduced toxicity when given respectful system prompts; no improvement without explicit prompting; becomes MORE toxic than GPT-3 when explicitly instructed to be toxic.
5. **Bias:** No improvement over GPT-3; instructed models show lower entropy (higher certainty) regardless of whether output is stereotypical.
6. **Alignment tax:** PPO without pretraining mix hurts NLP benchmark performance; PPO-ptx mitigates this at the cost of slight preference reduction.

### Performance Trends

- SFT > few-shot GPT-3 > raw GPT-3 (quality ladder).
- PPO > SFT in preference ratings.
- PPO-ptx ≈ PPO in preference but better on NLP benchmarks.
- Larger models do not always win on preference — 1.3B InstructGPT beats 175B GPT-3.

### Failure Cases

- Bias is not reduced — RLHF does not fix demographic bias, possibly worsens certainty.
- Toxicity reduction is prompt-dependent — without explicit safety instructions, InstructGPT is not reliably safer.
- Simple mistakes remain — the model still makes errors on basic reasoning tasks.
- The 1.3B PPO-ptx model performs slightly worse on TruthfulQA than same-size GPT-3 — small models may be more sensitive to RLHF instability.

### Publishability Strength Check

| Result | Strength |
|---|---|
| 100x parameter efficiency in preference | Strong — consistent across labeler groups and model sizes |
| Truthfulness improvement | Moderate — TruthfulQA is one benchmark; real-world hallucination harder to measure |
| Toxicity reduction | Weak standalone — highly prompt-dependent |
| Bias findings | Inconclusive — needs stronger methodology |
| Alignment tax mitigation via pretraining mix | Moderate — practical contribution but simple technique |

---

## 7. Strengths – Weaknesses – Assumptions

### Technical Strengths

| Strength | Description |
|---|---|
| Real-world evaluation | Uses actual API prompts, not academic datasets |
| Held-out labeler validation | Confirms model generalizes beyond training labelers |
| Scale of experiment | Three model sizes, multiple baselines, diverse task types |
| RLHF pipeline is complete and reproducible in principle | All three steps clearly described |
| Alignment tax mitigation (PPO-ptx) | Practical solution to capability regression |

### Explicit Weaknesses

| Weakness | Description |
|---|---|
| Non-diverse labeler pool | ~40 English-speaking contractors ≠ global human values |
| Reward hacking risk | RM is static; policy may learn to exploit it |
| Toxicity reduction is prompt-dependent | Not a robust safety guarantee |
| No bias reduction | RLHF does not fix demographic bias |
| High compute cost | Not reproducible by most researchers |
| Single scalar reward | Conflates multiple dimensions of quality |

### Hidden Assumptions

| Assumption | Why it is risky |
|---|---|
| Labeler preferences represent user intent | Labelers are a proxy, not actual users |
| RM scores generalize to all tasks | RM trained on API distribution; may fail on novel task types |
| KL penalty prevents reward hacking | Only slows it; does not eliminate it |
| Pairwise comparisons capture quality ordering | Assumes transitivity and consistency in human judgments |
| Pretraining data mix fixes alignment tax | May not hold for all task types or model architectures |

---

## 8. Weakness → Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| Non-diverse labeler pool | Practical constraint of one lab's annotation budget | Cross-cultural RLHF: train separate RMs per demographic group | Federated RL, multi-stakeholder reward aggregation |
| Single scalar reward conflates quality dimensions | Simplification for tractability | Multi-objective RLHF with Pareto-optimal policies | Multi-head reward model, constrained RL |
| Static RM becomes stale as policy improves | No RM retraining during PPO | Online RL with iterative RM updates | Constitutional AI, iterative DPO |
| Reward hacking | RM is imperfect; PPO exploits flaws | Reward model robustness evaluation | Adversarial probing of RM, uncertainty-aware RM |
| Bias not reduced | RLHF optimizes preference, not fairness | Fairness-constrained RLHF | Constrained optimization with fairness regularizer |
| High cost of human annotation | Requires many human comparisons | LLM-as-annotator (replace humans with AI) | RLAIF (RL from AI Feedback), preference distillation |
| Alignment tax | PPO moves away from pre-training distribution | Better regularization during RL | Elastic weight consolidation, task-specific KL schedules |
| Prompt-dependent toxicity | RLHF teaches conditional safety, not unconditional | Unconditional safety alignment | Constitutional AI, red-teaming during training |

---

## 9. Novel Contribution Extraction

### From the Paper's Contribution

"We propose RLHF-based fine-tuning (SFT → RM → PPO) that improves GPT-3 instruction-following by replacing the next-token prediction objective with a human preference ranking signal."

### Possible Novel Claim Templates

1. "We propose **multi-objective reward modeling** that improves alignment by **separately scoring helpfulness, harmlessness, and honesty** while constructing Pareto-optimal responses."

2. "We propose **cross-cultural RLHF** that improves value alignment by **collecting preference data from demographically diverse annotator groups** and aggregating rewards via multi-stakeholder optimization."

3. "We propose **online iterative reward model refinement** that improves RLHF stability by **continuously retraining the RM on policy-generated outputs** rather than fixing it before PPO begins."

4. "We propose **uncertainty-aware PPO** that improves reward hacking resistance by **down-weighting reward signals in regions of high RM uncertainty** during policy optimization."

5. "We propose **RLHF with fairness constraints** that improves demographic equity by **adding bias penalty terms to the PPO objective** measured via counterfactual probing datasets."

---

## 10. Future Research Expansion Map

### Author-Suggested Future Work (from paper)

- Reduce the alignment tax more effectively.
- Handle cases where helpfulness and harmlessness conflict.
- Align to diverse groups of users, not just labelers.
- Evaluate real-world downstream harms of InstructGPT outputs.
- Study how InstructGPT performs on non-English languages.

### Missing Directions (not mentioned by authors)

- **Continual alignment:** How to update alignment as user preferences and societal norms change over time.
- **Alignment interpretability:** What internal representations change between GPT-3 and InstructGPT?
- **Label efficiency:** How few human comparisons are needed to achieve good alignment?
- **Alignment transfer:** Can an RM trained on one language/task transfer to another?
- **Adversarial alignment:** What prompts can jailbreak InstructGPT, and how to make it robust?

### Modern Extensions

- **DPO (Direct Preference Optimization):** Eliminates the RM entirely; directly fine-tunes on preference pairs — simpler and more stable than PPO.
- **RLAIF (RL from AI Feedback):** Uses another LLM (Claude, GPT-4) as the labeler instead of humans — scales annotation cheaply.
- **Constitutional AI (Anthropic):** Uses a set of principles (a "constitution") to guide RM training — makes alignment rules explicit.
- **Reward Model Ensembles:** Train multiple RMs and use agreement as a confidence signal — reduces reward hacking.

### Cross-Domain Combinations

- **RLHF + Retrieval-Augmented Generation:** Align a RAG system where the reward accounts for both helpfulness and factual accuracy of retrieved content.
- **RLHF + Code Generation:** Use test execution results (pass/fail) as objective reward to complement human preference.
- **RLHF + Robotics:** Apply the same pipeline to physical agent behavior alignment (already done partially in OpenAI's robotics work).

---

## 11. How to Write a New Paper from This Work

### Reusable Elements

- **Evaluation style:** Win-rate comparison against a fixed baseline (here: 175B SFT model) is a clean, interpretable metric — adopt it.
- **Three-stage pipeline structure:** SFT → RM → RL is now standard; papers can modify any one stage as their contribution.
- **Human evaluation with held-out labelers:** Validate that results are not labeler-specific — use separate train/eval annotator pools.
- **Dataset composition table:** Reporting the distribution of task types shows evaluators you understand your data.

### What Must NOT Be Copied

- The specific prompt dataset (proprietary OpenAI API data).
- The specific labeler instructions (Appendix B of the paper — copyrighted).
- Claimed win-rates as baselines — your setup will differ.
- The RM architecture details without citation.

### How to Design a Novel Extension

**Step 1:** Pick ONE weakness from Section 8 (e.g., "single scalar reward").  
**Step 2:** Design a minimal modification to the pipeline that addresses it (e.g., multi-head RM).  
**Step 3:** Use an open-source LLM (Llama, Mistral) instead of GPT-3 to make it reproducible.  
**Step 4:** Compare against: (a) SFT only, (b) standard PPO, (c) your modified method.  
**Step 5:** Measure improvement on at least two dimensions (preference + bias, preference + truthfulness, etc.).

### Minimum Publishable Contribution Checklist

- [ ] Clear formulation of ONE limitation in existing RLHF work
- [ ] A method that directly addresses that limitation
- [ ] Human preference evaluation (win rate against SFT or PPO baseline)
- [ ] At least one automatic metric beyond preference (truthfulness, toxicity, bias)
- [ ] Ablation study showing each component of the proposed method contributes
- [ ] Results on at least two model sizes to check generalization
- [ ] Discussion of limitations and failure modes of the proposed approach

---

## 12. Complete Paper Writing Template

### Abstract
- **Purpose:** Summarize problem, method, and key result in 150–250 words.
- **Include:** (1) the gap in existing methods, (2) what you propose, (3) the most impressive quantitative result, (4) one broader implication.
- **Common mistake:** Describing the method in too much detail; omitting the quantitative result.
- **Reviewer expectation:** Should be able to judge novelty and contribution from the abstract alone.

### Introduction
- **Purpose:** Motivate the problem, summarize contributions, and preview results.
- **Include:** (1) Why current LLMs fail at X, (2) your proposed fix, (3) bulleted list of contributions, (4) high-level preview of results.
- **Common mistake:** Over-reviewing related work here — save it for Section 2.
- **Reviewer expectation:** Contributions must be clearly listed and falsifiable.

### Related Work
- **Purpose:** Position your work relative to prior art.
- **Include:** RLHF prior work (Christiano 2017, Stiennon 2020), instruction tuning (FLAN, T0), alignment approaches (Constitutional AI, DPO), and reward modeling.
- **Common mistake:** Listing papers without explaining how yours differs.
- **Reviewer expectation:** Show you know the field; show your gap is real.

### Method
- **Purpose:** Explain the full pipeline clearly.
- **Include:** (1) Formal problem setup, (2) step-by-step description with equations, (3) design choices and why alternatives were rejected, (4) pseudocode or diagram.
- **Common mistake:** Missing the "why" behind design decisions — reviewers want motivation, not just description.
- **Reviewer expectation:** Enough detail to reproduce the core method.

### Theory (if applicable)
- **Purpose:** Provide mathematical justification for why the method works.
- **Include:** Reward model objective, PPO objective with KL term, any convergence guarantees or bounds.
- **Common mistake:** Presenting proofs without intuition — always explain what a theorem means in plain language.

### Experiments
- **Purpose:** Empirically validate the method.
- **Include:** (1) Dataset details, (2) baselines and why they were chosen, (3) metrics and why they capture the right thing, (4) main results table/figure, (5) ablation study.
- **Common mistake:** Only reporting when your method wins — show failure cases too.
- **Reviewer expectation:** Baselines must be strong and fairly tuned; evaluation must match real-world use.

### Discussion
- **Purpose:** Interpret results beyond the numbers.
- **Include:** What the results mean for the field, unexpected findings, connections to broader alignment challenges.
- **Common mistake:** Repeating numbers already in the results section.

### Limitations
- **Purpose:** Honestly acknowledge what the paper does not solve.
- **Include:** Scope constraints (e.g., English-only), known failure modes, assumptions that may not hold.
- **Common mistake:** Omitting this section — reviewers will add it in their critique if you don't include it.
- **Reviewer expectation:** Shows intellectual honesty and scopes the contribution properly.

### Conclusion
- **Purpose:** Summarize and state impact in 100–150 words.
- **Include:** One sentence per contribution + one forward-looking sentence.
- **Common mistake:** Introducing new claims in the conclusion.

### References
- Cite all papers mentioned; use consistent citation format (NeurIPS/ACL/ICLR style).
- Include: Christiano 2017, Stiennon 2020, Brown 2020 (GPT-3), Schulman 2017 (PPO), Wei 2021 (FLAN), Sanh 2021 (T0).

---

## 13. Publication Strategy Guide

### Suitable Venue Types

| Venue Type | Examples | Why Suitable |
|---|---|---|
| Top ML conferences | NeurIPS, ICML, ICLR | Core audience for RLHF and alignment work |
| NLP conferences | ACL, EMNLP, NAACL | Instruction-following and language model alignment |
| AI Safety venues | AAAI, SafeAI workshop | Alignment and safety framing |
| Human-Computer Interaction | FAccT, CSCW | If focus is on labeler diversity or fairness |

### Required Baseline Expectations

- Must include: SFT-only baseline, standard PPO (if proposing an improvement), at least one public instruction-tuning model (FLAN or T0).
- Should include: DPO (now the standard comparison) for any new RLHF variant.
- Must report both human evaluation AND at least one automatic metric.

### Experimental Rigor Level

- Minimum: 1 model size, 100+ human evaluations, 1 held-out evaluator group.
- Preferred: 2+ model sizes, 1000+ human evaluations, automatic metric corroboration.
- Gold standard: Labeler agreement statistics, cross-validation of reward model, ablations on all components.

### Common Rejection Reasons

1. "Baselines are weak or not fairly tuned."
2. "Human evaluation sample size is too small to be significant."
3. "Novelty is incremental without sufficient ablation justification."
4. "Claims about safety are not adequately validated."
5. "Results only shown on one model size — may not generalize."
6. "No comparison to DPO or Constitutional AI."

### Increment Needed for Acceptance

- **Workshop:** 1 clear improvement over PPO baseline with human eval.
- **Conference (EMNLP/NAACL):** 2+ improvements, held-out labeler eval, ablations.
- **Top conference (NeurIPS/ICLR/ACL):** Novel framework, 3+ baselines, multiple model sizes, theoretical insight or strong empirical finding.

---

## 14. Researcher Quick Reference Tables

### Key Terminology Table

| Term | Simple Meaning |
|---|---|
| RLHF | Training using human preference rankings as reward signal |
| SFT | Supervised fine-tuning on human-written demonstrations |
| RM (Reward Model) | Neural net that scores response quality like a human would |
| PPO | RL algorithm that makes stable, conservative policy updates |
| PPO-ptx | PPO variant that mixes in pre-training data to prevent alignment tax |
| Alignment tax | Drop in standard NLP benchmark scores after RLHF fine-tuning |
| KL divergence | Measure of how far the current model drifts from the SFT baseline |
| Reward hacking | Model exploiting RM weaknesses instead of genuinely improving |
| Alignment | Making model behavior match user intent: helpful, honest, harmless |
| Win rate | Fraction of comparisons where model A is preferred over model B |

### Important Equations Summary

| Equation | Purpose |
|---|---|
| `Loss_RM = -E[log σ(r(x,y_w) - r(x,y_l))]` | Train RM to prefer winner over loser |
| `Obj_PPO = E[r_θ - β·KL(π_θ || π_SFT)]` | Maximize reward while staying close to SFT |
| `Obj_PPO-ptx = Obj_PPO + γ·E[log π_θ(x_pretrain)]` | Add pretraining term to prevent alignment tax |

### Parameter Meaning Table

| Parameter | Default Behavior | Effect of Increasing |
|---|---|---|
| β (KL weight) | Moderate constraint | Slows policy from diverging; reduces reward hacking risk |
| γ (pretraining weight) | Zero (standard PPO) | Reduces alignment tax; may slightly reduce preference gains |
| RM rank comparisons | 4–9 outputs per prompt | More comparisons = better RM signal; higher annotation cost |
| SFT epochs | Standard fine-tuning | Overtraining may reduce diversity of RL exploration |

### Algorithm Flow Summary

| Stage | Input | Output | Objective |
|---|---|---|---|
| SFT | Prompts + human demonstrations | Fine-tuned policy (π_SFT) | Cross-entropy on demonstrations |
| RM Training | Prompts + ranked output pairs | Reward model r(x,y) | Pairwise ranking loss |
| PPO Fine-tuning | Prompts (no labels) | InstructGPT (π_θ) | Maximize RM reward − KL penalty |

---

## 15. One-Page Master Summary Card

### Problem
Large language models are trained to predict text, not to follow user instructions. They hallucinate, produce toxic content, and ignore user intent. This is the **alignment problem**.

### Idea
Instead of asking humans to write better training data, ask them to **rank** which of several model outputs is better — this is easier, scalable, and captures nuanced preferences.

### Method
Three stages:
1. **SFT** — fine-tune GPT-3 on human-written demonstrations (~13k prompts).
2. **RM** — train a reward model to predict human preference from rankings (~33k prompt pairs).
3. **PPO** — use the RM as a reward signal to further optimize the policy via RL (~31k prompts), with KL penalty to prevent reward hacking.

### Results
- 1.3B InstructGPT is preferred over 175B GPT-3 (100x fewer parameters).
- 175B InstructGPT preferred over 175B GPT-3 in 85±3% of comparisons.
- ~2x improvement in truthfulness (TruthfulQA).
- Slight toxicity reduction (prompt-dependent).
- No bias improvement.
- Alignment tax mitigated by PPO-ptx (pretraining data mixing).

### Weakness
- Labeler pool is not diverse — represents a narrow slice of global preferences.
- Bias is not reduced; certainty increases regardless of fairness.
- Toxicity reduction requires explicit safety prompting.
- High annotation and compute cost — not reproducible by most researchers.

### Research Opportunity
- Multi-dimensional RM (separate scores for helpfulness, truthfulness, harmlessness).
- Cross-cultural RLHF for diverse demographic alignment.
- Online iterative RM updates to prevent reward hacking.
- LLM-as-annotator (RLAIF) to reduce human annotation cost.

### Publishable Extension
Design a **multi-objective RLHF** system that trains a reward model with separate heads for helpfulness, harmlessness, and honesty. Optimize the policy using Pareto-optimal reward aggregation. Compare against standard InstructGPT/PPO on preference, TruthfulQA, and CrowS-Pairs. Show that safety-capability trade-off is improved without an alignment tax.
