# ScienceWorld: Is Your Agent Smarter than a 5th Grader?
### Research Companion & Publication Blueprint
**Paper:** Wang et al., 2022 | University of Arizona, Microsoft Research, Allen Institute for AI

---

# 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Problem Domain** | Scientific Reasoning + Grounded Agent Intelligence |
| **Paper Type** | Systems / Engineering + Experimental / Empirical |
| **Core Contribution** | A new interactive text-based benchmark environment for testing agents on elementary science tasks requiring real experimental reasoning |
| **Key Idea** | Language models that ace science QA tests fail when they must actually *perform* a science experiment in a simulated world — showing that retrieval ≠ reasoning |
| **Required Background** | Reinforcement Learning basics, NLP/transformers, text-based game environments, POMDP framework |
| **Primary Baseline** | DRRN (Deep Reinforcement Relevance Network) — small RL agent that outperforms all others |
| **Main Innovation Type** | Benchmark creation + empirical evaluation of agent types |
| **Difficulty Level** | Intermediate (systems-heavy, but concepts are accessible) |
| **Reproducibility Level** | High — code and environment publicly available on GitHub |

---

# 1. Research Context & Core Problem

## Exact Problem Formulation

Current AI models are very good at answering science questions in text form. For example, if you ask: "What happens to ice when placed on a stove?", a language model trained on enough data will say "it melts." But does the model actually *understand* melting, or is it just pattern-matching from training data?

This paper asks a sharper question:
> **Can an AI agent demonstrate scientific understanding by performing the experiment, not just answering a question about it?**

## Why the Problem Exists

- Large language models (LLMs) have surpassed human performance on standardized science benchmarks (like ARC).
- But "answering" a question and "doing" the underlying experiment are fundamentally different cognitive tasks.
- Existing benchmarks are all static: they give a question, expect a text answer. None require the agent to *act* in a simulated world to verify scientific knowledge.
- This creates a dangerous illusion — models appear to "know science" when they are actually retrieving memorized associations.

## Historical / Theoretical Gap

- Interactive text environments existed before (TextWorld, Zork, AlfWorld) but were not designed around science curriculum.
- Science-domain QA benchmarks (ARC, OpenBookQA) do not test grounded procedural reasoning.
- No prior work combined: (a) physics simulation + (b) multi-step agent tasks + (c) elementary science curriculum.

## Limitations of Previous Approaches

| Prior Approach | What It Lacks |
|---|---|
| Science QA benchmarks (ARC, OpenBookQA) | No grounded action; static text in/out |
| TextWorld / Zork-based games | No science-specific simulation engines |
| TextLabs (chemistry protocols only) | Narrow scope; no thermodynamics/genetics/physics |
| AlfWorld (embodied + text) | Household pick-and-place tasks, not science reasoning |

## Contribution Category

- **System design**: Built a novel simulator (40k lines Scala + Python API) with 7 physics engines
- **Empirical insight**: Showed that online RL agents (small, interactive) outperform large offline-trained LLMs on grounded science tasks

---

## Why This Paper Matters

This paper challenges the assumption that **scale = understanding**. An 11-billion-parameter model pre-trained on millions of science examples fails at 5th-grade experiments that a small 1.5M-parameter RL agent can partially learn. This forces the community to separate:
- **Declarative knowledge** (knowing facts): LLMs excel here
- **Procedural knowledge** (doing experiments): LLMs fail here

---

## Remaining Open Problems

- How to efficiently combine declarative LLM knowledge with procedural RL learning?
- Can agents generalize science procedures across domains (e.g., learn "heating" in one context and apply it to unknown substances)?
- How do language model capabilities scale when given interactive training rather than static training?
- What architectural changes enable agents to navigate and reason simultaneously?
- How to automatically evaluate the correctness of multi-step procedural explanations without a simulator?

---

# 2. Minimum Background Concepts

## 2.1 Interactive Text Environments

- **What they are**: Software simulations where an agent reads a text description of a world and types actions (like "pick up fork", "go to kitchen").
- **Role in paper**: ScienceWorld is one such environment, but with physics engines instead of simple rule-based logic.
- **Why needed**: Provides a controlled, grounded testbed for measuring reasoning.

## 2.2 POMDP (Partially Observable Markov Decision Process)

- **Plain definition**: A mathematical model for decision-making when the agent can't see the full world — only partial observations (like reading text about a room, not seeing the whole map).
- **Role in paper**: All agent models in this paper are trained to optimize a policy under POMDP assumptions.
- **Formal components used**:
  - `S` = all possible world states
  - `A` = available text actions
  - `O` = text observations the agent receives
  - `R` = reward function (how well the agent is doing)
  - `γ` = discount factor (how much future rewards matter)
  - **Goal**: learn `π(a | o)` — the best action to take given current observation

## 2.3 Reinforcement Learning (RL)

- **What it is**: Learning by trial and error — the agent takes actions, gets rewards, and updates its strategy.
- **Role in paper**: DRRN and KG-A2C are RL agents trained online (interactively inside the environment).
- **Why needed**: RL enables the agent to *explore* the environment, unlike static language model training.

## 2.4 Behavior Cloning (Imitation Learning)

- **What it is**: Learning from expert demonstrations — the model watches a correct solution and learns to copy it.
- **Role in paper**: The BC-T5 agent is trained on oracle (expert) action sequences, then tested in the environment.
- **Why needed**: Tests whether large pre-trained models can transfer expert demonstrations to new situations.

## 2.5 Decision Transformer

- **What it is**: A transformer model that treats RL as a sequence prediction problem — given a goal reward, predict the sequence of actions that achieves it.
- **Role in paper**: Authors created "Text Decision Transformer" (TDT), adapting this idea to text environments.
- **Why needed**: Tests if conditioning on return-to-go (future reward sum) improves planning.

## 2.6 Knowledge Graph + RL (KG-A2C)

- **What it is**: An RL agent that builds a structured graph of facts from text observations (e.g., "glass bottle contains water") to guide action selection.
- **Role in paper**: Compared against simpler and larger agents.

---

# 3. Mathematical / Theoretical Understanding Layer

## 3.1 POMDP Objective

The learning agents aim to maximize:

```
E[ Σ_t  γ^t · R(s_t, a_t) ]
```

**Variable meanings:**

| Symbol | Meaning |
|---|---|
| `E[...]` | Expected value (average over many episodes) |
| `Σ_t` | Sum over all timesteps t |
| `γ` | Discount factor: value between 0 and 1; controls how much future rewards count |
| `R(s_t, a_t)` | Reward received at step t for taking action `a_t` in state `s_t` |
| `s_t` | True environment state at time t (hidden from agent) |
| `a_t` | Action chosen by agent at time t |

**Intuition**: The agent wants to accumulate as many rewards as possible over time, with recent rewards being worth more than distant ones.

**Assumption**: The optimal policy depends only on the current observation (not full history), though full history is often included in practice.

## 3.2 Returns-to-Go (Decision Transformer)

```
R̂_t = Σ_{t'=t}^{T} r_{t'}
```

**Intuition**: Instead of predicting just the next action, the Decision Transformer predicts what action leads to the *maximum total future reward* by conditioning on the desired future reward sum. The model is asked: "Given that you want to earn reward `R̂`, what action should you take now?"

### Mathematical Insight Box
> The key insight is that if you tell the model what future reward you want and show it past context, it should be able to plan backward — selecting actions that are consistent with achieving that target. In ScienceWorld, this fails because the model has no grounded understanding of how actions actually change physical states.

## 3.3 Subgoal-Based Reward Shaping

To avoid **sparse reward** (agent gets no signal until the very end), each task provides 2–15 subgoals. The normalized task score:

```
Score ∈ [0, 1] = (subgoals completed + final goal) / max_achievable_score
```

**Why important**: Without subgoal rewards, agents rarely receive positive signal and learn nothing. Reward shaping is a critical design decision in benchmark creation.

---

# 4. Proposed Method / Framework (MOST IMPORTANT)

## 4.1 Overall: What ScienceWorld Is

ScienceWorld is an **interactive text environment** where agents must complete 30 elementary science experiments using text commands. It is:
- Written in 40,000 lines of **Scala** (for speed)
- Wrapped in a **Python API** for ML research
- Containing **10 interconnected locations** (kitchen, workshop, garden, etc.)
- Populated with **200+ types of objects**
- Supporting **25 high-level actions** (~200k possible action-object combinations per step)

## 4.2 Simulation Engines (The Core System Innovation)

### Thermodynamics Engine
- Every object has a temperature and thermal properties.
- Heat transfers between objects in the same container.
- Uses a simplified conductive heat model with a `thermal_conduction_coefficient`.
- Objects change phase (solid → liquid → gas) when temperature crosses their melting/boiling points.
- Combustion is modelled — objects can catch fire.
- Heat sources (stove, oven) and heat sinks (fridge, freezer) are modelled.

**Why authors did this**: Enables experiments like "boil water", "melt ice", "observe what happens when metal is heated."
**Weakness**: Simplified model — no convection within gases, no radiation, no latent heat precision.
**Research opportunity**: A higher-fidelity thermodynamics model could test more nuanced reasoning.

---

### Electricity Engine
- Models simple **series circuits** with batteries, wires, bulbs, motors.
- Each object has two terminals (polarized: anode/cathode; unpolarized: terminal 1/2).
- Non-electrical objects also have virtual terminals — so a metal fork can substitute for a wire.
- Agents can test conductivity by checking if a light bulb lights up.

**Why authors did this**: Enables tasks like "test if this unknown material conducts electricity."
**Weakness**: Only series circuits, no parallel circuits, no resistance or inductance modelling.

---

### Chemistry Engine
- Models specific chemical reactions: mixing substances in containers produces new substances.
- Covers: water reactions, rust, food reactions, paint mixing.
- Only elementary reactions are included.

**Weakness**: Combinatorial space of chemicals is not explored; arbitrary reactions not supported.

---

### Life Stages Engine
- Plants and animals progress through stages (seed → seedling → juvenile → adult → reproducing → dead).
- Stage progression depends on meeting needs (water, soil, light).
- Failing to meet needs causes death.

---

### Genetics Engine
- Genes are inherited using **Punnett square** logic.
- Plants have dominant/recessive alleles for traits (flower color, leaf size, seed shape).
- Pollination (via bee or manual action) triggers genetic inheritance.
- Enables Mendelian genetics experiments.

---

### Friction Engine
- Models gravity + friction on a 1-dimensional inclined plane.
- Objects slide down at a speed proportional to plane angle and friction coefficient.
- Agent reads positional descriptions to infer relative friction or angle.

---

### Container Engine
- Containers can be open (pots) or closeable (cupboards).
- Objects inside closed containers are hidden from the agent.
- Heat from internal objects can spread to container and surroundings.

---

## 4.3 The 30 Tasks Across 10 Topics

| Topic | # Tasks | Example Task |
|---|---|---|
| Changes of State (Matter) | 4 | Boil, melt, freeze, any state change |
| Temperature Measurement | 3 | Use thermometer, measure boiling point (known/unknown) |
| Electrical Circuits | 4 | Create circuit, renewable energy, test conductivity (known/unknown) |
| Object Classification | 4 | Find living thing, non-living thing, plant, animal |
| Plant Biology | 2 | Grow a plant, grow a fruit |
| Chemistry | 3 | Generic mixing, secondary paint colors, tertiary paint colors |
| Life Spans | 3 | Identify longest/shortest-lived animal |
| Life Stages | 2 | Identify plant life stages, animal life stages |
| Forces / Friction | 3 | Determine incline angle, known surfaces, unknown surfaces |
| Mendelian Genetics | 2 | Known plants, unknown plants |

**Critical design**: Tasks come in "known" and "unknown" pairs. In the known version, the agent can retrieve an answer from training data. In the unknown version (e.g., "unknown substance B"), the agent *must* perform the experiment.

## 4.4 Parametric Variation System

- Each subtask has 10–1,400 parametric variations.
- Variations change: target objects, agent start location, room contents.
- Train/Dev/Test split: 50% / 25% / 25%.
- Critical unseen variations (new substances, animals) are placed in Dev/Test.

**Why**: Forces agents to *generalize*, not memorize specific scenarios.

## 4.5 Oracle Agents

- 30 hand-coded oracle programs provide gold-standard action trajectories.
- Each represents one canonical solution (e.g., always uses stove to boil water, not campfire).
- Used to generate expert demonstrations for Behavior Cloning and Text Decision Transformer.

**Weakness**: Only canonical solutions — alternate valid paths are not covered, biasing imitation agents.

---

## 4.6 Simplified Pseudocode Flow (Agent Interaction Loop)

```
Initialize environment with task T and variation V
Agent receives: task description d, initial observation o_0

For each timestep t:
    Agent observes: current room description + inventory + previous action
    Agent produces: action a_t (text string)
    Environment:
        - checks if a_t is valid
        - updates simulation engines (thermodynamics, electricity, etc.)
        - returns new observation o_{t+1} and reward r_t
        - checks if subgoals or final goal are met
    If episode end (goal reached or max steps hit): stop
```

---

# 5. Experimental Setup / Evaluation Design

## 5.1 Dataset / Environment Characteristics

- 30 subtasks, 7,200 total parametric variations
- Tasks span 10 science topics from a real elementary school curriculum
- Train (50%) / Dev (25%) / Test (25%) splits — unseen objects in Dev/Test

## 5.2 Agents Evaluated

| Agent | Type | Parameters | Training Method |
|---|---|---|---|
| Random-Valid | Baseline | — | Random from valid actions |
| DRRN | RL (online) | 1.5M | Interactive RL in environment |
| KG-A2C | RL + Knowledge Graph | 5.5M | Interactive RL with graph state |
| CALM (GPT-2) | LM + RL (hybrid) | 131M* | Fine-tuned GPT-2 + RL re-ranking |
| BC-T5 (Macaw) | Imitation Learning | 11,000M | Fine-tuned on expert demos, zero-shot inference |
| TDT-T5 (Macaw) | Decision Transformer | 11,000M | Fine-tuned on expert demos with returns-to-go |

*131M includes GPT-2 generator; only 6.9M policy parameters updated in RL.

## 5.3 Training Protocol

- All RL models: 8 environment threads × 100,000 steps/thread
- Episodes reset on success, failure, or after 100 steps
- All models use identical experiment configurations where possible

## 5.4 Metrics

- **Normalized task score** ∈ [0, 1]: based on subgoals completed + final goal
- RL performance: **averaged over last 10% of evaluation episodes**
- T5 performance: **averaged over all test variations**
- RL results: **averaged over 5 random seeds** (80% of standard deviations < 0.05)

**Why subgoal-based scoring**: Allows partial credit — agents that make progress but don't fully complete get rewarded, giving a more nuanced performance signal than binary success/failure.

## 5.5 Valid Action Detection Aid

- All agents except CALM and Random were given access to ScienceWorld's **valid action detection system** at test time.
- This substantially reduces the action space by eliminating clearly invalid commands.
- Important caveat: this makes results not directly comparable to fully open-ended settings.

---

## Experimental Reliability Analysis

| Aspect | Trustworthy | Questionable |
|---|---|---|
| RL results | 5-seed averaging, low variance | Only 100 max steps — may miss long-horizon solutions |
| T5/LM results | Zero-shot inference is consistent | Single evaluation pass — no seed variance reported |
| Baseline fairness | Identical training regimes | T5 pre-trained on science QA data (unfair advantage in knowledge, not planning) |
| Generalization | Unseen objects in test set | Oracle solutions are canonical only — alternate solutions not tested |
| Environment fidelity | Physics engines grounded in real science | Simplified — no advanced chemistry, no 3D spatial reasoning |

---

# 6. Results & Findings Interpretation

## 6.1 Main Outcomes

| Agent | Avg Score (all 30 tasks) | Parameters |
|---|---|---|
| Random-Valid | 0.03 | — |
| DRRN | **0.17** | 1.5M |
| KG-A2C | 0.11 | 5.5M |
| CALM (GPT-2) | ~0.05 | 131M |
| BC-T5 (Macaw) | 0.08 | 11,000M |
| TDT-T5 (Macaw) | 0.08 | 11,000M |

## 6.2 Critical Observations

**Finding 1 — Size does not equal intelligence here**
- DRRN (1.5M params) beats T5/Macaw (11B params) by a factor of 2x.
- Conclusion: How a model interacts with the environment matters more than how many parameters it has.

**Finding 2 — Valid action detection is a major aid**
- Models relying on valid action detection (DRRN, KG-A2C) outperform those that must generate valid actions (CALM, BC, TDT).
- LLMs generate "plausible-sounding" but often **invalid** actions in this environment.

**Finding 3 — Known vs. Unknown substance gap**
- Task 3-3 (known conductivity substance) vs. Task 3-4 (unknown substance B): similar performance.
- Agents have not learned to leverage retrieval for known cases — they don't reach the point where prior knowledge would help.

**Finding 4 — Easiest task: Find a non-living thing (Task 4-2)**
- Random baseline scores 0.63! Agents do similarly (0.44–0.56).
- Explanation: Many non-living objects exist; the environment is full of valid answers.

**Finding 5 — Hardest tasks: Matter changes (boiling, melting, freezing)**
- All agents score near 0 on Tasks 1-1 through 1-4.
- Agents fail to navigate to a heating source, pick up the object, and place it correctly.

**Finding 6 — Agents lack commonsense navigation**
- Trajectories show agents struggle with basics (navigating rooms, storing liquids) before even reaching science-specific actions.

## 6.3 Unexpected Observations

- Behavior cloning from 11B expert demonstrations performs the same as a 1.5M RL agent — static training fails to transfer to interactive settings.
- The Text Decision Transformer, designed to use future reward conditioning, shows no benefit over standard behavior cloning — the model cannot use returns-to-go effectively in a novel interactive environment.

---

## Publishability Strength Check

| Result | Strength |
|---|---|
| DRRN > T5/Macaw (size vs. interaction) | Strong — publication-grade, counterintuitive finding |
| Known vs. Unknown substance comparison | Moderate — requires more analysis to fully isolate retrieval vs. reasoning |
| All agents fail at matter changes | Strong — clear systematic failure, reproducible |
| Valid action detection dependency | Strong — practical limitation clearly identified |
| Performance variance analysis (5 seeds) | Strong — statistically credible |

---

# 7. Strengths – Weaknesses – Assumptions

## Technical Strengths

| Strength | Details |
|---|---|
| Multi-engine physics simulation | 7 simulation engines covering thermodynamics, electricity, genetics, chemistry, friction |
| Generalizable task design | Parametric variations prevent memorization |
| Known/Unknown task pairs | Clean isolation of retrieval vs. reasoning ability |
| Partial credit scoring | Subgoal rewards provide richer learning signal than binary success |
| Open-source, accessible | Code and environment publicly released |
| Procedural explanation evaluation | Agent action sequences serve as executable explanations |
| Scale comparison | 1.5M vs. 11B parameter agents gives dramatic, clear contrast |

## Explicit Weaknesses

| Weakness | Impact |
|---|---|
| Only canonical oracle solutions | Imitation agents biased toward one solution path |
| Valid action detection aid | Artificially reduces problem difficulty; not realistic |
| Short descriptions (transformer sequence length limits) | Agents receive simplified observations of complex environments |
| Only series circuits modelled | Limits electrical reasoning depth |
| No 3D or spatial reasoning | Some real experiments require spatial cognition |
| All tasks closed-ended | No open-ended science discovery tasks |
| 100-step episode cap | Long-horizon tasks artificially truncated |

## Hidden Assumptions

| Assumption | Why It Matters |
|---|---|
| Elementary science = good reasoning proxy | Complex science may show different patterns |
| Oracle trajectories represent correct behavior | Alternate valid solutions are not captured |
| Text agents can be directly compared to LLMs | Training objectives differ fundamentally |
| Valid action detection is an "aid" not the task | In real deployment, no such aid exists |
| 5 seeds sufficient for RL variance | Complex environments may require more seeds |
| Normalized score captures true capability | Partial scores may mask systematic failures |

---

# 8. Weakness → Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| Only canonical oracle solutions | Hand-coding multiple solutions is expensive | **Multi-solution imitation learning**: train agents on diverse trajectories | Procedural generation of alternate solutions; coverage-based sampling |
| Valid action detection required | LLMs generate invalid actions in novel environments | **Grounded language generation**: constrain LLM outputs to valid environment actions | Constrained decoding, action-grounded fine-tuning |
| Agents fail at navigation + reasoning together | Navigation and science tasks require different skills | **Hierarchical agent architectures**: separate navigation from science reasoning | Options framework, skill chaining, modular RL |
| LLMs can't transfer procedural knowledge | Static training has no feedback loop | **Online LLM fine-tuning**: update LLMs using environment interaction signals | RLHF adapted to interactive environments, online learning |
| Only 30 tasks across 10 topics | Manual task design is bottleneck | **Automated task generation**: LLM-assisted task creation from curriculum | Curriculum-based automatic task synthesis |
| No cross-domain generalization | Each task is trained/tested independently | **Cross-task transfer learning**: test if skills learned in one task transfer | Meta-learning, zero-shot task transfer experiments |
| Short 100-step episodes | Long experiments truncated artificially | **Long-horizon planning evaluation**: design benchmarks for extended experiments | Hierarchical planning, memory-augmented agents |
| 2D only (no spatial reasoning) | Text-only environment limitations | **Hybrid text + spatial environments**: add spatial components | Grid-based hybrid environments, symbolic spatial reasoning |

---

# 9. Novel Contribution Extraction

## Explicit Contribution Statements from This Paper

1. A new **interactive text benchmark** grounded in elementary science curriculum with 30 tasks and 7 simulation engines.
2. An **empirical demonstration** that interactive RL agents are more parameter-efficient than large offline-trained LLMs for grounded reasoning.
3. A **procedural explanation framework** — agent action sequences serve as executable, automatically evaluable explanations for scientific reasoning.

## 3–5 Novel Claim Templates for Your Own Paper

1. "We propose **[new agent architecture]** that improves grounded scientific reasoning by **[integrating prior knowledge with interactive feedback]**, achieving **[X% higher task completion]** on ScienceWorld without relying on valid action detection aids."

2. "We introduce **[automated curriculum task generator]** that expands ScienceWorld's 30 tasks to **[N tasks]** by using LLMs to synthesize new experimental scenarios from science textbooks, enabling broader evaluation of procedural science reasoning."

3. "We demonstrate that **[hierarchical RL agent]** separating navigation from reasoning improves average ScienceWorld performance from 0.17 to **[target score]** by decomposing the search problem."

4. "We show that **[online RLHF-adapted LLM]** achieves performance parity with small RL agents on ScienceWorld's known-unknown task pairs, demonstrating that interactive fine-tuning closes the grounded reasoning gap for large language models."

5. "We propose **[cross-task transfer framework]** where skills learned in simpler ScienceWorld tasks (e.g., object manipulation) transfer to harder tasks (e.g., circuit building), reducing sample complexity by **[X%]**."

---

# 10. Future Research Expansion Map

## Author-Suggested Future Work

- Cuing agents to generate **high-level explanatory scaffolds** alongside action sequences (e.g., "I need to heat the substance, so I'll use the stove") to improve both task performance and interpretability.
- Developing agents that can **leverage internal LLM knowledge** when solving known-substance tasks (the paper shows this doesn't happen yet).
- Using ScienceWorld as pre-training for **3D hybrid environments** like AlfWorld (transferring text-trained skills to visual environments).

## Missing Directions Not Discussed in Paper

- **Multi-agent science experiments**: some real experiments require collaboration (one agent heats, another measures).
- **Natural language instruction following**: can agents follow free-form instructions like "figure out what temperature this substance boils at" without pre-coded task structures?
- **Causal reasoning evaluation**: does completing the experiment imply the agent understands the causal mechanism?
- **Mistake detection and recovery**: can agents recognize and correct failed sub-procedures?
- **Human-in-the-loop experiments**: partial automation where human provides hints when agent gets stuck.

## Modern / LLM-Era Extensions (2023+)

- **LLM-as-planner frameworks** (e.g., ReAct, Toolformer, ProgPrompt): Test if chain-of-thought prompting helps agents plan multi-step experiments without RL training.
- **LLM agent with environment memory** (e.g., Reflexion): Use textual self-reflection to improve across episodes.
- **Foundation model + RL grounding**: Pre-train with LLM for knowledge, fine-tune with RL for procedural skill — testing if this hybrid approach closes the gap shown in this paper.
- **World model learning**: Train agents to build internal models of ScienceWorld's physics engines — testing systematic generalization.
- **Vision-Language Agents**: Extend ScienceWorld observations with diagrams or visual representations to test multimodal science reasoning.

---

# 11. How to Write a New Paper From This Work

## Reusable Elements

- **ScienceWorld environment**: Directly usable as a benchmark — no need to rebuild the simulator.
- **Known/Unknown task pair design**: Apply to any benchmark to isolate retrieval from reasoning.
- **Parametric variation system**: Reuse this design pattern for new benchmarks.
- **Subgoal reward structure**: Reuse for reward shaping in any hierarchical task.
- **Procedural explanation = executable evaluation**: This framing is reusable for any task where actions can be verified.

## What MUST NOT Be Copied

- Do not reuse the 30 task descriptions without citation.
- Do not reuse the oracle action trajectories without attribution.
- Do not claim to evaluate on ScienceWorld without explicitly following the train/dev/test split.

## How to Design a Novel Extension

**Option A — New Agent Architecture**
1. Identify one failure mode from the paper (e.g., navigation failure).
2. Design an agent that explicitly handles this failure (e.g., hierarchical navigation module).
3. Compare to DRRN as primary baseline.
4. Show improvement on ScienceWorld subtasks affected by that failure.

**Option B — New Benchmark Extension**
1. Add new simulation engines (e.g., advanced chemistry, 2D spatial reasoning).
2. Add new task categories from middle-school curriculum.
3. Re-evaluate all baseline agents.
4. Show new capabilities exposed by the extension.

**Option C — LLM Integration Study**
1. Apply modern LLM agents (GPT-4, Claude) with prompting strategies to ScienceWorld.
2. Compare to the 2022 baselines.
3. Analyze which ScienceWorld tasks have been "solved" and which remain hard.
4. Publish a new state-of-the-art table.

**Option D — Transfer Learning Study**
1. Pre-train agents on ScienceWorld.
2. Test transfer to AlfWorld or other text environments.
3. Compare to agents trained from scratch.

---

## Minimum Publishable Contribution Checklist

- [ ] Novel agent or method that beats DRRN (0.17 avg) by meaningful margin
- [ ] OR new benchmark extension with at least 10 new tasks
- [ ] OR systematic analysis of modern LLMs (GPT-4 era) on ScienceWorld
- [ ] Comparison against all 5 original agents as baselines
- [ ] Ablation study isolating the key contribution
- [ ] Results across multiple random seeds (min. 5)
- [ ] Analysis of known vs. unknown task performance
- [ ] Discussion of valid action detection dependency (use or justify removing it)

---

# 12. Complete Paper Writing Template

## Abstract
- **Purpose**: Summarize problem, method, key result in 150–250 words.
- **Include**: What task/problem, what environment, what agent/method, 1–2 key quantitative results.
- **Common mistakes**: Overpromising generality; omitting the benchmark/dataset; vague results ("improved performance").
- **Reviewer expectation**: Clear problem statement + concrete numbers.

---

## 1. Introduction
- **Purpose**: Motivate why this matters, what is missing, what you do.
- **Include**:
  - The gap between LLM QA success and grounded reasoning failure.
  - Reference to ScienceWorld as benchmark.
  - Your specific contribution (1 paragraph of clear bullet contributions at the end).
- **Common mistakes**: Overselling ("we solve scientific reasoning"); unclear contribution vs. related work.
- **Reviewer expectation**: Sharp problem definition + crisp 3-bullet contribution list.

---

## 2. Related Work
- **Purpose**: Situate your work relative to: (a) science QA, (b) interactive text environments, (c) agent architectures.
- **Include**:
  - Prior benchmarks (ARC, OpenBookQA) and their limitations.
  - Prior text environments (TextWorld, AlfWorld, ScienceWorld) and your differences.
  - Prior agents (DRRN, BC, Decision Transformer) and how yours differs.
- **Common mistakes**: Citing without explaining why they're insufficient; missing key ScienceWorld-adjacent papers.
- **Reviewer expectation**: You must cite Wang et al. (2022) and explain what you add beyond it.

---

## 3. Method
- **Purpose**: Describe your agent or system clearly.
- **Include**:
  - Architecture diagram or pseudocode.
  - How your method addresses the specific weakness you identified.
  - Training procedure.
- **Common mistakes**: Skipping design rationale; not explaining how components connect.
- **Reviewer expectation**: Enough detail to reproduce results.

---

## 4. Theory (if applicable)
- **Purpose**: Provide formal grounding for your method.
- **Include**: Formal definitions, objective function, convergence claims if any.
- **Common mistakes**: Including proofs without connecting them to the empirical work.

---

## 5. Experiments
- **Purpose**: Show your method works and why.
- **Include**:
  - ScienceWorld benchmark with official train/dev/test splits.
  - All 5 original baselines + your method.
  - Ablation study.
  - Per-task breakdown (like Table 2 in the paper).
  - Known vs. unknown task analysis.
- **Common mistakes**: Only reporting average; skipping ablations; not running multiple seeds.
- **Reviewer expectation**: Table comparing all baselines, statistical significance, ablation.

---

## 6. Discussion
- **Purpose**: Interpret findings, discuss implications.
- **Include**:
  - Why your method works (connect to design choices).
  - Where it still fails (honest analysis of limitations).
  - Connection to open questions in the field.
- **Common mistakes**: Pure positive spin; no failure case analysis.

---

## 7. Limitations
- **Purpose**: Demonstrate academic integrity + guide future work.
- **Include**:
  - What your method cannot do.
  - Simplifying assumptions in the environment.
  - What results may not generalize.
- **Common mistakes**: Generic statements ("more data would help") instead of specific limitations.
- **Reviewer expectation**: Concrete, honest, specific limitations — not vague hedging.

---

## 8. Conclusion
- **Purpose**: 1 paragraph summary of contribution and impact.
- **Include**: Problem, method, key results, 1 future direction.
- **Common mistakes**: Repeating the abstract verbatim; adding new information not in the paper.

---

## 9. References
- Must cite: Wang et al. (2022) ScienceWorld, Côté et al. (2018) TextWorld, Hausknecht et al. (2020) DRRN, Clark et al. (2018/2020) ARC, Raffel et al. (2020) T5, Chen et al. (2021) Decision Transformer.
- Use ACL/EMNLP citation format.

---

# 13. Publication Strategy Guide

## Suitable Venues

| Venue Type | Examples | Why Suitable |
|---|---|---|
| Top NLP Conferences | ACL, EMNLP, NAACL | ScienceWorld is NLP benchmark; core audience |
| Top AI/ML Conferences | NeurIPS, ICML, ICLR | If method contribution is strong |
| Specialized Workshops | Grounded Language, Reasoning workshops at ACL/NeurIPS | Good for incremental or early-stage work |
| Robotics + NLP | CoRL (if physical grounding extended) | For work connecting text and embodied AI |

## Required Baseline Expectations

- **Must beat DRRN (0.17)** by a clear margin if claiming a new SOTA agent.
- **Must compare to all 5 original agents** for fair comparison.
- **Must use valid train/dev/test splits** from original paper.
- **Must report per-task results**, not just average.

## Experimental Rigor Level Required

- 5 random seeds minimum for RL agents.
- Ablation study isolating each component of your method.
- Statistical significance (even informal std dev reporting as in the original).
- Known vs. unknown task pair analysis.

## Common Rejection Reasons

- Weak baselines: only compared to random or one agent.
- No ablation: can't tell which part of your method contributes.
- Cherry-picked tasks: only reported easy tasks where improvement is visible.
- Ignoring valid action detection: did not clarify whether you use the aid.
- Claiming to solve the benchmark when average is still below 0.50.
- Missing the original paper's key comparison (small RL vs. large LLM finding).

## Increment Needed for Acceptance

- Top venue (ACL/NeurIPS): Average score improvement from 0.17 to 0.35+ AND clear methodological novelty.
- Workshop / smaller venue: Any principled improvement with good analysis.
- System paper: New benchmark extension (new tasks + re-evaluation) counts as contribution.
- Analysis paper: Systematic study of modern LLMs on ScienceWorld is publishable without a new method.

---

# 14. Researcher Quick Reference Tables

## Key Terminology Table

| Term | Meaning |
|---|---|
| ScienceWorld | Interactive text environment with 30 elementary science tasks |
| POMDP | Mathematical model for sequential decision-making under partial observation |
| DRRN | Deep Reinforcement Relevance Network — small RL agent, best performer |
| KG-A2C | Knowledge Graph + Actor-Critic — RL agent using structured state representation |
| CALM | Language model generates action candidates, RL re-ranks them |
| Behavior Cloning (BC) | Learning from expert demonstrations offline |
| Text Decision Transformer (TDT) | Predicts actions conditioned on desired future reward sum |
| Returns-to-go (R̂) | Sum of future rewards an agent aims to achieve |
| Parametric variation | Changing task objects/starting location to test generalization |
| Valid action detection | System that filters syntactically/semantically invalid actions |
| Subgoal | Intermediate goal step that provides partial reward to reduce sparsity |
| Oracle agent | Hand-coded program that follows the canonical correct solution |
| Procedural explanation | Agent's action sequence as an executable, verifiable "how to" explanation |
| Grounded reasoning | Understanding that connects language to physical actions in the real (or simulated) world |

## Important Equations Summary

| Equation | Purpose |
|---|---|
| `E[ Σ_t γ^t R(s_t, a_t) ]` | POMDP objective: maximize discounted future rewards |
| `R̂_t = Σ_{t'=t}^{T} r_{t'}` | Returns-to-go: total future reward from step t onward |
| `Score = achieved_subgoals / max_achievable` | Normalized task score ∈ [0,1] |
| `π(a | o)` | Policy: probability of action given observation |

## Parameter Meaning Table

| Parameter | Value Used | Role |
|---|---|---|
| Max episode steps | 100 | Prevents infinite exploration |
| RL training threads | 8 | Parallelism for faster training |
| Total RL steps | 100,000 / thread | Training budget |
| Random seeds | 5 | Statistical reliability |
| Evaluation window | Last 10% of episodes | Stable performance measurement |
| BC training examples | 211,092 (BC) / 224,902 (TDT) | Volume of expert demonstrations |
| Beam search width | 30 | Action candidate list size for CALM/BC/TDT |
| Discount factor γ | 0 < γ < 1 | Future reward weighting |

## Algorithm Flow Summary

| Agent | Key Steps |
|---|---|
| DRRN | Observe text → encode observation + action candidates → rank by relevance → pick best → update with RL reward |
| KG-A2C | Observe text → extract triples → build knowledge graph → select action template → fill with graph objects → update with RL |
| CALM | Observe text → GPT-2 generates 30 candidates → RL re-ranks → pick highest valid → update RL |
| BC-T5 | Offline: fine-tune T5 on expert demos → Online: generate actions with beam search → pick highest-ranked valid action |
| TDT-T5 | Offline: fine-tune T5 on demos + returns-to-go → Online: condition on desired reward, generate actions, pick highest-ranked valid |

---

# 15. One-Page Master Summary Card

| Section | Content |
|---|---|
| **Problem** | Current AI models answer science questions well but cannot perform science experiments in grounded, interactive environments. Static text benchmarks mask this failure. |
| **Key Insight** | Answering "What happens when ice is placed on a stove?" is retrieval. Actually placing virtual ice on a virtual stove, observing it melt, and recording the result requires procedural reasoning — a fundamentally different capability. |
| **Method** | ScienceWorld: a text-based simulator with 7 physics engines (thermodynamics, electricity, chemistry, genetics, friction, life stages, containers), 30 tasks across 10 topics, and 7,200 parametric variations to force generalization. |
| **Baselines Tested** | Random, DRRN (RL, 1.5M), KG-A2C (RL+graph, 5.5M), CALM (GPT-2+RL, 131M), BC-T5 (imitation, 11B), TDT-T5 (Decision Transformer, 11B) |
| **Key Results** | Best agent (DRRN) achieves only 0.17 average score. An 11B parameter model achieves 0.08. All agents score near 0 on matter changes. Valid action detection is required for most models to function at all. |
| **Critical Finding** | Online RL training (interactive) > offline imitation learning (static) for grounded science tasks — regardless of model size. |
| **Weakness** | Only canonical solutions in oracles; valid action detection aid used; simplified physics; 100-step episode cap; no cross-task generalization testing. |
| **Research Opportunity** | (1) Hierarchical agents separating navigation from reasoning. (2) Online fine-tuning of LLMs using environment feedback. (3) Modern LLM evaluation (GPT-4/Claude era) on ScienceWorld. (4) Automated task generation to expand beyond 30 tasks. |
| **Publishable Extension** | Beat DRRN baseline (0.17) with a principled method + ablation study + ACL/EMNLP submission. OR apply LLM agents with chain-of-thought/ReAct to ScienceWorld and publish a systematic analysis. |
