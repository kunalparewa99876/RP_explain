# Research Companion Sheet: DreamerV3 — Mastering Diverse Domains through World Models

**Paper:** Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering Diverse Domains through World Models. arXiv:2301.04104v2.

**Paper Classification:** Algorithmic / Method + Experimental ML / Empirical

**Adaptation Strategy:** Because this paper is both algorithmic (proposing DreamerV3's architecture and robustness techniques) and empirical (extensive benchmarks across 150+ tasks), explanations will:
- Provide workflow logic and pseudocode intuition for all algorithmic components
- Focus on design decisions, baselines, and metrics for experimental sections
- Explain intuition before equations for mathematical formulations
- Emphasize the reasoning behind each design choice

---

# 0. Quick Paper Identity Card

| Field | Detail |
|---|---|
| **Problem Domain** | General-purpose reinforcement learning across diverse domains |
| **Paper Type** | Algorithmic / Method + Experimental ML |
| **Core Contribution** | A single RL algorithm with fixed hyperparameters that outperforms specialized expert methods across 150+ tasks spanning 8 domains |
| **Key Idea** | Learn a world model to imagine future outcomes, then train actor-critic in imagination; use normalization, balancing, and transformation techniques to make this robust across all domains without tuning |
| **Required Background** | Reinforcement learning basics (MDP, policy, value function), variational autoencoders, recurrent neural networks, actor-critic methods |
| **Primary Baselines** | PPO (general), MuZero (Atari), DrQ-v2 (visual control), VPT (Minecraft), domain-specific expert algorithms |
| **Main Innovation Type** | Robustness techniques (symlog, twohot, return normalization, KL balancing) enabling a single configuration across all domains |
| **Difficulty Level** | Advanced (requires RL, latent dynamics, variational inference knowledge) |
| **Reproducibility Level** | High — open-source code, single A100 GPU per experiment, all hyperparameters disclosed |

---

# 1. Research Context & Core Problem

## 1.1 Exact Problem Formulation

The core problem is: **How do you build ONE reinforcement learning algorithm that works well across fundamentally different domains without changing any hyperparameters?**

Current RL algorithms are brittle. An algorithm tuned for Atari games fails at robot control. An algorithm designed for continuous motor control struggles with discrete actions in board games. Researchers spend enormous time and compute tuning hyperparameters for each new domain.

Specifically, the challenges that differ across domains include:
- **Action types:** Continuous (robotic joints) versus discrete (game buttons)
- **Input types:** Raw pixel images versus low-dimensional sensor vectors
- **Reward structures:** Dense rewards (constant feedback) versus sparse rewards (feedback only at rare milestones)
- **Reward scales:** Rewards ranging from tiny fractions to millions
- **Environment complexity:** Simple 2D tasks versus infinite procedurally generated 3D worlds

## 1.2 Why the Problem Exists

RL algorithms contain many design decisions (learning rates, loss functions, reward scaling, exploration amounts, network architectures) that interact with environment characteristics. A hyperparameter that works for one domain may catastrophically fail in another because:

- Reward magnitudes differ by orders of magnitude across environments
- Gradient scales become unstable when reward or observation scales change
- Exploration requirements differ drastically between dense and sparse reward settings
- The balance between learning from observations versus learning from rewards shifts across domains

## 1.3 Historical / Theoretical Gap

- **PPO** (Schulman et al., 2017): Widely used and relatively robust but requires massive amounts of experience and often underperforms specialized methods
- **SAC** (Haarnoja et al., 2018): Good for continuous control but needs entropy scale tuning and struggles with high-dimensional inputs
- **MuZero** (Schrittwieser et al., 2019): Powerful planning algorithm but complex, hard to reproduce, not open-source
- **DreamerV1** (Hafner et al., 2019): Limited to continuous control only
- **DreamerV2** (Hafner et al., 2020): Surpassed human performance on Atari but still needed domain-specific tuning for some benchmarks

No single algorithm before DreamerV3 could master continuous control, discrete games, sparse reward tasks, visual environments, and procedurally generated worlds with one set of hyperparameters.

## 1.4 Contribution Category

- **Algorithmic:** Novel robustness techniques (symlog transformation, symexp twohot loss, percentile return normalization, KL balancing with free bits)
- **System Design:** A unified world model-based RL system with fixed configuration
- **Empirical:** State-of-the-art or competitive results across 8 benchmarks (150+ tasks)
- **Milestone Achievement:** First algorithm to collect diamonds in Minecraft from scratch without human data

### Why This Paper Matters

DreamerV3 fundamentally changes the practical applicability of RL. Before this work, applying RL to a new domain required significant expertise to tune algorithms. DreamerV3 demonstrates that a single well-designed algorithm can be applied "out of the box" to diverse problems. This is analogous to how ImageNet pretrained models democratized computer vision — DreamerV3 aims to make RL accessible to practitioners who cannot afford extensive hyperparameter searches.

The Minecraft achievement is particularly significant: the agent learns a 12-step technology tree from sparse rewards in a procedurally generated infinite 3D world, using 1 GPU for 9 days. Previous solutions (VPT) required 720 GPUs for 9 days plus human expert data.

### Remaining Open Problems

1. **Sample efficiency at scale:** Even with 100M steps, Dreamer only obtains diamonds in 0.4% of Minecraft episodes
2. **Unsupervised pretraining:** The world model rests on reconstruction, opening the door for pretraining on unlabeled video data
3. **Cross-domain transfer:** Can a single world model learn across multiple domains simultaneously?
4. **Hierarchical planning:** The current flat policy may struggle with even longer planning horizons
5. **Real-world deployment:** Sim-to-real transfer and safety constraints are not addressed
6. **Language integration:** Combining world models with language understanding for instruction following

---

# 2. Minimum Background Concepts

## 2.1 Reinforcement Learning (RL)

- **Plain definition:** An agent interacts with an environment by taking actions, receiving observations and rewards, and learning a policy (decision rule) that maximizes cumulative reward
- **Role in paper:** DreamerV3 is an RL algorithm; the entire paper is about solving RL problems more robustly
- **Why authors needed it:** The paper proposes improvements to the RL training loop itself

## 2.2 World Model

- **Plain definition:** A neural network that learns how the environment works — it predicts what will happen next given the current state and an action, without actually interacting with the real environment
- **Role in paper:** The world model is the central component of DreamerV3; the agent "dreams" (imagines) future trajectories using this model instead of collecting real experience for every learning update
- **Why authors needed it:** Learning in imagination is far more data-efficient than learning from real interactions alone, because one real experience can be replayed and imagined many times

## 2.3 Actor-Critic Architecture

- **Plain definition:** Two neural networks working together — the actor chooses actions, the critic evaluates how good a state is (estimates future reward). The critic helps the actor improve by telling it which actions lead to better outcomes
- **Role in paper:** The actor and critic learn purely from imagined trajectories generated by the world model
- **Why authors needed it:** Combines the advantages of policy-based methods (actor) and value-based methods (critic) for stable learning

## 2.4 Recurrent State-Space Model (RSSM)

- **Plain definition:** A neural network architecture that maintains a recurrent hidden state (memory) plus a stochastic latent state. The recurrent part remembers history; the stochastic part captures uncertainty about the current situation
- **Role in paper:** The RSSM is the backbone of DreamerV3's world model, predicting future states given actions
- **Why authors needed it:** The separation of deterministic (recurrent) and stochastic (latent) components enables both long-term memory and modeling uncertainty — essential for accurate multi-step prediction

## 2.5 Variational Autoencoder (VAE) Principles

- **Plain definition:** A neural network that compresses high-dimensional data (like images) into compact codes and reconstructs the data from those codes. It introduces randomness in the codes to learn generalizable representations
- **Role in paper:** The world model uses VAE-like encoding to compress observations into discrete latent representations
- **Why authors needed it:** Raw observations (images, sensor readings) are too high-dimensional for efficient planning; compact representations enable fast imagination

## 2.6 KL Divergence

- **Plain definition:** A measure of how different two probability distributions are. In this paper, it measures the gap between what the encoder sees (posterior) and what the dynamics model predicts (prior)
- **Role in paper:** Used to train the world model so that its predictions of future states align with what actually happens
- **Why authors needed it:** Balancing the KL divergence ensures that latent representations are both informative (encode useful information from observations) and predictable (the dynamics model can forecast them)

## 2.7 Lambda-Returns (λ-returns)

- **Plain definition:** A method to estimate the total future reward by blending short-term reward predictions with long-term value estimates. The parameter λ controls this blend — higher λ gives more weight to actual rewards, lower λ relies more on the critic's estimates
- **Role in paper:** Used to compute target returns for critic training during imagination
- **Why authors needed it:** Pure reward summation requires very long imagination horizons; λ-returns provide good estimates with short horizons (T=16 steps)

---

# 3. Mathematical / Theoretical Understanding Layer

## 3.1 RSSM Equations (Equation 1)

**Intuition:** The world model maintains two types of state — a deterministic recurrent state h (like a memory) and a stochastic latent state z (capturing uncertainty). Together they form the "model state" s = {h, z}.

**What problem it solves:** Predicting what happens next in an environment from a history of observations and actions.

**The components:**

| Component | Formula | What It Does |
|---|---|---|
| Sequence model | h_t = f_ϕ(h_{t-1}, z_{t-1}, a_{t-1}) | Updates memory based on previous state and action |
| Encoder | z_t ~ q_ϕ(z_t \| h_t, x_t) | Extracts latent state from observation + memory |
| Dynamics predictor | ẑ_t ~ p_ϕ(ẑ_t \| h_t) | Predicts latent state from memory alone (no observation) |
| Reward predictor | r̂_t ~ p_ϕ(r̂_t \| h_t, z_t) | Predicts reward from model state |
| Continue predictor | ĉ_t ~ p_ϕ(ĉ_t \| h_t, z_t) | Predicts whether episode continues |
| Decoder | x̂_t ~ p_ϕ(x̂_t \| h_t, z_t) | Reconstructs observation from model state |

**Assumptions:**
- The model state (h_t, z_t) is sufficient to predict rewards, continuations, and observations (Markov assumption in latent space)
- Discrete categorical latent representations (vectors of softmax distributions) are sufficient for capturing environment dynamics

**Practical interpretation:** During imagination (actor-critic training), only the sequence model and dynamics predictor are needed — no encoder, no real observations. This makes imagination very fast.

## 3.2 World Model Loss (Equations 2-3)

**Intuition:** The world model has three learning signals that must be balanced:
1. **Prediction loss:** Can the model reconstruct observations and predict rewards? (practical utility)
2. **Dynamics loss:** Can the sequence model predict future latent states? (imagination accuracy)
3. **Representation loss:** Are the representations predictable? (makes imagination possible)

**Variable meaning:**

| Variable | Meaning | Value |
|---|---|---|
| β_pred | Weight on prediction loss | 1 |
| β_dyn | Weight on dynamics loss | 1 |
| β_rep | Weight on representation loss | 0.1 |
| sg(·) | Stop-gradient operator — blocks gradient flow | — |
| KL[q \|\| p] | KL divergence between encoder and predictor distributions | — |

**Key design decisions:**

- **Why β_rep = 0.1 (small)?** The representation loss prevents the model from encoding information that the dynamics predictor cannot forecast. But making this too strong removes useful information. A small weight ensures representations are predictable without losing important details.

- **Why free bits (clipping KL below 1 nat)?** Without this, the KL losses can dominate and force the representations to be trivially simple (containing no useful information). Free bits disable the KL losses when they are already small enough, letting the prediction loss drive learning.

- **Why separate dynamics and representation losses with stop-gradients?** The dynamics loss pushes the predictor toward the encoder (making predictions match reality). The representation loss pushes the encoder toward the predictor (making reality match predictions). Without the asymmetric stop-gradients, both could collapse to a trivial solution.

**Limitation:** The free bits threshold of 1 nat is a hyperparameter that could theoretically need domain-specific tuning, though the authors show it works universally.

## 3.3 Symlog and Symexp Transformations (Equations 8-9)

**Intuition:** When targets (rewards, observations) vary by orders of magnitude across domains, standard loss functions break. Squared loss explodes with large targets; absolute loss gives tiny gradients for large targets; normalization introduces non-stationarity. The symlog function compresses large values while keeping small values unchanged.

**What problem it solves:** Making gradient magnitudes independent of target scales.

| Function | Formula | What It Does |
|---|---|---|
| symlog(x) | sign(x) · ln(\|x\| + 1) | Compresses large magnitudes while preserving sign |
| symexp(x) | sign(x) · (exp(\|x\|) - 1) | Inverse of symlog — recovers original scale |

**Key properties:**
- Symmetric around origin (unlike log, which cannot handle negatives)
- Approximates identity near zero (does not distort small values)
- Compresses large values logarithmically (prevents gradient explosions)

**Practical interpretation:** The network learns to predict symlog(target), and predictions are converted back via symexp. This means the loss function treats a prediction error of 1000 vs 1010 similarly to 10 vs 20 — the relative error matters, not the absolute scale.

## 3.4 Symexp Twohot Loss (Equations 10-11)

**Intuition:** For stochastic targets like rewards (which can be noisy or multi-modal), a single-point prediction is insufficient. Instead, the network predicts a full distribution over possible values using a discrete set of bins.

**What problem it solves:** Predicting potentially multi-modal distributions of rewards/returns while decoupling gradient magnitudes from target scales.

**How it works:**
1. Define bins B = symexp({-20, ..., +20}) — exponentially spaced bins covering a huge range
2. Network outputs softmax probabilities over these bins
3. Prediction = weighted average of bin positions (can be any continuous value between bins)
4. Training target = twohot encoding (soft label over the two nearest bins)
5. Loss = categorical cross-entropy between predicted and target distributions

**Why twohot instead of onehot?** Onehot can only represent bin centers. Twohot interpolates between adjacent bins, allowing continuous target values.

**Why exponential bin spacing?** Covers values from -exp(20) ≈ -485 million to +exp(20) with only ~255 bins, giving fine resolution near zero and coarse resolution for extreme values.

## 3.5 Return Normalization (Equations 6-7)

**Intuition:** The actor needs to balance exploitation (maximizing returns) and exploration (maintaining entropy). But returns can range from 0.01 to 1,000,000 across domains. A fixed entropy scale would over-explore in high-reward domains and under-explore in low-reward domains.

**What problem it solves:** Making the actor's exploration-exploitation balance robust across reward scales.

**How it works:**
- Compute the range S from the 5th to 95th percentile of returns in the current batch
- Divide advantages by max(1, S) — this normalizes large returns but leaves small returns untouched
- Use exponential moving average to smooth the range estimate

**Why percentiles instead of min/max?** Randomized environments can have outlier episodes with extreme returns. Percentiles are robust to these outliers.

**Why max(1, S) instead of just S?** When rewards are sparse, S can be near zero. Dividing by a tiny S would amplify noise and destabilize learning. The limit of 1 ensures that small returns are not artificially inflated.

### Mathematical Insight Box

**Key insight for researchers:** The fundamental idea is that robustness across domains requires decoupling gradient magnitudes from target magnitudes. Three techniques achieve this: (1) symlog transformation compresses observation/target scales, (2) categorical cross-entropy on twohot bins makes reward/return gradients scale-independent, (3) percentile return normalization with a lower limit makes actor gradients scale-independent while preserving sparse reward information. These techniques are general and could be applied to many other RL or supervised learning problems with varying scales.

---

# 4. Proposed Method / Framework (MOST IMPORTANT)

## 4.1 Overall Pipeline

DreamerV3 operates in a repeating cycle with three concurrent processes:

```
ENVIRONMENT INTERACTION
    ↓ (collect real experience)
REPLAY BUFFER (stores transitions)
    ↓ (sample minibatches)
WORLD MODEL TRAINING (learn environment dynamics)
    ↓ (generate imagined trajectories)
ACTOR-CRITIC TRAINING (learn behavior in imagination)
    ↓ (updated policy)
ENVIRONMENT INTERACTION (use updated policy)
```

**Step 1: Environment Interaction**
- The actor network selects actions by sampling from its policy π(a|s) given the current model state
- No lookahead planning — just a single forward pass through the actor
- Observations, actions, rewards, and continuation flags are stored in the replay buffer

**Step 2: World Model Training**
- Sample minibatches of length T=64 from the replay buffer
- Run encoder to get latent representations z_t from observations x_t
- Train sequence model, decoder, reward predictor, continue predictor, and dynamics predictor
- Losses: prediction loss + dynamics KL loss + representation KL loss

**Step 3: Actor-Critic Training (Imagination)**
- Start from encoded representations of real observations
- Use the world model to imagine T=15 steps into the future
- During imagination: only the sequence model and dynamics predictor run (no real observations)
- Compute λ-returns from imagined rewards and critic value estimates
- Train critic to predict return distributions
- Train actor to maximize normalized returns with entropy regularization

✔ **Why authors did this:** Separating world model learning from behavior learning allows the agent to practice thousands of imagined scenarios for every real interaction, dramatically improving data efficiency.
✔ **Weakness:** Imagination accuracy depends on world model quality; errors compound over long horizons.
✔ **Research idea seed:** Incorporate uncertainty-aware imagination that reduces trust in predictions as they extend further into the future.

## 4.2 World Model Component

### Architecture: RSSM with Discrete Latents

The world model uses the Recurrent State-Space Model (RSSM) with the following architecture:

- **Sequence model:** GRU (Gated Recurrent Unit) with block-diagonal recurrent weights (8 blocks). This allows many recurrent units without quadratic parameter growth.
- **Encoder:** CNN for images (stride-2 convolutions to 6×6 or 4×4 resolution, then flattened) or MLP for vectors. Output is a vector of categorical distributions (softmax).
- **Decoder:** Transposed CNNs for images or MLPs for vectors.
- **Dynamics predictor:** MLP that predicts the next latent state from the recurrent state alone.
- **Reward/Continue predictors:** 1-layer MLPs.

**Discrete latent representations:** Instead of continuous Gaussian latents (as in standard VAEs), DreamerV3 uses 32 categorical distributions each with d/16 classes (where d is the model dimension). Sampling uses straight-through gradients.

✔ **Why discrete latents?** They provide a natural bottleneck for information, are more robust to posterior collapse, and enable the use of KL-based free bits more naturally than Gaussian latents.
✔ **Weakness:** Discrete latents may have limited expressivity for fine-grained continuous dynamics.
✔ **Research idea seed:** Hybrid continuous-discrete latent spaces that capture both fine-grained physics and high-level state abstractions.

### Training Objective

Three losses balanced with weights:
1. **Prediction loss (β_pred=1):** Reconstruct observations (symlog squared error for vectors, standard loss for images), predict rewards (symexp twohot), predict episode continuation (logistic regression)
2. **Dynamics loss (β_dyn=1):** KL between encoder posterior and dynamics predictor, with free bits at 1 nat, stop-gradient on encoder
3. **Representation loss (β_rep=0.1):** KL between encoder posterior and dynamics predictor, with free bits at 1 nat, stop-gradient on predictor

**1% uniform mixture ("unimix"):** All categorical distributions (encoder, dynamics, actor) are mixed with 1% uniform to prevent zero probabilities and infinite log-probabilities. This stabilizes training and prevents KL spikes.

✔ **Why this balancing?** The small β_rep prevents representations from losing useful information while still making them predictable. Free bits prevent the KL from dominating when it is already small.
✔ **Weakness:** The reconstruction objective may waste model capacity on visually complex but task-irrelevant details.
✔ **Research idea seed:** Task-aware reconstruction that selectively reconstructs only task-relevant features, potentially improving efficiency in visually complex environments.

## 4.3 Critic Component

### Architecture and Training

- **Network:** 3-layer MLP operating on model states s_t = {h_t, z_t}
- **Output:** Categorical distribution over symexp-spaced bins (same as reward predictor)
- **Target:** λ-returns computed from imagined rewards and bootstrapped critic values
- **Stabilization:** Exponential moving average (EMA) of critic parameters serves as regularization target (decay=0.98)
- **Initialization:** Output weights initialized to zero (prevents hallucinated initial values)

**Dual training signal:** The critic is trained both on imagined trajectories (β_val=1) and on replay buffer trajectories (β_repval=0.3). The replay training uses imagination returns at start states as value annotations.

✔ **Why distributional critic (bins instead of single value)?** Return distributions can be multi-modal (some episodes reach high rewards, others do not). A categorical distribution captures this heterogeneity.
✔ **Weakness:** The fixed bin range [-20, +20] in symexp space may not cover extreme return values in all conceivable environments.
✔ **Research idea seed:** Adaptive bin ranges that expand based on observed return magnitudes.

## 4.4 Actor Component

### Architecture and Training

- **Network:** 3-layer MLP, outputs action distribution
- **Discrete actions:** Categorical distribution (with 1% unimix)
- **Continuous actions:** Typically a squashed Gaussian
- **Gradient estimator:** REINFORCE for both discrete and continuous actions
- **Entropy regularization:** Fixed scale η = 3×10⁻⁴

### Return Normalization (Key Innovation)

Instead of normalizing advantages (as PPO does), DreamerV3 normalizes returns:

```
Normalized_advantage = (R_λ_t - v_t) / max(1, S)
S = EMA(Percentile(R_λ, 95) - Percentile(R_λ, 5), decay=0.99)
```

**Why normalize returns instead of advantages?**
- Advantage normalization puts equal emphasis on exploitation regardless of whether rewards are reachable
- Under sparse rewards, advantage normalization amplifies noise, overwhelming entropy and killing exploration
- Return normalization with max(1, S) preserves the exploration drive when rewards are sparse (S is small, so no scaling occurs)

✔ **Why REINFORCE instead of reparameterization?** Works uniformly for both discrete and continuous action spaces, enabling a single algorithm for all domains.
✔ **Weakness:** REINFORCE has higher gradient variance than reparameterization-based methods.
✔ **Research idea seed:** Variance reduction techniques (control variates, baseline optimization) specifically designed for world model imagination settings.

## 4.5 Simplified Pseudocode

```
Initialize: world model (ϕ), actor (θ), critic (ψ), replay buffer D

For each environment step:
    # 1. ACT
    Encode current observation: z_t ~ encoder(h_t, x_t)
    Update memory: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
    Sample action: a_t ~ actor(h_t, z_t)
    Execute a_t in environment, receive x_{t+1}, r_t, c_t
    Store (x_t, a_t, r_t, c_t) in replay buffer D

    # 2. TRAIN WORLD MODEL
    Sample batch from D: {x_{1:T}, a_{1:T}, r_{1:T}, c_{1:T}}
    Encode all steps: z_t ~ encoder(h_t, x_t)
    Compute prediction loss (reconstruction + reward + continue)
    Compute dynamics KL loss (with free bits, stop-grad on encoder)
    Compute representation KL loss (with free bits, stop-grad on predictor)
    Update ϕ by gradient descent on weighted sum of losses

    # 3. TRAIN ACTOR-CRITIC (IMAGINATION)
    Start from encoded states of batch
    For t = 1 to T_imagine (=15):
        Sample action: a_t ~ actor(s_t)
        Predict next state: h_{t+1} = GRU(h_t, z_t, a_t), z_{t+1} ~ dynamics(h_{t+1})
        Predict reward: r_t ~ reward_predictor(h_t, z_t)
        Predict continue: c_t ~ continue_predictor(h_t, z_t)
    
    Compute λ-returns from imagined rewards and critic values
    Normalize returns: divide advantages by max(1, S) with percentile range
    Update critic (ψ) to predict return distributions
    Update actor (θ) via REINFORCE with entropy bonus
```

---

# 5. Experimental Setup / Evaluation Design

## 5.1 Dataset / Benchmark Characteristics

| Benchmark | Tasks | Env Steps | Action Repeat | Action Type | Input Type | Reward Type |
|---|---|---|---|---|---|---|
| Atari | 57 | 200M | 4 | Discrete | Images (210×160) | Varies |
| ProcGen | 16 | 50M | 1 | Discrete | Images | Varies |
| DMLab | 30 | 100M | 4 | Discrete | Images (3D) | Varies |
| Atari100K | 26 | 400K | 4 | Discrete | Images | Varies |
| Proprio Control | 18 | 500K | 2 | Continuous | Vectors | Dense/Sparse |
| Visual Control | 20 | 1M | 2 | Continuous | Images | Dense/Sparse |
| BSuite | 23 (468 configs) | Varies | 1 | Varies | Varies | Varies |
| Minecraft | 1 | 100M | 1 | Discrete | Images + Vectors | Sparse (12 milestones) |

## 5.2 Experimental Protocol

- **Fixed hyperparameters:** All benchmarks use exactly the same hyperparameter table (Table 4 in paper)
- **Hardware:** Single Nvidia A100 GPU per experiment
- **Seeds:** 5 seeds per benchmark (except ProcGen: 1, BSuite: 10, Minecraft: 10)
- **Error bars:** Mean with one standard deviation shaded
- **Model size:** 200M parameters by default; 12M for control suites (achieves same performance)
- **Replay ratio:** Adjusted per benchmark to match step budget (controls compute vs. data efficiency trade-off)

## 5.3 Metrics and Why They Were Chosen

- **Normalized scores (Atari, DMLab, ProcGen):** Performance relative to random and human/expert baselines. Enables comparison across games with different score ranges.
- **Raw return (Control, Minecraft):** Absolute performance in standardized environments.
- **Aggregate mean/median across tasks:** Tests generality (not just good on some tasks).
- **Item discovery rate (Minecraft):** Fraction of trained agents that discover specific items. Tests exploration capability.
- **Ablation difference scores:** Percentage of full performance retained when removing components.

## 5.4 Baseline Selection Logic

- **PPO:** The "universal baseline" — most widely used RL algorithm, known for robustness. Run with high-quality implementation (Acme framework) and hyperparameters tuned across domains.
- **Domain experts:** Best published results on each benchmark (MuZero for Atari, DrQ-v2 for visual control, etc.). These are often specifically designed and tuned for that domain.
- **Minecraft-specific:** IMPALA and Rainbow additionally tuned because no prior end-to-end learning from scratch had been reported.

## 5.5 Hyperparameter Reasoning

Key hyperparameters and their justification:

| Parameter | Value | Reasoning |
|---|---|---|
| Discount γ | 0.997 (horizon 333) | Long enough for most tasks without excessive variance |
| Batch length T | 64 | Sufficient for temporal dependencies in most domains |
| Imagination horizon H | 15 | Balances computational cost with return estimation quality |
| λ (lambda-return) | 0.95 | Standard value that balances bias-variance |
| Learning rate | 4×10⁻⁵ | Conservatively small for stability across domains |
| Free nats | 1 | Prevents KL domination while ensuring non-trivial representations |
| Entropy η | 3×10⁻⁴ | Works with return normalization to explore appropriately |

### Experimental Reliability Analysis

**What is trustworthy:**
- Results across 8 diverse benchmarks with 150+ tasks provide strong evidence of generality
- Multiple seeds with standard deviations reported
- Open-source code enabling reproduction
- Single GPU requirement makes results verifiable by many labs
- PPO baseline is carefully validated against published reference scores

**What is questionable:**
- ProcGen uses only 1 seed (acknowledged as computational constraint)
- Minecraft comparison: IMPALA and Rainbow were additionally tuned by the authors for this domain, while other domain experts use published numbers — this creates asymmetric comparison effort
- EfficientZero comparison on Atari100K is noted as "difficult" due to different reset strategies
- The 200M model size may not be accessible to all researchers despite single-GPU claim (A100 is high-end)
- Some supplementary benchmark results (pages with memory allocation errors in extraction) may contain additional nuances

---

# 6. Results & Findings Interpretation

## 6.1 Main Outcomes

### Cross-Domain Performance
- **Atari (57 tasks, 200M steps):** Dreamer outperforms MuZero, Rainbow, and IQN while using a fraction of the compute. Significantly outperforms PPO.
- **ProcGen (16 tasks, 50M steps):** Matches the tuned expert PPG algorithm; outperforms Rainbow.
- **DMLab (30 tasks, 100M steps):** Exceeds IMPALA and R2D2+ performance at 10× fewer steps (1000% data efficiency gain).
- **Atari100K (26 tasks, 400K steps):** Outperforms IRIS, TWM, SPR, and SimPLe (excluding EfficientZero with early resets).
- **Proprio Control (18 tasks, 500K steps):** New state-of-the-art, outperforming D4PG, DMPO, MPO.
- **Visual Control (20 tasks, 1M steps):** New state-of-the-art, outperforming DrQ-v2 and CURL.
- **BSuite (23 envs, 468 configs):** New state-of-the-art, especially in scale robustness category.
- **Minecraft Diamond:** First algorithm to collect diamonds from scratch. Mean return 9.1 vs. IMPALA 7.1, Rainbow 6.3, PPO 5.1.

### Key Quantitative Highlights
- 100% of trained Dreamer agents discover diamonds in Minecraft (vs. 0% for all baselines)
- 1000%+ data efficiency gain over scaled baselines on DMLab
- Diamonds obtained in 0.4% of Minecraft episodes at 100M steps

## 6.2 Performance Trends

- Performance increases monotonically with model size (12M to 400M parameters)
- Larger models require fewer environment interactions (better data efficiency)
- Higher replay ratios predictably increase performance
- Model size and replay ratio provide "knobs" for practitioners to trade compute for performance

## 6.3 Failure Cases

- Minecraft diamond rate per episode is only 0.4% at 100M steps — the agent can find diamonds but not consistently
- Some individual Atari games may show lower performance than specialized methods (aggregate statistics hide per-game variation)
- The paper does not report environments where Dreamer fails significantly

## 6.4 Unexpected Observations

- **Reconstruction is more important than reward prediction:** Ablating reconstruction gradients hurts performance more than ablating reward/value gradients. This contradicts most RL algorithms that rely primarily on task-specific signals. This suggests the unsupervised world model objective is the primary driver of good representations.
- **Every robustness technique matters, but only on subsets:** Each technique is critical for some tasks but unnecessary for others. No single technique is universally needed — it is their combination that enables universality.

## 6.5 Statistical Meaning

The consistent outperformance across 8 fundamentally different domains with one configuration is statistically meaningful beyond individual benchmark improvements. The probability of achieving state-of-the-art or competitive results across all domains by chance with a single hyperparameter set is extremely low.

### Publishability Strength Check

**Publication-grade results:**
- The Minecraft diamond achievement alone is a strong publication contribution
- State-of-the-art on Proprio Control, Visual Control, and BSuite with fixed hyperparameters
- Matching or exceeding tuned expert algorithms across all domains

**Results needing stronger validation:**
- ProcGen with 1 seed is insufficient for confident claims
- Per-task Atari analysis would strengthen understanding of where Dreamer excels or struggles
- Real-world robotic validation is missing

---

# 7. Strengths – Weaknesses – Assumptions

## Table 1: Technical Strengths

| # | Strength | Evidence |
|---|---|---|
| 1 | Single configuration across 150+ tasks in 8 domains | Tables and figures across all benchmarks |
| 2 | State-of-the-art on multiple benchmarks simultaneously | Direct comparisons with tuned expert algorithms |
| 3 | First to collect diamonds in Minecraft from scratch | Figure 5: 100% of agents find diamonds |
| 4 | Predictable scaling with model size and replay ratio | Figure 6c,d: monotonic improvement curves |
| 5 | Data efficiency (1000% gain on DMLab vs. scaled baselines) | DMLab results: 100M steps vs. 1B steps |
| 6 | Low compute requirement (single A100 GPU) | All experiments reproducible on one GPU |
| 7 | Open-source implementation | Public code availability |
| 8 | Principled robustness techniques with clear mathematical motivation | Symlog/twohot/return normalization derivations |
| 9 | Thorough ablation study showing each component's contribution | Figure 6a,b: systematic ablation on 14 tasks |
| 10 | Unsupervised world model as primary representation learning signal | Ablation: removing reconstruction hurts more than removing reward gradients |

## Table 2: Explicit Weaknesses

| # | Weakness | Impact |
|---|---|---|
| 1 | Minecraft diamond per-episode rate is only 0.4% | Agent discovers the ability but cannot reliably execute the 12-step plan |
| 2 | Imagination errors compound over long horizons | Limits planning for tasks requiring very long-term reasoning |
| 3 | REINFORCE gradient estimator has high variance | May limit performance in environments where low-variance gradients would help |
| 4 | No hierarchical structure | Flat policy must learn both high-level strategy and low-level control |
| 5 | Reconstruction of full observations may waste capacity | In visually complex worlds, most visual details are irrelevant to the task |
| 6 | No real-world robotic evaluation | Sim-to-real gap is unaddressed |
| 7 | Fixed discrete latent structure may be suboptimal for some domains | Continuous physics may require finer-grained representations |
| 8 | Does not leverage language or semantic knowledge | Unlike VPT/Voyager, no external knowledge integration |

## Table 3: Hidden Assumptions

| # | Assumption | Risk |
|---|---|---|
| 1 | Markovian latent space is sufficient | May fail in environments requiring arbitrarily long memory beyond GRU capacity |
| 2 | Reconstruction signal correlates with task-relevant features | Could fail if task-relevant information is a tiny fraction of observation |
| 3 | Symlog compression is appropriate for all target distributions | May not be optimal for targets with specific distributional properties |
| 4 | Percentile-based return normalization captures meaningful scale | May be unreliable with very few samples or highly non-stationary environments |
| 5 | Fixed imagination horizon of 15 steps is sufficient | Tasks with reward delays longer than 15 steps rely entirely on critic accuracy |
| 6 | Replay buffer with uniform sampling is sufficient | Prioritized replay improved performance but was excluded for simplicity |
| 7 | Single-agent, single-environment training paradigm | No multi-task or multi-agent learning |

---

# 8. Weakness → Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| Low per-episode diamond rate (0.4%) | 12-step technology tree requires consistent long-horizon execution | Hierarchical world models with subgoal discovery | Option framework over learned world model; temporal abstraction in latent space |
| Imagination error compounding | World model predictions degrade with distance from real data | Uncertainty-aware imagination with adaptive horizon | Ensemble world models with disagreement-based horizon truncation |
| Full observation reconstruction wastes capacity | Reconstruction loss treats all pixels equally | Selective reconstruction or contrastive learning in world models | Masked world models that reconstruct only task-relevant patches; BYOL-style self-supervised objectives |
| No language/semantic knowledge | System learns purely from interaction | Language-conditioned world models | Integrate language embeddings into the latent space; use LLMs for goal specification |
| No hierarchical planning | Flat policy over primitive actions | Hierarchical imagination with learned subgoals | Learn latent subgoal space; train high-level policy to set subgoals for low-level world model |
| No real-world validation | Sim-to-real gap unaddressed | Sim-to-real transfer with world model adaptation | Domain randomization in imagination; fine-tune world model on limited real data |
| REINFORCE high variance | Using the same gradient estimator for all action types | Action-type-specific gradient estimators within unified framework | Reparameterization for continuous + REINFORCE for discrete with learned baselines |
| Fixed latent structure | One discrete categorical structure for all domains | Adaptive latent representations | Learnable number of latent dimensions; mixture of continuous and discrete latents |

---

# 9. Novel Contribution Extraction

## Explicit Contribution Statements from the Paper

1. "We propose DreamerV3, a general reinforcement learning algorithm that outperforms specialized methods across 150+ diverse tasks with a single configuration, eliminating the need for domain-specific hyperparameter tuning."

2. "We introduce symlog predictions and symexp twohot losses that decouple gradient magnitudes from target scales, enabling stable learning across environments with reward scales differing by orders of magnitude."

3. "We propose percentile-based return normalization with a lower limit that maintains exploration under sparse rewards while converging to high performance under dense rewards, using a fixed entropy scale."

4. "We demonstrate that reconstruction-based world model learning provides the primary representation learning signal, suggesting that future RL algorithms can leverage unsupervised pretraining on unlabeled data."

5. "We achieve the first successful collection of diamonds in Minecraft from scratch using reinforcement learning alone, without human data or curricula, using a single GPU."

## Possible Novel Claim Templates for Extension Papers

1. "We propose ______ that improves DreamerV3's long-horizon planning by ______, achieving ____% higher per-episode success rates on sparse-reward tasks."

2. "We propose ______ that replaces DreamerV3's full reconstruction objective with ______, reducing world model capacity requirements by ____% while maintaining performance."

3. "We propose ______ that integrates language-conditioned goals into the DreamerV3 world model, enabling ______ without additional reward engineering."

4. "We propose ______ that extends DreamerV3's robustness techniques to multi-agent settings, demonstrating fixed-hyperparameter generality across ______ cooperative and competitive domains."

5. "We propose ______ that pretrains DreamerV3's world model on unlabeled video data, improving downstream RL performance by ____% with ____% fewer environment interactions."

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Work

- Teaching agents world knowledge from internet videos (leveraging the unsupervised reconstruction objective)
- Learning a single world model across multiple domains simultaneously (universal world model)
- Building up increasingly general knowledge and competency across tasks

## 10.2 Missing Directions Not Addressed by Authors

- **Safety and constraints:** No mechanism for avoiding dangerous states during exploration
- **Multi-agent settings:** All experiments are single-agent
- **Continual learning:** No evaluation of catastrophic forgetting when sequentially learning multiple tasks
- **Interpretability:** No analysis of what the world model actually learns to represent
- **Offline RL:** Using DreamerV3's world model for learning from fixed datasets without further interaction

## 10.3 Modern Extensions (Post-2023)

- **Foundation world models:** Pretraining world models on diverse internet video data and fine-tuning for specific tasks
- **Diffusion-based world models:** Replacing the RSSM with diffusion models for higher-fidelity imagination
- **Transformer-based sequence models:** Replacing the GRU with transformers for better long-range memory
- **VLM integration:** Using vision-language models as initialization or auxiliary signal for world model representations

## 10.4 Cross-Domain Combinations

- **Robotics + Language:** DreamerV3 for robotic manipulation with language goal specification
- **World models + LLM agents:** Using DreamerV3-style imagination as a "physics engine" for LLM-based planning agents
- **World models + Neuroscience:** Comparing DreamerV3's learning dynamics with biological hippocampal replay and imagination

## 10.5 LLM-Era Extensions

- **LLM-guided exploration:** Use an LLM to suggest promising actions or subgoals that the world model evaluates
- **World model as LLM tool:** Allow LLMs to query a trained world model to predict consequences of proposed plans
- **Unified world-language models:** Joint training of language understanding and physical world prediction in a single model

---

# 11. How to Write a NEW Paper From This Work

## 11.1 Reusable Elements

- **Symlog/symexp transformations:** Applicable to ANY machine learning problem with varying target scales (not just RL)
- **Twohot distributional prediction:** Reusable for any regression task where the target distribution may be multi-modal
- **Percentile return normalization:** Applicable to any RL algorithm that balances exploitation and exploration
- **Free bits + small KL weight:** Reusable in any VAE-based model for balancing reconstruction and regularization
- **Evaluation methodology:** Testing with fixed hyperparameters across diverse domains as the standard for generality claims
- **Ablation design:** Systematic removal of individual components to prove necessity

## 11.2 What MUST NOT Be Copied

- The RSSM architecture itself (directly copying it without modification is not a novel contribution)
- The specific hyperparameter table (using identical values would be replication, not research)
- The exact benchmark selection and comparison tables
- Sentences or phrasing from the paper (always paraphrase)

## 11.3 How to Design a Novel Extension

1. **Identify a specific weakness** from Section 8 that aligns with your expertise
2. **Propose a concrete solution** with theoretical motivation
3. **Test on at least 3 diverse domains** (following DreamerV3's philosophy of generality)
4. **Compare against DreamerV3 as baseline** using their open-source code
5. **Show that your extension does not break generality** (improvement on target domain without degradation on others)
6. **Ablate your addition** to prove it is necessary

### Minimum Publishable Contribution Checklist

- [ ] Clear problem statement identifying a specific limitation of DreamerV3
- [ ] Novel method with theoretical or empirical justification
- [ ] Experiments on at least 3 diverse domains (not just one)
- [ ] Direct comparison with DreamerV3 using their code
- [ ] Ablation study proving each component is necessary
- [ ] Analysis of failure cases and limitations of your extension
- [ ] Open-source code for reproducibility

---

# 12. Complete Paper Writing Template

## Abstract (150-250 words)

**Purpose:** Concisely state problem, method, key result, and significance.

**What to include:**
- One sentence on the problem (e.g., "World model-based RL agents struggle with [specific limitation]")
- One sentence on your approach (e.g., "We propose [method name] that [key idea]")
- Two sentences on results (e.g., "Our method improves [metric] by [X%] on [benchmarks] while maintaining generality across [N] domains")
- One sentence on significance (e.g., "This enables [practical impact]")

**Common mistakes:** Being too vague ("we improve RL"), including unnecessary background, burying the main result.

**Reviewer expectations:** The abstract should make the contribution and its magnitude immediately clear.

## 1. Introduction (1-1.5 pages)

**Purpose:** Motivate the problem, position your work, and state contributions.

**Structure:**
1. Broad problem context (1 paragraph)
2. Specific challenge your work addresses (1 paragraph)
3. Why existing approaches (including DreamerV3) fall short (1 paragraph)
4. Your approach and key insight (1 paragraph)
5. Summary of contributions as a bulleted list (3-5 items)

**Common mistakes:** Too much general RL background, not clearly stating what is new, contributions that restate method steps instead of outcomes.

**Reviewer expectations:** By the end of the introduction, the reviewer should know exactly what gap exists and how you fill it.

## 2. Related Work (1-1.5 pages)

**Purpose:** Position your work relative to prior art and clarify differences.

**What to include:**
- World model-based RL (DreamerV1-V3, MuZero, SimPLe, IRIS, TWM)
- Model-free RL baselines relevant to your domains (PPO, SAC, etc.)
- Specific related work for your extension (e.g., hierarchical RL if you add hierarchy)
- Clear differentiation statements ("Unlike [X], our method [Y]")

**Common mistakes:** Listing papers without relating them to your work, missing important references, not explaining how your work differs from the closest approach.

**Reviewer expectations:** The reviewer should understand the landscape and see exactly where your work fits.

## 3. Background (0.5-1 page)

**Purpose:** Define notation and recap DreamerV3 components your work builds on.

**What to include:**
- RSSM formulation (Equation 1 from DreamerV3)
- World model loss (relevant parts of Equations 2-3)
- Actor-critic setup (Equations 4-7 as relevant)
- Only include what is necessary for understanding your extensions

**Common mistakes:** Rewriting the entire DreamerV3 paper, introducing notation you never use.

**Reviewer expectations:** Brief, precise, and directly supporting the method section.

## 4. Method (2-3 pages)

**Purpose:** Present your novel contribution with sufficient detail for reproduction.

**Structure:**
1. Overview of your approach (1 paragraph + figure)
2. Detailed description of each novel component
3. Mathematical formulation where applicable
4. Algorithm pseudocode
5. Relationship to DreamerV3 (what changes, what stays the same)

**Common mistakes:** Not enough detail for reproduction, mixing method description with experimental results, not justifying design choices.

**Reviewer expectations:** A reader should be able to implement your method from this section alone.

## 5. Theoretical Analysis (Optional, 0.5-1 page)

**Purpose:** Provide formal guarantees or analysis of your method.

**What to include:**
- Convergence properties
- Computational complexity analysis
- Bounds on improvement

**Reviewer expectations:** Formal analysis strengthens the paper significantly but is not always required for empirical contributions.

## 6. Experiments (2-3 pages)

**Purpose:** Provide evidence that your method works and understand why.

**Structure:**
1. Experimental setup (domains, baselines, metrics, compute)
2. Main results (tables and figures comparing to DreamerV3 and other baselines)
3. Ablation study (each component's contribution)
4. Analysis (why your method works, when it fails)

**Common mistakes:** Cherry-picking domains, insufficient baselines, no ablation, no error bars, not reporting failures.

**Reviewer expectations:** Fair comparisons, statistical significance, ablation, and analysis of both successes and failures.

## 7. Discussion (0.5 page)

**Purpose:** Interpret results, discuss implications, connect to broader themes.

**What to include:**
- Why your method works (high-level insight)
- Surprising findings
- Implications for the field

**Common mistakes:** Repeating results without interpretation, overclaiming.

## 8. Limitations (0.5 page)

**Purpose:** Honestly acknowledge weaknesses.

**What to include:**
- Domains or scenarios where your method may not work
- Computational costs
- Assumptions that may not hold generally

**Reviewer expectations:** Honest limitations increase trust in the paper. Reviewers respect authors who identify their own weaknesses.

## 9. Conclusion (0.5 page)

**Purpose:** Summarize contributions and suggest future work.

**What to include:**
- Restate the problem and your solution (1-2 sentences)
- Key findings (1-2 sentences)
- Future directions (1-2 sentences)

**Common mistakes:** Introducing new information, being too lengthy, excessive speculation.

## References

**What to include:** All cited works in consistent format. Ensure DreamerV3 and all baseline papers are cited. Use conference versions over arXiv where available.

---

# 13. Publication Strategy Guide

## 13.1 Suitable Venues

**Top-tier conferences (high bar, high impact):**
- **NeurIPS** (Neural Information Processing Systems) — General ML, strong RL track
- **ICML** (International Conference on Machine Learning) — Core ML venue
- **ICLR** (International Conference on Learning Representations) — Strong for representation learning and RL

**Robotics-focused (if extension involves real-world):**
- **CoRL** (Conference on Robot Learning) — If robotic applications are primary
- **RSS** (Robotics: Science and Systems)

**Journals:**
- **JMLR** (Journal of Machine Learning Research) — Extended version with additional experiments
- **Nature Machine Intelligence** — If results are sufficiently groundbreaking and broadly appealing

## 13.2 Required Baseline Expectations

- **Must compare to:** DreamerV3 (the direct predecessor), PPO (universal baseline), and domain-specific experts
- **Must use:** DreamerV3's open-source code for fair comparison
- **Must show:** Results on at least 3 diverse domains with multiple seeds
- **Nice to have:** Computational cost comparison (FLOPs, GPU hours)

## 13.3 Experimental Rigor Level

- **Minimum:** 3 domains, 3 seeds, error bars, ablation of your novel components
- **Standard:** 5+ domains, 5 seeds, comprehensive ablation, failure analysis
- **Gold standard:** 8+ domains (matching DreamerV3's coverage), 5+ seeds, ablation, scaling analysis, computational cost breakdown

## 13.4 Common Rejection Reasons

1. **"Incremental contribution"** — Simply tuning DreamerV3 or making minor architectural changes without principled motivation
2. **"Limited evaluation"** — Testing on only 1-2 domains (violates the generality spirit of DreamerV3)
3. **"Unfair comparison"** — Giving your method more compute/tuning than baselines
4. **"Missing ablation"** — Not proving that each component of your extension is necessary
5. **"No analysis of failures"** — Only reporting successes without discussing limitations
6. **"Unclear novelty"** — Not distinguishing clearly from DreamerV3 and concurrent work

## 13.5 Increment Needed for Acceptance

- **New robustness technique:** Must show improvement on domains where DreamerV3 struggles, without degradation elsewhere
- **Architectural improvement:** Must demonstrate significant performance gain (>5% across domains) or substantial efficiency improvement (>2× data or compute efficiency)
- **New capability:** Must enable something DreamerV3 cannot do (e.g., multi-agent, language conditioning, hierarchical planning) while maintaining competitive base performance
- **Theoretical contribution:** Must provide formal analysis that explains or predicts DreamerV3's behavior and suggests principled improvements

---

# 14. Researcher Quick Reference Tables

## 14.1 Key Terminology Table

| Term | Meaning in This Paper |
|---|---|
| World model | Neural network that learns environment dynamics and predicts future states, rewards, and continuations |
| RSSM | Recurrent State-Space Model — the specific world model architecture combining deterministic recurrent state and stochastic latent state |
| Imagination / Dreaming | Generating trajectories using only the world model (no real environment interaction) |
| Model state s_t | Concatenation of recurrent state h_t and latent state z_t |
| Symlog | sign(x)·ln(\|x\|+1) — symmetric logarithmic compression function |
| Symexp | sign(x)·(exp(\|x\|)-1) — inverse of symlog |
| Twohot encoding | Soft label over two adjacent bins for continuous target representation |
| Free bits | KL divergence clipping below 1 nat to prevent regularization from dominating |
| Unimix | Mixing 99% network output with 1% uniform distribution to prevent zero probabilities |
| Return normalization | Dividing advantages by percentile range max(1,S) for scale-invariant actor gradients |
| Replay ratio | Number of training steps per environment step (controls compute-data trade-off) |
| λ-returns | Bootstrapped return estimate blending actual rewards with critic value predictions |
| Dynamics predictor | Predicts next latent state from recurrent state alone (used during imagination) |
| Encoder | Maps observations to latent states (used only when real observations are available) |
| Continue predictor | Predicts probability of episode continuation (vs. termination) |

## 14.2 Important Equations Summary

| Eq. | Name | Formula (simplified) | Purpose |
|---|---|---|---|
| 1 | RSSM | h_t = f(h_{t-1}, z_{t-1}, a_{t-1}); z_t ~ q(z\|h,x) | World model state transitions |
| 2 | World model loss | L = Σ(β_pred·L_pred + β_dyn·L_dyn + β_rep·L_rep) | Total training objective |
| 3 | KL losses | L_dyn = max(1, KL[sg(q)\|\|p]); L_rep = max(1, KL[q\|\|sg(p)]) | Balance representation quality |
| 5 | Critic loss | L = -Σ ln p_ψ(R^λ_t \| s_t) | Train distributional critic |
| 6 | Actor loss | L = -Σ sg(normalized advantage) · log π(a\|s) + η·H(π) | Policy gradient with entropy |
| 7 | Return range | S = EMA(Per(R,95) - Per(R,5), 0.99) | Robust return scale estimate |
| 8 | Symlog loss | L = ½(f(x,θ) - symlog(y))² | Scale-robust regression |
| 9 | Symlog/symexp | symlog(x) = sign(x)·ln(\|x\|+1) | Compression/decompression |
| 10 | Twohot prediction | ŷ = softmax(f(x))ᵀ·B, B = symexp({-20,...,+20}) | Distributional prediction |
| 11 | Twohot loss | L = -twohot(y)ᵀ·log softmax(f(x,θ)) | Scale-independent regression |

## 14.3 Parameter Meaning Table

| Parameter | Symbol | Value | Meaning |
|---|---|---|---|
| Prediction loss weight | β_pred | 1 | Importance of observation/reward reconstruction |
| Dynamics loss weight | β_dyn | 1 | Importance of dynamics prediction accuracy |
| Representation loss weight | β_rep | 0.1 | Importance of representation predictability (kept small) |
| Critic value loss weight | β_val | 1 | Importance of imagination value learning |
| Critic replay loss weight | β_repval | 0.3 | Importance of replay-based value learning |
| Entropy regularizer | η | 3×10⁻⁴ | Exploration incentive (fixed across all domains) |
| Discount factor | γ | 0.997 | How much future rewards are valued (horizon ≈ 333 steps) |
| Lambda return | λ | 0.95 | Blend of actual vs. bootstrapped returns |
| Imagination horizon | H | 15 | Steps of imagination for actor-critic training |
| Batch size | B | 16 | Number of trajectory sequences per batch |
| Batch length | T | 64 | Length of each trajectory sequence |
| Learning rate | — | 4×10⁻⁵ | Gradient step size for all networks |
| Return norm limit | L | 1 | Minimum denominator to prevent noise amplification |
| Return norm decay | — | 0.99 | EMA decay for return range estimation |
| Critic EMA decay | — | 0.98 | Slow target network update rate |
| Free nats | — | 1 | KL clipping threshold |
| Unimix | — | 1% | Uniform mixture for preventing zero probabilities |

## 14.4 Algorithm Flow Summary

```
┌─────────────────────────────────────────────────────┐
│                  ENVIRONMENT                         │
│  x_t, r_t, c_t ← env.step(a_t)                     │
└────────────┬───────────────────────────┬─────────────┘
             │ store                     │ observe
             ▼                           │
┌─────────────────────┐                  │
│   REPLAY BUFFER     │                  │
│   (uniform + queue) │                  │
└─────────┬───────────┘                  │
          │ sample batch                 │
          ▼                              │
┌─────────────────────────────────────┐  │
│       WORLD MODEL TRAINING          │  │
│                                     │  │
│  Encoder: x_t → z_t                │  │
│  Sequence: (h,z,a) → h'            │  │
│  Losses:                            │  │
│    L_pred (reconstruct x, predict r,c) │
│    L_dyn  (KL with free bits)       │  │
│    L_rep  (KL with free bits)       │  │
│                                     │  │
│  Key techniques:                    │  │
│    • symlog transform on vectors    │  │
│    • symexp twohot for rewards      │  │
│    • 1% unimix on categoricals      │  │
│    • LaProp + AGC optimizer         │  │
└─────────┬───────────────────────────┘  │
          │ imagine trajectories         │
          ▼                              │
┌─────────────────────────────────────┐  │
│    ACTOR-CRITIC IN IMAGINATION      │  │
│                                     │  │
│  For H=15 steps:                    │  │
│    a ~ actor(s_t)                   │  │
│    s_{t+1} = dynamics(s_t, a_t)    │  │
│    r_t = reward_pred(s_t)           │  │
│                                     │  │
│  Compute λ-returns                  │  │
│  Normalize returns (percentile)     │  │
│  Train critic (distributional)      │  │
│  Train actor (REINFORCE + entropy)  │  │
└─────────┬───────────────────────────┘  │
          │ updated policy               │
          ▼                              │
┌─────────────────────┐                  │
│   ACTOR NETWORK     │←─────────────────┘
│   a_t ~ π(·|s_t)   │
└─────────────────────┘
```

---

# 15. One-Page Master Summary Card

## Problem
Current RL algorithms require domain-specific hyperparameter tuning. No single algorithm masters diverse tasks (continuous/discrete actions, image/vector inputs, dense/sparse rewards, 2D/3D worlds) with fixed configuration.

## Key Idea
Learn a world model (RSSM) that imagines future outcomes; train actor-critic purely in imagination. Apply robustness techniques (symlog, twohot, return normalization, KL balancing) to make gradients independent of target scales and reward structures.

## Method
Three concurrent components trained from replayed experience:
1. **World model (RSSM):** Encodes observations → discrete latents, predicts dynamics/rewards/continuations via reconstruction + KL losses with free bits and unimix
2. **Critic:** Distributional (symexp twohot bins), trained on imagined λ-returns with EMA regularization
3. **Actor:** REINFORCE with entropy, percentile-normalized returns with max(1,S) limit

## Results
- Outperforms specialized expert algorithms across 8 domains (150+ tasks) with ONE hyperparameter set
- First algorithm to collect diamonds in Minecraft from scratch (1 GPU, 9 days, no human data)
- State-of-the-art on Proprio Control, Visual Control, BSuite
- Performance scales monotonically with model size (12M–400M) and replay ratio

## Weakness
- Minecraft per-episode diamond rate only 0.4%
- Imagination errors compound over long horizons
- Full reconstruction may waste capacity on task-irrelevant details
- No hierarchical planning, language integration, or real-world validation

## Research Opportunity
- Hierarchical world models for long-horizon tasks
- Unsupervised pretraining of world models on video data
- Language-conditioned imagination for instruction following
- Selective reconstruction focusing on task-relevant features
- Uncertainty-aware adaptive imagination horizons

## Publishable Extension
Combine DreamerV3's world model with hierarchical subgoal discovery to improve long-horizon sparse-reward tasks. Train a high-level policy to set subgoals in latent space and a low-level DreamerV3 policy to achieve them. Evaluate on Minecraft (per-episode diamond rate), Crafter, and DMLab navigation tasks. Target: NeurIPS/ICML with generality across at least 3 domains.
