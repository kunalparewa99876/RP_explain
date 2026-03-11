# Research Companion: Federated Learning in Mobile Edge Networks — A Comprehensive Survey
**Lim et al., 2020 — IEEE Communications Surveys & Tutorials**

---

# 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Paper Title** | Federated Learning in Mobile Edge Networks: A Comprehensive Survey |
| **Authors** | Wei Yang Bryan Lim, Nguyen Cong Luong, Dinh Thai Hoang, Yutao Jiao, Ying-Chang Liang, Qiang Yang, Dusit Niyato, Chunyan Miao |
| **Year** | 2020 |
| **Problem Domain** | Federated Learning (FL) in Mobile Edge Computing (MEC) networks |
| **Paper Type** | Survey / Conceptual with Systems Engineering Components |
| **Core Contribution** | First comprehensive survey bridging FL implementation challenges AND FL applications for MEC optimization |
| **Key Idea** | FL enables privacy-preserving, collaborative ML model training at the network edge — this paper surveys every angle: foundations, communication costs, resource allocation, privacy/security, and edge network applications |
| **Required Background** | Basic ML/DNN training, gradient descent, mobile network architecture, basic privacy concepts |
| **Primary Baseline** | FedAvg (McMahan et al., 2017) — the foundational FL algorithm |
| **Main Innovation Type** | Survey synthesis — identifies gaps, maps challenges, and creates a unified research framework |
| **Difficulty Level** | Intermediate to Advanced |
| **Reproducibility Level** | Moderate — references open-source frameworks (TFF, PySyft, LEAF, FATE) |

---

# 1. Research Context & Core Problem

## What Problem Does This Paper Address?

Mobile devices generate enormous amounts of data every day — photos, health readings, location traces, and text. Traditional cloud-based machine learning requires all this raw data to be sent to a central server for model training. This creates three serious problems:

- **Privacy violation**: Personal and sensitive data leaves the user's device
- **Latency**: Sending data to the cloud and receiving decisions back takes too long for real-time applications
- **Bandwidth overload**: The backbone network gets congested when millions of devices upload data

Mobile Edge Computing (MEC) partially solves the latency issue by bringing computation closer to users. However, even in MEC, data still had to leave users' devices to reach edge servers.

**Federated Learning (FL)** solves this by flipping the approach: instead of sending data to the model, the model comes to the data. Each device trains locally and sends only model parameter updates (not raw data) to a server for aggregation.

## Why the Problem Exists

- Traditional distributed ML was designed for controlled data center environments — not heterogeneous wireless networks
- GDPR and similar privacy laws legally restrict data sharing
- IoT and mobile device proliferation created an enormous decentralized data ecosystem
- Existing surveys treat FL and MEC as separate topics — no unified treatment existed before this paper

## Historical and Theoretical Gap

| What Existed | What Was Missing |
|---|---|
| Surveys on FL algorithms and settings (Yang et al., 2019) | FL-specific implementation challenges in edge networks |
| Surveys on MEC architecture and offloading | FL as an optimization tool for MEC |
| Individual papers on privacy/communication in FL | Systematic comparison across all FL challenge dimensions |
| FL application papers | Unified framework connecting challenges to solutions |

## Contribution Category

- **Survey synthesis** — not a new algorithm, but a structured mapping of the field
- **System design insights** — analyzes implementation challenges
- **Research roadmap** — identifies open problems and future directions

### Why This Paper Matters

This is the most cited comprehensive bridge between FL theory and edge network deployment. It serves as a reference map: if you want to work on FL for real-world networks, this paper tells you what has been done, what works, and what is still open. Every subsequent FL-for-wireless paper cites this work.

### Remaining Open Problems (Identified by This Paper)

- Non-IID data degradation without ground truth IID datasets
- Communication-accuracy tradeoff formalization
- Fair and privacy-preserving participant selection at scale
- Asynchronous FL with convergence guarantees for non-convex, non-IID settings
- Multiple competing FL servers in the same network
- Intelligent incentive mechanisms without revealing private data quality
- Secure aggregation under adaptive adversarial models
- Personalised FL for heterogeneous user preferences

---

# 2. Minimum Background Concepts

## 2.1 Deep Neural Networks (DNNs)

**Plain definition**: A layered computational structure where each layer transforms inputs through weighted sums and non-linear activation functions.

**Role in this paper**: FL is primarily studied in the context of DNN model training — participants train DNN layers locally and share learned weights.

**Why authors needed it**: FedAvg is designed for SGD-based training, which is the foundation of DNN optimization.

---

## 2.2 Stochastic Gradient Descent (SGD)

**Plain definition**: An iterative method to minimize a loss function by updating model weights in small steps proportional to the gradient.

**SGD update rule**:
$$w \leftarrow w - \eta \cdot \nabla_w L(w; b)$$

Where:
- $w$ = model weights
- $\eta$ = learning rate (step size)
- $L$ = loss function
- $b$ = mini-batch of training samples
- $\nabla_w L$ = gradient of loss with respect to weights

**Role in paper**: Local training at each FL participant uses SGD. The FedAvg algorithm builds directly on local SGD.

---

## 2.3 Mobile Edge Computing (MEC)

**Plain definition**: A network architecture where computing servers are placed at the network edge (base stations, access points) rather than far-away cloud data centers.

**Role in paper**: MEC provides the physical infrastructure where FL is deployed. Edge servers act as intermediate aggregators in hierarchical FL.

**Why authors needed it**: FL at scale requires a network model — MEC provides the 3-tier model: end devices → edge nodes → cloud server.

---

## 2.4 Non-IID Data

**Plain definition**: When different participants hold data that follows different statistical distributions. "IID" means each sample is drawn from the same distribution independently.

**Role in paper**: Non-IID is the core statistical challenge of FL. Real-world devices hold data reflecting their individual users — hospital A sees mostly elderly patients; hospital B sees mostly young athletes.

**Why it matters**: FedAvg's accuracy drops significantly (51% accuracy loss on CIFAR-10) compared to centralized training when data is non-IID.

---

## 2.5 Differential Privacy (DP)

**Plain definition**: A mathematical privacy guarantee that adding random noise to model updates prevents an adversary from inferring whether any individual's data was used in training.

**Role in paper**: DP is a key defense mechanism reviewed for protecting FL participants' private information during model sharing.

---

## 2.6 Federated Averaging (FedAvg)

**Plain definition**: The baseline FL algorithm where each participant trains locally for several gradient steps, then sends weights to the server, which averages them weighted by dataset size.

**FedAvg aggregation**:
$$w_G^t = \frac{1}{\sum_{i \in \mathcal{N}} |D_i|} \sum_{i=1}^{N} |D_i| \cdot w_i^t$$

**Role in paper**: FedAvg is the baseline that virtually every improvement in the paper is compared against.

---

# 3. Mathematical / Theoretical Understanding Layer

## 3.1 Global Loss Function

The FL server minimizes the global loss:

$$L(w_G^t) = \sum_{i \in \mathcal{N}} \frac{|D_i|}{\sum_j |D_j|} L_i(w_i^t)$$

| Symbol | Meaning |
|---|---|
| $w_G^t$ | Global model at iteration $t$ |
| $D_i$ | Dataset of participant $i$ |
| $L_i(w_i^t)$ | Local loss function at participant $i$ |
| $\mathcal{N}$ | Set of all participants |

**Intuition**: The server wants the combined model to perform well on all participants' data. Each participant's contribution is weighted by how much data they have.

**Assumption**: Data owners are honest — they use real data and submit true gradients.

**Practical interpretation**: If a hospital has more patient records, its model update naturally receives more weight in the global model.

---

## 3.2 q-Fair Federated Learning Objective

$$\min_w \sum_{k=1}^{N} p_k F_k^{q+1}(w)$$

| Symbol | Meaning |
|---|---|
| $p_k$ | Ratio of local samples of participant $k$ to total samples |
| $F_k$ | Standard loss for participant $k$ (see Table III in paper) |
| $q$ | Fairness calibration parameter |

**Intuition**: When $q = 0$, this reduces to standard FL. Increasing $q$ gives more weight to participants who are performing worse (higher loss), forcing the model to attend to underrepresented participants.

**Purpose**: Prevent the FL model from becoming biased toward participants with better hardware or more data.

---

## 3.3 Loss Functions for Common ML Models

| Model | Loss Function |
|---|---|
| Neural Network | $\frac{1}{n}\sum_{j=1}^n (y_j - f(x_j; w))^2$ (MSE) |
| Linear Regression | $\frac{1}{2}\|y_j - w^T x_j\|^2$ |
| Squared-SVM | $\frac{1}{n}\sum_j \max(0, 1 - y_j(w^T x_j - \text{bias})) + \lambda \|w^T\|^2$ |

---

### Mathematical Insight Box

> **Key Idea to Remember**: FL does not directly minimize a global loss on centralized data. Instead, it minimizes a weighted average of local losses — and non-IID data breaks the assumption that these local losses align with the global optimum. This misalignment is why non-IID data is such a persistent challenge.

---

# 4. Proposed Method / Framework

## 4.1 Paper Structure Overview

This is a survey, so the "method" is the survey framework itself. The paper proposes a two-dimensional classification:

1. **FL at Mobile Edge Network** — solving implementation challenges of FL on edge devices
2. **FL for Mobile Edge Network** — using FL as a tool to optimize the network itself

---

## 4.2 Core FL Training Pipeline (Tutorial)

```
STEP 1: Task Initialization
├── Server decides: target task, data requirements, hyperparameters
├── Server initializes global model w_G^0
└── Server broadcasts model and task to selected participants

STEP 2: Local Model Training (at each participant i)
├── Receive global model w_G^t
├── Split local dataset D_i into mini-batches of size B
├── For each local epoch (1 to E):
│   └── For each mini-batch b: w ← w - η·∇L(w; b)
└── Send updated local parameters w_i^{t+1} to server

STEP 3: Global Model Aggregation (at server)
├── Collect local updates from all participants
├── Compute weighted average: w_G^{t+1} = Σ(|D_i|/Σ|D_j|)·w_i^{t+1}
└── Broadcast new global model back to participants

REPEAT STEPS 2–3 until:
- Global loss converges, OR
- Target accuracy is achieved
```

**Why each step is designed this way**:
- Local training (Step 2) reduces communication rounds by doing multiple gradient steps before each aggregation
- Weighted averaging (Step 3) ensures larger datasets contribute more to the global model

---

## 4.3 Communication Cost Reduction Methods

### 4.3.1 Edge and End Computation

**What it does**: Increases local computation per participant before each communication round, reducing total communication rounds needed.

| Method | How it Works | Weakness |
|---|---|---|
| FedAvg (McMahan et al.) | More local updates per round | Poor convergence with non-IID data |
| Two-stream FL | Global model as fixed reference during local training (MMD loss term) | High computation cost on devices |
| HierFAVG | Intermediate edge server aggregation before cloud aggregation | Doesn't converge to 90% accuracy with many edge servers and non-IID data |

---

### 4.3.2 Model Compression

**What it does**: Reduces the size of model updates transmitted each round.

| Method | Technique | Compression Rate | Accuracy Impact |
|---|---|---|---|
| Structured Updates (low rank) | Express update as product of two matrices | Moderate | Acceptable |
| Structured Updates (random mask) | Sparse update with random pattern | Better | Good |
| Probabilistic Quantization | Quantize scalars to fewer bits (e.g., 2 bits) | 256× fewer bits | ~85% accuracy |
| Federated Dropout | Remove activation functions to create smaller sub-model | ~43% reduction | 25% dropout acceptable |

**Design choice reasoning**: Direct data compression reduces upload size immediately. The tradeoff is accuracy loss and potentially slower convergence.

**Improvement opportunities**:
- Formal accuracy-compression tradeoff curves
- Adaptive compression that adjusts based on model convergence state

---

### 4.3.3 Importance-based Updating

**What it does**: Selectively transmits only "important" gradient updates.

| Algorithm | Selection Method | Communication Saving |
|---|---|---|
| eSGD | Track loss history; transmit gradients that reduce loss | Significant |
| CMFL | Compare local update sign-similarity with prior global update | 3.47×–13.97× fewer rounds |

---

## 4.4 Resource Allocation Methods

### 4.4.1 Participant Selection

| Protocol | Key Mechanism | Tradeoff |
|---|---|---|
| FedCS | Select max participants that complete within deadline (greedy) | Bias toward high-spec devices |
| Hybrid-FL | Select for IID-approximate distributed data + compute capability | Privacy risk from data sharing |
| DQL-based (MCML) | DDQN optimizes data/energy/CPU allocation | Doesn't scale well with many devices |
| q-FFL | Reweight loss to reduce accuracy variance across participants | Slower convergence |

---

### 4.4.2 Joint Radio and Computation Management

| Method | Core Idea | Latency Improvement |
|---|---|---|
| BAA (over-the-air computation) | Exploit signal superposition — all devices transmit simultaneously | 10×–1000× over OFDMA |
| BAA + Error Accumulation | Store untransmitted gradients, correct in next round | Higher accuracy than basic BAA |
| DC Algorithm | Maximize participants while keeping aggregation error below threshold | Near-optimal, scalable |

---

### 4.4.3 Adaptive Aggregation

| Method | Approach | Problem Solved |
|---|---|---|
| Asynchronous FL | Server aggregates whenever any update arrives | Eliminates straggler waiting |
| FedAsync | Stale updates weighted less by staleness factor | Reduces outdated updates dragging convergence |
| Adaptive Aggregation (control algorithm) | Vary global aggregation frequency based on real-time system state | Energy-efficient resource use |

---

### 4.4.4 Incentive Mechanisms

| Mechanism | Model | Advantage |
|---|---|---|
| Stackelberg Game | Server as buyer, participants as sellers | Unique equilibrium; incentivizes better data |
| Contract Theory | Self-revealing contracts for data quality types | Extracts more profit; handles information asymmetry |
| DRL-based Incentive | Server learns optimal payment policy without prior knowledge | Dynamic adaptation |
| Reputation Blockchain | Reputation scores from past interactions stored on blockchain | Filters unreliable participants |

---

## 4.5 Privacy and Security Defenses

### Privacy Attacks
- **Model inversion**: Reconstruct training samples from released model
- **Membership inference**: Determine if an individual's data was in training set (up to 90% accuracy)
- **GAN-based inference**: Reconstruct private data from partial shared parameters

### Privacy Defenses

| Defense | Method | Limitation |
|---|---|---|
| DP-SGD (Abadi et al.) | Add Gaussian noise to gradients during training | Accuracy degradation |
| Round-based DP | Randomize which participants are selected each round | Does not work if server is malicious |
| Selective Parameter Sharing | Each participant shares only a fraction of parameters (1–10%) | Tested only on simple datasets |
| Homomorphic Encryption | Encrypt parameters before sending; server aggregates on ciphertext | Multi-round overhead, no collusion prevention |
| Hybrid (HE + DP) | Combine encryption and noise | Accuracy impact not quantified |
| Federated GAN | Collaboratively train a GAN to generate synthetic data | Training instability |

### Security Attacks

| Attack Type | Description | Impact |
|---|---|---|
| Data Poisoning (dirty-label) | Malicious participant injects mislabeled samples | 90% misclassification with only ~50 injected samples |
| Sybil Attack | One attacker creates many fake participants | 96.2% attack success with just 2 attackers |
| Model Poisoning | Attacker directly manipulates model updates (not data) | Corrupts global model |

### Security Defenses

| Defense | Key Idea |
|---|---|
| FoolsGold | Detect sybil participants by similarity of their gradient updates (sybil gradients look alike in non-IID FL) |
| Byzantine-robust aggregation | Replace averaging with robust aggregators (e.g., coordinate-wise median) |

---

# 5. Experimental Setup / Evaluation Design

## 5.1 Datasets Used Across Reviewed Papers

| Dataset | Type | Used For |
|---|---|---|
| MNIST | Handwritten digit images | Basic FL convergence and compression tests |
| CIFAR-10 | Color image classification | Non-IID accuracy degradation, BAA tests |
| EMNIST | Extended MNIST | Personalization and dropout tests |
| Flickr-AES | Aesthetic rating dataset | FEDPER personalization |
| FaceScrub | Face images | Privacy/membership inference |
| Human Activity Recognition | Sensor signals | CMFL communication savings |
| KDDCup | Network intrusion data | FoolsGold sybil defense |
| Amazon Reviews | Text reviews | FoolsGold sybil defense |
| Next-Word-Prediction LSTM | Text prediction | FedAvg communication studies |

## 5.2 Standard Metrics Used

| Metric | Purpose |
|---|---|
| Test accuracy | Primary model performance |
| Communication rounds to target accuracy | Efficiency of communication |
| Number of bits communicated | Compression effectiveness |
| Training latency | System-level efficiency |
| Energy consumption | Device resource viability |
| Accuracy variance across participants | Fairness measurement |
| Attack success rate | Security evaluation |

## 5.3 Common Experimental Protocol

- Compare proposed method vs. FedAvg baseline
- Evaluate on both IID and non-IID data splits
- Vary number of participants and local update counts
- Measure accuracy vs. communication rounds curves

### Experimental Reliability Analysis

| What Is Trustworthy | What Is Questionable |
|---|---|
| FedAvg baseline comparisons are consistently reproducible | Non-IID simulation often uses artificial class-based splits that may not match real distributions |
| Accuracy results for simple datasets (MNIST) are stable | Many results tested on small-scale simulations (few tens of devices) — scalability to thousands unverified |
| Communication round reductions are clearly quantified | Hardware heterogeneity is rarely simulated realistically |
| DP-SGD noise-accuracy tradeoffs are well-established mathematically | Incentive mechanisms assume rational actors and single-server monopoly |

---

# 6. Results & Findings Interpretation

## 6.1 Key Numerical Findings

| Finding | Number | Significance |
|---|---|---|
| FedAvg accuracy drop with non-IID CIFAR-10 | 51% lower than centralized | Major unsolved challenge |
| 5% shared data restores accuracy | +30% accuracy gain | Data sharing is powerful but risky |
| Quantization to 2 bits | 256× fewer bits | Extreme compression viable |
| Federated dropout at 25% | ~43% model size reduction | Practical without major accuracy cost |
| BAA latency vs. OFDMA | 10×–1000× reduction | Massive potential for over-the-air FL |
| CMFL communication saving (LSTM) | 13.97× fewer rounds | Importance-based updating very effective for sequential models |
| DQL energy savings vs. greedy | ~31% reduction | RL well-suited to dynamic network |
| FoolsGold vs. sybil (2 attackers) | From 96.2% to mitigated | Gradient similarity detection works well |
| Membership inference success rate | Up to 90% | Privacy is not guaranteed by FL alone |

## 6.2 Performance Trends

- More local computation → fewer communication rounds → but worse accuracy with non-IID data
- More participants per round → better accuracy → but slower per-round completion
- Stronger privacy (more noise) → worse model accuracy
- Asynchronous FL → faster wall-clock time → but less stable convergence

## 6.3 Unexpected Observations

- FedAvg produces regularization effect similar to dropout (prevents overfitting), which can actually improve final accuracy
- CMFL improves accuracy beyond just saving communication rounds — eliminating irrelevant updates acts as noise reduction
- BAA latency advantage is independent of number of participants (constant latency regardless of scale) — this is a fundamentally different scaling property than OFDMA

### Publishability Strength Check

| Claim | Publication Grade | Comment |
|---|---|---|
| Non-IID degradation quantification | High | Well-replicated across papers |
| BAA latency improvement | High | Novel communication paradigm with strong results |
| FoolsGold sybil defense | High | Clear adversarial model with measurable outcomes |
| Contract theory incentive mechanism | Moderate | Strong theory but limited device scale tested |
| GAN-based privacy defense | Moderate | Needs comparison with more baselines |
| Adaptive aggregation control algorithm | Moderate | Convergence only for convex loss |

---

# 7. Strengths – Weaknesses – Assumptions

## Table 1: Technical Strengths

| Strength | Details |
|---|---|
| Comprehensive coverage | Covers FL foundations, communication, resource allocation, privacy, security, and applications in one document |
| Dual perspective | Analyzes both FL deployed at edge networks AND FL used to optimize edge networks |
| Clear taxonomy | Table IV and Table V provide structured comparison of approaches |
| Open-source framework review | Covers TFF, PySyft, LEAF, FATE — actionable for new researchers |
| Challenge-solution mapping | Each challenge section ends with lessons learned and open problems |
| Multiple resource dimensions | Considers computation, communication, energy, data quality, and willingness simultaneously |

## Table 2: Explicit Weaknesses

| Weakness | Details |
|---|---|
| Lacks unified theoretical framework | Surveys individual algorithms — no overarching convergence theory across all challenges |
| Scalability of simulations | Most reviewed papers test on tens of devices, not thousands |
| Non-IID realism | Non-IID is simulated by class-based splits, not real user behavior distributions |
| Mobility is underexplored | Only one paper ([126]) considers participant mobility |
| Multiple FL server competition | All incentive mechanisms assume monopoly server — unrealistic |
| Missing system-level evaluation | Latency, energy, and bandwidth are rarely jointly optimized |
| Incomplete privacy-accuracy tradeoff quantification | Many privacy papers don't measure accuracy cost of their defense |

## Table 3: Hidden Assumptions

| Assumption | Where It Appears | Why It May Not Hold |
|---|---|---|
| Participants are honest (use real data) | FL training process | Data poisoning attacks contradict this |
| Server is honest | Most privacy solutions | Adversarial server is a well-documented threat |
| Single FL server per federation | Incentive mechanism designs | Multiple competing servers exist in real deployments |
| Stable wireless connection | Most algorithms | Participants frequently disconnect and reconnect |
| Participants can always perform local training | FedAvg-based methods | Energy-constrained devices may refuse |
| Non-IID is the only statistical challenge | Most statistical analysis | Concept drift, temporal non-stationarity also exist |
| Ground truth labels are correct | Data poisoning baseline | Dirty-label attacks violate this |

---

# 8. Weakness → Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| Non-IID accuracy degradation | Participants hold personalized data; no shared IID pool | Personalized FL with global feature sharing | Meta-learning (MAML), FEDPER-style base + personalization layers |
| No convergence guarantee for non-convex asynchronous FL | Mathematical complexity of asynchronous updates on non-convex losses | Provably convergent async FL algorithms | Stochastic analysis, Lyapunov stability theory |
| Communication-accuracy tradeoff unformalized | Empirical tuning dominates | Theoretical communication complexity bounds | Information theory, rate-distortion theory |
| Mobility not considered in most FL designs | Wireless network models assume static participants | Mobility-aware FL with handover support | Predictive participant selection, track-and-reconnect mechanisms |
| Multiple competing FL servers not modeled | Incentive literature assumes monopoly | Multi-server federated market design | Multi-leader multi-follower Stackelberg game, auction theory |
| Privacy-accuracy tradeoff not quantified | Privacy and accuracy studied separately | Unified Privacy-Utility Pareto framework | Rényi DP, moments accountant method |
| Adversarial server not fully addressed | Most privacy papers assume honest server | Trustless FL without central aggregator | Blockchain-based aggregation, SMPC-based server-free FL |
| Real-device energy not jointly optimized | Most papers optimize one resource dimension | Joint latency–energy–accuracy co-optimization | Multi-objective optimization, Pareto-aware RL |
| Scalability to thousands of devices unverified | Most papers use few tens of devices | Large-scale FL system simulation | Hierarchical FL, clustered aggregation architectures |
| LLM fine-tuning in FL not addressed | Paper predates large model era | Federated LLM fine-tuning with parameter-efficient adapters | LoRA-based FL, split learning for large models |

---

# 9. Novel Contribution Extraction

## Contribution Claim Templates (Inspired by This Paper)

Use these as starting points for your own research contribution statements:

1. "We propose **[personalized FL algorithm]** that improves **[model accuracy on non-IID data]** by **[learning shared global features + individualized local layers]**, reducing accuracy gap to centralized training by X%."

2. "We propose **[mobility-aware participant selection scheme]** that improves **[FL training stability]** by **[predicting participant availability using trajectory models]**, reducing dropout-caused convergence degradation by X%."

3. "We propose **[communication-efficient FL with formal accuracy bounds]** that improves **[communication-accuracy tradeoff]** by **[applying rate-distortion theory to gradient compression]**, reducing communication by X× while maintaining Y% accuracy guarantee."

4. "We propose **[blockchain-secured federated aggregation]** that improves **[FL privacy against a malicious aggregation server]** by **[distributing aggregation across a decentralized validator network]**, eliminating the need for any trusted third party."

5. "We propose **[multi-server federated learning market]** that improves **[participant incentive alignment in competitive environments]** by **[designing multi-leader Stackelberg mechanisms for competing FL servers]**, achieving X% higher participant contribution quality than single-server baselines."

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Directions

- Formal convergence analysis for asynchronous FL with non-IID, non-convex settings
- Multiple competing FL server (federation) ecosystem modeling
- Scalable participant selection with fairness guarantees
- Better non-IID simulation datasets reflecting real user behavior
- Asynchronous FL convergence bounds for real device constraints

## 10.2 Missing Directions (Not Mentioned by Authors)

- **Federated Transfer Learning** across heterogeneous model architectures
- **Continual / lifelong FL** — models that keep learning without forgetting
- **Federated meta-learning** — fast adaptation to new tasks at edge devices
- **Cross-silo vs. cross-device FL** — fundamentally different challenges not unified
- **Client-side watermarking** for contribution attribution and IP protection

## 10.3 Modern and Emerging Extensions

| Extension Direction | Why It Matters |
|---|---|
| FL for Large Language Models (LLMs) | Fine-tuning LLMs in private settings is a major industry need |
| LoRA-based FL | Parameter-efficient fine-tuning reduces communication cost by 100× |
| FL + 5G/6G network slicing | New network architectures create new FL deployment opportunities |
| FL for digital twins | Edge network digital twins trained via FL |
| Semantic communication + FL | Transmit meaning rather than raw gradients |
| FL for autonomous vehicles | Vehicle-to-vehicle collaborative perception with strict latency |
| FL under adversarial wireless environments | Jammer-aware FL robustness |

## 10.4 LLM-Era Extensions

- **Federated instruction tuning**: Fine-tune LLM system prompts using private user data
- **Federated RLHF**: Gather human feedback signals for LLM alignment without exposing private preferences
- **Split learning for LLMs**: Split model across devices and cloud to enable large model FL
- **Prompt-based FL**: Share only learnable prompt tokens, not full model weights

---

# 11. How to Write a NEW Paper From This Work

## Reusable Elements

| Element | How to Reuse |
|---|---|
| Problem framing (FL challenges) | Use the 4-challenge taxonomy (communication, resource allocation, privacy, security) as a structure |
| FedAvg as baseline | Always compare against FedAvg — it is the standard |
| IID vs. non-IID evaluation | Every FL paper must test both settings |
| Communication rounds as efficiency metric | Use alongside wall-clock time and bits transmitted |
| Earth Mover's Distance for non-IID | Quantify data heterogeneity formally |
| Table-based comparison of related work | Organize your related work survey in a structured table |
| Open-source frameworks (TFF, PySyft, LEAF) | Use LEAF benchmarks for fair comparison |

## What MUST NOT Be Copied

- Do not copy the survey structure verbatim — create your own taxonomy
- Do not reuse the exact Table II abbreviations — build your own for your context
- Do not copy the lesson-learned summaries — these are the authors' synthesis
- Do not reproduce paper descriptions — all related work must be paraphrased

## How to Design a Novel Extension

1. **Select one weakness** from Table 2 or Section 8 of this document
2. **Identify a missing method** that addresses that weakness
3. **Formalize a problem**: Define the optimization objective, constraints, and variables
4. **Propose an algorithm**: Modify an existing FL algorithm or design a new one
5. **Prove convergence**: Even a partial convergence bound is valuable
6. **Evaluate on standard benchmarks**: MNIST, CIFAR-10, and one domain-specific dataset (medical or vehicular)
7. **Compare against FedAvg and at least 2 other baselines**

## Minimum Publishable Contribution Checklist

- [ ] Clear, formal problem statement with novel angle
- [ ] Comparison against FedAvg baseline
- [ ] Tested on both IID and non-IID settings
- [ ] At least one theoretical result (convergence bound, complexity analysis, or information-theoretic bound)
- [ ] Multiple datasets, including one practical scenario
- [ ] Ablation study showing which component contributes most
- [ ] Discussion of limitations
- [ ] Honest failure case analysis

---

# 12. Complete Paper Writing Template

## Abstract

**Purpose**: Summarize the entire paper in 150–250 words.

**What to include**:
- The problem (2 sentences)
- Why existing solutions fail (1 sentence)
- Your key idea (1 sentence)
- Your method in brief (2 sentences)
- Main result (1–2 numbers)
- Significance statement (1 sentence)

**Common mistakes**: Starting with "In this paper..." — instead, start with the problem. Claiming too much ("first ever"). Vague results without numbers.

**Reviewer expectation**: Clear problem–gap–solution–result structure. Must be self-contained.

---

## Introduction

**Purpose**: Motivate the problem and establish your contribution.

**What to include**:
1. Context: Why is FL/MEC important? (IoT growth, privacy laws)
2. Problem: What specific challenge are you solving?
3. Limitations of prior work: Use this paper as a starting point — cite specific weaknesses
4. Your contribution: 3–4 bullet points
5. Paper organization

**Common mistakes**: Too much background (save for Section 2). Making contributions sound incremental.

**Reviewer expectation**: Clear gap identification. Contributions must be distinct from prior work.

---

## Related Work

**Purpose**: Position your work within existing research.

**What to include**:
- Group related papers by sub-topic (communication efficiency, resource allocation, etc.)
- Include a comparison table similar to Table I of this paper
- Explicitly state what each group does and does not do
- End each group with: "Our work differs in that..."

**Common mistakes**: Merely listing papers without comparison. Including irrelevant work.

---

## System Model / Problem Formulation

**Purpose**: Formally define the setting and optimization problem.

**What to include**:
- Network model (number of devices, servers, topology)
- Communication model (channel assumptions, bandwidth constraints)
- Computation model (device capabilities, local training budget)
- Formal optimization problem with objective function and constraints

**Common mistakes**: Assumptions that are too idealized. Not explaining why assumptions are reasonable.

---

## Proposed Method / Algorithm

**Purpose**: Present your contribution in full technical detail.

**What to include**:
- Algorithm pseudocode
- Explanation of each step
- Why each design choice was made
- Comparison to the most similar existing method

**Common mistakes**: Algorithm without explanation. Pseudocode that is ambiguous.

---

## Theoretical Analysis

**Purpose**: Provide mathematical guarantees for your method.

**What to include**:
- Convergence theorem (state the theorem clearly before the proof)
- Assumptions required (be explicit — non-IID? non-convex?)
- Proof sketch or outline
- Interpretation of the bound in plain language

**Common mistakes**: Proving trivial results. Hiding strong assumptions. Not connecting the bound to practical behavior.

**Reviewer expectation**: At least one theorem. Clear statement of what the theorem says.

---

## Experiments

**Purpose**: Empirically validate your method.

**What to include**:
- Dataset descriptions and justification
- Baseline methods and why they were chosen
- Hyperparameter settings (full table in appendix)
- Primary results tables and figures
- Ablation study

**Common mistakes**: Not comparing against FedAvg. Testing only IID. Missing error bars or confidence intervals.

---

## Discussion

**Purpose**: Interpret results and contextualize findings.

**What to include**:
- Why your method works (mechanistic explanation)
- Where it does not work (failure cases)
- Practical deployment considerations
- Connection to broader impact

---

## Limitations

**Purpose**: Honest assessment of what your paper does not address.

**What to include**:
- Assumptions that may not hold in practice
- Scenarios not tested
- Theoretical gaps

**Common mistakes**: Listing only minor limitations to appear more thorough. Omitting known weaknesses.

---

## Conclusion

**Purpose**: Summarize key contributions and future directions.

**What to include**:
- One sentence per main contribution
- Two or three concrete future work directions

---

## References

**Reviewer expectation**: 30–60 citations for a research paper; 100–200+ for a survey. Cite foundational work (McMahan 2017 for FedAvg, Dwork 2014 for DP, etc.) consistently.

---

# 13. Publication Strategy Guide

## Suitable Venues

| Venue Type | Examples | Focus |
|---|---|---|
| IEEE Transactions on Communications | ToC, TCOM | Communication-centric FL methods |
| IEEE Transactions on Wireless Communications | TWC | Wireless-specific FL optimization |
| IEEE/ACM Transactions on Networking | ToN | Network-level FL deployment |
| Machine Learning Conferences | ICML, NeurIPS, ICLR | Algorithm-centric FL contributions |
| IEEE Transactions on Information Forensics and Security | TIFS | Privacy and security extensions |
| IEEE Internet of Things Journal | IoTJ | IoT/Edge FL applications |
| IEEE Communications Magazine | ComMag | Survey and tutorial papers |
| IEEE Transactions on Mobile Computing | TMC | Mobile edge computing deployment |

## Required Baseline Expectations

- FedAvg is mandatory
- At minimum one communication-efficient baseline (e.g., CMFL, eSGD) for communication papers
- For privacy papers: DP-SGD baseline
- For resource allocation: FedCS or a DRL baseline

## Experimental Rigor Level Required

- Minimum 3 datasets (including one real-world scenario beyond MNIST/CIFAR)
- Both IID and non-IID evaluation
- Statistical significance: run experiments at least 3–5 times, report mean ± std
- Convergence curves (accuracy vs. communication rounds) are expected for communication/aggregation papers

## Common Rejection Reasons

- "Insufficient novelty" — contribution is too similar to FedAvg extensions
- "Only tested on MNIST" — reviewers want realistic and diverse datasets
- "No theoretical analysis" — especially for algorithm papers
- "Unfair comparison" — baseline implementations not given the same parameter tuning
- "Non-IID not addressed" — any FL paper ignoring non-IID will be rejected

## Increment Needed for Acceptance

| Target Venue | Required Improvement |
|---|---|
| Top-4 ML Conferences | >5% accuracy gain with theoretical guarantee; novel problem formulation |
| IEEE Transactions | 3–5% improvement with convergence proof; system model contribution |
| IEEE Letters/Journals | Clear incremental contribution; solid empirical evaluation on multiple datasets |
| Application Journals (IoT, TMC) | Application-specific novelty; deployment feasibility analysis |

---

# 14. Researcher Quick Reference Tables

## Key Terminology Table

| Term | Simple Definition |
|---|---|
| Federated Learning (FL) | Collaborative ML training where data stays on devices; only model updates are shared |
| FedAvg | Baseline FL algorithm: locally trained weights are averaged at the server |
| Non-IID | Different participants hold different statistical distributions of data |
| IID | All participants hold data from the same distribution (ideal assumption) |
| Straggler | Slow participant device that delays the entire FL training round |
| MEC (Mobile Edge Computing) | Placing computing resources at network edges (near users), not in distant cloud |
| Differential Privacy (DP) | Add calibrated noise to model updates to protect individual data from being inferred |
| Homomorphic Encryption (HE) | Encrypt data such that computation can be done on the ciphertext directly |
| Over-the-air Computation | Exploit wireless channel superposition to perform aggregation during transmission |
| BAA (Broadband Analog Aggregation) | Multi-access scheme using signal superposition for FL aggregation |
| FedCS | Participant selection protocol choosing max devices completable within a deadline |
| Sybil Attack | One attacker creates many fake identities to amplify a data poisoning attack |
| FoolsGold | Defense using gradient similarity to detect sybil participants |
| Contract Theory | Mechanism design tool with self-revealing contracts to address information asymmetry |
| Stackelberg Game | Leader-follower game model for pricing between FL server and participants |
| EMD (Earth Mover's Distance) | Measures how different two data distributions are — used to quantify non-IID degree |
| SMPC (Secure Multiparty Computation) | Protocol allowing multiple parties to jointly compute a function without revealing inputs |

---

## Important Equations Summary

| Equation | What It Does |
|---|---|
| $w \leftarrow w - \eta \nabla_w L(w;b)$ | Core SGD update rule for local training |
| $w_G^t = \frac{\sum_i \|D_i\| w_i^t}{\sum_j \|D_j\|}$ | FedAvg weighted aggregation |
| $L(w_G) = \sum_i \frac{\|D_i\|}{\sum_j \|D_j\|} L_i(w_i)$ | Global loss as weighted sum of local losses |
| $\min_w \sum_k p_k F_k^{q+1}(w)$ | q-Fair FL objective (fairness-aware aggregation) |
| Membership inference accuracy: up to 90% | Privacy risk benchmark with no defense |

---

## Parameter Meaning Table

| Parameter | Meaning | Typical Values |
|---|---|---|
| $N$ | Total number of FL participants | 10–1000+ |
| $m$ | Participants selected per round | Fraction of $N$ |
| $E$ | Local training epochs per round | 1–50 |
| $B$ | Mini-batch size for local SGD | 10–256 |
| $\eta$ | Learning rate | 0.001–0.1 |
| $q$ | Fairness calibration in q-FFL | 0 (standard) to high (max fairness) |
| $\lambda$ | Regularization constant in SVM loss | Problem-dependent |

---

## Algorithm Flow Summary

| Algorithm | Steps | Key Feature |
|---|---|---|
| FedAvg | Init → Select participants → Local SGD (E epochs) → Send weights → Weighted average → Repeat | Baseline FL algorithm |
| FedCS | Resource request → Greedy participant selection by deadline → FedAvg training | Deadline-constrained selection |
| Hybrid-FL | Resource request + data upload → IID-approximating selection → Merged training | IID construction via data sharing |
| HierFAVG | Local training → Edge aggregation (multiple rounds) → Global aggregation → Repeat | Hierarchical MEC aggregation |
| FedAsync | Any participant sends update → Server immediately updates global model → Staleness weighting | Asynchronous, no straggler waiting |
| q-FFL | Assign higher loss weights to worst-performing participants → FedAvg with modified objective | Fairness-aware FL |
| FoolsGold | Compare gradient cosine similarities → Down-weight similar (sybil) updates | Sybil-resistant aggregation |
| Adaptive Aggregation | Monitor system resource state → Derive optimal global aggregation frequency → Adjust dynamically | Energy-efficient FL |

---

# 15. One-Page Master Summary Card

## Paper at a Glance

| Dimension | Content |
|---|---|
| **Problem** | FL on mobile edge networks faces three challenges simultaneously: high communication cost (large model updates over slow wireless), heterogeneous device resources (compute, energy, data quality), and privacy/security threats (model inversion, poisoning attacks) |
| **Idea** | Conduct a comprehensive survey that: (1) maps FL implementation challenges at the edge, (2) maps FL as an optimization tool for edge networks, and (3) identifies open problems for future research |
| **Key Methods Surveyed** | FedAvg (baseline), model compression, importance-based updating, FedCS, BAA over-the-air computation, asynchronous FL, contract theory incentives, DP-SGD, homomorphic encryption, FoolsGold |
| **Core Results** | Non-IID degrades FedAvg accuracy by 51%; BAA achieves 10–1000× latency reduction over OFDMA; FoolsGold mitigates 96.2% sybil attacks; membership inference succeeds at 90% without privacy mechanisms |
| **Key Weakness** | No unified theoretical framework; most results from small-scale simulations; multiple competing server scenarios not modeled; privacy-accuracy tradeoff rarely quantified |
| **Research Opportunities** | Mobility-aware FL, provably convergent async FL for non-convex non-IID, multi-FL-server market design, FL for LLM fine-tuning, formal communication-accuracy bounds |
| **Best Publishable Extension** | Design a mobility-aware asynchronous FL algorithm with convergence guarantees on non-convex losses for non-IID data, validated on vehicular or healthcare datasets with energy-latency-accuracy joint optimization |

---

## Summary of the Four Core Challenges

```
FL at Mobile Edge Networks
├── 1. Communication Cost
│   ├── Problem: Millions of parameters × slow upload speeds
│   ├── Solutions: More local updates (FedAvg), compression, selective uploading
│   └── Trade-off: Accuracy suffers; non-IID makes compression worse
│
├── 2. Resource Allocation  
│   ├── Problem: Heterogeneous devices; free riders; straggler effect
│   ├── Solutions: FedCS, BAA, asynchronous FL, contract theory incentives
│   └── Trade-off: Fairness vs. efficiency; incentive vs. privacy
│
├── 3. Privacy
│   ├── Problem: Gradients leak individual data (up to 90% membership inference)
│   ├── Solutions: DP-SGD, homomorphic encryption, selective sharing, federated GAN
│   └── Trade-off: Privacy noise degrades model accuracy
│
└── 4. Security
    ├── Problem: Poisoning attacks through malicious participants (96.2% attack success)
    ├── Solutions: FoolsGold (gradient similarity), Byzantine-robust aggregators
    └── Trade-off: Defense mechanisms add computation overhead
```

---

*Document generated from: Lim et al. (2020), "Federated Learning in Mobile Edge Networks: A Comprehensive Survey," IEEE Communications Surveys & Tutorials.*
*PDF extracted using Docling with OCR enabled. Pages 22–34 had partial OCR failures (memory constraints) — content from Sections VI–VII may be partially represented through Introduction references.*
