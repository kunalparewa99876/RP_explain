# Research Companion: Advances and Open Problems in Federated Learning
**Kairouz et al., 2021 — The Federated Learning "Grand Survey" (58 Authors)**

---

# 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Paper Title** | Advances and Open Problems in Federated Learning |
| **Authors** | Peter Kairouz, H. Brendan McMahan, + 56 co-authors (Google, Apple, CMU, Stanford, MIT, etc.) |
| **Published At** | Foundations and Trends® in Machine Learning, Vol. 14, No. 1–2, 2021 |
| **arXiv** | 1912.04977 |
| **Problem Domain** | Federated Learning — comprehensive mapping of challenges, algorithms, theory, privacy, robustness, systems, and applications |
| **Paper Type** | Survey / Conceptual + Theoretical (with formal problem statements and open questions) |
| **Core Contribution** | The most comprehensive reference document in FL: maps every known open problem, formalizes challenges, and proposes research directions across the entire FL landscape |
| **Key Idea** | FL is not a single algorithm but a complete research ecosystem with unique challenges in privacy, communication, systems heterogeneity, statistical heterogeneity, and fairness — each requiring dedicated research attention |
| **Required Background** | FedAvg (McMahan 2017), differential privacy basics, optimization theory (SGD convergence), distributed systems fundamentals, basic cryptography concepts |
| **Primary Baseline** | FedAvg (McMahan et al., 2017) — the FedSGD/FedAvg framework is the starting reference throughout |
| **Main Innovation Type** | Conceptual Framing + Problem Formalization + Research Roadmap |
| **Difficulty Level** | High — spans deep theory, systems, cryptography, optimization, and fairness simultaneously |
| **Reproducibility Level** | Low (survey/position paper — no single reproducible experiment; references individual papers for empirics) |

---

# 1. Research Context & Core Problem

## 1.1 Exact Problem Formulation

The foundational paper on FL (McMahan 2017) introduced FedAvg and proved it works in practice. But it left hundreds of questions unanswered. This paper systematically asks:

- Under what conditions does FL break?
- What assumptions are hidden inside current FL methods?
- What is the correct privacy model for FL?
- What happens when clients misbehave, fail, or act adversarially?
- How does FL scale to billions of devices?
- Can FL be made fair across heterogeneous populations?
- What formal theoretical guarantees can FL provide?

The paper collects and formalizes these questions across **six major problem areas**, with each area containing multiple open research problems, making it the definitive map of the FL research landscape.

## 1.2 Why the Problem Exists

FL was presented as a promising framework (see: McMahan 2017, Section 1 of this companion), but its real-world deployment at Google (Gboard, Android) exposed a large gap between the simple algorithm and production requirements:

- Devices drop out mid-round (partial participation).
- Data distributions across clients are wildly different (non-IID).
- Malicious clients can poison the global model.
- Model updates themselves can leak private information.
- Communication costs at 10M+ clients become a serious bottleneck.
- Some groups of users are systematically under-represented.

No single paper addressed all of these simultaneously. This paper is the response to that gap.

## 1.3 Historical and Theoretical Gap

| Era | State of Knowledge |
|---|---|
| Pre-2016 | Distributed optimization in data centers: IID data, fast networks, no privacy |
| 2017 (FedAvg) | First cross-device FL algorithm — empirically effective but theoretically under-justified |
| 2018–2019 | Scattered papers on FL + DP, FL + secure aggregation, non-IID convergence |
| 2021 (This Paper) | First systematic taxonomy of ALL open problems across the full FL stack |

The gap: Researchers were solving isolated sub-problems without a shared vocabulary or problem taxonomy. This paper creates that shared language.

## 1.4 Limitations of Previous Approaches

- **FedAvg alone**: No privacy guarantee, no convergence proof for non-convex non-IID, no robustness, no fairness mechanism.
- **Differential Privacy in FL**: Added noise degrades accuracy; the right noise level for FL across millions of clients was unknown.
- **Secure Aggregation**: Cryptographically sound but computationally expensive; scaling to cross-device settings unsolved.
- **Personalization**: No standard benchmark or evaluation framework existed.
- **Byzantine Robustness**: Designed for data-center settings; effectiveness in cross-device FL was unknown.

## 1.5 Contribution Categories

| Category | Present? | Description |
|---|---|---|
| Theoretical | Yes | Formal definitions of privacy, convergence bounds, fairness criteria |
| Algorithmic | Yes | Survey and comparison of FL algorithms |
| Optimization | Yes | Open problems in convergence under heterogeneity |
| System Design | Yes | Cross-device vs cross-silo; infrastructure gaps |
| Empirical Insight | Partial | References to empirical results in cited papers |
| Problem Taxonomy | Yes (primary) | Classification of all known FL challenges |

---

### Why This Paper Matters

This is the **canonical reference document for the entire FL research community**. Any researcher working on:
- Privacy in ML
- Distributed optimization
- Communication-efficient training
- Robustness against adversarial clients
- Fairness in ML
- Mobile/edge AI systems

...must read this paper. It defines the vocabulary, sets the research agenda, and explicitly lists which problems remain unsolved — each of which is a publishable research target.

---

### Remaining Open Problems (This Paper's Explicit Contributions)

The paper lists over 70 open problems across 6 domains. The highest-priority ones:

1. Convergence guarantees for FedAvg under non-IID data with non-convex objectives
2. Tight privacy-utility tradeoffs: how much accuracy must be sacrificed for DP?
3. Secure aggregation that scales efficiently to millions of clients
4. Personalization: how to adapt a global model to individual users without losing generalization
5. Byzantine-robust aggregation that works under realistic adversary models
6. Fairness: defining and enforcing equitable treatment across heterogeneous client populations
7. Asynchronous FL: handling delayed updates from slow devices
8. Vertical FL: different clients hold different features (not different samples)
9. Communication efficiency: structured compression without convergence degradation
10. Formal threat models for FL privacy (what exactly is being protected, from whom?)

---

# 2. Minimum Background Concepts

These are the only concepts needed to understand this paper. Every concept is explained as it relates to the paper.

## 2.1 Federated Learning Setup (Cross-Device vs Cross-Silo)

- **Plain definition**: Cross-device FL involves millions of mobile devices (each with small data). Cross-silo FL involves tens to hundreds of institutions (hospitals, banks), each with large data.
- **Role in paper**: The paper distinguishes these two regimes because challenges are fundamentally different — cross-device has massive scale + sparse participation + low compute per client; cross-silo has institutional trust issues, regulatory barriers, and small numbers of powerful participants.
- **Why authors needed it**: Many proposed algorithms only work in one regime. Conflating them leads to invalid solutions.

## 2.2 Non-IID Data (Heterogeneous Data)

- **Plain definition**: Data across clients does not follow the same probability distribution. Each client's data reflects their personal behavior, location, language, or domain.
- **Role in paper**: Identified as the single most impactful challenge — it causes FedAvg to diverge or perform poorly, and it invalidates standard convergence proofs.
- **Why authors needed it**: Almost every subsection of the paper discusses non-IID as a complicating factor.

## 2.3 Differential Privacy (DP)

- **Plain definition**: A mathematical guarantee that any individual client's data has negligible influence on the released model. Formally: adding or removing one person's data changes the output distribution by at most a factor $e^\varepsilon$.
- **Role in paper**: Section 4 extensively covers DP in FL — both local DP (noise added at client before upload) and central DP (noise added at server after aggregation).
- **Why authors needed it**: Without DP, model updates can be reverse-engineered to reconstruct private training data (gradient inversion attacks). DP is the primary privacy tool in FL.

## 2.4 Secure Aggregation

- **Plain definition**: A cryptographic protocol where the server learns only the SUM of all client updates, not individual updates. Uses techniques from secret sharing and multi-party computation.
- **Role in paper**: Presented as the complement to DP — DP prevents inference from the aggregate; secure aggregation prevents the server from seeing individual updates.
- **Why authors needed it**: Even if the aggregate is published with DP noise, if the server sees individual gradients, privacy is violated against a semi-honest server.

## 2.5 Byzantine Robustness

- **Plain definition**: The ability of a system to produce correct results even when some participants provide wrong, corrupted, or malicious inputs.
- **Role in paper**: Covered in Section 4 (robustness) — if even a few clients send poisoned updates, FedAvg's averaging will incorporate the poisoned signal.
- **Why authors needed it**: At the scale of millions of devices, some fraction will always be compromised (malware, adversarial users, hardware failures).

## 2.6 Personalization in FL

- **Plain definition**: Instead of one global model for all clients, personalization allows the model (or part of it) to be tailored to each individual client's data distribution.
- **Role in paper**: Section 3 — one of the most active research directions. Pure global models underperform for users whose data is very different from the global average.
- **Why authors needed it**: Real products need personalized predictions (e.g., next-word prediction should match YOUR typing style, not the average user's).

## 2.7 Communication Compression

- **Plain definition**: Techniques to reduce the size of model updates sent between clients and server — e.g., gradient quantization (fewer bits per value), sparsification (only send the largest gradient components), structured updates.
- **Role in paper**: Section 6 — communication is the primary bottleneck in FL. Reducing bits transmitted per round directly enables more rounds per unit time.
- **Why authors needed it**: A 100MB model update from 10M devices per round is 1 petabyte of data — physically impossible without compression.

## 2.8 Fairness in Machine Learning

- **Plain definition**: The requirement that a model does not systematically under-serve or discriminate against subgroups of users (based on race, gender, geography, data volume, etc.).
- **Role in paper**: Section 5 — FL makes fairness harder because some clients have much more data and thus disproportionate influence on the global model.
- **Why authors needed it**: A global model optimized for average performance may be terrible for minority-language users or under-represented demographics.

## 2.9 Convergence in Optimization

- **Plain definition**: A guarantee that an optimization algorithm eventually reaches (or approximates) the best solution. For non-convex objectives, "convergence" typically means reaching a stationary point (gradient norm approaching zero).
- **Role in paper**: Section 3 on optimization — proving FedAvg converges under non-IID, non-convex conditions was an open problem this paper explicitly identifies.
- **Why authors needed it**: Without convergence guarantees, FedAvg is an engineering heuristic with no theoretical foundation.

## 2.10 Split Learning / Vertical FL

- **Plain definition**: Vertical FL = different clients hold different features about the same entities (e.g., a bank has transaction data; a hospital has health data; both share user IDs). Split learning = a neural network is split such that different layers run on different parties.
- **Role in paper**: Section 6 and 7 — distinguishes horizontal FL (same features, different samples) from vertical FL (different features, same samples).
- **Why authors needed it**: Many real-world partnerships involve different data silos with complementary features, not just replicated datasets.

---

# 3. Mathematical / Theoretical Understanding Layer

## 3.1 The Core FL Optimization Problem

**Intuition first**: The server wants the model that minimizes average error across ALL clients combined — but it can never directly see any client's data.

$$\min_{w \in \mathbb{R}^d} \left[ f(w) := \sum_{k=1}^{N} p_k F_k(w) \right]$$

### Variable Meaning Table

| Symbol | Meaning |
|---|---|
| $w$ | Global model parameters |
| $d$ | Model dimensionality (number of parameters) |
| $N$ | Total number of clients participating in FL |
| $p_k$ | Weight of client $k$ — typically $p_k = n_k / \sum_j n_j$ where $n_k$ = local data size |
| $F_k(w)$ | Local objective function for client $k$ — average loss on its local data |
| $f(w)$ | Global objective: weighted average of all client objectives |

**Practical interpretation**: Because each $F_k$ is defined over a different data distribution, optimizing $f(w)$ by running separate local optimizations and averaging is only exactly correct when all $F_k$ share the same minimum. Non-IID data breaks this.

**Limitation**: The paper notes that even defining the "right" objective is non-trivial — should $p_k$ be proportional to data size, or equal across clients (for fairness)?

---

## 3.2 The Non-IID Divergence Problem (Gradient Diversity)

**Intuition**: When each client's data looks different, their local gradients point in different directions. Averaging after many local steps moves each client's model toward its local optimum, which may be far from the global optimum. The final average is pulled in conflicting directions.

**Formal measure — Gradient Divergence**:

$$\Gamma = f^* - \sum_{k=1}^{N} p_k F_k^*$$

| Symbol | Meaning |
|---|---|
| $f^*$ | Global minimum of the overall objective $f(w)$ |
| $F_k^*$ | Local minimum of client $k$'s objective $F_k(w)$ |
| $\Gamma$ | Degree of heterogeneity — how far apart the local optima are from the global optimum |

**Intuition of $\Gamma$**: If all clients had the same data distribution, $\Gamma = 0$ (all local minima equal the global minimum). As data heterogeneity increases, $\Gamma$ grows, and FedAvg's convergence degrades proportionally.

**What the paper says**: Most FL convergence bounds include a term involving $\Gamma$. The larger the non-IID-ness, the larger the error floor — FedAvg will not converge to the true optimum, only to a neighborhood around it.

### Mathematical Insight Box
> **Key researcher takeaway**: The $\Gamma$ term is the "non-IID penalty." Any algorithm that claims to handle non-IID FL better than FedAvg must reduce its dependence on $\Gamma$ — either theoretically (smaller coefficient in the bound) or practically (consistently better accuracy on highly non-IID benchmarks).

---

## 3.3 Differential Privacy in FL — Formal Definition

**Intuition**: DP asks: "If I change one person's data, does the output change noticeably?" If the answer is "barely," the person has plausible deniability — their data could not have meaningfully influenced the result.

**Formal definition ($\varepsilon, \delta$-DP)**:

$$\Pr[\mathcal{M}(D) \in S] \leq e^\varepsilon \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$

### Variable Meaning Table

| Symbol | Meaning |
|---|---|
| $\mathcal{M}$ | The randomized mechanism (e.g., the trained model release procedure) |
| $D, D'$ | Two datasets differing in exactly one individual's data |
| $S$ | Any possible output set |
| $\varepsilon$ | Privacy budget — smaller = stronger privacy (typical useful range: 1–10) |
| $\delta$ | Failure probability — allows rare catastrophic leakage (typically $\delta \ll 1/n$) |

**Practical interpretation**: DP is achieved in FL by adding calibrated Gaussian or Laplace noise to gradients before uploading. The noise magnitude scales with the model's sensitivity (how much one client can change the gradient).

**Limitation of formulation**: DP protects against adversaries who see the published model. It does NOT protect against the server seeing individual raw updates (that requires secure aggregation). The paper emphasizes that DP + secure aggregation must be used together.

---

## 3.4 The Privacy-Utility Tradeoff

**Intuition**: Adding noise for privacy reduces model accuracy. The paper formalizes this as a fundamental tradeoff:

For a model trained on $n$ clients each contributing $m$ samples, with $(\varepsilon, \delta)$-DP, the expected accuracy loss scales roughly as:

$$\text{Accuracy Loss} \propto \frac{\sqrt{d}}{n \cdot m \cdot \varepsilon}$$

| Symbol | Meaning |
|---|---|
| $d$ | Model dimensionality (more parameters = more noise needed) |
| $n$ | Number of clients (more clients = less noise per client needed) |
| $m$ | Samples per client (more data = stronger signal) |
| $\varepsilon$ | Privacy budget (smaller = more noise = more accuracy loss) |

**Key insight**: This is why FL at scale (millions of clients) can achieve meaningful DP. The noise per client becomes small when $n$ is massive, making the privacy-utility tradeoff favorable at Google-scale but unfavorable at small-scale FL deployments.

### Mathematical Insight Box
> **Key researcher takeaway**: DP accuracy loss scales as $\sqrt{d}/(n \varepsilon)$. This means: (1) larger models need more data for the same privacy guarantee, (2) more clients makes DP "cheaper," and (3) there is a hard floor on how tight $\varepsilon$ can be while maintaining useful accuracy.

---

## 3.5 Communication Cost Model

**Intuition**: Every round, every participating client uploads a model update. If the model has $d$ parameters in 32-bit float, that is $4d$ bytes per client per round.

**Total communication per round**:

$$C_{\text{round}} = C \cdot N \cdot 4d \text{ bytes}$$

| Symbol | Meaning |
|---|---|
| $C$ | Fraction of clients participating per round (typically 0.01%–1%) |
| $N$ | Total number of clients |
| $4d$ | Bytes per full-precision model update |

**With $k$-bit quantization**:

$$C_{\text{round, quantized}} = C \cdot N \cdot \frac{k \cdot d}{8} \text{ bytes}$$

**The paper's key point**: Even at $C = 0.01\%$ of $N = 10^6$ clients with $d = 10^7$ parameters, one round requires transmitting 5GB. Compression from 32-bit to 2-bit reduces this to 312MB — still large but tractable. This motivates the entire Section 6 of the paper on communication efficiency.

### Mathematical Insight Box
> **Key researcher takeaway**: Communication cost is $O(C \cdot N \cdot d)$. Reducing $d$ (small models), $C$ (fewer clients per round), or bits-per-parameter by compression are the only three knobs. Research that tightens ALL three simultaneously is the holy grail of communication-efficient FL.

---

## 3.6 Byzantine Fault Tolerance — Formal Threat Model

**Intuition**: If a fraction $q$ of clients are adversarial and can send arbitrary updates, standard averaging will be corrupted. The question is: how large can $q$ be while still allowing correct learning?

**The impossibility result the paper cites**: For standard FedAvg with mean aggregation, even ONE Byzantine client can shift the aggregated gradient arbitrarily. Mean is not Byzantine-robust.

**Robust aggregation condition**: An aggregator $\text{Agg}(\cdot)$ is $(q, c)$-Byzantine-robust if, given $N$ client updates where at most $q \cdot N$ are adversarial:

$$\| \text{Agg}(g_1, ..., g_N) - \nabla f(w) \| \leq c \cdot \sigma$$

| Symbol | Meaning |
|---|---|
| $q$ | Fraction of adversarial clients |
| $c$ | Robustness constant (smaller = better) |
| $\sigma$ | Variance of honest gradients |
| $\nabla f(w)$ | True gradient of the global objective |

**Practical implication**: Methods like coordinate-wise median, Krum, and trimmed mean achieve Byzantine robustness but sacrifice accuracy under non-IID data or require strong assumptions (bounded gradient variance).

### Mathematical Insight Box
> **Key researcher takeaway**: Mean aggregation has ZERO Byzantine robustness. Any FL system deployed in a setting with untrusted clients needs a robust aggregator — but robustness and non-IID performance are in tension. Designing aggregators that handle both simultaneously is a major open problem.

---

# 4. Proposed Method / Framework (The Paper's Structure as a Framework)

This paper is a survey — its "method" is a structured taxonomy and problem formalization framework for FL research. The framework has six pillars:

---

## Pillar 1: Relaxing Core FL Assumptions

**What the authors did**: Identified the six assumptions hidden inside FedAvg and analyzed what breaks when each is relaxed.

| Hidden Assumption | What Happens When Relaxed |
|---|---|
| Clients are stationary | Distribution shift over time — model becomes stale |
| Everyone participates every round | Partial participation — biased aggregation |
| Data is private (not attackable) | Gradient inversion attacks expose raw data |
| Server is honest | Honest-but-curious server can infer sensitive attributes |
| Clients are not adversarial | Byzantine clients corrupt the global model |
| Communication is synchronous | Asynchronous FL — stale gradients cause instability |

**Why authors did this**: Without explicitly listing assumptions, any algorithm that "works" in simulation may fail in deployment.

**Weakness of this step**: The paper lists assumptions but does not always provide solutions — many relaxations remain open problems.

**Research idea seed**: Design a single FL framework parametrized by the assumption set — researchers can plug in their relaxation level and get the matching algorithmic recommendation.

---

## Pillar 2: Privacy — Two-Layer Defense

**What the authors did**: Formalized the complete privacy threat model for FL and proposed a two-layer defense combining DP + secure aggregation.

### Step-by-step Privacy Pipeline:
```
[Client k]
  1. Compute local gradient g_k on private data
  2. Clip gradient: g_k ← g_k / max(1, ||g_k|| / C)   [sensitivity control]
  3. Add Gaussian noise: g_k ← g_k + N(0, σ²C²I)       [local DP]
  4. Encrypt using secure aggregation protocol            [hide individual update]
  
[Server]
  5. Receive ONLY the sum: Σ g_k (via secure aggregation)
  6. Divide by N to get the average
  7. Update global model: w ← w - η · average
  8. Optionally add more noise for central DP guarantee
```

- **Why authors designed this**: No single mechanism (either DP OR secure aggregation) provides complete privacy. DP protects the OUTPUT; secure aggregation protects individual INPUTS to the aggregation from the server.
- **Weakness**: Combining DP with secure aggregation is computationally expensive. The Bonawitz 2017 secure aggregation protocol has $O(n^2)$ communication overhead —  infeasible for millions of clients.
- **Research idea seed**: Design lightweight secure aggregation (e.g., using structured randomness or trusted execution environments) that achieves the same privacy guarantee at $O(n \log n)$ cost.

---

## Pillar 3: Robustness Against Malicious Clients

**What the authors did**: Surveyed Byzantine-robust aggregation algorithms and their limitations in the FL setting.

### Robust Aggregation Algorithms Covered:

| Algorithm | Mechanism | Limitation in FL |
|---|---|---|
| Coordinate-Wise Median | Replace mean with median per dimension | Slaw convergence under high variance |
| Krum | Select the client update closest to all others | Ineffective under highly non-IID data |
| Trimmed Mean | Discard top/bottom $q$ fraction of values | Assumes bounded fraction of adversaries |
| FLTrust | Server uses a small clean dataset to score client updates | Requires server-side clean data (not always available) |

- **Why authors did this**: A deployed FL system WILL have compromised clients. Without robustness, one malicious client can collapse the model.
- **Weakness**: All robust aggregators assume the fraction of adversaries is known. In practice, the adversary fraction is unknown and may vary.
- **Research idea seed**: Develop an adaptive robust aggregator that estimates the adversary fraction online and adjusts its clipping threshold accordingly.

---

## Pillar 4: Optimization Under Heterogeneity

**What the authors did**: Characterized the convergence failure mode of FedAvg under non-IID data and surveyed corrective algorithms.

### Key Algorithms Surveyed:

| Algorithm | Key Idea | Advantage Over FedAvg |
|---|---|---|
| FedProx (Li et al. 2018) | Add proximal term to keep local models close to global model | Bounded divergence from global optimum |
| SCAFFOLD (Karimireddy et al. 2020) | Use control variates to correct client drift | Provably corrects non-IID bias |
| FedNova (Wang et al. 2020) | Normalize local updates before aggregation | Removes objective inconsistency |
| MIME (Karimireddy et al. 2021) | Mimics centralized optimizer via variance reduction in FL | Matches centralized SGD convergence |

**Simplified pseudocode for the core fix (control variate idea in SCAFFOLD)**:
```
Server maintains global control variate c

Each round:
  [Client k]
    receives global model w, global control variate c, local control variate c_k
    updates: delta_w = local_SGD(w, correction = c - c_k)
    updates c_k based on local gradient
    sends delta_w and updated c_k to server
  
  [Server]
    averages delta_w → updates w
    averages c_k updates → updates c
```

- **Why**: Without correction, each client drifts toward its local minimum. The control variate "steers" each client back toward the global gradient direction.
- **Weakness**: SCAFFOLD requires TWICE the communication per round (both model update AND control variate update).
- **Research idea seed**: Design a communication-compressed version of SCAFFOLD that sends quantized control variates without losing convergence speed.

---

## Pillar 5: Personalization

**What the authors did**: Formalized four distinct approaches to personalization in FL.

| Approach | How It Works | Best For |
|---|---|---|
| Global model + fine-tuning | Train global, then fine-tune locally | When local data is sufficient for fine-tuning |
| Multi-task learning | Treat each client as a different but related task | When clients share structural similarity |
| Meta-learning (Per-FedAvg) | Learn a good initialization for fast local adaptation | Few-shot adaptation per client |
| Mixture models | Each client gets a weighted mix of global + local model | When clients cluster into groups |

- **Why authors surveyed this**: Personalization fundamentally changes the objective — the goal is no longer one shared model but $N$ different models that benefit from collaboration.
- **Weakness**: No unified theoretical framework covers all four approaches. Each needs separate analysis.
- **Research idea seed**: Develop a single FL framework (e.g., Bayesian hierarchical model) that unifies all four personalization approaches as special cases.

---

## Pillar 6: Communication Efficiency

**What the authors did**: Surveyed and categorized compression techniques for FL into three families.

```
Compression Strategy Decision Tree:
  ├── Structured Updates
  │     └── Constrain update to low-rank matrix or random mask
  │           → smaller dimensionality → less to send
  │
  ├── Sketched Updates
  │     └── Random projection → send compressed sketch
  │           → server reconstructs approximate update
  │
  └── Quantization
        ├── Deterministic: round each value to k-bit representation
        └── Stochastic: round up/down with probability proportional to residual
              → unbiased compression → convergence preserved
```

- **Why**: Communication is 10–1000x more expensive than computation in cross-device FL. Compression directly translates to more rounds per day.
- **Weakness**: Most compression methods assume independent compression across rounds. In non-IID settings, error accumulation across rounds can dominate.
- **Research idea seed**: Design error-feedback compression (EF21 / memory-based) adapted for the FL setting to handle non-IID error accumulation.

---

# 5. Experimental Setup / Evaluation Design

## 5.1 Nature of This Paper's Empirical Evidence

This is a survey — it does not run its own experiments. Instead, it:
- Reviews and summarizes experimental results from cited papers.
- Identifies gaps in experimental methodology across the field.
- Proposes benchmarking standards that the field lacks.

## 5.2 Key Benchmarks Referenced in the Paper

| Benchmark | Description | What It Tests |
|---|---|---|
| MNIST / FEMNIST | Handwritten digits partitioned by writer | Basic non-IID image classification |
| Sentiment140 / Reddit | Text from individual users | NLP, non-IID language |
| CIFAR-10 / CIFAR-100 | Image classification with Dirichlet non-IID split | Heterogeneity stress test |
| Shakespeare | Lines assigned to characters | Highly non-IID language modeling |
| LEAF (Caldas et al.) | Federated version of standard benchmarks | First standardized FL evaluation suite |

## 5.3 Metrics Used and Why

| Metric | Why Used in FL |
|---|---|
| Rounds to target accuracy | Communication, not computation, is the bottleneck |
| Final test accuracy | Absolute quality of global model |
| Per-client accuracy | Reveals fairness/personalization gaps |
| Privacy budget $\varepsilon$ at target accuracy | Privacy-utility frontier |
| Compression ratio vs. accuracy | Communication efficiency tradeoff |

## 5.4 Identified Problems in Field-Wide Experimental Practice

The paper explicitly criticizes the evaluation culture in FL research:
- Different papers use different non-IID partitioning strategies — not comparable.
- Hyperparameters (learning rate, local epochs $E$) are tuned differently — unfair comparison.
- Most papers test on small numbers of clients (100–1000) but claim applicability to millions.
- Privacy evaluations (DP bounds) are rarely comparable across papers due to different accounting methods.

---

### Experimental Reliability Analysis

| What Is Trustworthy | What Is Questionable |
|---|---|
| Qualitative comparison of algorithm classes (e.g., SCAFFOLD > FedAvg under non-IID) | Specific accuracy numbers — depend heavily on tuning |
| Privacy analysis (DP accounting follows formal proofs) | Scale claims — most experiments use 100–500 clients, not millions |
| Communication compression factors (ratio analysis) | Generalizability of robustness results (attack scenarios are synthetic) |
| Identification of failure modes (well-documented empirically) | Personalization comparisons (no standard benchmark existed at time of writing) |

---

# 6. Results & Findings Interpretation

## 6.1 Major Findings of the Paper

### Finding 1: FedAvg is Insufficient for Production FL
FedAvg, despite its practical success, lacks:
- Convergence guarantees under heterogeneous data
- Privacy against gradient inversion
- Robustness against even one malicious client
- Fairness for under-represented groups

**Interpretation**: FedAvg is a useful prototype but should NOT be treated as a production solution without additional components.

### Finding 2: Privacy and Utility Are Fundamentally in Tension
The paper demonstrates that:
- Achieving $\varepsilon < 1$ (strong DP) with models larger than 1M parameters requires either an impractical number of clients OR significant accuracy degradation.
- Local DP (noise at client) is much worse than central DP (noise at server) for the same $\varepsilon$ — because local DP amplifies noise by $\sqrt{n}$ at each client independently.

**Interpretation**: Strong privacy guarantees in small FL deployments (e.g., cross-silo with 10 hospitals) require new mechanisms, not just application of existing DP tools.

### Finding 3: Non-IID Degree Determines Algorithm Choice
- Low heterogeneity ($\Gamma$ small): FedAvg works fine.
- Medium heterogeneity: FedProx or FedNova improve results.
- High heterogeneity: Only SCAFFOLD-type correction or personalized methods maintain good performance.

**Interpretation**: Measuring and reporting the heterogeneity level of a dataset should be standard practice in FL papers.

### Finding 4: Communication Compression Is Largely Solved for IID
- For IID data, 4-bit or even 1-bit quantization with error feedback loses less than 1% accuracy versus full precision.
- For non-IID data, compressed FL interacts with gradient diversity in complex, unsolved ways.

### Finding 5: Byzantine Robustness and FL Are Deeply Incompatible Without a Trusted Server Reference
- No existing algorithm simultaneously achieves: non-IID convergence + Byzantine robustness + no server-side clean data.
- The problem is fundamentally hard: distinguishing a Byzantine update from a legitimate update on an extreme non-IID client is nearly impossible without additional information.

## 6.2 Failure Cases Explicitly Identified

| Failure Case | Cause | Consequence |
|---|---|---|
| Client drift | Too many local steps on non-IID data | Global model diverges from global optimum |
| Privacy amplification failure | Too few clients per round | DP guarantee degrades rapidly |
| Secure aggregation dropout | Clients drop out during cryptographic protocol | Aggregation must restart; wastes communication |
| Fairness collapse | Aggregation weighted by data size | Majority language/demographic dominates |
| Gradient inversion | Large batch, few updates per round | Raw images/text reconstructible from gradients |

---

### Publishability Strength Check

| Result Type | Publication-Grade? | Notes |
|---|---|---|
| DP formal bounds for FL | Yes | Mathematically rigorous |
| Convergence bounds under non-IID | Yes (with caveats) | Bounds often not tight |
| Communication compression tradeoffs | Yes | Well-validated empirically across papers |
| Byzantine robustness limits | Yes | Supported by theoretical impossibility results |
| Fairness analysis | Partial | Needs formal metrics + empirical validation |
| Vertical FL analysis | Weak | Mostly conceptual; needs experimental grounding |
| Asynchronous FL guarantees | Weak | Active area but understudied at time of writing |

---

# 7. Strengths – Weaknesses – Assumptions

## Technical Strengths

| Strength | Description |
|---|---|
| Comprehensive scope | Covers every major FL challenge in one document |
| Formal problem statements | Each open problem is precisely defined — actionable for researchers |
| Privacy framework completeness | DP + secure aggregation combination is rigorously justified |
| Algorithm taxonomy | Clean classification of optimization, personalization, and compression approaches |
| Cross-regime distinction | Cross-device vs cross-silo distinction prevents category errors |
| Multi-disciplinary integration | Connects distributed systems, cryptography, optimization, fairness, and ML in one framework |
| Influence on research agenda | Explicitly shaped the direction of FL research for 3–5 years post-publication |

## Explicit Weaknesses

| Weakness | Description |
|---|---|
| No reproducible experiments | No single runnable experiment; all empirical claims are referenced, not verified |
| Breadth over depth | Some sections are superficial surveys; topics like vertical FL get limited treatment |
| Temporal limitation | Written in 2019–2020; newer advances (FL + LLMs, FL + transformers) not covered |
| No unified framework | Eleven research themes but no single mathematical framework connecting them |
| Limited hardware analysis | Under-discusses inference-time FL and on-device memory/compute constraints |
| Fairness section is underdeveloped | Compared to privacy and optimization sections, fairness lacks formal treatment |
| Assumes gradient-based learning | All analysis is for gradient-descent-based models; FL for non-differentiable models is unexplored |

## Hidden Assumptions

| Hidden Assumption | Where It Appears | Why It Matters |
|---|---|---|
| Server is computationally powerful | All aggregation algorithms | Breaks for IoT/edge server deployments |
| Clients have known IDs across rounds | Privacy amplification analysis | Violates by anonymous participation |
| Gradient updates are the communication primitive | All compression methods | Alternative: forward passes, activations (split learning) |
| Model architecture is fixed across all clients | Aggregation formulas | Breaks for heterogeneous model FL |
| Clients can run full local SGD | All local optimization methods | Breaks for streaming data (one-pass) clients |
| The number of adversaries is bounded and known | All robustness algorithms | Unknown adversary fraction is the reality |
| Privacy threat is an external attacker seeing the model | DP formulation | Ignores adversary models targeting communication channels |

---

# 8. Weakness → Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| No unified FL framework | Paper grew from isolated sub-communities | Build a single parameterized FL framework covering privacy, robustness, and personalization simultaneously | Modular FL architecture with composable components |
| Privacy-utility tradeoff too harsh for small $n$ | DP noise scales as $1/(n\varepsilon)$ — small $n$ means large noise | Design DP amplification-by-shuffling mechanisms for cross-silo FL | Shuffle model + DP with amplified $\varepsilon$ via shuffler |
| Non-IID + compression unresolved | Quantization error interacts non-trivially with gradient diversity | Adaptive compression that increases precision for high-diversity dimensions | Per-dimension bit allocation based on gradient variance |
| No Byzantine robustness without clean server data | Hard to distinguish malicious from extreme non-IID updates | Develop client reputation systems that track update history | Online trust scoring + CUSUM-based anomaly detection |
| Fairness lacks formal metrics | No agreed-upon definition of "fair" FL | Define and implement multiple fairness criteria in one FL framework | Pareto-frontier analysis of accuracy vs fairness across groups |
| Asynchronous FL is understudied | Synchronous assumption simplified analysis | Develop convergence-guaranteed asynchronous FL with staleness bounds | Lyapunov-based convergence analysis with delay compensation |
| Vertical FL has little formal theory | Focuses on horizontal (sample-split) FL | Derive convergence bounds for split-learning with misaligned entity IDs | Private set intersection + vertical FL optimization |
| Mobile compute constraints ignored | Paper treats computation as free | Co-design model architecture and FL algorithm for on-device resource limits | Neural architecture search (NAS) + FL joint optimization |
| LLMs not addressed | Paper predates LLM dominance | FL for large language model fine-tuning at scale | LoRA-based federated fine-tuning with DP guarantees |
| Model heterogeneity unexplored | Aggregation assumes identical architectures | Enable cross-architecture FL (different clients run different model sizes) | Knowledge distillation-based aggregation (no weight averaging) |

---

# 9. Novel Contribution Extraction

## Template Statements Inspired by This Paper

**Template 1 — Privacy Enhancement**:
> "We propose a *shuffled differentially private federated learning mechanism* that improves the *privacy-utility tradeoff in cross-silo FL* by *amplifying the effective privacy budget through an intermediate shuffler layer*, achieving $\varepsilon < 1$ with less than 2% accuracy degradation on benchmark medical datasets."

**Template 2 — Robustness + Non-IID**:
> "We propose *Heterogeneity-Aware Byzantine-Robust Federated Aggregation (HABRA)* that improves *global model quality under simultaneous non-IID data and Byzantine clients* by *using per-client gradient statistics to jointly detect adversarial updates and correct client drift*, outperforming existing robust aggregators by 8% under high heterogeneity and 20% Byzantine fraction."

**Template 3 — Communication Efficiency**:
> "We propose *Variance-Weighted Federated Quantization (VWQ)* that improves *communication efficiency in non-IID federated learning* by *allocating bit-width per gradient dimension proportional to cross-client gradient variance*, reducing communication cost by 4x while matching full-precision FedAvg accuracy."

**Template 4 — Personalization**:
> "We propose *Hierarchical Federated Meta-Learning (HierFML)* that improves *personalized model quality in cross-device FL* by *organizing clients into clusters with shared priors and learning cluster-level meta-initializations*, achieving 12% better per-client accuracy than Per-FedAvg on FEMNIST."

**Template 5 — LLM Extension**:
> "We propose *FedLoRA-DP*, a *differentially private federated fine-tuning framework for large language models* that improves *privacy-utility tradeoff in NLP applications* by *applying low-rank adapter training with per-layer DP noise calibration*, enabling fine-tuning of 7B-parameter models with $\varepsilon = 3$ and less than 3 perplexity points degradation."

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Work (Explicit in Paper)

- Tight convergence bounds for FedAvg under arbitrary non-IID — especially non-convex objectives
- Scalable secure aggregation: $O(n)$ or $O(n \log n)$ protocols
- Formal fairness guarantees in FL — both training fairness and model output fairness
- Unified personalization theory covering all four personalization paradigms
- FL for structured data beyond images and text (graphs, time series, tabular)
- Vertical FL theory — formal algorithms, convergence, privacy guarantees
- Asynchronous FL with delay-compensated convergence guarantees
- Cross-silo FL governance — how to coordinate institutions without a central authority

## 10.2 Missing Directions (Not in the Paper)

| Missing Direction | Why It Matters |
|---|---|
| FL + foundation models / LLMs | Dominant paradigm post-2022; FL must adapt to parameter-efficient fine-tuning |
| FL + continual learning | Clients' data distributions change over time; catastrophic forgetting in FL |
| FL + reinforcement learning | Distributed policy learning without sharing environment transitions |
| FL + graph neural networks | FL over knowledge graphs, social networks, molecular graphs |
| FL on neuromorphic hardware | Sparsity-aware FL for energy-efficient computation |
| FL + federated evaluation | How to evaluate a global model without accessing test data |
| Incentive mechanisms for FL participation | Game-theoretic analysis of why rational clients would participate honestly |

## 10.3 Modern Extensions (Post-2021)

| Extension | Key Papers | What It Addresses |
|---|---|---|
| FedPEFT / FedLoRA | Li et al. 2023, Zhang et al. 2023 | DP-efficient fine-tuning of LLMs via low-rank adapters |
| FedDF / Ensemble Distillation FL | Lin et al. 2020 | Model heterogeneity — no weight aggregation needed |
| FLUTE, Flower, PySyft v2 | Systems papers 2022 | Production-grade FL infrastructure |
| DP-FTRL | McMahan et al. 2022 | Better DP optimizer for FL — replaces DP-SGD |
| Federated Unlearning | Liu et al. 2021 | Right to be forgotten in FL (GDPR compliance) |
| FL + Homomorphic Encryption | Phong et al. 2018, newer works | Secure computation without decryption at server |

## 10.4 Cross-Domain Combinations

| Combination | Motivation | Research Gap |
|---|---|---|
| FL + Quantum Computing | Quantum secure aggregation; quantum-enhanced privacy | Mostly theoretical; no practical implementation |
| FL + Blockchain | Decentralized FL (no central server) via smart contracts | Byzantine robustness + consensus overhead |
| FL + Digital Twins | Simulate FL rounds before actual training | Simulation fidelity vs real-world gap |
| FL + Causal Inference | Learn causal relationships across distributed datasets | Combining FL with do-calculus formalism |
| FL + AutoML / NAS | Automate model design for FL constraints | NAS is expensive; needs FL-specific search spaces |

## 10.5 LLM-Era Extensions

- **Federated prompt tuning**: Fine-tune only soft prompts with DP guarantees — extremely communication-efficient
- **Federated RLHF (Reinforcement Learning from Human Feedback)**: Privacy-preserving collection of human preference data
- **FL for RAG systems**: Federated retrieval-augmented generation — private document databases across institutions
- **Membership inference in LLMs via FL**: New privacy attacks specific to foundation model FL
- **Cross-device LLM personalization**: On-device LoRA adapters updated via FL without accessing base model weights

---

# 11. How to Write a NEW Paper From This Work

## Reusable Elements

| Element | How to Reuse |
|---|---|
| Non-IID benchmark setup | Use Dirichlet distribution $\text{Dir}(\alpha)$ to partition data with controllable heterogeneity parameter $\alpha$ |
| Privacy accounting method | Use Rényi DP (RDP) accountant for tighter bounds than basic composition |
| Communication measurement | Report total bits transferred to reach target accuracy (not just rounds) |
| Gradient diversity metric $\Gamma$ | Compute and REPORT $\Gamma$ as a dataset characterization — few papers do this |
| Cross-device vs cross-silo distinction | Always specify which regime your paper targets |
| Open problem citation style | Each claim of "improvement" must reference the specific open problem from this paper it resolves |

## What MUST NOT Be Copied

- The exact open problem formulations (these are the paper's intellectual contribution)
- The survey tables (structure new comparisons with new columns)
- The threat model text (restate formally for your specific setting)
- The algorithm classification scheme (use it as inspiration, not copy)

## How to Design a Novel Extension

**Step 1**: Pick ONE open problem from Section 1 (Remaining Open Problems) of this companion.

**Step 2**: Identify which two properties are currently in tension (e.g., non-IID convergence vs Byzantine robustness).

**Step 3**: Design a mechanism that trades off one property slightly to gain significant improvement in the other.

**Step 4**: Derive a formal convergence/privacy/robustness bound for your mechanism.

**Step 5**: Validate on 2–3 standard FL benchmarks using the Dirichlet non-IID setup with $\alpha \in \{0.1, 0.5, 1.0\}$.

**Step 6**: Compare against the appropriate baseline from Section 4 (Pillar 3 or Pillar 4 of this companion).

## Minimum Publishable Contribution Checklist

- [ ] Addresses at least ONE explicitly stated open problem from Kairouz et al. 2021
- [ ] Provides formal theoretical guarantee (convergence bound OR privacy bound OR robustness bound)
- [ ] Compares against at least 3 relevant baselines from the FL literature
- [ ] Tests on at least 2 different heterogeneity levels ($\alpha$ values)
- [ ] Reports results in terms of rounds to accuracy AND total communication cost
- [ ] Discusses privacy implications even if privacy is not the primary contribution
- [ ] Analyzes failure cases — where does the proposed method break?
- [ ] Specifies clearly whether the setting is cross-device or cross-silo
- [ ] Reproducibility: code + experiment configuration provided

---

# 12. Complete Paper Writing Template

---

## Abstract
- **Purpose**: Communicate the contribution, method, and result in 150–250 words.
- **What to include**:
  - The open problem being addressed (cite Kairouz et al. 2021 here)
  - The proposed method name and core idea (one sentence)
  - The theoretical guarantee (state the type: convergence/privacy/robustness)
  - Two key experimental results (specific numbers)
  - One sentence on why this matters
- **Common mistakes**:
  - Starting with "In this paper, we..." (weak)
  - Omitting quantitative results
  - Using jargon without definition
- **Reviewer expectations**: A reviewer should know exactly what problem you solve, what you propose, and how well it works after reading only the abstract.

---

## Introduction
- **Purpose**: Motivate the problem, review the context, state your contribution explicitly.
- **What to include**:
  - Para 1: FL is important because [application motivation] but faces [specific challenge]
  - Para 2: Prior work addresses this via [Approach A], [Approach B] — but both fail because [gap]
  - Para 3: We propose [method name] which addresses [gap] by [mechanism]
  - Para 4: Bulleted list of contributions (3–4 explicitly stated, claiming novelty)
  - Para 5: Paper organization (section map)
- **Common mistakes**:
  - Overstating prior work failures
  - Burying contributions in paragraph 4+
  - Making claims in the introduction that are not backed up in the experiments
- **Reviewer expectations**: Crisp, honest, precisely scoped contribution claims. No vague claims like "we significantly improve performance."

---

## Related Work
- **Purpose**: Position your work in the research landscape.
- **What to include**:
  - Category 1: Papers solving the same problem with different methods — explain why yours is different
  - Category 2: Papers solving adjacent problems — explain what you borrow and what you don't
  - Category 3: Foundational papers (FedAvg, Kairouz 2021) — cite as context setters
- **Common mistakes**:
  - Only citing favorable comparisons
  - Not clearly stating how YOUR work differs from EACH cited group
  - Listing papers without analysis
- **Reviewer expectations**: Evidence that you know the field and you've correctly identified what's novel.

---

## Method
- **Purpose**: Describe exactly what you do, precisely enough to be reproducible.
- **What to include**:
  - Formal problem setup (reproduce the FL objective from Section 3.1 of this companion and extend it)
  - Algorithm description: numbered steps or pseudocode block
  - Explanation of each component and why it was designed this way
  - Comparison to the closest baseline — what changed, and why
- **Common mistakes**:
  - Pseudocode that is ambiguous (missing initialization, loop conditions, output)
  - No justification for design choices
  - Hiding important details in the appendix
- **Reviewer expectations**: The method section should be self-contained. A reader should be able to implement your algorithm from this section alone.

---

## Theory
- **Purpose**: Provide formal guarantees for your method.
- **What to include**:
  - Key theorem statement (convergence rate, privacy bound, or robustness guarantee)
  - List of assumptions — clearly stated, justified as realistic
  - Theorem proof sketch: intuition only in main paper, full proof in appendix
  - Comparison of your bound to the closest prior bound (table format preferred)
- **Common mistakes**:
  - Stating assumptions that contradict the experimental setup
  - Providing bounds that are weaker than baselines (if so, explain the tradeoff)
  - Not interpreting the bound in plain language after the theorem
- **Reviewer expectations**: Assumptions must be explicitly verified in the experimental setup. Bounds must improve upon prior art in at least one dimension.

---

## Experiments
- **Purpose**: Empirically validate every theoretical claim.
- **What to include**:
  - Dataset table: name, size, modality, how non-IID split was created ($\alpha$ value)
  - Baseline table: algorithm name, citation, why it is the appropriate comparison
  - Main result table: your method vs ALL baselines on ALL datasets
  - Ablation study: remove each component of your method one at a time
  - Communication cost plot: bits vs accuracy curve
  - Privacy analysis: if privacy claimed, show $(\varepsilon, \delta)$ at each accuracy level
- **Common mistakes**:
  - No ablation study — reviewer cannot tell which component matters
  - Only reporting best-case hyperparameter setting
  - Missing error bars / confidence intervals
  - Testing on IID data when claiming non-IID improvement
- **Reviewer expectations**: Ablation is non-negotiable. At least 3 baselines. At least 2 datasets. Must use the same evaluation metric as prior work for fairness.

---

## Discussion
- **Purpose**: Interpret results — what do they mean, not just what numbers say.
- **What to include**:
  - Why your method works (mechanism explanation)
  - When does it fail? (boundary conditions)
  - The most surprising result and its implication
  - Practical deployment guidance (how should a practitioner use this?)
- **Common mistakes**:
  - Repeating the results section without adding interpretation
  - Not addressing negative results or failure cases
- **Reviewer expectations**: Shows depth of understanding beyond the experiments.

---

## Limitations
- **Purpose**: Honest self-assessment — shows rigor and intellectual honesty.
- **What to include**:
  - Assumptions that may not hold in practice
  - Settings where your method fails or is not applicable
  - What was NOT tested and why
  - Computational cost compared to baselines (if higher, justify)
- **Common mistakes**:
  - Omitting limitations section entirely (automatic red flag for reviewers)
  - Vague statements like "future work may improve this"
- **Reviewer expectations**: Specific, honest, quantified limitations. Shows you understand your own method's boundaries.

---

## Conclusion
- **Purpose**: Summarize contribution and open new directions.
- **What to include**:
  - One sentence on the problem
  - Two sentences on what you proposed and your key result
  - Two sentences on what this enables or opens up for future work
- **Common mistakes**:
  - Repeating the abstract verbatim
  - Making new claims not supported in the paper
- **Reviewer expectations**: Concise, accurate, forward-looking without overclaiming.

---

## References
- **What to include**:
  - ALL papers compared against in experiments
  - All theoretical results you build on
  - Kairouz et al. 2021 (mandatory for any FL paper post-2021)
  - McMahan et al. 2017 (mandatory for any FL paper)
  - Specific open problem papers from the literature
- **Common mistakes**:
  - Missing foundational citations (FedAvg, DP-SGD, secure aggregation)
  - Citing arXiv preprints where published versions exist
  - Inconsistent citation formatting

---

# 13. Publication Strategy Guide

## Suitable Venues

### Top-Tier Conferences
| Venue | Focus | FL Relevance |
|---|---|---|
| ICML | ML theory + algorithms | Convergence theory, optimization, compression |
| NeurIPS | Broad ML + systems | Privacy, robustness, personalization, LLM FL |
| ICLR | Representation learning + systems | Model quality, personalization, communication |
| AISTATS | Probabilistic ML + stats | Convergence bounds, Bayesian FL |
| CCS / IEEE S&P | Security + cryptography | Privacy, Byzantine robustness, secure aggregation |

### Journals
| Venue | Focus |
|---|---|
| JMLR | Long-form ML theory |
| Foundations and Trends in ML | Surveys and position papers (same venue as this paper) |
| IEEE TIFS | Information forensics and security — DP in FL |
| IEEE TNNLS | Neural networks and learning systems — algorithmic FL |

### Workshops (for early/exploratory work)
- FL workshops at NeurIPS, ICML, ICLR (run annually since 2020)
- Privacy in ML (PriML) workshops
- Trustworthy ML workshops

## Required Baseline Expectations

- **Minimum**: FedAvg (McMahan 2017) always required
- **For privacy papers**: DP-FedAvg (Geyer et al. / McMahan et al. 2018)
- **For non-IID papers**: FedProx + SCAFFOLD
- **For communication papers**: Top-K sparsification + QSGD
- **For robustness papers**: Coordinate-wise median + Krum
- **For personalization papers**: Per-FedAvg + local fine-tuning

## Common Rejection Reasons in FL Papers

| Rejection Reason | How to Avoid |
|---|---|
| "Insufficient comparison to baselines" | Include ALL relevant baselines from Section 13 above |
| "Theoretical bounds are not tight or not new" | Verify your bound improves on the closest prior bound in at least one parameter |
| "Non-IID setup is too simple" | Use Dirichlet partitioning; report results for $\alpha \in \{0.1, 0.5, 1.0\}$ |
| "No ablation study" | ALWAYS include ablation — removes each component one at a time |
| "Claims don't match experiments" | Be conservative in claims; match every claim to a specific table/figure |
| "Missing related work" | Check Kairouz et al. 2021 Section-by-Section reference list |
| "Privacy analysis is informal" | Use formal RDP accounting; report exact $(\varepsilon, \delta)$ |
| "Experiments are at toy scale" | Use at least 100 clients; reference cross-device scale issues honestly |

## Increment Needed for Acceptance

| Venue Tier | Minimum Required Increment |
|---|---|
| Top-tier (NeurIPS/ICML) | Formal theorem + 3+ baselines + significant empirical gain OR fundamental new insight |
| Mid-tier (AISTATS/UAI) | Formal contribution (theory OR method) + solid experiments + 2+ baselines |
| Workshop | Proof of concept + one baseline + clear motivation |
| Journal (JMLR) | Comprehensive study + full theoretical treatment + extensive experiments |

---

# 14. Researcher Quick Reference Tables

## Key Terminology Table

| Term | Plain Meaning | Section Reference |
|---|---|---|
| FedAvg | Federated Averaging — baseline FL algorithm | Section 4 (Pillar 4) |
| Non-IID | Data distributions differ across clients | Section 2.2 |
| $\Gamma$ (gradient diversity) | Measure of how different local optima are | Section 3.2 |
| $(\varepsilon, \delta)$-DP | Formal privacy guarantee with privacy budget $\varepsilon$ | Section 2.3, 3.3 |
| Secure aggregation | Cryptographic protocol hiding individual updates from server | Section 2.4 |
| Client drift | Local models moving away from global optimum due to local over-training | Section 4 (Pillar 4) |
| Byzantine client | Client sending arbitrary malicious updates | Section 2.5, 3.6 |
| Cross-device FL | FL with millions of mobile/IoT devices per deployment | Section 2.1 |
| Cross-silo FL | FL with tens to hundreds of institutions | Section 2.1 |
| Personalization | Adapting global model to individual client distributions | Section 2.6, 4 (Pillar 5) |
| Communication round | One cycle of model broadcast → local training → aggregation | Section 2.5 |
| Gradient inversion | Attack reconstructing raw data from gradient updates | Section 6 |
| SCAFFOLD | Algorithm using control variates to correct client drift | Section 4 (Pillar 4) |
| FedProx | FedAvg + proximal term penalizing local deviation | Section 4 (Pillar 4) |
| Dirichlet partition | Non-IID data split method with heterogeneity parameter $\alpha$ | Section 5 |

## Important Equations Summary

| Equation | Description | Section |
|---|---|---|
| $f(w) = \sum_k p_k F_k(w)$ | Global FL objective | Section 3.1 |
| $\Gamma = f^* - \sum_k p_k F_k^*$ | Non-IID heterogeneity measure | Section 3.2 |
| $\varepsilon$-DP: $\Pr[\mathcal{M}(D) \in S] \leq e^\varepsilon \Pr[\mathcal{M}(D') \in S] + \delta$ | Differential privacy definition | Section 3.3 |
| Accuracy loss $\propto \sqrt{d}/(n\varepsilon)$ | Privacy-utility tradeoff scaling | Section 3.4 |
| $C_{\text{round}} = C \cdot N \cdot 4d$ bytes | Communication cost per round | Section 3.5 |
| Robust bound: $\|\text{Agg} - \nabla f\| \leq c\sigma$ | Byzantine robustness condition | Section 3.6 |

## Parameter Meaning Table

| Parameter | Meaning | Typical Values |
|---|---|---|
| $N$ | Total number of clients | Cross-device: $10^5$–$10^9$; Cross-silo: 10–100 |
| $K$ | Clients sampled per round | Cross-device: 50–500; Cross-silo: all |
| $C$ | Fraction of clients per round ($K/N$) | 0.001–0.1 |
| $E$ | Local epochs per round | 1–10 |
| $B$ | Local minibatch size | 10–256 |
| $\eta$ | Learning rate (local) | 0.001–0.1 |
| $\eta_g$ | Server learning rate (global) | 0.1–2.0 |
| $\varepsilon$ | DP privacy budget | 1–10 (practical); <1 (strong) |
| $\delta$ | DP failure probability | $10^{-5}$–$10^{-7}$ |
| $\alpha$ | Dirichlet non-IID parameter | 0.1 (high heterogeneity) to $\infty$ (IID) |
| $\Gamma$ | Gradient diversity (non-IID degree) | 0 (IID) to large positive (extreme non-IID) |
| $q$ | Fraction of Byzantine clients | 0.01–0.3 in robustness experiments |
| $d$ | Model dimensionality | $10^4$–$10^{10}$ parameters |
| $\sigma$ | DP noise standard deviation | Calibrated to sensitivity and $\varepsilon$ |
| $C_{\text{clip}}$ | Gradient clipping norm for DP | Median gradient norm (tuned) |

## Algorithm Flow Summary

| Algorithm | Type | Key Difference from FedAvg | Use When |
|---|---|---|---|
| FedAvg | Baseline | — | IID data, no privacy/robustness needs |
| FedProx | Optimization | Proximal term limits local drift | Moderate non-IID, dropped clients |
| SCAFFOLD | Optimization | Control variates correct gradient bias | High non-IID, full client participation |
| FedNova | Optimization | Normalizes local updates by step count | Variable local epochs across clients |
| DP-FedAvg | Privacy | Clips + noises gradients before upload | Any privacy requirement |
| SecAgg + DP-FedAvg | Privacy | Hides individual updates from server | Strong privacy against semi-honest server |
| Krum | Robustness | Selects most "agreed-upon" update | Byzantine clients present; IID data |
| Trimmed Mean | Robustness | Discards extreme updates | Known fraction of Byzantine clients |
| Per-FedAvg | Personalization | Meta-learning initialization | Fast local adaptation needed |
| FedDF | Personalization/Heterogeneity | Model distillation instead of averaging | Heterogeneous model architectures |

---

# 15. One-Page Master Summary Card

---

## Problem
Federated Learning (FL) was introduced as a privacy-preserving distributed training paradigm, but a single algorithm (FedAvg) cannot simultaneously address all real-world deployment challenges: heterogeneous data, adversarial clients, privacy leakage, communication bottlenecks, unfairness, and system failures.

---

## Idea
Map the entire FL challenge landscape into a structured taxonomy of open problems across six areas: (1) privacy, (2) robustness, (3) optimization under heterogeneity, (4) personalization, (5) communication efficiency, and (6) systems. Formally define each problem, identify what prior work cannot solve, and establish research directions.

---

## Method
A survey/position paper approach: systematically review all known FL algorithms, identify their failure conditions, state formal open problems, and propose solution directions — combined with new formal analyses of privacy-utility tradeoffs, convergence under non-IID conditions, and robustness bounds.

---

## Key Results
- FedAvg fails under: (a) high non-IID degree, (b) any adversarial client, (c) honest-but-curious server threat model, (d) strict fairness requirements.
- DP accuracy loss scales as $\sqrt{d}/(n\varepsilon)$ — only feasible at massive client scale.
- No existing algorithm simultaneously achieves: convergence + DP + Byzantine robustness + personalization.
- The field lacks standard benchmarks, evaluation protocols, and cross-paper comparable results.

---

## Weakness
- No runnable experiments — all evidence is survey-based.
- Many open problems stated without proposed solutions.
- Vertical FL, asynchronous FL, and FL+LLM directions are underexplored.
- No unified mathematical framework connecting all six problem areas.

---

## Research Opportunity
Design FL systems that provably combine two or more of: (privacy, robustness, non-IID convergence, personalization, communication efficiency) — most papers address only one. The intersection of any two is a valid publishable contribution.

---

## Publishable Extension (Top 3 Directions)
1. **DP + Non-IID Convergence**: Derive tight convergence bounds for DP-FedAvg under Dirichlet non-IID data — show where the privacy-utility-heterogeneity three-way tradeoff lies.
2. **Federated LLM Fine-Tuning with DP**: LoRA-based FL for 7B+ parameter models with formal $(\varepsilon, \delta)$ guarantees and communication compression — addresses the LLM-era gap.
3. **Model-Heterogeneous FL**: Design a knowledge-distillation-based aggregation scheme allowing clients to run different model architectures — no weight averaging needed, enabling cross-device FL with diverse hardware.

---

*This companion file covers the complete structure of Kairouz et al. (2021). Cross-reference with 01_McMahan2017_FederatedLearning_CS2.md for FedAvg foundations, which this paper extends and critiques.*
