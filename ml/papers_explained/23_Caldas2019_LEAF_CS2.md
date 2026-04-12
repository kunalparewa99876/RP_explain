# 23_Caldas2019_LEAF — Complete Research Companion

**Paper:** LEAF: A Benchmark for Federated Settings
**Authors:** Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Konecný, H. Brendan McMahan, Virginia Smith, Ameet Talwalkar
**Affiliations:** Carnegie Mellon University, Google, Determined AI
**Year:** 2019
**Website:** https://leaf.cmu.edu
**Code:** https://github.com/TalwalkarLab/leaf/

---

# 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Problem Domain** | Federated Machine Learning Benchmarking |
| **Paper Type** | Systems / Engineering + Empirical |
| **Core Contribution** | A modular open-source benchmark (LEAF) for evaluating learning methods in federated settings |
| **Key Idea** | Existing benchmarks for federated, meta-, and multi-task learning use artificial or inaccessible datasets; LEAF provides realistic, publicly available, reproducible federated datasets with a standardized evaluation protocol |
| **Required Background** | Federated Learning basics, FedAvg algorithm, Meta-Learning (Reptile, MAML), Multi-Task Learning, distributed systems fundamentals |
| **Primary Baseline** | FedAvg (McMahan et al., 2017) |
| **Main Innovation Type** | Infrastructure / Benchmark Design |
| **Difficulty Level** | Beginner–Intermediate (concept-heavy, low math) |
| **Reproducibility Level** | High — fully open-source, preprocessing scripts provided |

---

# 1. Research Context & Core Problem

## 1.1 Exact Problem Formulation

The field of federated learning and related paradigms (meta-learning, multi-task learning) lacked a standard, reproducible, and realistic benchmarking infrastructure. Researchers were either:
- Using artificial data (e.g., randomly splitting MNIST across "fake" clients), or
- Relying on proprietary or hard-to-reproduce datasets.

This made it impossible to fairly compare methods, reproduce results, or understand how solutions behave in realistic conditions.

## 1.2 Why the Problem Exists

- Federated data is generated on millions of real devices (phones, wearables, vehicles).
- Real federated data is **non-IID** (each user has a different data distribution), **unbalanced** (different amounts of data per user), and **privacy-sensitive**.
- Existing datasets like MNIST, CIFAR-10, Omniglot, and miniImageNet are either too simple or treat all data points as interchangeable — they do not reflect real federated properties.
- Proprietary federated datasets (from Google, Huawei, etc.) cannot be shared with the research community.

## 1.3 Historical / Theoretical Gap

Three learning paradigms all need realistic federated benchmarks but none had them:

| Paradigm | Gap Before LEAF |
|---|---|
| Federated Learning | Used artificial partitions of MNIST/CIFAR-10 |
| Meta-Learning | Used Omniglot/miniImageNet which assume balanced k-shot tasks |
| Multi-Task Learning | Used small benchmarks (≤200 tasks), far fewer than real federated networks |

## 1.4 Limitations of Previous Approaches

- **Artificial partitions**: Do not capture natural data heterogeneity across users.
- **Proprietary datasets**: Cannot be reproduced externally.
- **Existing public datasets used in ad-hoc ways**: No standard preprocessing, metrics, or splits — results cannot be compared across papers.
- **Meta-learning benchmarks**: Assume equal samples per task (k-shot), which does not match federated reality where data volumes differ dramatically per user.

## 1.5 Contribution Category

- **System Design**: A modular infrastructure for benchmarking
- **Empirical Insight**: Shows how different methods behave under realistic federated conditions
- **Dataset Curation**: Defines and releases 6 federated datasets
- **Evaluation Protocol**: Introduces new metrics specific to federated settings

### Why This Paper Matters

Before LEAF, two papers using the same federated learning algorithm on "MNIST" could produce incomparable results because each paper partitioned the data differently. LEAF standardizes the entire pipeline — data, metrics, and baselines — so that results become meaningful and comparable across the research community.

### Remaining Open Problems

- No audio, video, or sensor modalities in the initial release.
- No adversarial/poisoning benchmarks built in.
- No official privacy-preserving (differential privacy) evaluation track.
- Synthetic dataset design may not cover all failure modes of meta-learning.
- No benchmark for asynchronous federated settings.
- No cross-silo (institution-level) federated benchmark component.
- Fairness metrics across demographic subgroups are missing.

---

# 2. Minimum Background Concepts

## 2.1 Federated Learning

**Definition:** A way to train machine learning models where data stays on each user's device. Instead of sending raw data to a central server, each device trains locally and only sends model updates (gradients or weights).

**Role in paper:** The primary learning paradigm LEAF is designed to benchmark.

**Why authors needed it:** LEAF's datasets and metrics are specifically designed around the challenges (non-IID data, communication cost, device heterogeneity) that federated learning must solve.

## 2.2 Non-IID Data (Non-Independent and Identically Distributed)

**Definition:** In federated settings, each user's data comes from a different distribution. For example, one user only writes in French while another only writes in English. This contrasts with standard ML datasets where all data comes from the same distribution.

**Role in paper:** The core statistical challenge that LEAF's datasets are designed to reflect. The natural partitioning (by writer, by Twitter user, by Shakespeare character) ensures non-IID data.

**Why authors needed it:** Artificial benchmarks ignore this property; it makes federated learning much harder than standard distributed learning.

## 2.3 FedAvg (Federated Averaging)

**Definition:** An algorithm by McMahan et al. (2017). Each selected device trains for several local steps, then the server averages all device model weights to produce a new global model.

**Role in paper:** The primary reference implementation in LEAF; used as the baseline in almost all experiments.

**Why authors needed it:** FedAvg is the de-facto standard federated learning algorithm; reproducing its known behavior validates LEAF's correctness.

## 2.4 Meta-Learning

**Definition:** Learning how to learn. A model is trained across many tasks so it can quickly adapt to a new task with very few examples.

**Role in paper:** One of the three paradigms LEAF targets. Reptile (a first-order meta-learning algorithm) is used as an additional pipeline on FEMNIST.

**Why authors needed it:** Meta-learning is a natural fit for federated settings (each device = one task), but existing meta-learning benchmarks (Omniglot, miniImageNet) use artificial balanced tasks, not realistic federated-style tasks.

## 2.5 Multi-Task Learning (MTL)

**Definition:** Training one model that simultaneously handles multiple related tasks, sharing representations across tasks to improve overall performance.

**Role in paper:** Another target paradigm. Mocha (a federated MTL algorithm) is included as a reference implementation.

**Why authors needed it:** Real federated networks can be viewed as hundreds of thousands of related tasks (one per device), but existing MTL benchmarks have at most 200 tasks.

## 2.6 FLOPS and Communication Budget (Systems Metrics)

**Definition:**
- **FLOPS**: Floating Point Operations — a measure of computational cost.
- **Communication Budget**: Total bytes uploaded/downloaded between devices and server.

**Role in paper:** LEAF tracks both to evaluate the systems cost of federated methods, not just their accuracy.

**Why authors needed it:** Federated devices have limited battery, compute, and bandwidth. A method that saves communication but uses more compute (or vice versa) may be more or less practical depending on the deployment context.

## 2.7 Percentile-Based Metrics

**Definition:** Instead of reporting only average accuracy, LEAF reports accuracy at the 10th, 50th (median), and 90th percentile across all devices.

**Role in paper:** Captures how methods perform for the weakest devices (10th percentile) vs. the strongest (90th percentile).

**Why authors needed it:** In federated settings, average accuracy can be misleading. A model with high average accuracy might completely fail for a subset of users with rare data.

---

# 3. Mathematical / Theoretical Understanding Layer

This paper is **not mathematics-heavy**. The only formal mathematical content appears in the Synthetic Dataset (Appendix A). The core contributions are architectural and empirical.

## 3.1 Synthetic Dataset Generation (Appendix A)

### Purpose
To create a controlled federated dataset where tasks are naturally clustered but heterogeneous — designed to expose the failure modes of current meta-learning methods.

### Intuition Before Math
Imagine you have groups of users. Within each group, users behave similarly, but between groups, behavior differs substantially. This is more realistic than a single global model but harder than having completely independent users. The synthetic dataset creates this structure mathematically.

### Step-by-Step Logic

**Inputs:** Number of devices T, cluster probability vector (p₁, ..., pₖ)

**Preparation Phase (shared across all tasks):**

| Step | What Happens | Intuition |
|---|---|---|
| Sample cluster means µⱼ | Draw each mean from a Gaussian centered at a random point Bⱼ | Creates k "clusters" of similar tasks in a latent space |
| Draw matrix Q | Random projection matrix from latent space to feature space | Mixes latent cluster identity into observable features |
| Create Σ (diagonal) | Σᵢᵢ = i⁻¹·² | Features along higher indices have less variance (anisotropic input space) |

**Per-Task Generation:**

| Step | What Happens | Intuition |
|---|---|---|
| 1. Assign cluster | Sample cluster center µₜ according to probabilities (p₁,...,pₖ) | Each task belongs to a cluster |
| 2. Draw task model wₜ | Draw local latent vector uₜ ~ N(µₜ, I), set wₜ = Quₜ | Each task has its own model, close to others in its cluster |
| 3. Sample task size nₜ | Log-normal, clipped to [5, 1000] | Mimics realistic unbalanced data across devices |
| 4. Draw input center vₜ | Draw vₜ ~ N(Cₜ, I) | Each task also has its own input distribution |
| 5. Sample inputs xᵢₜ | xᵢₜ ~ N(vₜ, Σ) | Inputs are task-specific and anisotropic |
| 6. Generate labels yᵢₜ | yᵢₜ = argmax(sigmoid(wₜ · xᵢₜ + noise)) | Labels determined by task-specific model with small noise |

### Variable Meaning Table

| Symbol | Meaning |
|---|---|
| T | Total number of devices/tasks |
| k | Number of clusters |
| pⱼ | Probability of a task belonging to cluster j |
| µⱼ | Mean of cluster j in latent space |
| Q | Projection matrix from latent to observable space |
| Σ | Diagonal covariance for inputs (anisotropic) |
| wₜ | True model (weight vector) for task t |
| nₜ | Number of data samples for task t |
| vₜ | Center of input distribution for task t |
| xᵢₜ | i-th input for task t |
| yᵢₜ | Label for i-th sample of task t |

### Practical Interpretation
- The synthetic dataset can expose whether a meta-learning method correctly identifies task clusters.
- The anisotropic input (Σ with decreasing variance) means higher-indexed features are noisy — a robust algorithm should learn to focus on lower-indexed features.
- The task model wₜ = Quₜ ensures that tasks in the same cluster share model structure but remain distinct — exactly the right level of similarity for federated personalization to be beneficial.

### Mathematical Insight Box
> **Key Research Insight:** The critical design choice is that task models wₜ are clustered in latent space (via µⱼ) but not identical. This forces any method to balance between: (a) sharing information across tasks within a cluster, and (b) maintaining task-specific adaptation. If a meta-learning algorithm ignores this multi-cluster structure, it will fail on tasks from minority clusters.

---

# 4. Proposed Method / Framework (MOST IMPORTANT)

## 4.1 LEAF Framework Overview

LEAF is not a learning algorithm — it is a **modular infrastructure** that wraps around any learning algorithm to enable fair evaluation. It has three components:

```
[Raw Data Sources]
       |
       v
[Module 1: Datasets]
  - Download + preprocess scripts
  - Natural user-level partitioning
  - Small / full versions
  - Standardized output format
       |
       v
[Any ML Pipeline / Algorithm]
  (FedAvg, Mocha, Reptile, local training, IID training, etc.)
       |
       v
[Module 2: Reference Implementations]
  - Standard algorithm implementations
  - Produce logs in a standard format
       |
       v
[Module 3: Metrics]
  - Statistical metrics (percentiles, hierarchical)
  - Systems metrics (FLOPS, bytes)
  - Aggregation and analysis tools
```

## 4.2 Module 1: Datasets

### Design Criteria (Why these datasets were chosen)
1. **Natural keyed generation:** Data has a real-world owner (writer, Twitter user, Shakespeare character, celebrity, Reddit user) — not artificially assigned.
2. **Scale:** Thousands to millions of devices — realistic federated scale.
3. **Skewed distribution:** Number of samples per device follows a long-tail distribution, not uniform.

### The Six Datasets

| Dataset | Source | Partition Key | Task | Why Federated? |
|---|---|---|---|---|
| **FEMNIST** | Extended MNIST | Handwriting author | Image classification (62 classes) | Each writer has a unique handwriting style — true non-IID |
| **Sentiment140** | Twitter | Twitter user | Sentiment analysis | Each user has personal language patterns |
| **Shakespeare** | Project Gutenberg | Speaking role per play | Next-character prediction | Each character has a unique speaking style |
| **CelebA** | Large-scale CelebFaces | Celebrity identity | Binary classification (smiling/not) | Each celebrity appears in a natural cluster |
| **Reddit** | Reddit (Dec 2017) | Reddit user | Next-word prediction | Each user has unique vocabulary and topics |
| **Synthetic** | Generated | Per-task design | Multi-class classification | Controlled non-IID with cluster structure |

### Step: Preprocessing Pipeline
For each dataset:
1. Download raw data from public source.
2. Apply dataset-specific parsing (e.g., extract character dialogue from Shakespeare text).
3. Partition by natural key (user/author/role).
4. Create small (5% subsample) and full versions.
5. Output in a standardized JSON/format usable by any ML framework.

**Why authors did this:** Reproducibility. Anyone running LEAF will start from the same processed data.

**Weakness of this step:** Preprocessing is static — the benchmark does not support dynamic data arrival or time-evolving federated distributions.

**Research opportunity:** Design a streaming/temporal LEAF where data arrives over time (concept drift benchmarking).

## 4.3 Module 2: Metrics

### Statistical Metrics

| Metric | What it Measures | Why it Matters |
|---|---|---|
| Mean accuracy | Average across all devices | Standard baseline |
| Median accuracy (50th percentile) | Typical device performance | Less sensitive to outliers than mean |
| 10th percentile accuracy | Performance for the worst 10% of devices | Measures fairness / worst-case |
| 90th percentile accuracy | Performance for the best 10% of devices | Measures best-case ceiling |
| Accuracy by hierarchy | Performance within natural data groups (e.g., by play, by subreddit) | Reveals systematic biases |
| Equal weighting vs. proportional weighting | Every device equally vs. every sample equally | Determines whether power users dominate evaluation |

**Why authors did this:** Average accuracy hides the fact that a model may perform well for most users but catastrophically fail for a minority. In federated settings (healthcare, banking), this minority failure can be unacceptable.

**Weakness:** No fairness metrics (e.g., demographic parity) or differential privacy evaluation budget are built in.

### Systems Metrics

| Metric | What it Measures |
|---|---|
| Total FLOPS | Computational cost on devices (sum across all rounds and devices) |
| Bytes uploaded to server | Communication cost per round (client → server) |
| Bytes downloaded from server | Communication cost per round (server → client) |
| Budget to reach accuracy threshold | How many FLOPS or bytes are needed to achieve a target accuracy |

**Why authors did this:** In real federated deployments, battery life and bandwidth are constrained. A method with slightly lower accuracy but 10x lower communication cost may be far more practical.

**Weakness:** Latency, stragglers, and device availability are not modeled. Real federated systems must handle devices going offline mid-round.

## 4.4 Module 3: Reference Implementations

Three algorithms are included:

| Algorithm | Type | Key Behavior |
|---|---|---|
| **Minibatch SGD** | Distributed baseline | Each device processes a fraction of data; server averages gradients |
| **FedAvg** | Federated learning | Each device runs multiple local SGD steps; server averages weights |
| **Mocha** | Federated Multi-Task Learning | Solves a multi-task optimization problem with communication constraints |

**Why authors did this:** Reproducibility — researchers can compare against these implementations directly, knowing they are correctly implemented.

**Weakness:** Only covers federated learning paradigm. Meta-learning and multi-task learning reference implementations are limited.

---

# 5. Experimental Setup / Evaluation Design

## 5.1 Experiment 1: Reproducibility (Shakespeare + FedAvg)

| Item | Detail |
|---|---|
| Dataset | Shakespeare (subsample: 118 devices, ~5% of total) |
| Task | Next character prediction |
| Model | Character embedding (dim=8) → 2-layer LSTM (256 units) → softmax over vocabulary |
| Sequence length | 80 characters |
| Metric | Top-1 Accuracy |
| Rounds | Multiple, varying number of local epochs E |
| Learning rate | 0.8 |
| Devices per round | 10 |
| Goal | Replicate the divergence behavior noted in McMahan et al. (2017) |

**Why this experiment:** Demonstrates that LEAF produces the same behavior as the original paper, validating correctness.

## 5.2 Experiment 2: Granular Metrics (Sent140 + FEMNIST)

| Item | Sent140 Detail | FEMNIST Detail |
|---|---|---|
| Task | Sentiment analysis | Image classification |
| Model | Bag-of-words + logistic regression | 2 conv layers + pooling + dense (2048 units) |
| Variable | k (min samples per user): k=3 vs. default | C (clients per round), E (local epochs) |
| Metric | Percentile box plot of device accuracy | FLOPS and bytes to reach 0.75 accuracy threshold |
| Data subsample | Full | 5% |

**Why this experiment:** Shows that percentile metrics reveal what mean accuracy hides — a small number of data-poor users degrades the 25th percentile dramatically even when the median is stable.

## 5.3 Experiment 3: Modularity (CelebA, Synthetic, Reddit, FEMNIST)

| Dataset | Algorithm Compared | Split |
|---|---|---|
| CelebA | FedAvg vs. Local Models | 60/20/20 (train/val/test) per user |
| Synthetic | FedAvg vs. Local Models | 60/20/20 per user |
| Reddit | FedAvg vs. Global IID Model | 60/20/20 per user |
| FEMNIST | FedAvg vs. Reptile (meta-learning) | Standard |

**Why this experiment:** Demonstrates that LEAF's dataset module can plug into arbitrary ML pipelines, not just FedAvg.

## 5.4 Baseline Selection Logic

- **FedAvg** is the primary baseline because it is the dominant standard in federated learning literature.
- **Minibatch SGD** is included as a simpler distributed baseline.
- **Local Models** and **Global IID Model** are included as extreme baselines (full personalization vs. full centralization).
- **Reptile** is included to demonstrate compatibility with the meta-learning paradigm.

## 5.5 Hyperparameter Details

| Experiment | Key Hyperparameters |
|---|---|
| Shakespeare convergence | lr=0.8, 10 devices/round, vary E (local epochs) |
| Sent140 statistical | lr=3×10⁻⁴, vary k (min samples per user) |
| FEMNIST systems | lr=4×10⁻³ (FedAvg), lr=6×10⁻² (minibatch SGD) |
| CelebA | 10% clients, lr from {0.1, 0.01, 0.001, 0.0001}, 10 clients/round, 100 rounds |
| Synthetic | 1000 devices, 1 cluster, 60 features, 5 classes, lr from {10⁻³...10³} |
| Reddit | 819 devices, vocab=10000, seq_len=10, embedding_dim=200 |

### Experimental Reliability Analysis

| Aspect | Assessment |
|---|---|
| Reproducibility of Shakespeare convergence | High — matches prior work qualitatively |
| Statistical claim about percentile degradation | Trustworthy — consistent with federated learning theory on non-IID data |
| Systems comparison (FLOPS vs. bytes) | Trustworthy — concrete numbers, clear experimental setup |
| Modularity demonstrations | Trustworthy — straightforward pipeline substitution |
| Absolute accuracy numbers | Moderate — depend on hardware; no confidence intervals reported |
| Synthetic dataset meta-learning failure claim | Requires more experiments — only informally demonstrated |

---

# 6. Results & Findings Interpretation

## 6.1 Main Outcomes

### Shakespeare Convergence Reproduction
- LEAF successfully reproduces the training loss divergence for large numbers of local epochs (E) when using FedAvg on Shakespeare data.
- This validates LEAF as a reliable benchmark — results from the original paper are reproducible using LEAF's infrastructure.

### Statistical Metrics Insight (Sent140)
- Median device accuracy degrades only slightly when data-poor users (with as few as k=3 samples) are included.
- The 25th percentile accuracy degrades dramatically.
- **Interpretation:** If a researcher only reports average or median accuracy, they would conclude the method works well. In reality, a significant fraction of real users (those with few data points) experience much worse performance. LEAF's percentile metrics expose this hidden failure.

### Systems Metrics Insight (FEMNIST)
- FedAvg requires fewer total bytes uploaded compared to minibatch SGD to reach the same accuracy threshold.
- FedAvg requires more total FLOPS (local computation) compared to minibatch SGD.
- **Interpretation:** FedAvg trades local compute for communication savings — this is exactly the right tradeoff for bandwidth-constrained devices.

### Modularity Results (Table 2)

| Dataset | FedAvg | Alternative | Insight |
|---|---|---|---|
| CelebA | 89.46% | Local Models: 65.29% | Federated collaboration helps — global model beats local-only for this dataset |
| Synthetic | 71.89% | Local Models: 87.34% | Local models beat FedAvg — the synthetic dataset's heterogeneity is too high for FedAvg to benefit |
| Reddit | 13.35% | Global IID: 12.60% | FedAvg slightly outperforms centralized IID — minimal gain from federated structure |
| FEMNIST | 74.72% | Reptile: 80.24% | Meta-learning outperforms FedAvg — personalized adaptation is more appropriate for handwriting |

## 6.2 Unexpected Observations

- **Synthetic dataset local models outperform FedAvg (87.34% vs 71.89%):** This confirms the synthetic dataset's design goal — it creates scenarios where naive global model aggregation fails, exposing the need for personalized methods.
- **Reptile outperforms FedAvg on FEMNIST:** This is a strong empirical signal that meta-learning should be taken more seriously for federated benchmarking.

## 6.3 Failure Cases

- Reddit results are universally low (12-13%) — this dataset remains an open challenge.
- No method achieves near-human performance on Reddit next-word prediction in federated settings.

### Publishability Strength Check

| Result | Strength |
|---|---|
| Reproducibility of Shakespeare behavior | Strong — validates benchmark correctness |
| Percentile metric analysis on Sent140 | Strong — clear policy implication |
| Systems metric comparison (FEMNIST) | Strong — concrete, measurable |
| Modularity table (Table 2) | Moderate — limited runs, no confidence intervals |
| Synthetic dataset demonstrating FedAvg failure | Moderate — needs deeper analysis |

---

# 7. Strengths – Weaknesses – Assumptions

## 7.1 Technical Strengths

| Strength | Description |
|---|---|
| Open-source and reproducible | Full code, preprocessing scripts, and documentation are publicly available |
| Natural federated structure | Datasets use real-world user-level partitions, not artificial splits |
| Modular design | Any algorithm can plug into LEAF's pipeline |
| Multi-paradigm coverage | Supports federated learning, meta-learning, and multi-task learning |
| Granular metrics | Percentile-based and systems-based metrics go beyond simple accuracy |
| Scale | Reddit dataset alone has 1.66M users and 56.6M samples |
| Reference implementations | Lowers barrier to entry for new researchers |

## 7.2 Explicit Weaknesses

| Weakness | Description |
|---|---|
| No temporal/streaming data | Datasets are static snapshots; real federated data arrives continuously |
| Limited modalities | No audio, video, or sensor data in initial release |
| No privacy evaluation | No built-in differential privacy metric or attack simulation |
| No asynchronous setting | All experiments assume synchronous federated rounds |
| No adversarial evaluation | No poisoning, backdoor, or Byzantine robustness benchmarks |
| Small reference implementation set | Only FedAvg, minibatch SGD, and Mocha are included |
| No confidence intervals | Results tables show point estimates; no statistical significance testing |
| Static preprocessing | The partition of data into train/val/test is fixed; no dynamic splits |
| No fairness metrics | No measurement of performance across demographic subgroups |
| Limited cross-silo coverage | Designed for cross-device; cross-silo (institution-level) federated is not addressed |

## 7.3 Hidden Assumptions

| Assumption | Implication |
|---|---|
| Natural partition is a good proxy for real federated data | Writing style per author ≈ real phone usage per user — may not hold for all tasks |
| Synchronous rounds are representative | Real federated systems have device dropouts, stragglers, and asynchrony |
| Static dataset reflects real data distribution | User behavior and data distributions evolve over time |
| FLOPS and bytes are sufficient systems metrics | Ignores latency, stragglers, device failure, and power consumption |
| 60/20/20 train/val/test split per user is appropriate | Small users may have too few samples in any split for meaningful local evaluation |
| Single-round accuracy at a fixed threshold captures method quality | Ignores convergence speed, stability, and variance across runs |

---

# 8. Weakness → Research Direction Mapping

| Identified Weakness | Why it Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| No temporal/streaming data | LEAF uses static dataset snapshots | Build a temporal federated benchmark with concept drift | Collect time-stamped user data; simulate sliding-window federated rounds |
| No audio/video modalities | Initial scope limited to text and images | Extend LEAF with speech, sensor, or video datasets | Use LibriSpeech by speaker, or activity recognition by device |
| No privacy evaluation | Privacy metrics are complex to standardize | Benchmark federated methods under formal (ε, δ)-DP guarantees | Integrate the Opacus or TensorFlow Privacy library into LEAF's metrics module |
| No adversarial evaluation | Security requires separate threat model design | Add a Byzantine robustness benchmark track | Introduce controlled poisoning attacks and measure model degradation |
| No asynchronous federated support | Synchrony is simpler to implement and evaluate | Design an asynchronous federated benchmark | Simulate device availability schedules; use staleness-aware aggregation |
| No fairness metrics | Fairness definitions are contested and domain-specific | Introduce demographic parity and equalized odds into LEAF's metrics | Leverage CelebA's rich attribute labels (gender, age) for fairness evaluation |
| No cross-silo benchmarks | Cross-silo involves different threat and data models | Create an institution-level federated benchmark | Use hospital datasets or financial institution data for regulated-domain FL |
| Limited reference implementations | Implementing new algorithms requires research effort | Crowdsource reference implementations via community contributions | Open a plugin interface and accept community PRs (similar to Hugging Face) |
| No confidence intervals in results | Single-run experiments reduce statistical validity | Require multi-run reporting with variance in benchmark protocol | Mandate 5-run average with standard deviation for all benchmark submissions |
| No personalization metrics | Average accuracy hides per-user personalization quality | Add per-user accuracy variance as a first-class metric | Measure the standard deviation of accuracy across users |

---

# 9. Novel Contribution Extraction

## 9.1 What the Authors Claim

> "We propose LEAF, a modular benchmarking framework that improves the reproducibility and realism of federated learning research by providing open-source datasets with natural user-level partitions, a standardized evaluation protocol with percentile-based and systems metrics, and reference implementations of common federated algorithms."

## 9.2 Novel Contribution Templates for New Papers

1. **"We propose [TEMPORAL-LEAF] that improves [federated benchmarking] by [incorporating time-stamped data to simulate concept drift across federated rounds], enabling evaluation of methods on dynamic non-stationary user distributions."**

2. **"We propose [PRIVATE-LEAF] that improves [federated evaluation] by [integrating formal differential privacy guarantees as a first-class benchmark dimension], allowing researchers to jointly optimize accuracy-privacy tradeoffs under standardized privacy budgets."**

3. **"We propose [ADVERSARIAL-LEAF] that improves [federated security research] by [providing a controlled Byzantine-fault benchmark with configurable attacker fractions and attack strategies], enabling fair comparison of robust aggregation methods."**

4. **"We propose [FAIR-LEAF] that improves [federated evaluation] by [introducing demographic parity and per-group accuracy metrics across attributes present in CelebA and FEMNIST], enabling measurement of fairness implications of federated algorithms."**

5. **"We propose [ASYNC-LEAF] that improves [federated benchmarking realism] by [simulating realistic device availability schedules and staleness in model updates], enabling evaluation of asynchronous federated learning algorithms under resource-heterogeneous conditions."**

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Work

- Add datasets from audio and video domains.
- Expand ML tasks (text-to-speech, translation, compression).
- Add more reference implementations with community help.
- Keep LEAF updated with new methods from the research community.

## 10.2 Missing Directions

- **Personalization benchmarking:** LEAF does not have a standardized way to measure how well a method personalizes to individual users.
- **Cross-modal federated learning:** No benchmark covers scenarios where different devices have different input modalities (some have images, some have text).
- **Federated transfer learning:** LEAF does not address how a model pre-trained on a central dataset can be adapted using federated data.
- **Federated continual learning:** No benchmark for models that must learn new tasks without forgetting old ones across federated rounds.
- **Hierarchical federated settings:** Real networks have hierarchy (devices → base stations → cloud); LEAF assumes a flat topology.

## 10.3 Modern Extensions

- **Large Language Model (LLM) federated fine-tuning benchmark:** Evaluate methods like FedPTuning or LoRA in federated settings using realistic user text data.
- **Federated foundation model evaluation:** Benchmark how well foundation models can be adapted with federated data from diverse users.
- **Federated reinforcement learning benchmark:** LEAF-style infrastructure for multi-agent RL in distributed environments.

## 10.4 Cross-Domain Combinations

- **Federated Learning + Healthcare:** LEAF-style benchmark using de-identified patient records from multiple hospitals.
- **Federated Learning + Finance:** Transaction data benchmarks with institution-level privacy requirements.
- **Federated Learning + IoT Sensor Data:** Sensor readings from smart home devices or industrial machines.

## 10.5 LLM-Era Extensions

- **Federated instruction tuning benchmark:** Each user has personalized instruction-following preferences; evaluate FedAvg vs. local fine-tuning of LLMs.
- **Privacy-preserving LLM evaluation on LEAF-Reddit:** Use LEAF's Reddit dataset to benchmark how well LLMs can be fine-tuned with differential privacy without severe accuracy loss.
- **Retrieval-Augmented Generation (RAG) in federated settings:** Each device has a local knowledge base; benchmark how federated RAG compares to centralized RAG.

---

# 11. How to Write a NEW Paper From This Work

## 11.1 Reusable Elements

| Element | How to Reuse |
|---|---|
| Natural-partition dataset design | Apply the same principle to a new domain (healthcare, finance, IoT) |
| Percentile-based metric reporting | Always report 10th/50th/90th percentile accuracy alongside mean |
| Systems budget to reach threshold | Use FLOPS-to-accuracy and bytes-to-accuracy curves as standard figures |
| FedAvg as primary baseline | Any federated learning paper should include FedAvg as minimum baseline |
| Small/full dataset versions | Provide both a fast-prototyping subset and a full-scale version |
| Modular pipeline design | Design your benchmark so the dataset module is separable from the algorithm module |
| Synthetic dataset with controlled heterogeneity | Use the synthetic generation approach to ablate specific properties |

## 11.2 What MUST NOT be Copied

- The specific preprocessing scripts (these are LEAF's implementation).
- The exact dataset splits (unless you are explicitly building on LEAF).
- The LEAF brand name and LEAF website infrastructure.
- The specific model architectures in the reference implementations without citation.

## 11.3 How to Design a Novel Extension

**Step 1:** Identify which dimension LEAF does not cover (privacy, temporality, adversarial robustness, fairness, cross-silo, audio/video).

**Step 2:** Find or collect data that naturally exhibits this property AND has real-world user-level partitions.

**Step 3:** Design a metric that specifically measures performance along this new dimension.

**Step 4:** Include LEAF's existing datasets as baselines (use FEMNIST or Shakespeare) so your benchmark is directly comparable.

**Step 5:** Release preprocessing scripts, standardized splits, and at least one reference implementation.

**Step 6:** Demonstrate your benchmark reveals behaviors that LEAF's existing metrics cannot detect.

## 11.4 Minimum Publishable Contribution Checklist

- [ ] At least 2 new datasets OR 1 new dataset with significantly larger scale than LEAF's existing datasets
- [ ] At least 1 new metric dimension not covered by LEAF (privacy, fairness, adversarial, temporal, etc.)
- [ ] At least 2 baseline algorithms compared using the new metric
- [ ] Results reported with multiple runs and standard deviations
- [ ] Preprocessing code and data splits released open-source
- [ ] At least one finding that challenges a common assumption in the federated learning community
- [ ] Comparison to LEAF's original datasets to demonstrate that the new benchmark captures behaviors that LEAF misses

---

# 12. Complete Paper Writing Template

## Abstract
**Purpose:** State the problem, the gap, your contribution, and the key result in 150–200 words.

**What to include:**
- One sentence: what problem does federated learning face with benchmarking?
- One sentence: what existing approaches fail to do?
- Two sentences: what your framework/dataset/metric provides.
- One sentence: key empirical finding that demonstrates value.

**Common mistakes:** Starting with background instead of the problem. Being too vague about the contribution. Not stating a concrete result.

**Reviewer expectation:** Should be able to understand the entire paper's value from the abstract alone.

---

## 1. Introduction
**Purpose:** Motivate the problem, establish the gap, and summarize your contribution.

**What to include:**
- Paragraph 1: Why federated learning/meta-learning/multi-task learning exists and why it matters.
- Paragraph 2: What existing benchmarks do (and why they fail — use concrete examples of papers using artificial MNIST splits).
- Paragraph 3: Present your framework with a clear list of contributions (bullet points).
- Paragraph 4: Structure of the paper.

**Common mistakes:** Spending too much space on background and too little on the gap. Not citing the specific papers that use artificial benchmarks.

**Reviewer expectation:** The gap must be clearly established with citations.

---

## 2. Related Work
**Purpose:** Position your work relative to existing federated datasets, benchmarking frameworks, and evaluation methodologies.

**What to include:**
- Subsection on existing federated learning datasets (artificial, proprietary, partially public).
- Subsection on meta-learning benchmarks and why they fail for federated settings.
- Subsection on multi-task learning benchmarks and their scale limitations.
- Subsection on related benchmarking frameworks in other ML subfields (if applicable).

**Common mistakes:** Treating related work as a list of summaries instead of a structured argument for why your work is needed.

**Reviewer expectation:** Every related work category must connect to a specific limitation that your paper addresses.

---

## 3. [Your Framework Name]: Core Components
**Purpose:** Describe the datasets, metrics, and reference implementations in detail.

**What to include:**
- Dataset selection criteria (why these datasets, why this partition key).
- Statistics table (number of devices, total samples, mean/stdev per device).
- Metric definitions (with justification for each metric's relevance).
- Reference implementation list (with algorithmic brief for each).

**Common mistakes:** Describing what you built without explaining why each design decision was made.

**Reviewer expectation:** Every design choice should be justified by a specific challenge in federated settings.

---

## 4. Method / Theory (if applicable)
**Purpose:** If you introduce a new dataset generation method, a new metric formula, or a new evaluation protocol, formalize it here.

**What to include:**
- Formal definition of any new metric.
- If a synthetic dataset: full generative process with mathematical specification.
- If a new protocol: pseudocode or step-by-step procedure.

**Common mistakes:** Skipping formalization and leaving the method ambiguous.

**Reviewer expectation:** A reader should be able to reimplement your benchmark from scratch using this section.

---

## 5. Experiments
**Purpose:** Demonstrate that your benchmark is useful, reproducible, and reveals insights that prior benchmarks miss.

**What to include:**
- Experiment 1: Reproducibility — show that a known result can be replicated using your framework.
- Experiment 2: Granular metrics — show that your new metrics reveal something that average accuracy hides.
- Experiment 3: Modularity — demonstrate your framework working with multiple different algorithms.
- Ablation: Show effect of varying a key dataset property (e.g., minimum samples per user).

**Common mistakes:** Only showing results where your framework's baselines perform well. Not showing failure cases or surprising behaviors.

**Reviewer expectation:** At least one result should challenge an assumption common in prior work.

---

## 6. Discussion
**Purpose:** Interpret the results and draw actionable conclusions for the research community.

**What to include:**
- What the results mean for practitioners (which algorithm to choose and when).
- Which metrics should be prioritized for which deployment scenario.
- Which datasets are most challenging and why.

**Common mistakes:** Simply restating results without interpretation.

---

## 7. Limitations
**Purpose:** Honestly state what your benchmark does not cover.

**What to include:**
- Modalities not included.
- Settings not captured (asynchronous, cross-silo, adversarial).
- Metrics not included (privacy, fairness).
- Plans for future expansion.

**Common mistakes:** Making limitations sound like fatal flaws. Acknowledge them as design tradeoffs, not failures.

---

## 8. Conclusion
**Purpose:** Summarize contributions and call to action for the community.

**What to include:**
- One paragraph: What LEAF provides.
- One paragraph: What results showed.
- One paragraph: Invitation for community contributions.

**Common mistakes:** Introducing new ideas in the conclusion.

---

## 9. References
**Critical citations for any federated benchmarking paper:**
- McMahan et al. (2017) — FedAvg
- Smith et al. (2017) — Mocha / Federated MTL
- Finn et al. (2017) — MAML
- Nichol et al. (2018) — Reptile
- Bonawitz et al. (2017) — Secure Aggregation
- McMahan et al. (2018) — Differential Privacy + FL
- Li et al. (2019) — Federated learning challenges survey
- Caldas et al. (2019) — LEAF (this paper, cite when extending)

---

# 13. Publication Strategy Guide

## 13.1 Suitable Venues

| Venue Type | Examples | Why Suitable |
|---|---|---|
| ML Systems workshops | FL-ICML, FL-NeurIPS workshops | Direct audience for federated benchmarks |
| Top ML conferences (benchmarking tracks) | NeurIPS Datasets & Benchmarks, ICML | High visibility; benchmark papers are highly valued |
| Systems conferences | MLSys, SysML | If your contribution has significant systems engineering component |
| Applied ML conferences | ICLR | If your benchmark is accompanied by a new algorithm that uses it |

## 13.2 Required Baseline Expectations

Any paper extending LEAF must include:
- FedAvg as minimum federated learning baseline.
- At least one IID/centralized model as lower bound.
- At least one fully local (per-device) model as upper bound on personalization.
- If proposing new metrics: compare against existing LEAF metrics to show what is gained.

## 13.3 Experimental Rigor Requirements

- **Multiple runs with standard deviation** (LEAF's original paper did not do this — this is a weakness to fix in extensions).
- **Statistical significance testing** if claiming one method is better than another.
- **Ablation studies** on key dataset properties (e.g., vary level of non-IID-ness, vary data imbalance).
- **Wall-clock time** in addition to FLOPS (FLOPS are theoretical; real timing matters for practitioners).

## 13.4 Common Rejection Reasons

| Reason | How to Avoid |
|---|---|
| "This is just another dataset paper" | Show that your benchmark reveals empirically that existing metrics/methods are insufficient |
| "Insufficient scale" | Ensure at least one dataset has thousands of devices and millions of samples |
| "No new insights from the benchmark" | Every experiment must contain a finding that challenges or refines existing knowledge |
| "Results are not reproducible" | Release full code, dataset splits, and hyperparameters — not just scripts |
| "Benchmark is too narrow" | Support at least 3 different learning paradigms (FL, meta-learning, MTL) |

## 13.5 Increment Needed for Acceptance

To publish a paper that extends LEAF:
- Minimum: 2 new datasets + 1 new metric dimension + open-source release + 1 surprising empirical finding.
- Strong: 4+ new datasets + 2+ new metric dimensions + new reference implementation + theoretical analysis of benchmark properties.
- Best: All of the above + integration with existing LEAF codebase + community adoption before publication.

---

# 14. Researcher Quick Reference Tables

## 14.1 Key Terminology Table

| Term | Definition |
|---|---|
| Federated Learning | Training ML models on distributed devices without centralizing raw data |
| Non-IID | Each device's data comes from a different distribution |
| FedAvg | Algorithm that averages locally-trained model weights across devices |
| LEAF | Learning in Federated Environments — the benchmarking framework in this paper |
| FEMNIST | Federated Extended MNIST — handwritten digits/letters partitioned by writer |
| Sent140 / Sentiment140 | Tweet sentiment dataset partitioned by Twitter user |
| Shakespeare | Next-character prediction dataset partitioned by speaking role per play |
| CelebA | Celebrity face dataset partitioned by celebrity identity |
| Meta-Learning | Learning to adapt quickly to new tasks from few examples |
| Reptile | First-order meta-learning algorithm (simpler than MAML) |
| Multi-Task Learning | Jointly training on multiple related tasks to improve performance |
| Mocha | Federated multi-task learning algorithm |
| FLOPS | Floating Point Operations — measure of computation cost |
| Percentile metric | Performance at the 10th/50th/90th percentile across all devices |
| Power user | A device with disproportionately more data than average |
| k (min samples) | Minimum number of data points required for a device to be included |

## 14.2 Important Equations Summary

| Equation | Source | Purpose |
|---|---|---|
| wₜ = Q·uₜ | Synthetic dataset (Appendix A) | Task-specific model generated by projecting latent cluster membership |
| uₜ ~ N(µₜ, I) | Synthetic dataset | Local model vector sampled near the assigned cluster center |
| nₜ = min(mₜ + 5, 1000) | Synthetic dataset | Number of samples per task (log-normal, clipped) |
| xᵢₜ ~ N(vₜ, Σ) | Synthetic dataset | Input features drawn from task-specific anisotropic distribution |
| yᵢₜ = argmax(sigmoid(wₜxᵢₜ + noise)) | Synthetic dataset | Label generation with small noise injection |

## 14.3 Parameter Meaning Table

| Parameter | Meaning | Typical Value in Paper |
|---|---|---|
| C | Number of clients selected per round (FedAvg) | 10 |
| E | Number of local training epochs per round (FedAvg) | Varies (divergence study) |
| k | Minimum number of samples per user (Sent140 filtering) | 3 or default |
| lr | Learning rate | 0.8 (Shakespeare), 3×10⁻⁴ (Sent140), 4×10⁻³ (FEMNIST FedAvg) |
| T | Number of devices (synthetic dataset) | 1,000 |
| k (clusters) | Number of clusters in synthetic dataset | 1 (in paper's experiment) |
| d | Feature dimension (synthetic dataset) | 60 |
| s | Latent dimension (synthetic dataset) | Not specified, implied small |

## 14.4 Algorithm Flow Summary

### FedAvg (McMahan et al., 2017)
```
1. Server initializes global model w₀
2. For each round t:
   a. Server selects C devices randomly
   b. Each device downloads wₜ
   c. Each device runs E local SGD epochs on its own data
   d. Each device uploads updated weights w_device
   e. Server averages: wₜ₊₁ = average(all w_device)
3. Repeat until convergence
```

### Minibatch SGD (Distributed Baseline)
```
1. Server initializes global model w₀
2. For each round t:
   a. Each device computes gradient on a fraction of its data
   b. Server collects gradients and averages them
   c. Server updates: wₜ₊₁ = wₜ - lr × average_gradient
3. Repeat until convergence
```

### Reptile (Meta-Learning Baseline)
```
1. Initialize meta-model w₀
2. For each meta-iteration:
   a. Sample a batch of tasks (devices)
   b. For each task: run k steps of SGD starting from w₀ → get wₜₐₛₖ
   c. Update: w₀ = w₀ + ε × (average(wₜₐₛₖ) - w₀)
3. At test time: fine-tune w₀ on new task with few examples
```

---

# 15. One-Page Master Summary Card

| Section | Content |
|---|---|
| **Problem** | Federated learning, meta-learning, and multi-task learning lack realistic, reproducible, open-source benchmarks. Existing datasets are either artificial (random MNIST splits), proprietary, or hard to reproduce. |
| **Idea** | Create a modular benchmark framework (LEAF) with: (1) naturally-partitioned real federated datasets, (2) metrics that capture the full distribution of device performance and systems costs, and (3) reference implementations for comparison. |
| **Method** | Curate 6 datasets (FEMNIST, Sent140, Shakespeare, CelebA, Reddit, Synthetic) using real-world user-level partitions. Define percentile-based statistical metrics and FLOPS/bytes-based systems metrics. Implement FedAvg, minibatch SGD, and Mocha as baselines. Release all code and preprocessing scripts. |
| **Results** | (1) LEAF reproduces known FedAvg divergence behavior on Shakespeare. (2) Percentile metrics reveal that data-poor users suffer dramatically even when median accuracy is stable. (3) FedAvg saves bytes vs. minibatch SGD at the cost of more FLOPS. (4) Meta-learning (Reptile) outperforms FedAvg on FEMNIST; local models outperform FedAvg on the synthetic dataset. |
| **Weakness** | No temporal/streaming data. No audio/video. No privacy evaluation. No adversarial robustness track. No fairness metrics. Synchronous-only. No confidence intervals in results. |
| **Research Opportunity** | Build extensions targeting: temporal federated benchmarks, privacy-aware evaluation, Byzantine robustness tracks, fairness metrics, cross-silo benchmarks, LLM fine-tuning in federated settings. |
| **Publishable Extension** | Choose one missing dimension (privacy, fairness, adversarial, temporal, audio/video, LLM). Collect data with natural user partitions in that domain. Design a new metric. Show empirically that existing LEAF metrics fail to capture this new dimension. Release open-source. Submit to NeurIPS Datasets & Benchmarks or a federated learning workshop. |

---

*Generated using Docling (PDF extraction) + Claude Sonnet 4.6 (analysis)*
*Paper: 23_Caldas2019_LEAF | Date: 2026-04-11*
