# Paper Breakdown: 01_McMahan2017_FederatedLearning

**Full Title:** Communication-Efficient Learning of Deep Networks from Decentralized Data
**Authors:** H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas
**Published:** AISTATS 2017 (arXiv: 1602.05629)
**Institution:** Google, Inc.

---

## Role of This Paper in Your Research Journey

> This is the **foundational paper** of Federated Learning. All other papers in your reading list build on top of what this paper introduced. Understanding it deeply is non-negotiable — it defines the vocabulary, the core algorithm (FedAvg), and the key problems every subsequent paper tries to solve.

---

---

# SECTION 1 — Research Context & Core Problem

---

## 1.1 What Exact Problem Does This Paper Try to Solve?

- Mobile devices (phones, tablets) collect enormous amounts of user data every day.
  - Examples: everything you type, photos you take, GPS traces, voice recordings.
- Normally, to train a machine learning model, all this data would need to be **sent to a central server** (data center).
- This creates **two major problems:**
  1. **Privacy risk:** Sensitive personal data leaves the user's device and sits on a server — exposed to breaches, misuse, or surveillance.
  2. **Communication bottleneck:** Uploading massive amounts of raw data is slow, expensive, and often not feasible (limited bandwidth, metered connections).
- The paper asks: **Can we train a global machine learning model without ever moving the raw data off the device?**

---

## 1.2 Why Is This Problem Important?

- User data on devices is often **more representative and valuable** than data collected from the web (e.g., what users actually type in chat is very different from Wikipedia text).
- But this data is also **deeply private** — passwords, messages, medical details — making centralized collection legally and ethically problematic.
- In 2012, the U.S. White House report on consumer data privacy called for "focused collection" — collect only the minimum necessary. This paper is a direct engineering response to that principle.
- As smartphones became the primary computing device for hundreds of millions of people, this problem became practically urgent.

---

## 1.3 What Gap Existed Before This Paper?

### What existed before:
- **Distributed optimization research:** Papers on training models across multiple computers in data centers — but always assumed:
  - Fast, reliable network connections between machines.
  - Data distributed roughly equally and randomly (IID assumption).
  - A small number of machines (e.g., 16 workers in a cluster).
- **Partial gradient sharing:** Shokri & Shmatikov (2015) proposed sharing only a subset of gradients to save communication, but still did not handle non-IID or unbalanced data.

### What was missing:
- No existing method addressed training on data that is:
  - **Non-IID:** Each device has biased data (your phone only knows how *you* type, not how everyone types).
  - **Unbalanced:** Some users generate far more data than others.
  - **Massively distributed:** Millions of devices, not 16 servers.
  - **Communicationally constrained:** Devices are often offline, on slow Wi-Fi, or battery-limited.

---

## 1.4 What Type of Contribution Is This Paper?

- **Method-based contribution.**
- The paper defines a new problem setting (federated learning), proposes a practical algorithm (FedAvg), and validates it empirically on multiple tasks.
- It does **not** introduce a new dataset or a new model architecture — the contribution is the *optimization algorithm and problem framing*.

---

## 1.5 Research Opportunities That Remain After This Paper

The authors themselves acknowledge these openings at the end of the paper:
- **Privacy guarantees are not formal** — FedAvg does not mathematically guarantee privacy; differential privacy and secure aggregation are future work.
- **Convergence is not theoretically proven** for non-IID data — the paper is purely empirical.
- **Client failures and dropouts** are not handled — if a device goes offline mid-round, no recovery mechanism exists.
- **Heterogeneous hardware** — all clients are assumed to have similar compute capacity; in reality they don't.
- **Personalization** — the global model may not perform well on individual users with unique data distributions.
- **Communication cost reduction** — gradients and model weights can still be very large; further compression is needed.

> These gaps became the research agendas of papers #07 through #13 in your reading list.

---

---

# SECTION 2 — Background Concepts

---

## 2.1 What Is Stochastic Gradient Descent (SGD)?

- A method to train machine learning models by **updating model parameters a little bit at a time**.
- In each step, you pick a small random batch of your training data, calculate how wrong the model is (the "loss"), and nudge the model parameters to reduce that wrongness.
- **Why it matters here:** FedAvg is built on top of SGD. The authors start from SGD and ask: what happens if we run multiple SGD steps *locally on each device* before sending updates?

---

## 2.2 What Is IID vs. Non-IID Data?

- **IID (Independent and Identically Distributed):** If you randomly assigned data to each device, every device would see a representative sample of all users' patterns. This is IID — each device's data looks roughly the same statistically.
- **Non-IID:** In reality, each device only has *one specific user's* data. Your phone's keyboard model sees only how you type. A device that belongs to a Spanish speaker only sees Spanish text. This is non-IID — each device's data is skewed and different from others.
- **Why it matters here:** Almost all previous distributed learning algorithms assumed IID data. This paper is one of the first to explicitly test on non-IID data and show that model averaging still works.

---

## 2.3 What Is a Communication Round?

- In federated learning, a "round" is one complete cycle of:
  1. Server sends the current global model to selected devices.
  2. Each device trains locally.
  3. Each device sends its updated model back to the server.
  4. Server averages all received models into a new global model.
- **Why it matters here:** Each communication round uses bandwidth. The paper's central goal is to **reduce the number of rounds needed** to train a good model.

---

## 2.4 What Is Model Averaging?

- If you have two trained models — take all the numbers (weights/parameters) in Model A, add them to the corresponding numbers in Model B, divide by 2. The result is the average model.
- Intuitively, this should combine what each model learned.
- **Why it matters here:** FedAvg's key insight is that model averaging across devices that started from the **same initialization** works surprisingly well — even when each device trained on completely different data.
- The paper shows experimentally (Figure 1 in the paper) that when two models share the same starting point, averaging their parameters almost always gives a better model than either alone.

---

## 2.5 What Is a Global Model vs. Local Model?

- **Local model:** The model on a single device, trained only on that device's data.
- **Global model:** The model maintained by the central server — it is the weighted average of all local models received in a round.
- **Why it matters here:** The user never shares data. They only share how much their local model *changed* from the global starting point. The global model benefits from everyone's learning without seeing anyone's data.

---

---

# SECTION 3 — Proposed Methodology / Model

---

## 3.1 The FedAvg Algorithm — Overview

FedAvg is the core contribution of this paper. It has three hyperparameters:

| Parameter | What It Controls |
|-----------|-----------------|
| **C** (client fraction) | What fraction of devices participate in each round |
| **E** (local epochs) | How many times each device trains over its local data per round |
| **B** (local mini-batch size) | How many examples each device processes in one gradient step |

---

## 3.2 Step-by-Step Walkthrough of FedAvg

### Step 0 — Server Initialization
- The server starts with a randomly initialized global model (set of numbers representing the model).
- This is the shared starting point for all devices.

### Step 1 — Client Selection
- At the start of each round, the server selects a random **fraction C of all K clients** (e.g., if C=0.1 and there are 100 devices, 10 devices are chosen per round).
- Selected devices must be plugged in, on Wi-Fi, and not actively used (to respect user experience).
- **Why C matters:** Using more clients per round increases communication but gives more diverse gradient information. The paper found C=0.1 (10%) gives a good balance.

### Step 2 — Server Broadcasts Model
- The server sends the current global model weights to all selected clients.
- Each selected client now has an identical copy of the global model as their starting point for local training.

### Step 3 — Local Training on Each Client
- Each client trains this global model on its **own local data** for E epochs, using mini-batches of size B.
- Concretely, this means running multiple SGD update steps locally.
- After local training, each client has a slightly updated model that is better suited to its own data.
- **Key insight:** The more SGD steps run locally (higher E, smaller B), the less communication is needed — but too much local training on biased (non-IID) data can cause the local model to drift away from the global optimum.

### Step 4 — Clients Send Updates to Server
- Each client sends their updated model weights back to the server.
- Note: They send the **full model**, not just the gradient difference. (Though mathematically these are equivalent for one full step, it's slightly different here because multiple local steps are taken.)
- **What is NOT sent:** The actual training data — it never leaves the device.

### Step 5 — Server Aggregates (Weighted Average)
- The server computes a **weighted average** of all received local models.
- The weight for each client is proportional to how much data that client has:
  - A device with 1000 training examples contributes more to the average than a device with 100 examples.
- Formula (simplified): `new_global_model = sum over clients of (client_data_fraction × client_model)`

### Step 6 — Repeat
- The new global model is sent to the next randomly selected group of clients.
- This repeats for many rounds until the model converges.

---

## 3.3 Why Did the Authors Choose This Approach?

- **Simplicity:** FedAvg is just SGD + averaging. No exotic mathematics required.
- **Flexibility:** It reduces to FedSGD (the naive baseline) when E=1 and B=∞, so it strictly generalizes the simpler approach.
- **Empirical motivation:** The authors showed (Figure 1) that model averaging works well when models share the same starting point — which FedAvg guarantees by broadcasting the same global model each round.
- **Communication efficiency:** By doing more work locally (high E, small B), each round of communication does more "useful" training, dramatically reducing total rounds needed.

---

## 3.4 Comparison with FedSGD (The Baseline)

| Property | FedSGD | FedAvg |
|----------|--------|--------|
| Local steps per round | 1 | Many (controlled by E, B) |
| Communication rounds needed | High (baseline) | Much lower (10x–100x reduction) |
| Robustness to non-IID data | Very sensitive | Reasonably robust |
| Extra computation per device | Minimal | Moderate |

---

## 3.5 How Could You Modify FedAvg for a New Paper?

- **Adaptive local epochs:** Instead of fixed E, let clients decide how many local steps to take based on their data quality or model convergence. (Became FedProx — Paper #07 in your list)
- **Partial model sharing:** Only share the layers that changed the most, to reduce upload size.
- **Weighted averaging by validation performance:** Instead of weighting by data count, weight by how well each client's model performs on a held-out validation set.
- **Asynchronous rounds:** Allow fast clients to send updates without waiting for slow ones.

---

---

# SECTION 4 — Dataset / Experimental Setup

---

## 4.1 Datasets Used

### Dataset 1: MNIST (Handwritten Digit Recognition)

| Property | Details |
|----------|---------|
| Task | Classify handwritten digits (0–9) |
| Size | 60,000 training images, 10,000 test images |
| Distribution | Split across 100 simulated clients |
| IID version | Randomly shuffled and distributed equally |
| Non-IID version | Sorted by digit label, then divided into 200 shards of 300 examples each; each client gets 2 shards |

- **Why suitable:** A well-known benchmark that allows comparison. The pathological non-IID version (each client has only 2 different digit classes) is a stress test of the algorithm.
- **Limitation:** MNIST is extremely simple — models achieve 99%+ accuracy even with basic methods. Results may not generalize to hard real-world tasks.

---

### Dataset 2: Shakespeare (Language Modeling)

| Property | Details |
|----------|---------|
| Task | Predict the next character in a line of text |
| Source | Complete Works of William Shakespeare |
| Clients | 1,146 clients — one per speaking role per play |
| Size | ~3.5 million characters training, ~870k test |
| Distribution | Naturally non-IID (each role in a play is one client) and heavily unbalanced (some roles have many more lines) |

- **Why suitable:** This is a **naturally partitioned** dataset — the partition is meaningful (each user = one character in a play), not artificially constructed.
- **Limitation:** Shakespeare text does not realistically represent mobile keyboard usage. The natural language patterns are very different from modern chat/texting.

---

### Dataset 3: CIFAR-10 (Image Classification)

| Property | Details |
|----------|---------|
| Task | Classify 32×32 color images into 10 classes |
| Size | 50,000 training + 10,000 test images |
| Distribution | IID only — 100 clients, 500 training each |

- **Why used:** To validate that FedAvg scales to CNNs on a harder image dataset than MNIST.
- **Limitation:** Only tested in IID mode — non-IID CIFAR experiments were not conducted.

---

### Dataset 4: Large-Scale Social Network Posts (Language Modeling)

| Property | Details |
|----------|---------|
| Task | Predict the next word in a social media post |
| Size | 10 million posts, 500,000+ clients |
| Distribution | Naturally non-IID (each author = one client) |

- **Why suitable:** This is the closest proxy in the paper to real-world federated learning conditions — it has hundreds of thousands of clients with highly variable data sizes.
- **Limitation:** This dataset is not publicly available (internal Google data), so results cannot be independently reproduced.

---

## 4.2 Models Used

| Model | Task | Parameters |
|-------|------|-----------|
| 2NN (MLP, 2 hidden layers of 200 units) | MNIST | ~199,000 |
| CNN (2 conv layers + FC layer) | MNIST & CIFAR-10 | ~1.6M |
| 2-layer character LSTM | Shakespeare | ~866,000 |
| 256-node word LSTM | Social media next-word | ~4.95M |

---

## 4.3 Tools & Evaluation Method

- Implemented in **TensorFlow**.
- Over **2,000 individual training runs** conducted for hyperparameter sweeps.
- Metric: Number of communication rounds to reach a **target test accuracy**.
- Learning rate tuned over a grid of 11–13 values for fair comparison.

---

---

# SECTION 5 — Results & Key Findings

---

## 5.1 Main Result 1 — Massive Reduction in Communication Rounds

- FedAvg achieves **10x to 100x fewer communication rounds** compared to FedSGD (the naive baseline).
- Specific examples from the paper:

| Model | Setting | FedSGD rounds | FedAvg rounds | Speedup |
|-------|---------|--------------|--------------|---------|
| MNIST CNN | IID, 99% target | 626 rounds | 18 rounds | ~35x |
| Shakespeare LSTM | Non-IID, 54% target | 3,906 rounds | 41 rounds | ~95x |
| CIFAR-10 CNN | IID, 85% target | N/A (never converged) | 2,000 rounds | — |
| Large-Scale LSTM | Non-IID, 10.5% accuracy | 820 rounds | 35 rounds | ~23x |

---

## 5.2 Main Result 2 — Works on Non-IID Data (Surprisingly Well)

- Even with the **pathological non-IID MNIST** setup (each client has only 2 digit classes out of 10), FedAvg still converges to good accuracy.
- For the Shakespeare dataset, the non-IID partition actually converged **faster** than the IID version (95x speedup vs 13x speedup) — because some roles had very large datasets, making local training especially valuable.
- **Key lesson:** Non-IID data makes things harder but not impossible. The algorithm is robust.

---

## 5.3 Main Result 3 — More Local Computation = Fewer Rounds

- The hyperparameter `u = (E × n) / (K × B)` (expected local updates per round) is the key driver.
- Higher `u` (more local steps) → fewer communication rounds needed.
- But: too high `u` (very large E) can cause the model to plateau or even diverge — especially for the Shakespeare LSTM.

---

## 5.4 Main Result 4 — FedAvg Often Produces Better Final Models

- Interestingly, FedAvg not only converges faster but also reaches **higher final test accuracy** than FedSGD.
- Example: MNIST CNN — FedSGD reached 99.22% after 1,200 rounds; FedAvg reached 99.44% after only 300 rounds.
- The authors conjecture that model averaging acts like a **regularizer** (similar to dropout), preventing overfitting to local data distributions.

---

## 5.5 Where Performance Dropped

- For very large E on the Shakespeare LSTM, the model **plateaued or diverged** — local over-training caused each client's model to become too specialized, and averaging them produced a poor result.
- On **pathological non-IID MNIST** with full local dataset as a single batch (B=∞), FedAvg with E=5 only achieved a 0.5x "speedup" relative to FedSGD with multi-epoch — meaning it was *worse*.
  - This highlights: large batch + high epochs + non-IID = dangerous combination.

---

## 5.6 What Results Are Strong Enough to Publish?

- The 10-100x communication reduction is highly publishable — it proves federated learning is practical at scale.
- The non-IID robustness finding was novel and directly addressed a gap in prior work.
- The model averaging regularization hypothesis (FedAvg achieving better final accuracy) is noteworthy but not deeply analyzed — an opportunity for a follow-up paper.

---

## 5.7 What Results Need Improvement?

- The convergence guarantee on non-IID data is purely experimental — no theoretical proof exists in this paper.
- The large E divergence issue is flagged but not fully solved.
- The social media LSTM results are not reproducible (private dataset).

---

---

# SECTION 6 — Strengths, Weaknesses & Research Limitations

---

## 6.1 Technical Strengths

- **Elegance:** FedAvg is an extremely simple extension of SGD — easy to implement and understand.
- **Generality:** Tested on 5 model types (MLP, CNN, character LSTM, word LSTM) across 4 datasets — strong evidence of generality.
- **Scalability:** The large-scale LSTM experiment with 500,000 clients demonstrates real-world viability.
- **Non-IID robustness:** First major paper to explicitly test on pathologically non-IID data distributions.
- **Practical framing:** The federated learning problem formulation (Non-IID + Unbalanced + Massively Distributed + Communication-Limited) is precise and has guided an entire research field.

---

## 6.2 Methodological Weaknesses

- **No formal convergence proof:** All results are empirical. There is no mathematical guarantee that FedAvg converges, and under what conditions.
- **Fixed synchronous rounds:** All selected clients must complete their local training and report back before the round ends. If even one device is slow, the round is delayed (stragglers problem).
- **No client failure handling:** If a device goes offline mid-round, it is simply ignored. No mechanism for graceful degradation.
- **Hyperparameter sensitivity:** The optimal E and B values vary per dataset and model — there is no principled way to choose them in advance.
- **Data heterogeneity not fully characterized:** The paper tests one specific form of non-IID partition (2 classes per client on MNIST) but does not explore a range of IID-ness levels systematically.

---

## 6.3 Privacy Limitations

- FedAvg provides **practical privacy** (data stays on device) but **no formal privacy guarantee**.
- Gradient updates can still leak information about the training data — especially for sparse models (e.g., a bag-of-words representation could directly reveal which words a user typed).
- **No differential privacy, no secure aggregation** — these are explicitly listed as future work.
- An attacker with access to the server can attempt gradient inversion attacks to reconstruct training data from model updates.

---

## 6.4 Dataset / Experimental Limitations

- MNIST is too simple — near-perfect accuracy is trivially achievable; it cannot reveal failure modes.
- The social media LSTM dataset is not public — results cannot be reproduced by other researchers.
- CIFAR-10 was only tested under IID conditions — no non-IID CIFAR experiments.
- The Shakespeare dataset, while naturally non-IID, is from 1600s English text — very different from modern mobile text.

---

## 6.5 Assumptions Made by the Authors

- Clients are honest — they do not send manipulated updates (no adversarial clients considered).
- The server is trusted — it sees all model updates and produces the average.
- Synchronous communication — all selected clients complete in the same round.
- Sufficient client participation — at least C×K clients are always available each round.

---

---

# SECTION 7 — Future Scope & Research Opportunities

---

## 7.1 Future Work Suggested by the Authors

- **Formal differential privacy integration:** Apply DP-SGD (Abadi et al. 2016, Paper #04 in your list) to federated learning.
- **Secure aggregation:** Aggregate model updates without the server seeing individual client contributions (Bonawitz et al. 2017, Paper #05 in your list).
- Combining DP and secure aggregation together.

---

## 7.2 Research Directions Not Mentioned (New Opportunities)

### Convergence Theory
- **Opportunity:** Prove mathematically when and why FedAvg converges under non-IID conditions.
- **Why valuable:** Without theory, practitioners cannot trust FedAvg for safety-critical applications.
- **Papers that attempted this:** Li et al. (FedProx, 2020) — Paper #07 in your list.

### Client Drift Problem
- **Opportunity:** When local models deviate too far from the global model (a phenomenon called "client drift"), averaging produces poor results.
- **Solution direction:** Add a proximal term to the local loss function to prevent local models from drifting too far from the global model.

### Personalization
- **Opportunity:** The global model may perform worse than a purely local model for users with highly unique data (e.g., a user who types in a rare dialect).
- **Solution direction:** Fine-tune the global model per user, or train per-layer (shared global layers + personal output layer).
- **Papers that addressed this:** Kulkarni et al. (2020), Paper #09 in your list.

### Communication Compression
- **Opportunity:** Even though rounds are reduced, each round still transmits the full model — millions of floating-point numbers.
- **Solution direction:** Quantize gradients (use fewer bits per number) or sparsify updates (only send top-k changed parameters).
- **Papers that addressed this:** Sattler et al. (2019), Alistarh et al. (2017) — Papers #10, #11 in your list.

### Handling System Heterogeneity
- **Opportunity:** Different devices have different CPUs, memory, and battery levels. Assuming all clients can do the same amount of local computation is unrealistic.
- **Solution direction:** Dynamically assign E (local epochs) per client based on compute capability.

### Byzantine Robustness
- **Opportunity:** A malicious client could send a poisoned model update to corrupt the global model.
- **Solution direction:** Use robust aggregation (median instead of mean, or gradient clipping).
- **Papers that addressed this:** Yin et al. (2018) — Paper #15 in your list.

### Vertical Federated Learning
- **Opportunity:** This paper assumes each client has the same features but different samples. In reality, different organizations may have different features for the same users (e.g., a bank has financial data, a hospital has medical data, for the same person).
- **New area:** Vertical Federated Learning — Paper #25 in your list.

---

---

# SECTION 8 — How This Paper Helps Us Write a New Research Paper

---

## 8.1 What You Can Reuse (Ideas, Structure, Methods)

### Problem Formulation
- The four-property problem definition (Non-IID, Unbalanced, Massively Distributed, Communication-Limited) is a template you can extend.
- Example: Add a fifth property — "Heterogeneous compute" — and build a method that explicitly handles it.

### Algorithm Structure
- The three-parameter framework (C, E, B) is a reusable design pattern.
- Any new federated algorithm you propose can be positioned as a modification of FedAvg, making it easy for reviewers to understand your contribution.

### Experimental Design
- The paper's experimental methodology (fix two parameters, vary one, report rounds-to-target-accuracy) is the standard for federated learning papers.
- You should adopt the same evaluation protocol for direct comparability.

### Baseline
- FedAvg is the primary baseline for every federated learning paper published after 2017.
- Your paper must include FedAvg as a comparison point.

---

## 8.2 What You Must Avoid Copying

- Do not directly reuse the FedAvg pseudocode as your contribution. It is well-known and not novel.
- Do not use only MNIST + Shakespeare as your sole datasets — reviewers will ask why you didn't test on more realistic or harder benchmarks.
- Do not claim communication efficiency as your sole contribution without a theoretical bound — the field has moved beyond purely empirical efficiency claims.

---

## 8.3 What Improvements Are Required for a Novel Contribution

To publish a new paper extending McMahan et al. (2017), you need to address **at least one of:**

| Gap Identified | Possible Novel Contribution |
|---------------|---------------------------|
| No convergence guarantee for non-IID data | Prove convergence under specific non-IID conditions |
| Client drift under high local epochs | New regularization term in local objective |
| No privacy guarantee | Integrate differential privacy with formal epsilon guarantee |
| No handling of heterogeneous compute | Adaptive local epochs per client |
| Full model transmission overhead | Gradient quantization or sparsification with FedAvg |
| No personalization | Hybrid global + local model architecture |
| Synchronous assumption | Asynchronous FedAvg with convergence guarantee |
| Model trained on IID-simulated non-IID | Real-world federated benchmark |

---

## 8.4 How to Design a Publishable Extension

### Step 1 — Pick One Clear Gap
- Choose one weakness from Section 6 above.
- Frame it as a clearly stated research problem with its own motivating example.

### Step 2 — Propose a Modification to FedAvg
- Your modification should be minimal and explainable.
- Justify mathematically why your change addresses the gap.
- Example: Adding a proximal term → derived FedProx (Paper #07).

### Step 3 — Benchmark on Realistic Non-IID Data
- Use the LEAF benchmark (Paper #23, Caldas et al. 2019) which provides realistic non-IID datasets designed for federated learning.
- Run your method against FedAvg, FedSGD, and at minimum one other recent baseline.

### Step 4 — Provide Theoretical Analysis (If Possible)
- Even a convergence bound under simplified assumptions significantly strengthens the paper.
- If theory is too hard, provide extensive ablation studies.

### Step 5 — Target the Right Venue
- The original paper was published at **AISTATS** — appropriate for algorithm-focused FL papers.
- Other strong venues: **ICML, NeurIPS, ICLR, CVPR** (for vision FL), **EMNLP** (for NLP FL).

---

## 8.5 Suggested New Paper Directions Inspired by This Paper

### Direction A — "Adaptive FedAvg with Convergence Guarantee"
- **Problem:** Fixed E leads to client drift on non-IID data without any convergence guarantee.
- **Contribution:** Propose an adaptive E per client based on local gradient norms, and prove convergence.
- **Comparison:** FedAvg, FedProx.

### Direction B — "FedAvg with Quantized Updates"
- **Problem:** Each communication round transmits full model weights (millions of floats).
- **Contribution:** Combine FedAvg with 4-bit or 8-bit quantization, analyze the accuracy-communication tradeoff.
- **Comparison:** FedAvg, FedSGD, existing compression baselines.

### Direction C — "Personalized FedAvg via Layer-Wise Aggregation"
- **Problem:** A single global model cannot represent diverse user behaviors.
- **Contribution:** Propose aggregating only shared lower layers globally, keeping top layers local per user.
- **Comparison:** FedAvg, purely local models, existing personalization baselines.

### Direction D — "FedAvg for Healthcare with Differential Privacy"
- **Problem:** Medical data federated learning needs both efficient training and formal privacy.
- **Contribution:** Integrate DP-SGD into FedAvg's local update step, analyze the privacy-accuracy-rounds tradeoff.
- **Domain:** Healthcare FL (connects to Papers #18, #19 in your list).

---

---

# Quick Summary Card

| Item | Details |
|------|---------|
| **Core Algorithm** | FedAvg — Select fraction of clients, local SGD for E epochs with mini-batch B, weighted average on server |
| **Key Innovation** | More local computation = far fewer communication rounds (10x–100x reduction) |
| **Key Hyperparameters** | C (client fraction), E (local epochs), B (mini-batch size) |
| **Main Finding** | FedAvg is robust to non-IID and unbalanced data; converges in far fewer rounds than FedSGD |
| **Main Weakness** | No formal privacy guarantee, no convergence proof, no handling of malicious clients |
| **Your Baseline** | FedAvg must be your primary comparison in any federated learning paper |
| **Your First Step** | Reproduce FedAvg on MNIST/CIFAR-10 in TensorFlow or PyTorch before proposing modifications |

---

*Paper Breakdown created for research and publication purposes. All explanations are paraphrased and original. No text is copied from the source paper.*
