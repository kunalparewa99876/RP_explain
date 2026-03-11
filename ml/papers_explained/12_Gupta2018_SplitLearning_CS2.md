# Research Companion: Split Learning for Health
**Paper:** "Split Learning for Health: Distributed Deep Learning without Sharing Raw Patient Data"
**Authors:** Praneeth Vepakomma, Tristan Swedish, Otkrist Gupta, Ramesh Raskar (MIT)
**Year:** 2018
**Reference:** Gupta, O. & Raskar, R. (2018). Distributed learning of deep neural network over multiple agents. *Journal of Network and Computer Applications*, 116, 1–8.

---

# 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Paper Title** | Split Learning for Health: Distributed Deep Learning without Sharing Raw Patient Data |
| **Problem Domain** | Privacy-preserving distributed deep learning in healthcare |
| **Paper Type** | Systems / Engineering + Algorithmic |
| **Core Contribution** | Multiple SplitNN configurations for real-world federated healthcare scenarios |
| **Key Idea** | Split a neural network at a "cut layer" — clients process only the early layers; a server processes the rest — so raw data and full model never leave their owners |
| **Required Background** | Neural networks, backpropagation, federated learning basics, healthcare privacy regulations (HIPAA) |
| **Primary Baseline** | Federated Learning (FedAvg), Large Batch Synchronous SGD |
| **Main Innovation Type** | Architectural / System Design — new configurations of an existing framework |
| **Difficulty Level** | Beginner–Intermediate (conceptually simple; few equations) |
| **Reproducibility Level** | Moderate — results reference [32] (Gupta & Raskar 2018 journal paper); specific split-point and hyperparameter details require companion work |

---

# 1. Research Context & Core Problem

## 1.1 Exact Problem Formulation

Large healthcare institutions — hospitals, radiology centers, pathology labs, tele-health centers — each hold different fragments of patient data. Training one powerful AI model that uses all this data would be ideal, but the data cannot legally or ethically be moved from where it lives.

The question the paper formally asks is:

> "Can health entities collaboratively train deep learning models without sharing sensitive raw data?"

## 1.2 Why the Problem Exists

- **Legal barriers:** HIPAA (and equivalent international laws) prohibit sharing identifiable medical records with third parties.
- **Trust barriers:** Hospitals are unwilling to give competitors a view of their patient populations or imaging archives.
- **Technical barriers:** Even when sharing is permitted, moving large medical imaging datasets is expensive and slow.
- **Data modality fragmentation:** One institution has X-ray images; another has lab results; neither alone is sufficient for a high-quality diagnostic model.

## 1.3 Historical / Theoretical Gap

Before this paper, distributed deep learning had only two practical frameworks:

| Existing Method | Problem |
|---|---|
| **Federated Learning (FedAvg)** | Clients must run the full forward + backward pass locally — requires powerful client hardware; also clients must share model weights with a central server |
| **Large Batch Synchronous SGD** | Requires very high communication bandwidth; designed for data centers, not small hospitals |

Neither method addressed:
- Vertically partitioned data (different modalities at different sites)
- Learning without sharing labels
- Multi-task collaborative learning
- Low-resource clients (small clinics, edge devices)

## 1.4 Contribution Category

| Category | Present? |
|---|---|
| Theoretical | Partial — intuitive argument, no formal privacy proofs |
| Algorithmic | Yes — new split learning configurations |
| System Design | Yes — architecture for multi-entity healthcare collaboration |
| Optimization | Partial — resource efficiency shown empirically |
| Empirical Insight | Yes — accuracy vs. compute trade-off comparisons |

## 1.5 Why This Paper Matters

Split learning cuts client-side compute cost by roughly **190×** compared to federated learning on the same task. For small or resource-constrained health facilities (community clinics, mobile health units), this makes collaborative AI training accessible without specialized GPUs. It also extends distributed learning to scenarios that federated learning simply cannot handle, such as multimodal data split across institutions.

## 1.6 Remaining Open Problems

- No formal proof of privacy (inference attacks on cut-layer activations are known to be possible)
- Performance with heterogeneous or non-IID health data not deeply explored
- Communication cost scales with number of clients in vanilla configuration
- No mechanism to detect or mitigate poisoning / Byzantine clients
- Label privacy in some configurations still an open concern
- How to choose the cut layer optimally is not addressed

---

# 2. Minimum Background Concepts

## 2.1 Deep Neural Network — Forward and Backward Pass

**Plain definition:** A neural network consists of layers that transform input data step-by-step. "Forward pass" = data flows from input → output, making a prediction. "Backward pass" (backpropagation) = error signal flows from output → input, updating layer weights to reduce error.

**Role in the paper:** SplitNN physically divides a network at a chosen middle layer. The forward pass happens partly on the client, partly on the server. Backpropagation works the same way, just in reverse across the same split.

**Why authors needed it:** Without understanding layer-wise computation, you cannot design a system that separates part of the computation to different machines.

## 2.2 Cut Layer (Split Point)

**Plain definition:** The cut layer is the specific layer of the neural network where the model is divided. Everything before and including the cut layer runs at the client. Everything after runs at the server.

**Role in the paper:** It is the architectural pivot of the entire SplitNN design. The output of the cut layer (called "smashed data" or activations) is the only thing the client sends to the server.

**Why authors needed it:** It eliminates the need to share raw data. The server sees only an intermediate mathematical representation, not the original images or records.

## 2.3 Activations (Smashed Data)

**Plain definition:** The numerical output values produced at any layer after transforming the input. At the cut layer, these are the values shipped from client to server.

**Role in the paper:** They are the sole payload sent across the network. They are far smaller than the raw data in early layers of CNNs, making communication efficient.

**Note:** These activations are *not* fully privacy-proof — research has shown that some raw data can be reconstructed from them. The paper does not deeply address this.

## 2.4 Federated Learning (FedAvg)

**Plain definition:** A method where each client trains a full copy of the model locally and only shares model weight updates (gradients or parameters) with a center server for aggregation.

**Role in the paper:** Primary baseline for comparison. SplitNN is shown to require dramatically less client computation.

**Why authors needed it:** As the most well-known competitor, it anchors all resource comparisons.

## 2.5 Vertically Partitioned Data

**Plain definition:** Different institutions hold different *features* (columns) of the same patients — e.g., Hospital A has imaging, Hospital B has lab results — rather than different patients.

**Role in the paper:** SplitNN configuration 3 is specifically designed for this scenario. Clients each train a partial model on their feature subset, then their cut-layer outputs are concatenated before reaching the server.

**Why authors needed it:** Most distributed learning research assumes horizontally partitioned data (same features, different samples). Vertical partitioning is common in healthcare.

## 2.6 HIPAA

**Plain definition:** A U.S. law (Health Insurance Portability and Accountability Act) that strictly controls how patient health data is stored, used, and shared.

**Role in the paper:** Establishes the legal motivation — sharing raw data across institutions is not legally permissible, so a method must exist where raw data never leaves its origin.

## 2.7 Gradients

**Plain definition:** The mathematical signals computed during backpropagation that tell each weight how much to change to reduce error. In split learning, only the gradients *at the cut layer* are sent back to the client.

**Role in the paper:** The cut-layer gradient is the *only* information that flows from server back to client — not predictions, not model weights.

---

# 3. Mathematical / Theoretical Understanding Layer

*This paper is not mathematically heavy — it does not contain formal proofs or dense equations. This section covers the core computational ideas in plain terms.*

## 3.1 The Split Learning Computation Flow (Formal Intuition)

Let a full deep network be defined as a sequence of layers:

$$f = f_n \circ f_{n-1} \circ \cdots \circ f_2 \circ f_1$$

The cut layer is at position $k$. The client computes:

$$\mathbf{h}_k = f_k \circ f_{k-1} \circ \cdots \circ f_1(\mathbf{x})$$

where $\mathbf{x}$ is raw input data. Only $\mathbf{h}_k$ (the activation at cut layer $k$) is sent to the server.

The server computes:

$$\hat{y} = f_n \circ f_{n-1} \circ \cdots \circ f_{k+1}(\mathbf{h}_k)$$

### Variable Meaning Table

| Symbol | Meaning |
|---|---|
| $f_i$ | Transformation performed by layer $i$ |
| $k$ | Index of the cut layer |
| $\mathbf{x}$ | Raw input (e.g., medical image) — stays at client |
| $\mathbf{h}_k$ | Cut-layer activations — only value shared from client to server |
| $\hat{y}$ | Prediction (output of full network) |
| $n$ | Total number of layers |

## 3.2 Backpropagation Across the Cut

During backward pass:
- Server computes $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_k}$ (gradient of loss with respect to cut-layer activations)
- This gradient vector is sent back to the client
- Client uses it to backpropagate through layers $f_1, \ldots, f_k$ and update its weights

Only $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_k}$ crosses the network boundary — not loss values, not model architecture, not labels.

## 3.3 Mathematical Insight Box

> **Key Insight for Researchers:** The client computes a compressed learned representation ($\mathbf{h}_k$) rather than sending raw data. The communication efficiency comes from CNNs having fewer parameters in early layers. In VGG and ResNet, early convolutional layers are compact — so $\mathbf{h}_k$ is much smaller than the full model weights that federated learning must share.

## 3.4 Assumption Table

| Assumption | Implication |
|---|---|
| The server is "honest but curious" | Server does not actively attack but could try to reconstruct input from $\mathbf{h}_k$ |
| The cut layer is fixed in advance | No dynamic/adaptive split-point optimization |
| Network architecture is agreed upon | Clients and server must coordinate on model structure |
| Labels reside with server (vanilla config) | Not necessarily true in healthcare; U-shaped config resolves this |

---

# 4. Proposed Method / Framework

## 4.1 Overview of SplitNN

SplitNN physically divides a neural network across computing nodes. Rather than each node running the full model (as in federated learning), each node runs only a portion. The outputs of one portion become the inputs to the next.

**Main principle:** The raw data never leaves the client. The server only ever sees learned intermediate activations.

## 4.2 Configuration 1 — Vanilla Split Learning

**Pipeline:**
1. Client receives a batch of raw data (e.g., retinal images)
2. Client runs forward pass through layers 1 → cut layer
3. Client sends *activations at cut layer* to server
4. Server runs forward pass through cut layer+1 → output layer
5. Server computes loss, runs backward pass to cut layer
6. Server sends *gradients at cut layer* back to client
7. Client runs backward pass from cut layer → input layer
8. Both update their respective weights

**Why authors did this:** Simplest possible configuration; establishes the baseline for all comparisons and extensions.

**Weakness of this step:** Labels are held by the server, so the client reveals an association between its data and a task outcome (the server sees which client's activations produced each gradient). Also, only one client can train at a time per server (sequential training).

**Research idea seed:** Can activations be perturbed with differential privacy noise before sending? Can a scheduling protocol allow multiple clients to train in parallel or in a pipeline?

## 4.3 Configuration 2 — U-Shaped Split Learning (No Label Sharing)

**Pipeline:**
1. Client runs forward pass → cut layer → sends activations to server
2. Server runs forward pass → all its layers
3. Server sends final-layer outputs *back to the client* (the "U-shape")
4. Client computes loss (using its own labels) and generates gradients
5. Client sends gradients (not labels) back to server
6. Standard backpropagation continues from server through client

**Why authors did this:** In healthcare, labels themselves can be sensitive (e.g., whether a patient has a terminal disease). This configuration ensures labels never leave the client.

**Weakness:** The client runs slightly more computation (holds both early layers and loss layer). Server must return its last-layer outputs, increasing upward communication bandwidth.

**Research idea seed:** Can privacy guarantees be formalized for label protection under this configuration? What happens when labels are noisy or missing?

## 4.4 Configuration 3 — Vertically Partitioned Data

**Pipeline:**
1. Client A (radiology center) runs forward pass on imaging data → cut layer A → sends $\mathbf{h}_k^A$
2. Client B (pathology lab) runs forward pass on lab data → cut layer B → sends $\mathbf{h}_k^B$
3. Server concatenates [$\mathbf{h}_k^A$, $\mathbf{h}_k^B$] and runs remaining layers
4. Gradients split and returned to respective clients

**Why authors did this:** Directly models the real-world situation where no single institution has complete data for a patient. Cross-modal fusion happens inside the model, not by sharing raw data.

**Weakness:** Synchronization required — both clients must have activations available simultaneously. If one client is slow or offline, training stalls.

**Research idea seed:** Asynchronous vertical split learning; partial-data robustness (what if Client B is missing for some patients?).

## 4.5 Configuration 4 — Extended Vanilla Split Learning

Same as vanilla, but after the initial cut, a second client further processes the concatenated outputs before sending to the server. Adds an additional processing stage at a client node.

**Research idea seed:** Hierarchical or cascaded split learning for IoT-edge-cloud architectures.

## 4.6 Configuration 5 — Multi-Task Split Learning

Multiple servers, each trained on a different supervised task, all fed from the same client cut-layer outputs (possibly with multi-modal input from multiple clients).

**Why useful:** One institution's data contributes to multiple research problems simultaneously without exposing any raw data.

**Research idea seed:** Meta-learning across multiple split-learning tasks; shared representation quality with multi-task objectives.

## 4.7 Configuration 6 — Tor-like Multi-Hop Split Learning

Multiple clients train in sequence. Client 1 trains from input → cut1 → sends to Client 2, which trains cut1 → cut2 → sends to Client 3, etc., until the final client sends to the server. Inspired by Tor (onion routing architecture).

**Why useful:** When trust is very limited and no client should see too much of the representation.

**Weakness:** Sequential bottleneck; a single slow or unavailable intermediate client breaks the entire chain.

**Research idea seed:** Fault-tolerant multi-hop configurations; privacy amplification across hops.

## 4.8 Simplified Pseudocode: Vanilla SplitNN Training Round

```
For each mini-batch (x, y) — x at client, y at server:

  CLIENT (Forward):
    h_k = client_model_forward(x)   # x = raw data, stays at client
    send(h_k) → server

  SERVER (Forward + Backward):
    y_hat = server_model_forward(h_k)
    loss = compute_loss(y_hat, y)
    grad_h_k = server_model_backward(loss)  # gradient at cut layer
    update_server_weights()
    send(grad_h_k) → client

  CLIENT (Backward):
    client_model_backward(grad_h_k)  # use cut-layer gradient
    update_client_weights()
```

---

# 5. Experimental Setup / Evaluation Design

## 5.1 Datasets

| Dataset | Task | Architecture | Client Setup |
|---|---|---|---|
| CIFAR-10 | 10-class image classification | VGG | 100 clients |
| CIFAR-100 | 100-class image classification | ResNet-50 | 500 clients |

**Note:** These are computer vision datasets, not actual medical data. The paper uses them as representative benchmarks because labeled medical datasets at this scale are not publicly available.

## 5.2 Baselines

| Method | Description |
|---|---|
| **Federated Learning (FedAvg)** | McMahan et al. 2017 — full model at each client |
| **Large Batch Synchronous SGD** | Data-parallel training with synchronous gradient aggregation |
| **SplitNN (proposed)** | Split at early cut layer |

## 5.3 Primary Metrics

| Metric | What It Measures | Why Used |
|---|---|---|
| Validation Accuracy | Quality of trained model | Primary success metric |
| Client-Side TFlops | Computation required per client | Key resource efficiency metric for small health entities |
| Communication Bandwidth (GB/client) | Data transferred per client | Connectivity constraint in remote / under-resourced settings |

## 5.4 Hyperparameter and Design Notes

- Cut layer placement: early convolutional layers (exploits CNN parameter distribution — early layers have far fewer parameters)
- Number of clients: 100 and 500 (to test scalability)
- Exact hyperparameters (learning rate, batch size, number of rounds) reference the companion journal paper [32]

## 5.5 Hardware / Compute Assumptions

- Client devices are assumed to have very limited compute (no high-end GPU needed)
- Communication channel is low bit-rate (the paper references "snail-pace" connections)
- Server is assumed to have adequate compute to run the heavier layers

## 5.6 Experimental Reliability Analysis

| Aspect | Assessment |
|---|---|
| Trustworthy | Resource efficiency numbers (TFlops, bandwidth) are clearly stated with exact values; accuracy trends are shown graphically |
| Questionable | CIFAR is not medical data — generalization to real EHR or PACS images is assumed, not proven |
| Missing | No privacy attack evaluation; no experiments on vertically partitioned or U-shaped configurations; no real heterogeneous (non-IID) data tests |
| Dependency on [32] | Most experimental results are from the companion paper; this paper is primarily a configuration proposal paper |

---

# 6. Results & Findings Interpretation

## 6.1 Main Outcomes

**Client Computation (CIFAR-10 / VGG):**

| Method | 100 Clients | 500 Clients |
|---|---|---|
| Large Batch SGD | 29.4 TFlops | 5.89 TFlops |
| Federated Learning | 29.4 TFlops | 5.89 TFlops |
| SplitNN | **0.1548 TFlops** | **0.03 TFlops** |

SplitNN requires ~190× less computation per client than the baselines on 100-client setup.

**Communication Bandwidth (CIFAR-100 / ResNet-50):**

| Method | 100 Clients | 500 Clients |
|---|---|---|
| Large Batch SGD | 13 GB | 14 GB |
| Federated Learning | 3 GB | 2.4 GB |
| SplitNN | 6 GB | **1.2 GB** |

SplitNN beats all methods on bandwidth at 500 clients. At 100 clients, it is worse than FedAvg (6 GB vs 3 GB).

## 6.2 Why the Computation Advantage Is So Large

CNNs like VGG and ResNet have a property: early layers (where the cut is placed) have very few parameters compared to later fully-connected layers. The client running only early layers does minimal floating-point operations. This is the architectural insight the paper exploits.

## 6.3 Accuracy Trends

SplitNN achieves equal or *higher* accuracy than baselines while using a fraction of the compute. The figures show that as client-side flops increase, SplitNN's accuracy curve rises more steeply — meaning it gets more accuracy-per-flop.

## 6.4 Failure / Limitation Cases

- At low client counts (100 clients), communication bandwidth is higher for SplitNN than FedAvg
- No results on heterogeneous or non-IID data
- No formal privacy analysis under gradient-inversion or model-inversion attacks

## 6.5 Unexpected Observations

The paper does not investigate the case where cut-layer activations might leak private information. This is surprising since the stated motivation is privacy for healthcare. Subsequent research (e.g., Pasquini et al., 2021) demonstrated that activations can be reversed-engineered.

## 6.6 Publishability Strength Check

| Finding | Publication Strength |
|---|---|
| 190× compute reduction | Strong — concrete numbers, two architectures, two dataset scales |
| Accuracy at least matches baselines | Moderate — graphs shown but no statistical significance tests |
| Novel configurations for healthcare | Strong — clear practical motivation |
| Privacy claims | Weak — no formal guarantee; only architectural argument |

---

# 7. Strengths – Weaknesses – Assumptions

## 7.1 Technical Strengths

| Strength | Why It Matters |
|---|---|
| Dramatically lower client computation | Makes collaborative AI viable for resource-poor settings |
| Handles vertical data partitioning | Solves a problem federated learning ignores |
| Supports label-free sharing (U-shaped) | Addresses label privacy, a real healthcare concern |
| Modular and extensible configurations | Researchers can design new splits for new use cases |
| Compatible with any CNN architecture | Not limited to specific model families |
| Scales better in bandwidth at large client counts | Important for IoT or edge health deployments |

## 7.2 Explicit Weaknesses

| Weakness | Severity |
|---|---|
| Sequential client training (vanilla) | High — one client at a time creates a bottleneck |
| No formal privacy proof | High — activations can leak raw data |
| No non-IID analysis | High — medical data is inherently heterogeneous |
| Communication cost worse than FedAvg at small client counts | Medium |
| No convergence analysis | Medium — no theoretical guarantee on convergence |
| Results primarily from companion paper, not this paper | Medium — reproducibility concern |
| No Byzantine/poisoning resilience | Medium |

## 7.3 Hidden Assumptions

| Assumption | Problem If Violated |
|---|---|
| Server is trusted to not attack clients | If server is malicious, it can launch model-inversion attacks using activations |
| IID data distribution across clients | Non-IID distributions can cause accuracy collapse |
| Cut layer activations are not reconstructible | Subsequent research has shown this is false |
| Clients honestly participate | No mechanism to detect freeloading or model-poisoning clients |
| Network connectivity is stable | Multi-hop and vertical configs break under client dropout |
| All clients run same base architecture | Heterogeneous architectures (common in real deployments) require additional design |

---

# 8. Weakness → Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| Activations may leak raw data | No privacy mechanism at cut layer | Add formal differential privacy guarantee to cut-layer activations | Gaussian/Laplace noise injection, NoPeek (minimizing distance correlation) |
| Sequential training bottleneck | Vanilla config processes one client at a time | Design parallel split learning with client batching | Round-robin scheduling, pipeline parallelism, async updates |
| No non-IID analysis | Paper assumes homogeneous data | Evaluate SplitNN under label shift, covariate shift, feature shift | Federated benchmarks + non-IID generators |
| No convergence guarantee | Paper is empirical only | Prove convergence of SplitNN under standard assumptions | Optimization theory; similar to FedProx or SCAFFOLD analysis |
| Communication cost high at small client counts | Cut-layer activations larger than gradient diffs at small scale | Compress activations before transmission | Quantization, sparsification, autoencoders at cut layer |
| No Byzantine client defense | No aggregation or verification step | Integrate Byzantine-robust aggregation at the cut layer | Krum, Bulyan, or anomaly detection on activation space |
| No label privacy in vanilla config | Labels reside at server | Design a configuration where labels never leave any party | Secure Multi-Party Computation (SMPC) for label-blind loss computation |
| No adaptive cut-layer selection | Fixed cut point | Learn optimal cut layer during training | Neural Architecture Search (NAS) for split-point selection |

---

# 9. Novel Contribution Extraction

## 9.1 Explicit Novel Claim Templates

The following are ready-to-use templates for framing new research inspired by this paper:

1. "We propose **PrivSplit**, a differentially private extension of SplitNN that injects calibrated noise at the cut layer, providing $(\epsilon, \delta)$-DP guarantees while preserving model accuracy within X% of the baseline."

2. "We propose **AsyncSplit**, a parallel split learning framework that allows N clients to train simultaneously using a pipeline scheduling protocol, reducing total wall-clock training time by X× compared to vanilla SplitNN."

3. "We propose **AdaptiveSplit**, a method that dynamically selects the cut layer during training based on measured communication cost and privacy leakage metrics, outperforming fixed cut-layer strategies in heterogeneous network conditions."

4. "We propose **VertFedMix**, a split learning configuration for non-IID vertically partitioned data that introduces a cross-client attention mechanism at the concatenation point, improving convergence by X% on federated medical benchmarks."

5. "We propose **SplitGuard**, a Byzantine-robust split learning protocol that detects and filters adversarial cut-layer activations using statistical outlier detection, maintaining accuracy under X% of malicious clients."

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Work

- Generate novel SplitNN configurations beyond those proposed
- Combine SplitNN with neural network compression (quantization, pruning) for ultra-low bandwidth edge deployment
- Scale to even larger client numbers

## 10.2 Missing Directions (Not Mentioned by Authors)

- **Privacy attacks on activations:** Model-inversion and gradient-inversion attacks on cut-layer representations
- **Non-IID robustness:** Medical data is almost always non-IID across hospitals
- **Personalized split learning:** Different cut points or sub-networks per client
- **Fairness:** Ensuring all institutions benefit equally, not just large data holders
- **Vertical + horizontal hybrid:** Some clients share features; others share samples
- **Asynchronous training:** Handling clients that drop in and out
- **Semi-supervised extensions:** Leveraging unlabeled data in the split framework

## 10.3 Modern Extensions (Post-2018 Research Directions)

- **NoPeek (2020):** Adding distance correlation minimization to make activations less reconstructable
- **Split Federated Learning (SplitFed):** Combining FedAvg model aggregation with split architecture
- **Simulation-to-real transfer:** Using split learning for synthetic-to-real medical AI without data sharing
- **Foundation model fine-tuning under split learning:** Clients adapt a large pre-trained model via split learning, with only the adapter weights at the client

## 10.4 LLM / AI Era Extensions

- **Split fine-tuning of LLMs:** Hospital-specific prompt tuning or adapter fine-tuning without exposing pre-training data
- **Multimodal split learning:** Text (EHR) + image (radiology) fusion via separate client branches
- **SplitNN + RAG:** Retrieval-augmented generation where the retrieval index stays private at each institution

---

# 11. How to Write a New Paper From This Work

## 11.1 Reusable Elements

| Element | How to Reuse |
|---|---|
| Split architecture concept | Apply to new domains: finance, manufacturing, IoT sensor networks |
| Configuration design pattern | Design new split topologies for your specific collaboration structure |
| Compute / bandwidth comparison table | Use table format to justify your approach vs. baselines |
| Client-side TFlops metric | Standard metric for resource-limited scenario papers |
| Health data motivation | Highly accepted framing in medical AI — cite HIPAA / GDPR |

## 11.2 What Must NOT Be Copied

- The exact configurations described (vanilla, U-shaped, vertically partitioned) — already published
- The CIFAR/VGG/ResNet experimental setup without modification — too similar to existing work
- The "snail-pace" communication framing without new data
- The comparison to FedAvg and SGD only — reviewers expect broader baselines now (SCAFFOLD, FedProx, MOON, etc.)

## 11.3 How to Design a Novel Extension

**Step 1:** Choose ONE identified weakness (from Section 8)
**Step 2:** Define a specific healthcare scenario that this weakness makes worse
**Step 3:** Propose a method that addresses the weakness
**Step 4:** Run experiments on at least 2 medical or federated benchmarks (FedIXI, FedISIC, Chest X-ray, FLAIR NLP, etc.)
**Step 5:** Compare against: vanilla SplitNN, FedAvg (FedProx), and ideally one more recent baseline
**Step 6:** Add a privacy analysis section (even informal) — this is now expected

## 11.4 Minimum Publishable Contribution Checklist

- [ ] New mechanism / configuration not published before
- [ ] Tested on at least 2 datasets (ideally with real or realistic medical data)
- [ ] Compared to at least 3 baselines including vanilla SplitNN and FedAvg/FedProx
- [ ] Resource efficiency analysis (compute + communication)
- [ ] At least informal privacy analysis (ideally formal DP proof)
- [ ] Analysis under non-IID data conditions
- [ ] Ablation study (e.g., varying cut layer, noise level, number of clients)
- [ ] Statistical significance test (or confidence intervals) on accuracy results

---

# 12. Complete Paper Writing Template

## Abstract
**Purpose:** Compress the entire contribution into 150–250 words.

**What to include:**
- Problem: "Privacy-preserving distributed learning in [domain]"
- Gap: "Existing methods do not handle [specific weakness identified]"
- Proposed method: 1–2 sentences on your approach
- Key result: One concrete number (e.g., "X% accuracy improvement with Y× less computation")
- Significance: Why the community should care

**Common mistakes:** Describing what you did, not what you achieved. Being vague about the gap.

**Reviewer expectation:** Clear problem, clear solution, clear result. If any of the three is missing, rejection is likely.

---

## 1. Introduction
**Purpose:** Motivate the problem, establish context, state contributions.

**What to include:**
- Real-world consequence of the problem (privacy barrier, resource barrier)
- Brief overview of existing approaches and their limits (FedAvg, SplitNN baseline)
- Your paper's specific gap in 2–3 sentences
- Bullet-point list of contributions (exactly 3–5 bullets)
- Paper organization paragraph

**Common mistakes:** Too much related work in introduction. Overly broad framing. Not being specific about what is NEW in this paper.

**Reviewer expectation:** By end of introduction, reviewer understands exactly what problem is solved and how your solution is different from existing work.

---

## 2. Related Work
**Purpose:** Show you know the field; position your work clearly.

**Subsections to consider:**
- Distributed / Federated Learning (McMahan 2017, Li 2019 FedProx, Reddi 2021 FedOpt)
- Split Learning (Gupta & Raskar 2018, Vepakomma 2018, NoPeek 2020)
- Privacy-Preserving ML (DP-SGD Abadi 2016, Secure Aggregation Bonawitz 2017)
- Your specific domain (if medical: privacy in health AI)

**Common mistakes:** Listing papers without comparing them to your work. Missing key recent papers (reviewers will notice).

**Reviewer expectation:** Every paragraph in related work ends with "our work differs because..."

---

## 3. Problem Formulation
**Purpose:** Define the problem mathematically and clearly.

**What to include:**
- System model: number of clients, server, data partitioning type
- Privacy threat model (who is honest? who is curious? who is malicious?)
- Formal objective (what are you minimizing or optimizing?)
- Constraints (communication budget, compute budget, privacy budget $\epsilon$)

**Common mistakes:** Skipping formal definitions and going straight to method. Ambiguous threat model.

---

## 4. Proposed Method
**Purpose:** Explain your solution clearly enough to be reproduced.

**What to include:**
- System overview diagram (required)
- Step-by-step algorithm (pseudocode)
- Justification for each design choice
- How it handles the problem from Section 3

**Common mistakes:** Vague descriptions. Missing pseudocode. Not connecting method to the gap identified in introduction.

**Reviewer expectation:** Pseudocode or detailed algorithm. Diagram. Each design decision justified.

---

## 5. Theoretical Analysis (if applicable)
**Purpose:** Provide guarantees — convergence, privacy, or complexity.

**What to include:**
- Theorem statement (plain language first, then formal)
- Assumptions (list explicitly)
- Proof sketch (not full proof in main text — move full proof to appendix)
- What the theorem implies practically

**Common mistakes:** Stating theorems without interpreting what they mean practically.

---

## 6. Experiments
**Purpose:** Prove that your method works in practice.

**Required subsections:**
- Experimental setup (datasets, baselines, metrics, hardware)
- Main comparison table (accuracy + resource metrics)
- Ablation study (test individual components)
- Analysis under non-IID / challenging conditions
- Scalability analysis (varying number of clients)

**Common mistakes:** Only showing accuracy. No ablation. Only IID experiments. No error bars.

**Reviewer expectation:** At least one table with main results. Ablation. Non-IID test. Baseline comparison with at least 3 competitive methods.

---

## 7. Discussion
**Purpose:** Interpret results; explain surprising findings.

**What to include:**
- Why your method works (tie back to method design)
- When it performs worse (honest analysis)
- Comparison to intuitions from the problem formulation

---

## 8. Limitations
**Purpose:** Show intellectual honesty; propose future work.

**What to include:**
- Privacy limitations (especially if using activations without DP)
- Scalability limits
- Dataset limitations (if CIFAR was used — acknowledge it is not medical data)
- Threat model boundaries

**Common mistakes:** Missing this section entirely (red flag to reviewers), or making it too short.

---

## 9. Conclusion
**Purpose:** Close the paper; do not introduce new ideas.

**What to include:**
- One-paragraph summary of the problem
- One-paragraph summary of what was proposed and key results
- One sentence on broader impact or future work

---

## References
- Use consistent citation style (ACM, IEEE, or NeurIPS format)
- Include arxiv papers with full authors + year
- Do not include informal web sources

---

# 13. Publication Strategy Guide

## 13.1 Suitable Venues

| Venue Type | Examples | Fit |
|---|---|---|
| Federated Learning / Privacy workshops | NeurIPS FL Workshop, ICLR Workshop on Privacy ML | High — rapid feedback on split learning extensions |
| Medical AI conferences | MICCAI, MIDL, CHIL | High — if evaluated on real medical data |
| Systems/ML conferences | MLSys, SysML | High — if contribution is systems-level |
| Top ML conferences | ICML, NeurIPS, ICLR | Requires strong theoretical contribution + broad results |
| Applied ML journals | Nature Machine Intelligence, TPAMI, JAMIA | High if medical application is strong |

## 13.2 Required Baseline Expectations (2024+)

Reviewers will expect comparison beyond vanilla SplitNN and FedAvg:
- FedProx (Li et al., 2019)
- SCAFFOLD (Karimireddy et al., 2020)
- If privacy-focused: DP-FedAvg, NoPeek
- If communication-focused: SplitFed, QSGD or similar compression

## 13.3 Experimental Rigor Level Required

| Claim Type | Minimum Experimental Support |
|---|---|
| Resource efficiency | Benchmarked on ≥2 architectures, ≥2 client scales |
| Privacy protection | Formal DP proof OR empirical attack-defense evaluation |
| Accuracy parity | Confidence intervals or std deviation across ≥3 runs |
| Scalability | Tested on ≥3 different client counts |

## 13.4 Common Rejection Reasons

- "The only contribution is a new configuration of SplitNN — not sufficient novelty"
  → Include formal analysis (convergence or privacy) to elevate novelty
- "Experiments use CIFAR — not relevant to healthcare"
  → Include at least one medical/realistic federated dataset
- "No privacy formal guarantee while claiming privacy"
  → Add formal DP section or explicitly scope privacy claim
- "Baselines are outdated"
  → Add SCAFFOLD / FedProx / recent split-learning methods
- "No non-IID experiments"
  → Always test with Dirichlet non-IID distribution

## 13.5 Increment Needed for Acceptance

| Current Paper (Gupta 2018) | What You Must Add for Acceptance |
|---|---|
| New configurations | + Formal convergence or privacy proof |
| CIFAR experiments | + Medical/federated benchmark |
| 3 baselines | + 5–6 baselines including recent ones |
| Accuracy + resource metrics | + Statistical significance + ablation |
| Architectural argument for privacy | + Privacy attack evaluation or DP guarantee |

---

# 14. Researcher Quick Reference Tables

## 14.1 Key Terminology Table

| Term | Simple Meaning |
|---|---|
| SplitNN / Split Learning | Neural network cut in half — part runs on client, part on server |
| Cut Layer | The layer where the split happens; only its outputs are shared |
| Smashed Data | Activations at the cut layer sent from client to server |
| Vertical Partitioning | Different institutions have different features (columns) of same patients |
| Horizontal Partitioning | Different institutions have different patients (rows) |
| U-Shaped Configuration | Network wraps back from server to client; client keeps its labels |
| TFlops | Tera-floating-point-operations; measure of computation cost |
| FedAvg | Federated Averaging — the standard federated learning baseline |
| HIPAA | U.S. health data privacy law; motivates the entire paper |
| Non-IID | Data is not identically distributed across clients (realistic in healthcare) |

## 14.2 Important Equations Summary

| Equation / Idea | Meaning |
|---|---|
| $\mathbf{h}_k = f_k \circ \cdots \circ f_1(\mathbf{x})$ | Client computes activations up to cut layer $k$ |
| $\hat{y} = f_n \circ \cdots \circ f_{k+1}(\mathbf{h}_k)$ | Server completes forward pass from cut layer |
| $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_k}$ | Only value sent back from server to client during backpropagation |
| Client TFlops = (flops for layers 1 to k) | Drastically smaller than full network flops for CNN early layers |

## 14.3 Parameter Meaning Table

| Parameter | Meaning | Typical Value |
|---|---|---|
| Cut layer position $k$ | Which layer divides client and server model | Early layer in network (e.g., after conv block 1) |
| Number of clients | How many participating institutions | 100 or 500 in paper |
| Batch size | Samples per training iteration | Standard (not specified in this paper) |
| Dataset used | Benchmark for evaluation | CIFAR-10 (100 clients), CIFAR-100 (500 clients) |
| Architecture | Neural network model | VGG (CIFAR-10), ResNet-50 (CIFAR-100) |

## 14.4 Algorithm Flow Summary

| Step | Location | Action | Data Transferred |
|---|---|---|---|
| 1 | Client | Forward pass (input → cut layer) | None |
| 2 | Client → Server | Send cut-layer activations | $\mathbf{h}_k$ |
| 3 | Server | Forward pass (cut layer+1 → output) | None |
| 4 | Server | Compute loss; backward pass to cut layer | None |
| 5 | Server → Client | Send cut-layer gradient | $\nabla_{\mathbf{h}_k} \mathcal{L}$ |
| 6 | Client | Backward pass (cut layer → input) | None |
| 7 | Both | Update weights independently | None |

## 14.5 Configuration Quick Reference

| Config | Label Sharing? | Vertical Data? | Multi-Task? | Multi-Hop? |
|---|---|---|---|---|
| Vanilla | Yes (to server) | No | No | No |
| U-Shaped | No | No | No | No |
| Vertically Partitioned | Yes (to server) | Yes | No | No |
| Extended Vanilla | Yes (to server) | Yes | No | No |
| Multi-Task | Yes (to server) | Yes | Yes | No |
| Tor-Like Multi-Hop | Sequential | No | No | Yes |

---

# 15. One-Page Master Summary Card

| Category | Details |
|---|---|
| **Problem** | Healthcare institutions need to jointly train AI models but cannot share raw patient data due to HIPAA, trust, and bandwidth constraints |
| **Core Idea** | Split a neural network at a "cut layer." Client runs only early layers. Server runs the rest. Only intermediate activations and their gradients cross the network — not raw data. |
| **Proposed Method** | SplitNN with 6 configurations: vanilla, U-shaped (no label sharing), vertically partitioned data, extended vanilla, multi-task, and Tor-like multi-hop |
| **Key Result** | SplitNN uses ~0.1548 TFlops per client vs. 29.4 TFlops for FedAvg on CIFAR-10/VGG with 100 clients — roughly 190× less computation |
| **Communication Trade-off** | SplitNN needs more bandwidth than FedAvg at small client counts (100 clients: 6 GB vs. 3 GB), but outperforms at large counts (500 clients: 1.2 GB vs. 2.4 GB) |
| **Core Weakness** | No formal privacy guarantee; cut-layer activations can leak raw data; no analysis on non-IID data; sequential client training creates a bottleneck |
| **Research Opportunity #1** | Add differential privacy to cut-layer activations with formal $(\epsilon, \delta)$-DP guarantee |
| **Research Opportunity #2** | Design parallel/asynchronous SplitNN to eliminate sequential client bottleneck |
| **Research Opportunity #3** | Prove convergence of SplitNN under non-IID data distributions |
| **Publishable Extension** | PrivSplit (differentially private SplitNN), AsyncSplit (parallel SplitNN), or AdaptiveSplit (dynamic cut-layer selection) with proper theoretical + empirical validation |
| **Best Venue** | MICCAI, NeurIPS FL Workshop, MLSys, or federated learning privacy workshops |
