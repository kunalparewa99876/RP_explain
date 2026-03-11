# Research Companion: Survey of Personalization Techniques for Federated Learning
**Kulkarni, Kulkarni & Pant — 2020**
**Vishwakarma University & DeepTek Inc.**

---

> **Paper Classification: Survey / Conceptual**
> This paper surveys, organizes, and critically evaluates existing personalization strategies for federated learning. Explanations emphasize argument structure, reasoning flow, taxonomy logic, and conceptual gaps.

---

# 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Paper Title** | Survey of Personalization Techniques for Federated Learning |
| **Authors** | Viraj Kulkarni, Milind Kulkarni, Aniruddha Pant |
| **Year** | 2020 |
| **Problem Domain** | Federated Learning — Client-level Model Personalization |
| **Paper Type** | Survey / Conceptual |
| **Core Contribution** | First focused survey of personalization techniques for federated learning, organized into 7 distinct strategy categories |
| **Key Idea** | A single shared global model in federated learning fails heterogeneous clients; each client needs a model adapted to its own data distribution |
| **Required Background** | Federated Averaging (FedAvg), basic neural networks, non-IID data concepts, transfer learning, meta-learning basics |
| **Primary Baseline** | FedAvg — McMahan et al. (2016) |
| **Main Innovation Type** | Taxonomy + Critical Analysis (Survey) |
| **Difficulty Level** | Intermediate |
| **Reproducibility Level** | Low (survey paper — no new experiments; references existing works) |

---

# 1. Research Context & Core Problem

## 1.1 Exact Problem Formulation

- Standard federated learning trains ONE shared model across all clients.
- Clients have data that is **non-IID** (not identically distributed) — each client's data reflects its unique local environment.
- The global model is an average, so it is not optimized for any single client.
- For clients with enough good local data, their **own local model beats the global model**.
- These clients then have **no incentive to participate** in federated learning at all.

**Core Question the Paper Asks:**
> How can we adapt the globally trained federated model so that each individual client gets better predictions from it than they would from a purely local model?

## 1.2 Why the Problem Exists

- Federated learning was originally designed to improve privacy and reduce communication — personalization was not the primary concern.
- Real-world data across devices is naturally non-IID:
  - A user in one city has different typing patterns from a user in another city.
  - A hospital in one region sees different disease distributions than another.
- Model averaging (FedAvg) naturally dilutes client-specific signal.
- Privacy protection mechanisms (like differential privacy) make the global model even less sharp for minority or underrepresented clients.

## 1.3 Historical and Theoretical Gap

- Early federated learning papers (McMahan 2016, Kairouz 2021) measured only **global accuracy** — average performance across all clients.
- Global accuracy is misleading: a model can have excellent global accuracy while performing poorly for specific clients.
- There was no systematic study of techniques specifically designed to fix this *per-client performance gap*.
- This survey fills that gap by collecting, categorizing, and comparing all known personalization approaches as of 2020.

## 1.4 Limitations of Previous Approaches

| Approach | Limitation |
|---|---|
| Standard FedAvg | Single global model cannot serve heterogeneous clients well |
| Pure local training | Clients with small datasets get overfitted or biased models |
| Differential Privacy on FL | Reduces accuracy proportionally more for minority/underrepresented clients |
| Global accuracy metrics | Mask per-client performance failures |

## 1.5 Contribution Category

- **Survey / Taxonomy**: Organizes existing work into a clear framework
- **Empirical Insight**: Points out that personalized models mostly outperform both global and local-only models
- **System Design Commentary**: Identifies practical constraints (communication, straggler clients, privacy conflicts)
- **Open Problem Identification**: Names unresolved questions for future research

---

### Why This Paper Matters

This paper is important because it introduces a **shared vocabulary and taxonomy for personalization research** in federated learning. Before this survey, personalization papers used inconsistent terminology and compared against different baselines. By organizing methods into 7 categories with clear logic, this paper became a reference point for anyone designing or extending federated personalization systems. It also explicitly names the open problems — making it a direct source of research ideas.

---

### Remaining Open Problems (from paper + analysis)

- No formal theoretical framework for when a global model will outperform a local model
- No standard benchmark datasets or evaluation protocol for personalization
- User context featurization without privacy violation — still unsolved
- Personalization under differential privacy is still accuracy-limited
- Most methods assume full client participation — straggler problem not solved
- Cross-device vs. cross-silo personalization differences not formalized
- Communication efficiency of personalization steps is largely ignored
- Fairness across clients (not just average performance) not addressed

---

# 2. Minimum Background Concepts

## 2.1 Federated Learning (FL)

- **Plain definition**: A method where multiple devices collaboratively train a model without sharing their raw data. Each device keeps data locally.
- **Role in paper**: This is the foundational setting within which all personalization techniques operate.
- **Why authors needed it**: All personalization techniques are modifications or extensions of the standard FL process.

**Standard FL Process (FedAvg):**
```
Round starts → Server sends global model to clients
→ Each client trains locally on private data
→ Clients send updated model weights back to server
→ Server averages all updates → New global model
→ Repeat
```

## 2.2 Non-IID Data Distribution

- **Plain definition**: Data is non-IID when the data distribution is different across clients. Client A's data does not look like Client B's data.
- **Role in paper**: The root cause of the personalization problem. Non-IID data breaks the assumption that a single global model will work equally for all.
- **Why authors needed it**: Every personalization technique is essentially a response to non-IID data.

**Simple Example:**
- Client 1 (user in India): keyboard prediction data includes Hindi-English mixed words.
- Client 2 (user in France): keyboard prediction data is mostly French.
- A single global model will be mediocre for both — neither perfectly French nor perfectly Hindi-English.

## 2.3 Statistical Heterogeneity

- **Plain definition**: The variation in data quantity, quality, and distribution across participating clients.
- **Role in paper**: Wu et al.'s three-challenge framework (device, data, and model heterogeneity) sets the stage for why personalization is needed.
- **Why authors needed it**: To formally justify that a single global model is structurally insufficient.

## 2.4 Transfer Learning (Fine-tuning)

- **Plain definition**: Using a model trained on one task/dataset as a starting point, then retraining it on a specific task/dataset.
- **Role in paper**: The simplest personalization strategy — train a global model, then fine-tune it on each client's private data.
- **Why authors needed it**: Transfer learning is the most natural bridge between a global model and a personalized one.

## 2.5 Meta-Learning (MAML)

- **Plain definition**: A training approach that doesn't just train a model to perform a task — it trains a model to *quickly learn* new tasks with few examples.
- **Role in paper**: Meta-learning (especially MAML and Reptile) provides a principled theoretical framework where FL training becomes the "meta-training" phase and local fine-tuning becomes "meta-testing."
- **Why authors needed it**: It explains why FedAvg is actually implicitly doing meta-learning.

## 2.6 Knowledge Distillation

- **Plain definition**: Transferring knowledge from a large "teacher" model to a smaller "student" model. The student learns to mimic the teacher's outputs.
- **Role in paper**: Proposed as a way to prevent overfitting during personalization — the global model acts as the teacher / regularizer.
- **Why authors needed it**: Small local datasets cause personalized models to overfit; the global model as teacher provides regularization.

## 2.7 Multi-task Learning

- **Plain definition**: Training a single model to handle multiple related tasks simultaneously, sharing learned representations across tasks.
- **Role in paper**: Each client's learning problem is treated as a separate but related "task." One model per client is learned jointly.
- **Why authors needed it**: It allows sharing of knowledge between clients while keeping client-specific outputs.

## 2.8 Differential Privacy in FL

- **Plain definition**: A mathematical guarantee that adding or removing any single person's data does not significantly change the model's output.
- **Role in paper**: Identified as conflicting with personalization — privacy mechanisms reduce model accuracy, especially for minority clients.
- **Why authors needed it**: To explain why personalization becomes even more important when privacy constraints are applied.

---

# 3. Theoretical / Conceptual Understanding Layer

*Note: This survey paper does not introduce new mathematics. It references theoretical frameworks from other papers. The following section explains the key theoretical ideas referenced.*

## 3.1 Standard FL Objective

**Intuition:** FedAvg minimizes a weighted average of all clients' losses simultaneously. This means every client pulls the model in its own direction, and the result is a compromise — no single client's loss is fully minimized.

**What the formulation captures:**

| Symbol | Meaning |
|---|---|
| $F(w)$ | Global objective — weighted average of all client losses |
| $F_k(w)$ | Loss function for client $k$ |
| $n_k$ | Number of data samples at client $k$ |
| $n$ | Total samples across all clients |
| $w$ | Model parameters (weights) |

$$F(w) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(w)$$

**Limitation of this formulation:**
- It finds $w$ that is best "on average" across clients.
- A client with unique data distribution (minority) is outvoted by the majority.
- Personalization replaces or augments this objective with per-client objectives.

## 3.2 Personalization Objective (Hanzely et al. — Mixture Model)

**Intuition:** Instead of finding one $w$ for all clients, allow each client to find its own model $w_k$ that is a blended version of the global model and the client's own local model.

**Formulation:**

$$\min_{w, h_1, \ldots, h_K} \sum_{k=1}^{K} F_k(\alpha w + (1-\alpha) h_k)$$

| Symbol | Meaning |
|---|---|
| $w$ | Shared global model parameters |
| $h_k$ | Client $k$'s personal model parameters |
| $\alpha$ | Mixing parameter — controls how much global vs. local knowledge is used |

**Practical Interpretation:**
- When $\alpha = 1$: pure global model (standard FedAvg)
- When $\alpha = 0$: pure local model (no federation benefit)
- Optimal $\alpha$ is between 0 and 1 — best of both worlds

**Limitation:** Finding the right $\alpha$ requires tuning and may vary per client.

## 3.3 Meta-Learning Perspective (Jiang et al. / Fallah et al.)

**Intuition:** FedAvg is not just minimizing global loss — it is finding an initialization point that can be quickly fine-tuned for any client.

**MAML-inspired FL objective (Per-FedAvg):**

$$\min_w \sum_{k=1}^{K} F_k(w - \alpha \nabla F_k(w))$$

| Symbol | Meaning |
|---|---|
| $w$ | Initial global model (the "meta-initialization") |
| $\alpha$ | Learning rate for one personalization step |
| $\nabla F_k(w)$ | Gradient of client $k$'s loss at the current global model |

**What this says:**
- The global model $w$ is optimized not to minimize average loss directly, but to minimize loss *after each client takes one gradient step on their own data*.
- Small change from FedAvg in formulation, but large change in behavior — the global model learns to be personalization-friendly.

### Mathematical Insight Box

> **Key Researcher Insight:** The global model in federated learning should be understood as a meta-initialization point, not a final model. The quality of personalization depends heavily on how well the global model was trained — optimizing for raw global accuracy can accidentally make it harder to personalize. Design the global training objective to encourage fast local adaptation.

---

# 4. Proposed Framework — Taxonomy of 7 Personalization Techniques

*The paper's main contribution is organizing personalization approaches into 7 categories. Each is explained below with its logic, strengths, weaknesses, and research improvement seeds.*

## 4.1 Adding User Context

### What it does
- Instead of modifying the model, modify the **data**: add personal/contextual features to each client's input.
- If the client's preferences, location, history, or behavior are encoded as features, the same global model can produce personalized outputs.

### Workflow
```
Collect user context (location, preferences, history)
→ Featurize context into numerical features
→ Append context features to each data sample
→ Standard FL training proceeds
→ Global model now learns context-conditional patterns
```

### Why authors included it
- It is the simplest conceptual approach — no change to training, only to data representation.
- Useful as a baseline for showing that data-level personalization has limits.

### Weakness
- Most public research datasets do not contain contextual features — hard to benchmark.
- Privacy risk: context features may reveal more personal information than the raw data.
- No existing method for automatic, privacy-safe context featurization.

### Research Idea Seed
- Develop a **privacy-preserving context encoder** that learns to featurize client context using federated techniques without exposing raw context data.

---

## 4.2 Transfer Learning (Fine-tuning)

### What it does
- First, train a global model using standard FL.
- Then, each client fine-tunes the global model on its own local data.
- The global model provides a strong starting point; local fine-tuning adds specificity.

### Workflow
```
Phase 1 — FL Training:
  Server initializes model
  → FL rounds until global model converges
  → Global model captures shared knowledge

Phase 2 — Local Fine-tuning:
  Each client receives global model
  → Client trains on own data for T_local steps
  → Personalized model stored locally
```

### Design Variants
- **Full fine-tuning**: Retrain all layers on local data. Risk: catastrophic forgetting of global knowledge.
- **Top-layer fine-tuning**: Freeze lower layers (feature extractors), retrain only the final classification layers. Safer.

### Why authors included it
- Most widely used in practice. Simple to implement. Integrates naturally into FL workflow.

### Weakness
- If local fine-tuning runs too long → catastrophic forgetting (overwrites global knowledge).
- Works poorly when local datasets are very small.
- No clear rule for how many local fine-tuning steps to perform.

### Research Idea Seed
- Design an **adaptive fine-tuning stopper** that monitors when personalization starts to hurt global knowledge and stops training automatically.

---

## 4.3 Multi-task Learning (MOCHA)

### What it does
- Treats each client's problem as a separate but related "task."
- All tasks are learned jointly — shared representations are learned while task-specific outputs differ.
- Smith et al.'s MOCHA algorithm extends multi-task learning to the federated setting.

### Workflow
```
Each client k has task T_k with its own loss function F_k(w_k)
Shared representation: w_shared (trained jointly)
Task-specific layers: w_k (client-specific, trained locally)

Joint optimization:
  min Σ_k F_k(w_shared, w_k) + regularizer
```

### Why authors included it
- Theoretical foundation (Smith et al., NeurIPS 2017) is rigorous with communication and straggler analysis.
- Natural fit: clients with related data distributions share useful structure.

### Weakness
- **All clients must participate in every round** — violates partial participation assumption of real FL.
- Does not scale well to very large numbers of clients.
- Communication overhead is higher than FedAvg.

### Research Idea Seed
- Develop a **partial-participation multi-task FL** method where the shared representation is updated even when not all clients are present.

---

## 4.4 Meta-Learning (MAML / Per-FedAvg / ARUBA)

### What it does
- Meta-learning teaches the global model to be "fast-adaptable" — not just accurate, but easy to fine-tune.
- Key algorithms referenced: MAML (Finn et al.), Reptile (Nichol et al.), Per-FedAvg (Fallah et al.), ARUBA (Khodak et al.)
- Jiang et al. show FedAvg ≈ Reptile — federated training IS a form of meta-learning.

### Workflow (Per-FedAvg)
```
For each FL round:
  1. Server broadcasts global model w
  2. Each client k:
     a. Takes one gradient step on own data: w_k' = w - α ∇F_k(w)
     b. Evaluates loss after adaptation: F_k(w_k')
     c. Sends gradient of F_k(w_k') back to server
  3. Server updates global model to minimize sum of adapted losses
```

### Key Insight
- FedAvg optimizes: "what model minimizes average loss right now?"
- Per-FedAvg optimizes: "what model, after one client gradient step, minimizes average loss?"
- The second objective produces models that are dramatically easier to personalize.

### Why authors included it
- Most theoretically principled approach to personalization.
- Strongest connection to existing ML theory (MAML, meta-learning literature).

### Weakness
- Requires computing second-order gradients (computationally expensive).
- First-order approximations (like Reptile) are cheaper but less precise.
- Assumes a uniform number of personalization steps for all clients — not adaptive.

### Research Idea Seed
- Design a **client-adaptive meta-learning rate** where clients with harder distributions get more personalization gradient steps, determined automatically.

---

## 4.5 Knowledge Distillation (FedMD)

### What it does
- The global model acts as a "teacher." The personalized local model acts as a "student."
- Student is trained to:
  1. Match its own local data's true labels
  2. Not deviate too far from the teacher's outputs (regularization)
- Li and Wang's FedMD allows each client to use a **different neural architecture** — only knowledge (output distributions) is shared, not weights.

### Workflow
```
Global FL Training → Large global model (teacher)

Local Personalization:
  For each client:
    Input: local data sample x
    Teacher output: soft_labels = global_model(x)
    Student loss = cross_entropy(student(x), true_label)
                 + λ * KL_divergence(student(x), soft_labels)
    Train student on combined loss
```

### Why authors included it
- Solves the overfitting problem for clients with small local datasets — teacher provides a regularizer.
- Enables model **heterogeneity**: clients can have different model sizes.
- FedMD is particularly useful in cross-organizational FL where participants want to keep architectures private.

### Weakness
- Requires a shared public dataset for FedMD's cross-architecture protocol.
- Teacher's soft labels may contain biases from majority clients.
- λ (distillation weight) is a critical hyperparameter — hard to set without validation data.

### Research Idea Seed
- Create a **federated adaptive distillation** where λ is determined per-client based on local dataset size and confidence calibration of the global model.

---

## 4.6 Base + Personalization Layers (FedPer)

### What it does
- Neural network is split into two parts:
  - **Base layers** (lower layers): feature extractors — trained globally via FL
  - **Personalization layers** (top layers): task-specific heads — trained locally only
- This is a architectural personalization approach.

### Workflow
```
Global Training (FL):
  Server aggregates only base layer weights
  Base layers learn shared feature representations
  Top "personalization" layers are excluded from aggregation

Local Training:
  Each client independently trains its own personalization layers
  Base layers remain fixed during local training
```

### Comparison to Transfer Learning
| Aspect | Transfer Learning | FedPer |
|---|---|---|
| Base layer training | Global first, then local | Global only (never local) |
| Top layer training | Local (after global) | Local (only) |
| Risk of forgetting global knowledge | High | Eliminated |
| Personalization depth | Shallow (last N layers) | Structurally enforced |

### Why authors included it
- Clean architectural separation prevents catastrophic forgetting entirely.
- Training signal for each client only touches the right layers.

### Weakness
- Requires choosing WHERE to split the network (base vs. personalization boundary) — no clear rule.
- Different tasks may require different split points.
- Does not adapt the split based on data similarity across clients.

### Research Idea Seed
- Design a **learnable split point** where the boundary between base and personalization layers is determined automatically during FL training based on data heterogeneity signals.

---

## 4.7 Mixture of Global and Local Models (LLGD)

### What it does
- Instead of choosing either the global model or the local model, each client maintains a **weighted mixture** of both.
- Hanzely and Richtárik propose Loopless Local Gradient Descent (LLGD) as the optimizer for this mixed objective.
- The mixing parameter α controls how much global vs. local knowledge each client uses.

### Workflow
```
Each client k maintains:
  - w: global model (shared, updated via FL)
  - h_k: local model (private, updated locally)

Client k's effective model:
  θ_k = α * w + (1-α) * h_k

Training:
  w updated by averaging across clients (but partially — LLGD takes partial steps)
  h_k updated by local gradient on client k's own data
```

### Key Insight about LLGD
- Standard FedAvg performs **full averaging** — all client updates are averaged completely.
- LLGD performs **partial steps toward averaging** — less aggressive, more local knowledge preserved.
- This suggests FedAvg may be "over-aggregating" — losing too much local specificity.

### Why authors included it
- Only method that explicitly formulates the global-local tradeoff as a formal optimization problem.
- Most theoretically clean formulation of what personalization should mean.

### Weakness
- Single α for all clients — heterogeneous clients likely need different mixing ratios.
- No method for choosing α without per-client validation data.
- LLGD requires more complex optimization than FedAvg.

### Research Idea Seed
- Develop **client-adaptive mixture coefficients** where each client $\alpha_k$ is learned dynamically based on the client's data distribution distance from the global distribution.

---

## 4.8 User Clustering (bonus technique from Section 3.1)

### What it does
- Group clients with similar data distributions into clusters.
- Train one FL model per cluster — each cluster model is "semi-personalized."
- Mansour et al. provide learning-theoretic guarantees for this approach.

### Weakness
- Requires computing inter-client similarity — violates data privacy principles.
- Number of clusters is a hyperparameter with no clear selection criterion.

---

# 5. Experimental Setup / Evaluation Design

*Note: This is a survey paper. There are NO new experiments conducted. The following analysis evaluates how the referenced experiments in surveyed papers were designed and what constitutes valid personalization evaluation.*

## 5.1 What Good Personalization Evaluation Looks Like

| Metric | What It Measures | Why It Matters |
|---|---|---|
| Per-client test accuracy | How well the personalized model serves each individual client | Core personalization metric |
| Worst-case client accuracy | Performance of the weakest client | Fairness indicator |
| Global model accuracy (before personalization) | Quality of the FL starting point | Validates FL training step |
| Convergence rounds | How many FL rounds until stable personalization | Communication efficiency |
| Fine-tuning steps needed | How quickly personalization adapts | Practicality indicator |

## 5.2 Key Gap: Most FL Papers Use Wrong Metric

The paper identifies a **critical flaw** in how FL personalization is evaluated:
- Most papers report **global average accuracy** (average across all clients).
- This hides the fact that individual clients may perform poorly.
- A paper could claim excellent results while specific minority clients receive useless models.

**Correct Practice:** Always report per-client accuracy distributions — not just averages.

## 5.3 Datasets Used in Referenced Works

| Dataset Type | Used For | Challenge |
|---|---|---|
| Next-word prediction (mobile keyboard) | Real-world non-IID FL | Highly personalized, natural heterogeneity |
| Image classification (CIFAR, MNIST with non-IID splits) | Standard FL benchmark | Non-IID artificially induced |
| Medical / healthcare records | Cross-silo FL | Severe non-IID, small client count |

---

### Experimental Reliability Analysis

**What is trustworthy in the referenced experimental results:**
- Transfer learning / fine-tuning improvements are consistently validated across multiple independent papers
- FedPer results on image classification are reproducible
- MAML-based personalization improvements over FedAvg are verified in multiple works

**What is questionable:**
- Most results use artificially constructed non-IID data (not real user heterogeneity)
- No unified benchmark — each paper tests on different datasets, making comparison impossible
- Statistical significance testing is rarely reported
- Results often depend on hand-tuned hyperparameters (fine-tuning steps, λ, α) that would be unknown in practice

---

# 6. Results & Findings Interpretation

## 6.1 Main Findings of the Survey

- **Personalized models outperform both global models and local models** in most evaluated scenarios — this is the central empirical finding.
- The exception: when differential privacy is applied, personalized models sometimes fail to outperform pure local models for minority clients.
- Meta-learning (MAML-based) approaches show particularly strong personalization quality because the global model is explicitly trained to be fine-tunable.
- Knowledge distillation effectively prevents overfitting during personalization, especially for clients with small local datasets.
- FedPer's architectural split approach avoids catastrophic forgetting where transfer learning fails.

## 6.2 When Personalization Fails

| Scenario | Why Personalization Fails | Implication |
|---|---|---|
| Differential privacy + small client data | Privacy noise + data scarcity compound | Need privacy-compatible personalization |
| Very small local datasets | Fine-tuning overfits to noise | Need stronger regularization (distillation) |
| Rapidly changing local distributions | Personalized model becomes stale | Need online / continual personalization |
| Clients with IID data | Global model already good; no personalization needed | Detect IID clients and skip personalization |

## 6.3 Performance Trend Analysis

| Technique | Best When | Worst When |
|---|---|---|
| Transfer Learning | Large local dataset, model not too complex | Small local data, long fine-tuning forbidden |
| Multi-task Learning | Few clients, all reliable, all participate | Large-scale FL, high drop-out rate |
| Meta-Learning | Clients need quick adaptation, data is scarce | Large second-order gradient computation cost |
| Knowledge Distillation | Small local datasets, model architecture varies | No shared public dataset available (FedMD) |
| FedPer | Clear feature/task layer distinction possible | Ambiguous network architecture |
| Mixture Model | Explicit control over global/local tradeoff needed | Single α cannot fit all clients |

---

### Publishability Strength Check

**Publication-grade findings (strong):**
- The systematic taxonomy of 7 personalization categories — this structure is widely cited and reused
- Identification of the "global accuracy fallacy" — reporting global accuracy instead of per-client accuracy
- Formal meta-learning connection to FedAvg (Jiang et al. observation cited)
- Privacy-personalization conflict analysis

**Findings needing stronger validation:**
- Precise performance comparisons across techniques (no unified experimental comparison)
- Claims about which technique is "best" — highly context-dependent
- Scalability claims — no experiments at > 100 client FL

---

# 7. Strengths – Weaknesses – Assumptions

## Technical Strengths

| Strength | Description | Impact |
|---|---|---|
| Comprehensive taxonomy | 7 categories cover the entire personalization landscape as of 2020 | Provides stable reference framework |
| Problem motivation | Formally identifies why global accuracy is insufficient | Reframes the FL evaluation problem |
| Open problem identification | Explicitly lists unresolved questions | Research agenda-setting |
| Privacy-personalization conflict | Highlights the tension between DP and personalization | Identifies an important research gap |
| Multi-technique coverage | Covers theory, algorithms, architecture, and optimization approaches | Breadth for cross-domain researchers |

## Explicit Weaknesses

| Weakness | Description | Impact on Research |
|---|---|---|
| No new experimental results | Survey only — no unified empirical comparison | Cannot definitively rank techniques |
| No formal comparison protocol | Each method was evaluated in different original papers | Impossible to directly compare methods |
| 2020 snapshot | Misses post-2020 developments (clustered FL, pFL advances, foundation model era) | Outdated for cutting-edge work |
| No personalization metrics standardization | Each paper uses different metrics and datasets | Community lacks shared benchmark |
| Shallow privacy analysis | Privacy-personalization conflict mentioned but not analyzed deeply | Leaves a critical research gap |
| No federated NLP/LLM context | Personalization in language model FL not covered | Misses important practical domain |

## Hidden Assumptions

| Hidden Assumption | Why It May Be Wrong |
|---|---|
| Clients participate repeatedly across rounds | Many real clients participate sporadically |
| Local dataset is fixed and stable | Real data distributions shift over time |
| Clients are honest — no adversarial behavior | Byzantine/malicious clients can corrupt personalization |
| Personalization is always wanted by client | Some clients may prefer global model (e.g. cold-start users) |
| Fine-tuning is done after FL training completes | In practice, FL training never truly "completes" |
| Client has sufficient compute for meta-learning | Mobile/IoT devices may not support second-order gradient computation |

---

# 8. Weakness → Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| No unified benchmark for personalization | Each paper chose its own evaluation setup | Create a standardized personalization benchmark suite | Design non-IID splits across 5+ datasets, measure per-client distribution + fairness |
| Single global-to-local mixing ratio | Hanzely et al. used fixed α for all clients | Per-client adaptive mixing ratio | Learn α_k via client distribution similarity estimation (cosine distance in gradient space) |
| Meta-learning too expensive on edge devices | MAML requires second-order gradients | First-order personalized FL for constrained devices | Reptile-based federated meta-learning with communication compression |
| Personalization degrades with DP | DP noise destroys fine-tuning signal for minority clients | Privacy budget allocation aware of client data size | Allocate higher privacy budget to majority, lower to minority — adaptive DP |
| No dynamic personalization | Models are personalized once and frozen | Continual personalization for shifting distributions | Federated continual learning with elastic weight consolidation per client |
| No personalization fairness metric | Average accuracy conceals inequality | Define per-client fairness-aware personalization objective | Minimax fairness objective — optimize worst-case client accuracy |
| Knowledge distillation needs public dataset | Not always available in sensitive domains | Public-dataset-free federated distillation | Use synthetic data generation (GAN-based) at server for distillation |
| FedPer split point selection is manual | No theory for optimal layer split | Automatic architecture-aware split point selection | Meta-learnable split point updated via validation loss gradient |

---

# 9. Novel Contribution Extraction

## 9.1 Novel Claim Templates

**Template 1:**
> "We propose **Per-FedMix**, a personalized federated learning method that improves **per-client accuracy under data heterogeneity** by **learning client-adaptive global-local mixing coefficients using gradient space similarity**, unlike fixed-ratio mixture models that use a single shared α."

**Template 2:**
> "We propose **FairFedPer**, a federated learning framework that improves **fairness in personalization** by **applying minimax optimization over per-client personalization layers** rather than standard average-accuracy optimization, ensuring worst-case clients receive high-quality personalized models."

**Template 3:**
> "We propose **DP-PersonaFL**, a differentially private personalized federated learning system that improves **personalization accuracy for minority clients under privacy constraints** by **allocating privacy budgets proportional to client data size and representation**, resolving the disparate accuracy impact of uniform differential privacy."

**Template 4:**
> "We propose **ContinualFedPer**, a continually personalized federated learning architecture that improves **model freshness for individual clients with non-stationary data distributions** by **incorporating elastic weight consolidation within federated personalization layers** to prevent forgetting while adapting to distribution shifts."

**Template 5:**
> "We propose **MetaFedLight**, a resource-efficient personalized federated learning algorithm that improves **per-client adaptation quality for compute-constrained devices** by **replacing second-order MAML gradients with a memory-efficient reptile-style federated meta-learner** with adaptive learning rate scheduling."

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Directions

- Formal theoretical conditions under which global models outperform local models
- Privacy-safe user context featurization for personalization
- Evaluation of personalization techniques on real heterogeneous data (not synthetically created non-IID)

## 10.2 Missing Directions (Not Covered in Paper)

- **Clustered Federated Learning**: Train multiple sub-global models for groups of similar clients (middle ground between one global and all-local)
- **Federated Continual Learning**: Handle non-stationary distributions where client data changes over time
- **Cold-Start Personalization**: Handle new clients who have no local data history
- **Hierarchical Personalization**: Multi-level aggregation (device → edge server → cloud server) for better regional personalization
- **Cross-Device to Cross-Silo Personalization Transfer**: Methods that work at both scale extremes

## 10.3 Modern Extensions (Post-2020)

- **Federated Foundation Model Personalization**: Using pre-trained large language models as the global model base, then personalizing via LoRA or adapter layers at each client
- **Federated Parameter-Efficient Fine-Tuning (FedPEFT)**: Only fine-tune a small number of additional parameters per client (adapters, prompt tuning) instead of full model layers
- **Personalized Federated Learning with Graph Neural Networks**: When client relationships can be expressed as a graph, GNN-based message passing improves personalization
- **Federated Personalization with Uncertainty Quantification**: Clients get not just personalized predictions but calibrated confidence scores

## 10.4 LLM-Era Extensions

- **Prompt-based Personalization in FL**: Each client learns a personal prompt prefix that specializes a shared LLM — low communication, high personalization
- **Personalized Federated Retrieval-Augmented Generation (FedRAG)**: Each client maintains a personal retrieval database; the shared LLM uses client-specific retrieved context
- **Federated LoRA**: Each client trains low-rank adaptation matrices that are aggregated globally while keeping client-specific ranks locally
- **Privacy-Preserving Personalized Instruction Tuning**: Fine-tune shared LLMs on private instruction-following data without revealing client instructions

---

# 11. How to Write a NEW Paper From This Work

## Reusable Elements

- **Survey structure**: 7-category taxonomy as a conceptual framework for organizing your own literature review
- **Problem motivation**: Global accuracy vs. per-client accuracy argument — reusable in any personalization paper
- **Privacy-personalization conflict framing**: Sets up a compelling research problem
- **Evaluation logic**: Per-client accuracy, fairness metrics, convergence speed — these metrics are directly adoptable
- **Baseline selection**: FedAvg + local-only + one personalization method from each category as baselines

## What MUST NOT be Copied

- The taxonomy itself (7 categories as organized by this paper) — this is the paper's core IP
- Direct descriptions of any referenced algorithm (MOCHA, FedPer, Per-FedAvg, FedMD) without proper citation
- The exact framing of "three challenges" (device, data, model heterogeneity) — cite Wu et al.

## How to Design a Novel Extension

**Step 1: Pick ONE weakness from Section 8 as your problem.**
Example: "Single mixing ratio α cannot fit all clients" (Mixture model weakness)

**Step 2: Formulate it as a formal problem.**
Example: "Given non-IID client distributions, find per-client mixing coefficients α_k that minimize the worst-case client loss while maintaining communication efficiency."

**Step 3: Propose a method that directly solves the problem.**
Example: "Estimate client distribution similarity via gradient cosine distance; set α_k proportional to similarity with global distribution."

**Step 4: Compare against appropriate baselines.**
Example: FedAvg, FedPer, Mixture model with fixed α, Per-FedAvg.

**Step 5: Report per-client accuracy distribution (not just average).**
This alone will differentiate your paper from most prior work.

## Minimum Publishable Contribution Checklist

- [ ] Identifies a specific gap in existing personalization literature
- [ ] Proposes a method with clear algorithmic description and convergence argument
- [ ] Compares against at minimum 3 baselines (FedAvg + at least 2 personalization methods)
- [ ] Reports per-client accuracy distribution (not global average only)
- [ ] Tests on at least 2 non-IID dataset splits
- [ ] Includes communication cost analysis
- [ ] Addresses at least one of: fairness, privacy, efficiency, scalability
- [ ] Conducts ablation study isolating each component's contribution

---

# 12. Complete Paper Writing Template

## Abstract
**Purpose:** State the problem, gap, method, and result in 150–200 words.

**What to include:**
- First sentence: state the core problem (global FL model vs. heterogeneous clients)
- Second sentence: identify the gap in existing work (what's missing)
- Third sentence: state your proposed method name and key idea
- Fourth sentence: key experimental result (X% improvement over FedAvg on Y metric)
- Fifth sentence: significance statement

**Common mistakes:**
- Claiming too broad a contribution ("solves the personalization problem")
- Missing concrete numbers in results

**Reviewer expectation:** Abstract should be self-contained and concrete. Reviewers reject vague abstracts immediately.

---

## Introduction
**Purpose:** Expand abstract, motivate deeply, clarify contribution, and outline structure.

**What to include:**
- Paragraph 1: Federated learning setup and importance
- Paragraph 2: Non-IID problem and why global model fails specific clients
- Paragraph 3: Existing personalization approaches and their limitations (cite 4–6 papers)
- Paragraph 4: Clearly stated research gap that motivates your work
- Paragraph 5: Your contributions, listed as bullet points (3–4 items)
- Paragraph 6: Paper organization

**Common mistakes:**
- Burying the contribution in paragraph 5 or later
- Not explicitly stating what is new vs. what is existing
- Failing to connect motivation to proposed approach

**Reviewer expectation:** By end of introduction, reviewer must know exactly what you did and why it matters.

---

## Related Work
**Purpose:** Position your paper within the existing literature. Show you know the field.

**What to include:**
- Subsection 2.1: Federated Learning basics (McMahan 2016, Kairouz 2021)
- Subsection 2.2: Personalization in FL — organize by category (use Kulkarni 2020 survey as reference)
- Subsection 2.3: Your closest prior work — explain specifically why it is insufficient
- End each subsection with: "Our work differs from these approaches by..."

**Common mistakes:**
- Listing papers without explaining how they relate to your gap
- Not citing the closest competitor works (reviewers will notice)
- Writing related work before knowing exactly what your contribution is

**Reviewer expectation:** Related work should naturally lead to the gap that your paper fills.

---

## Method
**Purpose:** Explain your proposed approach with enough detail for reproducibility.

**What to include:**
- Subsection: Problem formulation (mathematical objective)
- Subsection: Proposed algorithm (pseudocode)
- Subsection: Key design choices and why alternatives were rejected
- Subsection: Computational complexity analysis (communication rounds, per-round cost)
- Subsection: Convergence argument (even informal one is better than none)

**Common mistakes:**
- Pseudocode without explanation
- No convergence or complexity analysis
- Not explaining why simpler approaches would not work

**Reviewer expectation:** Reproducibility is paramount. Another researcher should be able to re-implement from this section.

---

## Theory (if applicable)
**Purpose:** Provide convergence guarantees or theoretical justification.

**What to include:**
- Theorem: Convergence bound for your algorithm
- Key assumptions listed explicitly
- Comparison to FedAvg convergence bound
- Interpretation: what does the bound mean in practice?

**Common mistakes:**
- Assuming convergence without proof
- Theorems with so many assumptions they are practically useless
- Not connecting theory to experimental results

**Reviewer expectation:** NeurIPS/ICML papers increasingly require convergence proofs. Workshop papers may skip formal proofs.

---

## Experiments
**Purpose:** Empirically validate your claimed improvements.

**What to include:**
- Datasets: at least 2, with non-IID splitting method described
- Baselines: FedAvg + at least 2 personalization baselines + local-only
- Metrics: per-client accuracy distribution, fairness metric, convergence curve
- Ablation study: remove each component of your method and show its impact
- Hyperparameter sensitivity: show results are stable across key hyperparameter ranges

**Common mistakes:**
- Only reporting average accuracy
- No ablation study (reviewers will ask for it)
- Comparing against weak baselines only
- No statistical significance (error bars / variance across runs)

**Reviewer expectation:** Strong baselines, clear ablations, and per-client (not just global) evaluation.

---

## Discussion
**Purpose:** Interpret results beyond numbers; connect back to motivation.

**What to include:**
- Which experimental results confirm your hypothesis?
- Are there cases where your method underperforms? Why?
- What do the results imply for practical deployment?

---

## Limitations
**Purpose:** Show intellectual honesty and prevent reviewer surprises.

**What to include:**
- Scenarios where your method would not work
- Assumptions required for good performance
- Computational requirements that may not be met
- What was NOT tested (and why)

**Common mistakes:**
- Claiming no limitations
- Vague limitations ("our method is computationally expensive")

---

## Conclusion
**Purpose:** Concise summary + future directions.

**What to include:**
- Restate the problem in one sentence
- Restate your contribution in one sentence
- Key result in one sentence
- 2–3 future work directions

---

## References
- Cite all surveyed personalization papers from Kulkarni 2020
- Always cite: McMahan 2016 (FedAvg), Kairouz 2021 (challenges survey), Kulkarni 2020 (personalization survey)
- Use consistent citation format (ACM/IEEE/NeurIPS format per venue)

---

# 13. Publication Strategy Guide

## Suitable Conference/Journal Types

| Venue Type | Examples | Suitability |
|---|---|---|
| Top ML Conferences | NeurIPS, ICML, ICLR | Strong theory + experiments required; very competitive |
| Applied ML / Systems | MLSys, SysML | Systems efficiency angle required |
| FL Workshops | FL-NeurIPS, FL-ICML workshops | Good entry point; less experimental rigor required |
| AI/Data Privacy | CCS, USENIX Security | Privacy-personalization conflict angle |
| IEEE Transactions | TNNLS, TMC, TIFS | Journal format; more room for thorough evaluation |
| ACM Transactions | TIST, TOMM | Applied personalization, mobile/IoT settings |

## Required Baseline Expectations

For **workshops**: FedAvg + one personalization baseline (FedPer or Per-FedAvg)

For **top conferences**: FedAvg + FedPer + Per-FedAvg + MOCHA + local-only + at minimum 2 recent ICLR/NeurIPS personalization papers

## Experimental Rigor Level

| Conference Tier | Expected Evaluation |
|---|---|
| Workshop | 1–2 datasets, per-client accuracy, basic convergence |
| IEEE/ACM Journal | 3+ datasets, statistical significance, fairness analysis, hardware benchmarks |
| NeurIPS/ICML | Strong convergence theory, 4+ datasets, thorough ablation, communication efficiency analysis |

## Common Rejection Reasons

- Only global average accuracy reported — per-client analysis missing
- Weak baselines (only FedAvg compared)
- No ablation study
- Method is a minor modification of existing technique without sufficient theoretical or empirical evidence of significance
- Claims not supported by experiments (e.g., claims fairness improvement but doesn't measure fairness)
- No reproducibility details (code, hyperparameters, datasets not specified)

## Increment Needed for Acceptance

- **Workshop**: A clear problem statement + preliminary positive results + compelling future directions
- **IEEE/ACM journal**: Solid empirical improvement (>2–3% per-client accuracy over best baseline) + one novel algorithm + convergence argument
- **NeurIPS/ICML**: Strong theoretical contribution (proofs) OR large-scale comprehensive empirical study with new benchmark + 5%+ improvement with statistical significance

---

# 14. Researcher Quick Reference Tables

## Key Terminology Table

| Term | Simple Meaning |
|---|---|
| Federated Learning (FL) | Multi-device collaborative training without sharing raw data |
| FedAvg | The standard FL algorithm — average model weights across clients |
| Non-IID | Data distributions differ across clients |
| Statistical Heterogeneity | Variation in data quantity, quality, and distribution across clients |
| Personalization | Adapting the global model for each individual client |
| Transfer Learning / Fine-tuning | Use global model as starting point; train more on local data |
| Meta-Learning | Train model to be quickly adaptable (not just accurate) |
| MAML | Model-Agnostic Meta-Learning — trains a universal initialization point |
| Per-FedAvg | FedAvg modified to optimize for post-adaptation performance |
| Knowledge Distillation | Student model learns from teacher (global) model's soft outputs |
| FedPer | Federated Personalized — base layers global, personalization layers local |
| MOCHA | Multi-task learning algorithm for federated settings |
| LLGD | Loopless Local Gradient Descent — partial averaging for mixture models |
| Catastrophic Forgetting | Fine-tuning destroys previously learned global knowledge |
| Differential Privacy (DP) | Mathematical guarantee of individual data privacy in training |

## Important Equations Summary

| Equation | Name | Key Insight |
|---|---|---|
| $F(w) = \sum_k \frac{n_k}{n} F_k(w)$ | Standard FL Objective | Global model minimizes weighted average loss |
| $\theta_k = \alpha w + (1-\alpha) h_k$ | Mixture Model | Per-client model is blend of global and local |
| $\min_w \sum_k F_k(w - \alpha \nabla F_k(w))$ | Per-FedAvg Objective | Optimize initialization for post-adaptation performance |
| $\mathcal{L}_{student} = \mathcal{L}_{CE} + \lambda \cdot KL(student \| teacher)$ | Distillation Loss | Student trained to match both labels and teacher |

## Parameter Meaning Table

| Parameter | Method | Meaning | Typical Range |
|---|---|---|---|
| $\alpha$ | Mixture Model | Global vs. local model blend ratio | [0, 1] |
| $\alpha$ | Meta-Learning | Personalization learning rate | 0.01 – 0.1 |
| $\lambda$ | Knowledge Distillation | Weight of teacher regularization term | 0.1 – 1.0 |
| $T_{local}$ | Transfer Learning | Number of local fine-tuning steps | 1 – 20 |
| $K$ | All FL methods | Number of participating clients | 10 – 10,000+ |
| $E$ | FedAvg | Local training epochs per round | 1 – 10 |
| Split point | FedPer | Layer index separating base from personalization layers | Architecture-dependent |

## Algorithm Flow Summary

| Method | Phase 1 (Global) | Phase 2 (Local) | Key Differentiator |
|---|---|---|---|
| FedAvg (baseline) | Full model averaging | None | No personalization |
| Transfer Learning | Standard FL | Fine-tune all/top layers | Simplest; risk of forgetting |
| Meta-Learning (Per-FedAvg) | MAML-style FL training | One-step gradient adaptation | Globally optimized for adaptability |
| Knowledge Distillation | Standard FL (large model) | Student trained with teacher | Supports varied architectures |
| FedPer | Aggregate base layers only | Train personalization layers | Architectural separation |
| MOCHA | Joint multi-task FL | Task-specific layers locally | All clients in every round |
| Mixture Model | Update global component | Update local component | Explicit global-local tradeoff |
| Clustering | Train per-cluster global model | Optional local fine-tuning | Groups similar clients together |

---

# 15. One-Page Master Summary Card

| Element | Content |
|---|---|
| **Problem** | A single shared global model in federated learning performs poorly for clients with non-IID data; those clients have no incentive to participate |
| **Core Insight** | Global accuracy is the wrong metric — per-client accuracy is what matters for personalization; optimizing the global model for adaptability (not just accuracy) is key |
| **Method (Survey)** | 7 personalization strategies: (1) User Context, (2) Transfer Learning, (3) Multi-task Learning, (4) Meta-Learning, (5) Knowledge Distillation, (6) Base+Personalization Layers, (7) Mixture of Global and Local |
| **Key Finding** | Personalized models outperform both global and local models in most settings; meta-learning approaches produce the most adaptable global models |
| **When Personalization Fails** | Under differential privacy, small local datasets, rapidly shifting distributions |
| **Critical Gap** | No unified benchmark for personalization evaluation; global average accuracy still dominates the field despite its inadequacy |
| **Top Research Opportunities** | (1) Per-client adaptive mixing coefficients, (2) DP-aware personalization with adaptive privacy budget, (3) Continual federated personalization for non-stationary data, (4) LLM-era personalization via LoRA/adapters in FL |
| **Publishable Extension** | Pick one weakness, formulate as optimization problem, propose adaptive method, evaluate with per-client fairness metric — ready for FL workshop or IEEE Transactions |

---

*Research Companion generated from: Kulkarni V., Kulkarni M., Pant A. — "Survey of Personalization Techniques for Federated Learning" — 2020*
*Date: March 2026*
