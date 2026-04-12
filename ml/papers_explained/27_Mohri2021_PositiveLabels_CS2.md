# Research Companion Sheet: Federated Learning with Only Positive Labels

**Paper:** Federated Learning with Only Positive Labels  
**Authors:** Felix X. Yu, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar  
**Affiliation:** Google Research, New York  
**Published:** April 2020, arXiv:2004.10342  
**Venue:** ICML 2020

---

## Paper Type Classification

**Primary Type:** Algorithmic / Method with Theoretical Justification  
**Secondary Type:** Mathematical / Theoretical

**Adaptation Strategy:**
- Explain algorithm logic and workflow first, then theoretical backing
- Provide intuition before equations
- Explain theorem purpose and proof strategy (not full derivation)
- Focus on design decisions, baselines, and metrics for experimental sections

---

# 0. Quick Paper Identity Card

| Attribute | Details |
|---|---|
| **Problem Domain** | Federated Learning, Multi-class Classification |
| **Paper Type** | Algorithmic + Theoretical |
| **Core Contribution** | FedAwS: a server-side geometric regularizer that enables federated training when users only have data from one class (no negative labels) |
| **Key Idea** | Instead of requiring negative labels at clients, push class embeddings apart on the server using a "spreadout" regularizer, preventing collapse to a trivial solution |
| **Required Background** | Federated learning basics, embedding-based classifiers, contrastive loss functions, basic optimization |
| **Primary Baseline** | Softmax cross-entropy with access to all labels (oracle), Fixed random class embeddings (Baseline-2) |
| **Main Innovation Type** | New training framework combining positive-only client updates with server-side geometric regularization |
| **Difficulty Level** | Medium-High (math proofs require linear algebra and optimization background) |
| **Reproducibility Level** | High (standard datasets, clear algorithm, well-defined hyperparameters) |

---

# 1. Research Context & Core Problem

## 1.1 Exact Problem Formulation

- In standard federated learning, multiple users (clients) collaboratively train a shared model while keeping their data local
- This paper addresses a **stricter** variant: each user has data from **only one class** (only positive labels for that class)
- Users cannot share data with each other or with the server
- Users cannot access model parameters (class embeddings) belonging to other users
- The model is an **embedding-based classifier**: both inputs and classes are mapped into the same vector space, and classification is based on similarity (dot product) between input embedding and class embeddings

## 1.2 Why the Problem Exists

- **Privacy-sensitive applications:** In face recognition or speaker identification, a user's facial images belong to only one identity (their own). Sharing class embeddings could reveal sensitive identity information
- **Data isolation by nature:** Each user naturally possesses data from only their own class
- **Standard FL fails here:** Normal federated learning requires users to have data from multiple classes so the loss function can push apart negative classes. With only positive data, there is nothing to prevent all class embeddings from collapsing to a single point

## 1.3 The Collapse Problem (Core Technical Gap)

- Standard classification losses have two components:
  - **Positive part:** Pull instance embedding close to its true class embedding
  - **Negative part:** Push instance embedding away from other class embeddings
- With only positive labels, only the positive part is available
- Minimizing only the positive part leads to a **trivial solution**: all instances and all class embeddings converge to a single point in embedding space (zero distance everywhere = zero loss)
- This makes the classifier completely useless (random guessing)

## 1.4 Limitations of Previous Approaches

| Approach | Why It Falls Short |
|---|---|
| **Positive-Unlabeled (PU) Learning** | Requires access to unlabeled data from both positive and negative classes; in this FL setting, clients have no unlabeled data at all |
| **One-Class Classification** | Designed for anomaly/novelty detection for a single class; not for collaborative multi-class model training |
| **Generative Models (GAN-based)** | Could synthesize negative data, but training GANs in federated settings is expensive and a separate unsolved problem |
| **Standard Federated Averaging** | Leads to embedding collapse when users only have positive labels |

## 1.5 Contribution Category

- **Algorithmic:** New training framework (FedAwS) for federated learning with only positive labels
- **Theoretical:** Formal justification that FedAwS approximates conventional training with both positive and negative labels
- **Empirical:** Experiments on CIFAR-10/100 and extreme multi-class datasets showing near-oracle performance

### Why This Paper Matters

- Opens a **completely new problem setting** in federated learning that is directly relevant to privacy-sensitive applications (face recognition, speaker identification)
- Provides a **principled solution** (not a heuristic) with theoretical guarantees
- Demonstrates that a simple server-side geometric regularizer can substitute for the absence of negative labels
- The spreadout idea has **independent value** beyond FL, applicable to extreme multi-class classification as an alternative to negative sampling

### Remaining Open Problems

1. Extending to settings where class embeddings depend on class-level features (not just learned ID vectors)
2. Formal differential privacy guarantees integrated with FedAwS (currently only noted as compatible, not analyzed)
3. Handling non-IID data distributions more robustly within this positive-only setting
4. Communication efficiency improvements specific to the positive-only setting
5. Extension to multi-label settings where users may have positive data for more than one class but still lack negatives
6. Convergence rate analysis under partial client participation

---

# 2. Minimum Background Concepts

### 2.1 Embedding-Based Classifier

- **Plain definition:** A model that maps both input data and class labels into the same vector space. Classification is done by finding which class vector is closest (most similar) to the input's vector
- **Role in paper:** This is the model architecture the entire paper is built around. The input is embedded by a neural network `g_theta(x)` into a d-dimensional vector. Each class c has a learned embedding vector `w_c`. The score for class c is the dot product `w_c^T * g_theta(x)`
- **Why authors needed it:** Embedding-based models naturally decompose into per-class parameters (each `w_c` is independent), making it possible to send only the relevant class embedding to each user without sharing other classes' information

### 2.2 Federated Averaging (FedAvg)

- **Plain definition:** An iterative training protocol where a central server sends the current model to clients, clients update it using their local data, and the server averages all updates to produce a new global model
- **Role in paper:** FedAvg is the base training protocol that the authors modify. Their FedAwS adds an extra server-side step on top of FedAvg
- **Why authors needed it:** This is the standard federated learning algorithm; the paper's contribution is showing how to adapt it when clients only have positive labels

### 2.3 Contrastive Loss

- **Plain definition:** A loss function that simultaneously (a) pulls similar items together and (b) pushes dissimilar items apart in embedding space
- **Role in paper:** The standard contrastive loss requires both positive and negative pairs. The paper's key challenge is that only the positive part is available at clients. The authors show their method implicitly recovers an approximation of the full contrastive loss
- **Why authors needed it:** To formally define what is missing (the negative part) when training with only positive labels, and to show that their regularizer compensates for it

### 2.4 Cosine Distance

- **Plain definition:** For two unit-norm vectors u and u', the cosine distance is `d_cos(u, u') = 1 - u^T * u'`. It equals 0 when vectors point in the same direction and 2 when they point in opposite directions
- **Role in paper:** The specific distance measure used in the theoretical analysis and experiments. Using normalized embeddings with cosine distance simplifies the math and slightly improves empirical performance
- **Why authors needed it:** Provides a clean mathematical framework where the spreadout regularizer and contrastive loss have nice analytical properties

### 2.5 Surrogate Loss

- **Plain definition:** A continuous, differentiable loss function that upper-bounds the true (often non-differentiable) loss of interest (e.g., 0-1 misclassification error). Minimizing the surrogate also drives down the true loss
- **Role in paper:** The authors prove that the cosine contrastive loss is a valid surrogate for misclassification error, which means minimizing it leads to good classifiers
- **Why authors needed it:** This is a critical step in justifying FedAwS theoretically: if FedAwS approximates minimizing the cosine contrastive loss, and the cosine contrastive loss is a valid surrogate, then FedAwS leads to good classifiers

### 2.6 Stochastic Negative Mining

- **Plain definition:** Instead of comparing against all possible negative classes (which is expensive), only compare against the k hardest (most confusing) negative classes
- **Role in paper:** Used to make the spreadout regularizer scalable to settings with huge numbers of classes (e.g., 670K classes)
- **Why authors needed it:** Computing the full spreadout regularizer over all class pairs is O(C^2), which is infeasible for large C. Mining only the k nearest classes makes it tractable

---

# 3. Mathematical / Theoretical Understanding Layer

## 3.1 The Scorer Function (Eq. 1)

**Equation:** `f(x) = W * g_theta(x)`

| Symbol | Meaning |
|---|---|
| `x` | Input instance (e.g., an image) |
| `g_theta` | Neural network that maps input to a d-dimensional embedding |
| `theta` | Parameters of the embedding network |
| `W` | Class embedding matrix of size C x d |
| `w_c` | c-th row of W; embedding vector for class c |
| `f(x)_c = w_c^T * g_theta(x)` | Score (logit) for class c |

**Intuition:** The model embeds the input into a shared space, then computes similarity with each class embedding. The class with the highest similarity score is the predicted class.

## 3.2 Empirical Risk (Eq. 2)

**Equation:** `R_hat(f; S_i) = (1/n_i) * sum_{j in [n_i]} l(f(x_ij), y_ij)`

**Intuition:** The average loss computed on client i's local data. Each client tries to minimize this using gradient descent.

## 3.3 Federated Averaging Update (Eqs. 3-5)

**Client update:**
- `theta_i_t = theta_t - eta * gradient_theta R_hat(f_t; S_i)`
- `W_i_t = W_t - eta * gradient_W R_hat(f_t; S_i)`

**Server aggregation:**
- `theta_{t+1} = sum_{i} omega_i * theta_i_t`
- `W_{t+1} = sum_{i} omega_i * W_i_t`

**Intuition:** Each client takes a gradient step on its local data. The server averages all client updates, weighted by `omega_i = n_i / n` (proportional to data size).

## 3.4 Contrastive Loss (Eq. 6)

**Equation:**
```
l_cl(f(x), y) = alpha * [d(g_theta(x), w_y)]^2        (positive part)
              + beta * sum_{c != y} [max(0, nu - d(g_theta(x), w_c))]^2  (negative part)
```

| Component | Purpose |
|---|---|
| Positive part `l_pos` | Penalizes large distance between instance and its true class |
| Negative part `l_neg` | Penalizes small distance between instance and non-true classes |
| `nu` (margin) | Minimum desired distance between instance and negative classes |

**Intuition:** The loss says: "pull the input close to its true class, and push it away from all other classes by at least margin nu." Without negative labels, only the pull part is available, leading to collapse.

## 3.5 Spreadout Regularizer (Eq. 7)

**Equation:**
```
reg_sp(W) = sum_{c in [C]} sum_{c' != c} [max(0, nu - d(w_c, w_{c'}))]^2
```

**What problem it solves:** Prevents class embeddings from collapsing to a single point by penalizing any pair of class embeddings that are closer than margin nu.

**Intuition:** Think of it as placing C magnets on a sphere and adding repulsive forces between any two magnets that get too close. The regularizer has zero penalty when all classes are at least nu apart, and increases quadratically as classes get closer.

**Assumptions:**
- Applied only on the server (server has access to all class embeddings W)
- Does not involve any client data
- Margin nu must be chosen appropriately (addressed by stochastic negative mining variant)

**Limitation:** Computing this over all C^2 pairs is expensive for large C.

## 3.6 Stochastic Negative Mining Regularizer (Eq. 8)

**Equation:**
```
reg_sp_top(W) = sum_{c in C_t} sum_{y in C', y != c} -d^2(w_c, w_y) * I[y in N_k(c)]
```

| Symbol | Meaning |
|---|---|
| `C_t` | Set of classes participating in round t |
| `C'` | Subset of classes sampled for comparison |
| `N_k(c)` | Set of k classes closest to class c in embedding space |
| `I[.]` | Indicator function (1 if true, 0 otherwise) |

**Intuition:** Instead of pushing ALL class pairs apart (O(C^2) cost), only push apart each class from its k nearest neighbors. This adaptively focuses the repulsion where it matters most (on classes that are at risk of being confused). The margin nu is effectively set adaptively as the distance to the (k+1)-th nearest class.

## 3.7 Proposition 1: Classification Error Bound

**Statement:** If the minimum distance between any two class embeddings is `rho`, and the expected distance between an instance and its true class is `epsilon`, then:
```
P(misclassification) <= 2 * epsilon / rho
```

**Intuition:** This is a fundamental result saying: if classes are well separated (large rho) and instances are close to their true classes (small epsilon), then classification error is low. This directly motivates the two parts of FedAwS:
- Client updates reduce epsilon (pull instances toward their class)
- Spreadout regularizer increases rho (push classes apart)

**Proof strategy:** Uses triangle inequality to show that misclassification implies `d(g(x), w_y) >= rho/2`, then applies Markov's inequality to bound the probability.

## 3.8 Cosine Contrastive Loss (Definition 1, Eqs. 12-13)

**Equation (using logits s_c = g_theta(x)^T * w_c):**
```
l_ccl(f(x), y) = (1 - s_y)^2 + sum_{c != y} [max(0, nu - 1 + s_c)]^2
```

**Intuition:** Specialized contrastive loss for cosine distance with unit-norm embeddings. The positive part `(1 - s_y)^2` wants the logit of the true class to be 1 (perfect alignment). The negative part wants logits of non-true classes to be below `1 - nu` (sufficiently dissimilar).

## 3.9 Lemma 1: Surrogate Loss Property

**Statement:** For `nu in (1, 2)`:
```
l_ccl(f(x), y) >= 2(nu - 1) * I[y not in Top1(f(x))]
```

**Intuition:** The cosine contrastive loss is always at least `2(nu - 1)` whenever the classifier makes a mistake. This means minimizing this loss also minimizes the misclassification rate. This is essential because it proves the cosine contrastive loss is a **valid training objective** for classification.

**Proof strategy:** Two cases: (1) correct prediction: trivially true since loss >= 0. (2) incorrect prediction: uses convexity of the hinge-squared function to show the loss is lower-bounded by `2(nu - 1)`.

## 3.10 Proposition 2: FedAwS Objective Equivalence

**Statement:** When `lambda = 1/C` and all clients have equal data (`n_1 = ... = n_C = n/C`), the FedAwS objective equals the empirical risk with respect to the loss:
```
l_sp(f(x), y) = (1 - s_y)^2 + sum_{c != y} [max(0, nu - 1 + w_y^T * w_c)]^2
```

**Key insight:** The FedAwS objective can be rewritten as a standard empirical risk minimization problem with a specific loss function. This loss has:
- **Positive part:** Same as cosine contrastive loss: `(1 - s_y)^2`
- **Negative part:** Uses class-to-class similarity `w_y^T * w_c` instead of instance-to-class similarity `g_theta(x)^T * w_c`

**Why this matters:** It shows FedAwS is not a heuristic; it is equivalent to minimizing a well-defined loss function.

## 3.11 Theorem 1: Approximation Quality

**Statement:** The FedAwS loss `l_sp` approximates the cosine contrastive loss `l_ccl` with error bounded by:
```
|l_sp(f(x), y) - l_ccl(f(x), y)| <= (1 + 2*nu) * sum_{c != y} |w_c^T * r_{x,y}|
```
where `r_{x,y} = w_y - g_theta(x)` is the mismatch between the true class embedding and the instance embedding.

**Intuition:** The approximation error depends on how close the instance embedding is to its true class embedding. As training progresses and instances move closer to their classes (as encouraged by the positive part of the loss), the approximation becomes tighter. In other words, FedAwS becomes a better and better approximation of the full contrastive loss as training progresses.

**Practical interpretation:** Early in training, FedAwS may behave differently from standard training, but as the model improves, the two converge.

### Mathematical Insight Box

> **Key takeaway for researchers:** The spreadout regularizer on class embeddings serves as a **proxy for the negative part of the contrastive loss** that would normally require negative labels. The quality of this proxy improves as the model trains, creating a self-reinforcing cycle: better positive alignment leads to a better proxy, which leads to better class separation, which leads to better overall classification. This "bootstrap" property is what makes FedAwS work despite never seeing negative labels.

---

# 4. Proposed Method / Framework (FedAwS)

## 4.1 Overall Pipeline

FedAwS modifies standard Federated Averaging with two key changes:

```
STANDARD FEDAVG:                    FedAwS:
                                    
Server sends full model       -->   Server sends theta + only w_i to client i
Clients update full model     -->   Clients update theta + w_i using positive loss only
Server averages all W updates -->   Server collects individual w_i updates (no averaging of W)
                              -->   Server applies spreadout regularizer to push W apart
```

## 4.2 Step-by-Step Algorithm (Algorithm 1)

### Step 1: Server Initialization
- Initialize model parameters `theta_0` and class embedding matrix `W_0`

**Why:** Standard initialization; all embeddings start random

**Weakness:** If initial embeddings are already clustered, the spreadout regularizer may need many rounds to separate them

**Improvement idea:** Use structured initialization (e.g., vertices of a regular polytope) to start with well-separated embeddings

### Step 2: Server Communication (per round t)
- Server sends `theta_t` (shared embedding network) and `w_t_i` (only the i-th class embedding) to client i

**Why:** Privacy constraint: client i should not see other clients' class embeddings. The shared `theta` is not class-specific, so sharing it does not leak class identity information

**Weakness:** Communication of the full `theta` to every client may be expensive for large models

**Improvement idea:** Combine with communication compression techniques (e.g., gradient quantization)

### Step 3: Client Local Update
- Client i computes:
  ```
  (theta_{t,i}, w_{t,i}_i) = (theta_t, w_t_i) - eta * gradient l_pos(f(x), y)
  ```
- Where `l_pos` is the positive part of the loss only (e.g., squared hinge loss):
  ```
  l_pos(f(x), y) = [max(0, 0.9 - g_theta(x)^T * w_y)]^2
  ```

**Why:** Client only has positive labels, so can only minimize distance between instances and their class embedding. The threshold 0.9 means the loss is zero when the cosine similarity exceeds 0.9

**Weakness:** Without negative labels, the positive-only update creates a "pull" force but no "push" force, which would cause collapse without the server-side regularizer

**Improvement idea:** If some unlabeled data is available at clients, incorporate it via pseudo-labeling or PU learning techniques alongside FedAwS

### Step 4: Server Collects and Assembles W
- Server receives updated parameters from all clients
- Averages `theta`: `theta_{t+1} = (1/C) * sum_i theta_{t,i}`
- Assembles class embedding matrix: `W_tilde_{t+1} = [w_{t,1}_1, ..., w_{t,C}_C]^T`

**Why:** Unlike standard FedAvg which averages W, here each class embedding is updated only by its owning client. The server simply collects and stacks them. There is no averaging of W because each row of W is "owned" by a different client

**Weakness:** No cross-pollination between class embeddings during local updates

**Improvement idea:** Server could apply additional transformations beyond spreadout (e.g., whitening or decorrelation)

### Step 5: Server Applies Spreadout Regularizer
- `W_{t+1} = W_tilde_{t+1} - lambda * eta * gradient reg_sp(W_tilde_{t+1})`

**Why:** This is the core innovation. The spreadout step pushes apart any class embeddings that are too close, preventing collapse. It acts as a substitute for the missing negative labels

**Weakness:** The margin nu is a hyperparameter that is hard to set optimally

**Improvement idea:** Use the stochastic negative mining variant (Section 4.2) which adaptively sets the margin

### Step 6: Repeat for T rounds, output final model

## 4.3 FedAwS with Stochastic Negative Mining (Scalability Extension)

**Problem with basic spreadout:** Two issues for large C (many classes):
1. Computing `reg_sp(W)` requires O(C^2) pairwise distances
2. The optimal margin nu depends on the problem and is hard to tune

**Solution:** For each class c, only push apart from its k nearest neighbor classes:
```
reg_sp_top(W) = sum_c sum_{y in N_k(c)} -d^2(w_c, w_y)
```

**Simplified pseudocode:**
```
For each class c in participating set:
    Find k nearest classes to c (by embedding distance)
    Push c away from those k classes (maximize distance)
```

**Why this works:** Classes that are already far apart do not need additional repulsion. Only nearby (confusing) classes need to be pushed apart. This is analogous to hard negative mining in standard contrastive learning, but applied at the class level instead of instance level.

**Design choices:**
- k = 10 and lambda = 10 used as defaults
- Too small k: misses important negatives; too large k: wastes computation and may over-spread naturally similar classes
- C' (subset of classes for comparison) can be randomly sampled each round for further scalability

## 4.4 Key Design Decisions Summary

| Decision | Reasoning | Alternative Considered |
|---|---|---|
| Embedding-based classifier | Allows per-class parameter isolation | Generative models (rejected: expensive, don't fit FL framework) |
| Positive-only loss at clients | Privacy constraint; clients cannot see other class embeddings | PU learning (rejected: no unlabeled data available) |
| Server-side spreadout regularizer | Server has access to all class embeddings; compensates for missing negative part | GAN-generated negatives (rejected: additional expensive training) |
| Stochastic negative mining | Scalability to large C; adaptive margin | Fixed margin (rejected: hard to tune, O(C^2) cost) |
| L2 normalization of embeddings | Improves empirical performance, simplifies theory | Unnormalized (works but slightly worse) |

---

# 5. Experimental Setup / Evaluation Design

## 5.1 Datasets

### Image Classification (Small-scale)
| Dataset | Classes | Purpose |
|---|---|---|
| CIFAR-10 | 10 | Validate on small number of classes |
| CIFAR-100 | 100 | Validate on moderate number of classes |

### Extreme Multi-class Classification (Large-scale)

| Dataset | Features | Labels | Train Points | Test Points | Avg Instances/Label | Avg Labels/Instance |
|---|---|---|---|---|---|---|
| AmazonCat | 203,882 | 13,330 | 1,186,239 | 306,782 | 448.57 | 5.04 |
| WikiLSHTC | 1,617,899 | 325,056 | 1,778,351 | 587,084 | 17.46 | 3.19 |
| Amazon670K | 135,909 | 670,091 | 490,449 | 153,025 | 3.99 | 5.45 |

**Note:** Extreme multi-class datasets are originally multi-label. Authors uniformly sample one positive label per instance to create multi-class problems.

## 5.2 Models

| Experiment | Architecture | Embedding Dim |
|---|---|---|
| CIFAR-10 | ResNet-8, ResNet-32 | 64 |
| CIFAR-100 | ResNet-32, ResNet-56 | 64 |
| Extreme multi-class | Linear embedding + 3-layer NN (1024-1024-512) + L2 norm | 512 |

## 5.3 Methods Compared

| Method | Has Negative Labels? | Description |
|---|---|---|
| **Baseline-1** | No | Positive-only squared hinge loss (no collapse prevention) |
| **Baseline-2** | No | Positive-only loss with class embeddings FIXED at random initialization |
| **FedAwS** | No | Proposed method with stochastic negative mining |
| **Softmax (Oracle)** | Yes | Standard training with softmax cross-entropy (full label access) |
| **SLEEC (Oracle)** | Yes | Extreme classification method with full label access |

## 5.4 Metrics

- **Precision@1 (P@1):** Fraction of times the top-1 prediction is correct
- **Precision@3 (P@3):** Fraction of times the true label is among top-3 predictions
- **Precision@5 (P@5):** Fraction of times the true label is among top-5 predictions

**Why these metrics:** P@1 is standard classification accuracy. P@3 and P@5 are critical for extreme multi-class settings where the original data is multi-label and top-k predictions matter for applications like product recommendation

## 5.5 Hyperparameters

| Parameter | Value | Reasoning |
|---|---|---|
| `lambda` (spreadout weight) | 10 | Controls strength of spreadout; needs to be large enough to prevent collapse |
| `k` (nearest neighbors in mining) | 10 | Number of hard negative classes per class |
| Positive threshold | 0.9 | Cosine similarity target for positive pairs |
| Optimizer (extreme) | SGD (embeddings) + Adagrad (other) | Different learning rate needs |
| Clients per round (extreme) | 4,000 | Random subset of all clients |

## 5.6 Experimental Reliability Analysis

**What is trustworthy:**
- CIFAR experiments use standard architectures and datasets with well-understood baselines
- Extreme multi-class experiments use established benchmark datasets
- Comparison against oracle methods (with full label access) provides a meaningful upper bound
- Hyperparameter sensitivity analysis provided for lambda and k on AmazonCat

**What is questionable:**
- Only one run reported; no error bars or confidence intervals
- The conversion from multi-label to multi-class (by sampling one label) may not fully represent real federated scenarios
- No experiments on true federated scenarios with heterogeneous compute/network
- WikiLSHTC shows a significant gap between FedAwS (37.2 P@1) and Oracle (54.1 P@1), suggesting the method has limitations for certain data distributions
- No communication cost analysis despite this being a federated learning paper
- Privacy analysis is mentioned but not formally conducted

---

# 6. Results & Findings Interpretation

## 6.1 CIFAR Results (Table 1)

| Dataset | Model | Baseline-1 | Baseline-2 | FedAwS | Softmax (Oracle) |
|---|---|---|---|---|---|
| CIFAR-10 | ResNet-8 | 10.7 | 83.3 | **86.3** | 88.4 |
| CIFAR-10 | ResNet-32 | 9.8 | 92.1 | **92.4** | 92.4 |
| CIFAR-100 | ResNet-32 | 1.0 | 65.1 | **67.9** | 68.0 |
| CIFAR-100 | ResNet-56 | 1.1 | 67.5 | **69.6** | 70.0 |

**Key findings:**
- **Baseline-1 confirms collapse:** ~10% on CIFAR-10 (random guessing for 10 classes), ~1% on CIFAR-100 (random guessing for 100 classes)
- **FedAwS nearly matches oracle:** On CIFAR-10 with ResNet-32, FedAwS achieves 92.4% vs Oracle's 92.4% (identical). On CIFAR-100, the gap is only 0.1-0.4%
- **Baseline-2 works surprisingly well on CIFAR-10:** Random fixed embeddings in 64 dimensions are nearly orthogonal when C=10, providing natural separation. This insight explains why simply fixing embeddings can work for few classes but fails for many

**Statistical meaning:** The near-zero gap between FedAwS and Oracle on CIFAR validates the theoretical result that FedAwS approximates the full contrastive loss well when instances are close to their class embeddings

## 6.2 Extreme Multi-class Results (Table 3)

| Dataset | Metric | Baseline-1 | Baseline-2 | FedAwS | Softmax | SLEEC |
|---|---|---|---|---|---|---|
| AmazonCat | P@1 | 3.4 | 64.1 | **92.1** | 92.1 | 90.5 |
| Amazon670K | P@1 | 0.0 | 4.3 | **33.1** | 35.2 | 35.1 |
| WikiLSHTC | P@1 | 7.6 | 7.9 | **37.2** | 54.1 | 54.8 |

**Key findings:**
- **FedAwS matches Oracle on AmazonCat:** 92.1 P@1 for both, a remarkable result for a method without negative labels
- **Baseline-2 fails for large C:** Fixed random embeddings give 4.3% on Amazon670K (670K classes) vs 64.1% on AmazonCat (13K classes). In high dimensions, random vectors become less orthogonal as the number of vectors approaches or exceeds the dimensionality
- **WikiLSHTC gap is significant:** FedAwS achieves 37.2 vs Oracle's 54.1 (a 17-point gap). This dataset has 325K labels with few instances per label (avg 17.46), making it the hardest scenario

**Failure cases:**
- WikiLSHTC: The large gap suggests FedAwS struggles when classes have very few training instances (avg 17 per class) and the label space is extremely large
- The spreadout regularizer may over-spread classes that should be semantically close

## 6.3 Hyperparameter Sensitivity (Table 4, AmazonCat)

| Setting | P@1 | P@3 | P@5 |
|---|---|---|---|
| k = 10 (too few) | 26.3 | 21.5 | 18.2 |
| k = 100 (sweet spot) | **92.1** | **70.8** | **58.7** |
| k = 500 | 86.9 | 66.1 | 49.3 |
| k = all | 87.7 | 69.7 | 52.2 |
| lambda = 1 (too weak) | 73.2 | 50.2 | 40.4 |
| lambda = 10 (good) | **92.1** | **70.8** | **58.7** |
| lambda = 100 | 92.2 | 71.7 | 57.9 |

**Key findings:**
- **k matters significantly:** k=10 performs terribly (26.3%) because in a multi-label dataset, the nearest classes are often actual positives that should NOT be pushed apart. k=100 is the sweet spot
- **lambda needs to be sufficiently large:** lambda=1 gives 73.2% (inadequate spreadout), while lambda=10 and lambda=100 both give ~92% (sufficient spreadout)
- **Robustness to lambda:** Performance is stable for lambda >= 10, suggesting the method is not overly sensitive to this parameter once it is large enough

### Publishability Strength Check

**Publication-grade results:**
- FedAwS matching Oracle on CIFAR and AmazonCat (strong, convincing evidence)
- Clear demonstration of the collapse problem (Baseline-1)
- Theoretical framework connecting FedAwS to contrastive loss

**Results needing stronger validation:**
- WikiLSHTC gap is significant and not fully explained
- No error bars or multiple runs
- No real federated experiments (all simulated)
- No communication efficiency analysis
- No comparison with differential privacy overhead

---

# 7. Strengths - Weaknesses - Assumptions

## 7.1 Technical Strengths

| # | Strength | Evidence |
|---|---|---|
| 1 | **Novel and well-motivated problem formulation** | First paper to formally study FL with only positive labels; clear practical applications (face/speaker recognition) |
| 2 | **Elegant solution with theoretical backing** | FedAwS is simple (one extra server-side step) yet principled (provably approximates full contrastive loss) |
| 3 | **Near-oracle performance** | Matches softmax training on CIFAR-10/100 and AmazonCat despite having no negative labels |
| 4 | **Scalable extension** | Stochastic negative mining makes the method practical for 670K+ classes |
| 5 | **Clean separation of client/server responsibilities** | Clients handle positive learning, server handles class separation; no additional client-side complexity |
| 6 | **Self-reinforcing approximation** | Theorem 1 shows the approximation improves as training progresses |
| 7 | **Applicable beyond FL** | Spreadout regularizer has independent value for extreme multi-class classification |

## 7.2 Explicit Weaknesses

| # | Weakness | Impact |
|---|---|---|
| 1 | **No formal privacy analysis** | Claims privacy benefit but does not provide differential privacy guarantees or privacy budget analysis |
| 2 | **Large gap on WikiLSHTC** | 17-point gap from oracle suggests the method breaks down with very large label spaces and sparse labels |
| 3 | **No communication cost analysis** | Key metric in FL; unclear how FedAwS compares to standard FL in communication rounds and bandwidth |
| 4 | **Simulated federated setting** | Experiments simulate FL; no real distributed training with actual edge devices |
| 5 | **No error bars** | Single-run results make it hard to assess statistical significance |
| 6 | **Hyperparameter sensitivity** | k and lambda require tuning; k=10 fails badly on AmazonCat |
| 7 | **Equal-data assumption in theory** | Propositions 2 and Theorem 1 assume `n_1 = ... = n_C`, which rarely holds in practice |

## 7.3 Hidden Assumptions

| # | Assumption | Potential Issue |
|---|---|---|
| 1 | Each client has data from exactly one class | In reality, users may have data from 2-3 classes but still lack negatives for most classes |
| 2 | Class embeddings are the only sensitive parameters | In some applications, `theta` (shared model) may also leak information about classes |
| 3 | Server is honest and trustworthy | Server sees all class embeddings W; a malicious server could infer class relationships |
| 4 | All classes have sufficient data | WikiLSHTC results suggest FedAwS degrades when some classes have very few instances |
| 5 | Embedding dimensionality is large enough to separate all classes | For C >> d, perfect separation is geometrically impossible |
| 6 | Classes are semantically distinct | The spreadout regularizer pushes ALL non-identical classes apart equally; does not account for class hierarchy or semantic similarity |
| 7 | Cosine distance is appropriate for all domains | Theory relies on unit-norm embeddings with cosine distance; may not suit all problem types |

---

# 8. Weakness to Research Direction Mapping

| Identified Weakness | Why it Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| No formal privacy guarantee | Paper focuses on the positive-labels problem, not privacy | **FedAwS with Differential Privacy:** Formally analyze privacy-accuracy tradeoffs when adding DP noise to FedAwS updates | Add calibrated Gaussian noise to client updates and spreadout gradients; analyze privacy budget per round |
| Large gap on WikiLSHTC | Sparse labels (avg 17/class) make positive-only learning harder; spreadout may over-separate related classes | **Adaptive spreadout with class similarity awareness:** Modify regularizer to allow semantically related classes to stay closer | Learn a class hierarchy or similarity graph; modify spreadout margin per class pair based on semantic distance |
| No communication cost analysis | Not the paper's focus | **Communication-efficient FedAwS:** Analyze and reduce communication overhead | Compress embedding updates; send only changed dimensions; use top-k sparsification |
| Hyperparameter sensitivity (k, lambda) | Fixed k and lambda cannot adapt to different data distributions | **Self-tuning FedAwS:** Automatically adapt k and lambda during training | Use validation-based tuning or meta-learning to adjust hyperparameters per round |
| Equal-data assumption in theory | Simplifies proof but is unrealistic | **Heterogeneous FedAwS:** Extend theory and algorithm to handle unequal data sizes | Weighted spreadout regularizer with weights proportional to class frequency |
| No multi-label support | Paper converts multi-label to single-label | **Multi-positive FedAwS:** Handle the case where clients have multiple positive classes | Extend loss function to handle multiple positive labels per client; adjust spreadout to not push apart co-occurring classes |
| Server sees all embeddings | Architectural limitation | **Privacy-preserving spreadout:** Apply spreadout without server seeing raw class embeddings | Use secure aggregation or homomorphic encryption to compute pairwise distances on encrypted embeddings |
| No class-feature embeddings | Classes are simple ID-based learned vectors | **Feature-conditioned FedAwS:** Generate class embeddings from class-level features | Use a class-feature encoder network; the server applies spreadout on generated embeddings |

---

# 9. Novel Contribution Extraction

## 9.1 Paper's Novel Contributions (Author Claims)

1. **First problem formulation:** "We formulate the novel problem of federated learning with only positive labels, where each client has access to data from only a single class"
2. **FedAwS algorithm:** "We propose Federated Averaging with Spreadout (FedAwS), which uses server-side geometric regularization to prevent embedding collapse"
3. **Theoretical justification:** "We prove that FedAwS approximates the cosine contrastive loss (which requires negative labels) and that this approximation improves as training progresses"
4. **Scalable extension:** "We extend FedAwS using stochastic negative mining to handle settings with hundreds of thousands of classes"

## 9.2 Novel Claim Templates for New Research

**Template 1 (Privacy Extension):**
> "We propose DP-FedAwS that achieves (epsilon, delta)-differential privacy guarantees while maintaining the positive-labels-only training paradigm, improving upon FedAwS by providing formal privacy protection with only X% accuracy degradation."

**Template 2 (Adaptive Method):**
> "We propose AdaSpread that automatically tunes the spreadout margin and mining depth during federated training, improving upon FedAwS by eliminating hyperparameter sensitivity and achieving Y% higher accuracy on sparse-label datasets."

**Template 3 (Multi-Positive Extension):**
> "We propose FedAwS-MP that extends positive-only federated learning to the multi-positive setting where each client has data from a small subset of classes, improving upon FedAwS by handling Z% more realistic data distributions."

**Template 4 (Communication Efficiency):**
> "We propose Compressed-FedAwS that reduces communication cost by W% through embedding-specific compression while maintaining the spreadout regularization effectiveness of FedAwS."

**Template 5 (Hierarchical Classes):**
> "We propose HierFedAwS that incorporates class hierarchy into the spreadout regularizer, improving upon FedAwS by allowing semantically related classes to maintain proximity while preventing unrelated class collapse."

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Work

1. Extend ID-based class embeddings to **class-feature-conditioned embeddings** (e.g., class descriptions generating embeddings)
2. Apply the spreadout-as-replacement-for-negative-sampling idea to **conventional extreme multi-class classification** (not just FL)

## 10.2 Missing Directions (Not Mentioned by Authors)

1. **Convergence rate analysis:** How many rounds does FedAwS need compared to standard FL? Formal convergence guarantees?
2. **Client heterogeneity:** What happens when some clients have 10x more data than others? Non-uniform data distributions?
3. **Robustness to adversarial clients:** Can a malicious client corrupt the shared model by sending bad updates?
4. **Partial participation analysis:** Formal analysis of how random client sampling affects the spreadout step
5. **Non-embedding classifiers:** Can the spreadout idea be extended to non-embedding architectures?

## 10.3 Modern Extensions (Post-2020)

1. **Personalized FL + Positive Labels:** Combine FedAwS with personalized FL methods (e.g., FedPer, FedProx) for per-client model customization
2. **Foundation models:** Apply FedAwS to fine-tune large pre-trained models (e.g., CLIP, ViT) in federated positive-only settings
3. **Self-supervised FL:** Combine FedAwS with self-supervised learning to leverage unlabeled data alongside positive labels
4. **Federated continual learning:** Extend FedAwS to handle new classes appearing over time without forgetting old ones
5. **Cross-device FL with positive labels:** Scale FedAwS to millions of devices with extreme communication and compute constraints

## 10.4 Cross-Domain Combinations

1. **Medical imaging + FedAwS:** Each hospital has images of certain diseases only; train a shared diagnostic model
2. **Recommendation systems + FedAwS:** Each user's interactions are positive signals only; collaborative filtering without negative feedback
3. **NLP + FedAwS:** Each user has documents from specific topics; train a shared text classifier
4. **Biometrics + FedAwS:** Train speaker/face verification models across organizations without sharing identity data

## 10.5 LLM-Era Extensions

1. **Federated fine-tuning of LLMs with positive-only labels:** Users provide only examples of desired outputs; server applies spreadout to prevent output collapse
2. **Federated RLHF with positive-only preference data:** Each user provides only "preferred" completions; extend FedAwS to the preference learning setting
3. **Federated retrieval-augmented generation:** Each client contributes embeddings for their documents; spreadout prevents embedding collapse in the shared retrieval space

---

# 11. How to Write a NEW Paper From This Work

## 11.1 Reusable Elements

| Element | How to Reuse |
|---|---|
| Problem formulation style | Define the constrained FL setting first, then show why naive approaches fail, then propose solution |
| Collapse argument | Use the "trivial solution" argument to motivate any regularization-based approach |
| Theoretical framework | Follow the pattern: define surrogate loss -> prove it is valid -> show method approximates it |
| Evaluation design | Compare against: (1) naive baseline, (2) simple fix baseline, (3) proposed method, (4) oracle with full information |
| Hyperparameter sensitivity study | Analyze two most impactful parameters in detail on one dataset |

## 11.2 What MUST NOT Be Copied

- The exact spreadout regularizer formula (Eq. 7) without modification or extension
- The exact algorithm pseudocode (Algorithm 1) without adding novel steps
- The exact proof structure of Theorem 1 (create your own proof for your novel loss)
- Specific experimental results or numbers
- Exact sentence constructions from the paper

## 11.3 How to Design a Novel Extension

1. **Identify one weakness** from Section 7 or 8 above
2. **Formulate it as a research question:** "Can FedAwS maintain performance under differential privacy constraints?"
3. **Design a modification:** Add DP noise mechanism + analyze its interaction with spreadout
4. **Prove something new:** Show that your modified method still approximates the contrastive loss (even with noise)
5. **Evaluate incrementally:** Same datasets + your new setting + ablation studies

## 11.4 Minimum Publishable Contribution Checklist

- [ ] Novel problem formulation OR novel method OR novel theoretical result (at least one must be genuinely new)
- [ ] Formal problem statement with clear notation
- [ ] At least one theoretical guarantee (convergence, approximation, or privacy bound)
- [ ] Experiments on at least 3 datasets (small, medium, large scale)
- [ ] Comparison against FedAwS as a baseline (since it is the state-of-the-art for this setting)
- [ ] Ablation study showing each component of your method contributes
- [ ] Error bars or confidence intervals over multiple runs
- [ ] Communication cost analysis (missing in original paper; adding it strengthens your contribution)
- [ ] Clear discussion of limitations

---

# 12. Complete Paper Writing Template

## Abstract (150-250 words)

**Purpose:** Concisely state the problem, your solution, key theoretical/empirical results

**What to include:**
- Problem: FL with only positive labels and its limitation (embedding collapse)
- Gap: Previous methods [cite FedAwS] lack [your identified gap]
- Solution: Your proposed method name and its key idea
- Theory: One-sentence summary of your main theoretical result
- Results: Quantitative improvement over FedAwS on specific benchmarks

**Common mistakes:** Too vague ("we improve FL"), too long (exceeding 250 words), no quantitative results

**Reviewer expectations:** Can immediately tell what is new, why it matters, and whether results are significant

## 1. Introduction (1.5-2 pages)

**Purpose:** Motivate the problem, state contributions clearly

**What to include:**
- Paragraph 1: General context (FL + privacy + positive-only data)
- Paragraph 2: The specific problem and why existing solutions fall short
- Paragraph 3: Your approach at a high level
- Paragraph 4: Summary of contributions (numbered list)

**Common mistakes:** Too much background (save for Related Work), contributions too vague, not clearly differentiating from FedAwS

**Reviewer expectations:** Clear statement of what is novel and why it matters

## 2. Related Work (0.75-1 page)

**Purpose:** Position your work relative to existing literature

**What to include:**
- Federated learning (general)
- FL with limited labels (FedAwS and follow-up works)
- Privacy in FL (DP, secure aggregation)
- Your specific related area (e.g., PU learning, metric learning)
- Clear statement of how your work differs from each group

**Common mistakes:** Just listing papers without explaining differences, missing key references, not citing FedAwS

**Reviewer expectations:** Comprehensive coverage, honest comparison, clear differentiation

## 3. Problem Setup (0.5-1 page)

**Purpose:** Formally define the setting and notation

**What to include:**
- Notation table
- Formal problem statement
- Assumptions (clearly stated)
- What information each party (client, server) has access to

**Common mistakes:** Inconsistent notation, missing assumptions, ambiguous problem statement

**Reviewer expectations:** Precise, unambiguous, all symbols defined

## 4. Method (2-3 pages)

**Purpose:** Present your algorithm and design choices

**What to include:**
- Overview of approach
- Algorithm pseudocode
- Explanation of each step with justification
- Comparison with FedAwS (what you add/change)
- Computational complexity analysis
- Communication cost per round

**Common mistakes:** Missing pseudocode, no complexity analysis, unclear motivation for design choices

**Reviewer expectations:** Reproducible from the paper alone, clear improvement over baseline

## 5. Theoretical Analysis (1-2 pages)

**Purpose:** Provide formal guarantees for your method

**What to include:**
- Main theorem statement
- Intuitive explanation of what the theorem means
- Proof sketch (full proof in appendix)
- Corollaries or special cases
- Comparison with FedAwS theoretical guarantees

**Common mistakes:** Theorem without intuition, proof without context, assumptions buried in proof

**Reviewer expectations:** Correct proofs, meaningful guarantees, clear assumptions

## 6. Experiments (2-3 pages)

**Purpose:** Empirically validate your method

**What to include:**
- Datasets and preprocessing
- Baselines (must include FedAwS)
- Implementation details (reproducibility)
- Main results table
- Ablation studies
- Hyperparameter sensitivity
- Communication/compute overhead comparison

**Common mistakes:** Missing baselines, no ablations, no error bars, unfair comparisons

**Reviewer expectations:** Rigorous, reproducible, statistically sound, improvements are meaningful

## 7. Discussion (0.5 page)

**Purpose:** Interpret results and connect to broader implications

**What to include:**
- When does your method help most? When does it not?
- Practical considerations for deployment
- Relationship between theory and empirical observations

**Common mistakes:** Just repeating results, over-claiming, ignoring failure cases

## 8. Limitations (0.25-0.5 page)

**Purpose:** Honestly discuss what your method cannot do

**What to include:**
- Scenarios where your method may underperform
- Assumptions that may not hold in practice
- Computational limitations
- Open questions

**Common mistakes:** Being too defensive, listing trivial limitations

**Reviewer expectations:** Honest, thoughtful self-assessment

## 9. Conclusion (0.25 page)

**Purpose:** Summarize contributions and impact

**What to include:**
- One-sentence problem summary
- One-sentence solution summary
- Key results
- Future work (1-2 sentences)

**Common mistakes:** Introducing new information, repeating abstract verbatim

## References

**What to include:**
- All works cited in the paper
- Proper formatting per venue style
- Consistent citation style

---

# 13. Publication Strategy Guide

## 13.1 Suitable Venues

| Venue Type | Examples | Why Suitable |
|---|---|---|
| **Top ML conferences** | ICML, NeurIPS, ICLR | Original paper was ICML; follow-up work fits here if novel enough |
| **FL-specific workshops** | FL@NeurIPS, FL@ICML | Lower bar for incremental improvements; good for early-stage ideas |
| **Privacy ML venues** | PPML Workshop, PriML | If extension focuses on formal privacy guarantees |
| **Systems ML venues** | MLSys, SysML | If extension focuses on communication efficiency or deployment |
| **Journals** | JMLR, IEEE TPAMI | For comprehensive extensions with extensive experiments |

## 13.2 Required Baseline Expectations

For a paper extending FedAwS, reviewers will expect comparison against:
1. **FedAwS** (the method from this paper)
2. **Vanilla FedAvg** (even though it fails, to show the problem exists)
3. **At least one other FL baseline** (e.g., FedProx, SCAFFOLD)
4. **Oracle method** (full label access, to show upper bound)
5. **Your method** with ablations (each component removed)

## 13.3 Experimental Rigor Level

| Venue | Expected Rigor |
|---|---|
| ICML/NeurIPS | 4+ datasets, 3+ baselines, error bars, ablations, theoretical guarantee, hyperparameter analysis |
| Workshop | 2+ datasets, 2+ baselines, preliminary results acceptable |
| Journal | Everything above + additional experiments, deeper analysis, reproducibility package |

## 13.4 Common Rejection Reasons

1. **"Insufficient novelty over FedAwS"** — Your modification is too minor (e.g., just changing the regularizer without new insights)
2. **"Missing baselines"** — Did not compare against FedAwS or other FL methods
3. **"Theoretical claims not supported"** — Proofs have errors or assumptions are too strong
4. **"Limited experiments"** — Only CIFAR-10, no large-scale experiments
5. **"No privacy analysis"** — For a privacy-motivated paper, formal privacy guarantees are expected
6. **"Unclear practical motivation"** — Why would someone use your method over standard FL?

## 13.5 Increment Needed for Acceptance

- **Top conference (ICML/NeurIPS/ICLR):** Must address a fundamental limitation of FedAwS with novel theory + comprehensive experiments (e.g., formal DP guarantee with accuracy analysis, or extension to multi-positive setting with new convergence theory)
- **Workshop:** Novel preliminary idea with promising initial results (e.g., adaptive hyperparameter tuning for FedAwS with results on 2 datasets)
- **Journal:** Comprehensive study combining theory + multiple extensions + extensive experiments (e.g., full communication-privacy-accuracy tradeoff analysis)

---

# 14. Researcher Quick Reference Tables

## 14.1 Key Terminology Table

| Term | Definition in This Paper |
|---|---|
| FedAwS | Federated Averaging with Spreadout; the proposed algorithm |
| Spreadout regularizer | Penalty that pushes class embeddings apart if they are closer than margin nu |
| Positive-only loss | Loss function using only `l_pos` (pulls instances toward their class) |
| Embedding collapse | Failure mode where all embeddings converge to one point |
| Scorer function | `f(x) = W * g_theta(x)` mapping inputs to per-class scores |
| Cosine contrastive loss | Contrastive loss using cosine distance with unit-norm embeddings |
| Stochastic negative mining | Optimization trick: only push apart each class from its k nearest neighbors |
| Class embedding `w_c` | Learned vector representing class c in the shared embedding space |
| Instance embedding `g_theta(x)` | Neural network output mapping input x into the shared embedding space |
| Oracle method | Baseline with access to both positive AND negative labels (upper bound on performance) |

## 14.2 Important Equations Summary

| Eq. # | Formula | Purpose |
|---|---|---|
| (1) | `f(x) = W * g_theta(x)` | Scorer function: computes class scores |
| (6) | `l_cl = alpha * d(g(x), w_y)^2 + beta * sum_{c!=y} [max(0, nu - d(g(x), w_c))]^2` | Contrastive loss (needs both positive and negative labels) |
| (7) | `reg_sp(W) = sum_{c} sum_{c'!=c} [max(0, nu - d(w_c, w_{c'}))]^2` | Spreadout regularizer (server-side) |
| (8) | `reg_sp_top(W) = sum_c sum_{y in N_k(c)} -d^2(w_c, w_y)` | Stochastic negative mining variant |
| (11) | `d_cos(u, u') = 1 - u^T * u'` | Cosine distance for unit-norm vectors |
| (13) | `l_ccl = (1 - s_y)^2 + sum_{c!=y} [max(0, nu - 1 + s_c)]^2` | Cosine contrastive loss (in terms of logits) |
| (18) | `l_sp = (1 - s_y)^2 + sum_{c!=y} [max(0, nu - 1 + w_y^T * w_c)]^2` | FedAwS equivalent loss |
| (23) | `l_pos = [max(0, 0.9 - g_theta(x)^T * w_y)]^2` | Positive squared hinge loss used in experiments |

## 14.3 Parameter Meaning Table

| Parameter | Symbol | Range | Role |
|---|---|---|---|
| Margin | `nu` | (1, 2) for theory; problem-dependent in practice | Minimum desired distance between class embeddings |
| Spreadout weight | `lambda` | 10 (default) | Controls strength of spreadout regularization |
| Mining depth | `k` | 10-100 (default 10) | Number of nearest neighbor classes in stochastic negative mining |
| Learning rate | `eta` | Problem-dependent | Gradient step size |
| Embedding dimension | `d` | 64 (CIFAR), 512 (extreme) | Dimensionality of shared embedding space |
| Number of classes | `C` | 10 to 670K | Also equals number of clients in simplified setting |
| Positive threshold | 0.9 | Fixed | Target cosine similarity for positive pairs |

## 14.4 Algorithm Flow Summary

```
INITIALIZATION:
    Server initializes theta_0, W_0

FOR each round t = 0, 1, ..., T-1:
    
    [SERVER -> CLIENTS]
    Send theta_t to all clients
    Send w_t_i (only own class embedding) to client i
    
    [CLIENT-SIDE (parallel for all clients)]
    Each client i:
        Compute positive-only loss: l_pos = [max(0, 0.9 - g(x)^T * w_i)]^2
        Update: (theta_{t,i}, w_{t,i}_i) -= eta * gradient(l_pos)
        Send updated (theta_{t,i}, w_{t,i}_i) back to server
    
    [SERVER-SIDE]
    Average theta: theta_{t+1} = (1/C) * sum_i theta_{t,i}
    Collect class embeddings: W_tilde = stack all w_{t,i}_i
    
    [SPREADOUT STEP - KEY INNOVATION]
    For each class c:
        Find k nearest classes N_k(c) by embedding distance
        Push c away from N_k(c): W_{t+1} -= lambda * eta * gradient(reg_sp)
    
RETURN theta_T, W_T
```

---

# 15. One-Page Master Summary Card

## Problem
In federated learning, each user has data from only one class (e.g., their own face images). Users cannot share data or access other classes' model parameters. Standard federated learning fails because training with only positive labels causes all class embeddings to collapse to a single point.

## Idea
Add a server-side "spreadout" regularizer that pushes class embeddings apart after each federated averaging round. This compensates for the missing negative labels that would normally prevent collapse.

## Method
**FedAwS (Federated Averaging with Spreadout):**
1. Server sends shared model + individual class embedding to each client
2. Clients train locally using positive-only loss (pull instances toward their class)
3. Server averages shared model updates
4. Server applies spreadout regularizer to push class embeddings apart (prevents collapse)
5. For large number of classes: use stochastic negative mining (only push apart k nearest classes)

## Results
- CIFAR-10/100: FedAwS matches oracle performance (92.4% vs 92.4% on CIFAR-10 with ResNet-32)
- AmazonCat (13K classes): FedAwS matches oracle (92.1% P@1)
- Amazon670K (670K classes): FedAwS achieves 33.1% vs oracle's 35.2%
- WikiLSHTC (325K classes): FedAwS achieves 37.2% vs oracle's 54.1% (significant gap)

## Weakness
- No formal privacy guarantees (only mentions compatibility with DP)
- Significant performance gap on datasets with sparse labels (WikiLSHTC)
- Hyperparameter k and lambda require tuning
- No communication cost analysis
- Theory assumes equal data per client

## Research Opportunity
1. Integrate differential privacy with formal accuracy-privacy tradeoff analysis
2. Adaptive spreadout that respects class hierarchy/semantics
3. Extension to multi-positive settings (users have data from a few classes)
4. Communication-efficient variants with theoretical guarantees
5. Federated fine-tuning of foundation models with positive-only labels

## Publishable Extension
**Strongest angle:** Combine FedAwS with formal differential privacy guarantees. Add calibrated noise to client updates and spreadout gradients. Prove that the method achieves (epsilon, delta)-DP while maintaining bounded accuracy loss relative to non-private FedAwS. Evaluate on the same benchmarks + real-world biometric datasets. This addresses the paper's biggest gap (claimed privacy motivation but no formal guarantee) and adds both theoretical and empirical contributions suitable for a top venue.
