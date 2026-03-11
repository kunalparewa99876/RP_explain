# Research Companion: Analyzing Federated Learning through an Adversarial Lens
**Paper:** Bhagoji, A. N., Chakraborty, S., Mittal, P., & Calo, S. (2019). Analyzing Federated Learning through an Adversarial Lens. *ICML 2019*, PMLR 97.

---

# 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Paper Title** | Analyzing Federated Learning through an Adversarial Lens |
| **Authors** | Arjun Nitin Bhagoji, Supriyo Chakraborty, Prateek Mittal, Seraphin Calo |
| **Venue / Year** | ICML 2019 |
| **Problem Domain** | Security & Privacy in Federated Learning / Adversarial ML |
| **Paper Type** | Algorithmic / Experimental ML |
| **Core Contribution** | Designed model poisoning attacks against federated learning that achieve targeted misclassification while maintaining convergence and evading two stealth detectors |
| **Key Idea** | A single malicious agent can "boost" its parameter update to override many honest agents, then disguise the attack using extra loss terms and alternating minimization |
| **Required Background** | Federated learning (FedAvg), stochastic gradient descent, neural network training, basic threat-model terminology |
| **Primary Baseline** | Standard FedAvg with weighted averaging aggregation; comparisons vs. Krum and coordinate-wise median |
| **Main Innovation Type** | New attack formulation (model poisoning + stealth + alternating minimization) |
| **Difficulty Level** | Intermediate (heavy experimentation, moderate math) |
| **Reproducibility Level** | High — code released at https://github.com/inspire-group/ModelPoisoning |

---

# 1. Research Context & Core Problem

## 1.1 Exact Problem Formulation

Federated learning trains a shared neural network across many participants (agents) without centralising raw data. Each agent trains locally and sends only parameter updates to a server that averages them. The authors ask: **can one single malicious participant silently corrupt the global model so that it deliberately misclassifies a specific input — while the model still works perfectly for everyone else?**

## 1.2 Why the Problem Exists

- Privacy-driven design hides each agent's local data and training process from the server.
- That same opacity gives a malicious agent freedom to send *any* parameter update it chooses.
- The server sees only numbers, not intent.

## 1.3 Historical / Theoretical Gap

| Prior Work Assumption | Gap Exposed by This Paper |
|---|---|
| Data poisoning: attacker injects bad training samples | In FL, data is never shared — data poisoning is therefore ineffective unless the update is also boosted |
| Byzantine-resilient aggregation: attacker drives convergence to a bad model overall | These defences were designed against disruption, not against *targeted stealth* poisoning that keeps the model accurate |
| Adversarial examples: modify the input at inference time | Here the input is untouched; the model itself is corrupted during training |

## 1.4 Contribution Category
- **Algorithmic** — new attack formulation (explicit boosting + stealthy objective + alternating minimisation)
- **Empirical insight** — Byzantine-resilient methods are not robust to targeted poisoning; interpretability tools cannot detect a poisoned model

---

### Why This Paper Matters

It demonstrates that the privacy guarantee of federated learning (hiding data) is itself the attack surface. Any FL deployment that only aggregates updates without deeper inspection is vulnerable today.

---

### Remaining Open Problems

1. No theoretically-grounded detection method for targeted model poisoning exists.
2. Attacks beyond a single malicious agent (collusion) are not studied here.
3. Non-i.i.d. data distributions may make attacks harder or easier — not fully explored.
4. Defences that reason about *weight distributions* are suggested but not designed.
5. Attacks on large modern architectures (Transformers, LLMs under federated fine-tuning) are unexplored.
6. The fragility of interpretability tools is observed but not fully explained theoretically.

---

# 2. Minimum Background Concepts

### 2.1 Federated Learning (FedAvg)
- **What it is:** A training protocol where K agents each hold private data; a server iteratively selects a subset, asks them to train locally, collects their weight updates, and averages them.
- **Role in paper:** The training protocol being attacked.
- **Why needed:** Defines the rules the attacker must work within.

### 2.2 Model Poisoning vs. Data Poisoning
- **Data poisoning:** Injects corrupted training samples. In FL this requires the malicious agent to rely only on its own shard — the paper shows this alone fails.
- **Model poisoning:** The malicious agent directly crafts its submitted parameter update to implant a false behaviour.
- **Role in paper:** The attacking category; the paper shows model poisoning strictly subsumes data poisoning in FL.

### 2.3 Targeted Misclassification
- The attacker picks specific inputs and a false label. The goal is that the global model maps those inputs to the false label — not just that the model degrades overall.
- **Role in paper:** The adversarial objective being optimised.

### 2.4 Boosting (in this context)
- After a malicious agent runs its local optimisation and gets an update vector, it multiplies that vector by a large factor λ before sending it.
- **Why needed:** The honest agents' updates collectively drown out any single small update; boosting compensates.

### 2.5 Byzantine-Resilient Aggregation (Krum, Coordinate-wise Median)
- **Krum:** Picks the single update whose sum of distances to its nearest neighbours is smallest.
- **Coordinate-wise median:** Takes the statistical median of each weight dimension independently.
- **Role in paper:** Proposed defences being tested and shown to be insufficient.

### 2.6 Cross-Entropy Loss
- The standard loss for classification that the paper uses both for poisoning and for stealth (honest training term).
- Required as background because all adversarial objectives are expressed through it.

### 2.7 Interpretability Techniques (LRP, Guided Backprop, Integrated Gradients, SmoothGrad)
- Tools that try to explain which input pixels drove a neural network decision.
- **Role in paper:** Side study showing these tools cannot distinguish benign from poisoned models.

---

# 3. Mathematical / Theoretical Understanding Layer

## 3.1 Federated Aggregation Update Rule

$$w_G^{t+1} = w_G^t + \sum_{i \in [k]} \alpha_i \delta_i^{t+1}$$

| Symbol | Meaning |
|---|---|
| $w_G^t$ | Global model weights at round $t$ |
| $\delta_i^{t+1}$ | Weight update sent by agent $i$ in round $t+1$ |
| $\alpha_i = l_i / l$ | Agent $i$'s fraction of total training data |
| $k$ | Number of agents selected in this round |

**Intuition:** The new global model is the old one shifted by a weighted average of all agents' updates. The malicious agent controls only its own $\delta_m$.

---

## 3.2 Targeted Model Poisoning Objective (Eq. 1)

$$\min_{\delta_m^t} \sum_{i=1}^{r} \mathbf{1}\left[f(x_i; \hat{w}_G^t) \neq \tau_i\right]$$

- **What it does:** Counts how many of the $r$ target inputs are *not* yet misclassified as the desired class $\tau_i$.
- **Problem:** This is a hard combinatorial (0-1 loss) function — gradients cannot be computed through it.
- **Solution:** Replace the indicator with cross-entropy loss and use automatic differentiation (SGD/Adam).

| Symbol | Meaning |
|---|---|
| $x_i$ | Input the attacker wants to misclassify |
| $\tau_i$ | Desired false target class |
| $\hat{w}_G^t$ | Malicious agent's *estimate* of the global model at round $t$ |
| $r$ | Number of target examples |

---

## 3.3 Explicit Boosting Mechanism

The malicious agent computes a local update $\tilde{\delta}_m^t$ by running an optimiser on the adversarial objective, then:

$$\delta_m^t = \lambda \cdot \tilde{\delta}_m^t, \quad \text{where} \quad \lambda = \frac{1}{\alpha_m} = K$$

- **Intuition:** If the malicious agent's update is scaled down by $\alpha_m = 1/K$ during averaging, multiplying by $K$ beforehand makes it as though the malicious agent's update replaces the global model exactly.
- **Assumption:** The malicious agent ignores other agents' updates in its estimate of $\hat{w}_G^t$.

---

## 3.4 Full Stealthy Objective (Eq. 3)

$$\min_{\delta_m^t} \; L(\{x_i, \tau_i\}_{i=1}^r, \hat{w}_G^t) + L(D_m, w_m^t) + \rho \|\delta_m^t - \bar{\delta}_{ben}^{t-1}\|_2$$

| Term | Purpose |
|---|---|
| $L(\{x_i, \tau_i\}, \hat{w}_G^t)$ | Achieve targeted misclassification |
| $L(D_m, w_m^t)$ | Maintain high accuracy on the agent's own honest data → passes accuracy check |
| $\rho \|\delta_m^t - \bar{\delta}_{ben}^{t-1}\|_2$ | Keep the malicious update close to the average benign update from the previous round → passes distance check |
| $\rho$ | Hyperparameter balancing stealth vs. poisoning strength |

---

## 3.5 Accuracy Stealth Condition

$$\sum_{\{x_j, y_j\} \in D_{test}} \mathbf{1}[f(x_j; w_m^t) = y_j] - \mathbf{1}[f(x_j; w_{G \setminus m}^t) = y_j] < \gamma_t$$

- **Meaning:** The model produced by the malicious update alone must perform within $\gamma_t$ of the model produced by all other agents. If it does not, the server flags it.
- **Practical implication:** The malicious agent must train on honest data too — hiding its attack inside normal-looking performance.

---

## 3.6 Weight Distance Stealth Condition

For the malicious update $\delta_m^t$, the range of pairwise L2 distances to all honest updates is $R_m$. The condition:

$$\max\{|R_m^u - R_{min,[k]\setminus m}^l|, |R_m^l - R_{max,[k]\setminus m}^u|\} < \kappa_t$$

- **Meaning:** The spread of distances from the malicious update must look as normal as the spread seen between any two honest agents.
- **Why hard to satisfy:** The boosted update is orders of magnitude larger than honest ones in naive attacks — this condition forces the attacker to reduce raw magnitude while still achieving its goal.

---

### Mathematical Insight Box

> **Key idea to remember:** Federated averaging is a linear aggregation. Any single agent's contribution is multiplicatively suppressed by $\alpha_i$. Boosting by $1/\alpha_i$ exactly cancels this suppression. Stealth constraints then shape the attack to *look* like a normal update while hiding the poisoning inside.

---

# 4. Proposed Method / Framework

## 4.1 Overall Pipeline

```
[Round t begins]
  │
  ├─ Server selects k agents (including malicious agent m)
  │
  ├─ Each HONEST agent i:
  │     trains on D_i starting from w_G^t
  │     sends δ_i^{t+1} = w_i^{t+1} - w_G^t
  │
  └─ MALICIOUS agent m:
        [Choose attack strategy → see 4.2–4.4]
        sends boosted/stealthy δ_m^{t+1}
  │
  Server aggregates: w_G^{t+1} = w_G^t + Σ α_i δ_i^{t+1}
  │
[Round t+1 begins]
```

---

## 4.2 Attack Strategy 1: Targeted Model Poisoning (Explicit Boosting)

**Steps:**
1. Start from the current global model $w_G^{t-1}$.
2. Run $E_m = 5$ optimisation steps on the adversarial cross-entropy loss targeting $\{x_i, \tau_i\}$.
3. Compute raw update $\tilde{\delta}_m^t = \tilde{w}_m^t - w_G^{t-1}$.
4. Boost: send $\delta_m^t = K \cdot \tilde{\delta}_m^t$ (where $K$ is the total number of agents).

**Why authors did this:** Boosting exactly counteracts the averaging dilution.

**Weakness:** The boosted update looks nothing like an honest update — extreme magnitude, sparse distribution, detectable immediately by either stealth metric.

**Research seed:** Can we learn the optimal boosting factor analytically given only partial information about other agents?

---

## 4.3 Attack Strategy 2: Stealthy Model Poisoning

**Steps:**
1. Use the same adversarial objective as Strategy 1.
2. Add an honest training loss term over the malicious agent's own data shard.
3. Add an L2 penalty pulling $\delta_m^t$ toward the average benign update from the *previous* round.
4. Only the adversarial component of the loss is boosted by $\lambda$; the honest terms are not.

**Why authors did this:** Forces the update to simultaneously poison the model AND look like a normal update.

**Weakness:** Because both objectives compete, neither is fully satisfied in every round; the attack sometimes fails stealth checks.

**Research seed:** Use adversarial training (minimax) to jointly solve stealth and poisoning more reliably.

---

## 4.4 Attack Strategy 3: Alternating Minimisation

**Steps (per epoch $i$ of round $t$):**
1. Minimise adversarial objective from $w_m^{i-1,t}$ → get $\tilde{\delta}_m^{i,t}$.
2. Boost: $\tilde{w}_m^{i,t} = w_m^{i-1,t} + \lambda \tilde{\delta}_m^{i,t}$.
3. Minimise *stealth objective only* starting from $\tilde{w}_m^{i,t}$ → get $w_m^{i,t}$.
4. Repeat for multiple epochs; optimise stealth for more steps than poisoning.

**Why authors did this:** Decoupling the two objectives gives independent control over how hard to push each one per epoch.

**Weakness:** More computation required on the malicious agent; assumes the agent has substantial local compute.

**Research seed:** Formulate as a Pareto-optimisation problem to find the exact trade-off frontier between stealth and poisoning effectiveness.

---

## 4.5 Attack Against Byzantine-Resilient Aggregation

- **Against Krum:** Use alternating minimisation with a small $\lambda=2$. The malicious update needs only to have the lowest "sum of distances to nearest neighbours" score — not to be close to honest updates absolutely.
- **Against Coordinate-wise Median:** Use plain targeted model poisoning with $\lambda=1$ (no explicit boosting). Because the median uses information from *all* agents, a single large outlier in each dimension still influences the median, allowing poisoning with high confidence.

**Key design insight:** Krum is fooled by *relative* proximity, not absolute scale. Coordinate-wise median distributes influence across all agents, so even small shifts accumulate.

---

## 4.6 Better Estimation of Other Agents' Updates

Rather than ignoring other agents, the malicious agent estimates their cumulative contribution:

$$\hat{\delta}_{[k]\setminus m}^t = \frac{w_G^t - w_G^{t'} - \delta_m^{t'}}{t - t'}$$

where $t'$ is the last round the malicious agent participated in. Attack starts from $w_G^{t-1} + \hat{\delta}_{[k]\setminus m}^t$ instead of just $w_G^{t-1}$.

**Effect:** Faster convergence to target misclassification in the early rounds (see Table 1 in paper).

---

# 5. Experimental Setup / Evaluation Design

## 5.1 Datasets

| Dataset | Task | Model | Clean Accuracy |
|---|---|---|---|
| Fashion-MNIST | Image classification (10 classes) | 3-layer CNN + dropout | 91.7% |
| UCI Adult Census | Income prediction (binary) | Fully connected NN | 84.8% |

**Why two datasets:** Shows attacks generalise across image vs. tabular data and CNNs vs. fully connected networks.

## 5.2 Federated Setup

| Parameter | Value |
|---|---|
| Number of agents K | 10 or 100 |
| Agents per round | All 10 (K=10) or 10% randomly (K=100) |
| Data distribution | i.i.d. (hardest case for the attacker — most honest updates are similar and easy to discriminate from anomalous ones) |
| Training stops when | Target accuracy reached (91% / 84%) or max rounds (40 / 50) |

## 5.3 Target Setup (Primary)
- Fashion-MNIST: misclassify class 5 (sandal) as class 7 (sneaker), $r=1$ example.
- Adult Census: misclassify class 0 as class 1, $r=1$ example.

## 5.4 Metrics

| Metric | What It Measures |
|---|---|
| Confidence on malicious objective | How certain the global model is in the *wrong* label for target inputs |
| Validation accuracy of global model | Whether the honest performance is preserved (convergence check) |
| Stealth pass rate | How many rounds the malicious update avoids the accuracy and distance detectors |
| Range of L2 pairwise distances | Whether the malicious update "looks" statistically normal |

## 5.5 Hyperparameters
- Boosting factor $\lambda = K = 10$ for standard attacks.
- $\rho = 1\text{e-4}$ for distance penalty term in stealthy attack.
- Accuracy stealth threshold $\gamma_t = 10\%$.
- Optimiser: Adam for malicious objective; SGD for honest training terms.

---

### Experimental Reliability Analysis

| Aspect | Assessment |
|---|---|
| Dataset diversity | Limited — two small-scale datasets; no large-scale vision (ImageNet) or NLP |
| i.i.d. assumption | Explicitly chosen as the *hardest* case for the attacker — results are therefore lower bounds on attack success in the real world |
| Single vs. colluding agents | Only tested with 1 malicious agent — multi-agent collusion not explored |
| Reproducibility | Code released; results likely reproducible |
| Statistical reporting | No confidence intervals or standard deviations reported across seeds |
| Byzantine aggregation tests | Limited to 2 mechanisms; others (DRACO, FLTrust) not tested |

---

# 6. Results & Findings Interpretation

## 6.1 Main Outcomes

| Attack | Convergence Maintained? | Targeted Poisoning Achieved? | Stealth Passed? |
|---|---|---|---|
| Explicit boosting | Yes | Yes (100% confidence by round 3) | No — detected every round |
| Stealthy poisoning | Yes | Yes | Partial — passes 10 out of 15 rounds |
| Alternating minimisation | Yes | Yes (high confidence by round 2) | Yes — almost every round |
| Against Krum | Yes | Yes | Partial |
| Against Coord. Median | Yes | Yes (0.9 confidence, 11/14 rounds) | — |

## 6.2 Data Poisoning Comparison

Even adding 1,000 corrupted copies to the malicious agent's local training set *completely fails* to cause targeted misclassification at the global model. Model poisoning succeeds; data poisoning does not. Reason: the scaling factor $\alpha_m$ dilutes a data-poisoning-derived update the same way it dilutes any other update — unless the update is independently boosted.

## 6.3 Interpretability Fragility

Using LRP, Guided Backprop, DeConvNet, Integrated Gradients, SmoothGrad, and Gradient Saliency Maps, the feature-importance visualisations for a poisoned global model and a benign global model on the *same input* are nearly identical to the human eye. The poisoning effect exists only for the specific targeted input, leaving all other representations essentially unchanged.

## 6.4 Unexpected Observations

- Coordinate-wise median is broken *without* any boosting ($\lambda=1$), which is qualitatively surprising — even a unit-scale malicious update is sufficient because the median operator always "hears" every agent.
- The weight update histogram for the basic boosted attack is noticeably sparser than honest updates — a qualitative detection signal the authors notice but the server does not automatically exploit.

---

### Publishability Strength Check

| Result | Strength |
|---|---|
| 100% targeted poisoning with 1 malicious agent | Strong — clean, reproducible, released code |
| Alternating minimisation achieving near-complete stealth | Strong — concrete improvement with clear mechanism |
| Byzantine-resilient mechanisms fail against targeted poisoning | Strong — clear theoretical explanation provided |
| Interpretability tools cannot detect poisoning | Moderate — observation only, no quantitative metric |
| Data poisoning comparison | Strong — controlled experiment, 1000 samples tested |

---

# 7. Strengths – Weaknesses – Assumptions

## Technical Strengths

| Strength | Detail |
|---|---|
| Formulation is principled | Attack directly derived from the FL aggregation formula |
| Stealth is formally defined | Two concrete, checkable metrics — not hand-waving |
| Alternating minimisation is elegant | Decouples two conflicting objectives with independent step counts |
| Code released | High reproducibility |
| Covers multiple aggregation mechanisms | Goes beyond FedAvg to Krum and coordinate-wise median |
| Comparison with prior attacks | Explicit data poisoning comparison with quantitative results |

## Explicit Weaknesses

| Weakness | Detail |
|---|---|
| Only i.i.d. setting | Real FL is almost always non-i.i.d.; results may not transfer |
| Only 1 malicious agent | Multiple colluding agents not studied |
| Small-scale datasets | Fashion-MNIST and Adult Census — not representative of real FL deployments |
| No adaptive defence | Authors do not propose or test any counter-strategy |
| Stealth metrics are not complete | Only two server-side checks; more sophisticated anomaly detection not tested |
| No theoretical guarantee | Attack success proven empirically, not analytically |

## Hidden Assumptions

| Assumption | Impact if Violated |
|---|---|
| Server performs no per-update inspection of raw data | If secure aggregation is used, server sees only the sum — some attacks change |
| Malicious agent is selected in early rounds | If selected late (near convergence), attack windows shrink |
| Non-colluding single adversary | With multiple colluding agents, boosting may not be needed at all |
| i.i.d. training data | Non-i.i.d. makes benign updates more variable, potentially hiding the malicious one more easily |
| Honest agents do not detect anomalous global model behaviour | Assumes no peer-to-peer verification |

---

# 8. Weakness → Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| Only i.i.d. data studied | Non-i.i.d. adds complexity to the experimental setup | Attack and defence performance under heterogeneous data distributions | Dirichlet-distributed data assignment; FedNova or SCAFFOLD baselines |
| Single malicious agent | Colluding agents were out of scope | Coordinated multi-agent model poisoning with communication constraints | Game-theoretic multi-agent attack formulation |
| No adaptive defence proposed | Paper is attack-focused | Design server-side detection using learned anomaly detection on update distributions | Variational autoencoders or normalising flows trained on benign update statistics |
| Byzantine aggregation broken | Aggregation rules designed for convergence disruption, not targeted stealth | New aggregation with explicit stealth-resistance guarantees | Certified-radius aggregation; randomised smoothing over update space |
| Interpretability tools fail | Tools trained on benign models only | Poisoning-aware interpretability verification | Consistency testing of explanations across model variants |
| No theoretical guarantee | Empirical-only paper | Formal convergence + poisoning bounds under adversarial FL | PAC-learning or information-theoretic analysis of update influence |
| Small datasets | Benchmarking choice | Scaling study on CIFAR-10, ImageNet, and NLP tasks | PyTorch Flower or FedML framework |

---

# 9. Novel Contribution Extraction

## Contribution Templates for New Papers

**Template 1 (defence):**
> "We propose a learned update anomaly detector that improves detection of stealthy model poisoning attacks by training a generative model on benign update distributions, achieving detection rates above X% while maintaining less than Y% reduction in honest accuracy."

**Template 2 (heterogeneous setting):**
> "We extend targeted model poisoning to the non-i.i.d. federated setting and show that data heterogeneity increases attack surface by W% due to increased inter-agent update variance that masks boosted malicious updates."

**Template 3 (certified defence):**
> "We propose a certified aggregation mechanism that provides a formal upper bound on any single agent's influence on the global model, reducing targeted misclassification confidence from X% to below Y% regardless of the attack strategy used."

**Template 4 (multi-agent collusion):**
> "We design a coordinated multi-agent model poisoning strategy that achieves targeted backdoor injection in Z rounds without requiring boosting, defeating both accuracy-based and distance-based stealth detectors."

**Template 5 (LLM extensions):**
> "We demonstrate that model poisoning generalises to federated fine-tuning of large language models where a single malicious participant can inject targeted hallucination triggers while maintaining benchmark performance, using alternating minimisation adapted for parameter-efficient fine-tuning."

---

# 10. Future Research Expansion Map

## Author-Suggested Future Work
- Stronger server-side detection strategies (mentioned in conclusion).
- Notions of distance between weight distributions as defensive tools.

## Missing Directions in This Paper
- Non-i.i.d. partition effects on attack success.
- Multi-agent collusion analysis.
- Theoretical (non-empirical) attack success bounds.
- Defence mechanism design and evaluation.
- Attacks on models beyond classification (regression, generation).

## Modern Extensions
- **Federated fine-tuning of LLMs:** PEFT methods (LoRA, prefix tuning) change the update space — do boosting attacks still work?
- **Secure aggregation:** Cryptographic FL where the server only sees the aggregate — how does this affect attack strategy?
- **Asynchronous FL:** Agents submit updates at different times; timing-based attacks become possible.
- **Cross-silo vs. cross-device FL:** Fewer, more powerful participants vs. many mobile devices change the aggregation dynamics.

## Cross-Domain Combinations
- **Differential privacy + model poisoning:** DP noise clipping changes the effective scaling of updates — attack strategies must adapt.
- **Federated anomaly detection systems:** What if the shared model is an anomaly detector itself? Poisoning it could suppress legitimate alerts.
- **Healthcare FL:** Patient data privacy is paramount; targeted poisoning of a medical diagnostic model is a critical safety concern.

---

# 11. How to Write a NEW Paper From This Work

## Reusable Elements

| Element | How to Reuse |
|---|---|
| Adversarial objective formulation (Eq. 1-3) | Use as the attack baseline in any FL security paper |
| Alternating minimisation strategy | Applicable whenever two conflicting objectives must both be satisfied |
| Two stealth metrics | Use as standard evaluation protocol in FL attack papers |
| Data poisoning vs. model poisoning comparison | Standard comparison expected by reviewers in this area |
| Byzantine aggregation evaluation | Any FL security paper must now test against Krum and coordinate-wise median at minimum |

## What MUST NOT Be Copied
- The specific attack algorithm without citing the paper.
- The experimental results or figures.
- The exact objective function wording.
- The boosting derivation ($\lambda = 1/\alpha_m$) without attribution.

## How to Design a Novel Extension

**Option A (Fix a weakness):**
1. Choose non-i.i.d. data setting.
2. Redesign the distance stealth term — benign updates are now more spread out; what does "normal" look like?
3. Evaluate whether alternating minimisation still provides stealth.

**Option B (Build a defence):**
1. Observe that the malicious update has a bimodal histogram (sparse, small-range).
2. Train a classifier on honest update histograms.
3. Flag updates whose histogram diverges from the learned distribution.
4. Show that this breaks all three attack variants from this paper.

**Option C (Scale up):**
1. Implement attacks on CIFAR-10 with ResNet; Adult Census → credit scoring.
2. Measure how increased model dimensionality affects stealth (larger models may be *easier* to hide attacks in).

## Minimum Publishable Contribution Checklist

- [ ] Clear problem statement that extends or fixes a gap in this paper
- [ ] New attack variant OR new detection/defence with formal or empirical guarantees
- [ ] Evaluation on at least 2 datasets beyond Fashion-MNIST
- [ ] Baseline includes all three attack variants from this paper
- [ ] Byzantine aggregation mechanisms included in evaluation
- [ ] Ablation study on key hyperparameters ($\lambda$, $\rho$, $\gamma$)
- [ ] Statistical significance tested (multiple seeds)
- [ ] Comparison to at least one post-2019 concurrent attack/defence paper

---

# 12. Complete Paper Writing Template

## Abstract
**Purpose:** Motivate the problem, state the method, state the key result.
**What to include:** 1 sentence on FL privacy model → 1 sentence on identified threat → 1 sentence on proposed method → 1-2 sentences on key results → 1 sentence on implications.
**Common mistakes:** Too vague on the method; omitting quantitative results.
**Reviewer expectation:** Should be self-contained and cite a specific, concrete result (e.g., "100% misclassification confidence").

---

## Introduction
**Purpose:** Expand abstract, motivate problem deeply, position against related work, list contributions as bullet points.
**What to include:** Federated learning background (2-3 sentences) → Why the privacy guarantee creates vulnerability → Threat model overview → Key contributions (numbered list) → Paper structure roadmap.
**Common mistakes:** Contributions buried in paragraphs, not extractable as a bulleted list.
**Reviewer expectation:** Contributions must be clearly enumerable and technically specific.

---

## Related Work
**Purpose:** Show awareness of the field and differentiate clearly.
**What to include:** Data poisoning attacks → Byzantine-resilient aggregation → Adversarial examples → Model poisoning (concurrent work: Bagdasaryan et al., 2018) → Interpretability methods.
**Common mistakes:** Describing, not differentiating — every related work entry must state how your work differs.
**Reviewer expectation:** A clear differentiation table or explicit "unlike X, we Y" statements.

---

## Threat Model & Problem Formulation
**Purpose:** Define exactly who the attacker is, what they know, and what they want.
**What to include:** Attacker capabilities (number of agents controlled, data access, knowledge of other agents) → Attack goal (targeted misclassification with confidence) → Constraints (convergence must be maintained) → Stealth requirements.
**Common mistakes:** Leaving attacker assumptions implicit — reviewers will probe these.
**Reviewer expectation:** Every assumption must be justified and quantified.

---

## Method
**Purpose:** Present the attack algorithm step by step.
**What to include:** Adversarial objective formulation → Explicit boosting derivation → Stealthy objective with all loss terms → Alternating minimisation pseudocode → Estimation strategy.
**Common mistakes:** Skipping the motivation for design choices; not explaining why alternatives (implicit boosting) were rejected.
**Reviewer expectation:** Method should be reproducible from the section alone (with supplementary details).

---

## Theory (if applicable)
**Purpose:** Provide formal analysis of why the attack works.
**What to include:** For this paper style — a proposition showing boosting cancels averaging exactly; conditions under which stealth metrics are satisfiable simultaneously.
**Common mistakes:** Providing only informal arguments where formal proofs are expected by theory-focused venues.
**Reviewer expectation:** Security conferences (CCS, S&P, USENIX) expect formal threat models; ML conferences (ICML, NeurIPS) may accept empirical evidence with good justification.

---

## Experiments
**Purpose:** Empirically validate all claims in the method section.
**What to include:** Dataset and model details → Federated setup (K, rounds, data split) → Baseline attacks → Evaluation metrics → Results tables and figures → Ablation study.
**Common mistakes:** Only showing best-case results; not testing limits of the attack (small K, large K, near-convergence timing).
**Reviewer expectation:** Results must match claims in the abstract exactly; ablation must show each component contributes.

---

## Discussion
**Purpose:** Interpret results beyond the numbers.
**What to include:** Why Byzantine-resilient mechanisms fail (mechanistic explanation) → Why interpretability tools fail → What the results imply for FL deployment security.
**Common mistakes:** Repeating results instead of interpreting them.
**Reviewer expectation:** Show genuine insight — why do results behave as they do?

---

## Limitations
**Purpose:** Show intellectual honesty; pre-empt reviewer criticism.
**What to include:** i.i.d. restriction → Single-agent restriction → Small-scale datasets → No defence proposed.
**Common mistakes:** Being too vague ("future work may explore...") or omitting limitations entirely.
**Reviewer expectation:** Limitations should be specific and honest; suggests the authors understand boundaries of their contribution.

---

## Conclusion
**Purpose:** Summarise in 1 paragraph; point to future directions.
**What to include:** Problem re-stated in one sentence → Key finding summarised → Most important future direction.
**Common mistakes:** Introducing new ideas in the conclusion.
**Reviewer expectation:** Brief and forward-looking.

---

## References
- All citations in the introduction and related work sections must appear here.
- Use consistent citation style (BibTeX; PMLR format for ICML papers).
- Include arXiv versions for recent work.

---

# 13. Publication Strategy Guide

## Suitable Venues

| Type | Examples | Why Suitable |
|---|---|---|
| Top ML conferences | ICML, NeurIPS, ICLR | Experimental rigour, new attack/defence algorithms |
| Security conferences | IEEE S&P, USENIX Security, CCS, NDSS | Formal threat models, practical relevance |
| Privacy-focused workshops | FL-NeurIPS Workshop, PrivateNLP | Early-stage results, new directions |
| Journals | JMLR, IEEE TIFS, TDSC | Extended version with full proofs |

## Required Baseline Expectations (2024+)

Any follow-up paper must include:
1. All three attack variants from this paper as baselines.
2. Evaluation on at least Krum and coordinate-wise median.
3. Testing under non-i.i.d. data split (Dirichlet $\alpha \in \{0.1, 0.5\}$ is standard).
4. Comparison to Bagdasaryan et al. 2018 (concurrent model replacement paper).
5. At least one post-2020 defence (e.g., FLTrust, FLAME, CRFL).

## Common Rejection Reasons

| Reason | How to Avoid |
|---|---|
| "Only tested on Fashion-MNIST" | Add CIFAR-10 and at least one NLP/tabular dataset |
| "i.i.d. assumption is unrealistic" | Always provide non-i.i.d. experiments |
| "No comparison to recent defences" | Survey post-2019 FL security literature thoroughly |
| "Statistical significance not shown" | Report mean ± std over 5+ seeds |
| "Threat model is over-privileged" | Justify every assumed capability with realistic scenario |
| "No defence or mitigation proposed" | Propose even a simple counter-measure and evaluate it |

## Experimental Rigor Level Needed
- Minimum 3 datasets
- Minimum 5 random seeds per experiment
- Ablation of every hyperparameter ($\lambda$, $\rho$, $\gamma$, number of malicious agents)
- Tests at multiple values of K (10, 100, 1000 if feasible)

---

# 14. Researcher Quick Reference Tables

## Key Terminology Table

| Term | Simple Definition |
|---|---|
| Model poisoning | Sending a deliberately manipulated parameter update to corrupt the global model |
| Explicit boosting | Multiplying the malicious update by factor K to overcome averaging dilution |
| Stealth metric | A server-side check (accuracy or distance) that the adversary must bypass |
| Alternating minimisation | Switch between optimising for the attack goal and optimising for stealth in alternating epochs |
| Krum | Aggregation rule that selects the single update closest to its nearest neighbours |
| Coordinate-wise median | Aggregation rule that takes the median value of each model weight independently |
| Auxiliary data $D_{aux}$ | The specific inputs the attacker wants to misclassify |
| Byzantine adversary | An agent that can send any arbitrary update |

## Important Equations Summary

| Equation | What It Does |
|---|---|
| $w_G^{t+1} = w_G^t + \sum \alpha_i \delta_i^{t+1}$ | Standard FedAvg aggregation |
| $\delta_m^t = K \cdot \tilde{\delta}_m^t$ | Explicit boosting with factor K |
| Eq. 3 (stealthy objective) | Combines poisoning loss + honest training loss + distance penalty |
| $\hat{\delta}_{[k]\setminus m}^t = (w_G^t - w_G^{t'} - \delta_m^{t'}) / (t - t')$ | Previous-step estimation of other agents' cumulative update |

## Parameter Meaning Table

| Parameter | Meaning | Typical Value |
|---|---|---|
| K | Total number of agents | 10, 100 |
| k | Agents selected per round | K (when K=10), K/10 (when K=100) |
| $\alpha_m$ | Malicious agent's weight in aggregation | $1/K$ |
| $\lambda$ | Boosting factor | K (standard), 2 (for Krum), 1 (for coord. median) |
| $\rho$ | Distance penalty coefficient | $1\text{e-4}$ |
| $\gamma_t$ | Accuracy detection threshold | 10% |
| $E_m$ | Optimisation epochs for malicious agent | 5 |
| r | Number of targeted examples | 1 (primary), 10 (supplementary) |

## Algorithm Flow Summary

```
TARGETED MODEL POISONING:
  local_opt(adversarial_loss, E_m steps) → raw_update
  send λ × raw_update  [λ = K]

STEALTHY MODEL POISONING:
  local_opt(adversarial_loss + honest_loss + distance_penalty, combined) → update
  boost only adversarial component by λ

ALTERNATING MINIMISATION:
  for each epoch:
    step 1: minimise adversarial_loss   → temp_weights
    step 2: boost temp_weights by λ
    step 3: minimise stealth_loss only  → final_weights for this epoch
  send final_weights

ESTIMATION IMPROVEMENT:
  estimate other_agents_update from last observed global model shift
  start optimisation from w_G + estimated_other_update
```

---

# 15. One-Page Master Summary Card

| Category | Content |
|---|---|
| **Problem** | A single malicious agent in federated learning can poison the global model to misclassify specific inputs, while the model remains accurate for all other inputs. |
| **Idea** | The FL aggregation formula is linear — boosting one agent's update by the number of agents exactly cancels the averaging step. Add stealth constraints to hide the attack. |
| **Method (3 Stages)** | (1) Explicit boosting for raw attack; (2) Stealthy objective adding honest+distance terms; (3) Alternating minimisation decoupling the two goals for full stealth. |
| **Key Results** | 100% targeted misclassification confidence with 1 malicious agent; alternating minimisation avoids detection in almost all rounds; Krum and coordinate-wise median are both defeated; data poisoning with 1000 samples completely fails in the same setting; interpretability tools cannot distinguish poisoned from benign models. |
| **Weakness** | Only i.i.d. setting; only 1 malicious agent; small datasets; no defence proposed; no statistical significance reporting. |
| **Research Opportunity** | (1) Non-i.i.d. attacks and defences; (2) Multi-agent collusion strategies; (3) Learned anomaly detection on weight distributions; (4) Certified aggregation with formal influence bounds; (5) Scaling to federated LLM fine-tuning. |
| **Publishable Extension** | Design a server-side defence using a normalising flow trained on benign update distributions, evaluate against all three attack variants on CIFAR-10 under non-i.i.d. splits with 5 random seeds, and provide a formal detection bound. Submit to USENIX Security or IEEE S&P. |
