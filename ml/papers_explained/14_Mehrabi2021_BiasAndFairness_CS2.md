# Research Companion: A Survey on Bias and Fairness in Machine Learning
**Mehrabi, Morstatter, Saxena, Lerman & Galstyan — ACM Computing Surveys, 2021**

---

# 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Problem Domain** | Bias and fairness in AI/ML systems |
| **Paper Type** | Survey / Conceptual |
| **Core Contribution** | Unified taxonomy of bias types, formal fairness definitions, and domain-specific mitigation methods across ML/NLP/deep learning |
| **Key Idea** | Bias enters ML systems through data, algorithms, and user interactions in a feedback loop; fairness requires both detecting these biases and choosing the right fairness definition for the context |
| **Required Background** | Basic ML (classification, regression), probability, basic NLP concepts |
| **Primary Baseline** | COMPAS risk assessment system (used as the canonical real-world bias example throughout) |
| **Main Innovation Type** | Systematic synthesis and taxonomy (empirical survey) |
| **Difficulty Level** | Beginner–Intermediate (conceptual; no heavy math) |
| **Reproducibility Level** | High (surveys existing reproducible work; includes toolkits like AIF360 and Aequitas) |

---

# 1. Research Context & Core Problem

## Exact Problem Formulation
Machine learning algorithms make high-stakes decisions (bail, hiring, loans, medical diagnosis) that affect millions of people. These decisions are often **unfair** — they systematically disadvantage certain groups based on race, gender, age, or other protected attributes. The problem is threefold: (1) where does this unfairness come from, (2) how do we define and measure fairness mathematically, and (3) how do we build systems that are actually fair?

## Why the Problem Exists
- ML models learn from historical data that already reflects human and societal biases
- Algorithms can amplify small biases into large discriminatory outcomes
- User interaction with biased systems generates more biased training data (a feedback loop)
- There is no single agreed-upon definition of "fair" — different stakeholders prefer different definitions
- Legal frameworks (Fair Housing Act, Equal Credit Opportunity Act) prohibit discrimination but do not specify how to achieve it technically

## Historical and Theoretical Gap
Prior to surveys like this one, bias and fairness research was **fragmented** across computer science, law, social science, and philosophy. Researchers in classification fairness, NLP fairness, and deep learning fairness were working largely in isolation. No unified reference existed that connected:
- The sources of bias (data vs. algorithm vs. user)
- The mathematical definitions of fairness
- The domain-specific mitigation methods

## Limitations of Previous Approaches
- **Fairness toolkits** existed (AIF360, Aequitas) but lacked a unified conceptual framework
- **Individual papers** addressed one type of bias or one fairness definition
- **No taxonomy** organized bias types systematically within the data–algorithm–user feedback loop
- **Incompatibility** of fairness definitions was not well-recognized (it is mathematically impossible to satisfy all definitions simultaneously)

## Contribution Category
- **Conceptual / Theoretical**: Formal taxonomy of biases and fairness definitions
- **Empirical Synthesis**: Survey of state-of-the-art mitigation methods across ML domains
- **Practical Guidance**: Toolkits, protected attributes list, categorization of algorithms

### Why This Paper Matters
This is one of the most comprehensive references in AI fairness research. It established the standard taxonomy that papers now cite when discussing bias. Any new fairness paper must position itself against this survey's framework — either extending a definition, addressing a missing bias type, or solving a problem in a domain not yet covered.

### Remaining Open Problems
1. Temporal fairness — current definitions are static, but fairness must hold over time
2. Intersectional fairness — individuals belong to multiple protected groups simultaneously
3. Causal fairness at scale — causal graphs are expensive to build for complex systems
4. Fairness in federated learning — decentralized data makes global fairness harder
5. LLM fairness — foundation models encode bias at scale, with limited post-hoc control
6. Fairness–accuracy–privacy trilemma — optimizing all three simultaneously is unsolved
7. Cultural relativism — fairness definitions from the Western legal tradition may not apply globally

---

# 2. Minimum Background Concepts

## 2.1 Protected/Sensitive Attributes
**Definition**: Characteristics of a person that are legally or ethically protected from being used in decisions (race, gender, age, religion, disability, national origin, etc.).
**Role in paper**: Every fairness definition in the paper is framed around ensuring that the model output is not influenced by these attributes.
**Why authors need it**: The core question is "does the model treat people differently because of their protected attributes?"

## 2.2 Disparate Treatment vs. Disparate Impact
- **Disparate treatment**: A model explicitly uses a protected attribute to make different decisions
- **Disparate impact**: A model uses neutral-looking features, but the outcome still disadvantages a protected group (e.g., using zip code as a proxy for race)
**Role in paper**: The distinction motivates why "just remove the protected attribute" (fairness through unawareness) is insufficient as a fairness fix

## 2.3 The Data–Algorithm–User Feedback Loop
ML models train on data → produce outcomes → outcomes affect user behavior → users generate new data → data is used to retrain models. Each step can introduce or amplify bias.
**Role in paper**: The entire taxonomy of 18 bias types is organized around the three arrows of this loop (data→algorithm, algorithm→user, user→data).

## 2.4 Pre-processing / In-processing / Post-processing
Three categories for where bias mitigation can be applied:
- **Pre-processing**: Change the training data before training (e.g., resampling, re-weighting)
- **In-processing**: Modify the training objective (add fairness constraints or penalties)
- **Post-processing**: Modify predictions after training (relabeling, threshold adjustments)
**Role in paper**: All mitigation methods surveyed are classified into one of these three buckets.

## 2.5 Causal Graphs
Directed acyclic graphs where nodes are variables and edges indicate causal influence. Used to trace whether a protected attribute causally drives an outcome, accounting for indirect (mediated) paths.
**Role in paper**: The causal fairness section argues that statistical correlations alone cannot identify discrimination; causal structure is needed.

## 2.6 Word Embeddings
Numerical vector representations of words trained on large text corpora. They capture semantic similarity but also encode human biases present in training text (e.g., "nurse" closer to "woman" than "doctor").
**Role in paper**: Word embedding debiasing is one of the most active NLP fairness sub-areas.

---

# 3. Mathematical / Theoretical Understanding Layer

This is a survey paper, so math is used to precisely state existing definitions — not to derive new theorems. Below are the most important formal definitions and their plain-language meaning.

## 3.1 Core Fairness Definitions

### Definition 1 — Equalized Odds
**Formal**: $P(\hat{Y}=1 \mid A=0, Y=y) = P(\hat{Y}=1 \mid A=1, Y=y)$ for $y \in \{0,1\}$

**Plain meaning**: Both the true positive rate AND the false positive rate must be equal across protected and unprotected groups. The model must be equally accurate and equally wrong for all groups.

**Variable table**:
| Symbol | Meaning |
|---|---|
| $\hat{Y}$ | Model prediction |
| $A$ | Protected attribute (0 = unprotected, 1 = protected group) |
| $Y$ | True label |
| $y$ | A specific label value (0 or 1) |

**Practical interpretation**: In criminal justice (COMPAS), Black defendants and White defendants should have the same false positive rate (being wrongly classified as high-risk) AND the same true positive rate.

**Limitation**: Impossible to satisfy simultaneously with calibration (see Impossibility Result below).

---

### Definition 2 — Equal Opportunity
**Formal**: $P(\hat{Y}=1 \mid A=0, Y=1) = P(\hat{Y}=1 \mid A=1, Y=1)$

**Plain meaning**: Among people who truly deserve a positive outcome, both groups should be correctly identified at the same rate. Only the true positive rate must be equal (not false positive).

**Practical interpretation**: In hiring, equally qualified candidates from both groups should have equal chances of being offered a job. Easier to satisfy than equalized odds.

---

### Definition 3 — Demographic Parity (Statistical Parity)
**Formal**: $P(\hat{Y} \mid A=0) = P(\hat{Y} \mid A=1)$

**Plain meaning**: The probability of receiving a positive prediction must be the same regardless of group membership. The overall acceptance rates must be equal.

**Limitation**: Can be satisfied even when the model is making wrong predictions, as long as the error is balanced. Does not account for differences in base rates between groups.

---

### Definition 4 — Fairness Through Awareness
**Plain meaning**: Similar individuals (measured by some similarity metric) must receive similar predictions. Requires defining a task-specific similarity measure.

**Limitation**: Defining the "right" similarity metric is non-trivial and itself can introduce bias.

---

### Definition 5 — Fairness Through Unawareness
**Plain meaning**: Simply do not use protected attributes in the model. 

**Critical limitation**: Fails because proxy variables (zip code → race, name → gender) still allow the model to discriminate indirectly (disparate impact).

---

### Definition 6 — Treatment Equality
**Formal**: $\frac{FN}{FP}$ is equal for both groups (ratio of false negatives to false positives must be equal).

---

### Definition 7 — Test Fairness (Calibration)
**Formal**: $P(Y=1 \mid S=s, R=b) = P(Y=1 \mid S=s, R=w)$

**Plain meaning**: A risk score of, say, 70% means 70% chance of recidivism regardless of race.

**Impossibility result**: It is mathematically provable that calibration and equalized odds (balanced false positive rates) **cannot both be satisfied** except in degenerate cases. This is one of the most important theoretical results cited by this paper.

---

### Definition 8 — Counterfactual Fairness
**Plain meaning**: A decision is fair if it would be the same in a counterfactual world where the individual belonged to a different demographic group, holding everything else constant. Uses causal modeling.

---

### Definition 9 — Fairness in Relational Domains
**Plain meaning**: Fairness that considers not just individual attributes but also social connections and organizational relationships between individuals.

---

### Definition 10 — Conditional Statistical Parity
**Formal**: $P(\hat{Y} \mid L=1, A=0) = P(\hat{Y} \mid L=1, A=1)$

**Plain meaning**: Equal positive prediction rates conditioned on a set of "legitimate" factors L (e.g., qualifications). Allows for some predictive differences that are legitimately justified.

---

### Mathematical Insight Box
> **Key researcher insight**: Different fairness definitions formalize different notions of justice. Demographic parity reflects "equal outcomes." Equal opportunity reflects "equal chance for qualified individuals." Calibration reflects "reliable predictions." These notions come from different ethical theories and CANNOT all be satisfied at once. Choosing the right definition is a policy decision, not a technical one.

---

## 3.2 Taxonomy of Bias Types (18 Types Organized in the Feedback Loop)

### Data → Algorithm Biases
| Bias Type | Core Cause | Example |
|---|---|---|
| Measurement Bias | Wrong proxy variables used | Arrest rate as proxy for criminality (COMPAS) |
| Omitted Variable Bias | Missing important predictors | Market competitor not captured in churn model |
| Representation Bias | Non-representative sampling | ImageNet dataset dominated by Western cultures |
| Aggregation Bias | Treating heterogeneous groups as homogeneous | One diabetes model for all ethnicities despite different HbA1c patterns |
| Sampling Bias | Non-random sub-group sampling | Study sample differs from target population |
| Longitudinal Data Fallacy | Cross-sectional analysis of temporal data | Reddit comment length analysis across mixed cohorts |
| Linking Bias | Network links misrepresent user behavior | Social network degree vs. actual user interaction patterns |

### Algorithm → User Biases
| Bias Type | Core Cause | Example |
|---|---|---|
| Algorithmic Bias | Design choices in optimization | Regularization or objective function that favors majority group |
| User Interaction Bias | Interface design and user self-selection | Clickthrough data only captures seen content |
| Presentation Bias | How content is displayed | Web results: only visible items get clicks |
| Ranking Bias | Top-ranked assumed most important | More clicks on top search results regardless of quality |
| Popularity Bias | Popular items over-exposed | Recommender systems reinforce already-popular content |
| Emergent Bias | Behavior change after deployment | User population shifts over time post-design |
| Evaluation Bias | Inappropriate benchmarks used | Adience/IJB-A are 80%+ light-skinned subjects |

### User → Data Biases
| Bias Type | Core Cause | Example |
|---|---|---|
| Historical Bias | World reflects existing inequalities | Search for "CEO" returns mostly men |
| Population Bias | Platform users ≠ target population | Pinterest vs. Reddit gender demographics |
| Self-Selection Bias | Participants self-select into studies | Political polls filled by most enthusiastic supporters |
| Social Bias | Social influence distorts ratings | High existing ratings pull new ratings higher |
| Behavioral Bias | Different behavior across platforms | Emoji meaning differs across platforms |
| Temporal Bias | Behavior changes over time | Twitter topic hashtag usage shifts |
| Content Production Bias | Lexical/structural differences | Language use differs by gender and age |

---

# 4. Proposed Method / Framework (MOST IMPORTANT)

This is a survey paper. The "method" is the taxonomic framework and the systematic organization of the field. Below is the framework architecture.

## 4.1 Overall Framework Architecture

```
REAL-WORLD BIAS CASES (Section 2)
         ↓
TAXONOMY OF BIAS SOURCES (Section 3)
   ┌─────────────────────────────────────┐
   │   DATA ──→ ALGORITHM ──→ USER       │
   │    └─────────────────────┘          │
   │         (feedback loop)             │
   └─────────────────────────────────────┘
         ↓
FORMAL FAIRNESS DEFINITIONS (Section 4)
   • Individual / Group / Subgroup fairness
   • 10 mathematical definitions
   • Incompatibility results
         ↓
DOMAIN-SPECIFIC MITIGATION METHODS (Section 5)
   • Pre-processing → In-processing → Post-processing
   • Classification, Regression, PCA
   • Community detection, Graph embedding, Clustering
   • Causal inference methods
   • NLP: Word embeddings, Translation, NER, Language models
   • Deep learning: VAEs, Adversarial learning, Representation learning
         ↓
FUTURE DIRECTIONS (Section 6)
```

## 4.2 Bias Taxonomy Construction

**Step 1: Identify the loop structure**
The authors model the ML system as a feedback loop: data feeds algorithms, algorithms affect users, users generate data. All bias types map onto one of the three arrows.

**Why they did this**: Existing categorizations treated bias as either data or algorithm. The loop model captures more complex dynamics (e.g., emergent bias, feedback amplification).

**Weakness**: The loop is a simplification. In real systems (e.g., federated learning, multi-agent systems), the feedback is more complex.

**Research seed**: A more granular graph-based model of bias propagation paths (not just a loop) in complex sociotechnical systems.

---

**Step 2: Define formal fairness criteria**
The authors collect 10 mathematical fairness definitions from the literature, categorize them (group/individual/subgroup), and prove their incompatibility.

**Why they did this**: Without formal definitions, fairness claims are unverifiable. Practitioners need precise conditions to test.

**Weakness**: All 10 definitions assume a binary classification setting. Fairness in multi-class, regression, or generative models is less covered.

**Research seed**: New fairness definitions for regression, ranking, and generation (e.g., GPT responses).

---

**Step 3: Categorize mitigation methods**
All mitigation methods are organized by domain (classification, NLP, etc.) and stage (pre/in/post-processing). Referenced papers are provided for each category.

**Why they did this**: Practitioners need to know what method to apply when. The categorization guides method selection.

**Weakness**: The survey covers papers up to ~2020. LLMs and foundation models are absent.

**Research seed**: Extension of this taxonomy to cover foundation model fairness (prompt engineering, RLHF, fine-tuning biases).

---

## 4.3 Key Mitigation Methods Surveyed

### Pre-processing Methods
| Method | Idea | Reference |
|---|---|---|
| Preferential Sampling | Reweight samples to reduce discrimination in training data | [75,76] |
| Disparate Impact Removal | Transform features to remove disparate impact | [51] |
| Optimized Pre-processing | Minimize combined fairness and accuracy loss via optimization | [27] |
| Data Augmentation | Balance representation through synthetic data | Various |
| Datasheets for Datasets | Document bias properties of datasets for downstream users | [55] |

### In-processing Methods
| Method | Idea | Reference |
|---|---|---|
| Equalized Odds / Equal Opportunity Constraints | Add fairness constraint to classifier objective | [63] |
| Adversarial Debiasing | Train predictor and adversary simultaneously | [90,156] |
| Disparate Learning Process (DLP) | Use protected attributes in training but not at test time | [94] |
| Multitask Learning with Fairness | Train separate heads per group within one model | [122] |
| Fair PCA | Equal reconstruction fidelity per group | [137] |
| CLAN (Community Detection) | Two-step: structural + attribute-based fairness in graphs | [104] |
| Causal Pathway Constraints | Block discriminative causal paths in causal graph | [116] |
| Fair Word Embedding | Debias word vectors via geometric projections | [20,169] |

### Post-processing Methods
| Method | Idea | Reference |
|---|---|---|
| Threshold Optimization | Different decision thresholds per group post-training | [63] |
| Attention-based Relabeling | Use attention mechanism to rebalance classification output | [102] |
| Reject Option Classification | Assign uncertain predictions near decision boundary via fairness | Various |

---

## 4.4 NLP-Specific Fairness Methods (Surveyed)

- **Word Embedding Debiasing**: Remove gender/race direction from embedding space
- **Coreference Resolution Debiasing**: Fix gender pronoun resolution biases in parsing
- **Machine Translation**: Reduce gender bias in translated text
- **Semantic Role Labeling**: Calibrate role assignments to match training set distribution (RBA algorithm)
- **Named Entity Recognition**: Address under-recognition of minority group entities
- **Language Models**: Perplexity-based bias measurement across demographic groups
- **Sentence Embeddings**: Geometric debiasing analogous to word embedding methods

---

## 4.5 Deep Learning Fairness Methods (Surveyed)

- **Variational Autoencoders (VAEs)**: Learn disentangled representations where latent components do not encode protected attributes
- **Learning Fair Representations (LFR)**: Encode data such that the representation is independent of sensitive attributes while remaining useful for prediction
- **Adversarial Learning**: Train a classifier that cannot be used to predict protected attributes (adversary cannot recover group membership from representation)

---

# 5. Experimental Setup / Evaluation Design

This is a **survey paper** — it does not run new experiments. Instead, it synthesizes and compares results from referenced papers. The "evaluation" of the survey itself is:

## 5.1 Survey Scope and Coverage
- **Coverage**: 170+ referenced papers across ML, NLP, deep learning, causal inference
- **Application areas**: Criminal justice, hiring, advertising, medical, facial recognition, recommender systems, social networks
- **Time window**: Papers from ~2010–2020
- **Selection criteria**: Papers that (a) identify bias, (b) propose mitigation, or (c) formalize fairness definitions

## 5.2 Key Real-World Evaluation Examples Used
| Case | System | Bias Observed | Measurement |
|---|---|---|---|
| Criminal justice | COMPAS | Higher false positive rate for Black defendants | ProPublica analysis |
| Hiring | STEM ad algorithm | Fewer women saw STEM job ads | Google/Facebook ad audit |
| Facial recognition | Commercial systems | Lower accuracy on dark-skinned women | Buolamwini & Gebru (2018) |
| Medical | Clinical datasets | European over-representation | 87% European in 23andMe public data |
| Search | Google Image Search | "CEO" search returns mostly men | Audit study |
| NLP datasets | ImageNet, Open Images | 60% of data from 6 Western countries | Shankar et al. |

## 5.3 Benchmark Datasets Frequently Referenced
- **UCI Adult**: Income prediction; used for fairness in classification experiments
- **COMPAS Recidivism**: Used by ProPublica to demonstrate racial bias
- **LFW (Labeled Faces in the Wild)**: Used for face recognition fairness
- **IJB-A and Adience**: Facial datasets biased toward lighter-skinned subjects
- **imSitu**: Visual scene understanding dataset used to study structured prediction bias

### Experimental Reliability Analysis
| What is trustworthy | What is questionable |
|---|---|
| Real-world bias examples (audited by independent researchers) | That fairness metrics from one dataset generalize to others |
| Mathematical incompatibility proofs (formally proven) | Whether surveyed methods maintain fairness at deployment scale |
| Protected attributes list from FHA and ECOA (legally grounded) | The claim that any single fairness definition is "correct" |
| Bias taxonomy structure (supported by multiple prior categorization papers) | Inclusion of all relevant mitigation methods; some areas may be underrepresented |

---

# 6. Results & Findings Interpretation

## 6.1 Main Findings

**Finding 1: Bias sources are interconnected, not isolated**
Bias in data, algorithms, and user interactions form a self-reinforcing loop. Addressing only one component (e.g., debiasing training data) is insufficient if the algorithm amplifies residual bias or user behavior re-introduces it.

**Finding 2: No universal fairness definition exists**
The 10+ fairness definitions surveyed are mathematically incompatible with each other. Calibration + equalized odds cannot simultaneously hold unless base rates are equal across groups. This means fairness is inherently a policy choice, not a purely technical one.

**Finding 3: Mitigation methods exist but are domain-specific and fragmented**
No single method achieves all fairness goals simultaneously. Pre-processing methods sacrifice information; in-processing methods require access to training; post-processing methods require access to predictions but not internals. Each comes with accuracy trade-offs.

**Finding 4: Bias amplification is a real and documented phenomenon**
Models do not merely reflect bias in data — they amplify it. The imSitu semantic role labeling example showed that a 33% → 16% drop in male agent representation in cooking images occurred post-training.

**Finding 5: Protected attribute selection is critical**
The legal list of protected attributes (Table 3 in paper) provides guidance but is incomplete for all real-world scenarios. Cultural and legal frameworks differ across countries.

## 6.2 Failure Cases and Gaps Identified
- Temporal fairness: current definitions ignore how fairness changes over time
- Fairness in relational/graph data: harder to define than i.i.d. settings
- Subgroup intersectionality: a model can be fair on gender AND race separately, but unfair at the intersection (e.g., Black women)
- Feedback loop effects: no method yet addresses the full loop simultaneously

### Publishability Strength Check
| Aspect | Strength |
|---|---|
| Comprehensiveness of taxonomy | Very high — cited extensively post-publication |
| Formalization of incompatibility results | Strong — grounded in cited proofs |
| Coverage of application domains | High for 2021; gaps exist for post-2021 (LLMs, federated learning) |
| Actionability for practitioners | High — specific tools (AIF360, Aequitas) and method categories given |
| Novel theoretical contributions | Low — this is a synthesis, not a new theorem paper |

---

# 7. Strengths – Weaknesses – Assumptions

## Table 1: Technical Strengths

| Strength | Detail |
|---|---|
| Comprehensive bias taxonomy | 18 bias types organized systematically in a theoretical loop model |
| Formal fairness definitions | 10 precise mathematical definitions with variable explanations |
| Incompatibility results | Mathematical proof that fairness definitions cannot all be satisfied simultaneously |
| Domain breadth | Covers classification, regression, PCA, NLP, deep learning, causal inference, graphs |
| Practical toolkits | References AIF360 and Aequitas for practitioners |
| Legal grounding | Protected attributes list from FHA and ECOA |
| Pre/in/post-processing categorization | Actionable for practitioners choosing where to intervene |

## Table 2: Explicit Weaknesses

| Weakness | Detail |
|---|---|
| Survey cutoff | Covers literature up to ~2020; misses LLMs, RLHF, foundation model fairness entirely |
| Binary classification focus | Most fairness definitions assume binary outcomes; multi-class, regression, and generative fairness less covered |
| Static definitions | All definitions treat fairness as a single snapshot; temporal dynamics not addressed |
| Binary protected attributes | Most definitions treat sensitive attributes as binary (male/female); intersectionality and non-binary identities not covered |
| Western legal bias | Protected attribute lists reflect Western legal systems; global applicability is limited |
| No new empirical results | Survey synthesizes existing results but adds no new empirical validation |
| Feedback loop simplification | The data–algorithm–user loop is a simplification; real-world bias propagation is more complex |

## Table 3: Hidden Assumptions

| Assumption | Why it is hidden |
|---|---|
| Protected attributes are known and discrete | In practice, they may be continuous, unknown, or correlated with non-protected features |
| A single "correct" fairness definition exists per context | The paper lists incompatibilities but does not resolve how to choose; assumes context will determine this |
| Training data exists for retraining | Pre/in-processing methods require access to training data; inapplicable for black-box API systems |
| Groups are well-defined | Individual people belong to multiple, overlapping groups; the methods often treat groups as disjoint |
| Discrimination has a clear legal definition | Across different jurisdictions, what constitutes "illegal" discrimination varies substantially |
| Bias mitigation does not harm all groups | Some debiasing methods shift bias rather than remove it |

---

# 8. Weakness → Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| No coverage of LLM / foundation model fairness | Survey predates ChatGPT and GPT-3 era | Taxonomy of prompt-level, fine-tuning-level, and RLHF-level biases in LLMs | Audit framework + red-teaming framework for LLMs |
| Binary protected attributes assumed | Historical data collects binary labels; non-binary identities were under-researched | Fairness with continuous or multi-valued sensitive attributes | Optimal transport-based fairness; representation-theoretic approaches |
| Static fairness definitions | Mathematical convenience; temporal modeling is harder | Longitudinal fairness: ensuring fairness holds across model updates and time | Causal time-series fairness; online learning with fairness constraints |
| Intersectional fairness not addressed | Testing all subgroup combinations is exponentially expensive | Efficient intersectional fairness testing and mitigation | Subgroup leakage detection; multi-objective fairness optimization |
| Feedback loop not solved end-to-end | No single paper addresses data + algorithm + user simultaneously | Systemic fairness: designing the full production loop to be fair | Multi-agent simulation + feedback-aware training objective |
| Fairness–privacy–accuracy trilemma unsolved | Adding differential privacy reduces signal; fairness constraints reduce accuracy | Pareto-optimal fairness–privacy–accuracy deployment strategies | Multi-objective optimization; information-theoretic bounds |
| Federated learning fairness absent | Federated learning was emerging in 2020 | Fair federated learning: global fairness when data is distributed | Fairness-constrained aggregation; client-level fairness weighting |
| Causal graphs expensive to build | Manual domain knowledge required for causal structure | Automated causal discovery for fairness auditing | Constraint-based causal discovery + sensitivity analysis |
| Evaluation metrics for NLP fairness are weak | Word-level bias proxies (WEAT score) don't capture decision-level harm | Downstream-task fairness metrics for NLP | Task-specific equity metrics; new benchmark datasets with documented diversity |
| Western-centric legal grounding | Most cited laws are US/EU (FHA, ECOA, GDPR) | Cross-cultural fairness taxonomy | Comparative legal AI audit framework |

---

# 9. Novel Contribution Extraction

## Authors' Novel Claim
The paper introduces the **data–algorithm–user feedback loop** as the organizing principle for a unified taxonomy of 18 bias types, maps 10+ fairness definitions into group/individual/subgroup categories, and provides a multi-domain reference for mitigation methods. This is a **conceptual architecture**, not an algorithm.

## Template Novel Claim Statements (For Your Paper)

**Template 1 — LLM Fairness Extension**
> "We propose a bias taxonomy for large language models that extends the Mehrabi et al. framework to prompt-level, training-time, and RLHF-level bias sources, identifying five new bias types specific to generative AI."

**Template 2 — Temporal Fairness**
> "We propose a dynamic fairness framework that extends static fairness definitions to temporal sequences, demonstrating that models satisfying equalized odds at deployment can develop significant fairness drift over six months."

**Template 3 — Federated Fairness**
> "We propose a fairness-constrained federated aggregation algorithm that enforces demographic parity across heterogeneous client data distributions, addressing the gap identified by Mehrabi et al. in decentralized ML settings."

**Template 4 — Intersectional Fairness**
> "We propose an efficient intersectional fairness audit method that detects discrimination at subgroup intersections (e.g., Black women, elderly immigrants) using structured sampling, reducing computational cost by X% compared to exhaustive subgroup testing."

**Template 5 — Causal Fairness Automation**
> "We propose an automated causal discovery pipeline for fairness auditing that constructs probabilistic causal graphs from observational data, enabling discrimination pathway detection without manual domain knowledge."

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Directions
- Better benchmarks for group/individual/subgroup fairness evaluation
- More work on causal methods for fairness discovery and mitigation
- Temporal analysis of fairness interventions
- Cross-domain fairness methods (not domain-specific)
- Responsible reporting practices (datasheets, model cards, nutrition labels)

## 10.2 Missing Directions (Not in Paper)
- **Foundation model / LLM fairness**: Bias in GPT-style models, instruction tuning, RLHF
- **Federated learning fairness**: Fairness when data is distributed across clients
- **Multimodal fairness**: Vision-language models (CLIP, BLIP) have compounded biases
- **Agentic AI fairness**: Autonomous agents making sequential decisions over time
- **Robustness–fairness relationship**: Adversarially robust models can be less fair
- **Fairness in synthetic data generation**: Diffusion models and GANs can reflect or amplify bias
- **Fairness with missing data**: Most methods assume complete data

## 10.3 Modern Extensions
- **RLHF bias**: Human feedback used to align LLMs reflects human biases and can entrench them
- **RAG fairness**: Retrieval-augmented generation systems can inherit bias from knowledge bases
- **Embedding model bias**: BERT, RoBERTa, and sentence embeddings all demonstrate measurable social biases
- **Edge AI fairness**: Models deployed on devices trained on non-representative data

## 10.4 Cross-Domain Combinations
| Source Domain | Target Domain | Opportunity |
|---|---|---|
| Differential privacy | Fairness | Privacy-preserving fair learning |
| Federated learning | Fairness | Cross-client demographic fairness |
| Causal inference | NLP | Causal debiasing for language models |
| Game theory | Fairness | Multi-stakeholder fairness optimization |
| Formal verification | Fairness | Provably fair model certification |

---

# 11. How to Write a NEW Paper From This Work

## Reusable Elements

| Element | How to Reuse |
|---|---|
| Bias taxonomy (18 types) | Use as your problem definition and motivation; extend with new types for your domain |
| Pre/in/post-processing framework | Use to position your method within the mitigation landscape |
| Fairness incompatibility result | Use as motivation for why choosing one definition requires explicit justification |
| COMPAS example | Classic motivating example for any fairness paper |
| Protected attributes list | Use directly for your dataset analysis and fairness testing |
| Real-world case studies | Use to ground your introduction in tangible societal harm |

## What MUST NOT Be Copied
- Do not reproduce any original fairness definition verbatim from this paper (paraphrase or cite)
- Do not copy any table directly (tables 1–4 are from this paper; create your own)
- Do not reproduce the feedback loop diagram without citation
- Do not use the same exact set of real-world examples as if they were your own motivation (acknowledge they come from this survey)

## How to Design a Novel Extension

**Step 1**: Pick one specific weakness from Section 8 of this document
**Step 2**: Find 2–3 recent papers (2022–2025) that have started to address it
**Step 3**: Identify the specific gap those recent papers leave open
**Step 4**: Propose a method that fills exactly that gap
**Step 5**: Design experiments that prove your method's advantage over those recent papers

**Example path**: 
1. Weakness: No LLM fairness taxonomy
2. Recent papers: Gallegos et al. 2024 (survey of LLM bias), WinoBias for pronoun bias
3. Gap: No systematic framework for RLHF-stage bias introduction
4. Proposal: Taxonomy + measurement framework for RLHF bias with empirical case studies
5. Experiments: Measure bias before/after RLHF across 3 demographic dimensions on 5 LLMs

## Minimum Publishable Contribution Checklist
- [ ] Addresses at least one clearly identified gap from this survey
- [ ] Provides either a new formal definition, new algorithm, or new empirical evidence
- [ ] Evaluates on at least 2 standard fairness benchmarks (UCI Adult, COMPAS, etc.)
- [ ] Compares against at least one state-of-the-art prior method
- [ ] Reports both accuracy AND fairness metrics (accuracy alone is insufficient)
- [ ] Includes ablation study showing which component drives your improvement
- [ ] Discusses limitations and remaining open problems

---

# 12. Complete Paper Writing Template

## Abstract
**Purpose**: Summarize the complete research story in 150–250 words.
**Include**:
- Problem (one sentence): "Bias in ML leads to discriminatory outcomes in high-stakes applications."
- Gap (one sentence): "Existing methods do not address [specific gap your paper fills]."
- Method (one sentence): "We propose [name] which [verb phrase]."
- Results (one sentence): "On [datasets], our method achieves [metric] while reducing [bias measure] by [X]%."
- Significance (one sentence): "This demonstrates that [broader claim]."
**Common mistakes**: Overloading with technical jargon; omitting quantitative results; not stating the gap.
**Reviewer expectation**: Self-contained; no acronyms undefined; clear novelty signal.

## Introduction
**Purpose**: Motivate the problem, describe limitations of prior work, and state your contributions.
**Include**:
1. Hook: real-world motivating example (COMPAS, ad systems, facial recognition)
2. Why it matters: list stakeholders affected
3. Problem statement: formal or semi-formal characterization of your exact gap
4. What you do: bulleted list of 2–4 specific contributions
5. Paper organization paragraph
**Common mistakes**: Vague contributions; contributions that are not actually novel; failing to distinguish from Mehrabi et al.
**Reviewer expectation**: Contribution list should be specific and verifiable (not "we improve fairness" but "we reduce false positive rate disparity by X% on COMPAS").

## Related Work
**Purpose**: Position your work relative to prior art without disparaging other work.
**Include**:
1. Reference Section 3 (bias types) of Mehrabi et al. — place your bias type in taxonomy
2. Reference Section 4 (fairness definitions) — state which definition(s) your method satisfies
3. Reference Section 5 (mitigation methods) — classify your method as pre/in/post-processing
4. Cite 5–10 most recent related papers (within your specific sub-domain)
**Common mistakes**: Only citing old work; not citing Mehrabi et al. (reviewers will check); confusing related work with a bibliography dump.
**Reviewer expectation**: Clear explanation of HOW your work differs from each group of related work.

## Method
**Purpose**: Provide a reproducible description of your approach.
**Include**:
1. Problem formulation (mathematical notation for your specific task)
2. Overview figure or pipeline diagram
3. Step-by-step algorithm (with pseudocode if applicable)
4. Justification for each design choice
5. Time/space complexity analysis (if algorithmic)
**Common mistakes**: Skipping justification for choices; algorithm not reproducible from description alone; missing fairness constraint formalization.
**Reviewer expectation**: Expert in the field should be able to re-implement from Method section alone.

## Theory (If Applicable)
**Purpose**: Provide formal guarantees for your method.
**Include**:
1. Statement of theorem/proposition
2. Interpretation in plain language before proof
3. Proof sketch or full proof
4. Statement of assumptions (state clearly what must hold for theorem to apply)
5. Discussion of when assumptions may be violated
**Common mistakes**: Proving something trivial; proving something that doesn't correspond to your practical method; hiding assumptions.
**Reviewer expectation**: Assumptions must be stated and practically meaningful.

## Experiments
**Purpose**: Empirically validate your method against baselines.
**Include**:
1. Datasets: name, size, demographic breakdown, fairness-relevant characteristics
2. Baselines: include BOTH accuracy-focused AND fairness-focused baselines
3. Metrics: BOTH fairness metric (e.g., equalized odds difference, disparate impact ratio) AND accuracy (AUC, F1)
4. Experimental protocol: train/val/test splits, hyperparameter search details
5. Main results table: your method vs. all baselines on all metrics
6. Ablation study: remove one component at a time to show what drives performance
**Common mistakes**: Only reporting accuracy; unfair comparison (your method tuned, baselines not); no ablation.
**Reviewer expectation**: Results must be reproducible; random seed and variance reported.

## Discussion
**Purpose**: Explain what results mean for the field.
**Include**:
1. Why your method works — connect to theory or intuition
2. When it fails — honest analysis of failure cases
3. Which fairness definition you satisfy and why that choice was correct for your application
4. Broader implications for practitioners
**Common mistakes**: Overselling results; ignoring failure cases; not connecting to fairness definition choice.

## Limitations
**Purpose**: Demonstrate intellectual honesty and set scope for follow-up work.
**Include**:
1. Which bias types (from Mehrabi taxonomy) your method does NOT address
2. Which fairness definitions your method violates
3. Dataset limitations (e.g., binary gender labels, US-centric data)
4. Computational limitations (scalability)
5. What would invalidate your claims
**Common mistakes**: Saying "future work will address this" without specifying what
**Reviewer expectation**: Limitations section now mandatory at top venues (NeurIPS, ICML, FAccT).

## Conclusion
**Purpose**: One-paragraph synthesis of what you proved and why it matters.
**Include**: Problem restatement, your method (one sentence), key result (one sentence), broader impact (one sentence).
**Common mistakes**: Introducing new claims; repeating introduction verbatim.

## References
- Cite Mehrabi et al. [1]
- Cite foundational fairness papers: Dwork et al. (individual fairness), Hardt et al. (equalized odds), Calmon et al., Zemel et al. (LFR)
- Cite recent related work (2022–2025) to demonstrate currency
- Cite the dataset papers for any dataset you use

---

# 13. Publication Strategy Guide

## Suitable Venues

| Venue | Type | Focus |
|---|---|---|
| FAccT (ACM Fairness, Accountability, Transparency) | Conference | Best venue for fairness-first papers |
| AIES (AAAI/ACM AI Ethics and Society) | Conference | Ethics + societal implications |
| NeurIPS (Datasets and Benchmarks Track) | Conference | New fairness datasets or benchmarks |
| ICML | Conference | Fairness as constrained optimization; formal guarantees |
| ICLR | Conference | Representation learning fairness; deep learning fairness |
| KDD | Conference | Practical fairness at scale; industrial systems |
| ACM Computing Surveys | Journal | If proposing a new survey/taxonomy |
| Journal of Machine Learning Research (JMLR) | Journal | Theory-heavy fairness papers |
| IEEE Transactions on Neural Networks and Learning Systems | Journal | Deep learning fairness |

## Required Baseline Expectations
- Always include a **vanilla unfair baseline** (baseline without any fairness constraint)
- Include at least one **pre-processing** and one **in/post-processing** baseline
- Standard required baselines: Hardt et al. equalized odds post-processor [63], Calmon et al. optimized pre-processing, Zemel et al. LFR for representation learning
- For classification: report results on **COMPAS** and/or **UCI Adult**

## Experimental Rigor Level
- **Minimum**: 2 datasets, 3 baselines, accuracy + fairness metrics, statistical significance test
- **Good**: 3+ datasets, 5+ baselines, multiple fairness definitions tested, ablation study
- **Strong**: Real-world deployment scenario, human study, intersectional fairness analysis

## Common Rejection Reasons
1. "This is incremental — it only applies one of Mehrabi et al.'s listed techniques without new insight"
2. "Fairness metric is improved at unsustainable accuracy cost"
3. "Only one fairness definition is satisfied; the paper does not justify why this definition is appropriate"
4. "No comparison to standard baselines (AIF360 implementations)"
5. "Dataset used lacks demographic diversity metadata"
6. "Results not reproducible — no seed, no variance reported"
7. "Protected attributes assumed binary; no discussion of non-binary cases"
8. "Broader impact section absent (required at NeurIPS/ICML/FAccT)"

## Increment Needed for Acceptance
- **FAccT**: New insight about fairness (social, legal, or technical); empirical results secondary to insight quality; interdisciplinary framing valued
- **NeurIPS/ICML**: Strong theoretical contribution OR strong empirical results on standard benchmarks with significant improvement
- **KDD**: Practical impact, scalability, industry relevance; deployment results valued

---

# 14. Researcher Quick Reference Tables

## Key Terminology Table

| Term | Definition |
|---|---|
| Protected attribute | Characteristic (race, gender, age, etc.) legally protected from use in decisions |
| Disparate treatment | Explicit use of protected attribute in decision (direct discrimination) |
| Disparate impact | Neutral features used but protected group still harmed (indirect discrimination) |
| Equalized odds | Equal TPR and FPR across groups |
| Equal opportunity | Equal TPR across groups only |
| Demographic parity | Equal positive prediction rate across groups |
| Counterfactual fairness | Decision unchanged if protected attribute were different |
| Calibration (test fairness) | Predicted scores reflect true probabilities regardless of group |
| Pre-processing | Remove bias from training data before model training |
| In-processing | Add fairness constraint or penalty during training |
| Post-processing | Adjust predictions after training to achieve fairness |
| Bias amplification | Model increases bias beyond what was in the training data |
| Feedback loop | Biased outcomes generate biased user behavior generates biased new data |
| Intersectionality | Individuals belong to multiple protected groups simultaneously |
| Proxy discrimination | Non-protected feature used but correlated with protected attribute |
| Historical bias | Pre-existing societal inequalities reflected in data |
| Explainable discrimination | Differential treatment justified by legitimate factors |
| Unexplained discrimination | Differential treatment not justifiable by any legitimate factor |

## Important Equations Summary

| Definition | Equation | Meaning |
|---|---|---|
| Equalized Odds | $P(\hat{Y}=1\mid A=a, Y=y) = \text{const w.r.t. } a$ | Same TPR and FPR across groups |
| Equal Opportunity | $P(\hat{Y}=1\mid A=a, Y=1) = \text{const w.r.t. } a$ | Same TPR across groups |
| Demographic Parity | $P(\hat{Y}\mid A=0) = P(\hat{Y}\mid A=1)$ | Same acceptance rate across groups |
| Conditional Stat. Parity | $P(\hat{Y}\mid L=l, A=a) = \text{const w.r.t. } a$ | Equal outcomes given legitimate factors |
| Counterfactual Fairness | $P(\hat{Y}_{A\leftarrow a}=y\mid x,a) = P(\hat{Y}_{A\leftarrow a'}=y\mid x,a)$ | Same outcome in factual and counterfactual world |

## Parameter / Notation Table

| Symbol | Meaning | Appears In |
|---|---|---|
| $\hat{Y}$ | Model prediction | All fairness definitions |
| $Y$ | True label | Equalized odds, equal opportunity |
| $A$ | Protected attribute variable | All fairness definitions |
| $a, a'$ | Specific values of protected attribute | Counterfactual fairness |
| $S$ | Risk score / predicted probability | Test fairness (calibration) |
| $L$ | Set of legitimate (non-sensitive) factors | Conditional statistical parity |
| $R$ | Group membership variable | Test fairness |
| TPR | True Positive Rate = $P(\hat{Y}=1\mid Y=1)$ | Equalized odds, equal opportunity |
| FPR | False Positive Rate = $P(\hat{Y}=1\mid Y=0)$ | Equalized odds |

## Algorithm Flow Summary

| Method | Stage | Steps |
|---|---|---|
| Fair PCA | In-processing | 1. Formulate Fair PCA as SDP; 2. Solve SDP; 3. Reduce rank via LP |
| CLAN (Community Detection) | In-processing | 1. Detect communities via modularity; 2. Reclassify low-degree nodes via supervised classifier on attributes |
| Preferential Sampling | Pre-processing | 1. Measure discrimination in data; 2. Resample to reduce discrimination; 3. Train standard classifier |
| Adversarial Debiasing | In-processing | 1. Train predictor; 2. Train adversary to predict protected attribute from representation; 3. Optimize predictor to fool adversary |
| RBA (Semantic Role Labeling) | Post-processing | 1. Train model; 2. Calibrate predictions to match training distribution for each category |
| Threshold Optimization | Post-processing | 1. Train standard classifier; 2. Find per-group decision thresholds that satisfy equalized odds |

---

# 15. One-Page Master Summary Card

| Field | Content |
|---|---|
| **Paper** | "A Survey on Bias and Fairness in Machine Learning" — Mehrabi et al., ACM Computing Surveys 2021 |
| **Problem** | ML systems make discriminatory decisions in high-stakes contexts (bail, hiring, loans, medical care) due to bias in data, algorithms, and user interactions |
| **Core Idea** | Bias enters via a self-reinforcing feedback loop: biased data → biased algorithm → biased user behavior → more biased data. Understanding AND breaking this loop requires both formal fairness definitions and targeted mitigation methods |
| **Method (Survey Architecture)** | 1. Taxonomy of 18 bias types organized in the data–algorithm–user feedback loop; 2. Formal catalog of 10+ mathematical fairness definitions with incompatibility proofs; 3. Domain-specific survey of pre/in/post-processing mitigation methods across classification, NLP, deep learning, causal inference |
| **Key Results** | Different fairness definitions are mathematically incompatible; models amplify existing data bias; no single method achieves all fairness goals; fairness choice is a policy decision, not just a technical one |
| **Core Weakness** | Predates LLMs; binary protected attributes assumed; static (non-temporal) definitions; no coverage of federated learning fairness; Western legal framework as sole reference |
| **Research Opportunity** | LLM-era fairness taxonomy; temporal fairness definitions; intersectional subgroup fairness; federated fairness; privacy–fairness–accuracy trilemma; automated causal fairness discovery |
| **Publishable Extension** | Pick any one specific gap → propose a method to address it → evaluate on COMPAS + UCI Adult + one domain-specific dataset → compare against AIF360 baselines → report accuracy AND fairness metrics → submit to FAccT or ICML |
