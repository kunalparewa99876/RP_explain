# Model Selection — ML: Federated Learning (30 Papers)

> **Purpose**: Assign **one model** (Sonnet 4.6 or Opus 4.6) per paper for the upcoming task of explaining each paper in simple, structured, non-technical language to support research understanding and paper writing.

---

## Selection Criteria

| Factor | Sonnet 4.6 | Opus 4.6 |
|---|---|---|
| **Best for** | Surveys, system designs, application papers, benchmarks, practical frameworks, tools | Cryptographic protocols, convergence proofs, privacy-budget analysis, novel mathematical formulations, quantum/neuromorphic paradigms |
| **Strength** | Precise instruction-following, clean structured output, efficient summarisation | Nuanced multi-step reasoning, handling formal proofs, synthesising cryptography + ML + distributed systems |

---

## Paper-by-Paper Assignments

### FOUNDATIONAL PAPERS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 01 | McMahan2017 — FedAvg (Federated Learning) | **Sonnet 4.6** | Foundational FL algorithm; weighted averaging of local updates; clearly defined algorithm with intuitive communication reduction |
| 02 | Kairouz2021 — Advances and Open Problems in FL | **Sonnet 4.6** | Comprehensive 400+ page survey; covers the entire FL landscape; survey format ideal for Sonnet's structured summarisation |

### PRIVACY & SECURITY

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 03 | Dwork2014 — Algorithmic Foundations of Differential Privacy | **Opus 4.6** | Foundational mathematical textbook; (ε,δ)-differential privacy; Laplacian and Gaussian mechanisms; heavy formal probability theory |
| 04 | Abadi2016 — Deep Learning with Differential Privacy | **Opus 4.6** | DP-SGD algorithm; privacy accounting via moments accountant; gradient clipping analysis; requires mathematical precision |
| 05 | Bonawitz2017 — Practical Secure Aggregation | **Opus 4.6** | Cryptographic protocol for secure gradient aggregation; secret sharing, Diffie-Hellman, fault tolerance; deep cryptographic reasoning |
| 06 | Evans2018 — Practical Secure Multi-Party Computation | **Sonnet 4.6** | Accessible introduction to SMPC; Yao's garbled circuits and secret sharing explained pedagogically; designed to be understandable |

### OPTIMISATION & ALGORITHMS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 07 | Li2019 — FedProx | **Sonnet 4.6** | Adds proximal term to FedAvg for non-IID stability; clear algorithmic modification; moderate mathematical complexity |
| 08 | Reddi2021 — Adaptive Federated Optimisation | **Opus 4.6** | Applies Adam/Yogi to FL with server-side momentum and variance; convergence analysis under heterogeneity; mathematical optimisation theory |
| 09 | Kulkarni2020 — Survey of Personalisation Techniques for FL | **Sonnet 4.6** | Survey of meta-learning, multi-task learning, local fine-tuning for personalisation; review format |

### COMMUNICATION EFFICIENCY

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 10 | Sattler2019 — Gradient Compression | **Sonnet 4.6** | Practical Top-K and Random-K sparsification; empirical communication reduction; application-oriented |
| 11 | Alistarh2017 — Quantisation (QSGD) | **Opus 4.6** | Quantised SGD with stochastic rounding; variance analysis; theoretical convergence guarantees; mathematical treatment of bit-width reduction |
| 12 | Gupta2018 — Split Learning | **Sonnet 4.6** | Client-server model partitioning architecture; clear split-and-forward methodology; intuitive design |

### FAIRNESS & BIAS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 13 | Li2020 — Fair Resource Allocation in FL | **Opus 4.6** | q-FedAvg algorithm with fairness-constrained optimisation; mathematical treatment of multi-objective trade-offs across clients |
| 14 | Mehrabi2021 — Survey on Bias and Fairness in ML | **Sonnet 4.6** | Comprehensive survey; taxonomies of bias types and fairness metrics; review format ideal for Sonnet |

### ROBUSTNESS & ATTACKS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 15 | Yin2018 — Byzantine-Robust Distributed Learning | **Opus 4.6** | Geometric median aggregation with theoretical robustness guarantees; optimal statistical rates; formal proofs of Byzantine fault tolerance |
| 16 | Bagdasaryan2020 — Model Poisoning Attacks in FL | **Sonnet 4.6** | Backdoor injection and attack methodology; constrained optimisation attack; clear threat model description |
| 17 | Bhagoji2019 — Analysing FL Through an Adversarial Lens | **Sonnet 4.6** | Comprehensive threat model; targeted and untargeted poisoning; empirical attack evaluation; well-structured security analysis |

### DOMAIN APPLICATIONS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 18 | Kaissis2020 — Privacy-Preserving DL in Healthcare | **Sonnet 4.6** | Overview of FL applications in healthcare; HIPAA/GDPR compliance; medical imaging use cases; application review |
| 19 | Rieke2020 — Future of Digital Health with FL | **Sonnet 4.6** | Real-world FL deployment across hospitals; brain tumour segmentation; operational challenges; practical deployment paper |
| 20 | Lim2020 — FL in Mobile Edge Networks | **Sonnet 4.6** | Survey of FL for IoT and edge computing; resource allocation; 5G/6G integration; survey format |

### SYSTEMS & IMPLEMENTATION

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 21 | Bonawitz2019 — Towards FL at Scale: System Design | **Sonnet 4.6** | Google's production FL system (Gboard); device selection, scheduling, fault tolerance; systems engineering paper |
| 22 | Ziller2021 — PySyft | **Sonnet 4.6** | Open-source FL library; PyTorch integration; privacy-preserving ML toolkit; tool/library documentation |
| 23 | Caldas2019 — LEAF Benchmark | **Sonnet 4.6** | Standardised FL benchmarks; FEMNIST, Shakespeare, Reddit datasets; evaluation metrics; benchmark paper |

### BENCHMARKS & EVALUATION

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 24 | Hsieh2020 — Non-IID Data Quagmire | **Opus 4.6** | Deep convergence analysis under data heterogeneity; when FedAvg fails mathematically; theoretical treatment of statistical heterogeneity |
| 25 | Yang2021 — Vertical Federated Learning | **Sonnet 4.6** | Architectural concepts for feature-partitioned FL; entity alignment protocols; clear categorical description of VFL approaches |

### ADVANCED TOPICS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 26 | Xie2020 — Asynchronous Federated Optimisation | **Opus 4.6** | Convergence guarantees with stale gradients and unbounded delays; mathematical analysis of asynchronous distributed optimisation |
| 27 | Mohri2021 — FL with Only Positive Labels | **Opus 4.6** | Unbiased risk estimator for positive-unlabeled learning in FL; novel mathematical formulation for label-scarce settings |
| 28 | Chen2021 — Quantum Federated Learning | **Opus 4.6** | Intersection of quantum computing and FL; quantum neural networks in distributed settings; novel paradigm requiring deep conceptual reasoning across two complex fields |
| 29 | Roy2022 — Neuromorphic Computing for FL | **Opus 4.6** | Spiking neural networks on neuromorphic hardware for FL; novel paradigm combining brain-inspired computing with distributed learning; interdisciplinary depth |
| 30 | Nature2022 — Neuromorphic Computing | **Sonnet 4.6** | Overview/review of neuromorphic computing; broader survey of the field; less FL-specific mathematical depth |

---

## Summary

| Model | Count | Percentage |
|---|---|---|
| **Sonnet 4.6** | 17 | 57% |
| **Opus 4.6** | 13 | 43% |
| **Total** | **30** | **100%** |

> **Rationale**: The ML/Federated Learning category has a mix of practical systems papers, surveys, and application papers (healthcare, edge networks, benchmarks) for Sonnet 4.6, alongside cryptographic protocols, convergence proofs, privacy theory, and quantum/neuromorphic paradigms that require Opus 4.6's deeper mathematical and interdisciplinary reasoning.
