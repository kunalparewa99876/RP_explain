# Model Selection — ANN: Advanced Neural Architecture (34 Papers)

> **Purpose**: Assign **one model** (Sonnet 4.6 or Opus 4.6) per paper for the upcoming task of explaining each paper in simple, structured, non-technical language to support research understanding and paper writing.

---

## Selection Criteria

| Factor | Sonnet 4.6 | Opus 4.6 |
|---|---|---|
| **Best for** | Clear empirical papers, surveys, well-known architectures, practical methods | Heavy math (PDEs, Fourier, spectral theory), novel theoretical paradigms, abstract mathematical frameworks |
| **Strength** | Precise instruction-following, clean structured output, efficient summarisation | Nuanced multi-step reasoning, handling complex mathematical derivations, synthesising interdisciplinary ideas |

---

## Paper-by-Paper Assignments

### FOUNDATIONAL PAPERS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 01 | LeCun2015 — Deep Learning (Nature) | **Sonnet 4.6** | Comprehensive overview paper by DL pioneers; survey-style coverage of CNNs, RNNs, and backpropagation; well-structured for summarisation |
| 02 | Krizhevsky2012 — AlexNet (ImageNet) | **Sonnet 4.6** | Landmark CNN paper; empirical results on ImageNet; architecture is well-documented and clear |
| 03 | He2016 — ResNet (Deep Residual Learning) | **Sonnet 4.6** | Residual connections (F(x) + x); clear core idea; widely understood architecture with moderate mathematical complexity |

### TRAINING DYNAMICS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 04 | Glorot2010 — Xavier Initialization | **Opus 4.6** | Analyses variance propagation through deep networks; derives initialization schemes from mathematical analysis of activation functions |
| 05 | Ioffe2015 — Batch Normalisation | **Opus 4.6** | Normalisation theory with internal covariate shift analysis; mathematical treatment of batch statistics and their effect on training dynamics |
| 06 | Kingma2015 — Adam Optimizer | **Opus 4.6** | Adaptive learning rate optimiser combining momentum and RMSprop; bias correction derivation; convergence analysis requires mathematical rigour |

### STATE-SPACE MODELS (SSMs)

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 07 | Gu2022 — S4 (Structured State Spaces) | **Opus 4.6** | Highly mathematical: HiPPO framework, continuous-time state-space discretisation, Cauchy kernel computation; one of the most math-heavy papers in the set |
| 08 | Gu2023 — Mamba (Selective State Spaces) | **Opus 4.6** | Selective state-space models with input-dependent parameters; hardware-aware algorithm design; complex mathematical and systems reasoning |
| 09 | Fu2023 — H3 (Hungry Hungry Hippos) | **Sonnet 4.6** | Combines SSMs with gating; builds on S4 with clearer modifications; less mathematical novelty than S4/Mamba themselves |

### NEURO-SYMBOLIC AI

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 10 | Serafini2016 — Logic Tensor Networks | **Opus 4.6** | Integrates first-order logic with neural networks via fuzzy logic; novel paradigm requiring deep understanding of both symbolic and connectionist AI |
| 11 | Marcus2020 — Next Decade in Robust AI | **Sonnet 4.6** | Argumentative position paper; critiques pure deep learning; a roadmap, not a mathematical derivation |
| 12 | Garcez2023 — Neurosymbolic AI: The 3rd Wave | **Sonnet 4.6** | Survey of neuro-symbolic approaches; categorises methods; review-style paper well-suited for structured summary |

### MODEL COMPRESSION

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 13 | Hinton2015 — Knowledge Distillation | **Sonnet 4.6** | Clear teacher-student framework; soft targets concept is intuitive; moderate mathematical complexity |
| 14 | Sanh2019 — DistilBERT | **Sonnet 4.6** | Practical compression of BERT; empirical focus on speed/accuracy trade-offs; application-oriented |
| 15 | Frankle2019 — Lottery Ticket Hypothesis | **Opus 4.6** | Theoretical insight into sparse networks; pruning at initialisation; requires deep reasoning about network structure and trainability |
| 16 | Jacob2018 — Quantisation (Integer Inference) | **Sonnet 4.6** | Practical quantisation scheme; 8-bit integer arithmetic; production-focused (TF Lite); more implementation than theory |

### SELF-SUPERVISED LEARNING

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 17 | Chen2020 — SimCLR (Contrastive Learning) | **Sonnet 4.6** | Clear contrastive framework; augmentation + agreement maximisation; well-structured methodology |
| 18 | Radford2021 — CLIP (Vision-Language) | **Sonnet 4.6** | Large-scale contrastive pre-training; clear dual-encoder architecture; empirical zero-shot results |
| 19 | Caron2021 — DINO (Self-Supervised ViT) | **Sonnet 4.6** | Self-distillation without labels; clear method with impressive emergent properties; moderate complexity |
| 20 | He2022 — MAE (Masked Autoencoders) | **Sonnet 4.6** | Mask-and-reconstruct paradigm; clear self-supervised method; well-structured for explanation |

### PHYSICS-INFORMED NEURAL NETWORKS (PINNs)

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 21 | Raissi2019 — Physics-Informed Neural Networks | **Opus 4.6** | Neural networks constrained by PDEs; PDE residual in loss function; requires understanding of differential equations and physics modelling |
| 22 | Wang2021 — DeepONet (Operator Learning) | **Opus 4.6** | Branch-trunk architecture for learning PDE solution operators; mathematical operator theory; parametric PDE generalisation |
| 23 | Li2021 — Fourier Neural Operator (FNO) | **Opus 4.6** | Fourier-space learning of PDE operators; resolution-invariance via spectral methods; deep mathematical content |
| 24 | Kovachki2023 — Neural Operators Review | **Sonnet 4.6** | Survey paper covering DeepONet, FNO, and variants; review format well-suited for structured summarisation |

### GRAPH NEURAL NETWORKS (GNNs)

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 25 | Kipf2017 — Graph Convolutional Networks (GCN) | **Opus 4.6** | Spectral graph convolution theory; Chebyshev polynomial approximation; requires mathematical understanding of graph Laplacians |
| 26 | Veličković2018 — Graph Attention Networks (GAT) | **Sonnet 4.6** | Adds attention to graph convolutions; clear extension of existing GNN methods; moderate complexity |
| 27 | Hamilton2017 — GraphSAGE (Inductive Learning) | **Sonnet 4.6** | Practical inductive GNN; neighbour sampling and aggregation functions; application-focused |
| 28 | Guo2021 — Molecular Property Prediction (GNN) | **Opus 4.6** | GNNs for quantum chemistry; bond/angle/torsion interactions; interdisciplinary (ML + chemistry) requiring nuanced explanation |

### CONTINUAL LEARNING

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 29 | Kirkpatrick2017 — EWC (Elastic Weight Consolidation) | **Opus 4.6** | Fisher information matrix for weight importance; mathematical treatment of catastrophic forgetting; Bayesian justification |
| 30 | Rusu2016 — Progressive Neural Networks | **Sonnet 4.6** | Clear modular architecture; lateral connections; no-forgetting-by-design approach; intuitive method |
| 31 | DeLange2021 — Continual Learning Survey | **Sonnet 4.6** | Comprehensive survey with taxonomy; review format ideal for Sonnet's structured summarisation |

### CAPSULE NETWORKS & EQUIVARIANCE

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 32 | Sabour2017 — Capsule Networks (Dynamic Routing) | **Opus 4.6** | Novel routing-by-agreement algorithm; vector representations of part-whole hierarchies; paradigm shift from standard CNNs |
| 33 | Chen2018 — Neural ODEs | **Opus 4.6** | Continuous-depth networks via ODE solvers; adjoint method for memory-efficient training; deep mathematical content (NeurIPS Best Paper) |
| 34 | Bronstein2021 — Geometric Deep Learning | **Opus 4.6** | Unified mathematical framework (symmetries, equivariance, gauges); abstract algebra applied to deep learning; one of the most theoretically dense papers |

---

## Summary

| Model | Count | Percentage |
|---|---|---|
| **Sonnet 4.6** | 18 | 53% |
| **Opus 4.6** | 16 | 47% |
| **Total** | **34** | **100%** |

> **Rationale**: ANN has the heaviest concentration of mathematical papers (SSMs, PINNs, PDEs, spectral graph theory, Neural ODEs, geometric DL). These require Opus 4.6's deep reasoning. The remaining papers — surveys, well-known architectures (AlexNet, ResNet), practical methods (DistilBERT, quantisation), and self-supervised learning — are well-suited for Sonnet 4.6's structured output.
