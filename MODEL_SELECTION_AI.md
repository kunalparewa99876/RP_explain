# Model Selection — AI Safety & Governance (30 Papers)

> **Purpose**: Assign **one model** (Sonnet 4.6 or Opus 4.6) per paper for the upcoming task of explaining each paper in simple, structured, non-technical language to support research understanding and paper writing.

---

## Selection Criteria

| Factor | Sonnet 4.6 | Opus 4.6 |
|---|---|---|
| **Best for** | Clear empirical papers, surveys, frameworks, policy docs, practical tools | Heavy math/proofs, novel theoretical paradigms, formal methods, deep interdisciplinary reasoning |
| **Strength** | Precise instruction-following, clean structured output, efficient summarisation | Nuanced multi-step reasoning, handling ambiguity, synthesising complex ideas |

---

## Paper-by-Paper Assignments

### FOUNDATIONAL PAPERS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 01 | Amodei2016 — Concrete Problems in AI Safety | **Sonnet 4.6** | Well-structured empirical safety agenda; 5 clearly defined problems; straightforward to summarise |
| 02 | Katz2017 — Reluplex: SMT Solver for DNN Verification | **Opus 4.6** | Heavy formal methods; extends simplex algorithm; requires deep mathematical reasoning about SMT solving and ReLU constraints |
| 03 | Katz2019 — Marabou Framework for Verification | **Opus 4.6** | Production-grade DNN verifier with complex formal verification theory; ACAS Xu safety proofs require careful mathematical exposition |
| 04 | Gehr2018 — DeepPoly (Abstract Interpretation) | **Opus 4.6** | Abstract interpretation with zonotope and polyhedra domains; heavy mathematical formalism requiring deep reasoning |
| 05 | Xu2021 — α,β-CROWN Verifier | **Opus 4.6** | GPU-accelerated bound propagation with complex mathematical underpinnings; combines incomplete and complete verification |

### AI GOVERNANCE & POLICY

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 06 | NIST2023 — AI Risk Management Framework | **Sonnet 4.6** | Policy/governance document; clear 4-function structure (Govern, Map, Measure, Manage); well-suited for structured summary |
| 07 | Mitchell2019 — Model Cards for Model Reporting | **Sonnet 4.6** | Documentation framework; practical and well-structured; clear template format |
| 08 | Gebru2021 — Datasheets for Datasets | **Sonnet 4.6** | Documentation framework; practical checklist-style content; easy to structure |

### EXPLAINABLE AI (XAI)

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 09 | Ribeiro2016 — LIME | **Sonnet 4.6** | Clear perturbation-based method; well-defined algorithm; moderate mathematical complexity |
| 10 | Lundberg2017 — SHAP | **Opus 4.6** | Built on Shapley values from game theory; unifies 6 existing methods; requires mathematical depth to explain properly |
| 11 | Vaswani2017 — Attention Is All You Need | **Opus 4.6** | Foundational transformer architecture; multi-head attention, positional encoding, and scaled dot-product attention require mathematical precision |
| 12 | Zheng2018 — NOTEARS (Causal Structure Learning) | **Opus 4.6** | Converts combinatorial DAG learning to continuous optimisation; novel mathematical formulation |

### CAUSAL REASONING & FAIRNESS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 13 | Loftus2018 — Fairness Through Causal Awareness | **Opus 4.6** | Counterfactual fairness grounded in causal graphs; requires careful causal reasoning and mathematical rigour |
| 14 | Barocas2023 — Fairness and Machine Learning (Book) | **Sonnet 4.6** | Comprehensive textbook/reference; well-structured content; survey-style coverage |
| 15 | Buolamwini2018 — Gender Shades | **Sonnet 4.6** | Empirical audit study; intersectional analysis with clear results; data-driven, not mathematically complex |
| 16 | Reisman2018 — Algorithmic Impact Assessments | **Sonnet 4.6** | Policy framework paper; stakeholder engagement and transparency; no heavy math |

### ADVERSARIAL ROBUSTNESS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 17 | Szegedy2014 — Intriguing Properties of Neural Networks | **Opus 4.6** | Foundational adversarial examples paper; explores counterintuitive properties of high-dimensional spaces; requires nuanced explanation |
| 18 | Goodfellow2015 — FGSM | **Sonnet 4.6** | Clear one-step attack method; well-known and well-documented; straightforward mathematical formulation |
| 19 | Madry2018 — PGD (Projected Gradient Descent) | **Opus 4.6** | Min-max robust optimisation framework; theoretical analysis of adversarial training; deeper mathematical treatment than FGSM |
| 20 | Cohen2019 — Randomised Smoothing (Certified Robustness) | **Opus 4.6** | Provable robustness certificates via Gaussian smoothing; probabilistic guarantees require careful mathematical reasoning |

### CLIMATE AI

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 21 | Pathak2022 — FourCastNet | **Sonnet 4.6** | Practical weather forecasting model using Fourier neural operators; application-focused with clear architecture |
| 22 | Lam2023 — GraphCast | **Sonnet 4.6** | GNN-based weather model; empirical performance comparison; clear architectural description |
| 23 | Lam2023 — GraphCast Supplemental Material | **Opus 4.6** | Supplemental with detailed ablations, mathematical derivations, and technical proofs requiring deep analysis |
| 24 | Schneider2022 — Hybrid Climate Modeling | **Sonnet 4.6** | Review paper on combining physics and ML; survey-style coverage of approaches |
| 25 | Vaghefi2023 — ClimateGPT | **Sonnet 4.6** | LLM application for climate; practical knowledge synthesis tool; not mathematically complex |

### CONTINUOUS ASSURANCE & MLOPS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 26 | Sculley2015 — Hidden Technical Debt in ML Systems | **Sonnet 4.6** | Systems-level paper; identifies practical problems; clear categorisation of debt types |
| 27 | Sato2019 — Continuous Delivery for ML | **Sonnet 4.6** | Practical MLOps guide; CI/CD principles applied to ML; process-focused |
| 28 | Breck2019 — Monitoring Models in Production | **Sonnet 4.6** | Practical monitoring paper; drift detection and alerting; systems-oriented |

### BRAIN-COMPUTER INTERFACES

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 29 | Willett2021 — Brain-to-Text via Handwriting (Nature) | **Opus 4.6** | Neural decoding from intracortical signals; complex neuroscience + ML intersection; Nature-level depth |
| 30 | Goering2023 — BCI Safety | **Sonnet 4.6** | Safety and ethics paper; neuroethics considerations; regulatory discussion; policy-oriented |

---

## Summary

| Model | Count | Percentage |
|---|---|---|
| **Sonnet 4.6** | 18 | 60% |
| **Opus 4.6** | 12 | 40% |
| **Total** | **30** | **100%** |

> **Rationale**: The AI Safety & Governance category has a significant number of policy, survey, and practical papers (governance frameworks, documentation standards, MLOps) well-suited for Sonnet 4.6. Formal verification, mathematical fairness/causality, and adversarial robustness theory papers require Opus 4.6's deeper reasoning capability.
