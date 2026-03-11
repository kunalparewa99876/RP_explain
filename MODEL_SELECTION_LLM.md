# Model Selection — LLM: Memory Management & Cognitive Symbiosis (34 Papers)

> **Purpose**: Assign **one model** (Sonnet 4.6 or Opus 4.6) per paper for the upcoming task of explaining each paper in simple, structured, non-technical language to support research understanding and paper writing.

---

## Selection Criteria

| Factor | Sonnet 4.6 | Opus 4.6 |
|---|---|---|
| **Best for** | Clear architectures, empirical scaling papers, agent frameworks, application papers, surveys | Novel memory architectures, mathematical foundations, complex multi-modal reasoning, philosophical/interdisciplinary depth |
| **Strength** | Precise instruction-following, clean structured output, efficient summarisation | Nuanced multi-step reasoning, handling mathematical derivations, synthesising across disparate fields |

---

## Paper-by-Paper Assignments

### FOUNDATIONAL PAPERS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 01 | Vaswani2017 — Attention Is All You Need | **Opus 4.6** | Foundational transformer; multi-head attention, positional encoding, scaled dot-product attention all require precise mathematical exposition |
| 02 | Devlin2019 — BERT (Bidirectional Transformers) | **Sonnet 4.6** | Well-documented architecture; masked LM and next-sentence prediction are clearly defined; empirical fine-tuning results |
| 03 | Radford2019 — GPT-2 | **Sonnet 4.6** | Autoregressive LM; clear scaling approach; zero-shot transfer; well-structured for summary |
| 04 | Brown2020 — GPT-3 (Few-Shot Learners) | **Sonnet 4.6** | Extensive empirical evaluation of in-context learning; scaling laws; clear structure despite length |

### STATE-SPACE MODELS (SSMs) — EFFICIENCY

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 05 | Katharopoulos2020 — Transformers Are RNNs | **Opus 4.6** | Mathematical reformulation of attention as linear RNN; kernel approximation theory; bridges two paradigms with formal proofs |
| 06 | Gu2022 — S4 (Structured State Spaces) | **Opus 4.6** | Highly mathematical: HiPPO framework, continuous-time discretisation, Cauchy kernel; one of the most technically dense papers |
| 07 | Gu2023 — Mamba (Selective State Spaces) | **Opus 4.6** | Selective SSMs with input-dependent dynamics; hardware-aware parallel scan; complex mathematical and systems-level reasoning |

### MEMORY ARCHITECTURES

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 08 | Graves2014 — Neural Turing Machines | **Opus 4.6** | Novel differentiable memory with attention-based read/write; content and location addressing; paradigm-shifting architecture requiring deep technical understanding |
| 09 | Graves2016 — Differentiable Neural Computer (Nature) | **Opus 4.6** | Extends NTM with temporal links and dynamic allocation; complex memory management mechanisms; Nature-level depth |
| 10 | Lewis2020 — RAG (Retrieval-Augmented Generation) | **Sonnet 4.6** | Clear retrieval + generation pipeline; well-defined architecture; practical approach to LLM knowledge augmentation |
| 11 | Arslan2026 — Aeon (Neuro-Symbolic Memory) | **Opus 4.6** | Novel neuro-symbolic memory with episodic/semantic/working memory; sub-100ms retrieval; complex multi-component architecture requiring deep analysis |

### AGENTIC AI & MULTI-AGENT SYSTEMS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 12 | Schick2023 — Toolformer (Tool Use) | **Sonnet 4.6** | Clear self-supervised tool-learning framework; API call integration; well-defined method |
| 13 | Yao2023 — ReAct (Reasoning + Acting) | **Sonnet 4.6** | Interleaved thought-action-observation chains; clear framework; widely replicated and well-understood |
| 14 | Park2023 — Generative Agents (Simulation) | **Sonnet 4.6** | Memory stream + reflection + planning; well-structured agent architecture; empirical simulation results |
| 15 | Wu2023 — AutoGen (Multi-Agent) | **Sonnet 4.6** | Framework paper; multi-agent conversation orchestration; practical system design |

### VISION-LANGUAGE-ACTION (VLA) MODELS

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 16 | Brohan2023 — RT-1 (Robotics Transformer) | **Sonnet 4.6** | Transformer for robot control; clear VLA architecture; empirical robot manipulation results |
| 17 | Brohan2023 — RT-2 (Vision-Language-Action) | **Opus 4.6** | Combines vision-language models with robot control; web-knowledge transfer to robotics; complex multi-modal reasoning across vision, language, and action spaces |
| 18 | Driess2023 — PaLM-E (Embodied Multimodal) | **Opus 4.6** | 562B parameter embodied LLM; processes vision, language, and sensor data simultaneously; requires understanding of multi-modal fusion at massive scale |
| 19 | Team2024 — Octo (Generalist Robot Policy) | **Sonnet 4.6** | Open-source VLA; practical generalisation across robots; implementation-focused paper |

### WORLD MODELS & VIDEO UNDERSTANDING

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 20 | Ha2018 — World Models | **Sonnet 4.6** | VAE encoder + RNN dynamics + controller; foundational concept is intuitive; clear three-component architecture |
| 21 | Hafner2023 — DreamerV3 (World Models) | **Opus 4.6** | Single algorithm mastering diverse domains; complex latent-space world model with imagined rollouts; sophisticated RL + world model integration |
| 22 | Kerbl2023 — 3D Gaussian Splatting | **Opus 4.6** | Novel 3D scene representation; differential rasterisation; mathematical treatment of Gaussian primitives and real-time rendering |
| 23 | ArXiv2411 — Video Generation | **Sonnet 4.6** | Video generation techniques; likely diffusion-based; application-oriented content |

### ALIGNMENT & SAFETY

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 24 | Ouyang2022 — InstructGPT (RLHF) | **Sonnet 4.6** | Clear 3-step RLHF pipeline (SFT → reward model → PPO); well-documented training process; foundational but clearly structured |
| 25 | Bai2022 — Constitutional AI (AI Feedback) | **Sonnet 4.6** | Self-critique and revision with a constitution of principles; clear methodology; builds on RLHF with understandable extensions |
| 26 | Rafailov2023 — DPO (Direct Preference Optimisation) | **Opus 4.6** | Mathematical derivation showing policy directly optimisable from preferences without separate reward model; KL-constrained optimisation theory |

### COGNITIVE SYMBIOSIS & HUMAN-AI COLLABORATION

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 27 | Eloundou2023 — AI Automation & Labor Impact | **Sonnet 4.6** | Economic analysis of AI impact on jobs; survey-style methodology with clear statistics; data-driven |
| 28 | Clark1998 — Extended Mind (Cognitive Augmentation) | **Opus 4.6** | Foundational philosophy paper; "extended mind" thesis requires nuanced philosophical reasoning and contextualisation for modern AI research |
| 29 | Bansal2021 — Human-AI Collaboration | **Sonnet 4.6** | Literature review and design framework; survey-style; clear practical guidelines |
| 30 | CognitiveOffloading_Societies2023 | **Sonnet 4.6** | Cognitive science perspective on AI over-reliance; skill atrophy discussion; accessible social science content |

### SCIENTIFIC DISCOVERY & HYPOTHESIS GENERATION

| # | Paper | Model | Reasoning |
|---|---|---|---|
| 31 | Wang2022 — ScienceWorld (Reasoning Benchmark) | **Sonnet 4.6** | Benchmark paper; evaluation of agent reasoning; clear experimental setup and results |
| 32 | Bran2023 — ChemCrow (Chemistry Tools) | **Sonnet 4.6** | LLM agent with 18 chemistry tools; practical tool-augmented approach; application-focused |
| 33 | Brunton2016 — SINDy (Equation Discovery) | **Opus 4.6** | Sparse regression for discovering governing equations from data; mathematical treatment of dynamical systems; PNAS-level depth |
| 34 | Unknown_Paper955 | **Opus 4.6** | Unknown content requires Opus 4.6's superior ability to handle ambiguity and perform deep, adaptive analysis of unfamiliar material |

---

## Summary

| Model | Count | Percentage |
|---|---|---|
| **Sonnet 4.6** | 20 | 59% |
| **Opus 4.6** | 14 | 41% |
| **Total** | **34** | **100%** |

> **Rationale**: The LLM category spans transformers, memory architectures, agentic AI, robotics, alignment, and cognitive science. Agent frameworks, application papers, and well-documented architectures (BERT, GPT-2/3, RAG, RLHF) suit Sonnet 4.6. The mathematically dense papers (S4, Mamba, NTM/DNC, Gaussian splatting, DPO) and paradigm-defining works (Extended Mind, Aeon) require Opus 4.6's deeper reasoning.
