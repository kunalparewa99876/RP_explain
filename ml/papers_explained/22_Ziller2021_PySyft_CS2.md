# Research Companion: PySyft / Syft 0.5 — Structured Transparency Platform

> **Source Paper**: "SYFT 0.5: A Platform for Universally Deployable Structured Transparency"
> **Authors**: Adam James Hall, Madhava Jay, Tudor Cebere, Bogdan Cebere, et al. (OpenMined)
> **Institutions**: University of Oxford, OpenMined, Edinburgh Napier University
> **Reference Tag**: 22_Ziller2021_PySyft

---

## 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Problem Domain** | Privacy-Preserving Machine Learning (PPML), Federated / Distributed AI |
| **Paper Type** | Systems / Engineering (with experimental validation) |
| **Core Contribution** | A unified open-source framework (PySyft/Syft 0.5) that integrates multiple privacy-enhancing technologies (PETs) to support Structured Transparency (ST) in ML information flows |
| **Key Idea** | Combine homomorphic encryption + split neural networks to perform private inference — shifting the model split point reduces ciphertext size and computation time dramatically, at a controlled cost to model secrecy |
| **Required Background** | Federated Learning basics, Homomorphic Encryption (conceptual), Differential Privacy, Secure MPC, Python/PyTorch, Peer-to-Peer networking |
| **Primary Baseline** | Fully HE-encrypted inference (no split); other federated frameworks (TFF, FATE, PaddleFL, Flower, etc.) |
| **Main Innovation Type** | System design + algorithmic hybrid (split-HE inference) + theoretical framing (Structured Transparency) |
| **Difficulty Level** | Moderate–Advanced (multi-domain: cryptography + systems + ML + social theory) |
| **Reproducibility Level** | Medium — source code available via OpenMined, but hardware-dependent timing results |

---

## 1. Research Context & Core Problem

### 1.1 Exact Problem Being Solved

When machine learning is applied to sensitive data (medical records, financial data, personal images), the standard workflow forces the data owner to give up a copy of their data to a data scientist. Once copied, data can never be guaranteed to be deleted, creating long-term privacy risk.

The paper targets two simultaneous problems:

- **The data copy problem**: data once copied cannot be un-copied (illustrated by the DukeMTMC dataset scandal — faces used without consent, dataset revoked, but copies remained freely online).
- **The privacy-utility trade-off in ML inference**: existing approaches that protect privacy (e.g., full HE) are computationally too heavy for practical use, while lighter approaches (e.g., sharing raw activations) leak statistical information.

### 1.2 Why the Problem Exists

- Centralised ML architectures are the default — all data flows to a single processor.
- Legal frameworks (GDPR) require explicit consent for data processing, blocking privacy-sensitive research.
- Trust is missing between data subjects (who own the data) and data processors (who run ML models).
- Existing tools address only one privacy concern at a time (just DP, or just HE, or just federated aggregation), with no unified platform.

### 1.3 Historical and Theoretical Gap

Prior frameworks (TFF by Google, FATE by WeBank, PaddleFL by Baidu, Flower by Cambridge) primarily handle **Horizontally Federated Learning (HFL)** — the setting where data is distributed across many users but the model type is the same. Very few frameworks simultaneously support:
- Vertical FL (different feature sets across parties)
- Private Set Intersection (PSI)
- Secure Multi-Party Computation (SMPC)
- Homomorphic Encryption (HE)
- Zero-knowledge Access Control (ZkAC)

PySyft is the only framework at publication time with all of these.

### 1.4 Contribution Categories

- **System design**: Syft's Node/Client/Store/AST/Pointer architecture
- **Theoretical**: Structured Transparency as a privacy framework extending Contextual Integrity
- **Algorithmic**: Hybrid split-HE inference pipeline
- **Empirical**: Measurements showing split point effect on compute time and ciphertext size

### Why This Paper Matters

It bridges social privacy theory (Nissenbaum's Contextual Integrity), legal compliance (GDPR), and practical ML cryptography in one deployable framework. Researchers who want to run ML on data they cannot see now have a reference platform and a formal theory to justify the design.

### Remaining Open Problems

- Activation signals in SplitNN still leak statistical information even without HE — current solutions (Shredder noise, HE) are either weak or slow.
- Model inversion and membership inference attacks remain threats even with DP training.
- Output verification (ensuring model is unbiased and correct) is not yet fine-grained — relies on credential authority trust.
- Multi-party settings beyond two parties are not demonstrated.
- Integration with cloud orchestration (PyGrid) is a stated future goal, not complete.

---

## 2. Minimum Background Concepts

### 2.1 Contextual Integrity (Nissenbaum, 2004, 2009)

- **Plain definition**: Privacy is not just about who has access to data — it is about whether information flows in ways that match the social norms of the context in which data was originally shared.
- **Role in paper**: Provides the philosophical foundation for Structured Transparency (ST). The authors build ST as a technical extension of this social theory.
- **Why needed**: Justifies why purely access-control approaches to privacy are insufficient.

### 2.2 Structured Transparency (ST)

- **Plain definition**: A set of formal properties that a privacy-preserving information flow should satisfy: input privacy, output privacy, input verification, output verification, and flow governance.
- **Role in paper**: ST is the design target — every component of PySyft is evaluated against whether it delivers one of these properties.
- **Why needed**: Gives the engineers a checklist to build against, and gives users a way to audit privacy guarantees.

### 2.3 Homomorphic Encryption (HE) — specifically CKKS

- **Plain definition**: A special form of encryption where arithmetic operations (addition, multiplication) can be performed on encrypted data without decrypting it first. The result, when decrypted, equals what you would have gotten by operating on the unencrypted data.
- **CKKS variant**: Supports approximate arithmetic on real numbers — suitable for neural network activations.
- **Role in paper**: Used to encrypt the activation signals that travel from the Data Owner (DO) to the Data Scientist (DS). The DS computes on ciphertext, returning an encrypted prediction that only the DO can decrypt.
- **Why needed**: Protects input privacy when the DO sends data-derived signals to an untrusted DS.

### 2.4 Split Neural Networks (SplitNN)

- **Plain definition**: A neural network is split into two parts at a chosen layer (the "cut layer"). One party (DO) runs the first part on their local data, producing an activation signal. The other party (DS) runs the second part on that activation signal to produce the final prediction.
- **Role in paper**: Reduces how deep the HE computation needs to go. Fewer encrypted operations → smaller ciphertext → faster computation.
- **Why needed**: Pure HE inference is too slow; SplitNN reduces the encrypted computation burden.

### 2.5 Secure Multi-Party Computation (SMPC)

- **Plain definition**: A family of cryptographic protocols where multiple parties jointly compute a function over their private inputs without any party learning the others' inputs. PySyft uses Beaver triples (multiplicative secret sharing) for SMPC.
- **Role in paper**: Offered as an alternative to HE for certain computation tasks via SyMPC library.
- **Why needed**: For scenarios where HE is too expensive but full data sharing is unacceptable.

### 2.6 Differential Privacy (DP)

- **Plain definition**: A mathematical guarantee that the output of a computation does not reveal whether any specific individual's data was included in the training set. Achieved by adding calibrated noise during training.
- **Role in paper**: Applied during model training (via Opacus library) to prevent membership inference and model inversion attacks from a curious DO who queries the model repeatedly.
- **Why needed**: Protects the DS's training data from being reconstructed by the DO over many inference queries.

### 2.7 Verifiable Credentials and Zero-Knowledge Access Control

- **Plain definition**: Digital certificates that prove attributes about an actor (e.g., "this person is a licensed data scientist") without revealing other personal information. Zero-knowledge proofs allow proving group membership without revealing which group member you are.
- **Role in paper**: Manages who is allowed to connect to whose data store. Uses Aries agents (from Hyperledger) and CL Signatures.
- **Why needed**: Provides the governance and verification layers required by ST — actors must authenticate their roles before an information flow begins.

### 2.8 Abstract Syntax Tree (AST) in Syft

- **Plain definition**: A tree structure that maps Python library objects to their remote equivalents. When a user calls a function on a local pointer object, Syft looks up the AST to find the right remote procedure call to send to the data owner's machine.
- **Role in paper**: Core architectural innovation — makes PySyft nearly transparent to users who write normal Python/PyTorch code.
- **Why needed**: Lowers the barrier for researchers who do not know cryptography to use privacy-preserving tools.

### 2.9 WebRTC / STUN Protocol

- **Plain definition**: WebRTC is a peer-to-peer communication standard. STUN servers help two peers discover each other's network addresses even behind firewalls/NAT, then step aside while the peers communicate directly.
- **Role in paper**: Provides the network layer for the Duet 2-party connection. After pairing, all computation messages travel peer-to-peer over encrypted UDP (DTLS).
- **Why needed**: Enables direct, secure communication without a persistent central server.

---

## 3. Mathematical / Theoretical Understanding Layer

> Classification: This paper is primarily a **Systems/Engineering** paper. Mathematics appears in the context of cryptographic primitives. Full proofs are not presented — the paper uses established schemes.

### 3.1 CKKS Homomorphic Encryption Parameters

The CKKS scheme requires choosing:

| Parameter | Meaning | Effect |
|---|---|---|
| Polynomial degree (`n`) | Size of the polynomial ring | Larger → more security + more capacity but slower |
| Coefficient modulus (bits) | Controls how many multiplications can happen before noise overwhelms the result (called "multiplicative depth") | More layers encrypted → more bits needed |
| Scale (`2^26`) | Controls precision of approximate arithmetic | Higher scale → more accurate but uses more modulus budget |
| Security level (128-bit) | Minimum security against known attacks | Fixed by NIST standards |

**Key insight for this paper**: Every neural network layer that is computed under HE consumes modulus budget. If you compute more layers in plaintext first (SplitNN), you need a smaller modulus, which directly reduces ciphertext size.

### 3.2 The Split-Point Trade-off (Core Quantitative Insight)

Let:
- `L_DO` = number of layers computed by the Data Owner in plaintext
- `L_DS` = number of layers computed by the Data Scientist on ciphertext
- `d` = multiplicative depth of L_DS (determines required modulus bits)

When `L_DO` increases (split point moves deeper into the model on the DO side):
- `d` decreases → coefficient modulus decreases → ciphertext size decreases → computation time decreases
- **But**: the DO now possesses more model weights → model secrecy decreases

This is the core **privacy-efficiency trade-off** formalized by the experimental results.

**Experiment 1** (split after Conv2, DO computes 2 conv layers):
- Modulus: 140 bits, Ciphertext: 269.86 KB, DS computation time: 4.17s

**Experiment 2** (split after FC1, DO computes 2 conv + 1 FC layers):
- Modulus: 88 bits, Ciphertext: 139.62 KB, DS computation time: 97ms

Moving split one layer deeper: **52 bits less modulus**, **~2x smaller ciphertext**, **~43x faster inference**.

### Mathematical Insight Box

> **Key researcher takeaway**: In hybrid split-HE inference, the split point is a first-class design variable, not an afterthought. Each layer moved to the plaintext side has multiplicative effects on ciphertext size and inference speed. Optimal split point selection can be formulated as a constrained optimization problem: minimize compute time subject to a minimum acceptable model secrecy level.

---

## 4. Proposed Method / Framework

> **Paper type**: Systems / Engineering → focus is on pipeline architecture, component design, and information flow.

### 4.1 Overall System Architecture (Duet)

Duet is the 2-party configuration of PySyft:
- **Data Owner (DO)**: Holds private data + initiates the Duet session. Has a Store (local key-value store of objects). Has the root verification key.
- **Data Scientist (DS)**: Connects to DO via WebRTC. Operates on DO's data through pointer objects. Cannot see raw data.

```
[DS Machine]                        [DO Machine]
  Local Pointer ──── RPC message ──→  Node → AST → Execute → Store
  ← ← ← ← ← ← ← ← ← Result pointer returned ← ← ← ← ← ← ←
```

### 4.2 The Private Inference Information Flow (5 Phases)

#### Phase 1 — Governance (Step 1 in paper's Figure 4)
- DO and DS exchange verifiable credentials via Aries agents.
- Each proves their role (e.g., "I am a credentialed data scientist") using zero-knowledge proofs (CL Signatures).
- Credential schemes are defined on a distributed ledger — no central authority needed.
- **Why**: ST requires flow governance — identity must be established before any data moves.
- **Weakness**: Trust depends on the credentialing authority. If the issuing authority is compromised, fake credentials can be created.
- **Research seed**: Explore decentralized credential revocation or reputation-weighted trust for governance.

#### Phase 2 — Input Verification (Steps 2–3)
- DS stores their model as a verifiable credential — the model hash is attested.
- DO can verify: "Is this the model I agreed to use?"
- Dataset properties (completeness, provenance, schema) can also be attested.
- **Why**: ST requires input verification — actors should know what model and data are actually being used.
- **Weakness**: Verifying correctness of a model (e.g., no backdoor) goes beyond just verifying its hash.
- **Research seed**: Commit to model behavior (accuracy, fairness metrics) not just model identity.

#### Phase 3 — Input Privacy / Split-HE Inference (Steps 4–8)

This is the central technical contribution.

```
DO side (plaintext):
  Raw data → Conv1 → Conv2 → [optional: FC1] → Activation Signal → CKKS Encrypt → Send

DS side (ciphertext):
  Receive Ciphertext → [optional: FC1] → Sq. Activation → FC2 → Return Encrypted Output
```

- The DO runs early layers in plaintext to extract a compact activation.
- The activation is encrypted with CKKS before transit.
- The DS runs the remaining layers on ciphertext.
- **Why hybrid**: Pure HE inference (all layers on DS encrypted) requires deep multiplicative depth → large modulus → large ciphertext → slow. SplitNN alone leaks statistical information in activations.
- **Weakness of this step**: Statistical leakage still exists in the activation signal even before encryption — a malicious DS could analyze patterns in many activation signals to reconstruct DO data.
- **Research seed**: Apply activation obfuscation (randomized projections, noise injection) before encryption, or use information-theoretically private activation compression.

#### Phase 4 — Output Privacy (Step 9)
- Encrypted prediction returned to DO.
- DO decrypts locally — DS never sees the plaintext result.
- Model is trained with Differential Privacy (Opacus) to prevent membership inference attacks via repeated queries.
- **Why**: Even if individual queries are private, aggregating many queries can leak training set membership.
- **Weakness**: DP training always trades model accuracy for privacy — the ε budget must be chosen carefully and reported.
- **Research seed**: Adaptive DP mechanisms that allocate privacy budget based on query sensitivity.

#### Phase 5 — Output Verification (Post-inference)
- Currently relies on the credentialing system (if DS's credential is valid, their model is assumed correct).
- Planned extension: track model performance statistics (accuracy, fairness) via Aries infrastructure.
- **Weakness**: No technical mechanism today for verifying model correctness or absence of bias.
- **Research seed**: Zero-knowledge proofs of model properties (e.g., ZK-proof that test accuracy > 90%) without revealing the model weights.

### 4.3 Syft Architecture Components

#### Node and Client
- **Node**: The fundamental communication unit. Receives and forwards signed messages. Has optional key-value Store. Verifies all messages via Ed25519 (256-bit).
- **Client**: The user-facing API object. Provides a handle to invoke remote functions and inspect remote metadata.
- Asymmetry: DO side has the Store; DS side has a Client that requests operations.

#### Abstract Syntax Tree (AST)
- Maps Python library objects (PyTorch tensors, functions) to remote counterparts.
- When user invokes a method on a Pointer, AST translates this to a remote procedure call.
- Allows near-zero-change migration of existing PyTorch code to privacy-preserving remote execution.
- **Weakness**: Only explicitly allowlisted libraries are supported. Arbitrary Python code cannot run remotely.

#### Store and Pointers
- Objects created during remote execution are held in the DO's Store.
- The DS holds Pointer objects — local proxies that map to real objects in the DO's Store.
- Garbage collection: when a DS Pointer goes out of scope, a Protobuf message is sent to delete the real object from the Store.
- **Weakness**: Single-user GC assumption — if multiple DS nodes hold pointers to the same object, naive GC could prematurely delete it.

#### Communication Protocol
- WebRTC over DTLS/UDP for peer-to-peer, after STUN server brokering.
- After initial pairing, no data passes through the STUN server.
- Self-hosted STUN is supported via a single URL parameter.

### 4.4 Privacy-Enhancing Technology Libraries in Syft

| Library | PET Type | Function |
|---|---|---|
| TenSEAL | Homomorphic Encryption (CKKS, BFV via SEAL) | Encrypted tensor operations |
| SyMPC | Secure Multi-Party Computation | Secret sharing with Beaver triples |
| Opacus | Differential Privacy | DP-SGD training of PyTorch models |
| PyDP | Differential Privacy | DP aggregate statistics |
| Aries/AriesExchanger | Zero-Knowledge Access Control | Verifiable credentials, governance |
| PSI library | Private Set Intersection | Based on ECDH + Bloom Filters |
| Syfertext | Secure NLP | Private text preprocessing |

---

## 5. Experimental Setup / Evaluation Design

### 5.1 Dataset

- **MNIST**: 28×28 grayscale handwritten digit images, 10 classes.
- Standard benchmark — not privacy-sensitive itself, but used to demonstrate the inference pipeline.
- **Limitation**: MNIST is too simple to stress-test real-world privacy-utility trade-offs. Results may not generalize to high-dimensional data (medical images, genomics).

### 5.2 Model Architecture

- Deep convolutional neural network.
- Architecture: Conv1 → Conv2 → FC1 → Sq. Activation → FC2 → Output
- Trained with Opacus DP (not counted in the inference flow timing).

### 5.3 Two Experimental Conditions

| Condition | Split Point | DO computes (plaintext) | DS computes (encrypted) |
|---|---|---|---|
| Experiment 1 | After Conv2 | Conv1, Conv2 | FC1, Sq.Act, FC2 |
| Experiment 2 | After FC1 | Conv1, Conv2, FC1, Sq.Act | FC2 only |

### 5.4 Metrics and Why

- **Ciphertext size (KB)**: Practical constraint for network transmission. Smaller = more deployable.
- **Computation time (ms/s)**: Practical constraint for real-time inference. Faster = more deployable.
- **CKKS parameters (modulus bits)**: Proxy for computational cost and security level.
- **Structured Transparency properties**: Qualitative checklist — does the flow satisfy each ST property?

### 5.5 Hardware

- Intel Core i7-6600U @ 2.60GHz, 4 cores.
- Consumer-grade CPU — intentionally modest to show feasibility without specialized hardware.

### 5.6 Hyperparameters (CKKS)

| Parameter | Exp 1 (split at FC2) | Exp 2 (split at FC1) |
|---|---|---|
| Polynomial degree | 8192 | 8192 |
| Coefficient modulus | 140 bits | 88 bits |
| Security level | 128 bits | 128 bits |
| Scale | 2^26 | 2^26 |

### Experimental Reliability Analysis

| What is Trustworthy | What is Questionable |
|---|---|
| Relative comparison between split points (same hardware, same settings) | MNIST is too clean — does not stress-test real image complexity or feature leakage |
| CKKS parameter derivation follows standard practice | Single hardware configuration — no GPU, no ARM, no distributed setting |
| Timing reproducible given the code (available open source) | No statistical significance reported (no variance / confidence intervals across runs) |
| Architecture comparison table (Table 1) is factually verifiable | Table 1 may be outdated quickly — federated framework landscape changes fast |

---

## 6. Results & Findings Interpretation

### 6.1 Main Outcomes

**Quantitative (Timing & Size)**:
- Moving the split point from Conv2→DS to FC1→DS reduces:
  - Ciphertext from **269.86 KB to 139.62 KB** (~48% reduction)
  - DS computation time from **4.17s to 97ms** (~43x speedup)
- The modulus drops from 140 to 88 bits — this is what drives both reductions.

**Qualitative (ST Properties)**:
- Input Privacy: achieved via HE on activation signals.
- Output Privacy: achieved via DO-side decryption + DP training.
- Input Verification: achieved via Aries credential attestation of model and dataset.
- Output Verification: partially achieved (credential trust, not technical model validation).
- Flow Governance: achieved via credential exchange and role verification before flow begins.

### 6.2 Performance Trends

- Each additional layer moved to DO plaintext computation causes a **non-linear drop** in ciphertext size and compute time (because modulus reduction is multiplicative across depth).
- The relationship is: fewer encrypted multiplications → shallower multiplicative depth → smaller modulus → everything downstream gets cheaper.

### 6.3 Failure Cases and Limitations

- **Model secrecy degrades** as the split moves deeper — the DO learns more architecture details and weights.
- **Statistical leakage in activations** is not eliminated by encryption — it is present before encryption. An adversarial DO who sees many activation patterns could still infer training data statistics.
- **Sq. (Square) activation** is used instead of ReLU because HE cannot compute ReLU exactly (it requires comparison operations that are expensive in HE). This forces a specific activation function that may hurt model accuracy.

### 6.4 Unexpected Observations

- The jump from 140-bit to 88-bit modulus (reducing by 52 bits) produces a 43x speedup — this super-linear effect is counter-intuitive at first but makes sense because HE operations scale roughly as O(n log n) where n is the polynomial degree, and modulus reduction lowers the effective depth.

### Publishability Strength Check

| Result | Strength Level | Notes |
|---|---|---|
| 43x speedup from split point change | Strong — concrete number | Needs more datasets and architectures to generalize |
| Framework comparison table (Table 1) | Moderate — factual, comprehensive | Could become stale; needs periodic updates |
| ST property evaluation | Moderate — qualitative | Lacks formal security proofs; relies on argument |
| DP training integration | Moderate | No ε budget values reported in main paper |
| MNIST demo | Weak as a standalone result | Too simple; needs medical or NLP data for impact |

---

## 7. Strengths – Weaknesses – Assumptions

### Technical Strengths

| Strength | Why It Matters |
|---|---|
| Unified framework covering 7+ PETs | No comparable open-source platform at publication time |
| Near-transparent API (AST-based) | Researchers don't need cryptography expertise to use it |
| Hybrid split-HE is a genuinely novel design | Combines two established ideas into a new flow |
| 43x speedup shows practical feasibility | Moves HE inference from theoretical to potentially deployable |
| Structured Transparency formal framework | Gives a principled evaluation methodology for any PPML system |
| Peer-to-peer WebRTC removes central server requirement | Reduces trust requirements and attack surface |
| Open source + modular libraries | Community can extend each component independently |

### Explicit Weaknesses

| Weakness | Impact |
|---|---|
| Activation signal leakage even before encryption | Invalidates full input privacy claim |
| Output verification relies solely on credential trust | Cannot technically guarantee model correctness |
| Square activation function forced by HE constraints | Reduces model expressiveness and potentially accuracy |
| MNIST only for experiments | Results may not generalize to complex tasks |
| No formal privacy budget (ε) reported for DP training | Cannot assess actual DP protection level |
| SyMPC still immature at publication | SMPC functionality not production-ready |
| Single-user GC assumption for Store | Cannot support multiple DS nodes sharing objects safely |
| No GPU support at publication | Limits practical inference speed |

### Hidden Assumptions

| Assumption | Why It's Hidden | Risk If Wrong |
|---|---|---|
| DO is honest (no adversarial DO) | Threat model focuses on DS, not DO | If DO is adversarial, they could abuse their model access |
| Credentialing authority is trustworthy | Aries system delegates root trust to an authority | Compromised authority breaks governance |
| Model trained with DP is sufficient against inversion attacks | DP training doesn't give formal inference-query protection | DO with many queries may still reconstruct training data |
| Polynomial degree 8192 sufficient for chosen network depth | Standard practice, but network-specific | Deeper models would violate this assumption |
| STUN server operated by OpenMined is available and trustworthy | Peer-to-peer only after pairing | Downtime or compromise of STUN server blocks pairing |
| Activation statistical leakage is acceptable after encryption | Not formally analyzed | Adversarial ML attacks on encrypted activation patterns could still work |

---

## 8. Weakness → Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| Activation signals leak statistical information before HE | SplitNN was designed for bandwidth reduction, not full input privacy | Design activation-private split learning | Information-theoretic activation compression; randomized projections; private activation quantization |
| Square activation forced by HE | ReLU requires comparison (not supported in approx. HE) | Design HE-compatible activations with better model quality | Polynomial approximations of ReLU (degree 2–4); learnable polynomial activations; TFHE for exact comparisons |
| No formal ST proof — only argument-based | ST is an extension of social theory, not yet formalized into a security proof | Formalize ST as a cryptographic security definition | Define ST as an ideal functionality in the UC (Universal Composability) framework |
| DP budget not reported | Opacus integration exists but wasn't evaluated as a variable | Study the ε-accuracy trade-off in split-HE inference | Systematic ε sweep + accuracy measurement on non-trivial datasets |
| Output verification is only credential-based | No ZK-proof of model properties exists yet in framework | Add verifiable ML model properties | ZK-SNARKs for provable model accuracy bounds; differentially private model cards |
| Split point selection is manual | No formal optimization criterion for split point | Automate split point selection | Multi-objective optimization: minimize computation cost subject to minimum model secrecy and maximum leakage bounds |
| MNIST only — no complex datasets | System paper uses simple demo to prove concept | Extend to real-world PPML benchmarks | Apply framework to MIMIC-III (medical), FinBench (financial), or LFW (face recognition) with formal privacy evaluation |
| Garbage collection single-user assumption | Engineering convenience | Multi-party reference counting for shared objects | Distributed reference counting protocol; epoch-based reclamation |

---

## 9. Novel Contribution Extraction

### From the Paper's Contributions

**Contribution Statement 1**:
> "We propose PySyft 0.5, a unified framework that integrates homomorphic encryption, secure multi-party computation, differential privacy, and zero-knowledge access control into a single API, enabling Structured Transparency in machine learning workflows."

**Contribution Statement 2**:
> "We propose a hybrid split-HE inference protocol where the data owner computes early network layers in plaintext, encrypts the resulting activation signal with CKKS, and transmits only the compact ciphertext to the data scientist — reducing inference time by 43x and ciphertext size by 48% compared to fully encrypted inference."

### Novel Extension Templates

**Template 1**:
> "We propose an **activation-private split learning protocol** that applies [randomized projection / polynomial noise injection] to activation signals before CKKS encryption, providing [formal ε-information leakage bounds] that existing split-HE approaches lack."

**Template 2**:
> "We propose an **automated split point selection algorithm** for hybrid split-HE networks that minimizes [inference latency + communication overhead] subject to [model secrecy constraints], demonstrated on [medical imaging / NLP] benchmarks."

**Template 3**:
> "We propose a **formally verified Structured Transparency framework** by expressing ST properties as ideal functionalities in the Universal Composability (UC) security model, enabling composable proofs for arbitrary combinations of PETs."

**Template 4**:
> "We propose **ZK-attested model deployment** where a data scientist produces a zero-knowledge proof of model accuracy, fairness metrics, and DP training guarantees before an information flow begins, strengthening the output verification component of Structured Transparency."

**Template 5**:
> "We propose **HE-compatible neural architecture search (HE-NAS)** that jointly optimizes model accuracy and multiplicative depth to identify network architectures that are both high-performing and computationally efficient for private inference."

---

## 10. Future Research Expansion Map

### Author-Suggested Future Work

- Integration of PySyft with **PyGrid**: adds cloud/on-prem worker management, dataset dashboards, and organizational policy controls.
- Support for multiple tensor backends: NumPy, TensorFlow, JAX, scikit-learn (beyond PyTorch).
- Maturing SyMPC for more complex SMPC protocols.

### Missing Directions (Not Mentioned by Authors)

- **Multi-party split learning (>2 parties)**: extend Duet architecture to N-party settings with chained splits.
- **GPU/TPU support for HE inference**: hardware-accelerated CKKS operations exist (HElib, SEAL-GPU).
- **Communication compression**: the WebRTC channel transmits full Protobuf-serialized ciphertext — compression of ciphertext before transmission could further reduce bandwidth.
- **Adaptive split with dynamic layer allocation**: allow split point to shift between inference calls based on network conditions and privacy requirements.
- **Formal DP accounting integrated into Syft**: report cumulative ε budget consumed across multiple inference queries automatically.

### Modern Extensions (2022–2025 Landscape)

- **LLM-era private inference**: apply split-HE to transformer architectures (attention heads are expensive under HE; split after embedding layers is more feasible).
- **Cross-silo federated learning integration**: extend Duet to cross-institutional settings with formal audit trails.
- **TEE + HE hybrid**: combine Trusted Execution Environments (Intel SGX, AMD SEV) with HE for higher performance.
- **Post-quantum HE**: CKKS is not post-quantum resistant under standard assumptions; quantum-resistant variants are an open research frontier.
- **Private RAG (Retrieval-Augmented Generation)**: apply ST principles to private LLM inference with knowledge bases.

### Cross-Domain Combinations

- **Healthcare**: private genomic analysis, federated EHR analysis with ST guarantees.
- **Finance**: cross-bank fraud detection using SMPC without exposing transaction data.
- **Legal**: GDPR-compliant model training pipelines with fully auditable ST information flows.
- **IoT / Edge AI**: split learning where IoT devices compute early layers locally and encrypted activations go to edge server.

---

## 11. How to Write a NEW Paper From This Work

### Reusable Elements

- **Evaluation methodology**: measure ciphertext size, computation time, AND qualitative ST property satisfaction — use all three together.
- **Framework comparison table format** (Table 1): adopt this to compare your system against baselines — rows are systems, columns are capabilities/features.
- **Information flow diagram format** (Figures 2/4): draw actor→step→actor flows with labeled privacy guarantees at each step.
- **Split-point variable design**: use the layer-by-layer table (Tables 2/3) as a template for showing how a design variable affects multiple metrics simultaneously.
- **Structured Transparency as evaluation framework**: use the 5 ST properties (input/output privacy, input/output verification, governance) as a checklist for any new PPML system.

### What MUST NOT Be Copied

- Do not copy the specific CKKS parameter choices without justification for your model and dataset.
- Do not copy the ST qualitative evaluations without providing formal proofs or at least adversarial threat model analysis.
- Do not use MNIST as the primary evaluation dataset — this will be rejected by reviewers as insufficient.
- Do not present the Duet architecture as novel if you are building on it — cite it clearly.

### How to Design a Novel Extension

1. **Pick one weakness from Section 8** (e.g., activation leakage before encryption).
2. **Formalize the problem**: define a privacy leakage metric for pre-encryption activations.
3. **Propose a technical fix**: e.g., apply a learned privacy-preserving transformation to activations before HE.
4. **Build on PySyft as a platform**: implement your modification as a Syft module — this gives you the baseline and framework for free.
5. **Evaluate on a harder dataset** (CIFAR-10, CheXpert, or MIMIC) with both accuracy and privacy metrics.
6. **Compare against**: (a) no activation protection, (b) Shredder noise, (c) full HE baseline.

### Minimum Publishable Contribution Checklist

- [ ] Novel technical mechanism that improves at least one dimension (privacy, accuracy, efficiency) without sacrificing the others.
- [ ] Formal threat model defining adversary capabilities.
- [ ] Evaluation on at least one non-trivial dataset (beyond MNIST).
- [ ] Comparison against at least two baselines.
- [ ] Privacy budget (ε) or security parameter reported quantitatively.
- [ ] Reproducible: code available or experiments fully described.
- [ ] Addresses at least one of the explicit weaknesses in this paper.

---

## 12. Complete Paper Writing Template

### Abstract
- **Purpose**: Summarize problem, method, and key results in ~150 words.
- **Include**: (1) The privacy problem being solved, (2) Your proposed approach in one sentence, (3) Two concrete results (a number + a qualitative gain).
- **Common mistake**: Writing "we propose a framework" without stating what specifically it enables that nothing else could.
- **Reviewer expectation**: Abstract must contain at least one quantitative claim.

---

### Introduction
- **Purpose**: Motivate the problem, establish the gap, and state contributions.
- **Structure**:
  1. Hook paragraph: real-world data privacy incident or regulation.
  2. Why existing approaches fail (cite 3–5 specific works and their limitations).
  3. Your insight / approach (1 paragraph).
  4. Explicit contribution list: "In summary, our contributions are: (1)...(2)...(3)..."
  5. Paper structure sentence ("The rest of the paper is organized as follows...").
- **Common mistake**: Vague contributions ("we improve privacy") with no concrete claims.
- **Reviewer expectation**: Contributions must be falsifiable and verifiable in the experiments.

---

### Related Work
- **Purpose**: Position your work against prior art.
- **Sections to include**:
  - Privacy-Enhancing Technologies (HE, SMPC, DP)
  - Split Learning and Federated Learning
  - Privacy-Preserving Inference systems
  - (If applicable) Formal security frameworks
- **Common mistake**: Listing papers without explaining how yours differs.
- **Reviewer expectation**: Show you have read the most recent (2020–2025) work; do not just cite papers from the original paper.

---

### Method / Framework
- **Purpose**: Describe your technical contribution in enough detail to reproduce.
- **Structure**:
  1. System overview diagram (actor, data flow, components).
  2. Phase-by-phase description of the information flow.
  3. Technical details for each novel component.
  4. Pseudocode or algorithm block for the core mechanism.
  5. Complexity analysis (computation, communication, storage).
- **Common mistake**: Describing implementation details instead of algorithm logic.
- **Reviewer expectation**: A competent reader should be able to reimplement from this section alone.

---

### Theoretical Analysis (if applicable)
- **Purpose**: Prove or formally argue privacy guarantees.
- **Include**: Threat model definition, security definition, theorem statement, proof sketch (full proof in appendix).
- **Common mistake**: Claiming "our method is secure" without a formal proof or clearly scoped argument.
- **Reviewer expectation**: At minimum, a clearly stated threat model and informal security argument.

---

### Experiments
- **Purpose**: Empirically validate your claims.
- **Structure**:
  1. Datasets: description + why chosen + statistics.
  2. Baselines: each baseline explained and justified.
  3. Metrics: each metric defined + justified.
  4. Implementation details: hardware, libraries, hyperparameters.
  5. Main results table/figure.
  6. Ablation study: vary one component at a time to show each part matters.
- **Common mistake**: No ablation study; cherry-picked metrics; no variance reported.
- **Reviewer expectation**: Results must be reproducible; variance/confidence intervals required for claimed speedups.

---

### Discussion
- **Purpose**: Interpret results, connect to theory, explain surprising findings.
- **Include**: Why results match or deviate from expectations; practical deployment implications; comparison to baselines.
- **Common mistake**: Repeating numbers from the results section without interpretation.

---

### Limitations
- **Purpose**: Honest scoping of your contribution.
- **Include**: Dataset limitations, threat model scope, compute assumptions, open problems.
- **Common mistake**: Omitting limitations that reviewers will identify anyway — it is better to state them first.
- **Reviewer expectation**: Shows intellectual honesty and research maturity.

---

### Conclusion
- **Purpose**: Summarize contributions and point to future work.
- **Include**: 1-paragraph contribution summary + 3–5 specific future directions.
- **Common mistake**: Introducing new content not in the paper.

---

### References
- Use consistent citation style.
- Include at minimum: foundational PPML papers, framework comparison papers, any paper whose method you extend or compare against.
- Check every citation is actually cited in text.

---

## 13. Publication Strategy Guide

### Suitable Venues

| Type | Specific Venues | Why |
|---|---|---|
| Top ML/AI Conferences | NeurIPS, ICML, ICLR | If novel algorithm + strong experiments |
| Security/Privacy Conferences | IEEE S&P, CCS, USENIX Security, PETS | If formal security proof is included |
| Systems Conferences | OSDI, SOSP, EuroSys | If architecture/systems contribution dominates |
| Applied ML Workshops | FL-ICML Workshop, PrivateNLP Workshop | Early-stage or preliminary results |
| Journals | TPDS, IEEE TDSC, JMLR | Extended versions with comprehensive evaluation |

### Required Baseline Expectations

- Must compare against at least: full HE (no split), pure SplitNN (no HE), and at least one other framework (TFF, FATE, or Flower).
- Must evaluate on at least two datasets — one of which should be domain-relevant (e.g., medical).
- Must include ablation across split points.

### Experimental Rigor Level

- Top venues require: statistical significance testing, multiple hardware configurations, comprehensive ablation.
- Workshop papers: single hardware, single dataset, but clear novelty.

### Common Rejection Reasons

1. **MNIST-only evaluation** — too easy, not representative.
2. **No formal threat model** — claiming "private" without defining the adversary.
3. **No comparison to recent baselines** — especially if newer hybrid methods exist.
4. **System paper without performance at scale** — single-machine demos insufficient for systems venues.
5. **Qualitative security claims without proofs** — reviewers at security venues require formal analysis.
6. **Missing ablation** — reviewers want to know which parts of your system are necessary.

### Increments Needed for Acceptance

- **Workshop**: framework + one novel mechanism + MNIST proof-of-concept.
- **ML Conference**: novel mechanism + non-trivial dataset + ablation + comparison to 2+ baselines.
- **Security Conference**: novel mechanism + formal security proof in recognized model (UC, game-based) + non-trivial evaluation.
- **Systems Conference**: system + scale evaluation (multi-node, multi-user) + measured overhead vs. non-private baseline.

---

## 14. Researcher Quick Reference Tables

### Key Terminology Table

| Term | Short Definition |
|---|---|
| Structured Transparency (ST) | Framework requiring: input privacy, output privacy, input verification, output verification, flow governance |
| Contextual Integrity | Social theory: privacy = information flowing appropriately within its originating social context |
| CKKS | HE scheme for approximate arithmetic on real numbers; suitable for neural network activations |
| Multiplicative Depth | Number of sequential HE multiplications a ciphertext can undergo before noise destroys it |
| Coefficient Modulus | Budget parameter in CKKS controlling max multiplicative depth; more bits = more depth |
| SplitNN | Neural network split across two parties at a chosen cut layer |
| Hybrid Split-HE | Run early layers in plaintext (DO side), encrypt activation, run remaining layers on ciphertext (DS side) |
| Duet | PySyft's 2-party architecture (Data Owner + Data Scientist) |
| Data Owner (DO) | Party that holds private data; controls the Store; creates Duet session |
| Data Scientist (DS) | Party that holds private model; connects to DO; operates on data via pointers |
| Node | Smallest Syft communication unit; receives, verifies, forwards signed messages |
| Pointer | Local proxy object on DS side mapping to a real object in DO's Store |
| AST | Abstract Syntax Tree — maps Python objects to remote execution paths |
| Store | Key-value store on DO's machine holding all intermediate computation results |
| Aries Agent | Hyperledger component for verifiable credentials and zero-knowledge attribute proofs |
| CL Signatures | Camenisch-Lysyanskaya anonymous credential scheme for zero-knowledge group membership proofs |
| PSI | Private Set Intersection — compute intersection of two private sets without revealing the sets |
| SyMPC | PySyft's in-house SMPC library using Beaver triples for secret-shared computation |
| Opacus | PyTorch library for Differential Privacy training (DP-SGD) |
| WebRTC / STUN | Peer-to-peer connection protocol; STUN server brokers initial connection only |
| DTLS | Datagram TLS — encrypted UDP transport used by WebRTC DataChannel |
| Beaver Triples | Precomputed multiplication triples for efficient SMPC without online interaction |

### Important Equations / Relationships Summary

| Relationship | Meaning |
|---|---|
| More layers on DO (plaintext) → lower multiplicative depth d | Each layer moved to DO reduces required HE depth |
| Lower d → smaller coefficient modulus (bits) | Modulus is determined by depth; less depth = fewer bits |
| Smaller modulus → smaller ciphertext size | Ciphertext size grows with modulus; 140→88 bits ≈ 48% size reduction |
| Smaller modulus → faster HE operations | Polynomial arithmetic cost scales with modulus bit-length |
| More model layers on DO → less model secrecy | DO learns more architecture weights by running them locally |

### Parameter Meaning Table (CKKS)

| Parameter | Exp 1 Value | Exp 2 Value | Meaning |
|---|---|---|---|
| Polynomial degree n | 8192 | 8192 | Ring size; fixes security level and slot capacity |
| Coefficient modulus | 140 bits | 88 bits | Budget for multiplicative operations |
| Security level | 128 bits | 128 bits | Minimum attack resistance; matches NIST standard |
| Scale | 2^26 | 2^26 | Precision of real-number encoding |
| Resulting ciphertext (activation) | 269.86 KB | 139.62 KB | Wire size of encrypted activation signal |
| DS computation time | 4.17 s | 97 ms | Time for DS to process encrypted layers |

### Algorithm Flow Summary

```
HYBRID SPLIT-HE INFERENCE ALGORITHM

SETUP (once):
  1. DO and DS exchange verifiable credentials via Aries
  2. DS stores model hash as verifiable credential
  3. DO selects split_layer ∈ {Conv1, Conv2, FC1, ...}

TRAINING (offline):
  4. DS trains model with DP-SGD (Opacus)

INFERENCE (per query):
  5. DO loads input data
  6. DO runs model[1..split_layer] in plaintext → activation_plaintext
  7. DO encrypts activation_plaintext with CKKS → activation_ciphertext
     (modulus bits determined by depth of remaining layers)
  8. DO sends activation_ciphertext to DS via WebRTC/Protobuf
  9. DS runs model[split_layer+1..end] on activation_ciphertext → output_ciphertext
  10. DS returns output_ciphertext to DO
  11. DO decrypts output_ciphertext → prediction

PRIVACY GUARANTEES:
  - Input privacy: data never leaves DO; only CKKS-encrypted activation sent
  - Output privacy: prediction decrypted only by DO; model trained with DP
  - Input verification: DS model hash attested via Aries credential
  - Output verification: via DS's credential authority (not technically verified yet)
  - Governance: credential exchange before any data flows
```

---

## 15. One-Page Master Summary Card

| Dimension | Content |
|---|---|
| **Problem** | Sensitive data cannot be used for ML inference because it must be copied to a data scientist's machine. Existing privacy solutions are either too computationally heavy (full HE) or leak statistical information (SplitNN). |
| **Idea** | Combine SplitNN + HE: let the data owner run early neural network layers locally in plaintext, then encrypt only the compact activation signal with CKKS before sending it. Evaluate the whole system against a formal Structured Transparency framework. |
| **Method** | Syft 0.5 framework: Node/Client/Store/AST architecture + Duet (2-party) + CKKS encrypted split-inference + Aries verifiable credentials for governance + Opacus for DP training. Split point is a design variable that trades model secrecy for computation efficiency. |
| **Key Results** | Splitting one layer deeper (Conv2→FC1 as cut point) reduces ciphertext size from 269.86 KB to 139.62 KB (48% reduction) and DS computation time from 4.17s to 97ms (43x speedup). PySyft is the most feature-complete open-source PPML framework at publication time. |
| **Weakness** | (1) Activation signals still leak statistical information before encryption. (2) Output verification is not technically guaranteed — only credential-trust-based. (3) Square activation function forced by HE reduces model expressiveness. (4) Only MNIST used for experiments. (5) No formal security proof for the overall ST framework. |
| **Research Opportunity** | (1) Design activation-private transformations before CKKS encryption with formal leakage bounds. (2) Automate split point selection via multi-objective optimization. (3) Formalize ST as a UC security framework. (4) Add ZK-proof of model properties for output verification. (5) Extend to LLM inference and medical imaging with full privacy accounting. |
| **Publishable Extension** | Pick any one weakness above. Example: propose an information-theoretically private activation compression scheme that reduces leakage to ε bits per activation, demonstrate it on CIFAR-10 and a medical dataset, compare against SplitNN (no protection), Shredder, and full HE — and show it fits within the PySyft ST evaluation framework. |

---

*End of Research Companion Document*
*Generated for: 22_Ziller2021_PySyft | Date: 2026-04-11*
