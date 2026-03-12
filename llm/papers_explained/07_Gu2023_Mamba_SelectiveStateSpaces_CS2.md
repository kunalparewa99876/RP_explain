# 07 — Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**Authors:** Albert Gu (Carnegie Mellon University), Tri Dao (Princeton University)  
**Published:** December 2023  
**Venue:** Preprint (arXiv)

---

# 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Problem Domain** | Efficient sequence modeling for foundation models across language, audio, and genomics |
| **Paper Type** | Algorithmic / Method + Experimental ML / Empirical |
| **Core Contribution** | A selective state space model (S6) with a hardware-aware scan algorithm, combined into a simple architecture (Mamba) that replaces Transformers for sequence modeling |
| **Key Idea** | Making SSM parameters input-dependent (selective) enables content-based reasoning while retaining linear-time complexity; a fused GPU kernel avoids the resulting memory bottleneck |
| **Required Background** | State space models (SSMs), recurrent neural networks (RNNs), convolutions, Transformer attention, GPU memory hierarchy (HBM vs. SRAM), parallel scan algorithm |
| **Primary Baseline** | Transformer++ (GPT3 architecture enhanced with RoPE, SwiGLU, RMSNorm — i.e., LLaMa-style recipe) |
| **Main Innovation Type** | Algorithmic (selection mechanism) + Systems (hardware-aware implementation) + Architectural (simplified Mamba block) |
| **Difficulty Level** | High — requires understanding of continuous-time systems, discretization, GPU memory, and parallel algorithms |
| **Reproducibility Level** | High — code and pretrained checkpoints are publicly released on GitHub |

---

# 1. Research Context & Core Problem

## 1.1 Exact Problem Formulation

- Foundation models across domains (language, audio, genomics, vision) overwhelmingly rely on Transformer architectures
- The core attention mechanism in Transformers has **quadratic time and memory complexity** in sequence length, making it prohibitively expensive for long sequences
- Existing subquadratic alternatives (linear attention, gated convolutions, structured state space models / S4) scale efficiently but **fail to match Transformer quality on discrete modalities** like language
- The fundamental bottleneck: these efficient models use **time-invariant (LTI) dynamics** — their parameters do not change based on the input content, so they cannot selectively focus on or ignore specific tokens

## 1.2 Why the Problem Exists

- Attention works well precisely because it does **no compression** — it stores the entire context (KV cache), which gives it powerful content-based reasoning but at quadratic cost
- Recurrent models (including SSMs) are efficient because they compress context into a fixed-size state, but the quality of their output depends entirely on **how well that compression captures relevant information**
- Prior SSMs use fixed (learned but time-invariant) parameters, meaning they process every input token the same way regardless of content — they cannot "choose" to remember an important token or forget an irrelevant one

## 1.3 Historical / Theoretical Gap

- The S4 family of models (S4, S4D, S5, DSS, Hyena, H3) demonstrated that structured state space models could scale efficiently and dominate benchmarks on continuous signals (audio, vision, Long Range Arena)
- However, none of these models matched Transformer performance on **language modeling** — the single most important modality for foundation models
- The gap was caused by the **linear time invariance (LTI) constraint**: efficient convolution-based computation required parameters to be constant over time, preventing content-aware reasoning

## 1.4 Limitations of Previous Approaches

| Previous Approach | Limitation |
|---|---|
| Standard Transformer | Quadratic complexity in sequence length; cannot handle very long sequences |
| Linear Attention | Degenerate linear SSM; weak modeling power |
| S4 / S4D / DSS | Time-invariant; cannot do content-based selection; inferior language modeling |
| H3 | Sandwiches S4 with gating but the SSM core is still LTI |
| Hyena | Replaces S4 with MLP-parameterized convolution; still fundamentally LTI |
| RetNet | Simplified SSM (state dim N=1); alternative parallel path but limited expressivity |
| RWKV | Linear attention variant; LTI recurrences; ratio of two SSMs |

## 1.5 Contribution Category

- **Algorithmic**: Selection mechanism that makes SSM parameters input-dependent
- **Systems/Engineering**: Hardware-aware parallel scan algorithm (kernel fusion + recomputation)
- **Architectural**: Simplified neural network block combining SSM with gated MLP (the Mamba block)
- **Empirical**: State-of-the-art results on language, audio, and DNA, validating the approach at scale

### Why This Paper Matters

- Mamba is the **first linear-time sequence model to match Transformer-quality performance** on language modeling
- It achieves **5× higher inference throughput** than Transformers of the same size because it has no KV cache
- It scales linearly with sequence length, enabling practical modeling of sequences up to **1 million tokens**
- The Mamba-3B model matches or exceeds Transformers twice its size (e.g., Pythia-7B) on downstream tasks
- It provides a **drop-in replacement** for the Transformer backbone in foundation models

### Remaining Open Problems

1. Scaling beyond a few billion parameters — unclear if advantages persist at 7B+ scale
2. Whether downstream affordances of Transformers (in-context learning, instruction tuning, RLHF, prompting) transfer fully to Mamba
3. The selection mechanism hurts performance on continuous signals (audio) — no single approach dominates across all modalities
4. Hybrid architectures (Mamba + Attention) are not extensively explored
5. Theoretical understanding of why selection is necessary and sufficient for language modeling
6. Quantization and efficient deployment techniques for Mamba are unexplored

---

# 2. Minimum Background Concepts

## 2.1 State Space Models (SSMs)

- **Definition:** A mathematical framework that maps an input sequence to an output sequence through a hidden (latent) state, governed by a set of matrices (A, B, C, D)
- **Role in paper:** The foundational building block that this paper modifies with a selection mechanism
- **Why authors needed it:** SSMs provide the only known family of recurrent models with linear-time complexity that also have principled mechanisms for long-range dependencies

## 2.2 Discretization

- **Definition:** The process of converting a continuous-time system (defined by differential equations) into a discrete-time system (that operates on sequences step by step), using a step size parameter Δ
- **Role in paper:** Connects continuous SSM parameters (Δ, A, B) to discrete parameters (A̅, B̅) via formulas like zero-order hold (ZOH); Δ becomes the key parameter for the selection mechanism
- **Why authors needed it:** Discretization is the first computational step in any SSM forward pass and provides the mathematical bridge to RNN gating (Theorem 1)

## 2.3 Linear Time Invariance (LTI)

- **Definition:** A system whose parameters (A̅, B̅, C) do not change over time — every input token is processed with the same dynamics
- **Role in paper:** LTI is the constraint that all prior SSMs required for efficient computation via convolution; this paper removes LTI by making parameters time-varying (input-dependent)
- **Why authors needed it:** Understanding LTI's limitation is the core motivation — LTI models cannot perform content-based reasoning

## 2.4 Parallel Scan (Prefix Sum)

- **Definition:** An algorithm that computes cumulative operations (like running sums or recurrences) in parallel instead of sequentially, achieving O(L) work in O(log L) parallel time
- **Role in paper:** Enables efficient parallel training of the time-varying recurrence that cannot be computed as a convolution
- **Why authors needed it:** Without the parallel scan, the selective SSM would be sequential and prohibitively slow during training

## 2.5 GPU Memory Hierarchy (HBM vs. SRAM)

- **Definition:** GPUs have two main memory levels: HBM (High Bandwidth Memory, large but slow) and SRAM (on-chip registers/shared memory, small but fast). Most non-matmul operations are memory-bandwidth-bound
- **Role in paper:** The hardware-aware algorithm avoids materializing the expanded state (B, L, D, N) in HBM by performing all computation in SRAM
- **Why authors needed it:** The naive implementation of selective scan would require O(BLDN) memory reads/writes to HBM, making it impractically slow

## 2.6 Kernel Fusion

- **Definition:** Combining multiple GPU operations into a single kernel so that intermediate results stay in fast SRAM rather than being written back to slow HBM
- **Role in paper:** The authors fuse discretization, parallel scan, and output multiplication into one kernel, reducing memory I/O by a factor of N (the state dimension)
- **Why authors needed it:** This fusion is what makes the selective scan 20–40× faster than the naive implementation

## 2.7 Gating Mechanisms in RNNs

- **Definition:** Learned scalar gates (values between 0 and 1) that control how much of the current input versus the previous hidden state is retained at each time step (e.g., LSTM forget gate, GRU reset gate)
- **Role in paper:** The selection mechanism on Δ generalizes RNN gating (Theorem 1); a large Δ resets state and focuses on current input, a small Δ preserves state and ignores current input
- **Why authors needed it:** This connection provides principled parameterization and initialization for the selection mechanism

---

# 3. Mathematical / Theoretical Understanding Layer

## 3.1 The Continuous-Time SSM (Equation 1)

### Intuition

A continuous-time SSM takes a 1D input signal x(t) and produces a 1D output signal y(t) by routing information through a hidden state h(t) of dimension N. Think of it as a dynamical system where:
- h'(t) = A·h(t) + B·x(t) — the hidden state evolves based on itself (via A) and the input (via B)
- y(t) = C·h(t) — the output reads from the hidden state via C

### What Problem It Solves

Models long-range dependencies using principled continuous-time dynamics, which provides resolution invariance and proper normalization

### Variable Meaning Table

| Variable | Shape | Meaning |
|---|---|---|
| x(t) | scalar (per channel) | Input signal at continuous time t |
| y(t) | scalar (per channel) | Output signal at continuous time t |
| h(t) | N-dimensional vector | Latent (hidden) state |
| A | N × N | State transition matrix — governs how the hidden state evolves over time |
| B | N × 1 | Input projection matrix — controls how input enters the hidden state |
| C | 1 × N | Output projection matrix — controls how hidden state maps to output |
| Δ | scalar | Step size for discretization — controls the temporal resolution |

### Assumptions

- The system is linear (no nonlinearities inside the recurrence)
- A is diagonal (for computational efficiency) — reduces N×N matrix to N numbers
- One SSM operates per channel independently (SISO: single-input single-output)

### Practical Interpretation

Each channel of the model has its own recurrent hidden state of dimension N. The total hidden state across D channels is D×N, which is the "effective state" of the model — much larger than traditional RNNs (which typically have N=1 per channel).

## 3.2 Discretization (Equation 4 — Zero-Order Hold)

### Intuition

To use the continuous-time SSM on discrete sequences (like text tokens), we "sample" the continuous system at intervals of size Δ. The zero-order hold method assumes the input is constant between sample points.

### Formulas

- A̅ = exp(Δ·A) — the discrete state transition matrix
- B̅ = (Δ·A)⁻¹ · (exp(Δ·A) - I) · Δ·B — the discrete input matrix

### What Problem It Solves

Bridges continuous-time theory (with nice mathematical properties) to discrete-time computation (which is what we actually run on GPUs)

### Key Insight

Δ controls the "speed" of the system: a **large Δ** means the system takes a big time step (focusing on the current input and forgetting past state), while a **small Δ** means a tiny time step (preserving past state and barely incorporating the current input). This is exactly what gates do in LSTMs/GRUs.

## 3.3 Dual Computation Modes (Equations 2 and 3)

### Recurrent Mode (Equation 2)

- h_t = A̅·h_{t-1} + B̅·x_t
- y_t = C·h_t
- **When used:** Autoregressive inference (one token at a time)
- **Complexity:** O(BLDN) time, O(BDN) memory per step
- **Advantage:** Constant time per step, no KV cache needed

### Convolutional Mode (Equation 3)

- Unrolling the recurrence yields a global convolution kernel K̅ = (C·B̅, C·A̅·B̅, C·A̅²·B̅, ...)
- y = x * K̅ (convolution)
- **When used:** Training (entire sequence available)
- **Complexity:** O(BLD log L) via FFT
- **Advantage:** Fully parallelizable
- **Restriction:** Only works when parameters are time-invariant (LTI)

### Why This Matters

The convolutional mode is HOW prior SSMs achieved efficiency. But selective SSMs **break time-invariance**, so the convolutional mode no longer applies. The authors must find an alternative efficient computation path — this is where the hardware-aware parallel scan comes in.

## 3.4 The Selection Mechanism (Algorithm 2 — S6)

### Intuition

Instead of using fixed B, C, and Δ for all time steps, make them **functions of the current input**:
- B_t = Linear_N(x_t) — input-dependent: controls how the current token enters the state
- C_t = Linear_N(x_t) — input-dependent: controls how the state maps to output
- Δ_t = softplus(Parameter + Linear_1(x_t)) — input-dependent: controls the step size (gating)

### What Problem It Solves

Allows the model to:
1. **Selectively remember** relevant tokens (large Δ)
2. **Selectively forget** irrelevant tokens (small Δ)
3. **Content-aware gate** what enters the state (B_t) and what leaves the state (C_t)

### Shape Changes (Critical Detail)

| Parameter | S4 (LTI) Shape | S6 (Selective) Shape | Change |
|---|---|---|---|
| B | (D, N) | (B, L, N) | Now varies per batch element and time step |
| C | (D, N) | (B, L, N) | Now varies per batch element and time step |
| Δ | (D) | (B, L, D) | Now varies per batch element and time step |
| A̅, B̅ | (D, N) | (B, L, D, N) | Discretized parameters now time-varying |

### Limitation

The expanded discrete parameters A̅, B̅ of shape (B, L, D, N) are much larger than the input/output of shape (B, L, D). Naively materializing them causes a memory blowup by factor N. This is solved by the hardware-aware algorithm (Section 3.3 of the paper).

## 3.5 Theorem 1 — Connection to RNN Gating

### Intuition

When the SSM state dimension is N=1 and A=−1, B=1, the selective SSM recurrence simplifies exactly to a gated RNN:

h_t = (1 − g_t)·h_{t-1} + g_t·x_t

where g_t = σ(Linear(x_t)) is a sigmoid gate, and σ is effectively softplus-based.

### What This Means

- The selection mechanism is a **mathematically principled generalization** of RNN gating
- Δ plays the role of the gate: large Δ → g_t ≈ 1 → focus on current input; small Δ → g_t ≈ 0 → ignore current input, preserve state
- This provides theoretical justification for the specific parameterization choices (softplus activation, broadcast from dim 1)

### Mathematical Insight Box

> The key equation to remember: h_t = A̅_t · h_{t-1} + B̅_t · x_t, where A̅_t and B̅_t are now **time-varying** (depend on the input x_t through Δ_t). This single change — making the recurrence input-dependent — is what transforms a weak LTI model into a powerful selective model. The rest is engineering to make it efficient.

---

# 4. Proposed Method / Framework (MOST IMPORTANT)

## 4.1 Overall Pipeline

The Mamba architecture is a deep neural network composed of repeated identical blocks (the "Mamba block"). Each block processes a sequence of embeddings and produces a sequence of the same shape. Multiple blocks are stacked with residual connections and normalization layers.

**High-level flow for a single Mamba block:**

```
Input x (B, L, D)
  │
  ├──→ Linear projection (expand D → E·D, with E=2)
  │       │
  │       ├──→ Branch 1: Conv1d → SiLU activation → Selective SSM → output z₁
  │       │
  │       └──→ Branch 2: SiLU activation → output z₂
  │
  ├──→ Elementwise multiply: z₁ ⊙ z₂
  │
  └──→ Linear projection (contract E·D → D) → Output (B, L, D)
```

## 4.2 Step-by-Step Component Breakdown

### Step 1: Input Linear Projection

- Input x of shape (B, L, D) is projected to (B, L, 2·E·D) where E=2
- This is split into two branches of shape (B, L, E·D) each
- **Why:** Expands the model dimension to increase expressivity while keeping a gated structure (similar to SwiGLU in Transformers)

✔ **Why authors did this:** Combining two branches (SSM branch + gate branch) into a single block merges the traditional H3-style SSM architectures with MLP blocks, eliminating the need for separate interleaved blocks  
✔ **Weakness:** The expansion factor doubles memory for intermediate activations  
✔ **Research idea seed:** Explore different expansion factors or dynamic expansion based on input complexity

### Step 2: Short Convolution (Conv1d)

- Applied to Branch 1 only, with a small kernel (typically 4)
- **Why:** Provides local context mixing before the SSM processes global context — analogous to the "shift-SSM" in H3

✔ **Why authors did this:** Local patterns (like bigrams in language) are captured cheaply before global selective processing  
✔ **Weakness:** Fixed kernel size; may not adapt well to all modalities  
✔ **Research idea seed:** Use input-dependent convolution kernel sizes, or replace with a lightweight local attention

### Step 3: SiLU (Swish) Activation

- Applied after Conv1d on Branch 1, and directly on Branch 2
- SiLU(x) = x · σ(x) where σ is the sigmoid function
- **Why:** Provides nonlinearity; the combination of Branch 1 × Branch 2 with SiLU creates a SwiGLU-like gating structure

✔ **Why authors did this:** SwiGLU has been shown to outperform other activation functions in large language models (LLaMa, PaLM)  
✔ **Weakness:** SiLU is smooth but not the only option; other activations might work comparably  
✔ **Research idea seed:** Minimal impact expected from changing this, but adaptive activation functions could be investigated

### Step 4: Selective SSM (S6) — The Core Innovation

This is where the selection mechanism operates:

1. **Parameter generation:** From the input x_t at each time step:
   - B_t = Linear_N(x_t) — projects input to state dimension N
   - C_t = Linear_N(x_t) — projects input to state dimension N
   - Δ_t = softplus(bias + Linear_1(x_t)) — projects input to scalar, broadcast to D dimensions

2. **Discretization:** Convert continuous parameters to discrete:
   - A̅_t = exp(Δ_t · A) — A is a learned diagonal matrix (fixed, not selective)
   - B̅_t = (Δ_t · A)⁻¹ · (exp(Δ_t · A) − I) · Δ_t · B_t

3. **Selective scan (recurrence):**
   - h_t = A̅_t · h_{t-1} + B̅_t · x_t
   - y_t = C_t · h_t
   
   This is computed using the hardware-aware parallel scan algorithm (not convolution)

4. **Output:** y of shape (B, L, E·D)

✔ **Why authors did this:** Making B, C, Δ input-dependent allows the model to decide per-token whether to store, ignore, or retrieve information from the hidden state  
✔ **Weakness:** Breaks the LTI property, preventing efficient convolution-based computation; relies on custom CUDA kernels for efficiency  
✔ **Research idea seed:** Explore making A also selective; investigate different selection functions beyond linear projections; combine selection with attention for hybrid models

### Step 5: Gated Multiplication

- Multiply the output of the SSM branch (z₁) with the gated branch (z₂) elementwise
- **Why:** Multiplicative gating is a proven mechanism for controlling information flow in neural networks

✔ **Why authors did this:** Creates a nonlinear interaction similar to the multiplicative gate in GLU/SwiGLU  
✔ **Weakness:** One branch sees no sequential context (only pointwise activation)  
✔ **Research idea seed:** Add a lightweight sequential processing to the gate branch as well

### Step 6: Output Linear Projection

- Projects from expanded dimension (E·D) back to model dimension D
- **Why:** Restores the original dimensionality for residual connections with the block input

### Step 7: Residual Connection + LayerNorm

- Standard residual: output = LayerNorm(block_output + x)
- Optional: the paper uses an additional normalization layer inside the block (inspired by RetNet)

## 4.3 Hardware-Aware Selective Scan Algorithm

### The Problem

The selective scan produces intermediate states of shape (B, L, D, N). For typical values (B=64, L=2048, D=2048, N=16), this is ~128 GB — far too large for GPU memory.

### The Solution: Three Techniques Combined

1. **Kernel Fusion**
   - Fuse discretization + parallel scan + output multiplication into a single CUDA kernel
   - Load parameters (Δ, A, B, C) from HBM → SRAM → compute → write output back to HBM
   - Intermediate states of size (B, L, D, N) exist only in SRAM (never in HBM)
   - Reduces memory I/O by factor N (typically 10–100×)

2. **Parallel Scan (Blelloch 1990)**
   - Even though the recurrence is sequential, it can be parallelized using the associative property of the scan operation
   - Work complexity: O(BLDN), same as sequential
   - Parallel time: O(log L), enabling GPU parallelism

3. **Recomputation (Gradient Checkpointing)**
   - During forward pass: do NOT save intermediate states (would cost O(BLDN) memory)
   - During backward pass: recompute intermediate states from inputs (which are O(BLD + DN) — much smaller)
   - Net effect: same memory as FlashAttention-based Transformers (~16 bytes per token per selective SSM layer)

### Pseudocode-Style Explanation

```
FORWARD PASS:
  Load (Δ, A, B, C) from HBM to SRAM          # O(BLD + DN) read
  For each chunk of sequence length:
    Discretize: compute A̅, B̅ in SRAM            # No HBM access
    Parallel scan: compute all h_t in SRAM       # No HBM access
    Multiply h_t by C to get y_t in SRAM         # No HBM access
  Write y (output) from SRAM to HBM             # O(BLD) write
  
BACKWARD PASS:
  Load (Δ, A, B, C) and output gradients from HBM to SRAM
  Recompute intermediate states in SRAM
  Compute gradients in SRAM
  Write input gradients to HBM
```

## 4.4 Architecture Design Choices

| Design Choice | Decision | Rationale |
|---|---|---|
| Expansion factor E | 2 | Matches parameter count of Transformer (MHA + MLP) when using 2 Mamba blocks per "layer" |
| State dimension N | 16 | Best trade-off: increasing N gives large quality improvements at negligible parameter cost |
| Activation function | SiLU / Swish | Makes the gated branch equivalent to SwiGLU (proven effective in LLaMa/PaLM) |
| Conv1d kernel size | 4 | Small local context; cheap computation |
| A initialization | S4D-Real: A_n = −(n+1) | Simpler real-valued initialization outperforms complex-valued S4D-Lin for language |
| Δ initialization | τ_Δ⁻¹(Uniform(0.001, 0.1)) | Following prior SSM work for stable training |
| A selectivity | Not selective (fixed) | Δ already provides sufficient selectivity through A̅ = exp(Δ·A); simplicity wins |
| Normalization | RMSNorm (pre-norm) | Following proven Transformer++ practices |
| Bias terms | No linear bias | Following LLaMa/PaLM conventions |

---

# 5. Experimental Setup / Evaluation Design

## 5.1 Synthetic Tasks

### Selective Copying Task

- **Configuration:** Sequences of length 4096, vocab size 16, memorize 16 colored tokens among white noise tokens
- **What it tests:** Content-aware reasoning — the model must distinguish relevant tokens from irrelevant ones based on content, not position
- **Why it matters:** LTI models solve vanilla Copying (fixed spacing) trivially via time-tracking, but fail on Selective Copying (random spacing) because they lack content awareness

### Induction Heads Task

- **Configuration:** 2-layer models, sequence length 256 during training, tested up to 1M tokens
- **What it tests:** Associative recall — after seeing "Harry Potter", predict "Potter" when "Harry" appears again
- **Why it matters:** Induction heads are hypothesized to be the core mechanism behind in-context learning in LLMs

## 5.2 Language Modeling

- **Dataset:** The Pile (800GB diverse text dataset)
- **Tokenizer:** GPT-NeoX-20B tokenizer (same as Pythia and RWKV for fair comparison)
- **Training recipe:** Following GPT3 specifications with Chinchilla-optimal token counts
- **Model sizes:** 130M, 370M, 790M, 1.4B, 2.8B parameters
- **Training tokens:** Proportional to model size (Chinchilla scaling), up to 300B tokens for largest models
- **Context lengths:** 2048 and 8192 for scaling laws
- **Optimizer:** AdamW with β=(0.9, 0.95), weight decay 0.1, cosine LR decay
- **Downstream tasks:** LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande (zero-shot)

## 5.3 DNA Modeling

- **Dataset:** HG38 (Human Genome) — ~4.5 billion DNA base pairs
- **Task 1:** Causal language modeling (next base pair prediction) — scaling law evaluation
- **Task 2:** Great Apes species classification — classify DNA segments as human, chimpanzee, gorilla, orangutan, or bonobo (99% DNA similarity, very challenging)
- **Sequence lengths:** 1024 up to 1,048,576 (1M)
- **Model sizes:** 200K to 40M parameters

## 5.4 Audio Modeling

- **Dataset:** YouTubeMix (4 hours solo piano, 16kHz sampling rate) for pretraining
- **Dataset:** SC09 (1-second speech clips of digits 0–9, 16kHz) for generation
- **Architecture:** U-Net backbone with Mamba blocks replacing S4+MLP blocks
- **Metric:** Bits per byte (BPB) for pretraining; FID, IS, mIS, AM for generation quality

## 5.5 Metrics Used and Why

| Metric | Domain | Why This Metric |
|---|---|---|
| Perplexity | Language, DNA | Standard measure of how well the model predicts next token; lower is better |
| Zero-shot accuracy | Language downstream | Measures generalization without fine-tuning; directly comparable across models |
| BPB (Bits per Byte) | Audio | Normalized negative log-likelihood; standard for audio generation quality |
| FID (Fréchet Inception Distance) | Audio generation | Measures how similar generated audio is to real audio in feature space |
| IS (Inception Score) | Audio generation | Measures quality and diversity of generated samples |
| Throughput (tokens/s) | Efficiency | Practical measure of inference speed |

## 5.6 Hardware / Compute Assumptions

- All efficiency benchmarks on A100 80GB PCIe GPU
- Training used standard mixed-precision (BF16/FP16)
- Scaling laws followed Chinchilla protocol (tokens proportional to parameters)

### Experimental Reliability Analysis

**What is trustworthy:**
- Language modeling comparisons are strong: same dataset, tokenizer, and training tokens as Pythia and RWKV
- Scaling law experiments follow rigorous Chinchilla protocol
- Synthetic tasks convincingly isolate the selection mechanism's contribution
- Code and checkpoints are publicly released for verification

**What is questionable:**
- DNA experiments use relatively small models (up to 40M parameters) — unclear if findings scale
- Audio experiments show that selection actually hurts for continuous signals — the "universal backbone" claim is nuanced
- Largest language model is only 2.8B — competitive landscape at 7B+ is unknown
- No comparison with very recent models that appeared after the paper (e.g., Mixtral, newer RWKV versions)
- Downstream evaluations only cover common sense reasoning; no evaluation on code, math, or instruction following

---

# 6. Results & Findings Interpretation

## 6.1 Main Outcomes

### Synthetic Tasks

- **Selective Copying:** The selection mechanism (S6) is necessary and sufficient — S4 alone achieves 18.3% accuracy, S6 achieves 97.0%, and the full Mamba (Mamba block + S6) achieves 99.8%
- **Induction Heads:** Mamba perfectly solves the task AND extrapolates to 1M token sequences (4000× longer than training), while attention models fail beyond 2× training length and run out of memory at 16K

### Language Modeling

- **Scaling Laws:** Mamba is the first attention-free model to match Transformer++ (the strongest Transformer recipe with RoPE, SwiGLU, RMSNorm). The gap widens in Mamba's favor at longer sequence lengths
- **Downstream (Table 3):** At every model size (130M to 2.8B), Mamba achieves the **best results on every single evaluation task** compared to models of the same size. Mamba-2.8B (63.3% avg) exceeds GPT-J-6B (63.0%) and Pythia-6.9B (61.7%) — models with 2–2.5× more parameters

### DNA Modeling

- **Model size scaling:** Mamba matches Transformer++ and HyenaDNA with 3–4× fewer parameters
- **Context length scaling:** Mamba's perplexity **improves monotonically** with longer context up to 1M tokens, while HyenaDNA's perplexity **gets worse** with longer context
- **Species classification:** Mamba-7M achieves 81.3% accuracy at 1M context length on the Great Apes task (random guessing is 20%), far exceeding HyenaDNA

### Audio Modeling

- **Pretraining (YouTubeMix):** Mamba consistently outperforms SaShiMi (S4+MLP) baseline, with the gap widening at longer context lengths
- **Generation (SC09):** Small Mamba (6.1M params) outperforms all baselines including much larger GAN and diffusion models. Larger Mamba (24.3M) achieves FID of 0.67 (previous best: 1.42)

### Efficiency

- Mamba's selective scan is up to **40× faster** than naive PyTorch scan and matches or exceeds FlashAttention-2 speed beyond 2K sequence length
- **Inference throughput:** Mamba-1.4B achieves 4–5× higher throughput than a similarly-sized Transformer (due to no KV cache requirement)
- **Memory:** Comparable to Transformer with FlashAttention-2 (~16 bytes per token per layer)

## 6.2 Performance Trends

- Selection is most beneficial for **discrete, information-dense** data (language, DNA)
- Selection can actually **hurt** on **continuous, smooth** data (audio waveforms) where LTI inductive bias is beneficial
- The outer layers of audio models (closer to raw signal) benefit from LTI, while inner layers (after tokenization/compression) benefit from selection
- Performance improvements scale well with model size and context length simultaneously

## 6.3 Failure Cases

- On audio waveforms, the selection mechanism degrades performance compared to LTI S4 (Figure 10)
- Complex-valued SSMs outperform real-valued ones for audio, contrary to language where real-valued wins
- The paper does not evaluate on tasks requiring complex reasoning (math, code) where Transformers might have stronger advantages

## 6.4 Unexpected Observations

- Architectural gating (multiplicative interactions) alone does NOT solve the Selective Copying task — it must be combined with selection along the sequence dimension
- The choice of inner LTI SSM (S4 vs. Hyena vs. S4D-Real vs. S4D-Complex) barely matters — what matters is whether selection is used
- Increasing state dimension N from 1→16 with selective B, C gives >1.0 perplexity improvement for only 1% more parameters — without selection, increasing N has almost no effect
- Hybrid Mamba-MHA architecture (adding attention to Mamba) provides only marginal improvement, suggesting Mamba already captures most of what attention provides

### Publishability Strength Check

**Publication-grade results:**
- Language modeling scaling laws matching Transformer++ (Figure 4) — very strong
- Zero-shot downstream evaluations (Table 3) — comprehensive and reproducible
- Synthetic tasks cleanly isolating the mechanism (Tables 1, 2) — elegant and convincing
- Inference throughput benchmarks (Figure 8) — practically important
- Model ablations systematically validating each component (Tables 6–10)

**Needs stronger validation:**
- Scaling beyond 3B parameters
- Evaluation on more diverse downstream tasks (MMLU, HumanEval, GSM8K)
- Comparison with concurrent/newer efficient architectures
- Long-context evaluations on practical tasks (not just perplexity)

---

# 7. Strengths – Weaknesses – Assumptions

## Table 1: Technical Strengths

| # | Strength | Evidence |
|---|---|---|
| 1 | First linear-time model matching Transformer quality on language | Figure 4: scaling laws match Transformer++ |
| 2 | 5× higher inference throughput than Transformers | Figure 8: no KV cache needed |
| 3 | Scales to million-length sequences | DNA and audio experiments with 1M tokens |
| 4 | Clean theoretical motivation via Theorem 1 (gating connection) | Mathematical proof connecting Δ to RNN gates |
| 5 | Simple architecture — single repeated block, no attention | Figure 3: only Conv1d + SSM + gated MLP |
| 6 | Comprehensive ablations isolating each contribution | Tables 6–10 systematically test each component |
| 7 | Hardware-aware implementation competitive with FlashAttention | Memory and speed benchmarks on A100 |
| 8 | Open-source code and checkpoints | GitHub repository publicly available |

## Table 2: Explicit Weaknesses

| # | Weakness | Impact |
|---|---|---|
| 1 | Only tested up to 2.8B parameters | Unknown if advantages persist at production scale (7B+) |
| 2 | Selection hurts on continuous signals (audio) | Not truly universal — modality-dependent tuning needed |
| 3 | Requires custom CUDA kernels | Hardware vendor lock-in; not easily portable to TPUs/other accelerators |
| 4 | No evaluation on complex reasoning tasks | Math, code, instruction-following capabilities unknown |
| 5 | Downstream affordances (RLHF, prompting, ICL) not validated | Unclear if post-training techniques work equally well |
| 6 | Real vs. complex parameterization is modality-specific | Practitioners must choose based on domain, adding tuning burden |
| 7 | No principled theory for when selection helps vs. hurts | Practical guidance is empirical, not theoretical |

## Table 3: Hidden Assumptions

| # | Assumption | Risk |
|---|---|---|
| 1 | Diagonal A matrix is sufficient | Non-diagonal structures might capture richer dynamics |
| 2 | SISO formulation (one SSM per channel) is optimal | MIMO formulation (channels interacting inside SSM) could be more expressive |
| 3 | Selection on Δ, B, C is sufficient (not A) | Authors acknowledge A-selection might help but did not test extensively |
| 4 | State dimension N=16 is near-optimal | Explored up to 16; higher N might help more at larger model scales |
| 5 | SiLU/SwiGLU activation is optimal | Marginal contribution not isolated |
| 6 | Training improvements (RMSNorm, no bias, higher LR) apply equally to Mamba | Some improvements might benefit Transformers more proportionally |
| 7 | Chinchilla-optimal training is appropriate for SSMs | SSMs might have different compute-optimal trade-offs than Transformers |

---

# 8. Weakness → Research Direction Mapping

| Identified Weakness | Why it Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| Only tested up to 2.8B params | Compute constraints at time of publication | Validate Mamba at 7B–70B scale with production workloads | Partner with labs running large-scale training; study scaling laws at 10B+ |
| Selection hurts on audio | Audio is continuous/smooth; LTI inductive bias is beneficial | **Adaptive selection** — learn per-layer whether to use LTI or selective mode | Meta-learned selection mask or continuous relaxation between LTI and selective |
| Requires custom CUDA kernels | Parallel scan is not a standard GPU primitive | Develop portable scan primitives for multiple backends | JAX-compatible or Triton-based implementations; XLA custom ops for TPU |
| No evaluation on reasoning tasks | Paper focused on perplexity and common sense | Evaluate Mamba on GSM8K, HumanEval, MMLU, BBH | Extend downstream evaluation suite; may reveal need for hybrid approaches |
| RLHF/ICL not validated | Post-training infrastructure is Transformer-centric | **Study SSM-specific alignment techniques** | Apply DPO, RLHF to Mamba; measure in-context learning curves vs. Transformers |
| Modality-specific parameterization | Continuous vs. discrete data need different inductive biases | **Universal adaptive SSM** that automatically detects modality | Learnable complex-real interpolation; per-layer modality detection |
| No theory for when selection helps | Empirical discovery, not theoretical | **Formal characterization** of task families where selection is necessary | Computational complexity analysis; relate to formal language hierarchies |
| Diagonal A only | Efficiency constraint | **Structured non-diagonal A** with efficient computation | Block-diagonal, low-rank + diagonal, or Toeplitz-structured A matrices |
| Fixed state dimension N=16 | Larger N increases computation | **Dynamic state expansion** — use larger N only when needed | Input-dependent N selection; mixture of SSMs with different state sizes |

---

# 9. Novel Contribution Extraction

## 9.1 Explicit Contribution Statements from the Paper

1. "We propose selective state space models that make SSM parameters functions of the input, enabling content-based reasoning while maintaining linear-time complexity."
2. "We design a hardware-aware parallel scan algorithm that materializes intermediate states only in fast SRAM, achieving 20–40× speedup over naive implementations."
3. "We introduce the Mamba architecture — a simplified block combining selective SSMs with gated MLPs — that requires no attention mechanism."

## 9.2 Reusable Novel Claim Templates (for your own research)

**Template 1 — Mechanism Transfer:**
> "We propose ______ [modification to SSM parameters] that improves ______ [specific capability] by ______ [making specific component input-dependent], enabling ______ [new application domain]."

**Template 2 — Efficiency Innovation:**
> "We design a hardware-aware algorithm for ______ [operation] that exploits ______ [memory hierarchy property], achieving ______ [speedup factor] over standard implementations while maintaining ______ [quality constraint]."

**Template 3 — Architecture Simplification:**
> "We simplify ______ [existing complex architecture] by combining ______ [component A] and ______ [component B] into a single block, reducing ______ [complexity metric] while achieving ______ [comparable or better quality]."

**Template 4 — Selective Mechanism Extension:**
> "We extend the selection mechanism of Mamba to ______ [new domain/modality] by ______ [adaptation strategy], demonstrating that selective state spaces can effectively model ______ [type of data]."

**Template 5 — Hybrid Architecture:**
> "We propose a hybrid architecture combining selective SSMs with ______ [complementary mechanism] that achieves ______ [benefit of SSMs] and ______ [benefit of other mechanism] simultaneously."

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Work

- Validate scaling properties at 7B+ parameters
- Investigate whether Mamba supports Transformer-like downstream affordances (fine-tuning, RLHF, prompting, in-context learning, instruction tuning, quantization)
- Further engineering for scaling SSMs (training stability, distributed training, etc.)

## 10.2 Missing Directions Not Explored by Authors

- **Multimodal Mamba:** Using selective SSMs for vision-language models, video understanding, or cross-modal reasoning
- **Encoder-only Mamba:** The paper only studies causal (decoder-only) models; bidirectional Mamba for classification/retrieval tasks is unexplored
- **Mamba for structured prediction:** Machine translation, summarization, or other seq2seq tasks
- **Continual learning:** The fixed-size state makes Mamba a natural candidate for lifelong/continual learning
- **Edge deployment:** Linear-time inference with small state is ideal for mobile/embedded devices

## 10.3 Modern Extensions (post-publication)

- **Mamba-2 (Dao & Gu, 2024):** Improved version with structured state space duality (SSD), connecting selective SSMs to a form of structured attention
- **Vision Mamba / VMamba:** Applying Mamba to image classification and dense prediction tasks
- **Jamba:** Hybrid Mamba-Transformer architecture from AI21 Labs
- **MambaByte:** Applying Mamba to byte-level (tokenizer-free) language modeling
- **Mamba in diffusion models:** Using Mamba as the backbone for diffusion-based image/video generation

## 10.4 Cross-Domain Combinations

| Domain 1 | Domain 2 | Research Idea |
|---|---|---|
| Mamba + Reinforcement Learning | Robotics | Linear-time state tracking for long-horizon robotic tasks |
| Mamba + Graph Neural Networks | Molecular modeling | Selective state propagation over molecular graphs |
| Mamba + Retrieval-Augmented Generation | Information retrieval | Efficient long-context QA without KV cache limitations |
| Mamba + Mixture of Experts | Large-scale LLMs | Sparse Mamba layers with expert-specific selection patterns |
| Mamba + Knowledge Distillation | Model compression | Distill large Transformer knowledge into compact Mamba models |

## 10.5 LLM-Era Extensions

- Selection mechanism as a **learned attention alternative** in multimodal LLMs
- Mamba as the **backbone for AI agents** — infinite context window for long-running agent sessions
- **Speculative decoding with Mamba** — the recurrent mode makes Mamba a natural draft model
- **State caching** instead of KV caching — trading fixed-size state snapshots for arbitrary-length context

---

# 11. How to Write a NEW Paper From This Work

## 11.1 Reusable Elements

### Ideas You Can Build Upon

- The principle that **selectivity enables content-based reasoning in recurrent models** — apply this to ANY recurrent architecture
- The hardware-aware algorithm design pattern: **identify memory bottleneck → fuse operations → recompute backward pass**
- The evaluation methodology: synthetic task (isolate mechanism) → scaling laws → downstream evaluation
- The architectural pattern of **combining sequential processing with gated MLPs** in a single block

### Evaluation Methodology to Reuse

- Chinchilla-style scaling laws as primary evidence of quality
- Selective Copying and Induction Heads as diagnostic synthetic tasks
- The same downstream benchmark suite (LAMBADA, HellaSwag, PIQA, ARC, WinoGrande)
- Throughput and memory benchmarks for practical efficiency claims

### Experimental Protocol to Follow

1. Show ablation proving your new mechanism works on a synthetic task
2. Show scaling laws on The Pile (or comparable large corpus)
3. Show downstream zero-shot evaluations at matched model size and training tokens
4. Show efficiency benchmarks (throughput, memory) on standardized hardware

## 11.2 What MUST NOT Be Copied

- The exact hardware-aware kernel implementation (covered by their specific code license)
- Their exact training hyperparameters presented as "novel" — clearly cite these as from their work
- The Mamba block architecture diagram verbatim — redraw with your modifications
- Their benchmark numbers as your own baseline — always rerun baselines yourself or cite clearly

## 11.3 How to Design a Novel Extension

### Strategy 1: Fix a Weakness

Pick one weakness from Section 7 (Table 2) and address it rigorously:
- Example: "Mamba requires custom CUDA kernels" → Develop a portable Triton-based implementation and benchmark parity

### Strategy 2: Apply to New Domain

Take the selective SSM mechanism and apply it where Transformers dominate but struggle with efficiency:
- Example: Protein structure prediction with million-residue sequences
- Example: Real-time video understanding with hour-long streams

### Strategy 3: Improve the Selection Mechanism

The current selection is simple linear projections — there is room for more expressive selection:
- Example: Attention-based selection (use local attention window to compute Δ, B, C)
- Example: Multi-scale selection (different SSM heads with different state dimensions)

### Strategy 4: Theoretical Analysis

The paper is mostly empirical — theoretical contributions are highly valued:
- Example: Prove that selective SSMs can represent any function that attention can (or prove separation)
- Example: Derive optimal state dimension N as a function of task complexity

## 11.4 Minimum Publishable Contribution Checklist

- [ ] A clear and novel modification to the selective SSM mechanism or architecture
- [ ] Ablation showing the modification's isolated effect on a synthetic task
- [ ] Scaling law experiments on at least one standard benchmark (Pile, C4, etc.)
- [ ] Downstream evaluation on at least 4–5 standard tasks
- [ ] Efficiency comparison (throughput and/or memory) against Mamba and Transformer baselines
- [ ] Clear articulation of when/why your approach works better than vanilla Mamba

---

# 12. Publication Strategy Guide

## 12.1 Suitable Venues

| Venue | Type | Fit Level | Notes |
|---|---|---|---|
| NeurIPS | Conference | Excellent | Top venue for this type of architectural/algorithmic work |
| ICML | Conference | Excellent | Strong fit for ML methods with theoretical grounding |
| ICLR | Conference | Excellent | Good for representation learning and architecture innovations |
| COLM | Conference | Good | Focused specifically on language models |
| TMLR | Journal | Excellent | For thorough empirical work that may not fit conference page limits |
| ACL/EMNLP | Conference | Good | If the contribution is NLP-specific |

## 12.2 Required Baseline Expectations

For any Mamba-extension paper:
- **Mandatory baselines:** Original Mamba, Transformer++ (LLaMa-style), at least one recent efficient model (RetNet, RWKV, or Mamba-2)
- **Strongly recommended:** Pythia or GPT-Neo (for standardized comparisons), vanilla Transformer (GPT3-style)
- **If claiming long-context improvements:** Hyena/HyenaDNA (for subquadratic baselines)
- **If claiming efficiency improvements:** FlashAttention-2 as the Transformer efficiency baseline

## 12.3 Experimental Rigor Level

- **Minimum:** 3 model sizes, 1 dataset, downstream eval on 4+ tasks
- **Competitive:** 4+ model sizes in scaling laws, 2+ modalities, 6+ downstream tasks, efficiency benchmarks
- **Strong:** Full Chinchilla scaling laws, multiple modalities, comprehensive ablations, open-source code

## 12.4 Common Rejection Reasons

| Rejection Reason | How to Avoid |
|---|---|
| "Incremental over Mamba" | Must show clear qualitative difference, not just marginal perplexity gains |
| "Only tested at small scale" | Include at least one model at ≥1B parameters |
| "Missing baselines" | Always include Transformer++, original Mamba, and at least one concurrent method |
| "No ablation" | Systematically ablate every new component you add |
| "Unfair comparison" | Use same tokenizer, dataset, training tokens, and hyperparameter tuning budget for all models |
| "Limited tasks" | Evaluate on both perplexity AND downstream tasks; include at least one non-language modality if claiming generality |

## 12.5 Increment Needed for Acceptance

- **Perplexity improvement:** ≥0.1 perplexity at 1B+ scale (with matched compute) is notable
- **Downstream improvement:** ≥1–2% average accuracy across benchmark suite
- **Efficiency improvement:** ≥1.5× throughput improvement at matched quality
- **New capability:** Demonstrating a capability that Mamba lacks (e.g., working well on vision without modification, or supporting bidirectional processing) — even without numerical improvement, new capabilities are publishable

---

# 13. Researcher Quick Reference Tables

## 13.1 Key Terminology Table

| Term | Meaning in This Paper |
|---|---|
| SSM | Structured State Space Model (specifically S4 family, not general state space models) |
| S4 | The original structured SSM with LTI (time-invariant) parameters computed via convolution |
| S6 | The authors' selective SSM — S4 with selection mechanism, computed via scan (not convolution) |
| Mamba | The complete neural network architecture using S6 layers in a simplified block design |
| Selection mechanism | Making SSM parameters (Δ, B, C) input-dependent to enable content-based reasoning |
| LTI | Linear Time Invariant — parameters constant across all time steps |
| Selective scan | The hardware-aware parallel scan algorithm for computing S6 recurrence efficiently |
| State expansion | Using hidden state dimension N >> 1 to increase model capacity per channel |
| HBM | High Bandwidth Memory — main GPU memory (large, slow) |
| SRAM | Static Random Access Memory — on-chip GPU memory (small, fast) |
| Kernel fusion | Combining multiple GPU operations into one kernel to avoid HBM round-trips |
| Recomputation | Not saving intermediate states; recomputing them in backward pass to save memory |
| Discretization | Converting continuous-time SSM parameters to discrete-time using step size Δ |
| ZOH | Zero-Order Hold — a specific discretization rule assuming constant input between samples |
| Expansion factor (E) | Ratio of inner dimension to model dimension in the Mamba block (E=2 by default) |

## 13.2 Important Equations Summary

| Equation | Formula | Purpose |
|---|---|---|
| Continuous SSM | h'(t) = A·h(t) + B·x(t); y(t) = C·h(t) | Defines the underlying dynamical system |
| Discrete recurrence | h_t = A̅·h_{t-1} + B̅·x_t; y_t = C·h_t | Actual computation in recurrent mode |
| ZOH discretization | A̅ = exp(Δ·A); B̅ = (Δ·A)⁻¹(exp(Δ·A)−I)·Δ·B | Converts continuous to discrete parameters |
| Convolution kernel | K̅ = (C·B̅, C·A̅·B̅, C·A̅²·B̅, ...) | LTI-only parallel computation mode |
| Selection functions | B_t = Linear(x_t); C_t = Linear(x_t); Δ_t = softplus(bias + Linear(x_t)) | Makes parameters input-dependent |
| Gated RNN (Theorem 1) | h_t = (1−g_t)·h_{t-1} + g_t·x_t | Special case showing SSM selection = RNN gating |

## 13.3 Parameter Meaning Table

| Parameter | Shape (S6) | Learned? | Role |
|---|---|---|---|
| A | (D, N) | Yes (fixed over time) | State transition matrix — controls dynamics of hidden state evolution |
| B | (B, L, N) | Yes (input-dependent) | Input-to-state matrix — controls which inputs enter the state |
| C | (B, L, N) | Yes (input-dependent) | State-to-output matrix — controls which state information reaches output |
| Δ | (B, L, D) | Yes (input-dependent) | Step size / gate — controls focus vs. ignore of current input |
| D | scalar | - | Model dimension — number of channels |
| N | scalar (16) | - | State dimension per channel — controls capacity of recurrent state |
| E | scalar (2) | - | Expansion factor — inner dimension = E × D |

## 13.4 Algorithm Flow Summary

```
MAMBA BLOCK (single pass):
  1. x ∈ (B,L,D) → Linear → (B,L,2ED)
  2. Split into branch_1 (B,L,ED) and branch_2 (B,L,ED)
  3. branch_1 → Conv1d(kernel=4) → SiLU activation
  4. From branch_1, compute:
     - Δ = softplus(param + Linear₁(branch_1))     # (B,L,ED)
     - B = Linear_N(branch_1)                        # (B,L,N)
     - C = Linear_N(branch_1)                        # (B,L,N)
  5. Discretize: A̅ = exp(Δ·A), B̅ = f(Δ,A,B)        # (B,L,ED,N)
  6. Selective scan: h_t = A̅_t·h_{t-1} + B̅_t·x_t    # Parallel scan in SRAM
  7. Output: y_t = C_t · h_t                          # (B,L,ED)
  8. branch_2 → SiLU activation
  9. y = y ⊙ branch_2                                 # Gated multiply
  10. y → Linear → (B,L,D) → Add residual + Norm

FULL MAMBA ARCHITECTURE:
  Embedding → [Mamba Block + Residual + RMSNorm] × num_layers → LM Head
```

---

# 14. One-Page Master Summary Card

## Problem

Transformers have quadratic complexity in sequence length. Existing subquadratic alternatives (S4, Hyena, linear attention) cannot match Transformer quality on language because their time-invariant dynamics prevent content-based reasoning.

## Key Idea

Make SSM parameters (Δ, B, C) **functions of the input** so the model can **selectively** propagate or forget information along the sequence. This enables content-aware reasoning while retaining linear-time complexity.

## Method

1. **Selection mechanism (S6):** B_t, C_t, Δ_t are generated from input via linear projections; this makes the recurrence time-varying
2. **Hardware-aware algorithm:** Fuse discretization + parallel scan + output computation into a single CUDA kernel that operates entirely in SRAM; recompute states in backward pass
3. **Mamba block:** Combine selective SSM with gated MLP into a single repeating block (no attention, no separate MLP block)

## Results

- **Language:** First attention-free model matching Transformer++ quality; Mamba-2.8B exceeds Pythia-6.9B on downstream tasks
- **DNA:** Monotonically improving perplexity up to 1M context; 81% accuracy on Great Apes classification at 1M length
- **Audio:** State-of-the-art generation quality; FID 0.67 on SC09 (previous best: 1.42)
- **Efficiency:** 5× inference throughput over Transformers; 40× faster than naive scan; linear memory scaling

## Weakness

- Untested beyond 2.8B parameters
- Selection hurts on continuous signals (audio waveforms)
- Requires custom CUDA kernels (not portable)
- Downstream affordances (RLHF, ICL depth) not validated
- No evaluation on reasoning-heavy tasks (math, code)

## Research Opportunity

- Scale to 7B+ and validate end-to-end LLM training pipeline
- Adaptive selection that automatically switches between LTI and selective modes per layer/modality
- Develop portable implementations (Triton, XLA)
- Hybrid Mamba-Attention architectures with principled layer allocation
- Apply to multimodal, agentic, and continual learning settings

## Publishable Extension (Example)

> "We propose Adaptive Mamba, which uses a learned per-layer gating mechanism to interpolate between LTI and selective modes, achieving state-of-the-art performance across both discrete (language) and continuous (audio, video) modalities without modality-specific tuning. On The Pile, Adaptive Mamba matches Mamba quality while improving audio BPB by X% and matching Transformer++ quality, all with linear-time inference."
