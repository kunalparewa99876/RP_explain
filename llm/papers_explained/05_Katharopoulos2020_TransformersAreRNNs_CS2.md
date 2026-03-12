# Research Companion: Transformers are RNNs — Fast Autoregressive Transformers with Linear Attention (Katharopoulos et al., 2020)

---

**Paper Classification**: Mathematical / Algorithmic Method  
**Adaptation Mode**: Explain intuition BEFORE equations, explain meaning of symbols, explain theorem purpose, provide workflow logic + pseudocode intuition, focus on design decisions, baselines, metrics

---

# 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Paper Title** | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention |
| **Authors** | Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret |
| **Year** | 2020 |
| **Venue** | ICML 2020 (37th International Conference on Machine Learning) |
| **Problem Domain** | Efficient Transformer Architectures / Sequence Modeling |
| **Paper Type** | Mathematical / Algorithmic Method |
| **Core Contribution** | Reformulates self-attention using kernel feature maps and the associativity of matrix products, reducing complexity from O(N²) to O(N), and reveals that transformers with causal masking are equivalent to RNNs |
| **Key Idea** | Replace the softmax in attention with a kernel-based dot product, then reorder the matrix multiplications so that attention can be computed in linear time — this also lets autoregressive transformers run as fast recurrent networks during inference |
| **Required Background** | Self-attention mechanism (Vaswani et al., 2017), kernel methods (basic), matrix associativity, autoregressive models, recurrent neural networks (RNNs), causal masking, computational complexity (Big-O) |
| **Primary Baseline** | Vanilla Transformer (softmax attention), Reformer (LSH-based attention) |
| **Main Innovation Type** | Algorithmic — replaces the attention computation strategy to achieve linear complexity without changing the transformer's overall architecture |
| **Difficulty Level** | Intermediate-to-Advanced |
| **Reproducibility Level** | High — code publicly released at https://linear-transformers.com/, CUDA kernels provided, standard datasets used |

---

# 1. Research Context & Core Problem

## 1.1 Exact Problem Formulation

- Standard transformers use **self-attention** which compares every token (position) with every other token in the sequence
- This pairwise comparison creates an **N × N attention matrix**, where N is the sequence length
- Both **time and memory** scale as **O(N²)** — quadratic in sequence length
- This means:
  - Training on very long sequences is extremely slow and memory-hungry
  - During **autoregressive inference** (generating one token at a time), each new token requires recomputing attention over ALL previous tokens, and the cost grows quadratically as the generated sequence gets longer
- The paper asks: **Can we reformulate self-attention to run in O(N) time and memory, while keeping performance comparable to standard transformers?**

## 1.2 Why the Problem Exists

- Self-attention's power comes from its **global receptive field** — every position can directly attend to every other position
- But this power has a direct cost: the attention matrix has N² entries, one for every (query, key) pair
- In practice, this limits the context window (the maximum sequence length) that a transformer can process
- For **autoregressive tasks** (e.g., text generation, image pixel-by-pixel generation), this is especially painful because:
  - The model generates one output at a time
  - At each step, it must attend to all previously generated outputs
  - With softmax attention, cost per step grows linearly with current sequence length, and total cost grows quadratically
  - This makes generating long sequences (thousands of tokens/pixels) extremely slow

## 1.3 Historical and Theoretical Gap

- Before this paper, approaches to make transformers faster included:
  - **Sparse attention** (Child et al., 2019): reduces complexity to O(N√N) by only attending to a subset of positions — but introduces sparsity patterns that may miss important connections
  - **Locality-sensitive hashing** (Kitaev et al., 2020 — Reformer): reduces to O(N log N) using hashing to find similar query-key pairs — but forces keys = queries, limiting applicability to decoding tasks
  - **Weight pruning, quantization, distillation**: reduce model size but do NOT change the fundamental O(N²) attention complexity
- **None of these methods** speed up autoregressive inference, because they still require recomputing attention from scratch at each generation step
- The key insight missing was: **if you change HOW the attention is computed (not just which pairs to attend to), you can break the quadratic barrier entirely**

## 1.4 Limitations of Previous Approaches

| Approach | Complexity | Autoregressive Speedup? | Limitations |
|---|---|---|---|
| Vanilla Transformer | O(N²) | No | Prohibitively slow for long sequences |
| Sparse Transformers | O(N√N) | No | Fixed sparsity patterns, may miss dependencies |
| Reformer (LSH) | O(N log N) | No | Keys must equal queries (cannot be used for decoding where keys ≠ queries) |
| Transformer-XL | O(N²) | No | Extends context via memory but keeps quadratic cost |
| Adaptive Attention Span | O(N²) asymptotically | No | Learns per-head span but asymptotic complexity unchanged |

## 1.5 Contribution Category

- **Algorithmic**: new way to compute attention
- **Theoretical**: formal proof that transformers with causal masking are equivalent to RNNs
- **Empirical**: demonstrations on image generation and speech recognition

### Why This Paper Matters

- It provides the **first linear-time, constant-memory formulation** for autoregressive transformers
- It reveals a **deep theoretical connection**: every transformer with causal masking is actually a recurrent neural network — this bridges two paradigms that were considered fundamentally different
- It enables autoregressive inference that is **up to 4,000× faster** than vanilla transformers
- It opens a new research direction: instead of just making attention sparse, you can change the attention kernel to achieve linear scaling

### Remaining Open Problems

- The feature map used (elu + 1) is a simple choice — **finding optimal feature maps** for different tasks remains open
- Linear attention may lose some **representation power** compared to softmax attention, especially for tasks that need sharp, peaked attention distributions
- The paper does not explore **language modeling** at scale — performance on large-scale NLP benchmarks is not tested
- The connection between the RNN formulation and **gating mechanisms** (like LSTM gates) is not fully explored
- **Hybrid approaches** (combining linear and softmax attention in the same model) are not investigated
- The effect on **very long sequences** (e.g., >100K tokens) for practical NLP tasks is not demonstrated

---

# 2. Minimum Background Concepts

## 2.1 Self-Attention (Softmax Attention)

- **Plain definition**: A mechanism where each position in a sequence computes a weighted average of all other positions, with weights determined by how "similar" their representations are
- **Role inside paper**: This is what the authors are trying to speed up — it is the computational bottleneck they target
- **Why authors needed it**: The entire paper is about reformulating THIS specific operation to be faster

### How it works (review)
- Input sequence is projected into three representations: **Queries (Q)**, **Keys (K)**, and **Values (V)**
- For each query, compute a similarity score with every key using dot product, then apply softmax to get weights, then take a weighted sum of values
- The formula for position i is: V'ᵢ = Σⱼ [exp(QᵢᵀKⱼ / √D) / Σⱼ' exp(QᵢᵀKⱼ' / √D)] × Vⱼ
- Computing all N outputs requires an N×N attention matrix → O(N²)

## 2.2 Kernel Methods (Basic)

- **Plain definition**: A kernel is a function k(x, y) that measures similarity between two inputs. Many kernels can be decomposed as k(x, y) = φ(x)ᵀφ(y), where φ is a "feature map" that transforms inputs into a new space
- **Role inside paper**: The authors reinterpret the attention similarity function as a kernel, then use the kernel's feature map decomposition to algebraically rearrange the computation
- **Why authors needed it**: The kernel decomposition is what allows them to break apart the N×N matrix into products that can be computed in O(N) time using the associativity of matrix multiplication

### Key property used
- If similarity = φ(q)ᵀ φ(k), then the attention numerator Σⱼ φ(Qᵢ)ᵀ φ(Kⱼ) Vⱼ can be rewritten as φ(Qᵢ)ᵀ [Σⱼ φ(Kⱼ) Vⱼᵀ]
- The term in brackets is a **single matrix** that can be computed once and reused for all queries
- This is the core trick that eliminates the quadratic dependence on N

## 2.3 Associativity of Matrix Multiplication

- **Plain definition**: For matrices A, B, C, we have (A × B) × C = A × (B × C) — you can choose which multiplication to do first
- **Role inside paper**: By changing the order in which matrices are multiplied in the attention formula, the authors change the complexity from O(N²) to O(N)
- **Why authors needed it**: In standard attention, you first compute Q × Kᵀ (an N×N matrix), then multiply by V. In linear attention, you first compute Kᵀ × V (a D×M matrix, independent of N), then multiply Q by this result

## 2.4 Causal Masking

- **Plain definition**: A constraint used in autoregressive models that prevents position i from attending to any position j > i (future positions)
- **Role inside paper**: The authors show that even with causal masking, linear attention maintains O(N) complexity, and this formulation directly reveals the RNN structure
- **Why authors needed it**: Autoregressive tasks (generation) require causal masking; without handling it, their method would only work for bidirectional tasks (like classification)

## 2.5 Recurrent Neural Networks (RNNs)

- **Plain definition**: A neural network that processes sequences one step at a time, maintaining a hidden state that gets updated at each step: hᵢ = f(hᵢ₋₁, xᵢ)
- **Role inside paper**: The authors prove that their linear transformer with causal masking is mathematically equivalent to an RNN with a specific state update rule — this enables constant-time, constant-memory inference per step
- **Why authors needed it**: This equivalence is the key to fast autoregressive inference — instead of recomputing attention over the full sequence at each step, you just update a small state

## 2.6 Feature Maps

- **Plain definition**: A function φ(x) that transforms an input vector into a (possibly higher-dimensional) representation such that the dot product φ(x)ᵀφ(y) approximates some desired similarity function
- **Role inside paper**: The choice of feature map determines the attention behavior and computational cost of the linear transformer
- **Why authors needed it**: They need a feature map that (a) gives non-negative similarity scores (required for valid attention weights), (b) has finite and low dimensionality (for computational efficiency), and (c) performs well empirically

---

# 3. Mathematical / Theoretical Understanding Layer

## 3.1 Generalized Attention (Equation 3 from paper)

### Intuition
- Standard softmax attention uses exp(qᵀk/√D) as the similarity function
- But you could use ANY non-negative similarity function and still have a valid attention mechanism
- The paper formalizes this: as long as sim(q, k) ≥ 0, the weighted average over values is well-defined

### Formula
$$V'_i = \frac{\sum_{j=1}^{N} \text{sim}(\mathbf{Q}_i, \mathbf{K}_j) \mathbf{V}_j}{\sum_{j=1}^{N} \text{sim}(\mathbf{Q}_i, \mathbf{K}_j)}$$

### Variable Meaning Table

| Variable | Meaning | Dimensions |
|---|---|---|
| Qᵢ | Query vector for position i | D-dimensional |
| Kⱼ | Key vector for position j | D-dimensional |
| Vⱼ | Value vector for position j | M-dimensional |
| V'ᵢ | Output vector for position i | M-dimensional |
| sim(·,·) | Any non-negative similarity function | Scalar output |
| N | Total sequence length | Scalar |

### What problem it solves
- Generalizes the attention mechanism beyond softmax — opens the door to choosing sim() strategically for computational benefits

### Assumptions
- sim(·,·) must be non-negative (otherwise attention weights could be negative, which breaks the "weighted average" interpretation)

## 3.2 Kernel-Based Attention (Equation 4 — The Core Insight)

### Intuition
- If the similarity function is a kernel, it can be decomposed: sim(q, k) = φ(q)ᵀ φ(k)
- Substituting this into the attention formula makes the numerator: Σⱼ [φ(Qᵢ)ᵀ φ(Kⱼ)] Vⱼ
- This is a sum over j of: (scalar) × (vector)
- But we can rewrite it as: φ(Qᵢ)ᵀ [Σⱼ φ(Kⱼ) Vⱼᵀ]
- The key insight: the term in brackets does NOT depend on i — it can be computed ONCE

### Formula (Linearized Attention)
$$V'_i = \frac{\phi(\mathbf{Q}_i)^\top \left[\sum_{j=1}^{N} \phi(\mathbf{K}_j) \mathbf{V}_j^\top \right]}{\phi(\mathbf{Q}_i)^\top \left[\sum_{j=1}^{N} \phi(\mathbf{K}_j) \right]}$$

### What problem it solves
- Reduces the attention computation from O(N²) to O(N)
- The summation Σⱼ φ(Kⱼ) Vⱼᵀ produces a single C×M matrix (where C is the feature map dimension)
- The summation Σⱼ φ(Kⱼ) produces a single C-dimensional vector
- Both are computed once over all j, then each query just does a matrix-vector multiply → O(N) total

### Why this works (the associativity trick)
- **Standard order**: For each query i, compute similarity with all N keys → O(N) per query → O(N²) total
- **New order**: First aggregate all key-value pairs into a summary matrix → O(N) to build it → then each query just reads from this summary → O(1) per query → O(N) total

### Practical interpretation
- Think of Σⱼ φ(Kⱼ) Vⱼᵀ as a **compressed summary** of the entire sequence
- Each query extracts its relevant information from this summary via a simple matrix multiplication
- This is fundamentally different from softmax attention, where each query individually compares against every key

### Limitation of formulation
- The exact softmax kernel exp(qᵀk) has an **infinite-dimensional** feature map — you cannot linearize exact softmax attention
- You must use an **approximate kernel** or a different kernel with a finite-dimensional feature map
- This means linear attention is NOT identical to softmax attention — it is an approximation or alternative

## 3.3 Computational Cost Analysis

### Softmax attention cost
- O(N² × max(D, M)): because you form the full N×N attention matrix

### Linear attention cost
- Feature map computation: O(N × C) where C is the feature map dimension
- Summary matrix Σⱼ φ(Kⱼ)Vⱼᵀ: O(N × C × M)
- Query-summary multiplication: O(N × C × M)
- Total: **O(N × C × M)**

### When is linear attention cheaper?
- When N > C × M (i.e., when sequence length exceeds the product of feature and value dimensions)
- In practice, D and M are typically 32–64 per head, while N can be hundreds or thousands → linear attention wins

### Specific feature map costs

| Feature Map | Feature Dim C | Total Cost | When Favorable |
|---|---|---|---|
| Polynomial degree 2 | D² | O(N × D² × M) | When N > D² |
| elu(x) + 1 (paper's choice) | D | O(N × D × M) | Always when N > D (almost always true) |
| Random Fourier Features | User-defined R | O(N × R × M) | When N > R |

## 3.4 Causal Linear Attention (Equations 8–12)

### Intuition
- For autoregressive tasks, position i can only attend to positions j ≤ i
- With standard attention, you mask the upper triangle of the N×N attention matrix
- With linear attention, you cannot precompute the full summary Σⱼ₌₁ᴺ φ(Kⱼ)Vⱼᵀ because each position needs a DIFFERENT partial sum (only up to its own position)
- Solution: compute the partial sums **incrementally** using a running accumulation

### Key Recurrences
$$S_i = S_{i-1} + \phi(\mathbf{K}_i) \mathbf{V}_i^\top$$
$$Z_i = Z_{i-1} + \phi(\mathbf{K}_i)$$
$$V'_i = \frac{\phi(\mathbf{Q}_i)^\top S_i}{\phi(\mathbf{Q}_i)^\top Z_i}$$

### Variable Meaning Table

| Variable | Meaning | Dimensions | Update Rule |
|---|---|---|---|
| Sᵢ | Accumulated key-value summary up to position i | C × M matrix | Sᵢ = Sᵢ₋₁ + φ(Kᵢ) Vᵢᵀ |
| Zᵢ | Accumulated key normalizer up to position i | C-dim vector | Zᵢ = Zᵢ₋₁ + φ(Kᵢ) |
| V'ᵢ | Output at position i | M-dim vector | φ(Qᵢ)ᵀ Sᵢ / φ(Qᵢ)ᵀ Zᵢ |

### What problem it solves
- Enables causal (autoregressive) attention in O(N) time
- Each position's output depends only on the current summary Sᵢ and normalizer Zᵢ
- The summary can be updated in O(C × M) — constant with respect to N

### Why this is powerful
- During inference, you just maintain Sᵢ and Zᵢ as hidden states
- Each new token updates S and Z in constant time, then computes output in constant time
- This is EXACTLY how an RNN works → transformers with causal linear attention ARE recurrent networks

## 3.5 The Transformer-RNN Equivalence (Equations 16–20)

### Intuition
- The causal linear attention formulation reveals that each transformer layer has:
  - A **state** (the accumulated S and Z matrices)
  - An **update rule** (add the new key-value pair to the state)
  - An **output rule** (query the state to produce output)
- This is the definition of a recurrent neural network
- Therefore: **every transformer with causal masking is an RNN**

### Formal RNN Formulation

For each transformer layer, given input xᵢ at timestep i:

1. **Compute projections**: qᵢ = Wq × xᵢ, kᵢ = Wk × xᵢ, vᵢ = Wv × xᵢ
2. **Update attention memory**: sᵢ = sᵢ₋₁ + φ(kᵢ) × vᵢᵀ
3. **Update normalizer memory**: zᵢ = zᵢ₋₁ + φ(kᵢ)
4. **Compute attention output**: v'ᵢ = sᵢᵀ φ(qᵢ) / zᵢᵀ φ(qᵢ)
5. **Apply feedforward**: yᵢ = f(v'ᵢ)

### What this means for research
- Transformers and RNNs are NOT fundamentally different architectures — one is a special case of the other
- RNNs can theoretically be as powerful as transformers if given the right state update mechanism
- This opens a unified framework for studying information storage and retrieval in both model families

### Mathematical Insight Box
> **Key insight for researchers**: The quadratic cost of attention comes from computing all N² pairwise interactions. By decomposing the similarity function as φ(q)ᵀ φ(k), you can aggregate first and query second. When combined with causal masking, this aggregation becomes a running sum — which is exactly a recurrent state. This means the difference between transformers and RNNs is NOT about architecture but about the ATTENTION KERNEL used.

## 3.6 Gradient Computation for Causal Linear Attention (Equations 13–15)

### Intuition
- A naive implementation would store all intermediate Sᵢ matrices for backpropagation → this multiplies memory by max(D, M)
- The authors derive gradients that can also be computed as **cumulative sums** (forward and backward), requiring only O(1) extra memory per step

### Key Gradient Formulas
- ∇φ(Qᵢ)L = ∇V̄ᵢL × Sᵢᵀ (forward cumulative sum)
- ∇VᵢL = Sᵀ φ(Kᵢ), where S accumulates φ(Qᵢ)Gᵢᵀ in reverse order (backward cumulative sum)
- ∇φ(Kᵢ)L = S × Vᵢ, with the same reverse accumulation

### What problem it solves
- Enables training causal linear attention with **O(N × C × M) time** and **O(N × max(C, M)) memory**
- Without this trick, memory would be impractical for long sequences or deep models

### Practical interpretation
- The forward pass accumulates key-value pairs left to right
- The backward pass accumulates query-gradient pairs right to left
- Both are simple cumulative sums → implemented efficiently in ~200 lines of CUDA code

---

# 4. Proposed Method / Framework (MOST IMPORTANT)

## 4.1 Overall Pipeline

The linear transformer replaces ONLY the attention computation inside each transformer layer. Everything else (embeddings, feedforward layers, layer normalization, positional encoding) stays the same.

### Step-by-step pipeline:

**Step 1: Input Projection (unchanged)**
- Project input x into Queries Q, Keys K, Values V using learned weight matrices
- Q = xWQ, K = xWK, V = xWV

✔ **Why authors did this**: Standard transformer projection — provides the query/key/value representations needed for attention  
✔ **Weakness**: No weakness at this step — it is standard  
✔ **Improvement idea**: Could explore different projection strategies (e.g., shared projections) to reduce parameters

**Step 2: Feature Map Application (NEW)**
- Apply feature function φ(·) to each query and key vector
- φ(x) = elu(x) + 1

✔ **Why authors did this**: The feature map ensures all similarity values are non-negative (required for valid attention weights) and has the same dimensionality as the input (D), keeping computation efficient  
✔ **Why elu instead of relu**: relu would set gradients to zero for negative inputs, causing dead neurons; elu smoothly transitions for negative inputs, maintaining gradient flow  
✔ **Weakness**: This is a simple, hand-chosen feature map — it may not approximate the representational power of softmax attention well for all tasks  
✔ **Improvement idea (research seed)**: Learn the feature map end-to-end; use random Fourier features for better softmax approximation; use polynomial feature maps for higher expressiveness

**Step 3: Compute Summary Matrices (NEW — the linearization)**
- For **non-causal** (bidirectional) tasks:
  - Compute Σⱼ φ(Kⱼ) Vⱼᵀ → a single C×M matrix
  - Compute Σⱼ φ(Kⱼ) → a single C-dimensional vector
  - These summarize the entire sequence
- For **causal** (autoregressive) tasks:
  - Compute running sums: Sᵢ = Sᵢ₋₁ + φ(Kᵢ)Vᵢᵀ and Zᵢ = Zᵢ₋₁ + φ(Kᵢ)

✔ **Why authors did this**: This is the central innovation — by aggregating key-value pairs into a fixed-size summary, the quadratic N×N attention matrix is never formed  
✔ **Weakness**: The summary matrix has fixed capacity (C×M entries) regardless of sequence length — for very long sequences, this may bottleneck the amount of information that can be stored and retrieved  
✔ **Improvement idea (research seed)**: Use higher-dimensional feature maps for longer sequences; introduce forgetting mechanisms to prioritize recent information; combine with sparse attention for critical positions

**Step 4: Query the Summary (NEW)**
- For each query position i:
  - V'ᵢ = φ(Qᵢ)ᵀ Sᵢ / φ(Qᵢ)ᵀ Zᵢ (causal case)
  - V'ᵢ = φ(Qᵢ)ᵀ S / φ(Qᵢ)ᵀ Z (non-causal case)

✔ **Why authors did this**: Each query retrieves its output independently from the summary, in O(CM) time per query  
✔ **Weakness**: All queries read from the same summary — there is no position-specific weighting beyond what φ captures  
✔ **Improvement idea**: Add positional bias to the query-summary interaction; use different summaries for different heads or layers

**Step 5: Feedforward + Residual (unchanged)**
- Apply feedforward network and residual connections as in standard transformers

✔ **Why authors did this**: Standard transformer layer components — these provide per-position nonlinear transformations  
✔ **No change needed here**

## 4.2 Simplified Pseudocode

### Non-causal (bidirectional) linear attention:
```
Input: φ(Q) [N×C], φ(K) [N×C], V [N×M]

# Compute summary matrix (once for entire sequence)
S = sum over j of: φ(K_j) * V_j^T          # [C×M] matrix
Z = sum over j of: φ(K_j)                   # [C] vector

# Compute output for each query
for i = 1 to N:
    V'_i = φ(Q_i)^T * S / (φ(Q_i)^T * Z)   # [M] vector

return V'
```

### Causal (autoregressive) linear attention:
```
Input: φ(Q) [N×C], φ(K) [N×C], V [N×M]

S = 0   # [C×M] matrix (attention memory)
Z = 0   # [C] vector (normalizer memory)

for i = 1 to N:
    # Update state with new key-value pair
    S = S + φ(K_i) * V_i^T
    Z = Z + φ(K_i)
    
    # Compute output by querying state
    V'_i = φ(Q_i)^T * S / (φ(Q_i)^T * Z)

return V'
```

### Autoregressive INFERENCE (RNN mode):
```
# Initialize states
s = 0   # [C×M] per layer
z = 0   # [C] per layer

for each new input token x_i:
    # Project
    q_i, k_i, v_i = project(x_i)
    
    # Update state (constant time!)
    s = s + φ(k_i) * v_i^T
    z = z + φ(k_i)
    
    # Compute output (constant time!)
    v'_i = s^T * φ(q_i) / (z^T * φ(q_i))
    
    # Feedforward
    y_i = feedforward(v'_i)

return y_i
```

## 4.3 The Feature Map Choice

- The paper uses: **φ(x) = elu(x) + 1**
- elu(x) = x if x > 0, α(eˣ - 1) if x ≤ 0 (with α = 1)
- Adding 1 ensures φ(x) > 0 for all x → guarantees non-negative similarity
- This produces a feature map with the SAME dimensionality as the input (C = D)
- Computational cost: O(NDM) — same order as a single feedforward layer

### Why not other feature maps?

| Feature Map | Pros | Cons |
|---|---|---|
| Exact softmax kernel | Matches standard attention exactly | Infinite-dimensional → not computable |
| Polynomial (degree 2) | Exact finite-dim feature map, proven to work | C = D², expensive for large D |
| Random Fourier Features | Can approximate RBF/softmax kernels | Requires choosing number of features, introduces randomness |
| elu(x) + 1 (chosen) | Simple, same dim as input, good empirical results | Not a known kernel approximation, ad-hoc |
| relu(x) | Simple, non-negative | Zero gradients for x < 0, can cause dead neurons |

## 4.4 Training vs. Inference Comparison

| Aspect | Standard Transformer | Linear Transformer |
|---|---|---|
| **Training** | Parallelizable across positions, O(N²) per layer | Parallelizable across positions, O(N) per layer |
| **Inference (per step)** | Must attend to all previous positions, O(i) at step i | Update state + query, O(1) at step i |
| **Inference (total for N steps)** | O(N²) | O(N) |
| **Memory during inference** | Stores all previous keys/values: O(N) | Stores only S and Z: O(C×M) = constant |

---

# 5. Experimental Setup / Evaluation Design

## 5.1 Datasets Used

| Dataset | Task | Sequence Length | Type |
|---|---|---|---|
| Synthetic copy task | Sequence duplication | 128 tokens | Synthetic |
| MNIST | Autoregressive image generation | 784 pixels | Real-world |
| CIFAR-10 | Autoregressive image generation | 3,072 pixels (32×32×3) | Real-world |
| WSJ (80 hours) | Automatic speech recognition | 800 frames avg, 2,400 max | Real-world |

## 5.2 Experimental Protocol

### Experiment 1: Synthetic Copy Task (§4.1.1)
- **Purpose**: Test convergence — does linear attention converge stably to the same loss as softmax?
- **Setup**: 4 layers, 8 heads, batch size 64, RAdam optimizer, learning rate 10⁻³ → 10⁻⁴ after 3000 updates
- **Comparison**: softmax vs. linear vs. Reformer (lsh-1, lsh-4)

### Experiment 2: Memory and Computational Requirements (§4.1.2)
- **Purpose**: Measure actual (not just theoretical) time and memory scaling
- **Setup**: Vary N ∈ {2⁹, 2¹⁰, ..., 2¹⁶}, scale batch size inversely with N, measure peak GPU memory and time
- **Hardware**: NVidia GTX 1080 Ti (11 GB)
- **Comparison**: Maximum N that fits in memory for each method

### Experiment 3: MNIST Image Generation (§4.2.1)
- **Purpose**: End-to-end autoregressive performance comparison on a standard benchmark
- **Setup**: 8 layers, 8 heads, embedding size 256, feedforward dim 1024, mixture of 10 logistics output
- **Training**: 250 epochs, RAdam, learning rate 10⁻⁴, batch size 10 for all methods
- **Metric**: Bits per dimension (lower is better) and images/second throughput

### Experiment 4: CIFAR-10 Image Generation (§4.2.2)
- **Purpose**: Test on longer sequences (3,072 vs 784 pixels) to show scaling advantage
- **Setup**: 16 layers, same per-layer config as MNIST
- **Training**: 7 days on a single GPU (NVidia P40, 24 GB)
- **Key constraint**: Softmax could only use batch size 1; linear and Reformer used batch size 4
- **Metric**: Bits per dimension and images/second throughput

### Experiment 5: Automatic Speech Recognition (§4.3)
- **Purpose**: Show the method works for non-autoregressive tasks too
- **Setup**: WSJ dataset, 9 layers, 6 heads, CTC loss, also compared to bidirectional LSTM (3 layers, hidden 320)
- **Metric**: Phoneme Error Rate (PER) and training time per epoch

## 5.3 Metrics Used and Why

| Metric | Why Used | What It Measures |
|---|---|---|
| Bits/dimension | Standard for generative models — measures how well the model predicts each element | Lower = model assigns higher probability to data |
| Images/second | Directly measures inference throughput | Higher = faster generation |
| Phoneme Error Rate (PER) | Standard for speech recognition | Lower = fewer predicted phoneme errors |
| Training time/epoch | Measures training efficiency | Lower = faster training |
| Peak GPU memory | Measures memory footprint | Lower = can handle longer sequences or bigger batches |

## 5.4 Baseline Selection Logic

- **Softmax (vanilla transformer)**: The gold standard — shows the performance ceiling that linear attention should aim for
- **Reformer (LSH-1, LSH-4)**: The state-of-the-art efficient transformer at the time — shows whether linear attention is competitive with the best existing acceleration method
- **Bi-LSTM** (speech only): A classic recurrent baseline — shows whether transformers (of any kind) are better than RNNs for this task

## 5.5 Hyperparameter Reasoning

- Embedding size 256 (32 per head × 8 heads): Standard moderate-size transformer configuration
- Feedforward = 4× embedding: Standard transformer ratio (Vaswani et al., 2017)
- Learning rate 10⁻⁴ with RAdam: Adam variant that removes warmup requirement
- Reformer uses 64 buckets, ~32 elements per chunk: As recommended in the Reformer paper
- Batch size kept constant per experiment: Ensures fair comparison (avoids larger-batch advantages)

## 5.6 Hardware / Compute Assumptions

- Memory experiments: NVidia GTX 1080 Ti (11 GB)
- CIFAR-10: NVidia P40 (24 GB)
- All experiments on single GPUs — no multi-GPU setups

### Experimental Reliability Analysis

**What is trustworthy**:
- The convergence experiment clearly shows linear attention converges to the same loss as softmax
- The memory/time scaling curves match theory (O(N) vs O(N²))
- The throughput improvements (300× for MNIST, 4000× for CIFAR-10) are dramatic and consistent
- WSJ speech recognition shows the method generalizes beyond autoregressive tasks

**What is questionable**:
- CIFAR-10: Training for a fixed 7 days (not to convergence) makes the perplexity comparison confounded by training speed differences — linear transformer completed 3× more epochs
- No language modeling experiments — the most common transformer application is absent
- No large-scale experiments (e.g., large corpus NLP, long document modeling)
- Memory errors on pages 12-17 suggest some complex tables/figures may not be perfectly captured
- The feature map choice (elu+1) is not theoretically justified — it is an empirical choice
- No ablation on feature map choice across tasks

---

# 6. Results & Findings Interpretation

## 6.1 Main Outcomes

### Convergence (Synthetic Task)
- **Linear attention converges stably** and reaches the **same final loss** as softmax attention
- Reformer (LSH) converges to a slightly higher loss due to noise introduced by the hashing approximation
- This validates that linear attention does not sacrifice convergence properties

### Memory and Time Scaling
- Softmax attention: memory and time grow **quadratically** with N — maxes out at N=4,096 on an 11 GB GPU
- Linear attention: memory and time grow **linearly** with N — can handle N=65,536+ on the same GPU
- Reformer: also scales linearly (O(N log N) but log N is small) — but linear attention is faster at every point

### MNIST Image Generation

| Method | Bits/dim | Images/sec | Speedup |
|---|---|---|---|
| Softmax | 0.621 | 0.45 | 1× |
| LSH-1 | 0.745 | 0.68 | 1.5× |
| LSH-4 | 0.676 | 0.27 | 0.6× |
| **Linear (proposed)** | **0.644** | **142.8** | **317×** |

- Linear attention achieves nearly the same bits/dim as softmax (0.644 vs 0.621 — only 0.023 difference)
- It generates images **317× faster** — can simultaneously generate 10,000 MNIST images on a single GPU
- Reformer actually makes things slower (LSH-4: 0.27 img/s, worse than softmax's 0.45)

### CIFAR-10 Image Generation

| Method | Bits/dim | Images/sec | Speedup |
|---|---|---|---|
| Softmax | 3.47 | 0.004 | 1× |
| LSH-1 | 3.39 | 0.015 | 3.75× |
| LSH-4 | 3.51 | 0.005 | 1.25× |
| **Linear (proposed)** | **3.40** | **17.85** | **4,462×** |

- Linear transformer achieves **better bits/dim than softmax** (3.40 vs 3.47) because it completed 3× more epochs in the same wall-clock time
- Throughput improvement grows with sequence length: from 317× on MNIST (784 pixels) to **4,462× on CIFAR-10** (3,072 pixels)
- For every 1 image softmax generates, linear generates 4,460 images

### Automatic Speech Recognition (WSJ)

| Method | Validation PER | Time/epoch (s) |
|---|---|---|
| Bi-LSTM | 10.94 | 1,047 |
| Softmax | 5.12 | 2,711 |
| LSH-4 | 9.33 | 2,250 |
| **Linear (proposed)** | **8.08** | **824** |

- Linear transformer outperforms both LSTM and Reformer, and is the **fastest to train** (824s vs 2,711s for softmax)
- Softmax achieves the lowest PER (5.12) but is 3.3× slower to train
- Linear is 2.9 percentage points behind softmax — a meaningful gap for speech recognition
- Reformer performs surprisingly poorly here (9.33 PER)

## 6.2 Performance Trends

- The speedup of linear attention over softmax **grows with sequence length** — as sequences get longer, the advantage becomes more dramatic
- The quality gap (bits/dim or PER) is small but consistent — linear attention is slightly worse than softmax on quality metrics
- Reformer consistently underperforms on quality metrics (especially on MNIST and WSJ), suggesting that LSH hashing introduces too much noise for these tasks

## 6.3 Failure Cases

- **Speech recognition**: 8.08 PER vs 5.12 PER for softmax — linear attention loses ~3 percentage points. This suggests that for tasks where precise, sharp attention patterns are important, linear attention may not fully substitute for softmax
- **No language modeling results**: The paper does not test on text generation or language modeling, which are the most common transformer applications. This is a notable absence

## 6.4 Unexpected Observations

- **Reformer is often slower or worse than expected**: LSH-4 on MNIST is slower than softmax (0.27 vs 0.45 images/sec), and its quality is much worse (0.676 vs 0.621 bits/dim)
- **Linear attention works on non-autoregressive tasks too**: The WSJ experiment shows the linearization benefits training speed even when not using the RNN formulation

## 6.5 Statistical Meaning

- No error bars or confidence intervals are reported — this is a limitation
- Single-run results make it hard to judge whether small differences (e.g., 0.621 vs 0.644 bits/dim) are statistically significant
- The throughput comparisons are deterministic (measuring wall-clock time) and are highly reliable

### Publishability Strength Check

**Publication-grade results**:
- The O(N) complexity proof and the transformer-RNN equivalence are strong theoretical contributions
- The 317× to 4,462× speedup on image generation is dramatic and convincing
- The method is simple, principled, and easy to implement

**Results needing stronger validation**:
- No NLP benchmarks (language modeling, translation, summarization)
- No error bars or repeated runs
- Speech recognition gap (5.12 vs 8.08 PER) is significant — the method underperforms substantially
- No comparison with other efficient attention methods beyond Reformer (e.g., Longformer, Linformer)

---

# 7. Strengths – Weaknesses – Assumptions

## 7.1 Technical Strengths

| # | Strength | Evidence |
|---|---|---|
| 1 | Reduces attention complexity from O(N²) to O(N) with a principled mathematical derivation | Theoretical proof + empirical timing verification |
| 2 | Enables constant-time, constant-memory autoregressive inference | The RNN formulation stores only S and Z matrices, not all past keys/values |
| 3 | Reveals deep connection between transformers and RNNs | Mathematical proof that causal attention = recurrent computation |
| 4 | Drop-in replacement — only changes the attention mechanism, not the overall architecture | Compatible with any existing transformer codebase |
| 5 | Custom CUDA kernels for gradient computation enable practical deployment | ~200 lines of CUDA, publicly released |
| 6 | Dramatic empirical speedups (up to 4,462×) on real tasks | MNIST and CIFAR-10 generation experiments |
| 7 | Works for both causal (autoregressive) and non-causal tasks | Demonstrated on both image generation and speech recognition |

## 7.2 Explicit Weaknesses

| # | Weakness | Impact |
|---|---|---|
| 1 | Feature map choice (elu+1) is ad-hoc — no theoretical guarantee it approximates softmax well | May lose representational power for tasks needing sharp attention |
| 2 | 3 percentage point PER gap in speech recognition (8.08 vs 5.12) | Linear attention is not a free lunch — quality drops on some tasks |
| 3 | No language modeling or NLP experiments | Leaves the most common transformer application untested |
| 4 | Fixed-size summary matrix may become an information bottleneck for very long sequences | The C×M matrix must encode ALL historical context — capacity is limited |
| 5 | No forgetting mechanism — all past information is weighted equally in the running sum | In practice, recent context is often more relevant than distant context |
| 6 | No error bars or statistical significance testing | Cannot distinguish real improvements from noise |
| 7 | Training speed advantage is confounded with quality (CIFAR-10 trained for same wall-clock time) | Linear transformer did 3× more epochs — hard to separate speed vs. quality effect |

## 7.3 Hidden Assumptions

| # | Assumption | Why It May Not Hold |
|---|---|---|
| 1 | A low-dimensional feature map can capture the relevant structure of softmax attention | Softmax's infinite-dim kernel may encode important patterns that elu+1 misses |
| 2 | The running-sum accumulation acts as an effective memory for autoregressive tasks | Without gating or forgetting, the running sum may be dominated by older information |
| 3 | Tasks tested (image generation, speech) generalize to NLP and other domains | Language modeling may require sharper attention patterns that linear attention cannot produce |
| 4 | Short-to-medium sequence experiments (≤3072) predict performance on truly long sequences (>100K) | Very long sequences may expose the information bottleneck of the fixed-size summary |
| 5 | Single-GPU, moderate-scale experiments are representative of large-scale deployment | Large-scale training on many GPUs may have different scaling characteristics |

---

# 8. Weakness → Research Direction Mapping

| Identified Weakness | Why it Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| Ad-hoc feature map (elu+1) with no theoretical guarantee | Exact softmax kernel has infinite-dim feature map, so some approximation is needed | **Learn optimal feature maps** for specific tasks or datasets | Train the feature map φ as a small neural network end-to-end; use meta-learning to select feature maps |
| No forgetting mechanism in the running sum | The formulation treats all past positions equally because Sᵢ just accumulates | **Introduce exponential decay or gating** into the state update | Replace Sᵢ = Sᵢ₋₁ + φ(Kᵢ)Vᵢᵀ with Sᵢ = λSᵢ₋₁ + φ(Kᵢ)Vᵢᵀ where λ is learned; or use gating like LSTM |
| Fixed-capacity summary matrix bottleneck | The C×M summary must encode all context regardless of sequence length | **Adaptive-capacity summaries** that grow with sequence complexity | Use dynamic rank updates, hierarchical summaries, or periodic compression of the state |
| Quality gap on speech recognition | Linear attention cannot produce sharp, peaked attention distributions | **Hybrid attention**: combine linear attention (for most positions) with sparse softmax (for critical positions) | Use linear attention by default but allow top-k softmax attention for a small number of positions |
| No NLP benchmarks tested | Paper focused on image generation and speech as demonstrations | **Evaluate and adapt linear attention for language modeling** | Apply linear transformers to GPT-style language modeling with proper comparison |
| No multi-scale or hierarchical attention | Each layer uses the same flat linear attention | **Multi-resolution linear attention** with different feature maps at different layers | Use coarse feature maps in early layers, fine-grained in later layers |
| Equal weighting of all past positions | Running sum accumulation has no recency bias | **Position-aware feature maps** | Incorporate relative or learnable positional information into the feature map φ |

---

# 9. Novel Contribution Extraction

## 9.1 Explicit Novel Claim Statements

1. **"We propose a kernel-based reformulation of self-attention that reduces the computational and memory complexity from O(N²) to O(N) by exploiting the associativity of matrix multiplication."**

2. **"We propose a causal linear attention mechanism that enables autoregressive transformers to perform inference with constant time and memory per step, revealing their equivalence to recurrent neural networks."**

3. **"We demonstrate that linear transformers achieve comparable performance to vanilla transformers on image generation while providing up to 4,462× faster inference throughput."**

## 9.2 Possible Novel Claim Templates Inspired by This Paper

1. "We propose a **gated linear attention** mechanism that improves upon linear transformers by introducing learnable forgetting gates, achieving better performance on long-sequence language modeling while maintaining O(N) complexity."

2. "We propose a **learned feature map** for linear attention that is trained end-to-end via a meta-learning objective, closing the quality gap with softmax attention on [task] by X%."

3. "We propose a **hybrid attention architecture** that selectively applies softmax attention to a small fraction of critical positions and linear attention elsewhere, achieving softmax-level quality with near-linear-attention speed."

4. "We propose **hierarchical linear attention** that maintains multiple summary matrices at different temporal scales, enabling efficient processing of sequences exceeding 100K tokens without information loss."

5. "We propose **decaying linear attention** where the state update includes a position-dependent exponential decay factor, improving the ability of linear transformers to model local patterns while retaining efficient long-range access."

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Work

- **Better feature maps**: Approximating the RBF kernel using random Fourier features, which could allow using models pretrained with softmax attention
- **Understanding information storage**: The RNN equivalence opens study of how transformers store and retrieve information, connecting to research on RNN memory mechanisms

## 10.2 Missing Directions (Not Explored by Authors)

- **Language modeling at scale**: GPT-style autoregressive language modeling is the dominant use case for causal attention — conspicuously absent
- **Bidirectional NLP tasks**: BERT-style masked language modeling, question answering, summarization
- **Very long sequences** (>100K tokens): The theoretical advantage is greatest here, but no experiment tests it
- **Multi-head analysis**: How do individual attention heads in linear transformers differ from those in softmax transformers?
- **Attention pattern visualization**: What patterns does linear attention learn? Are they meaningful?
- **Combination with other efficient methods**: Could linear attention be combined with sliding window attention (Longformer) or sparse patterns?

## 10.3 Modern Extensions (Post-2020)

- **State Space Models (S4, Mamba)**: These can be seen as sophisticated extensions of the "transformer as RNN" idea, with structured state transitions and selective updates
- **RetNet**: Microsoft's Retentive Network explicitly uses the running-sum formulation with exponential decay, addressing the "no forgetting" weakness
- **RWKV**: A linear attention transformer that uses time-mixing and channel-mixing to achieve competitive performance with full transformers on language modeling
- **Flash Attention**: Hardware-aware attention computation that makes softmax attention much faster — changes the practical trade-off with linear attention
- **Gated Linear Attention (GLA)**: Adds gating mechanisms to the state update, directly addressing the weakness identified in this paper

## 10.4 Cross-Domain Combinations

- **Linear attention for robotics**: Real-time decision-making in embodied agents where inference speed is critical
- **Linear attention for genomics**: DNA/protein sequences can be extremely long (>100K) — ideal for linear complexity
- **Linear attention for video**: Video frames create very long sequences — linear attention could enable efficient video transformers
- **Linear attention for real-time systems**: The constant-memory, constant-time RNN formulation is ideal for streaming data (IoT, sensor networks)

## 10.5 LLM-Era Extensions

- **Linear attention for efficient LLM inference**: Could linear attention reduce the KV cache bottleneck in serving large language models?
- **Hybrid LLM architectures**: Mix linear attention layers (for most of the model) with a few softmax attention layers (for critical reasoning steps)
- **Linear attention for long-context LLMs**: Models like GPT-4 use 128K context — linear attention could make this cheaper
- **Speculative decoding with linear attention**: The fast RNN inference could serve as the "draft" model in speculative decoding

---

# 11. How to Write a NEW Paper From This Work

## 11.1 Reusable Elements

### Ideas you can build upon:
- The kernel-based attention framework — extend it with different kernels
- The transformer-as-RNN perspective — use it to design new hybrid architectures
- The cumulative-sum gradient trick — apply it to other recurrent formulations
- The feature map design space — propose and evaluate new feature maps

### Evaluation methodology patterns:
- The synthetic convergence test (copy task) is a great sanity check for any new attention method
- The memory/time scaling benchmark (varying N on fixed GPU) clearly shows complexity
- The throughput measurement (images/second) is a compelling metric for autoregressive tasks
- Training for a fixed wall-clock time (instead of fixed epochs) is useful when comparing methods of different speed

### Presentation patterns:
- The "reformulate → prove equivalence → demonstrate empirically" structure is very clean
- Starting with theory, then showing practical implications (speed, memory) is persuasive
- Including pseudocode (Algorithm 1) makes the method reproducible

## 11.2 What MUST NOT Be Copied

- The specific feature map φ(x) = elu(x) + 1 cannot be claimed as a new contribution
- The kernel-based attention derivation is from Tsai et al. (2019) — the LINEARIZATION trick is from this paper
- The exact experimental setup, numbers, and comparisons
- The mathematical derivations and proof structure

## 11.3 How to Design a Novel Extension

### Step 1: Pick a weakness from Section 8
- e.g., "No forgetting mechanism in the running sum"

### Step 2: Propose a concrete solution
- e.g., "Introduce a learnable decay factor λᵢ that controls how much of the past state is retained at each position"
- Formalize: Sᵢ = λᵢ Sᵢ₋₁ + φ(Kᵢ)Vᵢᵀ

### Step 3: Ensure the solution preserves the key advantage
- Does it still have O(N) complexity? Yes — the update is still constant-time per step
- Can it still be used as an RNN during inference? Yes — just store S and λ

### Step 4: Design experiments that demonstrate the improvement
- Show the decay helps on long-range tasks where the original running sum fails
- Compare against both softmax attention AND the original linear attention
- Use established benchmarks (language modeling perplexity, PER for speech)

### Step 5: Frame the contribution clearly
- "We identify that the running-sum accumulation in linear attention lacks a forgetting mechanism, leading to degraded performance on tasks with long-range dependencies. We propose Decayed Linear Attention (DLA) that introduces..."

## 11.4 Minimum Publishable Contribution Checklist

- [ ] A novel modification to linear attention that addresses a clearly identified weakness
- [ ] Mathematical proof or clear derivation that the modification preserves O(N) complexity
- [ ] Synthetic experiment showing convergence
- [ ] At least 2 real-world benchmarks showing improvement over both linear AND softmax attention
- [ ] Ablation study showing each component of the modification matters
- [ ] Comparison with at least 3 baselines (softmax, original linear, one other efficient method)
- [ ] Analysis of when the modification helps and when it does not
- [ ] Released code and documentation

---

# 12. Publication Strategy Guide

## 12.1 Suitable Conference/Journal Types

| Venue Type | Examples | Fit Level |
|---|---|---|
| Top ML conferences | ICML, NeurIPS, ICLR | Best fit — this paper was at ICML 2020 |
| NLP conferences | ACL, EMNLP, NAACL | Good fit if you include NLP experiments |
| Efficient ML workshops | EfficientNLP, Edge ML, Efficient Transformers | Good for preliminary or focused results |
| Systems ML | MLSys | Good if focusing on the systems/inference aspect |
| Journals | JMLR, TMLR | For more thorough, extended versions |

## 12.2 Required Baseline Expectations

For a follow-up paper to be accepted at a top venue, you MUST compare against:
- **Vanilla softmax transformer** (the gold standard)
- **Original linear transformer** (Katharopoulos et al., 2020 — this paper)
- **At least 2 modern efficient attention methods**: e.g., Performer, Longformer, Flash Attention, RetNet, Mamba
- **A standard RNN/LSTM** (to validate the "better than RNNs" claim still holds)

## 12.3 Experimental Rigor Level

- **Required**: Multiple runs with standard deviations or confidence intervals
- **Required**: At least one NLP benchmark (language modeling perplexity is standard)
- **Required**: Memory and throughput benchmarks, not just quality metrics
- **Strongly recommended**: Ablation studies for each design choice
- **Strongly recommended**: Scaling analysis (vary N from small to very large)

## 12.4 Common Rejection Reasons

| Rejection Reason | How to Avoid |
|---|---|
| "Just a small modification to existing work" | Frame clearly how your modification addresses a fundamental limitation, not just an engineering tweak |
| "No NLP experiments" (this paper's weakness) | Include language modeling perplexity on at least WikiText-103 or similar |
| "No comparison with recent methods" | Include post-2020 baselines (RetNet, Mamba, Flash Attention) |
| "Missing error bars" | Run every experiment at least 3 times with different seeds |
| "Unclear when the method helps vs. hurts" | Include failure analysis — show when and why the method underperforms |
| "The quality gap is too large" | If you cannot match softmax quality, clearly characterize the trade-off (speed vs. quality Pareto frontier) |

## 12.5 Increment Needed for Acceptance

- **Minimum**: A new feature map or state update mechanism that closes the quality gap on one task domain while preserving linear complexity
- **Strong**: A new formulation that matches softmax quality on NLP benchmarks with O(N) complexity
- **Very strong**: A theoretical result explaining WHEN and WHY linear attention loses quality, with a principled fix

---

# 13. Researcher Quick Reference Tables

## 13.1 Key Terminology Table

| Term | Definition in This Paper |
|---|---|
| Linear Attention | Attention using kernel feature maps instead of softmax, enabling O(N) computation |
| Feature Map φ(·) | A function that transforms queries/keys such that φ(q)ᵀφ(k) defines a positive similarity |
| Attention Memory (S) | The running sum Σ φ(Kⱼ)Vⱼᵀ — acts as the hidden state in the RNN interpretation |
| Normalizer Memory (Z) | The running sum Σ φ(Kⱼ) — used to normalize the attention weights |
| Causal Masking | Constraint that position i can only attend to positions j ≤ i |
| Associativity Trick | Reordering matrix multiplications: (Q × Kᵀ) × V → Q × (Kᵀ × V) to avoid the N×N matrix |
| elu(x) + 1 | The specific feature map used: Exponential Linear Unit shifted by 1 to ensure positivity |
| Reformer / LSH | Baseline method using locality-sensitive hashing to approximate attention in O(N log N) |
| Bits per dimension | Evaluation metric for generative models: average negative log-likelihood per data dimension |

## 13.2 Important Equations Summary

| Equation | Name | Purpose |
|---|---|---|
| V'ᵢ = Σⱼ sim(Qᵢ,Kⱼ)Vⱼ / Σⱼ sim(Qᵢ,Kⱼ) | Generalized Attention | Defines attention with any non-negative similarity |
| V'ᵢ = φ(Qᵢ)ᵀ[Σⱼ φ(Kⱼ)Vⱼᵀ] / φ(Qᵢ)ᵀ[Σⱼ φ(Kⱼ)] | Linearized Attention | The key linearization using kernel feature maps |
| Sᵢ = Sᵢ₋₁ + φ(Kᵢ)Vᵢᵀ | Causal State Update | Running accumulation for causal linear attention |
| Zᵢ = Zᵢ₋₁ + φ(Kᵢ) | Normalizer Update | Running normalization accumulation |
| V'ᵢ = φ(Qᵢ)ᵀSᵢ / φ(Qᵢ)ᵀZᵢ | Causal Output | Output computation from accumulated state |
| φ(x) = elu(x) + 1 | Feature Map | The specific feature map choice used in experiments |

## 13.3 Parameter Meaning Table

| Parameter | Symbol | Typical Value | Role |
|---|---|---|---|
| Sequence length | N | 784 – 3,072 (experiments) | Main variable — linear attention scales O(N) vs O(N²) |
| Query/Key dimension | D | 32 per head | Projection dimension for attention |
| Value dimension | M | 32 per head | Dimension of value vectors |
| Feature map dimension | C | D (same as query/key dim) | Dimension of φ(x) — determines summary matrix size |
| Number of heads | H | 6–8 | Multi-head attention parallelism |
| Number of layers | L | 4–16 | Transformer depth |
| Feedforward dimension | — | 4 × embedding = 1024 | Inner dimension of the two-layer FFN |
| Learning rate | — | 10⁻⁴ | RAdam optimizer |

## 13.4 Algorithm Flow Summary

### Training (Parallel Mode):
```
1. Input sequence x [N × F]
2. For each layer l = 1 to L:
   a. Project to Q, K, V
   b. Apply feature map: φ(Q), φ(K)
   c. IF causal: compute cumulative sums S₁...Sₙ and Z₁...Zₙ in parallel (prefix sum)
      ELSE: compute global S = Σ φ(Kⱼ)Vⱼᵀ and Z = Σ φ(Kⱼ)
   d. Compute output: V'ᵢ = φ(Qᵢ)ᵀSᵢ / φ(Qᵢ)ᵀZᵢ (causal) or φ(Qᵢ)ᵀS / φ(Qᵢ)ᵀZ
   e. Apply residual + feedforward + layer norm
3. Output predictions for all positions simultaneously
```

### Inference (RNN Mode — for autoregressive tasks):
```
1. Initialize: s = 0 [C×M], z = 0 [C] for each layer
2. For each new token xᵢ:
   a. For each layer l:
      - Project xᵢ to qᵢ, kᵢ, vᵢ
      - Update: sₗ = sₗ + φ(kᵢ) × vᵢᵀ     [constant time]
      - Update: zₗ = zₗ + φ(kᵢ)              [constant time]
      - Output: v'ᵢ = sₗᵀ φ(qᵢ) / zₗᵀ φ(qᵢ) [constant time]
      - Apply feedforward
   b. Predict next token from final layer output
3. Feed predicted token as next input → repeat
```

---

# 14. One-Page Master Summary Card

| Aspect | Summary |
|---|---|
| **Problem** | Standard transformer self-attention has O(N²) time and memory complexity, making it prohibitively slow for long sequences and autoregressive inference |
| **Key Idea** | Replace softmax similarity with a kernel feature map φ(·), then exploit matrix multiplication associativity to compute attention in O(N) time by aggregating key-value pairs into a fixed-size summary matrix before querying |
| **Method** | (1) Apply feature map φ(x) = elu(x)+1 to queries and keys, (2) compute summary matrices as running cumulative sums for causal tasks, (3) query the summary at each position for O(1) output per step, (4) use the resulting RNN formulation for constant-time autoregressive inference |
| **Results** | Near-identical quality to softmax on image generation (0.644 vs 0.621 bits/dim on MNIST, 3.40 vs 3.47 on CIFAR-10), inference throughput 317× faster on MNIST and 4,462× faster on CIFAR-10, 3× faster training per epoch on speech recognition |
| **Core Theoretical Insight** | Every transformer with causal masking is mathematically equivalent to a recurrent neural network — the difference between transformers and RNNs is the attention kernel, not the architecture |
| **Key Weakness** | Ad-hoc feature map with no theoretical guarantees; no forgetting mechanism in the running sum; 3 percentage point quality gap on speech recognition; no NLP/language modeling experiments |
| **Research Opportunity** | Design learnable or theoretically-grounded feature maps; add gating/decay to the state update (as RetNet and GLA later did); evaluate on language modeling benchmarks; explore hybrid linear+softmax architectures |
| **Publishable Extension** | Propose gated linear attention with learnable decay, show it closes the quality gap with softmax on language modeling while preserving O(N) complexity, compare against RetNet/Mamba/Flash Attention |
