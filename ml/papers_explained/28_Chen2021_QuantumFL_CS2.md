# 28_Chen2021_QuantumFL — Complete Research Companion Sheet

---

# 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Paper Title** | Federated Quantum Machine Learning |
| **Authors** | Samuel Yen-Chi Chen, Shinjae Yoo |
| **Affiliation** | Computational Science Initiative, Brookhaven National Laboratory |
| **Year** | 2021 |
| **Problem Domain** | Quantum Machine Learning + Federated Learning |
| **Paper Type** | Algorithmic / Method + Experimental |
| **Core Contribution** | First framework for training hybrid quantum-classical ML models in a federated manner |
| **Key Idea** | Combine federated learning with variational quantum circuits (VQC) on top of classical pre-trained CNNs to enable privacy-preserving, distributed quantum ML training |
| **Required Background** | Basics of federated learning, quantum computing fundamentals (qubits, quantum gates), variational quantum circuits, transfer learning, CNNs |
| **Primary Baseline** | Non-federated (centralized) training of the same hybrid VGG16-VQC architecture |
| **Main Innovation Type** | Framework / System Design — first to apply FL to QML |
| **Datasets Used** | Cats vs Dogs (23K train / 2K test), CIFAR-10 Planes vs Cars (10K train / 2K test) |
| **Difficulty Level** | Moderate — requires understanding of both quantum computing and federated learning |
| **Reproducibility Level** | High — uses public datasets, standard libraries (PyTorch, PennyLane, Qulacs), and clearly described architecture |

---

# 1. Research Context & Core Problem

## 1.1 Exact Problem Formulation

- Quantum machine learning (QML) models are becoming increasingly promising, but current quantum computers (called NISQ devices) are limited in their number of qubits and circuit depth.
- At the same time, machine learning models need large amounts of data, much of which is sensitive (medical records, personal recordings, browsing habits).
- Training QML models on a single centralized machine has two problems:
  1. **Privacy risk**: All data must be collected in one place, exposing it to leaks or adversarial attacks.
  2. **Scalability bottleneck**: A single NISQ device cannot handle large-scale training due to hardware limitations.
- **The core question**: Can we train quantum machine learning models across multiple distributed quantum devices (or simulators) without ever sharing raw data, while still achieving performance comparable to centralized training?

## 1.2 Why the Problem Exists

- **Quantum hardware constraints**: NISQ devices have limited qubits and noisy operations — they cannot run large circuits faithfully.
- **Data privacy regulations**: Sensitive data (health, financial, personal) cannot be freely moved to cloud servers for centralized model training.
- **No prior work existed**: At the time of writing, nobody had studied federated learning in the quantum machine learning setting. Classical FL was well-explored, but its extension to quantum models was entirely unexplored.

## 1.3 Historical / Theoretical Gap

- Classical federated learning (introduced by McMahan et al., 2017) was already well-developed, with many extensions (FedProx, secure aggregation, differential privacy).
- Quantum machine learning had advanced significantly with variational algorithms, enabling QML on NISQ devices.
- **The gap**: These two fields had never been connected. No one had investigated whether federated training principles could work when the local models are quantum circuits rather than classical neural networks.

## 1.4 Limitations of Previous Approaches

| Previous Approach | Limitation |
|---|---|
| Classical FL | Only works with classical models; does not leverage quantum computing potential |
| Centralized QML | Requires all data in one place; single point of failure; cannot scale across multiple quantum devices |
| Standalone NISQ devices | Limited capacity; cannot handle large datasets alone |
| Classical transfer learning | Does not explore quantum components for classification |

## 1.5 Contribution Category

- **Framework design**: Proposes the first federated quantum machine learning architecture.
- **Empirical validation**: Demonstrates through experiments that federated QML achieves comparable accuracy to centralized QML.
- **Efficiency insight**: Shows that distributed training uses significantly fewer computing resources per round while maintaining performance.

### Why This Paper Matters

- It is the **first paper** to propose and experimentally validate federated learning for quantum machine learning models.
- It bridges two rapidly growing fields — federated learning and quantum computing — opening a completely new research direction.
- It demonstrates a practical approach using hybrid quantum-classical models that can work on current NISQ hardware.
- The privacy and scalability implications are significant for real-world applications (healthcare, finance, speech recognition) where quantum computing meets sensitive data.

### Remaining Open Problems

1. No differential privacy or secure aggregation is implemented — model parameters could still leak information.
2. Only simple averaging is used for aggregation — no robustness against Byzantine (malicious) clients.
3. Experiments are on binary classification only — multi-class and more complex tasks are not explored.
4. Only the quantum classifier layer is federated — the pre-trained CNN feature extractor is frozen, limiting the novelty of quantum training.
5. No experiments on actual quantum hardware — all results are from quantum simulators.
6. Non-IID (non-identically distributed) data across clients is not addressed.
7. Communication overhead is not quantified or optimized.
8. Scalability to larger numbers of qubits remains untested.

---

# 2. Minimum Background Concepts

## 2.1 Federated Learning (FL)

- **Plain definition**: A distributed training paradigm where multiple devices (clients) train a shared model collaboratively without exchanging their raw data. Each client trains locally on its own data, then sends only model updates (parameters) to a central server, which aggregates them into a global model.
- **Role inside paper**: FL provides the overarching training framework. The authors apply FL to quantum machine learning models instead of classical neural networks.
- **Why authors needed it**: To enable privacy-preserving, distributed training of QML models across multiple quantum devices.

## 2.2 NISQ Devices (Noisy Intermediate-Scale Quantum Computers)

- **Plain definition**: Current-generation quantum computers that have a moderate number of qubits (tens to hundreds) but lack error correction, making them prone to noise and errors in computation.
- **Role inside paper**: The motivation for the hybrid approach — since NISQ devices are limited, the authors use classical CNNs for heavy feature extraction and only use the quantum circuit for the final classification step.
- **Why authors needed it**: To justify the hybrid quantum-classical architecture and explain why pure quantum models are impractical currently.

## 2.3 Variational Quantum Circuits (VQC) / Quantum Neural Networks (QNN)

- **Plain definition**: Quantum circuits that contain adjustable parameters (like weights in a classical neural network). These parameters are optimized using classical optimization techniques. The circuit applies quantum operations to input data and produces output measurements.
- **Role inside paper**: The VQC serves as the classifier layer in the hybrid model. It replaces the final dense layers of a classical CNN.
- **Why authors needed it**: VQCs are the primary quantum component that makes the model "quantum." They are suitable for NISQ devices because they can work with few qubits and moderate circuit depth.

## 2.4 Qubits

- **Plain definition**: The quantum equivalent of classical bits. While a classical bit can be 0 or 1, a qubit can be in a superposition of both states simultaneously. Multiple qubits can be entangled, allowing complex correlated computations.
- **Role inside paper**: The VQC in this paper uses 4 qubits. The number of qubits determines the input dimension of the quantum classifier (the compressed feature vector must match the qubit count).
- **Why authors needed it**: Qubits are the fundamental units of quantum computation; understanding them is essential to understand how the VQC processes data.

## 2.5 Quantum Gates (Rotation Gates, CNOT)

- **Plain definition**: Operations applied to qubits that transform their states. Rotation gates (Ry, Rz) rotate a single qubit's state by a specified angle. CNOT gates entangle two qubits, creating quantum correlations between them.
- **Role inside paper**: Ry and Rz gates are used in the encoder to convert classical data into quantum states. CNOT gates and parameterized single-qubit rotations R(alpha, beta, gamma) form the variational (trainable) layer.
- **Why authors needed it**: These gates constitute the building blocks of the VQC architecture.

## 2.6 Transfer Learning

- **Plain definition**: A technique where a model pre-trained on a large dataset (like ImageNet) is reused for a new task. Typically, the early layers (feature extractors) are kept frozen, and only the final classification layers are retrained.
- **Role inside paper**: A pre-trained VGG16 model extracts image features and compresses them into a 4-dimensional vector, which is then fed to the VQC for classification.
- **Why authors needed it**: Current quantum devices cannot process high-dimensional image data directly. Transfer learning bridges this gap by using a classical CNN to reduce dimensionality before quantum processing.

## 2.7 VGG16

- **Plain definition**: A well-known deep convolutional neural network with 16 layers, originally trained on ImageNet for image classification. It is commonly used as a feature extractor in transfer learning setups.
- **Role inside paper**: Serves as the classical component that processes raw images (224x224 or 32x32) and outputs a 4-dimensional latent vector for the VQC.
- **Why authors needed it**: Needed a powerful, pre-trained feature extractor to compress image information into a representation small enough for a 4-qubit quantum circuit.

## 2.8 Parameter-Shift Rule

- **Plain definition**: A method to compute exact gradients of quantum circuits analytically. Instead of using finite differences (which introduce approximation error), the parameter-shift rule evaluates the circuit at two shifted parameter values to compute the exact derivative.
- **Role inside paper**: Enables end-to-end gradient-based training of the hybrid model. The classical part uses standard backpropagation; the quantum part uses the parameter-shift rule.
- **Why authors needed it**: Standard backpropagation cannot directly compute gradients through quantum operations. The parameter-shift rule provides the analytical gradient needed for optimization.

## 2.9 PauliZ Measurement

- **Plain definition**: A quantum measurement along the Z-axis that projects a qubit into state |0> or |1>. The expectation value of this measurement gives a number between -1 and +1, which can be used as a classification score.
- **Role inside paper**: The VQC outputs PauliZ expectation values from the first two qubits, which serve as logits for binary classification (e.g., cat vs. dog).
- **Why authors needed it**: To convert quantum state information into classical numbers that can be used for classification decisions.

---

# 3. Mathematical / Theoretical Understanding Layer

## 3.1 Quantum State Representation

### Intuition
A quantum system with N qubits can exist in a superposition of all possible combinations of 0s and 1s. Each combination has an associated complex number (amplitude) that encodes the probability of observing that state when measured.

### Formal Expression
An N-qubit quantum state is:

|psi> = SUM over all (q1, q2, ..., qN) of c_{q1,...,qN} |q1> tensor |q2> tensor ... tensor |qN>

### Variable Meaning Table

| Symbol | Meaning |
|---|---|
| \|psi> | The full quantum state of the N-qubit system |
| N | Number of qubits (4 in this paper) |
| q_i | Each qubit's state, either 0 or 1 |
| c_{q1,...,qN} | Complex amplitude for each basis state — encodes both probability and phase information |
| \|q_i> | The state of the i-th qubit |
| tensor product | Combines individual qubit states into a joint multi-qubit state |

### Normalization Constraint
SUM over all (q1,...,qN) of |c_{q1,...,qN}|^2 = 1

This means the total probability of measuring any outcome must equal 1.

### Practical Interpretation
- For 4 qubits, there are 2^4 = 16 possible basis states.
- The quantum state is a weighted combination of all 16 states.
- The VQC manipulates these amplitudes through quantum gates to perform computation.

## 3.2 Quantum Encoding (Data Embedding)

### Intuition
To process classical data (like image features) on a quantum circuit, we need to convert classical numbers into quantum states. The encoding uses the input values as rotation angles for quantum gates.

### Encoding Scheme
For each input value x_i:
- Apply rotation gate Ry(arctan(x_i)) — rotates qubit around y-axis
- Apply rotation gate Rz(arctan(x_i^2)) — rotates qubit around z-axis

### Why arctan?
- arctan maps any real number to the range (-pi/2, pi/2), ensuring the rotation angles are bounded.
- Using both x_i and x_i^2 captures both linear and quadratic features of the input.

### Assumptions
- The 4-dimensional latent vector from VGG16 is sufficient to encode relevant image features.
- The arctan transformation preserves enough information for classification.

### Limitation
- This encoding is relatively simple; more sophisticated encoding methods might capture richer quantum representations.

## 3.3 Variational Layer Structure

### Intuition
After encoding data, the variational layer applies trainable quantum operations that learn to transform the encoded state into a useful classification output. It is analogous to trainable weight layers in classical neural networks.

### Structure (per layer)
1. **CNOT gates** between neighboring qubits — creates entanglement (quantum correlations).
2. **General rotation gates** R(alpha_i, beta_i, gamma_i) on each qubit — three trainable parameters per qubit.

### Parameter Count
- 4 qubits x 3 parameters x 2 repetitions = **24 trainable quantum parameters**.
- This is extremely compact compared to classical neural networks.

## 3.4 Federated Aggregation

### Intuition
After each client trains its local quantum circuit, the central server combines all the client models into one global model by simply averaging the parameters.

### Aggregation Rule
Theta_global = (1/K) * SUM over k=1 to K of theta_k

where:
- Theta_global = aggregated global model parameters
- theta_k = trained parameters from client k
- K = number of selected clients in each round (5 in this paper)

### Assumptions
- All clients have equal weight (equal amount of data).
- Simple averaging is sufficient; no weighting by dataset size or data quality.
- No client is malicious or Byzantine.

### Limitation
- Simple averaging is vulnerable to corrupted or adversarial model updates.
- Does not account for data heterogeneity across clients.

## 3.5 Parameter-Shift Rule for Quantum Gradients

### Intuition
To train the quantum circuit, we need to compute how the output changes when we slightly adjust each parameter. The parameter-shift rule computes this exactly by evaluating the circuit twice: once with the parameter shifted up by pi/4 and once shifted down by pi/4.

### Formula
d/d(theta) f(theta) = [f(theta + s) - f(theta - s)] / (2 sin(s))

where s = pi/2 for standard rotation gates.

### Practical Interpretation
- No need for finite difference approximations — the gradient is exact.
- Requires 2 circuit evaluations per parameter per gradient computation.
- This integrates seamlessly with classical backpropagation through the rest of the hybrid model.

### Mathematical Insight Box
> **Key insight for researchers**: The parameter-shift rule enables end-to-end differentiable hybrid quantum-classical models. The quantum circuit can be treated as a differentiable layer, just like any classical neural network layer, making it compatible with standard deep learning training pipelines (PyTorch, TensorFlow). This is what makes hybrid quantum-classical transfer learning practical.

---

# 4. Proposed Method / Framework (MOST IMPORTANT)

## 4.1 Overall Pipeline

The Federated Quantum Machine Learning (FedQML) framework operates as follows:

```
Step 1: Central server initializes global hybrid model (VGG16 + VQC)
Step 2: Central server distributes global model to all clients
Step 3: In each training round:
    a. Central server randomly selects K clients (K=5)
    b. Selected clients receive the current global model
    c. Each client trains locally on its own data for E epochs
    d. Clients upload trained circuit parameters to central server
    e. Central server averages parameters to form new global model
    f. Global model is evaluated on central test set
Step 4: Repeat Step 3 for R rounds (R=100)
```

## 4.2 Components / Modules

### Component 1: Classical Feature Extractor (VGG16)

- **What it does**: Takes raw images and extracts meaningful features, compressing them into a small latent vector.
- **Architecture modifications**: The original VGG16 classifier is replaced with:
  - Linear(25088 -> 4096) + ReLU + Dropout(0.5)
  - Linear(4096 -> 4096) + ReLU + Dropout(0.5)
  - Linear(4096 -> 4) — outputs a 4-dimensional vector
- **Training status**: Pre-trained on ImageNet, frozen during federated training (weights are NOT updated).
- **Why authors did this**: NISQ devices cannot handle high-dimensional inputs. VGG16 reduces 224x224x3 images to just 4 numbers.
- **Weakness**: The frozen pre-trained model may not extract optimal features for the specific task. Fine-tuning it could improve performance.
- **Research idea seed**: Investigate federated fine-tuning of both the classical and quantum components jointly. Study whether different classical backbones (ResNet, EfficientNet) improve QML performance.

### Component 2: Quantum Classifier (VQC)

- **What it does**: Takes the 4-dimensional latent vector from VGG16 and performs binary classification using quantum operations.
- **Architecture**:
  - **Encoder**: Ry(arctan(x_i)) and Rz(arctan(x_i^2)) for each of 4 qubits
  - **Variational layer**: CNOT entanglement + R(alpha, beta, gamma) rotations (repeated 2 times)
  - **Measurement**: PauliZ expectation on first 2 qubits (for binary classification)
- **Parameters**: 24 trainable quantum circuit parameters total.
- **Why authors did this**: VQC provides a quantum-enhanced classifier that can potentially offer advantages over classical classifiers with the same parameter count.
- **Weakness**: Only 4 qubits and 24 parameters — very limited model capacity. The circuit depth (2 repetitions) is shallow.
- **Research idea seed**: Explore deeper circuits, different entanglement patterns, or data re-uploading schemes. Test whether quantum advantage exists for this architecture vs. a classical layer with 24 parameters.

### Component 3: Federated Aggregation Server

- **What it does**: Receives trained quantum circuit parameters from selected clients and computes their average to form the new global model.
- **Aggregation method**: Simple parameter averaging (FedAvg-style).
- **Client selection**: Random selection of 5 out of 100 clients each round.
- **Why authors did this**: Simplest aggregation approach; serves as a proof-of-concept baseline.
- **Weakness**: No protection against malicious clients, no weighting by data quality or quantity, no momentum or adaptive aggregation.
- **Research idea seed**: Implement FedProx, FedAdam, or Byzantine-robust aggregation methods for quantum FL. Study how quantum parameter space affects aggregation dynamics.

### Component 4: Local Client Training

- **What it does**: Each selected client trains the VQC on its local data partition for E epochs using standard gradient-based optimization.
- **Training details**: Batch size = 32, local epochs E = {1, 2, 4} tested.
- **Gradient computation**: Classical backpropagation + parameter-shift rule for quantum gradients.
- **Why authors did this**: Local training is the fundamental operation in FL — clients improve the model on their own data before sharing updates.
- **Weakness**: No local regularization to prevent client drift. No gradient clipping or differential privacy noise addition.
- **Research idea seed**: Add local differential privacy to quantum gradient updates. Study the effect of client drift in quantum parameter space.

## 4.3 Data Flow

```
Raw Image (224x224 or 32x32)
    |
    v
VGG16 Feature Extractor (frozen, pre-trained)
    |
    v
Modified Classifier Layers (25088 -> 4096 -> 4096 -> 4)
    |
    v
4-dimensional Latent Vector
    |
    v
Quantum Encoder: Ry(arctan(x_i)), Rz(arctan(x_i^2)) on 4 qubits
    |
    v
Variational Layer 1: CNOT entanglement + R(alpha, beta, gamma) rotations
    |
    v
Variational Layer 2: CNOT entanglement + R(alpha, beta, gamma) rotations
    |
    v
PauliZ Measurement on Qubits 0 and 1
    |
    v
2 logit values -> Binary Classification (e.g., Cat vs Dog)
```

## 4.4 Simplified Pseudocode

```
FEDERATED QUANTUM ML TRAINING:

Initialize:
  - Global model M = VGG16(frozen) + VQC(random_params)
  - Distribute training data equally across N=100 clients
  - Place test data on central server

For round r = 1 to 100:
    1. Randomly select K=5 clients from N=100
    2. Send current global VQC parameters Theta to selected clients
    
    For each selected client k in parallel:
        3. Load global parameters Theta into local VQC
        4. For epoch e = 1 to E:
            For each batch in local_data[k]:
                a. Forward pass: image -> VGG16 -> latent_vector -> VQC -> prediction
                b. Compute loss (cross-entropy)
                c. Backward pass: 
                   - Classical backprop for VGG16 classifier layers
                   - Parameter-shift rule for VQC gradients
                d. Update local VQC parameters (only quantum params updated)
        5. Upload trained VQC parameters theta_k to central server
    
    6. Aggregate: Theta_new = average(theta_1, theta_2, ..., theta_K)
    7. Evaluate Theta_new on central test set
    8. Broadcast Theta_new to all clients

Return final global model
```

## 4.5 Key Design Choices and Rationale

| Design Choice | Rationale | Alternative Considered |
|---|---|---|
| Hybrid quantum-classical model | Pure quantum models need too many qubits for image data | Pure VQC (impractical on NISQ) |
| VGG16 as feature extractor | Well-established, pre-trained, produces reliable features | ResNet, other architectures |
| 4 qubits | Matches the 4-dimensional compressed latent vector | More qubits (would need different compression ratio) |
| 2 variational layer repetitions | Balances expressiveness with circuit depth constraints | Deeper circuits (more noise on real hardware) |
| Simple averaging aggregation | Proof-of-concept; simplest FL aggregation | FedAvg weighted, FedProx, etc. |
| 100 clients, 5 selected per round | Standard FL setting; manageable communication | Different client fractions |
| Binary classification only | Simplifies quantum measurement to 2 qubits | Multi-class (needs more measurement qubits) |

---

# 5. Experimental Setup / Evaluation Design

## 5.1 Dataset Characteristics

| Property | Cats vs Dogs | CIFAR-10 (Planes vs Cars) |
|---|---|---|
| **Training samples** | 23,000 | 10,000 |
| **Testing samples** | 2,000 | 2,000 |
| **Image dimensions** | Resized to 224x224 | 32x32 |
| **Task** | Binary classification | Binary classification |
| **Data distribution** | Equally split across 100 clients (230/client) | Equally split across 100 clients (100/client) |
| **Class balance** | Not explicitly stated | Not explicitly stated |

## 5.2 Experimental Protocol

- **Training rounds**: 100 rounds for both datasets.
- **Clients per round**: 5 randomly selected out of 100.
- **Local epochs tested**: E = {1, 2, 4}.
- **Batch size**: S = 32.
- **Aggregation**: Simple parameter averaging.
- **Evaluation**: Testing accuracy and testing loss computed on the central server after each aggregation round using the global model.
- **Baseline**: Non-federated (centralized) training with the same hybrid VGG16-VQC architecture on the full dataset.

## 5.3 Metrics Used and Why

| Metric | Why Used |
|---|---|
| **Testing Accuracy** | Primary measure of model correctness on unseen data; directly shows if federated training preserves performance |
| **Testing Loss** | Tracks convergence quality; reveals whether the model is still improving or has plateaued |
| **Training Loss** | Monitors local client training progress; averaged across selected clients per round |

## 5.4 Baseline Selection Logic

- Only one baseline: **non-federated training** with the exact same model architecture.
- This is the most direct comparison — same model, same data, only the training paradigm differs (centralized vs. federated).
- No comparison against classical (non-quantum) federated learning or other quantum architectures.

## 5.5 Hyperparameter Reasoning

| Hyperparameter | Value | Reasoning |
|---|---|---|
| Number of clients (N) | 100 | Standard FL testbed size |
| Clients per round (K) | 5 | 5% participation rate; standard in FL literature |
| Local epochs (E) | 1, 2, 4 | Tests the effect of local training intensity |
| Batch size (S) | 32 | Common batch size for image classification |
| Training rounds (R) | 100 | Sufficient for convergence observation |
| Qubits | 4 | Matches latent vector dimension |
| VQC layers | 2 | Balances expressiveness and circuit depth |
| Dropout | 0.5 | Standard regularization for VGG16 classifier |

## 5.6 Hardware / Compute Assumptions

- All experiments were run on **quantum simulators** (Qulacs), not actual quantum hardware.
- Classical components used PyTorch.
- Quantum components used PennyLane (quantum ML library).
- No mention of specific GPU/CPU hardware or training wall-clock times.

### Experimental Reliability Analysis

**What is Trustworthy:**
- The comparison between federated and non-federated training is fair (same model, same data).
- Testing accuracy results are reported consistently across both datasets.
- Multiple local epoch settings (1, 2, 4) provide some sensitivity analysis.
- The datasets are well-known public benchmarks.

**What is Questionable:**
- **No error bars or confidence intervals** — results from a single run per configuration; no statistical significance testing.
- **No real quantum hardware experiments** — simulator results may not reflect actual quantum device performance (noise, decoherence).
- **No non-IID data experiments** — data is uniformly distributed, which is unrealistic for real FL scenarios.
- **Only binary classification** — unclear how results extend to multi-class problems.
- **No comparison with classical FL** — we cannot tell if the quantum component adds any value over a classical classifier.
- **No communication cost analysis** — critical for practical FL but completely absent.
- **VGG16 is frozen** — most of the model's power comes from the pre-trained classical network, not the quantum circuit.
- **Very small quantum model** — 24 parameters make it hard to claim meaningful "quantum" contribution.

---

# 6. Results & Findings Interpretation

## 6.1 Main Outcomes

### Cats vs Dogs Results

| Configuration | Training Loss | Testing Loss | Testing Accuracy |
|---|---|---|---|
| Federated (1 local epoch) | 0.3506 | 0.3519 | 98.7% |
| Federated (2 local epochs) | 0.3405 | 0.3408 | 98.6% |
| Federated (4 local epochs) | 0.3304 | 0.3413 | 98.6% |
| Non-Federated | 0.3360 | 0.3369 | 98.75% |

### CIFAR-10 Planes vs Cars Results

| Configuration | Training Loss | Testing Loss | Testing Accuracy |
|---|---|---|---|
| Federated (1 local epoch) | 0.4029 | 0.4133 | 93.4% |
| Federated (2 local epochs) | 0.4760 | 0.4056 | 94.05% |
| Federated (4 local epochs) | 0.4090 | 0.3934 | 93.45% |
| Non-Federated | 0.4190 | 0.4016 | 93.65% |

## 6.2 Performance Trends

1. **Federated training matches centralized performance**: The accuracy gap between federated and non-federated training is negligible (within 0.15% for Cats vs Dogs, within 0.25% for CIFAR-10).
2. **Single local epoch suffices**: Even with E=1, the model achieves nearly the same accuracy as E=2 or E=4. This is a significant practical finding — fewer local epochs mean less computation per client.
3. **Training loss fluctuates in federated setting**: Because different clients are randomly selected each round, the training loss shows more variability than centralized training. However, testing loss converges smoothly.
4. **Convergence within 100 rounds**: Both datasets show convergence within the 100 training rounds across all federated configurations.

## 6.3 Efficiency Observations

- **Cats vs Dogs**: Per round, federated training uses data from 230 x 5 x 1 = 1,150 samples (with E=1), while centralized training processes all 23,000 samples per epoch. This is a **20x reduction** in per-round compute.
- **CIFAR-10**: Per round, federated training uses 100 x 5 x 1 = 500 samples, vs. 10,000 for centralized. This is also a **20x reduction**.

## 6.4 Failure Cases / Limitations in Results

- No failure cases are reported.
- The paper does not explore scenarios where federated training might fail (e.g., highly non-IID data, malicious clients, noisy quantum channels).
- The training loss fluctuation is noted but not analyzed deeply.

## 6.5 Unexpected Observations

- The fact that 1 local epoch performs essentially as well as 4 local epochs was notable. This suggests that quantum circuit parameters converge quickly even with minimal local training, possibly because the parameter space is very small (24 parameters).

## 6.6 Statistical Meaning

- The accuracy numbers are high (93-98%), but this is largely due to the powerful pre-trained VGG16 backbone.
- Without error bars or multiple runs, we cannot determine if the small differences between configurations are statistically significant.
- The small parameter count (24) means the quantum circuit has limited capacity to overfit, which might explain the consistent convergence.

### Publishability Strength Check

**Publication-grade results:**
- The demonstration that federated QML works at all — matching centralized performance — is the key publishable finding.
- The efficiency observation (20x fewer samples per round) is interesting.
- The proof-of-concept nature of the work is appropriate for a first-in-kind contribution.

**Results needing stronger validation:**
- Statistical significance testing (multiple runs with different random seeds).
- Real quantum hardware experiments.
- Non-IID data scenarios.
- Comparison with classical federated baselines.
- Communication cost measurement.
- Multi-class classification experiments.

---

# 7. Strengths -- Weaknesses -- Assumptions

## Table 1: Technical Strengths

| # | Strength | Why It Matters |
|---|---|---|
| 1 | **First-of-its-kind**: First paper to combine FL with QML | Opens an entirely new research direction |
| 2 | **Clean framework design**: Well-structured hybrid pipeline with clear component separation | Easy to reproduce, extend, and build upon |
| 3 | **Minimal accuracy degradation**: Federated training preserves 98%+ accuracy | Validates the feasibility of distributed quantum training |
| 4 | **Computational efficiency**: 20x reduction in per-round data usage | Practical benefit for distributed NISQ training |
| 5 | **Standard tools**: Uses PyTorch, PennyLane, Qulacs — accessible and reproducible | Lowers barrier to entry for follow-up research |
| 6 | **Multiple experiment configurations**: Tests 3 local epoch settings across 2 datasets | Shows robustness of the approach |
| 7 | **Practical hybrid architecture**: Combines proven CNN backbone with quantum classifier | Feasible on current NISQ devices |

## Table 2: Explicit Weaknesses

| # | Weakness | Impact |
|---|---|---|
| 1 | **No privacy guarantees**: No differential privacy, secure aggregation, or formal privacy analysis | The privacy claim is unsupported — model parameters could leak data |
| 2 | **Simulator only**: No real quantum hardware experiments | Results may not transfer to actual noisy quantum devices |
| 3 | **Trivial aggregation**: Simple averaging with no robustness | Vulnerable to Byzantine clients and adversarial attacks |
| 4 | **Binary classification only**: Only 2-class problems tested | Unclear scalability to multi-class or regression tasks |
| 5 | **No statistical rigor**: No error bars, confidence intervals, or multiple runs | Cannot assess significance of reported differences |
| 6 | **IID data only**: Data equally distributed across clients | Unrealistic for real-world FL where data is heterogeneous |
| 7 | **Frozen classical backbone**: VGG16 is not trained; only 24 quantum parameters are updated | Most of the model's power comes from the classical component |
| 8 | **No classical FL comparison**: Does not compare against classical FL with same architecture | Cannot demonstrate quantum advantage |
| 9 | **No communication analysis**: Communication overhead is not measured | Critical metric for FL practicality is missing |

## Table 3: Hidden Assumptions

| # | Hidden Assumption | Risk |
|---|---|---|
| 1 | All clients have the same quantum hardware/simulator capability | Real quantum devices have different noise profiles and qubit counts |
| 2 | Data is perfectly IID across clients | Real-world data is almost always non-IID |
| 3 | Communication channels are reliable and noise-free | Quantum parameter transmission could be subject to noise |
| 4 | Simple averaging is sufficient for quantum parameter spaces | Quantum parameters might have different aggregation dynamics than classical weights |
| 5 | 4-dimensional latent vector captures sufficient information | Information bottleneck may lose critical features |
| 6 | Pre-trained VGG16 features generalize to all classification tasks | Domain shift could degrade performance on specialized datasets |
| 7 | 100 training rounds is sufficient for convergence | More complex tasks may need many more rounds |
| 8 | No client dropout or failure during training | Real distributed systems have unreliable nodes |

---

# 8. Weakness to Research Direction Mapping

| Identified Weakness | Why it Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| No differential privacy | Paper focuses on proof-of-concept; DP adds complexity | **Differentially private federated QML** | Add Gaussian noise to quantum gradients before aggregation; study privacy-accuracy tradeoff for quantum models |
| Simulator only | Real quantum hardware access is expensive and limited | **Federated QML on real NISQ devices** | Run experiments on IBM Quantum, IonQ, or other cloud quantum services; study impact of real device noise |
| Simple averaging aggregation | Keeps framework simple for first paper | **Robust aggregation for quantum FL** | Implement Byzantine-robust methods (Krum, trimmed mean) for quantum parameters; study quantum parameter outlier detection |
| Binary classification only | Fewer measurement qubits needed | **Multi-class federated QML** | Extend to multi-qubit measurement for multi-class; study scaling behavior |
| IID data only | Simplifies experimental setup | **Non-IID federated QML** | Study convergence under label skew, feature skew; implement FedProx or SCAFFOLD for quantum models |
| Frozen classical backbone | Avoids complexity of training large classical model | **Joint quantum-classical federated fine-tuning** | Unfreeze some VGG layers; study whether federated training of both components improves results |
| No quantum advantage comparison | Quantum advantage is hard to establish | **Quantum vs. classical FL comparison** | Replace VQC with classical network of same parameter count; compare accuracy, convergence speed, communication cost |
| No communication analysis | Not the paper's focus | **Communication-efficient quantum FL** | Study quantum parameter compression, gradient sparsification for quantum circuits; quantify bits transmitted per round |
| Very small quantum model (24 params) | NISQ hardware limits | **Scaling federated QML** | Test with more qubits, deeper circuits, larger feature vectors; study when quantum models outperform classical ones of same size |
| No adversarial robustness | Not addressed | **Byzantine-robust federated QML** | Inject malicious quantum parameter updates; test defense mechanisms |

---

# 9. Novel Contribution Extraction

## 9.1 The Paper's Novel Claim

"We propose the first federated training framework for hybrid quantum-classical machine learning models that achieves comparable accuracy to centralized training while distributing computational load across multiple quantum devices."

## 9.2 Possible Novel Claim Templates for Future Research

1. **"We propose a differentially private federated quantum machine learning framework that provides formal (epsilon, delta)-privacy guarantees while maintaining classification accuracy within X% of non-private federated QML."**

2. **"We propose a Byzantine-robust aggregation method for federated quantum neural networks that maintains convergence even when up to Y% of participating quantum clients submit adversarial model updates."**

3. **"We propose a communication-efficient federated QML protocol that reduces quantum parameter transmission by X% through gradient compression while preserving model convergence."**

4. **"We propose a non-IID-robust federated quantum learning framework using [regularization method] that achieves convergence under heterogeneous data distributions across quantum clients, improving accuracy by X% over vanilla FedAvg in quantum settings."**

5. **"We demonstrate for the first time that federated quantum machine learning on real NISQ hardware achieves comparable performance to simulator-based federated QML, identifying key noise-induced challenges and proposing noise-aware aggregation strategies."**

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Work

1. **Integration with differential privacy (DP)** for stronger privacy guarantees.
2. **Secure multi-party computation** to further enhance security in decentralized quantum computing.
3. **Advanced aggregation methods** that are robust to corrupt models or noisy communication.
4. **Decentralized FL** (removing the central server) using blockchain or peer-to-peer protocols for quantum FL.
5. **Gossip learning** in the quantum regime as an alternative to server-based FL.
6. **Hybrid tensor network + quantum circuit architectures** in federated settings.
7. **Federated quantum convolutional neural networks (QCNN)** on larger quantum devices.
8. **Applications**: quantum-enhanced speech recognition, healthcare (dementia prediction), and financial applications.

## 10.2 Missing Directions (Not Mentioned by Authors)

1. **Non-IID data handling**: No discussion of data heterogeneity across quantum clients.
2. **Personalized federated QML**: Adapting the global model to individual client needs.
3. **Asynchronous federated QML**: Clients with different quantum hardware may have different training speeds.
4. **Quantum-specific gradient compression**: Exploiting the structure of quantum parameters for communication efficiency.
5. **Fairness in federated QML**: Ensuring the global model works equally well for all clients.
6. **Federated quantum generative models**: Extending beyond classification to generation tasks.
7. **Client selection strategies**: Using intelligent selection instead of random sampling to improve convergence.

## 10.3 Modern Extensions (Post-2021 Relevance)

1. **Quantum error mitigation in FL**: Modern error mitigation techniques could be applied to federated QML on real hardware.
2. **Quantum advantage benchmarking**: More rigorous testing of when quantum components actually help in FL.
3. **Larger quantum devices**: As quantum hardware grows (100+ qubits), federated training of more complex quantum models becomes feasible.
4. **Quantum federated transfer learning with foundation models**: Replace VGG16 with modern vision transformers or foundation models.
5. **Federated quantum reinforcement learning**: Extend the framework to distributed quantum RL.

## 10.4 Cross-Domain Combinations

1. **Federated QML + Homomorphic Encryption**: Train on encrypted quantum data.
2. **Federated QML + Continual Learning**: Clients learn incrementally without forgetting.
3. **Federated QML + Multi-task Learning**: Different clients work on related but different tasks.
4. **Federated QML + Network Architecture Search**: Automatically design optimal quantum circuits for federated settings.
5. **Federated QML + Edge Computing**: Deploy on quantum edge devices with limited connectivity.

---

# 11. How to Write a NEW Paper From This Work

## 11.1 Reusable Elements

- **The hybrid VGG16-VQC pipeline**: Proven and reproducible baseline architecture.
- **The federated training protocol**: Well-defined client-server interaction pattern.
- **Evaluation methodology**: Testing accuracy and loss across multiple local epoch settings.
- **Dataset choices**: Cats vs Dogs and CIFAR-10 are standard, easy-to-reproduce benchmarks.
- **The argument structure**: "We show federated [X] achieves comparable performance to centralized [X]."
- **Software stack**: PyTorch + PennyLane + Qulacs is a well-supported combination.

## 11.2 What MUST NOT Be Copied

- Exact figure reproductions or table formatting.
- Verbatim sentences or paragraph structures.
- The specific experimental numbers without re-running experiments.
- The exact modified VGG16 classifier layer configuration (design your own or justify reuse).

## 11.3 How to Design a Novel Extension

1. **Pick one weakness** from Section 8 as your primary research question.
2. **Formulate a clear hypothesis**: "Adding [technique X] to federated QML will improve [metric Y] by addressing [weakness Z]."
3. **Design controlled experiments**: Keep the base architecture similar for fair comparison, change only the variable under study.
4. **Add proper baselines**: Include both classical FL and centralized QML as baselines (which this paper lacks).
5. **Add statistical rigor**: Run multiple seeds, report confidence intervals.
6. **Test on real hardware if possible**: Even a small-scale real quantum experiment adds significant novelty.
7. **Address non-IID data**: This is the most impactful unaddressed challenge.

## 11.4 Minimum Publishable Contribution Checklist

- [ ] Clear problem statement that goes beyond this paper.
- [ ] At least one novel technical component (new aggregation, new architecture, new privacy mechanism, real hardware results).
- [ ] Comparison against this paper's approach as a baseline.
- [ ] Comparison against classical (non-quantum) federated learning.
- [ ] Statistical significance testing (multiple runs with error bars).
- [ ] At least one experiment addressing a realistic challenge (non-IID data, noisy hardware, adversarial clients).
- [ ] Communication cost analysis.
- [ ] Clear discussion of when quantum components help vs. when they do not.
- [ ] Reproducible code and experimental setup.

---

# 12. Complete Paper Writing Template

## Abstract
- **Purpose**: Summarize the entire paper in 150-250 words.
- **What to include**: Problem statement, proposed approach, key innovation, main results (with numbers), and significance.
- **Common mistakes**: Too vague, no concrete results, too much background, too long.
- **Reviewer expectations**: Should clearly state what is new and why it matters.
- **Template**: "Training quantum machine learning models in federated settings faces challenges including [challenge]. Existing approaches [limitation]. We propose [method], which [key innovation]. Our framework enables [benefit]. Experiments on [datasets] demonstrate [specific results]. Compared to [baselines], our approach achieves [improvement]. This work opens [future direction]."

## 1. Introduction
- **Purpose**: Motivate the problem, establish context, state contributions.
- **What to include**:
  - Opening hook: Growing importance of QML + FL.
  - Problem statement: Why federated QML is needed.
  - Gap in existing work: What has not been done.
  - Your contributions (explicit numbered list).
  - Paper organization.
- **Common mistakes**: Too much general background, contributions buried in text, not clearly distinguishing your work from prior art.
- **Reviewer expectations**: By the end of the introduction, reviewers should know exactly what is new and why they should care.

## 2. Related Work
- **Purpose**: Position your work within the literature.
- **What to include**:
  - Federated learning (classical) — key developments and limitations.
  - Quantum machine learning — variational approaches, NISQ challenges.
  - Privacy in ML — differential privacy, secure computation.
  - Hybrid quantum-classical models — transfer learning approaches.
  - Clearly state what each related work does NOT address that your paper does.
- **Common mistakes**: Listing papers without critical analysis, missing key references, not explaining how your work differs.
- **Reviewer expectations**: Comprehensive but focused coverage; every cited work should relate to your contribution.

## 3. Preliminaries / Background
- **Purpose**: Provide technical foundation needed to understand your method.
- **What to include**:
  - Federated learning formulation (FedAvg algorithm).
  - Quantum computing basics (qubits, gates, measurement).
  - Variational quantum circuits (encoding, variational layers, measurement).
  - Transfer learning setup.
- **Common mistakes**: Too much textbook material, not enough focus on what is specific to your paper.
- **Reviewer expectations**: Just enough to make the paper self-contained; no more.

## 4. Proposed Method
- **Purpose**: Describe your contribution in full technical detail.
- **What to include**:
  - System architecture diagram.
  - Detailed algorithm description (pseudocode).
  - Mathematical formulation of key components.
  - Design choices and justifications.
  - Complexity analysis (communication cost, computation cost).
- **Common mistakes**: Unclear descriptions, missing important details, no justification for design choices.
- **Reviewer expectations**: Reproducible from this section alone.

## 5. Theoretical Analysis (if applicable)
- **Purpose**: Provide convergence guarantees, privacy bounds, or other theoretical results.
- **What to include**:
  - Theorem statements with clear assumptions.
  - Proof sketches (full proofs in appendix).
  - Interpretation of theoretical results.
- **Common mistakes**: Assumptions too strong to be practical, proofs without intuition.
- **Reviewer expectations**: Honest about limitations of theoretical results.

## 6. Experiments
- **Purpose**: Empirically validate your approach.
- **What to include**:
  - Datasets and preprocessing.
  - Baselines and comparison methods.
  - Implementation details and hyperparameters.
  - Results tables with error bars.
  - Ablation studies.
  - Sensitivity analysis.
- **Common mistakes**: Unfair comparisons, cherry-picked results, no ablation, no error analysis.
- **Reviewer expectations**: Reproducible, fair, statistically rigorous, with ablations.

## 7. Discussion
- **Purpose**: Interpret results, discuss implications, and acknowledge limitations.
- **What to include**:
  - What the results mean in practice.
  - When the method works well and when it does not.
  - Comparison of computational costs.
  - Potential applications.
- **Common mistakes**: Overclaiming, ignoring negative results, no discussion of failure modes.
- **Reviewer expectations**: Honest, balanced, and insightful.

## 8. Limitations
- **Purpose**: Explicitly state what the paper does NOT address.
- **What to include**:
  - Assumptions that may not hold.
  - Scenarios where the method may fail.
  - Missing experiments or comparisons.
- **Common mistakes**: Downplaying limitations, being too brief.
- **Reviewer expectations**: Transparency increases trust; explicitly discuss limitations.

## 9. Conclusion
- **Purpose**: Summarize contributions and impact.
- **What to include**:
  - Recap of key contributions (1-2 sentences each).
  - Main takeaway message.
  - Future directions (2-3 concrete directions).
- **Common mistakes**: Introducing new information, being too repetitive of the abstract.
- **Reviewer expectations**: Clear, concise, forward-looking.

## References
- Use consistent citation format (usually conference-specific).
- Include all works discussed in the paper.
- Cite recent and seminal works appropriately.

---

# 13. Publication Strategy Guide

## 13.1 Suitable Venues

| Venue Type | Examples | Fit |
|---|---|---|
| **Quantum Computing Conferences** | IEEE Quantum Week, QIP, APS March Meeting | High — quantum ML focus |
| **Machine Learning Conferences** | NeurIPS, ICML, ICLR (workshops) | Medium — need strong quantum advantage argument |
| **Privacy/Security Conferences** | IEEE S&P, CCS (workshops) | Medium — need formal privacy analysis |
| **Journals** | Physical Review Research, Quantum, IEEE Trans. Quantum Engineering, npj Quantum Information | High — journals welcome thorough studies |
| **Applied ML/AI** | AAAI, IJCAI | Medium — need strong application motivation |
| **Federated Learning Workshops** | FL@NeurIPS, FL@ICML | High — specialized audience |

## 13.2 Required Baseline Expectations

- Classical federated learning with same architecture (replace VQC with small classical NN).
- Non-federated centralized quantum training.
- Non-federated centralized classical training.
- At least one advanced FL baseline (FedProx, FedAdam, SCAFFOLD).

## 13.3 Experimental Rigor Level

- **Minimum**: Multiple random seeds (at least 3-5), error bars, statistical tests.
- **Ideal**: Ablation studies, sensitivity analysis, scalability experiments, real hardware validation.
- **Expected datasets**: At least 2-3 datasets with varying complexity.

## 13.4 Common Rejection Reasons

1. "The quantum component is too small (24 parameters) to demonstrate quantum advantage."
2. "No comparison with classical models of equivalent size."
3. "No real quantum hardware experiments."
4. "IID data assumption is unrealistic."
5. "No formal privacy analysis despite privacy claims."
6. "Insufficient statistical rigor — no error bars."
7. "The high accuracy is primarily from VGG16, not the quantum circuit."

## 13.5 Increment Needed for Acceptance

To publish a paper extending this work, you need AT LEAST TWO of:
1. Real quantum hardware experiments.
2. Formal privacy guarantees (differential privacy bounds).
3. Non-IID data handling.
4. Rigorous quantum vs. classical comparison.
5. Advanced aggregation methods with robustness analysis.
6. Significant scaling (more qubits, more complex tasks).
7. Novel application domain with domain-specific insights.

---

# 14. Researcher Quick Reference Tables

## 14.1 Key Terminology Table

| Term | Definition | Context in Paper |
|---|---|---|
| NISQ | Noisy Intermediate-Scale Quantum — current quantum devices without error correction | Motivates hybrid approach |
| VQC | Variational Quantum Circuit — quantum circuit with trainable parameters | Core quantum component |
| QNN | Quantum Neural Network — synonymous with VQC in this paper | Alternative name for VQC |
| Qubit | Quantum bit — basic unit of quantum information | 4 qubits used in VQC |
| Entanglement | Quantum correlation between qubits created by CNOT gates | Enables multi-qubit computation in VQC |
| FL | Federated Learning — distributed training without sharing raw data | Training paradigm |
| FedAvg | Federated Averaging — averaging model parameters from clients | Aggregation method used |
| Transfer Learning | Using pre-trained model features for new task | VGG16 features fed to VQC |
| Parameter-Shift Rule | Exact gradient computation for quantum circuits | Enables end-to-end training |
| PauliZ Measurement | Quantum measurement along Z-axis producing expectation value in [-1, 1] | Output of VQC for classification |
| VGG16 | 16-layer deep CNN pre-trained on ImageNet | Classical feature extractor |

## 14.2 Important Equations Summary

| Equation | Purpose | Key Insight |
|---|---|---|
| \|psi> = SUM c_{q1...qN} \|q1...qN> | N-qubit quantum state representation | Quantum state is superposition of all basis states |
| SUM \|c\|^2 = 1 | Normalization constraint | Total measurement probability must be 1 |
| Ry(arctan(x_i)), Rz(arctan(x_i^2)) | Quantum encoding | Maps classical features to quantum rotations |
| Theta = (1/K) SUM theta_k | Federated aggregation | Simple averaging of client parameters |
| df/d(theta) = [f(theta+s) - f(theta-s)] / 2sin(s) | Parameter-shift gradient | Exact quantum gradient without finite differences |

## 14.3 Parameter Meaning Table

| Parameter | Symbol | Value in Paper | Meaning |
|---|---|---|---|
| Number of clients | N | 100 | Total participating quantum devices |
| Clients per round | K | 5 | Randomly selected each round |
| Local epochs | E | 1, 2, or 4 | Training iterations per client per round |
| Batch size | S | 32 | Samples per gradient step |
| Training rounds | R | 100 | Total federated rounds |
| Qubits | n | 4 | Quantum circuit width |
| VQC layers | L | 2 | Variational block repetitions |
| Quantum parameters | - | 24 | Total trainable quantum parameters (4 x 3 x 2) |
| Latent dimension | d | 4 | Output of modified VGG16 classifier |
| Dropout rate | p | 0.5 | Regularization in VGG16 classifier |

## 14.4 Algorithm Flow Summary

```
FEDERATED QUANTUM ML — SUMMARY FLOW

[INIT] Global model = VGG16 (frozen) + Modified Classifier + VQC (random)
       Data split: 100 clients, equal partitions
       Test data: central server

[ROUND r = 1..100]
  |-- Select 5 random clients
  |-- Send global VQC params to selected clients
  |-- Each client:
  |     |-- Train VQC on local data (E epochs, batch=32)
  |     |-- Compute gradients: classical backprop + parameter-shift
  |     |-- Upload trained VQC params
  |-- Server: Theta_global = mean(theta_1...theta_5)
  |-- Evaluate on central test set
  |-- Broadcast new global model

[OUTPUT] Final global model after 100 rounds
```

---

# 15. One-Page Master Summary Card

## Problem
How to train quantum machine learning models across distributed quantum devices without sharing sensitive raw data, while maintaining performance comparable to centralized training.

## Idea
Apply federated learning principles to hybrid quantum-classical models: clients train a variational quantum circuit locally, share only circuit parameters with a central server, which aggregates them into a global model.

## Method
- Hybrid architecture: Pre-trained VGG16 (frozen, feature extraction) + 4-qubit VQC (trainable, classification).
- 100 clients, 5 selected per round, simple parameter averaging aggregation.
- Quantum gradients via parameter-shift rule; classical gradients via backpropagation.
- 24 total trainable quantum parameters.

## Results
- Cats vs Dogs: 98.7% federated vs. 98.75% centralized (negligible gap).
- CIFAR-10 Planes vs Cars: 93.4-94.05% federated vs. 93.65% centralized.
- Single local epoch sufficient for convergence.
- 20x reduction in per-round data processing.

## Weakness
- No real quantum hardware validation.
- No privacy guarantees (no differential privacy or secure aggregation).
- IID data assumption only.
- No classical FL baseline comparison.
- No statistical rigor (no error bars).
- Very small quantum model (24 parameters).

## Research Opportunity
- Add differential privacy to quantum federated training with formal epsilon bounds.
- Test on real NISQ hardware and characterize noise impact.
- Handle non-IID data distributions with quantum-adapted FL algorithms.
- Compare rigorously against classical models of equivalent size.
- Implement robust aggregation for quantum parameter spaces.

## Publishable Extension
Combine any two of: (1) real hardware experiments, (2) formal privacy guarantees, (3) non-IID data handling, (4) quantum vs. classical comparison, (5) robust/adaptive aggregation — with proper statistical rigor and ablation studies — to produce a publishable extension that addresses the main gaps in this foundational work.

---
