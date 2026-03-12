# 18 — Driess et al. (2023): PaLM-E — An Embodied Multimodal Language Model

---

# 0. Quick Paper Identity Card

| Field | Detail |
|---|---|
| **Paper Title** | PaLM-E: An Embodied Multimodal Language Model |
| **Authors** | Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, Pete Florence |
| **Affiliations** | Robotics at Google, TU Berlin, Google Research |
| **Year** | 2023 |
| **Problem Domain** | Embodied AI, Multimodal Learning, Robot Planning, Vision-Language Models |
| **Paper Type** | Systems / Engineering + Experimental ML / Empirical |
| **Core Contribution** | A single large multimodal language model that integrates vision, language, and robot sensor data to perform embodied reasoning, robotic planning, visual question answering, and captioning — all in one unified model |
| **Key Idea** | Inject continuous sensor observations (images, robot states, 3D representations) directly into the token embedding space of a pre-trained large language model (PaLM), forming "multi-modal sentences" that allow the LLM to reason about the physical world |
| **Required Background** | Transformers, Large Language Models (LLMs), Vision Transformers (ViT), Autoregressive Language Modeling, Robotics basics, Visual Question Answering |
| **Primary Baselines** | PaLI (vision-language model), SayCan (LLM + affordance functions), Frozen (frozen LLM with vision encoder), CLIP-FT |
| **Main Innovation Type** | Architectural + Training Strategy (multimodal injection into LLM embedding space + multi-task co-training across robot embodiments and vision-language tasks) |
| **Difficulty Level** | Medium-High (conceptually accessible, but engineering-heavy at scale) |
| **Reproducibility Level** | Low (requires 562B parameter models, massive compute, proprietary robot data, Google-scale infrastructure) |

---

# 1. Research Context & Core Problem

## 1.1 Exact Problem Formulation

- Large language models (LLMs) are extremely powerful at text-based reasoning tasks — dialogue, math, code, step-by-step logic
- However, LLMs operate entirely in text space — they have **no direct access** to real-world sensory information like images, robot states, or 3D scene data
- This creates the **grounding problem**: LLMs can talk about the world but cannot perceive or interact with it
- For robotics, this means an LLM cannot look at a table with objects and decide which one to pick up — it needs someone to describe the scene in text first
- The core question: **Can we build a single model that takes in both language and raw sensor data (images, robot states, 3D representations) and reasons about embodied tasks like robot planning, while also doing well on vision-language tasks?**

## 1.2 Why the Problem Exists

- LLMs are trained on text-only internet data — they have no visual or physical experience
- Previous approaches (like SayCan) tried to solve this by keeping the LLM text-only and using separate vision/affordance modules to filter the LLM's outputs — but this creates a bottleneck because the LLM never actually "sees" the scene
- Standard vision-language models (like PaLI, Flamingo) are trained on image captioning and VQA tasks but are **not trained on robot data** — they cannot do embodied reasoning out of the box
- The geometric configuration of objects (exact positions, spatial relationships) matters enormously in robotics but is lost when you describe a scene in text

## 1.3 Historical / Theoretical Gap

- **SayCan (2022)**: Used LLMs for robot planning but only fed text descriptions to the LLM, relying on external affordance functions to ground decisions — limited because the LLM itself never perceives the environment
- **Flamingo (2022)**: Augmented LLMs with vision but focused on general VL tasks, not robotics
- **Frozen (2021)**: Trained vision encoders to produce embeddings that a frozen LLM could process — but limited to simple vision-language tasks and did not address embodied reasoning
- **Gato (2022)**: A generalist multi-embodiment agent, but did not demonstrate positive transfer across different tasks/embodiments
- **Gap**: No single model existed that could simultaneously do well on (a) robotic planning across multiple embodiments, (b) visual question answering, (c) image captioning, AND (d) general language tasks

## 1.4 Contribution Category

- **System Design**: Novel architecture for injecting multimodal continuous data into LLM embedding space
- **Empirical Insight**: Demonstration of positive transfer from vision-language data to robotics tasks
- **Architectural Innovation**: Neural scene representations (OSRT) and entity-labeling multimodal tokens as inputs
- **Scaling Insight**: Larger model scale leads to less catastrophic forgetting during multimodal finetuning

### Why This Paper Matters

- It is the **first demonstration** that a single large embodied multimodal model can simultaneously solve robotic planning, visual QA, captioning, and retain language abilities
- It proves that **transfer learning works across very different domains** — training on internet-scale vision-language data significantly improves robotics performance, even with very few robot examples
- It introduces **multi-modal sentences** as a clean, flexible way to mix any combination of modalities into an LLM
- At 562B parameters, it was the **largest vision-language model** reported at the time, showing that scale brings emergent multimodal capabilities (zero-shot chain-of-thought on images, multi-image reasoning)

### Remaining Open Problems

1. **Compute requirements** are prohibitive — 562B parameters requires enormous infrastructure
2. **Real robot data is still sparse** — the model relies on only thousands of robot demonstrations
3. **Low-level control is separate** — PaLM-E only generates high-level text plans, requiring separate low-level policies to execute actions
4. **Catastrophic forgetting** for smaller models — the 12B model loses 87% of NLG performance during multimodal training
5. **No true end-to-end robot control** — still needs the two-level hierarchy (PaLM-E for planning + RT-1 or similar for actions)
6. **Generalization to truly novel environments** remains underexplored
7. **Safety and robustness** of deploying such a massive model in real robots is not addressed

---

# 2. Minimum Background Concepts

## 2.1 Large Language Models (LLMs)

- **Definition**: Neural networks trained on massive text data to predict the next token in a sequence
- **Role in paper**: PaLM (the base LLM) provides the reasoning backbone — all of PaLM-E's intelligence comes from the pre-trained LLM, which the authors augment with multimodal inputs
- **Why needed**: The LLM's internalized world knowledge (from trillions of text tokens) is what enables planning and reasoning — without it, the model would need to learn everything from scratch

## 2.2 Decoder-Only Autoregressive Models

- **Definition**: Models that generate text one token at a time, left to right, where each new token depends on all previous tokens
- **Role in paper**: PaLM-E inherits this architecture from PaLM — it generates robot plans and answers as sequences of tokens, one at a time
- **Why needed**: This architecture naturally supports open-ended generation (plans of variable length, free-form answers)

## 2.3 Vision Transformer (ViT)

- **Definition**: A transformer model that processes images by dividing them into patches (like 16×16 pixels), treating each patch as a "token", and applying self-attention
- **Role in paper**: ViT is used as the image encoder — it converts raw images into a sequence of embedding vectors that can be fed into the LLM alongside text tokens
- **Why needed**: To give the LLM "eyes" — the ViT converts visual information into the same format the LLM understands (embedding vectors)

## 2.4 Token Embedding Space

- **Definition**: The internal vector space where the LLM represents words — each word/subword is mapped to a high-dimensional vector (e.g., 2048 dimensions)
- **Role in paper**: PaLM-E's key innovation is mapping ALL modalities (text, images, robot states, 3D representations) into this SAME embedding space, so the LLM treats them identically
- **Why needed**: By putting everything in one shared space, the LLM's self-attention can jointly reason over text and multimodal inputs without any architectural changes

## 2.5 Grounding (in AI/Robotics)

- **Definition**: Connecting abstract symbols (words, concepts) to real-world physical entities and their properties
- **Role in paper**: The entire paper is about solving the grounding problem — making the LLM understand not just the word "red block" but the actual visual appearance and spatial position of a red block on a table
- **Why needed**: Without grounding, an LLM can generate plausible-sounding robot plans that are physically impossible

## 2.6 Affordance Functions

- **Definition**: Functions that tell you whether a particular action is possible given the current state of the environment (e.g., "Can I pick up this object right now?")
- **Role in paper**: Previous work (SayCan) used separate affordance functions to constrain LLM outputs — PaLM-E learns affordances implicitly through its multimodal training, eliminating the need for separate modules
- **Why needed**: Understanding what previous approaches did helps explain why PaLM-E's integrated approach is better

## 2.7 Object Scene Representation Transformer (OSRT)

- **Definition**: A neural network that takes multiple views of a scene and learns to decompose it into individual object representations ("slots") in a 3D-aware, unsupervised manner
- **Role in paper**: One of the novel input encoders explored — OSRT provides object-centric 3D representations that are especially effective for robotic planning tasks
- **Why needed**: Standard ViT processes images as flat grids, but robotics requires understanding individual objects and their spatial relationships — OSRT addresses this

## 2.8 Multi-Modal Sentences

- **Definition (introduced by this paper)**: Sequences where text tokens and encoded continuous observations (images, states, 3D representations) are interleaved in a single input sequence to an LLM
- **Role in paper**: This is the core architectural idea — instead of processing text and images separately, they become one unified sequence
- **Why needed**: This allows flexible, arbitrary combinations of modalities at any position in the input, enabling prompts like "Given [IMAGE], what happened between [IMAGE1] and [IMAGE2]?"

---

# 3. Mathematical / Theoretical Understanding Layer

## 3.1 Autoregressive Language Modeling

### Intuition
- The LLM learns to predict "what comes next" given everything that came before
- Think of it as an extremely sophisticated autocomplete that can write entire paragraphs, solve math, or plan robot actions

### Core Equation

$$p(w_{1:L}) = \prod_{l=1}^{L} p_{LM}(w_l \mid w_{1:l-1})$$

| Symbol | Meaning |
|---|---|
| $w_{1:L}$ | A sequence of $L$ text tokens |
| $w_l$ | The $l$-th token in the sequence |
| $p_{LM}$ | The probability distribution computed by the large language model (a transformer) |
| $p(w_{1:L})$ | The joint probability of the entire text sequence |

- **What it solves**: Defines how the model assigns probability to any piece of text
- **Assumption**: Each token only depends on the tokens that came before it (causal/left-to-right modeling)
- **Practical interpretation**: At inference time, the model generates one token at a time, sampling from $p_{LM}(w_l \mid w_{1:l-1})$

## 3.2 Prefix-Conditioned Generation

### Intuition
- Instead of generating text from scratch, you give the model context (a "prompt") and ask it to continue
- The prompt can contain instructions, examples, or questions

### Core Equation

$$p(w_{n+1:L} \mid w_{1:n}) = \prod_{l=n+1}^{L} p_{LM}(w_l \mid w_{1:l-1})$$

| Symbol | Meaning |
|---|---|
| $w_{1:n}$ | The prefix/prompt (first $n$ tokens) |
| $w_{n+1:L}$ | The generated continuation |

- **What it solves**: Allows the model to be steered by providing appropriate context — this is the mechanism for task specification, few-shot learning, and multi-modal prompting

## 3.3 Token Embedding and Multi-Modal Injection

### Intuition
- Every text token gets mapped to a vector in a high-dimensional space before being processed by the transformer
- PaLM-E's key idea: map non-text inputs (images, states) into the SAME vector space, so the transformer sees them as "just more tokens"

### Core Mechanism

$$x_i = \begin{cases} \gamma(w_i) & \text{if token } i \text{ is a text token} \\ \phi_j(O_j) & \text{if token } i \text{ corresponds to observation } O_j \end{cases}$$

| Symbol | Meaning |
|---|---|
| $\gamma$ | The word token embedding function (a learned lookup table of size $k \times \|W\|$, where $\|W\| = 256{,}000$) |
| $w_i$ | A text token from vocabulary $W$ |
| $x_i$ | The embedding vector in $\mathbb{R}^k$ fed to the transformer |
| $\phi_j$ | An encoder for observation modality $j$ (maps continuous data to the embedding space) |
| $O_j$ | A continuous observation (e.g., image, robot state, 3D scene) |
| $k$ | Dimension of the embedding space |
| $q$ | Number of embedding vectors produced per observation |

### Key Design Choices
- An encoder $\phi : \mathcal{O} \rightarrow \mathcal{X}^q$ maps one observation into $q$ vectors (one image may produce many "tokens")
- These observation tokens are placed at DYNAMIC positions in the sequence — they can appear anywhere text would appear
- The LLM's existing positional encodings are reused, so observation tokens get position information naturally

### Practical Interpretation
- When the model sees "Given [IMAGE]. Q: What color is the block?", the [IMAGE] gets replaced by, say, 256 embedding vectors from the ViT encoder, sitting inline between the text embeddings for "Given" and ". Q: What"
- The transformer's self-attention then jointly processes all these vectors (text + image) identically

## 3.4 Training Loss

### Intuition
- The model is trained to predict the correct text continuation (answer, plan) given the multi-modal prompt
- Only the OUTPUT tokens are used to compute the loss — the multi-modal prompt tokens give context but are not predicted

### Equation
- Standard cross-entropy loss computed over non-prefix tokens $w_{n+1:L}$
- The special tokens in the text that mark where observations should be inserted get **replaced at runtime** by the actual encoder outputs

### Variable Meaning Table for Training

| Symbol | Meaning |
|---|---|
| $\mathcal{D}$ | Training dataset of $N$ examples |
| $I^i_{1:u_i}$ | $u_i$ continuous observations for example $i$ |
| $w^i_{1:L_i}$ | Text sequence for example $i$ |
| $n_i$ | Index separating the prefix (multi-modal prompt) from the prediction target |

### Mathematical Insight Box

> **Key Insight**: The genius of PaLM-E is that it does NOT change the LLM architecture. It only changes WHAT goes into the embedding layer. By training encoders to produce embedding vectors that "speak the language" of the LLM's internal representations, any continuous modality can be injected. The LLM's transformer layers, attention mechanisms, and token prediction head remain exactly the same. This means all the pre-trained knowledge is preserved and the model can jointly attend over text AND sensory data.

---

# 4. Proposed Method / Framework (MOST IMPORTANT)

## 4.1 Overall Pipeline

```
Raw Inputs (Images, Robot States, 3D Scenes, Text)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Input Encoders (one per modality)               │
│  • ViT for images → embedding vectors           │
│  • MLP for state vectors → embedding vectors     │
│  • OSRT for 3D scenes → object slot embeddings   │
│  • γ (word embedding) for text → token embeddings│
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Multi-Modal Sentence Construction               │
│  Interleave all embeddings in sequence order     │
│  e.g., [text][image][text][state][text]          │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Pre-trained PaLM (decoder-only LLM)             │
│  Self-attention processes ALL tokens identically  │
│  Generates text output autoregressively           │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Output: Generated Text                          │
│  • Answer to VQA question                        │
│  • High-level robot plan steps                   │
│  • Image caption                                 │
└─────────────────────────────────────────────────┘
        │ (if robot planning)
        ▼
┌─────────────────────────────────────────────────┐
│  Low-Level Policy (e.g., RT-1)                   │
│  Executes text instructions as robot actions     │
│  Returns new observations → re-query PaLM-E     │
└─────────────────────────────────────────────────┘
```

## 4.2 Input Encoders (Components/Modules)

### 4.2.1 State Estimation Vectors (Simplest)
- **What**: Robot joint positions, object poses, sizes, colors encoded as a flat vector $s \in \mathbb{R}^S$
- **How**: An MLP $\phi_{state}$ maps $s$ into the language embedding space
- **Why authors did this**: State vectors are the most direct representation — no information loss from perception
- **Weakness**: Requires ground-truth state information, which is not available in most real-world scenarios
- **Improvement idea**: Learn state estimation jointly from raw sensor data end-to-end

### 4.2.2 Vision Transformer (ViT) — Main Visual Encoder
- **What**: Takes a raw image $I$ and produces $m$ token embeddings
- **How**: Image is divided into 16×16 patches, each patch is linearly projected, and transformer layers process all patches with self-attention
- **Variants used**:
  - ViT-4B (4 billion parameters, pre-trained on image classification)
  - ViT-22B (22 billion parameters, largest ViT at the time)
  - ViT + Token Learner (TL) — adds learned aggregation of ViT tokens, trained from scratch
- **Projection**: Since ViT embedding dimension $\tilde{k}$ may differ from LLM embedding dimension $k$, a learned affine transformation $\psi$ maps $\tilde{x}_i \rightarrow x_i$
- **Why authors did this**: ViT is the standard, proven approach for extracting rich visual features; pre-trained ViTs already encode semantic understanding of images
- **Weakness**: ViT outputs a flat grid of patch tokens — no explicit notion of objects, which matters for robotic manipulation
- **Improvement idea**: Combine ViT with object detection or segmentation to get both global and object-level features

### 4.2.3 Object-Centric Representations
- **What**: Instead of treating the image as a flat grid, decompose it into per-object representations
- **How (with ground-truth masks)**: Given object instance masks $M_j$, apply ViT to each masked region: $x^j = \phi_{ViT}(M_j \circ I)$
- **How (with OSRT, unsupervised)**: OSRT takes multiple views, discovers objects automatically through inductive biases, and produces per-object "slots" $o_j$. Each slot is projected into $m$ embedding vectors via MLP $\psi$
- **Why authors did this**: Robots interact with individual objects — having per-object representations makes it easier to plan actions like "pick up object 3 and place it on object 5"
- **Weakness**: OSRT requires multiple views (not always available) and its object discovery may fail in cluttered scenes
- **Improvement idea**: Use modern single-view object-centric models (like SAM or DETR) to get object representations without needing multiple views

### 4.2.4 Entity Referrals
- **What**: A mechanism for PaLM-E to refer to specific objects in its generated plans
- **How**: Prepend the prompt with labels like "Object 1 is \<obj1\>. Object 2 is \<obj2\>." where \<obj_j\> is replaced by that object's encoded representation. PaLM-E can then use "obj1", "obj2" etc. in its output text
- **Why authors did this**: When multiple similar objects exist (e.g., same-colored blocks at different positions), natural language descriptions are ambiguous. Entity referrals provide unique identifiers
- **Weakness**: Requires knowing how many objects exist and establishing the correspondence between referral tokens and physical objects
- **Improvement idea**: Automatically discover and label entities without manual scene setup

## 4.3 The Multi-Modal Sentence Mechanism

- **Step 1**: Tokenize the text prompt as usual
- **Step 2**: At positions where special tokens like \<img\> or \<state\> appear, replace them with the encoder outputs (which may be multiple vectors each)
- **Step 3**: The resulting sequence of embedding vectors (mixed text and multimodal) becomes the prefix for the LLM
- **Step 4**: The LLM processes this mixed sequence with its standard self-attention and generates text output

**Critical design choice**: Observation embeddings are placed at DYNAMIC positions in the text, not fixed positions. This means you can construct prompts like:
- "Given \<img1\>. Q: What happened between \<img1\> and \<img2\>?"
- "\<state\> Robot, pick up the \<img\> object"

This flexibility is a significant advantage over models (like Flamingo) that insert images at fixed positions.

## 4.4 Robot Control Loop Integration

- **Step 1**: Human provides a high-level instruction (e.g., "Bring me something to clean a spill")
- **Step 2**: PaLM-E receives the instruction + current camera image as a multi-modal sentence
- **Step 3**: PaLM-E generates the FIRST step of the plan in text (e.g., "Find a sponge")
- **Step 4**: A low-level policy (e.g., RT-1) executes this step on the real robot
- **Step 5**: New camera observation is taken
- **Step 6**: PaLM-E receives the updated observation + history of completed steps, generates the NEXT step
- **Step 7**: Repeat until PaLM-E outputs "terminate"

**Why this closed-loop design**: If a step fails or the environment changes (adversarial disturbances), PaLM-E can re-observe and re-plan — it is not committed to a static plan

**Weakness of control loop**: Latency — going through a 562B parameter model for every decision step is slow; also dependent on low-level policy quality

**Improvement idea**: Train the model to also handle low-level actions directly or use a smaller, distilled version for real-time control

## 4.5 Training Strategy

### 4.5.1 Model Configurations
| Name | LLM | Vision Encoder | Total Parameters |
|---|---|---|---|
| PaLM-E-12B | PaLM 8B | ViT-4B | 12B |
| PaLM-E-84B | PaLM 62B | ViT-22B | 84B |
| PaLM-E-562B | PaLM 540B | ViT-22B | 562B |

### 4.5.2 Freezing vs. Finetuning
- **Option A (Frozen LLM)**: Only train the input encoders and projection layers. The LLM parameters stay fixed. This preserves language capabilities perfectly but may underperform on robotics tasks
- **Option B (Unfrozen/Full finetuning)**: Train everything end-to-end. Better performance on embodied tasks but risks catastrophic forgetting of language abilities
- **Key finding**: Freezing works reasonably well, but finetuning + scale is the better path — a 562B model finetuned end-to-end loses only 3.8% of NLG capability

### 4.5.3 Co-Training Across Tasks (Full Mixture)
- The model is trained on a mixture of datasets simultaneously:
  - 52.4% WebLI (web-scraped image-language pairs)
  - 13.1% VQ2A (visual QA)
  - 13.1% CC3M (conceptual captions)
  - 5.2% VQG, 5.2% Object Aware
  - 0.5% each for OK-VQA, VQAv2, COCO, Wikipedia
  - **Only 8.9% total is robot data** (3.1% mobile manipulation, 4.2% Language-Table, 1.6% TAMP)
- **Why this matters**: Despite containing less than 9% robot data, the model achieves state-of-the-art embodied reasoning — the internet-scale vision-language data transfers massively to robotics

### Simplified Pseudocode

```
ALGORITHM: PaLM-E Training and Inference

Training:
1. Load pre-trained PaLM (LLM) and pre-trained ViT (vision encoder)
2. Initialize projection layers ψ (randomly)
3. For each batch from the mixed dataset:
   a. Parse text to find special tokens (<img>, <state>, <obj>)
   b. Encode each continuous observation via its encoder φ
   c. Replace special tokens with encoder outputs in embedding space
   d. Feed mixed embedding sequence into PaLM
   e. Compute cross-entropy loss on prediction target tokens only
   f. Backpropagate and update encoder + projection + (optionally) LLM

Inference (Robot Planning):
1. Receive: high-level instruction + current observation
2. Construct multi-modal sentence: "Human: <instruction> I see <img>"
3. Feed to PaLM-E, generate text output
4. Parse generated text as the next action step
5. Execute via low-level policy on robot
6. Observe new state, append step history, repeat from step 2
7. Stop when PaLM-E generates "terminate"
```

---

# 5. Experimental Setup / Evaluation Design

## 5.1 Robot Environments

### 5.1.1 TAMP (Task and Motion Planning) — Simulation
- **Setup**: Tabletop with 3-8 cube-shaped objects of different sizes and colors
- **Tasks**: 
  - Planning tasks (p1: how to grasp an object when others block it; p2: how to stack objects)
  - VQA tasks (q1: object color; q2: object position on table; q3: vertical object relations; q4: plan feasibility)
- **Why chosen**: Tests combinatorial reasoning — many possible plans, most are infeasible
- **Dataset**: 96,000 training scenes (full), experiments also test with only 1% (320 examples per planning task)

### 5.1.2 Language-Table — Simulation + Real Robot
- **Setup**: Tabletop with multiple blocks, a real robot arm does pushing
- **Tasks**: Three long-horizon manipulation tasks like "sort blocks by color into corners", "push block to same-colored block"
- **Why chosen**: Tests complex dynamics understanding (pushing is harder than pick-and-place), requires fine-grained spatial reasoning
- **Control loop**: PaLM-E outputs text sub-goals at 1 Hz, low-level policy executes at 5 Hz
- **Few-shot regime**: Tested with only 10-80 demonstrations per task

### 5.1.3 Mobile Manipulation — Real Robot (Kitchen)
- **Setup**: Real robot in a kitchen environment (similar to SayCan setup)
- **Tasks**: 
  - Affordance prediction: "Is it possible to \<skill\> here?"
  - Failure detection: "Was \<skill\> successful?"
  - Long-horizon planning: "I spilled my drink, bring me something to clean it"
- **Low-level policy**: RT-1 (Robotics Transformer) executing navigation and manipulation skills
- **Training data**: 2,912 sequences from prior SayCan runs
- **Why chosen**: Most realistic and challenging setting — real kitchen, real robot, complex multi-step tasks

## 5.2 General Vision-Language Benchmarks
- **OK-VQA**: Visual QA requiring external/world knowledge
- **VQAv2**: Standard visual question answering
- **COCO Captioning**: Image captioning on MS COCO (Karpathy splits)

## 5.3 Language Benchmarks
- **21 benchmarks** covering NLU (natural language understanding) and NLG (natural language generation)
- Including: TriviaQA, Natural Questions, HellaSwag, Winograd, RACE, PIQA, ARC, BoolQ, etc.
- **Purpose**: Measure catastrophic forgetting — how much language ability is lost after multimodal training

## 5.4 Metrics
| Metric | Used For | Why This Metric |
|---|---|---|
| Planning Success Rate | TAMP, Language-Table, Mobile Manipulation | Measures whether generated plans actually achieve the goal when executed |
| VQA Accuracy | TAMP VQA, General VQA | Standard metric for question answering correctness |
| F1 Score | Affordance prediction, Failure detection | Balances precision and recall for binary classification |
| CIDEr Score | COCO Captioning | Standard caption quality metric |
| NLU/NLG Average | Language benchmarks | Tracks overall language ability retention |

## 5.5 Baseline Selection Logic
- **PaLI**: State-of-the-art VLM at the time — tests whether general VL models can solve embodied tasks (answer: no, they cannot out of the box)
- **SayCan (with oracle affordances)**: The best prior method for LLM-based robot planning — uses separate affordance functions; even with oracle (perfect) affordances, it struggles on TAMP
- **Frozen**: The closest prior architecture to PaLM-E for frozen-LLM vision-language learning — PaLM-E outperforms by >45% on VQAv2
- **CLIP-FT / CLIP-FT-hindsight**: Baselines for failure detection task
- **QT-OPT**: Value-function based baseline for affordance prediction

## 5.6 Hyperparameter and Compute Considerations
- Three model scales: 12B, 84B, 562B parameters
- Full mixture training on internet-scale data
- Fine-tuning experiments: 9,000 additional steps for Language-Table task-specific models
- Robot data constitutes only 8.9% of training mixture

### Experimental Reliability Analysis

**What is trustworthy**:
- Multiple environments (3 robot domains + general VL + language) provide comprehensive evaluation
- Real-world robot experiments (Language-Table real robot, kitchen mobile manipulation) validate beyond simulation
- Transfer experiments are well-controlled (same architecture, just different data mixtures)
- Comparison against strong baselines (PaLI, SayCan with oracle affordances)

**What is questionable**:
- Real robot evaluations are only "qualitative" for some tasks (long-horizon mobile manipulation)
- No error bars or confidence intervals reported for most experiments
- The TAMP environment is relatively simple (blocks on a table) — unclear how it extends to complex manipulation
- The 562B model's real-time feasibility for robot control is not discussed
- The dependence on low-level policy quality is not systematically analyzed

---

# 6. Results & Findings Interpretation

## 6.1 Main Outcomes

### Transfer Learning is the Central Finding
- Across ALL three robotic domains, training on the full mixture (internet VL data + all robot data) significantly outperforms training on robot-specific data alone
- For TAMP with only 1% data (320 examples): ViT-4B with full mixture achieves 74.1% planning success vs. 30.6% with single-robot data — more than **2x improvement**
- For Language-Table: Full mixture model with 40 demos achieves 80% on Task 1 vs. 50% for single-robot training

### Input Representation Comparison (TAMP Environment)
- **OSRT** (3D-aware, object-centric) achieves **best overall performance** — 82.5%/76.2% on planning tasks with only 320 examples, beating all other representations
- **State vectors** with pre-trained LLM work well when ground truth is available (97-100% on VQA, 55.9%/49.7% on planning with 1% data)
- **ViT-4B** alone struggles on planning with limited data (30.6%/32.9%), but excels when co-trained on full mixture (74.1%/74.6%)
- **ViT + Token Learner** trained from scratch performs poorly in low-data regime (24%/14.7%)

### Out-of-Distribution Generalization
- Pre-trained LLMs generalize to more objects (6, 8) much better than non-pre-trained ones
- With 62B LLM: reasonable OOD performance even on 6 objects with OOD tasks
- Without pre-training: "basically no out-of-distribution generalization"

### General Vision-Language Results
- PaLM-E-562B achieves **state-of-the-art on OK-VQA** (66.1%) as a GENERALIST model (one checkpoint for all tasks), surpassing even task-specific finetuned models like PaLI
- On VQAv2: 80.0% (generalist) — competitive with finetuned models
- With frozen LLM: PaLM-E-12B achieves 70.3% on VQAv2, significantly outperforming Frozen (48.4%)

### Catastrophic Forgetting and Scale
- **PaLM-E-12B (unfrozen)**: Loses 87.3% of NLG performance, 15.0% of NLU
- **PaLM-E-84B (unfrozen)**: Loses 61.6% of NLG, 4.3% of NLU
- **PaLM-E-562B (unfrozen)**: Loses only 3.8% of NLG, and actually GAINS 0.4% on NLU
- **Critical insight**: Scale is a powerful mitigation for catastrophic forgetting — the 562B model barely loses any language ability

### Emergent Capabilities (PaLM-E-562B)
- Zero-shot multimodal chain-of-thought reasoning
- Multi-image reasoning (despite being trained only on single-image examples)
- OCR-free math reasoning on images with handwritten numbers
- Visually-conditioned jokes
- Zero-shot egocentric video question answering

## 6.2 Performance Trends

- **More data diversity = better robot performance** (consistent across all environments)
- **Larger model = less forgetting + better generalization**
- **Pre-training (both LLM and ViT) is essential** — training from scratch fails in low-data robot regimes
- **Object-centric representations are most data-efficient** for structured robotic tasks
- Freezing the LLM is a **viable but suboptimal** path

## 6.3 Failure Cases

- SayCan and PaLI (zero-shot) **completely fail** on Language-Table tasks (0% success)
- ViT-4B trained only on single robot struggles severely on planning tasks with 1% data
- Small models (12B) with unfrozen LLM lose most language ability
- Non-pre-trained LLMs show essentially no OOD generalization

## 6.4 Statistical Observations

- No error bars or statistical tests are reported — results are presented as single numbers
- Real robot evaluations for mobile manipulation are qualitative only
- The "success rate" metric can mask partial successes

### Publishability Strength Check

**Publication-grade results**:
- Transfer learning demonstration across 3 robot domains + general VL tasks
- OK-VQA state-of-the-art as a generalist model
- Catastrophic forgetting vs. scale analysis
- Input representation comparison (OSRT vs ViT vs state)
- Few-shot / zero-shot generalization on Language-Table

**Needs stronger validation**:
- Mobile manipulation planning (only qualitative evaluation)
- No statistical significance testing
- Limited real-world diversity (kitchen environment only)
- Computational cost analysis missing
- Latency/throughput for real-time robot control not measured

---

# 7. Strengths – Weaknesses – Assumptions

## 7.1 Technical Strengths

| # | Strength | Why It Matters |
|---|---|---|
| 1 | Single model handles robotics + VQA + captioning + language | Proves general-purpose embodied AI is feasible |
| 2 | Positive transfer from internet VL data to robotics | Addresses robotics data scarcity |
| 3 | Multi-modal sentences allow flexible input composition | Any modality can appear anywhere in the prompt |
| 4 | OSRT as input encoder provides data-efficient 3D-aware representations | Best performance with minimal robot data |
| 5 | Scale mitigates catastrophic forgetting | 562B model retains language abilities |
| 6 | Closed-loop replanning handles disturbances | Robust to execution failures |
| 7 | Entity referral mechanism solves object ambiguity | Enables precise robotic manipulation instructions |
| 8 | Zero-shot emergent capabilities at scale | Chain-of-thought, multi-image reasoning without explicit training |

## 7.2 Explicit Weaknesses

| # | Weakness | Impact |
|---|---|---|
| 1 | Enormous computational requirements (562B params) | Not reproducible by most labs; impractical for real-time robot control |
| 2 | Depends on separate low-level policies (RT-1, etc.) | Not truly end-to-end; limited by low-level policy quality |
| 3 | Small models suffer severe catastrophic forgetting | 12B loses 87% NLG — not useful as a language model after training |
| 4 | Real robot evaluation is mostly qualitative | Hard to assess actual reliability and safety |
| 5 | TAMP environment is relatively simple (blocks only) | Unclear generalization to complex real-world manipulation |
| 6 | No latency / throughput analysis | Unknown whether model can operate at robot-useful speeds |
| 7 | Only 2,912 mobile manipulation training sequences | Limited training data for the most complex domain |

## 7.3 Hidden Assumptions

| # | Assumption | Risk |
|---|---|---|
| 1 | Low-level policies are reliable and cover needed skills | If a skill is missing or unreliable, the entire plan fails |
| 2 | Text is a sufficient interface between high-level planning and low-level control | Some nuances (force, speed, precision) are hard to express in text |
| 3 | Pre-trained PaLM contains enough world knowledge for robot reasoning | Domain-specific knowledge (e.g., chemical handling, delicate objects) may be absent |
| 4 | Object-centric decomposition can be done reliably | OSRT may fail in cluttered or occluded scenes |
| 5 | Internet-scale VL data is representative enough to transfer to robotics | Web images and robot camera images have different distributions |
| 6 | Cross-entropy training loss is sufficient for planning tasks | Planning may benefit from reward-based or RL-based objectives |
| 7 | Autoregressive generation is appropriate for plan generation | Plans have complex interdependencies that may not be well-captured left-to-right |

---

# 8. Weakness → Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| 562B params impractical for robots | Scaling laws favor large models; compression not explored | Efficient embodied multimodal models | Knowledge distillation, model pruning, quantization specifically for robot deployment |
| Separate low-level policies needed | PaLM-E outputs text, not motor commands | End-to-end embodied model that also outputs actions | Combine PaLM-E-style perception with action token prediction (similar to RT-2 but larger scale) |
| Catastrophic forgetting at small scale | Multimodal finetuning overwrites language representations | Scale-independent forgetting mitigation | LoRA / adapter-based finetuning, experience replay, progressive neural networks |
| Qualitative-only real robot evaluation | Automated metrics for long-horizon tasks are hard to define | Standardized embodied AI benchmarks | Community-driven benchmark suites with automated scoring |
| Simple tasks only (blocks, kitchen) | Data collection for complex manipulation is expensive | Learning from demonstration + simulation at scale | Combine sim-to-real transfer with foundation model planning |
| No latency analysis | Paper focuses on capability, not efficiency | Real-time embodied LLM inference | Speculative decoding, model cascading, edge deployment of smaller models |
| Text as only robot interface | Design choice for simplicity | Richer action representations | Predict action tokens, trajectory parameters, or code programs alongside text |
| OSRT needs multiple views | Architectural limitation of scene representation | Single-view 3D-aware object representations | Monocular depth + object segmentation (SAM) + 3D lifting |
| No safety analysis | Not in scope of the paper | Safe embodied AI planning | Add constraint verification layers, uncertainty quantification for plans |

---

# 9. Novel Contribution Extraction

## 9.1 Explicit Novel Claims from the Paper

1. "We propose **embodied language models** that directly incorporate real-world continuous sensor modalities into language models, establishing the link between words and percepts through multi-modal sentences."
2. "We demonstrate that a **single generalist, transfer-learned, multi-embodiment** decision-making agent can be trained by mixing embodied data into multimodal LLM training."
3. "We introduce **neural scene representations (OSRT) and entity-labeling multimodal tokens** as novel architectural ideas for injecting structured 3D-aware information into LLMs."
4. "We show that PaLM-E-562B is a **quantitatively competitive vision-language generalist** (SOTA on OK-VQA) while simultaneously being an effective embodied reasoner."
5. "We demonstrate that **scaling the language model size** enables multimodal finetuning with significantly less catastrophic forgetting."

## 9.2 Reusable Novel Claim Templates

1. "We propose ______ that integrates ______ modality into large language models by ______, enabling embodied reasoning with ______ improvement over the prior state of the art."

2. "We demonstrate that co-training on ______ and ______ data leads to positive transfer, achieving ______ performance on ______ tasks with only ______ domain-specific examples."

3. "We show that ______ representations, when injected into LLMs via ______, provide superior data efficiency for ______ tasks, outperforming ______ by ______."

4. "We introduce a ______ mechanism that allows multimodal language models to reference ______ in their generated outputs, enabling ______ without external ______ modules."

5. "We find that scaling model size to ______ parameters mitigates catastrophic forgetting during ______ finetuning, with only ______% degradation in ______ capabilities."

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Work

- Combine OSRT's geometric data efficiency with large-scale visual pre-training
- Further investigate the frozen LLM path as a route to general-purpose embodied models
- Explore whether scale alone can fully eliminate catastrophic forgetting

## 10.2 Missing Directions (Not Explored by Authors)

- **Model compression for deployment**: How to get PaLM-E performance in a model small enough for real-time robot control
- **Safety and constraint verification**: Ensuring generated plans do not cause harm
- **Multi-robot coordination**: Extending from single-robot to multi-robot scenarios
- **Longer time horizons**: Tasks spanning hours or days, not just minutes
- **Learning from failure**: Using execution failures to improve the model online
- **Human-in-the-loop refinement**: Interactive correction of plans

## 10.3 Modern Extensions (Post-2023)

- **PaLM-E → RT-2**: The direct successor — predicting action tokens directly rather than text plans
- **PaLM-E → Gemini-style models**: Modern multimodal models that natively process images, video, audio, and text
- **Integration with world models**: Combining PaLM-E-style planning with learned dynamics models (DreamerV3, etc.)
- **Open-source alternatives**: Applying the PaLM-E recipe to open-source LLMs (LLaMA, Mistral) to enable broader research

## 10.4 Cross-Domain Combinations

- **PaLM-E + surgical robotics**: Embodied multimodal reasoning for medical procedures
- **PaLM-E + autonomous driving**: Multimodal understanding of driving scenes + planning
- **PaLM-E + scientific experiments**: Embodied AI that can plan and execute laboratory experiments
- **PaLM-E + accessibility**: Embodied AI assistants for people with disabilities

## 10.5 LLM-Era Extensions

- **Multimodal chain-of-thought** for robotics (explicitly reasoning through steps before acting)
- **In-context learning for new robots** (few-shot adaptation to new embodiments without retraining)
- **Retrieval-augmented embodied reasoning** (looking up relevant past experiences or manuals when planning)
- **Natural language feedback** for plan correction (humans providing verbal corrections mid-execution)

---

# 11. How to Write a NEW Paper From This Work

## 11.1 Reusable Elements

- **Multi-modal sentence framework**: The idea of interleaving continuous observations with text tokens inside an LLM is broadly applicable — use it for any domain where you combine text with continuous data (audio, sensor streams, medical images)
- **Transfer learning evaluation protocol**: Showing that co-training on diverse data improves target domain performance is a powerful experimental design pattern
- **Frozen vs. unfrozen analysis**: Systematically comparing frozen and finetuned LLM settings on the same tasks
- **Scaling analysis for catastrophic forgetting**: Measuring language retention across model sizes
- **Input representation comparison**: Testing multiple encoders (simple MLP, ViT, object-centric) on the same task
- **Few-shot regime testing**: Evaluating with 1% of data, 10-80 examples to show data efficiency

## 11.2 What MUST NOT Be Copied

- The exact PaLM-E architecture and training recipe (proprietary)
- Specific dataset mixtures and sampling frequencies
- The exact PaLM or ViT-22B model weights
- Specific prompt templates and evaluation scripts
- The SayCan task definitions and environment setups (cite if reused)

## 11.3 How to Design a Novel Extension

1. **Pick a limitation**: Choose one weakness from Section 8 (e.g., model size, catastrophic forgetting at small scale, text-only output)
2. **Define your contribution scope**: Are you proposing a new architecture, training method, or application domain?
3. **Select an open-source base**: Use LLaMA/Mistral + open ViT as your foundation instead of PaLM
4. **Design a controlled experiment**: Compare your method against reproducing PaLM-E's approach on the same base model
5. **Add a novel component**: E.g., adapter-based finetuning to reduce forgetting, action token prediction for end-to-end control, novel scene representations
6. **Evaluate systematically**: Show improvement on at least one robot domain AND one VL benchmark

## 11.4 Minimum Publishable Contribution Checklist

| # | Requirement | Details |
|---|---|---|
| 1 | Novel technical component | Must go beyond simply reproducing PaLM-E with different models |
| 2 | At least one robotics evaluation | Real or sim, with quantitative success rates |
| 3 | At least one VL benchmark | To show generalist capabilities |
| 4 | Transfer learning analysis | Show benefit of multi-task co-training |
| 5 | Comparison with strong baselines | Must compare against PaLM-E or its open-source reproductions |
| 6 | Ablation study | Remove your contribution and show performance drops |
| 7 | Statistical significance | Report means, standard deviations, multiple seeds |

---

# 12. Publication Strategy Guide

## 12.1 Suitable Venues

| Venue Type | Specific Venues | Why Suitable |
|---|---|---|
| Top AI conferences | ICML, NeurIPS, ICLR | Multimodal learning, foundation models, transfer learning |
| Robotics conferences | CoRL, ICRA, RSS | Embodied AI, robot planning, manipulation |
| Computer Vision | CVPR, ECCV | Vision-language models, visual reasoning |
| Applied AI | AAAI, IJCAI | General AI systems, multi-task learning |

## 12.2 Required Baseline Expectations

- If claiming embodied multimodal improvements: **Must compare against PaLM-E** (or open-source equivalents like OpenFlamingo, LLaVA applied to embodied tasks)
- If claiming efficient embodied models: **Must show performance vs FLOPs/parameters tradeoff**
- If claiming transfer learning improvements: **Must include ablation of data mixtures**

## 12.3 Experimental Rigor Level

- Top venues expect: Multiple environments, both simulated and real; quantitative metrics with statistical analysis; ablation studies; computational cost reporting
- Reviewers will specifically look for: Evidence that improvements are not just from more compute or data

## 12.4 Common Rejection Reasons

1. "Only tested in simulation" — always include at least some real robot results or clearly justify simulation-only
2. "No comparison to PaLM-E or similar baselines" — must position against the state of the art
3. "Incremental improvement" — need a clear, novel technical contribution beyond engineering
4. "Missing ablations" — every claimed contribution must have an ablation
5. "Unreproducible" — if using proprietary models/data, provide as much reproducibility as possible
6. "No analysis of failure modes" — reviewers want to know when/why the method fails

## 12.5 Increment Needed for Acceptance

- For a top venue: Need either (a) a significantly new capability or (b) a strong efficiency improvement (same performance at 10x fewer params)
- Simply reproducing PaLM-E on different models is insufficient — need a novel angle
- Cross-domain transfer to a NEW domain (medical, agricultural, industrial) with domain-specific innovations can be sufficient

---

# 13. Researcher Quick Reference Tables

## 13.1 Key Terminology Table

| Term | Definition in This Paper's Context |
|---|---|
| Multi-modal sentence | A sequence interleaving text token embeddings and continuous observation embeddings as input to the LLM |
| Grounding | Connecting LLM representations to real-world sensory data (images, states) |
| Embodied language model | An LLM augmented with continuous sensor inputs to reason about the physical world |
| Entity referral | A mechanism to label object representations with unique tokens so PaLM-E can reference specific objects |
| Full mixture | The complete training dataset mixing internet VL data (91.1%) with robot data (8.9%) |
| Prefix/Prompt | The input multi-modal sentence conditioning the autoregressive generation |
| Affordance prediction | Predicting whether a specific skill/action is feasible in the current environment state |
| Failure detection | Determining whether a previously attempted action succeeded or failed |
| Catastrophic forgetting | Loss of previously learned capabilities (language) when training on new data (multimodal) |
| Token Learner (TL) | An architecture that learns to aggregate ViT tokens into fewer, more informative tokens |

## 13.2 Important Equations Summary

| Equation | Purpose | Key Idea |
|---|---|---|
| $p(w_{1:L}) = \prod_l p_{LM}(w_l \mid w_{1:l-1})$ | Autoregressive LM probability | Each token depends on all previous tokens |
| $x_i = \gamma(w_i)$ or $\phi_j(O_j)$ | Multi-modal embedding | Text and observations map to the same embedding space |
| $\phi: \mathcal{O} \rightarrow \mathcal{X}^q$ | Encoder definition | Maps observation to $q$ embedding vectors |
| $x_i = \psi(\tilde{\phi}_{ViT}(I)_i)$ | ViT projection | Affine transform aligns ViT dimension to LLM dimension |
| Cross-entropy loss on $w_{n+1:L}$ | Training objective | Only predict target text, not the multi-modal prefix |

## 13.3 Parameter Meaning Table

| Parameter/Config | Value | Significance |
|---|---|---|
| PaLM model sizes | 8B, 62B, 540B | Larger = better transfer, less forgetting |
| ViT model sizes | 4B (ViT-4B), 22B (ViT-22B) | Larger = richer visual features |
| Vocabulary size ($\|W\|$) | 256,000 | PaLM's token vocabulary |
| Embedding dimension ($k$) | Varies by model | Dimension of the shared embedding space |
| OSRT slots ($q$) | Multiple per object | Each object encoded as multiple embedding vectors |
| Robot data ratio | 8.9% of full mixture | Small fraction yet critical for embodied tasks |
| TAMP 1% data | 320 examples per planning task | Extreme few-shot regime for testing transfer |
| Language-Table demos | 10–80 per task | Few-shot real-world evaluation |
| Mobile manipulation sequences | 2,912 | Training data for kitchen robot tasks |

## 13.4 Algorithm Flow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PRE-TRAINING PHASE                                       │
│    • PaLM: trained on internet text (trillions of tokens)   │
│    • ViT: trained on image classification (billions of imgs) │
└─────────────┬───────────────────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. MULTI-MODAL TRAINING PHASE                               │
│    • Connect ViT/OSRT/MLP encoders to PaLM via projectors   │
│    • Train on full mixture (VL data + robot data)            │
│    • Optionally freeze LLM, train only encoders+projectors  │
└─────────────┬───────────────────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. INFERENCE (ROBOT PLANNING)                               │
│    • Receive instruction + camera image                      │
│    • Encode image → embeddings via ViT                       │
│    • Construct multi-modal sentence                          │
│    • PaLM-E generates next plan step (text)                  │
│    • Low-level policy (RT-1) executes step                   │
│    • Loop: new observation → re-plan → execute               │
└─────────────────────────────────────────────────────────────┘
```

---

# 14. One-Page Master Summary Card

## Problem
LLMs cannot perceive or reason about the physical world — they lack visual and sensory grounding, making them unable to plan robot actions from raw observations.

## Key Idea
Inject continuous sensor observations (images, states, 3D representations) directly into the token embedding space of a pre-trained LLM, creating "multi-modal sentences" that the LLM processes with its existing transformer architecture.

## Method
- PaLM-E = Pre-trained PaLM (LLM) + Pre-trained ViT (vision encoder) + learned projection layers
- Input encoders (ViT, MLP, OSRT) convert observations into embedding vectors in the LLM's space
- Multi-modal sentences interleave text and observation embeddings
- Trained end-to-end on a mixture of internet VL data (91%) + robot data (9%)
- Closed-loop robot control: PaLM-E generates text plans → low-level policies execute → new observations → re-plan

## Results
- Positive transfer from VL data to robotics across 3 robot domains (2x+ improvement with full mixture)
- OSRT provides most data-efficient representations for structured robot tasks
- PaLM-E-562B achieves SOTA on OK-VQA as a generalist model
- Catastrophic forgetting decreases dramatically with scale (87% loss at 12B → 3.8% at 562B)
- Emergent capabilities: zero-shot multimodal CoT, multi-image reasoning, OCR-free math

## Main Weaknesses
- Requires 562B parameters for best results (impractical for deployment)
- Depends on separate low-level policies (not truly end-to-end)
- Small models suffer severe catastrophic forgetting
- Real robot evaluations are limited and mostly qualitative

## Research Opportunity
Build efficient embodied multimodal models using knowledge distillation, adapter-based finetuning, or action token prediction to achieve PaLM-E-level performance at deployable scale. Alternatively, explore novel scene representations, safety-aware planning, or cross-domain transfer to new robot application areas.

## Publishable Extension Ideas
1. Adapter/LoRA-based multimodal training to prevent forgetting at small scale
2. End-to-end embodied model predicting both text plans AND action tokens
3. Open-source reproduction of PaLM-E recipe on LLaMA + open ViT
4. Novel domain application (medical robotics, warehouse automation) with domain-specific scene representations
5. Integration of world models for look-ahead planning in embodied LLMs

---
