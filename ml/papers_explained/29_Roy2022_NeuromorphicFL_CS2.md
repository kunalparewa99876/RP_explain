# 29 - Federated Neuromorphic Learning of Spiking Neural Networks for Low-Power Edge Intelligence

**Authors:** Nicolas Skatchkovsky, Hyeryung Jang, Osvaldo Simeone  
**Affiliation:** KCLIP Lab, CTR, Department of Engineering, King's College London, UK  
**Venue:** ICASSP 2020 (IEEE International Conference on Acoustics, Speech and Signal Processing)  
**Funding:** European Research Council (ERC), Horizon 2020 (Grant No. 725731)

---

## Paper Type Classification

**Primary:** Algorithmic / Method  
**Secondary:** Systems / Engineering

**Adaptation Strategy:**
- Provide workflow logic and pseudocode intuition for the FL-SNN algorithm
- Focus on design decisions, baselines, and metrics for the experimental component
- Explain intuition behind the probabilistic SNN formulation before diving into equations

---

# 0. Quick Paper Identity Card

| Attribute | Detail |
|---|---|
| **Problem Domain** | Federated Learning + Neuromorphic Computing (Spiking Neural Networks on edge devices) |
| **Paper Type** | Algorithmic / Method with experimental validation |
| **Core Contribution** | First federated learning protocol specifically designed for on-device spiking neural networks (FL-SNN) |
| **Key Idea** | Train SNNs collaboratively across edge devices using local bio-inspired learning signals (not backpropagation) plus global parameter averaging through a base station |
| **Required Background** | Basics of Federated Learning (FedAvg), neural network training, basic probability, understanding of what biological neurons do |
| **Primary Baseline** | Separate (isolated) training of individual SNNs without any cooperation |
| **Main Innovation Type** | Novel algorithm combining two paradigms: biologically plausible SNN training + FL communication protocol |
| **Difficulty Level** | Moderate-High (probabilistic SNN model is mathematically involved; FL protocol itself is relatively straightforward) |
| **Reproducibility Level** | Medium (MNIST-DVS dataset is public; SNN simulation details are provided but neuromorphic hardware is specialized) |

---

# 1. Research Context & Core Problem

## 1.1 Exact Problem Formulation

- Training powerful AI models (standard deep neural networks) demands massive computational resources: large GPU clusters, huge energy consumption, days or weeks of training time.
- Edge devices (phones, IoT sensors, wearables) have very limited computation, memory, and energy budgets. They cannot run standard deep learning training.
- Even if a low-power model could be trained on a single edge device, each device only has access to a small, potentially biased local dataset. Training on limited local data alone produces poor models.
- **The central question:** How can multiple resource-constrained edge devices collaboratively train a brain-inspired (spiking) neural network model without sharing their private data, while keeping energy and communication costs very low?

## 1.2 Why the Problem Exists

- **Energy wall:** Standard ANNs use floating-point multiplications that consume significant energy. Edge devices operating on batteries cannot afford this.
- **Data scarcity per device:** Each device sees only a small slice of the overall data distribution. For example, one sensor might only see certain types of events. Training locally leads to models that are biased and generalize poorly.
- **Privacy constraints:** Sharing raw data between devices is often unacceptable due to privacy regulations and security risks.
- **Communication bottleneck:** Even in federated learning, regularly transmitting full model parameters between devices and a server is expensive in terms of bandwidth and energy.

## 1.3 Historical / Theoretical Gap

- **Spiking Neural Networks (SNNs)** had been studied mainly as standalone on-device models. Researchers showed SNNs could learn with very low energy on specialized hardware (like Intel's Loihi chip), but always in isolation on a single device.
- **Federated Learning (FL)** had been developed and studied exclusively for conventional ANNs (standard deep networks using backpropagation). FedAvg and its variants assumed gradient-based backpropagation updates.
- **No one had combined these two worlds.** There was no protocol for making SNNs on multiple devices train collaboratively through federated learning. This paper fills that gap for the first time.

## 1.4 Limitations of Previous Approaches

| Previous Approach | Limitation |
|---|---|
| Standard ANN training | Too energy-hungry for edge devices; requires GPU clusters |
| Single-device SNN training | Limited by local data availability; each device learns a biased model |
| Conventional FL (FedAvg) | Designed for ANN backpropagation-based models; does not account for SNN's temporal spike-based dynamics |
| SNN learning rules (STDP, INST/FILT) | Only defined for individual devices; no framework for cooperative training across devices |

## 1.5 Contribution Category

- **Algorithmic:** A new federated learning algorithm (FL-SNN) tailored for spiking neural networks
- **System Design:** An edge computing architecture where mobile devices with on-device SNNs cooperate through a base station
- **Empirical Insight:** Demonstrating that selectively exchanging subsets of synaptic weights offers a useful communication-accuracy trade-off

### Why This Paper Matters

- It opens an entirely new research intersection: neuromorphic computing meets federated learning. Before this work, these were separate communities.
- It provides a practical path toward ultra-low-power collaborative AI on edge devices (think swarms of IoT sensors, medical wearables, autonomous drones).
- The biological plausibility of the learning rule (no backpropagation needed) means the approach is compatible with emerging neuromorphic hardware that physically cannot do backpropagation.
- The selective weight exchange mechanism addresses one of FL's biggest practical problems (communication overhead) in a way that is naturally suited to SNN's sparse representation.

### Remaining Open Problems

1. **Scalability:** Only 2 devices and 2 classes tested. What happens with hundreds of devices and complex multi-class tasks?
2. **Non-IID robustness:** The extreme non-IID setup (each device has only one class) was tested, but intermediate non-IID distributions and their effects on convergence are unexplored.
3. **Convergence guarantees:** No theoretical convergence analysis is provided for FL-SNN.
4. **Hardware validation:** All experiments are in simulation. Real neuromorphic hardware deployment (e.g., on Loihi chips) with wireless communication is not demonstrated.
5. **Security:** No analysis of adversarial attacks (model poisoning, Byzantine faults) in the neuromorphic FL setting.
6. **Complex datasets:** Only MNIST-DVS (a relatively simple benchmark) was used.

---

# 2. Minimum Background Concepts

## 2.1 Spiking Neural Networks (SNNs)

- **Plain definition:** Unlike standard neural networks where neurons pass continuous real numbers to each other, spiking neural networks communicate using discrete "spikes" -- brief electrical pulses -- spread over time. Think of it like Morse code versus continuous speech.
- **Role inside paper:** SNNs are the core model deployed on each edge device. The entire learning framework is built around how SNNs generate and learn from these spike patterns.
- **Why authors needed it:** SNNs can be implemented on neuromorphic hardware (like Intel Loihi) that consumes only picojoules per spike -- orders of magnitude less energy than GPUs running standard neural networks. This makes them ideal for battery-powered edge devices.

## 2.2 Federated Learning (FL) / FedAvg

- **Plain definition:** A distributed training approach where multiple devices each train a local copy of the same model on their own private data, then periodically send their model parameters (not data) to a central server. The server averages the parameters and sends the averaged model back to all devices. This cycle repeats.
- **Role inside paper:** FL provides the cooperative training framework that allows multiple SNN-equipped devices to benefit from each other's data without actually sharing that data.
- **Why authors needed it:** A single edge device has too little data to train an effective model. FL solves this by enabling collaboration while preserving privacy.

## 2.3 Membrane Potential

- **Plain definition:** A running tally inside each spiking neuron that accumulates incoming signals. When this tally gets high enough, the neuron fires a spike. It is analogous to voltage building up in a biological neuron.
- **Role inside paper:** The membrane potential is the key variable that determines whether a neuron spikes at each timestep. The entire probabilistic model of the SNN revolves around it.
- **Why authors needed it:** The learning rule depends on computing how the probability of spiking (which depends on the membrane potential) changes when synaptic weights change.

## 2.4 Synaptic Weights

- **Plain definition:** Numbers that control how strongly one neuron's spikes influence another neuron's membrane potential. Larger weight means a stronger connection. These are the "learnable parameters" of the SNN, analogous to weights in a standard neural network.
- **Role inside paper:** Synaptic weights are what gets updated during local training and what gets exchanged during FL communication rounds.
- **Why authors needed it:** Learning = adjusting synaptic weights so the network produces desired output spikes for given input spikes.

## 2.5 Generalized Linear Model (GLM) for Neurons

- **Plain definition:** A probabilistic framework that says: "given the current membrane potential, the probability that a neuron fires a spike is given by a sigmoid function applied to that potential." This turns the deterministic spike-or-no-spike decision into a probabilistic one.
- **Role inside paper:** The GLM formulation allows the authors to define a proper loss function (negative log-likelihood) and compute gradients for learning, even in the spiking regime.
- **Why authors needed it:** Without a probabilistic model, it is very hard to define a differentiable training objective for SNNs because spikes are discrete binary events.

## 2.6 Eligibility Trace

- **Plain definition:** A running average of how much each synaptic weight contributed to the neuron's recent spiking behavior. Think of it as a "credit assignment memory" that tracks which weight adjustments would most improve performance, accumulated over recent timesteps.
- **Role inside paper:** The eligibility trace is one of two key components of the local SNN learning rule. It provides the direction for weight updates at hidden neurons.
- **Why authors needed it:** Since SNNs process data over time (unlike feed-forward ANNs that process one sample at a time), the learning rule needs to track contributions across multiple timesteps.

## 2.7 Learning Signal

- **Plain definition:** A scalar score that tells hidden neurons whether the overall network is performing well or poorly at the current moment. If the learning signal is positive, hidden neurons should reinforce their current behavior; if negative, they should change.
- **Role inside paper:** The learning signal acts as a substitute for backpropagation. Instead of propagating error gradients backward through layers, a single global signal is broadcast to all hidden neurons.
- **Why authors needed it:** Backpropagation is biologically implausible and hard to implement on neuromorphic hardware. The learning signal provides a simpler, hardware-friendly alternative.

## 2.8 DVS Camera (Dynamic Vision Sensor)

- **Plain definition:** A special camera where each pixel independently reports brightness changes (not absolute brightness values). It outputs a stream of "events" over time, naturally producing spike-like data that is ideal for SNNs.
- **Role inside paper:** The MNIST-DVS dataset used in experiments was captured by a DVS camera, making it a natural fit for demonstrating SNN capabilities.
- **Why authors needed it:** To demonstrate the system on time-encoded data that represents a realistic use case for neuromorphic sensors on edge devices.

## 2.9 Communication Period (tau)

- **Plain definition:** The number of local training iterations that happen between successive FL synchronization rounds. A larger tau means devices train longer in isolation before sharing parameters, reducing communication but potentially hurting convergence.
- **Role inside paper:** tau is the primary knob for controlling the communication-accuracy trade-off in FL-SNN.
- **Why authors needed it:** Finding the right tau is critical because communication is expensive on edge devices, but too little communication causes the local models to diverge.

---

# 3. Mathematical / Theoretical Understanding Layer

## 3.1 Local Empirical Loss (Equation 1)

**Intuition:** Each device has its own dataset. The local loss measures how badly the model performs on that device's data, averaged over all samples.

**What it solves:** Provides a per-device measure of model quality.

| Variable | Meaning |
|---|---|
| F^(i)(theta) | Average loss on device i's data |
| D^(i) | Local dataset at device i |
| \|D^(i)\| | Number of data points at device i |
| f(theta, z) | Loss for a single data point z given model parameters theta |
| theta | All learnable parameters of the SNN model |

**Assumptions:** All devices share the same model architecture (same theta structure).

**Practical interpretation:** If this number is high, the model is doing poorly on device i's data.

## 3.2 Global Learning Objective (Equation 2)

**Intuition:** The global goal is to find one set of parameters that works well across ALL devices, weighted by how much data each device contributes.

**What it solves:** Defines the optimization target for federated learning -- a single model that balances performance across all devices.

| Variable | Meaning |
|---|---|
| F(theta) | Weighted average loss across all N devices |
| N | Total number of participating devices |
| \|D^(i)\| | Dataset size at device i (used as weight) |

**Assumptions:** A single shared model can adequately serve all devices. Weighted average by dataset size is fair.

**Limitation:** In highly heterogeneous settings, a single global model may not be optimal for any individual device (the classic personalization problem in FL).

## 3.3 Local SGD Update (Equation 3)

**Intuition:** Each device nudges its local copy of the model parameters in the direction that reduces its local loss, based on a single randomly chosen data sample.

**What it solves:** Performs one step of local learning at each device.

| Variable | Meaning |
|---|---|
| theta^(i)(t) | Updated parameters at device i after iteration t |
| theta_tilde^(i)(t-1) | Starting parameters at device i for this iteration (either from last local step or from global averaging) |
| alpha | Learning rate -- how big each update step is |
| z^(i)(t) | Randomly sampled data point from device i's dataset |
| nabla f | Gradient of the loss -- the direction of steepest increase (we go opposite) |

**Assumptions:** Learning rate alpha is fixed and the same across all devices.

## 3.4 Global Averaging (Equation 4)

**Intuition:** Every tau iterations, all devices send their parameters to the base station, which computes a weighted average and broadcasts it back. This pulls all devices' models back toward consensus.

**What it solves:** Prevents local models from drifting too far apart; enables cooperative learning.

| Variable | Meaning |
|---|---|
| theta(t) | Globally averaged parameter at time t |
| theta^(i)(t) | Local parameter at device i at time t (before averaging) |
| \|D^(i)\| | Weight for device i (proportional to data size) |

**Limitation:** Simple averaging may not be optimal when devices have very different data distributions.

## 3.5 Spiking Probability (Equation 5)

**Intuition:** Given a neuron's current membrane potential (how "charged up" it is), the probability of it firing a spike follows a sigmoid curve. High membrane potential means high chance of spiking; low potential means low chance.

**What it solves:** Makes the spiking decision probabilistic rather than deterministic, enabling gradient-based optimization.

| Variable | Meaning |
|---|---|
| p(o_n(s) = 1 \| u_n(s)) | Probability that neuron n spikes at time s |
| u_n(s) | Membrane potential of neuron n at time s |
| sigma(.) | Sigmoid function: maps any real number to (0,1) |

**Assumptions:** Spiking neurons follow a Generalized Linear Model. Conditional independence: given the membrane potential, the spike is independent of everything else.

**Practical interpretation:** This is the "activation function" of the SNN world. Just as a ReLU or sigmoid in a standard ANN determines a neuron's output, the sigmoid of the membrane potential determines spike probability.

## 3.6 Membrane Potential Dynamics (Equation 6)

**Intuition:** A neuron's membrane potential at any moment is the sum of three things: (1) filtered incoming spikes from connected neurons, (2) the neuron's own recent spiking history (self-feedback), and (3) a fixed bias term.

**What it solves:** Defines how signals propagate through the network over time.

| Variable | Meaning |
|---|---|
| u_n(s) | Membrane potential of neuron n at time s |
| P_n | Set of neurons that send connections to neuron n |
| o_vec_n,k(s-1) | Filtered spike trace from pre-synaptic neuron k (convolution of spikes with synaptic filter) |
| w_n * o_back_n(s-1) | Self-feedback: neuron's own past spikes filtered and weighted |
| gamma_n | Bias parameter for neuron n |
| a_n,k(s) | Synaptic impulse response filter from neuron k to neuron n |
| w^l_n,k | Learnable synaptic weight (l-th basis component from k to n) |
| K_a | Number of basis functions used to parameterize each synaptic filter |
| a_l(s) | l-th fixed basis function (raised cosine) |
| b(s) | Fixed feedback impulse response |

**Assumptions:** Synaptic filters can be well-approximated by a linear combination of K_a fixed basis functions. Feedback is captured by a single learnable weight multiplied by a fixed filter.

**Practical interpretation:** Think of each synapse as having a "filter" that shapes how incoming spikes influence the receiving neuron over time. Instead of learning the entire filter shape (which would have too many parameters), the filter is expressed as a weighted sum of K_a pre-chosen basic shapes (raised cosines). Learning only adjusts the weights in this combination.

## 3.7 Log-Loss for SNN (Equation 7)

**Intuition:** For a given input-output pair encoded as spike trains, this quantity measures how unlikely the observed output is under the current model. Lower log-loss means the model assigns higher probability to the correct output.

**What it solves:** Provides the training objective for each SNN -- the quantity we want to minimize.

| Variable | Meaning |
|---|---|
| L_y\|x(theta) | Negative log-likelihood of output y given input x |
| h | Latent spike signals of hidden neurons (summed over all possibilities) |
| S | Total number of local timesteps |
| o_n(s) | Output of neuron n at time s |
| u_n(s) | Membrane potential of neuron n at time s |

**Limitation:** Summing over all possible hidden spike configurations (the marginalization over h) is intractable for large networks. In practice, this is approximated.

## 3.8 FL-SNN Local Update Rule (Equation 10)

**Intuition:** The weight update rule works differently for output neurons versus hidden neurons:
- **Output neurons:** Updated directly using the eligibility trace (because the desired output is known).
- **Hidden neurons:** Updated using the product of the learning signal and the eligibility trace (because the desired behavior of hidden neurons is unknown; the learning signal provides indirect supervision).

**What it solves:** Performs biologically plausible local learning at each device without backpropagation.

| Variable | Meaning |
|---|---|
| theta^(i)_n(t) | Updated parameters for neuron n at device i |
| alpha | Learning rate |
| e^(i)_n(t) | Eligibility trace for neuron n at device i |
| l^(i)(t) | Learning signal at device i |

**Key insight:** This is a three-factor learning rule for hidden neurons: weight change = learning rate x learning signal x eligibility trace. This is consistent with observed biological synaptic plasticity mechanisms.

## 3.9 Learning Signal (Equation 11)

**Intuition:** The learning signal is an exponentially weighted running average of how well the observable neurons (input and output) are behaving. It accumulates the log-probability of the observed spiking behavior at input and output neurons. If these neurons are spiking in ways that the model predicts well (high probability), the signal is high (positive reinforcement); otherwise, it is low.

| Variable | Meaning |
|---|---|
| l^(i)(t) | Learning signal at device i at global time t |
| kappa | Decay factor (0 < kappa < 1) controlling how much past history is retained |
| [t] | Set of local timesteps corresponding to global iteration t |
| log p(o_n(s)\|u_n(s)) | Log-probability of the observed spike at input/output neuron n |

## 3.10 Eligibility Trace (Equation 12)

**Intuition:** The eligibility trace is an exponentially weighted running average of the gradients of the log-spiking-probability with respect to the parameters. It tracks "which parameter adjustments would increase the probability of the observed spiking pattern."

| Variable | Meaning |
|---|---|
| e^(i)_n(t) | Eligibility trace for neuron n at device i |
| kappa | Same decay factor as in the learning signal |
| nabla log p(o_n(s)\|u_n(s)) | Gradient of log-probability of spiking w.r.t. parameters |

### Mathematical Insight Box

**The key idea to remember:** FL-SNN replaces backpropagation (which requires global error signals to flow backward through layers) with a two-component local learning rule:
1. **Learning signal** (scalar, broadcast to all hidden neurons) -- tells "how well is the network doing overall right now?"
2. **Eligibility trace** (per-parameter, computed locally) -- tracks "which parameters contributed to recent behavior?"

The product of these two gives the update direction. This is fundamentally more hardware-friendly than backpropagation because each neuron only needs local information plus a single scalar broadcast, rather than layer-by-layer gradient propagation.

---

# 4. Proposed Method / Framework (FL-SNN)

## 4.1 Overall Pipeline

The FL-SNN system operates at the intersection of three layers:

```
Layer 1: EDGE DEVICES (N mobile devices)
   Each device has:
   - A local SNN (same architecture across devices)
   - A local private dataset D^(i)
   - Ability to run online SNN learning locally

Layer 2: WIRELESS COMMUNICATION
   - Devices communicate through a base station (BS)
   - Communication happens every tau global iterations
   - Only model parameters (not data) are transmitted

Layer 3: PARAMETER SERVER (at the base station)
   - Receives parameters from all devices
   - Computes weighted average
   - Broadcasts averaged parameters back to all devices
```

## 4.2 Step-by-Step Algorithm Walkthrough

### Step 1: Initialization
- All N devices initialize their SNN parameters to the same starting values theta_bar(0).
- Each device selects D examples randomly from its local dataset and encodes them as binary spike trains of total length S = D * S' timesteps.

**Why authors did this:** Identical initialization ensures all devices start from the same point, which is standard in FL. Random selection with replacement enables online learning.

**Weakness:** No strategy for how to choose D (number of examples per round). Fixed initialization may not be optimal.

**Research idea seed:** Investigate smart initialization strategies (e.g., transfer learning from a pre-trained SNN) to speed up convergence.

### Step 2: Local Spike Generation at Hidden Neurons
- At each local algorithmic time s, every hidden neuron n in H generates a spike with probability sigma(u_n(s)), where u_n(s) is the membrane potential computed from incoming filtered spikes and self-feedback (Equation 6).
- This is a stochastic process: the neuron randomly spikes or not, with probability determined by the sigmoid of its membrane potential.

**Why authors did this:** The probabilistic GLM model enables proper likelihood-based learning. Stochastic spiking is also more biologically realistic.

**Weakness:** Stochastic spiking introduces variance in the learning process. More hidden neurons = more randomness = potentially slower convergence.

**Research idea seed:** Explore variance reduction techniques adapted from reinforcement learning (e.g., control variates) to stabilize SNN training.

### Step 3: Compute Learning Signal
- Every delta_s local timesteps (i.e., at each global iteration t), each device computes a learning signal l^(i)(t).
- The learning signal is an exponentially decaying running average of the log-probabilities of observed spikes at input and output neurons.
- It tells hidden neurons: "Overall, the network is doing well (positive) or poorly (negative) at matching the desired output."

**Why authors did this:** Hidden neurons do not have direct access to the desired output. The learning signal is a compact scalar summary of the network's performance that can be broadcast to all hidden neurons without requiring backpropagation.

**Weakness:** A single scalar signal for all hidden neurons is a very coarse feedback mechanism. Different hidden neurons may be contributing very differently to performance.

**Research idea seed:** Develop layer-specific or neuron-group-specific learning signals to provide richer feedback without losing biological plausibility.

### Step 4: Compute Eligibility Trace and Update Weights
- Each neuron n (hidden or output) computes its eligibility trace e_n(t): an exponentially decaying running average of the gradient of the log-spiking-probability with respect to its parameters.
- **For output neurons:** weight update = -alpha * e_n(t) (direct gradient, since desired output is known)
- **For hidden neurons:** weight update = -alpha * l(t) * e_n(t) (modulated by the learning signal)

**Why authors did this:** The three-factor rule (learning rate x learning signal x eligibility trace) is the simplest biologically plausible learning mechanism that can handle hidden neurons. It has connections to the REINFORCE algorithm in reinforcement learning.

**Weakness:** The learning signal acts as a multiplicative gating factor. If l(t) is close to zero, hidden neurons barely learn regardless of their eligibility trace. If l(t) is noisy, it introduces instability.

**Research idea seed:** Investigate adaptive learning rates per neuron or per layer to compensate for learning signal variance.

### Step 5: Global Averaging (Every tau Iterations)
- When t is a multiple of tau, every device sends its current parameters theta^(i)(t) to the base station.
- The BS computes the weighted average theta(t) = weighted_mean(theta^(i)(t)), weighted by dataset sizes.
- The averaged parameter theta(t) is broadcast back to all devices, which reset their local parameters to this global average.
- When t is NOT a multiple of tau, devices simply continue with their local parameters (no communication).

**Why authors did this:** Periodic averaging is the standard FedAvg approach. It balances communication cost (fewer rounds when tau is large) against model consistency.

**Weakness:** Simple averaging may destroy locally learned specialized features. When data is highly non-IID, averaging can hurt performance.

**Research idea seed:** Replace simple averaging with more sophisticated aggregation methods (e.g., attention-weighted aggregation, or Bayesian aggregation that respects uncertainty).

### Step 6: Selective Weight Exchange (Communication Reduction)
- Instead of exchanging ALL synaptic weights, each device sends only K'_a out of K_a weights per synapse -- specifically, the K'_a weights with the largest gradients.
- The communication rate is defined as r = K'_a / tau (number of weights exchanged per global iteration).
- The BS averages weights that both devices sent; keeps weights sent by only one device; and sets un-sent weights to zero.

**Why authors did this:** Transmitting all parameters at every synchronization round is too expensive for bandwidth-limited edge networks. Sending only the most "important" weights (largest gradient magnitude) is an efficient approximation.

**Weakness:** Setting unsent weights to zero at the BS is aggressive and may discard useful information. The "largest gradient" selection heuristic may not always identify the most important weights.

**Research idea seed:** Explore more sophisticated sparsification strategies -- random selection, top-K by magnitude (not gradient), or learned sparsity masks. Also investigate error-feedback mechanisms where unsent gradients are accumulated and sent later.

## 4.3 Simplified Pseudocode

```
INITIALIZE all devices with same parameters theta_0

FOR each global iteration t = 1 to T:
    FOR each device i = 1 to N:
        
        # --- LOCAL COMPUTATION ---
        For each local timestep s in interval [t]:
            Each hidden neuron probabilistically spikes based on membrane potential
        
        Compute LEARNING SIGNAL: running average of log-probability at observable neurons
        
        For each neuron n in (hidden union output):
            Compute ELIGIBILITY TRACE: running average of gradient of log-probability
            
            IF n is output neuron:
                Update weight using eligibility trace only
            IF n is hidden neuron:
                Update weight using (learning signal) x (eligibility trace)
        
        # --- GLOBAL COMMUNICATION ---
        IF t is multiple of tau:
            Send local parameters (or subset) to base station
            Receive globally averaged parameters from base station
            Replace local parameters with global average
        ELSE:
            Keep local parameters unchanged
```

## 4.4 Time Scale Relationship

A unique aspect of FL-SNN is the multi-scale temporal structure:

| Time Scale | Symbol | Description |
|---|---|---|
| Local algorithmic time | s = 1, ..., S | Individual SNN timesteps; one spike decision per neuron per step |
| Global iteration | t = 1, ..., T | Spans delta_s local timesteps; one parameter update per global iteration |
| Communication round | tau | One FL synchronization every tau global iterations |
| Training epoch | D examples | Total S = D * S' local timesteps to process D examples |

**Key relationship:** S = D * S' = T * delta_s

This means the number of training examples D and the number of parameter updates T are decoupled (unlike conventional FL where one example = one update). This gives FL-SNN additional flexibility in balancing computation and communication.

---

# 5. Experimental Setup / Evaluation Design

## 5.1 Dataset Characteristics

| Property | Value |
|---|---|
| Dataset | MNIST-DVS |
| Data source | DVS camera recording of MNIST digits displayed on an LCD monitor |
| Original resolution | 128 x 128 pixels |
| Cropped resolution | 26 x 26 pixels (active region only) |
| Temporal duration per sample | 2 seconds, downsampled to S' = 80 timesteps |
| Training set | 900 examples per class |
| Test set | 100 examples per class |
| Data encoding | Event-based: +1 (brightness increase), -1 (brightness decrease), 0 (no change) |
| Classes used | '1' and '7' (binary classification) |

## 5.2 Experimental Protocol

- **Number of devices:** N = 2
- **Data partition:** Extremely non-IID: Device 1 has ONLY class '1', Device 2 has ONLY class '7'. This is the hardest possible partition for 2 classes.
- **Training examples per device:** D = 400 randomly selected (with replacement) from local data
- **Total local timesteps:** S = 400 * 80 = 32,000
- **SNN architecture:** Input layer (26x26 = 676 neurons) -> Hidden layer (N_H neurons, varied: 0 or 16) -> Output layer (2 neurons)
- **Connectivity:** All input neurons connect to all hidden and output neurons. Hidden and output neurons are fully connected to each other.

## 5.3 Metrics Used and Why

| Metric | Why Used |
|---|---|
| Normalized mean test loss | Measures how much FL improves over the separate training baseline. Normalization by baseline loss allows direct comparison of the improvement factor. |
| Final test accuracy | Standard classification performance metric. Directly interpretable for practical applications. |
| Standard deviation over trials | Captures variability due to stochastic spiking and random data sampling. Essential for assessing reliability. |

## 5.4 Baseline Selection Logic

- **Primary baseline:** Separate (isolated) training of each SNN with no communication (tau = infinity). This represents what happens without FL.
- **Why this baseline:** The most natural comparison to show that FL adds value. If FL-SNN does not beat separate training, there is no point in the communication overhead.
- **Missing baselines:** No comparison with conventional ANN + FL (e.g., FedAvg with a standard neural network of similar size). No comparison with centralized SNN training (upper bound).

## 5.5 Hyperparameter Choices

| Parameter | Value | Reasoning |
|---|---|---|
| alpha (learning rate) | 0.05 | Manual hyperparameter search |
| kappa (decay factor) | 0.2 | Manual hyperparameter search |
| delta_s (local steps per global iteration) | 5 | Manual hyperparameter search |
| K_a (number of synaptic basis functions) | 8 | Raised cosine functions with 10-timestep synaptic duration; biologically plausible filters |
| tau (communication period) | Varied: 8, 64, 400, infinity | Experimental variable to study communication-accuracy trade-off |
| K'_a (weights exchanged) | Varied: 1, 2, 4, 8 | Experimental variable for selective exchange study |

## 5.6 Hardware / Compute Assumptions

- All experiments are simulations (not on neuromorphic hardware).
- The paper does not report compute requirements for the simulations.
- The implicit assumption is that each device has a neuromorphic chip capable of running the SNN at the stated network size. Real-world deployment would target chips like Intel Loihi (mentioned in the introduction).

### Experimental Reliability Analysis

**What is trustworthy:**
- The clear advantage of FL over separate training (Fig. 3) is consistent across multiple trials with reported standard deviations.
- The trend that smaller tau (more frequent communication) leads to better performance is expected and validates the setup.
- The selective weight exchange experiment (Fig. 4) shows consistent trends across different rates.

**What is questionable:**
- Only 2 devices and 2 classes is an extremely simple setup. Real-world FL involves tens to thousands of devices with many classes.
- Only 3 trials for the loss experiment and 10 trials for the accuracy experiment. Statistical significance is limited.
- No comparison with any ANN-based FL method, making it hard to assess whether the SNN approach is competitive or just "better than nothing."
- The non-IID partition (each device has exactly one class) is extreme and somewhat artificial. More realistic data heterogeneity was not tested.
- No convergence analysis -- we do not know if the algorithm has converged by the end of training.

---

# 6. Results & Findings Interpretation

## 6.1 Main Outcomes

### Experiment 1: Effect of Communication Period (Fig. 3)

- **Setup:** No hidden neurons (N_H = 0), varying tau from 8 to infinity.
- **Key finding:** FL-SNN with tau = 8 (frequent communication) achieves normalized test loss below 1.0, meaning it outperforms separate training.
- **Increasing tau degrades performance:** tau = 64 achieves approximately 1.0 (matching separate training); tau = 400 and tau = infinity (no communication) perform worse, settling above 1.0.
- **Interpretation:** Frequent parameter sharing is essential for the cooperative advantage. When devices communicate rarely (large tau), their local models diverge too much and averaging becomes less effective.

### Experiment 2: Selective Weight Exchange (Fig. 4)

- **Setup:** N_H = 16 hidden neurons, fixed communication rates r = 1/8 and r = 1/4.
- **Key finding:** For a fixed communication budget (rate r), it is better to send fewer weights more frequently than to send more weights less frequently.
- **Example (r = 1/4):** Sending K'_a = 2 weights every tau = 8 iterations outperforms sending K'_a = 8 weights every tau = 32 iterations, even though the total amount of data transmitted is the same.
- **Best accuracy achieved:** Approximately 85-90% with r = 1/4 and small tau.
- **Worst accuracy:** Around 50% (random guessing for 2 classes) when tau is large with K'_a = 1.

## 6.2 Performance Trends

| Trend | Evidence | Explanation |
|---|---|---|
| More frequent communication helps | Fig. 3: tau=8 beats tau=64 beats tau=400 | Frequent averaging prevents model divergence in non-IID settings |
| Selective exchange works | Fig. 4: Good accuracy even with K'_a < K_a | Largest-gradient weights carry most information; other weights can be delayed |
| Frequent partial > infrequent full (at same rate) | Fig. 4: small tau + small K'_a beats large tau + large K'_a at same r | More synchronization points keep models aligned; the specific weights matter less than staying synchronized |
| Hidden neurons improve accuracy | Implied: N_H = 16 experiments achieve higher absolute accuracy | Hidden layer adds representational capacity |

## 6.3 Failure Cases

- **tau = 400 with N_H = 0:** Normalized loss above 1.05, meaning FL actually slightly hurts compared to separate training. Rare averaging with drifted parameters introduces harmful averaging.
- **K'_a = 1 with large tau:** Accuracy drops to approximately 50% (random chance), showing that too little communication makes FL ineffective.

## 6.4 Unexpected Observations

- The performance degradation from tau = 8 to tau = 64 is relatively small (both near 1.0), but the jump from tau = 64 to tau = 400 is dramatic. This suggests a critical threshold for communication frequency below which FL stops being beneficial.
- The standard deviations (shaded areas in Fig. 3) are relatively large, especially for large tau, indicating that the stochastic nature of SNN training combined with infrequent synchronization leads to highly variable outcomes.

## 6.5 Statistical Meaning

- Results are averaged over 3 trials (Fig. 3) or 10 trials (Fig. 4). The standard deviations shown are substantial, especially for intermediate tau values.
- The overlapping error bars in Fig. 4 suggest that some differences between configurations may not be statistically significant.
- No formal statistical tests (t-tests, confidence intervals) are reported.

### Publishability Strength Check

**Publication-grade results:**
- The clear demonstration that FL-SNN outperforms separate training for small tau is a solid proof-of-concept result.
- The selective weight exchange trade-off is a practical and interesting finding.
- The first-of-its-kind nature of combining FL with SNN makes even preliminary results publishable at ICASSP.

**Results needing stronger validation:**
- Scalability to more devices, more classes, and more complex datasets.
- Statistical rigor (more trials, confidence intervals, significance tests).
- Comparison with conventional ANN+FL baselines.
- Convergence analysis (theoretical or at least empirical convergence curves for more settings).
- Energy consumption measurements on actual neuromorphic hardware.

---

# 7. Strengths -- Weaknesses -- Assumptions

## 7.1 Technical Strengths

| # | Strength | Why It Matters |
|---|---|---|
| 1 | First-ever combination of FL and SNN | Opens an entirely new research direction at the intersection of two active fields |
| 2 | Biologically plausible learning rule (no backpropagation) | Compatible with neuromorphic hardware that cannot implement backpropagation |
| 3 | Online learning (processes data as streams over time) | Natural fit for real-time edge applications with continuous sensor data |
| 4 | Selective weight exchange mechanism | Addresses the critical communication bottleneck in FL, especially important for bandwidth-constrained edge devices |
| 5 | Flexible communication-accuracy trade-off via tau and K'_a | Allows system designers to tune the protocol for their specific bandwidth and latency constraints |
| 6 | Clean algorithmic formulation (Algorithm 1) | Makes the method reproducible and extensible |
| 7 | Demonstrates clear benefit of cooperation over isolation | Validates the core premise that FL helps even for non-standard models like SNNs |

## 7.2 Explicit Weaknesses

| # | Weakness | Impact |
|---|---|---|
| 1 | Only 2 devices and 2 classes tested | Cannot assess scalability; real FL involves many devices and classes |
| 2 | No theoretical convergence guarantees | Users cannot predict when or whether the algorithm will converge for new problems |
| 3 | Only one (simple) dataset: MNIST-DVS | Unclear how the method performs on harder, more varied datasets |
| 4 | No comparison with ANN+FL baselines | Cannot assess whether SNN+FL is competitive with conventional approaches in terms of accuracy |
| 5 | No real hardware validation | Energy savings are claimed based on neuromorphic hardware literature but not measured |
| 6 | Simple averaging may be suboptimal for non-IID data | More advanced aggregation strategies are not explored |
| 7 | Learning signal is a single scalar for all hidden neurons | Limits the richness of feedback and may slow learning |
| 8 | No privacy analysis | FL provides some privacy but model parameters can still leak information; no formal privacy guarantees |

## 7.3 Hidden Assumptions

| # | Assumption | Potential Issue |
|---|---|---|
| 1 | All devices have the same SNN architecture | In practice, different devices may have different hardware capabilities requiring different network sizes |
| 2 | Communication is reliable (no packet loss, delays) | Wireless communication is unreliable; message loss could disrupt the averaging scheme |
| 3 | Base station can handle all devices simultaneously | Scalability of the parameter server is not addressed |
| 4 | Identical initialization across all devices | May not always be possible in practice; different hardware may have different random seeds or pre-trained states |
| 5 | Data at each device is static throughout training | Real edge devices may receive new data continuously; the algorithm does not address concept drift |
| 6 | Synchronous communication: all devices communicate at the same time | Synchronous protocols require waiting for the slowest device (straggler problem) |
| 7 | Synaptic basis functions (raised cosines) are suitable for all data types | Different applications may benefit from different basis function choices |
| 8 | Setting unsent weights to zero at BS is acceptable | This discards potentially useful information and may bias the model |

---

# 8. Weakness to Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| Only 2 devices tested | Proof-of-concept paper with limited scope | Scale FL-SNN to 10-100+ devices and study convergence dynamics | Implement on cloud simulator with many virtual devices; study how aggregation quality changes with N |
| No convergence theory | Combining probabilistic SNN with FL creates a complex optimization landscape that is hard to analyze | Derive convergence bounds for FL-SNN under non-IID data | Extend FedAvg convergence proofs (Li et al. 2020) to account for SNN's stochastic spiking and online learning |
| Only MNIST-DVS tested | Limited dataset availability for DVS-based tasks; paper focused on algorithm design | Evaluate on DVS-Gesture, N-MNIST, CIFAR10-DVS, and non-vision temporal datasets (audio, EEG) | Benchmark FL-SNN on progressively harder DVS datasets and compare with ANN+FL |
| No ANN baseline comparison | Different paradigms are hard to compare fairly | Provide accuracy-energy-communication Pareto curves comparing FL-SNN vs. FL-ANN | Implement equivalent-parameter-count ANN with FedAvg and measure accuracy, simulated energy, and communication |
| Single scalar learning signal for all hidden neurons | Biological plausibility constraint; single global reward signal is simplest | Develop layered or neuron-group-specific learning signals | Use local error signals at each layer (e.g., local loss functions, target propagation) while maintaining hardware compatibility |
| No privacy guarantees | Paper focuses on algorithmic design, not privacy | Add differential privacy or secure aggregation to FL-SNN | Add calibrated noise to transmitted parameters (DP-SGD adapted for SNN); implement secure aggregation protocol |
| No asynchronous communication support | Synchronous protocol is simpler to analyze | Design asynchronous FL-SNN tolerant of stragglers | Use techniques from asynchronous FL (e.g., FedBuff) adapted for SNN's temporal dynamics |
| Unsent weights set to zero | Simple heuristic for handling incomplete parameter exchange | Design smarter aggregation for partial weight exchange | Use error feedback: accumulate unsent weight updates and add them to next communication round |
| No adversarial robustness | Beyond the scope of this proof-of-concept | Study Byzantine-resilient aggregation for FL-SNN | Apply robust aggregation methods (Krum, trimmed mean) and test against model poisoning attacks on SNN |
| No personalization | Single global model may not fit all devices | Add personalization layers or fine-tuning to FL-SNN | Keep some SNN layers local (not averaged) while federating others; per-device fine-tuning after FL |

---

# 9. Novel Contribution Extraction

## 9.1 Authors' Novel Claims (as stated in the paper)

1. "We propose FL-SNN, the first online federated learning protocol for cooperative training of on-device spiking neural networks."
2. "FL-SNN leverages local feedback signals (learning signal + eligibility trace) instead of backpropagation, combined with global feedback through periodic parameter averaging."
3. "We demonstrate that selective exchange of synaptic weights provides a flexible trade-off between communication load and accuracy."

## 9.2 Novel Claim Templates Inspired by This Paper

1. **"We propose [METHOD_NAME] that improves federated SNN training by [SPECIFIC MECHANISM], achieving [X]% higher accuracy than FL-SNN while reducing communication by [Y]%."**
   - Example: Replace simple averaging with attention-weighted aggregation based on each device's local loss.

2. **"We propose [METHOD_NAME] that extends neuromorphic federated learning to asynchronous settings by [MECHANISM], enabling scalability to [N] devices with heterogeneous compute capabilities."**
   - Example: Allow devices to communicate at their own pace, with the server maintaining a running weighted average.

3. **"We propose [METHOD_NAME] that provides differential privacy guarantees for federated SNN training by [MECHANISM], achieving formal (epsilon, delta)-differential privacy with only [X]% accuracy degradation."**
   - Example: Add calibrated Laplace noise to synaptic weights before transmission, with noise magnitude tuned to the SNN's sensitivity.

4. **"We propose [METHOD_NAME] that replaces FL-SNN's global scalar learning signal with [RICHER FEEDBACK MECHANISM], improving hidden neuron training efficiency by [X]% on complex tasks."**
   - Example: Use local target signals at each layer computed from the output error, propagated via fixed random feedback connections.

5. **"We propose [METHOD_NAME] that validates neuromorphic federated learning on real [HARDWARE_PLATFORM] hardware, demonstrating [X]x energy savings over conventional ANN+FL systems while maintaining competitive accuracy on [DATASET]."**
   - Example: Deploy FL-SNN on Intel Loihi 2 chips communicating via Bluetooth Low Energy, and measure actual energy consumption.

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Work

The paper's conclusion is very brief and does not explicitly list future directions. However, the following are implied:
- Scaling to more devices and more complex classification tasks
- Exploring more sophisticated communication strategies beyond simple selective exchange
- Implementing on actual neuromorphic hardware

## 10.2 Missing Directions Not Addressed by Authors

1. **Convergence theory:** Mathematical proof of convergence rate under non-IID data distributions.
2. **Privacy mechanisms:** Integration of differential privacy or secure multi-party computation.
3. **Heterogeneous architectures:** Allow different devices to have different SNN sizes and aggregate only shared sub-networks.
4. **Continual learning:** Handle streaming data where new classes or data distributions appear over time.
5. **Multi-task learning:** Each device learns a different but related task, with federated sharing of common representations.

## 10.3 Modern Extensions (Post-2020)

1. **Surrogate gradient methods for SNN:** Since 2020, surrogate gradient methods have become the dominant approach for training SNNs. Combining these with FL (replacing the three-factor rule) could yield better accuracy.
2. **FedAvg improvements:** Newer FL algorithms (FedProx, SCAFFOLD, FedNova) handle non-IID data better. Adapting these for SNN would be valuable.
3. **Neuromorphic hardware advances:** Intel Loihi 2, IBM NorthPole, and SpiNNaker 2 offer more powerful neuromorphic platforms. Implementing FL-SNN on these modern chips would be impactful.
4. **Split learning for SNN:** Instead of transmitting all parameters, transmit intermediate spike representations (split learning approach).
5. **Sparse and event-driven communication:** Leverage the inherent sparsity of spike signals to design more efficient communication protocols.

## 10.4 Cross-Domain Combinations

1. **Neuromorphic FL + autonomous driving:** DVS cameras on multiple vehicles could collaboratively train an SNN for pedestrian detection.
2. **Neuromorphic FL + healthcare wearables:** EEG or EMG sensors on patient wearables could collaboratively train seizure/gesture detection models.
3. **Neuromorphic FL + smart manufacturing:** Vibration sensors on factory equipment could collaboratively learn anomaly detection.
4. **Neuromorphic FL + environmental monitoring:** Acoustic sensors in wildlife monitoring networks could collaboratively classify animal calls.

## 10.5 LLM-Era Extensions

- **Knowledge distillation from LLMs to SNN:** Use a large language model or vision-language model to generate labels or embeddings that guide SNN training on edge devices, with FL aggregating the knowledge.
- **LLM-guided hyperparameter tuning for FL-SNN:** Use an LLM-based agent to automatically tune tau, K'_a, alpha, and other hyperparameters based on system conditions.

---

# 11. How to Write a NEW Paper From This Work

## 11.1 Reusable Elements

- **The overall framework:** FL + non-standard on-device models (not just SNNs -- could be binary neural networks, quantized networks, or other low-power models).
- **The experimental design:** Non-IID partition where each device has different classes is a clean way to demonstrate FL benefits.
- **The selective exchange idea:** Sending only the most important parameters is applicable to any federated learning scenario.
- **The multi-timescale structure:** Separating local computation time from global communication time is a useful modeling technique.
- **The evaluation methodology:** Normalizing by a baseline (separate training) to show relative improvement.

## 11.2 What MUST NOT Be Copied

- The specific FL-SNN algorithm (Algorithm 1) -- this is the authors' core contribution.
- The exact equations (5)-(12) -- these define the specific probabilistic SNN model and learning rule from [5].
- Figures, tables, or experimental data.
- Text or phrasing from the paper.

## 11.3 How to Design a Novel Extension

1. **Identify a specific weakness** from Section 8 (e.g., no convergence guarantees).
2. **Propose a concrete solution** (e.g., add proximal regularization term as in FedProx to prevent local model drift).
3. **Implement and test** on a broader experimental setup (more devices, more classes, harder datasets).
4. **Compare fairly** against FL-SNN as a baseline, using the same experimental protocol plus your additions.
5. **Frame as a contribution:** "We address the open problem of [weakness] in neuromorphic federated learning by proposing [solution], which achieves [result]."

## 11.4 Minimum Publishable Contribution Checklist

- [ ] Clearly identified limitation of FL-SNN (not just "we test on more datasets")
- [ ] Proposed a novel method or mechanism that addresses this limitation
- [ ] Theoretical justification or intuition for why the proposed method should work
- [ ] Experimental validation on at least 2-3 datasets, including at least one harder than MNIST-DVS
- [ ] Fair comparison against FL-SNN and at least one other relevant baseline
- [ ] Ablation study showing which component of the proposed method contributes most
- [ ] Analysis of communication cost vs. accuracy trade-off
- [ ] Discussion of limitations and future work for the proposed method

---

# 12. Complete Paper Writing Template

## 12.1 Abstract

**Purpose:** Concise summary (150-250 words) that covers problem, method, results, and significance.

**What to include:**
- One sentence on the problem (edge AI needs low-power collaborative learning)
- One sentence on the gap (existing FL-SNN has limitation X)
- One sentence on your method (we propose Y that addresses X by Z)
- One sentence on key results (achieves A% improvement on datasets B and C)
- One sentence on significance (enables practical deployment scenario D)

**Common mistakes:**
- Too vague ("we improve federated learning") -- be specific about what you improve and by how much
- Too long -- conferences typically limit to 200 words
- Including references or acronym definitions that are not self-contained

**Reviewer expectations:** Can I understand the contribution without reading the full paper? Is the improvement quantified?

## 12.2 Introduction

**Purpose:** Motivate the problem, establish context, and state contributions clearly.

**What to include:**
- Paragraph 1: Big picture problem (energy-efficient edge AI)
- Paragraph 2: SNN background and why they matter for edge devices
- Paragraph 3: FL background and why collaborative training is needed
- Paragraph 4: Existing work on FL-SNN and its specific limitation that you address
- Paragraph 5: Your contributions (bulleted list of 3-4 concrete contributions)

**Common mistakes:**
- Too much background, too little about your specific contribution
- Not clearly distinguishing your work from FL-SNN
- Overclaiming: "we solve the problem of..." (use "we address" or "we mitigate")

**Reviewer expectations:** By the end of the introduction, I should know exactly what is new and why it matters.

## 12.3 Related Work

**Purpose:** Position your work relative to existing literature and show awareness of the field.

**What to include:**
- Subsection on Federated Learning (FedAvg, FedProx, SCAFFOLD, communication-efficient FL)
- Subsection on Spiking Neural Networks (training methods, neuromorphic hardware)
- Subsection on Neuromorphic Federated Learning (FL-SNN and any subsequent work)
- Subsection on your specific technical area (e.g., convergence analysis, privacy, asynchronous FL -- whatever your contribution targets)
- Clear statement of how your work differs from each group

**Common mistakes:**
- Listing papers without explaining how they relate to YOUR work
- Missing recent (last 2 years) papers in the area
- Being dismissive of prior work rather than building on it

**Reviewer expectations:** Comprehensive coverage of relevant work, honest comparison, and a clear gap that your paper fills.

## 12.4 Method

**Purpose:** Fully describe your proposed method so that someone could reimplement it.

**What to include:**
- System model and assumptions
- Mathematical formulation of your optimization objective
- Detailed algorithm description (pseudocode)
- Explanation of each component and design choice
- How your method differs from FL-SNN (highlight modifications)
- Computational and communication complexity analysis

**Common mistakes:**
- Ambiguous notation (define all symbols in a table)
- Missing details that prevent reproduction (hyperparameter ranges, initialization)
- Not explaining WHY design choices were made, only WHAT they are

**Reviewer expectations:** I should be able to reimplement your method from this section alone. Every design choice should be justified.

## 12.5 Theoretical Analysis (if applicable)

**Purpose:** Provide formal guarantees about your method's properties.

**What to include:**
- Theorem statement with all assumptions explicitly listed
- Proof sketch (full proofs in appendix)
- Discussion of what the theorem implies practically
- Comparison of your bounds with existing bounds (if any)

**Common mistakes:**
- Hiding assumptions in the proof rather than stating them upfront
- Not discussing when assumptions are realistic vs. restrictive
- Proving something trivial or obvious

**Reviewer expectations:** Assumptions are reasonable and clearly stated. The theorem tells us something useful that we did not already know.

## 12.6 Experiments

**Purpose:** Empirically validate that your method works and provides improvement over baselines.

**What to include:**
- Datasets: at least 2-3, including one that is more challenging than MNIST-DVS (e.g., DVS-Gesture, N-Caltech101, CIFAR10-DVS)
- Baselines: FL-SNN, separate training, at least one other relevant method
- Metrics: test accuracy, test loss, communication cost, convergence speed
- Non-IID scenarios: multiple levels of data heterogeneity (not just extreme partition)
- Number of devices: test with N = 2, 5, 10, 20+
- Ablation study: remove/modify components of your method one at a time
- Error bars: standard deviation over at least 5 runs
- Hyperparameter sensitivity analysis

**Common mistakes:**
- Cherry-picking results or datasets where your method works best
- Not reporting standard deviations or confidence intervals
- Unfair baseline comparison (different hyperparameter tuning effort)
- Too few trials

**Reviewer expectations:** Results are comprehensive, fair, and statistically sound. I should be convinced the improvement is real, not due to noise or hyperparameter tuning.

## 12.7 Discussion

**Purpose:** Interpret results, explain surprising findings, and connect back to the big picture.

**What to include:**
- Why does your method work better? What is the mechanism?
- When does it fail or underperform? (Be honest.)
- What do the results mean for practical deployment?
- Connection to theoretical results (if applicable)

**Common mistakes:**
- Just repeating the numbers from the results section
- Not acknowledging when baselines perform comparably
- Overinterpreting small differences that may not be statistically significant

**Reviewer expectations:** Thoughtful analysis, not just cheerleading for your method.

## 12.8 Limitations

**Purpose:** Honestly acknowledge what your work does not cover.

**What to include:**
- Scalability limitations (if only tested with few devices)
- Assumptions that may not hold in practice
- Computational overhead of your method vs. baselines
- Scenarios where your method may not apply

**Common mistakes:**
- Listing limitations so severe they undermine the paper (save those for future work)
- Being too vague ("future work could improve results")

**Reviewer expectations:** Self-awareness. Reviewers respect honest limitation analysis and penalize papers that ignore obvious weaknesses.

## 12.9 Conclusion

**Purpose:** Summarize contributions and impact in 1 paragraph.

**What to include:**
- Restate the problem (one sentence)
- Restate your method (one sentence)
- Summarize key results (one sentence)
- Broader impact or future direction (one sentence)

**Common mistakes:**
- Introducing new information or results not in the main paper
- Being too long (keep it under 200 words)
- Overclaiming

**Reviewer expectations:** Clean wrap-up. No surprises.

## 12.10 References

**What to include:**
- All papers cited in the text
- Prefer published versions over arXiv preprints where available
- Follow the venue's citation format exactly

**Common mistakes:**
- Missing references for key claims
- Citing only your own prior work
- Not citing the paper you are building upon (FL-SNN)

---

# 13. Publication Strategy Guide

## 13.1 Suitable Conference/Journal Types

| Venue Type | Examples | Fit |
|---|---|---|
| Signal processing conferences | ICASSP, ICIP, GlobalSIP | High -- this paper was published at ICASSP |
| Machine learning conferences | NeurIPS, ICML, ICLR (workshops) | Medium-High -- if results are strong and scalable |
| Neuromorphic computing venues | NICE, IEEE BioCAS | High -- direct target audience |
| Federated learning venues | FL-ICML Workshop, NeurIPS FL Workshop | High -- novel application domain for FL |
| Edge computing conferences | SEC, IEEE/ACM Symposium on Edge Computing | High -- practical relevance |
| Journals | IEEE TNNLS, IEEE JSAC, Frontiers in Neuroscience | High -- for extended versions with theory and more experiments |

## 13.2 Required Baseline Expectations

- **Minimum:** FL-SNN (this paper), separate SNN training, centralized SNN training (upper bound)
- **Recommended:** FedAvg with equivalent-parameter ANN, at least one communication-efficient FL method (e.g., signSGD, sparse communication)
- **For top venues:** Include 3-4 recent neuromorphic FL methods published after this paper

## 13.3 Experimental Rigor Level

| Venue Tier | Datasets | Devices | Trials | Theory Required |
|---|---|---|---|---|
| Workshop | 1-2 | 2-5 | 3-5 | No |
| Mid-tier conference (ICASSP) | 2-3 | 2-10 | 5-10 | Preferred |
| Top conference (NeurIPS/ICML) | 3+ | 10-100 | 5+ with significance tests | Strongly preferred |
| Journal (IEEE TNNLS) | 3+ | 5-50 | 10+ | Expected |

## 13.4 Common Rejection Reasons

1. "Incremental contribution: just applying existing FL to SNNs" -- counter by showing SNN-specific challenges that require novel solutions.
2. "Limited experiments: only MNIST-DVS with 2 devices" -- use more datasets and scale to more devices.
3. "No theoretical analysis" -- provide at least convergence rate analysis or communication complexity bounds.
4. "No comparison with ANN baselines" -- include accuracy-vs-energy Pareto comparison.
5. "No real hardware validation" -- if possible, include even a small-scale hardware demo.
6. "Privacy claims unsupported" -- either add formal privacy analysis or carefully avoid privacy claims.

## 13.5 Increment Needed for Acceptance

- **For ICASSP-level:** A clear algorithmic novelty (e.g., new aggregation method, new learning signal) + 2-3 datasets + improvement over FL-SNN.
- **For NeurIPS/ICML-level:** Theoretical contribution (convergence analysis) + comprehensive experiments (5+ datasets, many devices) + practical insights (energy measurements or hardware demo).
- **For journal:** All of the above + detailed ablation + extended discussion + broader experimental coverage.

---

# 14. Researcher Quick Reference Tables

## 14.1 Key Terminology Table

| Term | Definition | Context in Paper |
|---|---|---|
| SNN | Spiking Neural Network: neural network communicating via binary spikes over time | Core model on each edge device |
| FL | Federated Learning: collaborative training without sharing data | Cooperative training framework |
| FL-SNN | Proposed algorithm combining FL with probabilistic SNN training | Main contribution |
| BS | Base Station: acts as parameter server in the FL protocol | Central aggregation point |
| GLM | Generalized Linear Model: probabilistic framework for neuron spiking | Foundation for SNN model |
| DVS | Dynamic Vision Sensor: event-driven camera producing spike-like output | Data source for experiments |
| STDP | Spike-Timing-Dependent Plasticity: biological learning rule based on spike timing | Special case of the paper's learning rule |
| Eligibility trace | Running average of parameter gradients over recent timesteps | Component of local learning rule |
| Learning signal | Scalar feedback indicating overall network performance | Component of local learning rule (for hidden neurons) |
| Membrane potential | Accumulated input signal at a neuron; determines spike probability | Key state variable in SNN |
| Synaptic weight | Learnable parameter controlling connection strength between neurons | What is learned and communicated |
| Communication period (tau) | Number of global iterations between FL synchronization rounds | Key hyperparameter |
| Communication rate (r) | Number of synaptic weights exchanged per global iteration: r = K'_a / tau | Metric for communication efficiency |
| Basis functions | Fixed filter shapes (raised cosines) used to parameterize synaptic filters | Architectural choice for synaptic model |

## 14.2 Important Equations Summary

| Eq. | Name | Purpose | Key Insight |
|---|---|---|---|
| (1) | Local empirical loss | Per-device loss function | Average of loss over local data |
| (2) | Global FL objective | Overall optimization target | Weighted average of local losses across all devices |
| (3) | Local SGD update | One step of local training | Standard gradient descent on local data |
| (4) | Global averaging | Parameter synchronization | Weighted average of all devices' parameters |
| (5) | Spiking probability | Probabilistic neuron model | Sigmoid of membrane potential gives spike probability |
| (6) | Membrane potential | Signal integration | Sum of filtered incoming spikes + self-feedback + bias |
| (7) | Log-loss for SNN | SNN training objective | Negative log-probability of correct output spike pattern |
| (8) | Local SNN loss | Per-device SNN loss | Equation 7 averaged over local dataset |
| (10) | Local update rule | Core FL-SNN learning | Different rules for output (direct) vs. hidden (modulated by learning signal) neurons |
| (11) | Learning signal | Global performance feedback | Exponential running average of log-probabilities at observable neurons |
| (12) | Eligibility trace | Per-parameter credit | Exponential running average of gradients of log-spiking-probability |

## 14.3 Parameter Meaning Table

| Parameter | Symbol | Type | Value in Experiments | Role |
|---|---|---|---|---|
| Learning rate | alpha | Hyperparameter | 0.05 | Controls step size of parameter updates |
| Decay factor | kappa | Hyperparameter | 0.2 | Controls memory length of learning signal and eligibility trace |
| Local steps per global iteration | delta_s | Parameter | 5 | Maps local SNN time to global iteration time |
| Communication period | tau | Parameter | 8, 64, 400, infinity | Controls frequency of FL synchronization |
| Number of devices | N | System parameter | 2 | Number of participating edge devices |
| Number of basis functions | K_a | Architecture parameter | 8 | Complexity of synaptic filter parameterization |
| Exchanged weights | K'_a | Communication parameter | 1, 2, 4, 8 | Number of synaptic weights sent per communication round |
| Communication rate | r = K'_a / tau | Derived parameter | 1/8, 1/4 | Weights exchanged per global iteration |
| Number of hidden neurons | N_H | Architecture parameter | 0, 16 | Representational capacity of the SNN |
| Input neurons | N_X | Architecture parameter | 676 (26x26) | Matches image dimensions |
| Output neurons | N_Y | Architecture parameter | 2 | Matches number of classes |
| Training examples | D | Training parameter | 400 | Number of examples per device per training run |
| Samples per example | S' | Data parameter | 80 | Temporal length of each spike-encoded example |
| Total local timesteps | S = D*S' | Derived | 32,000 | Total SNN simulation steps during training |

## 14.4 Algorithm Flow Summary

```
INITIALIZATION
    |
    v
[All devices share same initial parameters]
    |
    v
FOR t = 1 to T: ─────────────────────────────────────────┐
    |                                                      |
    v                                                      |
[Each device i, in parallel:]                              |
    |                                                      |
    ├── Generate hidden neuron spikes (stochastic, Eq 9)   |
    |                                                      |
    ├── Compute learning signal (Eq 11)                    |
    |                                                      |
    ├── Compute eligibility traces (Eq 12)                 |
    |                                                      |
    ├── Update output neuron weights (Eq 10, direct)       |
    |                                                      |
    ├── Update hidden neuron weights (Eq 10, modulated)    |
    |                                                      |
    ├── IF t mod tau == 0:                                 |
    |   ├── Send parameters to BS                          |
    |   ├── BS computes weighted average (Eq 4)            |
    |   └── Receive averaged parameters from BS            |
    |                                                      |
    └── ELSE: keep local parameters ──────────────────────>┘
    
FINAL OUTPUT: theta_F (converged model parameters)
```

---

# 15. One-Page Master Summary Card

## Problem
Edge devices need to train AI models locally but lack sufficient data individually. Standard deep learning is too energy-intensive for edge hardware. No framework existed for collaborative training of energy-efficient spiking neural networks across multiple devices.

## Idea
Combine biologically plausible spiking neural network (SNN) training with federated learning (FL) so that multiple edge devices can collaboratively learn without sharing data, while consuming minimal energy.

## Method
**FL-SNN:** Each device trains a probabilistic SNN using a local three-factor learning rule (learning signal x eligibility trace x learning rate) instead of backpropagation. Every tau iterations, devices send their synaptic weights (or a subset) to a base station, which averages them and broadcasts the result back.

## Results
- FL-SNN with frequent communication (small tau) clearly outperforms separate training (no collaboration).
- For a fixed communication budget, sending fewer weights more frequently beats sending more weights less frequently.
- Best accuracy: approximately 85-90% on binary classification of MNIST-DVS digits ('1' vs '7').

## Weakness
- Only 2 devices, 2 classes, and 1 dataset tested.
- No convergence theory, no privacy analysis, no hardware validation.
- Single scalar learning signal limits hidden neuron training quality.
- No comparison with conventional ANN+FL baselines.

## Research Opportunity
- Scale to many devices and complex tasks with convergence guarantees.
- Add differential privacy or secure aggregation.
- Design richer learning signals (layer-specific, neuron-group-specific).
- Implement asynchronous FL-SNN for heterogeneous device networks.
- Validate on real neuromorphic hardware (Loihi 2, SpiNNaker 2).

## Publishable Extension
Propose a communication-efficient, privacy-preserving, asynchronous federated learning protocol for SNNs with theoretical convergence guarantees, validated on 3+ DVS datasets with 10+ devices and compared against both FL-SNN and conventional ANN+FL baselines. Include energy measurements on neuromorphic hardware if accessible.
