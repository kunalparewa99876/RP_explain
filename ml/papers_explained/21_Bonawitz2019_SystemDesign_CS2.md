# Research Companion: Towards Federated Learning at Scale — A System Design
**Bonawitz et al., 2019 — ICLR Workshop on Federated Learning / SysML 2019**

---

# 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Paper Title** | Towards Federated Learning at Scale: A System Design |
| **Authors** | Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloé Kiddon, Jakub Konečný, Stefano Mazzocchi, H. Brendan McMahan, Timon Van Overveldt, David Petrou, Daniel Ramage, Jason Roselander |
| **Affiliation** | Google Inc., Mountain View, CA, USA |
| **Year** | 2019 |
| **Problem Domain** | Production-scale Federated Learning (FL) system for mobile devices |
| **Paper Type** | Systems / Engineering |
| **Core Contribution** | First description of a production-level FL system architecture deployed on tens of millions of Android devices using TensorFlow, covering protocol design, device/server architecture, Secure Aggregation, analytics, tooling, and real deployment results |
| **Key Idea** | Federated Learning can be deployed at massive scale on mobile devices through a carefully engineered system of synchronous round-based training, actor-based servers, ephemeral in-memory aggregation, Secure Aggregation, device eligibility gating, and pace steering — all without ever moving raw user data off the device |
| **Required Background** | Basic FL concepts (FedAvg), TensorFlow basics, distributed systems fundamentals, basic understanding of Secure Multi-Party Computation (MPC), mobile OS concepts (Android JobScheduler) |
| **Primary Baseline** | FedAvg (McMahan et al., 2017) — the canonical FL algorithm this system executes in production |
| **Main Innovation Type** | System design — not a new algorithm, but a novel engineering solution integrating privacy, fault tolerance, scalability, and usability into a production FL pipeline |
| **Difficulty Level** | Intermediate (systems concepts) / Advanced (security and MPC sections) |
| **Reproducibility Level** | Low — proprietary Google infrastructure; but the design principles are openly described and replicable |

---

# 1. Research Context & Core Problem

## What Problem Does This Paper Address?

Federated Learning (FL) was introduced as a concept in 2017 by McMahan et al. — the idea being that instead of sending user data to a central server, devices train models locally and only send model weight updates back. However, in 2019, **no one had actually built a production-scale FL system** that ran on real devices with real users at scale. This paper fills that gap.

The specific challenges the paper tackles:

- **Device heterogeneity**: Different phones have different hardware capabilities, memory, and OS versions
- **Unreliable connectivity**: Devices drop in and out constantly — WiFi disconnects, batteries die, users pick up their phones
- **Device eligibility**: Training should only happen when the device is idle, charging, and on WiFi — so the pool of available devices changes dramatically throughout the day
- **Scalability**: A system that can handle 10 devices in testing must also handle 10 million in production
- **Privacy**: Even model updates could leak user information, so individual updates must be kept private
- **Stragglers**: Some devices respond slowly or not at all — the system must handle this gracefully
- **Developer usability**: Model engineers must be able to deploy FL tasks easily, safely, and with version compatibility checks

## Why the Problem Exists

- FL was theoretically proposed but the engineering complexity of making it work in the real world was enormous and underestimated
- Data center ML systems (like parameter servers) assume reliable, fast, always-on compute nodes — mobile devices are the opposite
- Privacy regulations (GDPR, etc.) made data centralization legally risky, increasing urgency for on-device learning
- Secure Aggregation protocols existed in theory but had never been embedded into a production ML pipeline

## Historical and Theoretical Gap

| What Existed Before This Paper | What Was Missing |
|---|---|
| FedAvg algorithm (McMahan et al., 2017) — theory of FL | Practical system implementing FL at mobile-device scale |
| Secure Aggregation protocol (Bonawitz et al., 2017) — cryptographic theory | Integration of SecAgg into a real production system |
| Parameter server frameworks (Li et al., 2014) — datacenter ML | Mobile-adapted equivalent with eligibility gating and pace control |
| TensorFlow — ML computation framework | Way to package TF computations to run safely on millions of phones |
| Academic FL papers on convergence guarantees | Production operational data showing FL actually works at scale |

## Contribution Category

- **System design** — the primary contribution is the complete engineering architecture
- **Operational insight** — real deployment metrics from production FL runs (10M+ devices)
- **Privacy integration** — practical embedding of SecAgg and differential privacy hooks into a usable pipeline
- **Tooling** — developer workflow for building and deploying FL tasks safely

### Why This Paper Matters

This paper is the existence proof that Federated Learning works at scale. Every subsequent practical FL deployment paper cites this work. It bridges the gap between FL theory and FL practice, and defines the vocabulary (FL population, FL task, FL round, FL plan, pace steering) that the entire field now uses. If you want to build an FL system — for healthcare, finance, IoT, or mobile — this paper is your blueprint.

### Remaining Open Problems (Identified by This Paper)

- Participation bias: devices that charge overnight ≠ all users (geographic, demographic skew)
- Slow convergence: FL is ~7× slower wall-clock than datacenter training for equivalent models
- Multi-tenant on-device scheduling does not account for app recency or usage frequency
- Bandwidth: weight updates for RNNs can be larger than raw data uploads
- Device scheduling parameter tuning (round timeout windows) is still static, not adaptive
- Extending the system beyond ML to "Federated Computation" for analytics use cases
- Formal privacy guarantees for the full system end-to-end (not just individual components)

---

# 2. Minimum Background Concepts

## 2.1 Federated Learning (FL)

**Plain definition**: A machine learning approach where multiple devices each train a shared model using only their own local data, then send model updates (not raw data) to a central server for aggregation.

**Role in this paper**: The entire system is built to execute FL in production. The training algorithm used is FedAvg.

**Why authors needed it**: FL is the entire reason the system exists — the paper describes how to make FL actually work on real devices at real scale.

---

## 2.2 Federated Averaging (FedAvg)

**Plain definition**: A specific FL algorithm where selected devices run multiple local gradient descent steps on their own data, then send the resulting weight update (delta) to the server, which averages all deltas weighted by the number of training samples each device had.

**Role in this paper**: FedAvg is the primary algorithm running in production on this system. Its pseudocode is given in Appendix B.

**Why authors needed it**: FedAvg determines the shape of the communication round — local training followed by one aggregation — which defines the protocol structure.

---

## 2.3 Secure Multi-Party Computation (MPC)

**Plain definition**: A cryptographic technique where multiple parties jointly compute a function of their private inputs without any party learning anything about another party's input beyond what is revealed by the output.

**Role in this paper**: Secure Aggregation (a specific MPC protocol) is used so that the server can compute the sum of all device updates without ever seeing any individual device's update in plaintext.

**Why authors needed it**: Even if raw data never leaves the device, model updates themselves can sometimes be reversed to infer training data. SecAgg prevents this additional privacy risk.

---

## 2.4 Actor Programming Model

**Plain definition**: A concurrent computation paradigm where independent units called "actors" communicate only by sending messages to each other. Each actor processes messages one at a time.

**Role in this paper**: The entire FL server is designed using the Actor Model. Coordinators, Selectors, Master Aggregators, and Aggregators are all actors.

**Why authors needed it**: The actor model provides natural isolation, scalability, and fault tolerance. When one actor fails, only its message queue is lost — other actors continue. This is critical for handling millions of unreliable device connections.

---

## 2.5 Differential Privacy (DP)

**Plain definition**: A formal mathematical guarantee that the output of a computation does not change significantly whether or not any single individual's data was included.

**Role in this paper**: DP is mentioned as an additional privacy enhancement (McMahan et al., 2018 techniques are implemented). It adds noise to aggregated updates to limit information leakage.

**Why authors needed it**: SecAgg ensures server cannot see individual updates. DP ensures that the aggregate itself doesn't reveal too much about individual participants.

---

## 2.6 Android JobScheduler

**Plain definition**: An Android OS service that lets apps schedule background jobs to run only when specific system conditions are met (charging, WiFi, idle).

**Role in this paper**: The FL runtime uses JobScheduler to ensure FL training only happens under favorable device conditions — protecting battery life and user experience.

**Why authors needed it**: Without this constraint, FL training could drain battery or use mobile data, which would harm users and violate the principle that FL should be invisible to users.

---

## 2.7 TensorFlow Computation Graph

**Plain definition**: A declarative description of computation as a directed graph where nodes are operations and edges are tensors (multi-dimensional arrays flowing between operations).

**Role in this paper**: FL Plans contain serialized TensorFlow graphs. These graphs define exactly what computation runs on the device and what aggregation runs on the server.

**Why authors needed it**: By serializing TF graphs into plans, the authors can deploy computation to devices without running Python directly on them — which is impossible on Android.

---

## 2.8 Straggler

**Plain definition**: In distributed systems, a "straggler" is a worker node that takes much longer than the others to complete its assigned task.

**Role in this paper**: The FL protocol handles stragglers by selecting 130% of the target number of devices and discarding the slowest ones when enough responses arrive.

**Why authors needed it**: Waiting indefinitely for slow devices would block round progress. The straggler mitigation strategy trades some computation for round completion speed.

---

# 3. Mathematical / Theoretical Understanding Layer

*(This is primarily a systems paper. Mathematical content is concentrated in the FedAvg algorithm from Appendix B and the SecAgg complexity analysis.)*

## 3.1 FedAvg Algorithm (from Appendix B)

### Intuition Before Equations

The server wants a global model that is good across all users' data combined, but it cannot see that data. The solution: each selected device runs gradient descent on its own data, computes how much the weights changed (the "delta"), and reports that delta back. The server averages all the deltas, weighted by how many training samples each device contributed.

### Algorithm Steps

**Server side (per round t):**
1. Select 1.3K eligible clients (where K is target participants)
2. Wait for updates from K clients
3. Compute weighted average of deltas:

$$\bar{n}_t = \sum_{k=1}^{K} n_k^t \quad \text{(total samples across all devices)}$$

$$\Delta_t = \frac{1}{\bar{n}_t} \sum_{k=1}^{K} n_k^t \cdot \Delta_k^t \quad \text{(weighted average update)}$$

4. Update global weights: $w_{t+1} = w_t + \Delta_t$

**Client side (ClientUpdate function):**
1. Receive global weights $w$ from server
2. Divide local data into minibatches $B$
3. Run E local epochs of SGD:
   $$w \leftarrow w - \eta \cdot \nabla L(w; b) \quad \text{for each batch } b$$
4. Compute delta: $\Delta = w_{\text{final}} - w_{\text{init}}$
5. Report $(\Delta, n)$ to server where $n$ = number of local samples

### Variable Meaning Table

| Symbol | Meaning | Why It Matters |
|---|---|---|
| $w_t$ | Global model weights at round $t$ | The shared model being learned |
| $\Delta_k^t$ | Weight update from device $k$ in round $t$ | What the device sends to server (not raw data) |
| $n_k^t$ | Number of training samples on device $k$ | Used for weighted averaging — more data = more influence |
| $\bar{n}_t$ | Total samples across all devices in round | Normalization factor |
| $\eta$ | Learning rate | Controls how large each gradient step is |
| $K$ | Target number of devices per round | Typically a few hundred in production |
| $E$ | Number of local epochs | How much each device trains before reporting |

### Key Insight

The delta $\Delta$ is explicitly noted as more compressible than the full weight vector $w$. This is because deltas are sparse and small-magnitude compared to the absolute weights. This motivates communication compression techniques.

### Practical Interpretation

In production, typical values are: K = a few hundred devices, E = a small number of local epochs. The authors observe that increasing K beyond a few hundred shows diminishing convergence improvement. This is why each round is not global — involving all millions of devices would add no benefit while massively increasing coordination complexity.

---

## 3.2 Secure Aggregation Complexity

The SecAgg protocol has **quadratic computational cost on the server** with respect to the number of participating devices $n$:

$$\text{Server cost} \propto O(n^2)$$

### Practical Implication

This limits each SecAgg instance to handling at most a few hundred devices. The system resolves this by running one SecAgg instance per Aggregator actor, each over a small group. The Master Aggregator then aggregates these intermediate sums in plaintext, trading some privacy granularity for scalability.

### Mathematical Insight Box

> **Key idea for researchers**: Quadratic scaling is often a hard barrier in cryptographic protocols. The engineering solution here — hierarchical aggregation where only the leaves use crypto and the root aggregates in plaintext — is a reusable pattern for making cryptographic protocols scale.

---

# 4. Proposed Method / Framework — System Architecture

**Paper Classification: Systems / Engineering**

The contribution is an end-to-end system. Below is a component-by-component breakdown.

## 4.1 Overall System Flow

```
[Devices] → Check in with eligibility criteria (idle, charging, WiFi)
    ↓
[Selector Actors] → Accept or reject device connections, forward to Aggregators
    ↓
[Coordinator Actor] → Manages one FL population, schedules rounds, instructs Selectors
    ↓
[Master Aggregator Actor] → Manages one FL task's rounds, spawns Aggregators
    ↓
[Aggregator Actors] → Collect updates from subsets of devices, aggregate (with or without SecAgg)
    ↓
[Master Aggregator] → Final aggregation of Aggregator outputs
    ↓
[Global Model Storage] → Updated model written to persistent storage
    ↓
[Devices] → Pull updated global model for inference
```

---

## 4.2 Communication Protocol — Three-Phase Round

Every FL round has three phases:

### Phase 1: Selection

**What happens**: Devices that meet eligibility criteria (idle + charging + WiFi) open a bidirectional stream with the server. The server uses reservoir sampling to pick a subset of ~few hundred from potentially thousands of connected devices.

**Why authors did this**: Training on every available device is unnecessary (diminishing returns past ~hundreds) and would create coordination overhead. A small representative sample is sufficient for convergence.

**Weakness of this step**: Reservoir sampling is uniform — it does not account for which devices have the most valuable or most recent data.

**Research opportunity**: Implement principled, privacy-preserving device selection based on gradient diversity or local data quality signals.

---

### Phase 2: Configuration

**What happens**: The server sends each selected device: (a) the FL Plan (serialized TF graph + instructions), and (b) an FL Checkpoint (current global model weights).

**Why authors did this**: Decoupling the plan from the model allows the same plan to be used across multiple rounds. Plans are compiled in advance and versioned for TF compatibility.

**Weakness**: The FL Plan must be pre-compiled for every version of TF running in the field. Each incompatible TF change requires a new versioned plan.

**Research opportunity**: Design a plan representation that is TF-version-agnostic, or use a higher-level IR (like MLIR) for cross-version compatibility.

---

### Phase 3: Reporting

**What happens**: Devices perform local training and send their weight deltas back. The server aggregates them with FedAvg. Stragglers are discarded after a timeout. If fewer than the minimum required devices report, the round is abandoned.

**Why authors did this**: Discarding stragglers prevents slow devices from blocking round completion. The 130% overselection buffer ensures enough responses arrive even with dropout.

**Weakness**: Abandoned rounds waste all device computation from that round.

**Research opportunity**: Partial round utilization — if a round fails, use partial updates from reporting devices rather than discarding them entirely.

---

## 4.3 Pace Steering

**What it is**: A flow control mechanism where the server tells each device when to reconnect. The device attempts to comply.

**Two modes**:
- **Small population mode**: Server synchronizes reconnection times so enough devices are available simultaneously for one round. This is essential for SecAgg which requires a minimum group size.
- **Large population mode**: Server staggers reconnection times to avoid "thundering herd" — all devices checking in simultaneously and overwhelming the server.

**Why authors did this**: Without pace steering, device availability follows natural diurnal patterns that can either flood or starve the server. Pace steering creates a managed, predictable flow.

**Weakness**: Pace steering is stateless and probabilistic. In adversarial settings, devices that ignore reconnection suggestions could disrupt round timing.

**Research opportunity**: Adaptive pace steering using online learning to predict device availability patterns and preemptively adjust check-in windows per device.

---

## 4.4 Device Architecture (Android)

### Components

| Component | Description |
|---|---|
| **Example Store** | Application-managed data repository (e.g., SQLite). App implements an API to expose data to the FL runtime |
| **FL Runtime** | Background service triggered by Android JobScheduler when device is idle + charging + WiFi |
| **AIDL IPC** | Android inter-process communication mechanism allowing FL runtime to work across apps |
| **Remote Attestation** | Android SafetyNet API used to verify device authenticity — prevents non-genuine devices from participating |

### Control Flow

1. App registers with FL runtime, providing FL population name and example store
2. JobScheduler invokes FL runtime in a separate process when eligibility conditions are met
3. FL runtime contacts server, announces readiness
4. If selected, receives FL Plan and model checkpoint
5. Queries example store for training data
6. Runs local training per FL Plan
7. Reports updates to server
8. Cleans up temporary resources

**Why authors did this**: Running FL only when idle + charging + WiFi ensures zero impact on user experience. Separation of FL runtime from the app process means FL doesn't compete with the user's active app.

**Weakness**: The hard eligibility requirement (must be charging + WiFi) systematically excludes users who rarely charge on WiFi — a major source of participation bias.

**Research opportunity**: Soft eligibility constraints with battery-impact modeling — allow limited FL under partial conditions (e.g., charging but on cellular) with reduced computation budgets.

---

## 4.5 Server Architecture — Actor Model

### Actors and Their Roles

| Actor | Type | Responsibility |
|---|---|---|
| **Coordinator** | Persistent, one per FL population | Global synchronization, instructs Selectors on how many devices to accept, spawns Master Aggregators |
| **Selector** | Persistent, globally distributed | Accepts/rejects device connections, forwards device streams to Aggregators |
| **Master Aggregator** | Ephemeral, one per FL task round | Manages round lifecycle, spawns Aggregators, performs final aggregation |
| **Aggregator** | Ephemeral, one per device group | Collects and aggregates updates from a subset of devices (optionally with SecAgg) |

### Key Design Decisions

**All state is in-memory (ephemeral actors)**:
- No per-device updates are ever written to persistent storage
- This eliminates the possibility of datacenter-internal attacks targeting stored device logs
- If an Aggregator crashes, only the devices connected to it lose their round — not the entire round

**Pipelining**:
- The Selection phase of round $t+1$ runs in parallel with the Configuration/Reporting phases of round $t$
- This is free in the actor model — Selectors run their acceptance logic continuously

**Geographic distribution**:
- Selectors can be placed in data centers close to devices, reducing network latency
- Coordinators communicate with Selectors via message passing regardless of geographic distance

**Why authors did this**: The actor model naturally handles millions of concurrent device connections without shared mutable state, which would create lock contention and scalability bottlenecks.

**Weakness**: Actor systems can suffer from message queue overload under burst traffic. The paper does not describe backpressure mechanisms.

**Research opportunity**: Adaptive Coordinator scheduling using real-time device availability predictions to pre-position Aggregator instances in regions with upcoming high device activity.

---

## 4.6 Secure Aggregation Integration

**What it does**: Guarantees that even inside Google's data center, no individual device's weight update can be inspected. Only the sum across all devices in a group is ever visible in plaintext.

**Protocol structure** (4 rounds of interaction):

| Round | Phase | What Happens |
|---|---|---|
| 1 | Prepare (part 1) | Devices exchange public keys, establish shared secrets |
| 2 | Prepare (part 2) | Devices exchange additional secret shares |
| 3 | Commit | Devices upload cryptographically masked model updates |
| 4 | Finalization | Devices reveal cryptographic keys to unmask the aggregate |

**Robustness**: Devices that drop out during Prepare phase are excluded cleanly. Devices that drop out during Finalization phase are recoverable if enough others complete.

**Scalability solution**: SecAgg runs per Aggregator actor over groups of ~hundreds. The Master Aggregator adds intermediate sums in plaintext. Each FL task defines a parameter $k$ — the minimum group size for secure aggregation.

**Why authors did this**: SecAgg adds a privacy guarantee against honest-but-curious datacenter adversaries. Without SecAgg, a rogue employee with database access could reconstruct individual user behavior from stored update logs.

**Weakness**: The two-level aggregation (SecAgg at Aggregator level + plaintext at Master Aggregator level) means the privacy guarantee is over groups, not the full global population.

**Research opportunity**: Hierarchical SecAgg — running SecAgg at multiple levels of the hierarchy, potentially using more efficient cryptographic primitives like Homomorphic Encryption or PIR for the upper levels.

---

## 4.7 FL Plans

**What it is**: A data structure containing two parts:
- **Device portion**: TensorFlow graph, data selection criteria for example store, batching instructions, number of epochs, node labels for loading/saving weights
- **Server portion**: Aggregation logic (how to combine device updates)

**Why not just run Python?**: Python cannot run on Android devices, and running arbitrary Python on the server for untrusted device responses would be a security risk. The Plan serializes computation into a safe, inspectable format.

**Versioning**: Each plan is compiled into multiple versioned variants — one per TF runtime version deployed in the field. Graph transformations are applied to make the same logical computation compatible with older TF operators.

**Why authors did this**: Devices in the field may run TF runtimes months older than the current development version. Without versioning, new models would be incompatible with large portions of the device fleet.

**Weakness**: Versioning requires manual identification and transformation of incompatible operators — the paper reports ~one such incompatibility every three months.

**Research opportunity**: Automatic compatibility testing using fuzzing-based approaches to detect operator incompatibilities before deployment.

---

## 4.8 Analytics and Observability

**Device-side logging**: Non-PII health metrics logged to cloud:
- Device state at training activation
- Training duration and frequency
- Memory usage
- Error types
- Device model / OS / FL runtime version

**State machine visualization**: Every device's round participation is logged as a sequence of state transitions, then represented as ASCII symbols for dashboards:

| Symbol | Meaning |
|---|---|
| `-` | FL server check-in |
| `v` | Downloaded plan |
| `[` | Training started |
| `]` | Training completed |
| `+` | Upload started |
| `^` | Upload completed |
| `#` | Upload rejected (too late) |
| `!` | Interrupted |

**Example session shapes**:
- `-v[]+^` = success (75% of sessions)
- `-v[]+#` = trained but rejected due to late upload (22%)
- `-v[!` = interrupted during training (2%)

**Why authors did this**: The FL system runs on devices they don't control. Without detailed analytics, it is impossible to distinguish device-side bugs from server-side bugs, or to detect model degradation.

**Research opportunity**: Federated analytics — computing aggregate statistics about the FL system's health without any raw device data leaving the device.

---

# 5. Experimental Setup / Evaluation Design

This is a systems paper. There is no controlled experiment in the traditional ML sense. Instead, the authors report **operational metrics** from a production deployment.

## 5.1 Deployment Context

| Parameter | Value |
|---|---|
| Deployment platform | Android devices (recent versions, ≥2 GB RAM) |
| Production duration | Over 1 year of production operation |
| Cumulative FL population | ~10M daily active devices |
| Active simultaneous participants | Up to ~10,000 devices at peak |
| Primary application | Gboard mobile keyboard (next-word prediction, content suggestions) |
| Training algorithm | FedAvg (McMahan et al., 2017) |

## 5.2 Key Operational Metrics

| Metric | Observed Value | Significance |
|---|---|---|
| Device dropout rate | 6%–10% per round | Informs the 130% overselection buffer |
| Diurnal variation | 4× difference between peak and off-peak | Drives pace steering design |
| Successful session rate | 75% | Validates round completion feasibility |
| Late-upload rejection rate | 22% | Indicates round timeout windows need tuning |
| Round completion time | Capped by server timeout | Straggler mitigation in action |
| FL convergence (Gboard NWP) | 3000 rounds, 5 days, 1.5M users | Wall-clock convergence benchmark |
| Performance improvement (NWP) | Top-1 recall: 13.0% → 16.4% vs. n-gram baseline | Model quality validation |
| FL vs. server-trained parity | FL model matches server-trained RNN on live A/B test | Proves FL produces competitive models |
| Download/upload asymmetry | Download dominates (plan + model vs. delta only) | Motivates update compression research |

## 5.3 Evaluation Protocol

- Real-world A/B experiments (FL model vs. n-gram vs. server-trained RNN) in Gboard
- Production monitoring dashboards tracking device health, round completion rates, network throughput
- State-sequence visualizations aggregated across all devices to detect systemic failure patterns

### Experimental Reliability Analysis

| Aspect | Assessment |
|---|---|
| **Trustworthy** | Operational metrics from 1+ year production deployment — represents real-world behavior, not simulation |
| **Trustworthy** | A/B experiment results for Gboard — controlled comparison with real users |
| **Questionable** | All metrics are from a narrow set of applications (primarily keyboard) — generalizability unknown |
| **Questionable** | No controlled baseline comparison for the system architecture — no head-to-head with an alternative FL system design |
| **Questionable** | Device pool is limited to Android devices with ≥2 GB RAM — not representative of all global mobile users |
| **Questionable** | Performance depends heavily on network/device speed which varies by region — not characterized across geographies |

---

# 6. Results & Findings Interpretation

## 6.1 Key Results

**Gboard Next-Word Prediction**:
- RNN with ~1.4M parameters trained via FL
- 3,000 rounds over 5 days, 6×10⁸ sentences from 1.5×10⁶ users
- Top-1 recall improved from 13.0% (n-gram baseline) to 16.4% (FL model)
- FL model matches performance of a server-trained RNN (which required 1.2×10⁸ SGD steps)
- Live A/B test: FL model outperforms both n-gram and server-trained RNN

**Operational Scale**:
- System operates continuously at 10M daily active devices
- Up to 10,000 simultaneous participants
- 4× diurnal variation in participant count

**Session Outcomes** (from Table 1):
- 75% complete successfully
- 22% complete training but miss upload window
- 2% interrupted during training

## 6.2 What the Results Mean

- **FL can match datacenter ML quality**: The Gboard result shows FL is not a quality-sacrificing alternative — it is competitive with server-trained models and even outperforms them in live A/B (likely because FL trains on actual user typing, which no server data can replicate)
- **The 22% late-upload rejection rate is significant**: Nearly 1 in 4 devices trains but contributes nothing. This wasted computation motivates better round timeout tuning or partial aggregation of late arrivals
- **4× diurnal variation is large**: This must be accounted for in round scheduling — static scheduling would either flood or starve the system

## 6.3 Failure Patterns and Edge Cases

- Rounds are abandoned if too few devices report in time — represents wasted device computation
- Actor crashes result in partial round loss, requiring the Coordinator to restart the round from the previous committed checkpoint
- The 130% overselection buffer works in practice but is a static heuristic — adversarial or unusual device behavior patterns could make it inadequate

### Publishability Strength Check

| Result | Publication Grade? | Notes |
|---|---|---|
| FL matching datacenter RNN on Gboard (A/B test) | Yes — strong applied result | Real users, live system, controlled comparison |
| 10M+ device deployment at 10k simultaneous | Yes — unprecedented scale demonstration | No academic FL paper had operated at this scale before |
| 6–10% dropout rate characterization | Moderate — useful engineering insight | Not a controlled study; varies by application |
| State-sequence visualization methodology | Yes — reusable monitoring technique | Novel contribution to FL observability tooling |
| SecAgg at production scale | Yes — first production demonstration of SecAgg | Proves cryptographic protocol is practically feasible |

---

# 7. Strengths – Weaknesses – Assumptions

## Table 1: Technical Strengths

| Strength | Description |
|---|---|
| First production FL system | Existence proof that FL works at real scale, not just in simulation |
| Ephemeral in-memory aggregation | Eliminates persistent per-device logs — strong security property with no cryptographic overhead |
| Secure Aggregation integration | Individual device updates are uninspectable even to Google engineers |
| Hierarchical actor architecture | Scales naturally from 10 to 10M devices via actor spawning |
| FL Plan versioning | Handles TF runtime heterogeneity across the device fleet gracefully |
| Pace steering | Handles both very small and very large FL populations with the same mechanism |
| Pipelining of round phases | Selection of next round runs in parallel with reporting of current round — reduces idle server time |
| Real A/B experiment validation | Proves FL model quality is production-worthy, not just theoretically acceptable |
| Analytics state machine | Provides fine-grained observability without logging PII |

## Table 2: Explicit Weaknesses

| Weakness | Impact |
|---|---|
| Eligibility bias (charging + WiFi only) | Systematically underrepresents users in developing countries or with limited WiFi access |
| Static round timeout configuration | ~22% of device training effort is wasted because round windows are not adaptive |
| Uniform reservoir sampling for device selection | Ignores data quality, diversity, or recency — may not be optimal for convergence |
| Quadratic SecAgg limits group size to ~hundreds | Cannot achieve global SecAgg across all participants in one round — requires hierarchical compromise |
| No persistent per-device logs | Improves privacy but prevents federated debugging (cannot trace which device caused an anomaly) |
| ~7× slower convergence vs. datacenter | Practical limitation for model iteration speed |
| Versioned FL plans require manual intervention | Each TF incompatibility needs identification and graph transformation — ~4 per year |
| Single geographical server zone bias not analyzed | Impact of placing Coordinators and Selectors in different regions is not studied |

## Table 3: Hidden Assumptions

| Assumption | Why It Matters |
|---|---|
| Android devices are representative of the target user base | False for non-Android platforms (iOS) or older devices |
| FedAvg convergence holds for production data distributions | Production data is non-IID in unknown ways — convergence guarantees from theory may not hold |
| 130% device overselection is always sufficient to meet K | Under adversarial dropout or unusual device behavior patterns, rounds may frequently fail |
| Devices follow pace steering suggestions | Devices might ignore reconnection time suggestions due to OS interference or bugs |
| Proxy data (e.g., Wikipedia) is a valid substitute for real device data in simulation | Distribution shift between proxy and real data may cause simulation results to not generalize |
| Android attestation reliably prevents non-genuine device participation | Sophisticated attackers may break attestation; the paper acknowledges but does not fully address this |
| A few hundred devices per round is sufficient for convergence | Application-dependent — complex models may require more devices |

---

# 8. Weakness → Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| Participation bias from eligibility constraints | Hard binary eligibility (idle + charging + WiFi) excludes large user segments | Soft eligibility with resource budgets | Battery-impact-aware scheduling; partial FL runs on cellular with reduced update size |
| 22% wasted device computation (late uploads) | Round reporting windows are statically configured | Adaptive round timeout using historical device report-time distributions | Online learning of per-device report-time CDF; adaptive timeout per device class |
| Uniform device selection ignores data quality | Reservoir sampling is simple and privacy-preserving but uninformed | Privacy-preserving quality-aware selection | Federated bandit selection using proxy signals (e.g., local loss magnitude) without revealing raw data |
| SecAgg quadratic scaling limits global privacy | MPC protocol cost grows with group size | Efficient hierarchical MPC or Homomorphic Encryption for upper aggregation levels | Lightweight HE (CKKS) for intermediate sums; tree-structured SecAgg |
| ~7× slower convergence than datacenter | FedAvg cannot efficiently use more than ~hundreds of parallel devices | New FL algorithms that scale parallelism | Asynchronous FL with bounded staleness; FedProx; SCAFFOLD for variance reduction |
| Static FL Plan compilation per TF version | TF operator interface instability; no version-independent IR | Cross-version FL Plan intermediate representation | MLIR-based FL Plan compilation with version compatibility shims |
| No federated debugging capability | Ephemeral actors + no per-device logs prevent post-hoc analysis | Privacy-preserving anomaly attribution | Federated anomaly detection via aggregated error signatures; encrypted device health telemetry |
| Device scheduling ignores app usage recency | Multi-tenant scheduler uses simple FIFO queue | Usage-aware federated scheduling | Reinforcement learning-based device scheduler using app recency signals (not content) |

---

# 9. Novel Contribution Extraction

## Explicit Contribution Statements from This Paper

*"We propose [synchronous round-based FL protocol with eligibility gating] that enables [private model training] on [tens of millions of mobile devices] without moving raw user data off-device."*

*"We propose [actor-based FL server architecture with ephemeral in-memory aggregation] that improves [scalability and privacy] by [eliminating persistent per-device update logs while enabling elastic scaling from 10 to 10M devices]."*

*"We propose [pace steering] that improves [system reliability under diurnal device availability fluctuations] by [giving the server control over device reconnection timing without requiring persistent device state]."*

*"We propose [versioned FL plan generation with graph transformation] that improves [backward compatibility] by [automatically adapting computation graphs for older TensorFlow runtimes deployed on heterogeneous device fleets]."*

*"We propose [production integration of Secure Aggregation with hierarchical Aggregator actors] that improves [privacy of individual device updates] while [maintaining scalability to tens of thousands of simultaneous participants]."*

## 3–5 Novel Claim Templates for Your Own Research

1. **"We propose [MECHANISM] that achieves [PROPERTY] for federated learning on [CONSTRAINED DEVICE CLASS] by [KEY TECHNIQUE]."**
   - *Example fill*: "We propose battery-adaptive eligibility scheduling that achieves demographically representative participation for federated learning on low-end Android devices by learning per-device battery discharge models."

2. **"We propose [PRIVACY MECHANISM] that extends [EXISTING PROTOCOL] to [SCALE] while maintaining [GUARANTEE] by [EFFICIENCY IMPROVEMENT]."**
   - *Example fill*: "We propose tree-structured Secure Aggregation that extends pairwise SecAgg to 10,000+ simultaneous participants while maintaining per-device privacy by reducing per-device communication from O(n) to O(log n)."

3. **"We show that [EXISTING ASSUMPTION] breaks under [CONDITION] and propose [CORRECTIVE METHOD] to restore [PROPERTY]."**
   - *Example fill*: "We show that static round timeout configuration causes significant participation waste under diurnal availability variation, and propose an online adaptive timeout mechanism to restore efficient device utilization."

4. **"We propose [SYSTEM COMPONENT] that enables [NEW CAPABILITY] for production FL systems by [DESIGN CHOICE]."**
   - *Example fill*: "We propose a federated analytics layer that enables aggregate device health monitoring for production FL systems by computing differential-privacy-protected statistics without raw device data leaving the device."

5. **"We demonstrate that [FL TECHNIQUE] achieves [METRIC] comparable to [STRONG BASELINE] in a live deployment of [SCALE], proving [CLAIM]."**
   - *Example fill*: "We demonstrate that federated next-word prediction achieves top-1 recall comparable to fully server-trained RNNs in a live A/B deployment across 1.5M users, proving that production-quality models can be trained without data centralization."

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Work

- **Bias quantification and mitigation**: Characterize how eligibility constraints create demographic bias; develop algorithmic or systems countermeasures
- **Convergence time improvement**: Develop FL algorithms that can efficiently use more than hundreds of devices in parallel
- **Adaptive round configuration**: Use online ML to tune round timeout windows based on time-of-day device availability patterns
- **Better on-device scheduling**: Usage-aware multi-tenant scheduler that prioritizes apps the user actively uses
- **Bandwidth compression**: More aggressive compression (Konečný et al., Caldas et al.) and quantized model training
- **Federated Computation**: Generalize the system beyond ML to arbitrary MapReduce-like workloads
- **Federated Analytics**: Aggregate device statistics monitoring without raw data leaving devices

## 10.2 Missing Directions (Not Mentioned by Authors)

- **Federated continual learning**: How does the system handle concept drift when user behavior patterns change over time?
- **Cross-device FL across multiple OEMs**: The system is Android-only; a platform-agnostic FL runtime (e.g., via WebAssembly) would extend reach
- **Federated hyperparameter optimization**: Current configuration (learning rate, round size) is manually set; automated federated NAS or HPO would accelerate development
- **Adversarial robustness at scale**: Byzantine-robust aggregation for production deployments where a small fraction of devices may be compromised
- **Federated multi-modal learning**: Most FL literature focuses on single modality (text); production devices generate diverse data types
- **Energy-optimal FL scheduling**: Joint optimization of model quality, convergence speed, and aggregate battery consumption across the device fleet

## 10.3 Modern Extensions (2019 → 2025)

| Extension | Relevance to This Paper |
|---|---|
| Personalized FL (per-client fine-tuning, MAML) | Production FL serves one global model; personalization per user would require client-side adaptation layer |
| Federated Foundation Models (Fed-LLM) | The system can in principle run LLM fine-tuning; communication costs of billion-parameter updates need new compression |
| Privacy-preserving FL auditing | How can users verify that their data was actually used with FL's claimed privacy properties? |
| Edge-cloud continuum FL | System assumes binary device/cloud split; multi-tier FL (device → edge server → cloud) is an active direction |
| Differential Privacy with formal accounting | McMahan et al. 2018 DP is implemented; full privacy budget accounting over training rounds is an open integration problem |
| Federated unlearning | How to "forget" a user's contribution after training? Production FL has no mechanism for this |

## 10.4 LLM-Era Extensions

- **LLM fine-tuning via FL**: The Gboard use case already involves language models. RLHF/DPO fine-tuning via FL without centralizing human preference labels is a direct extension
- **On-device LLM adaptation**: Quantized LLMs (e.g., Gemma Nano, Phi-2) running on-device could use FL to personalize without full gradient updates — LoRA or adapter-based FL
- **Federated reward modeling**: Learning reward models for RLHF from user feedback signals (thumbs up/down, edits) via FL maintains privacy of preference data

---

# 11. How to Write a NEW Paper From This Work

## 11.1 Reusable Elements

| Element | How to Reuse |
|---|---|
| Round-based protocol design | Apply to non-mobile FL settings: IoT sensors, hospital networks, edge compute clusters |
| Actor model for FL server | Use as architectural template for your own FL system implementation |
| Pace steering concept | Adapt for any distributed system with fluctuating participant availability |
| State-sequence visualization for observability | Apply to any distributed learning system for debugging without PII |
| A/B experiment methodology for FL model evaluation | Industry-standard validation that FL models are production-quality |
| 130% overselection + straggler discarding | Apply as a default fault tolerance heuristic for synchronous distributed ML |
| Hierarchical aggregation for SecAgg scaling | Reuse for any cryptographic protocol with quadratic scaling |

## 11.2 What MUST NOT Be Copied

- The specific actor framework implementation (Google proprietary)
- The Android integration code (JobScheduler, SafetyNet attestation specifics)
- FL Plan generation from TensorFlow graphs
- Production operational data (belongs to Google)
- Specific TF graph transformation procedures for version compatibility

## 11.3 How to Design a Novel Extension

**Step 1** — Pick one weakness from Section 8 as your motivation

**Step 2** — Reformulate it as a research problem with a precise gap statement: "Existing production FL systems [do X] which causes [problem Y]; no prior work has addressed [gap Z]"

**Step 3** — Design a specific mechanism for gap Z. Make it a self-contained algorithmic or systems contribution that could be plugged into this paper's framework

**Step 4** — Define a clear evaluation: What metric improves? What is your baseline? (This paper's system is your baseline — use its reported numbers as the bar to beat)

**Step 5** — Implement using open-source FL frameworks (Flower, PySyft, TFF, OpenFL) as a substitute for the proprietary Google system

**Example Novel Extension Design**:
- **Problem**: 22% of device training rounds are wasted because round reporting windows are too narrow
- **Proposed method**: Online adaptive timeout prediction using a lightweight time-series model fit to each device's historical report-time distribution
- **Evaluation**: Simulate in Flower with diurnal availability patterns; measure wasted computation (rounds with <K completions) and round completion rate
- **Baseline**: Fixed timeout matching this paper's described behavior
- **Expected result**: 10–15% reduction in wasted computation, slight increase in round latency — present tradeoff curve

## 11.4 Minimum Publishable Contribution Checklist

- [ ] Novel component or mechanism not present in this paper's architecture
- [ ] Clear problem formulation with measurable gap
- [ ] Implementation that can be tested (even if simulated, not real production)
- [ ] At least one quantitative metric improved over this paper's reported numbers or this paper's described approach
- [ ] Ablation study showing each component of your contribution is necessary
- [ ] Discussion of limitations and failure cases of your approach
- [ ] Comparison to at least 2 alternative methods beyond this paper's baseline

---

# 12. Complete Paper Writing Template

## Abstract
**Purpose**: Communicate problem, approach, and key result in 150–250 words.
**Include**:
- The specific limitation of current FL systems that your paper addresses (1 sentence)
- Your proposed mechanism (1 sentence)
- Your key quantitative result (1–2 sentences)
- Brief implication for the field (1 sentence)

**Common mistakes**:
- Being too vague about the problem ("we improve FL efficiency")
- Not stating a concrete number (reviewers need a hook)
- Over-promising ("we solve the FL scalability problem")

**Reviewer expectation**: After the abstract, a reviewer should know exactly what you did and why it matters.

---

## Introduction
**Purpose**: Motivate the problem, establish the gap, preview the contribution.
**Include**:
- Paragraph 1: What FL is and why it matters for privacy (cite McMahan 2017, this paper)
- Paragraph 2: The specific problem you are solving — what fails in existing systems
- Paragraph 3: Why existing solutions (including this paper) are insufficient
- Paragraph 4: Your approach and contributions (numbered bullet list)
- Paragraph 5: Summary of experimental results
- Paragraph 6: Paper organization

**Common mistakes**:
- Too much background — introduction is not a tutorial
- Weak gap statement — must be precise and differentiated from all cited prior work
- Contributions listed without clarity on what is novel vs. what extends prior work

---

## Related Work
**Purpose**: Position your paper relative to the existing literature.
**Key areas to cover**:
1. FL system designs (this paper as primary reference)
2. FL algorithms (FedAvg, FedProx, SCAFFOLD, etc.)
3. Secure aggregation and privacy in FL
4. Distributed ML systems (parameter servers, MapReduce)
5. The specific sub-problem your paper addresses (e.g., device scheduling, compression, SecAgg scaling)

**Common mistakes**:
- Dumping a list of papers without explaining how yours differs
- Not citing this paper (Bonawitz 2019) as the primary systems baseline
- Missing the most recent 12 months of directly relevant work

**Reviewer expectation**: Reviewers will check that you know the state of the art and that your claimed novelty actually exists.

---

## Method
**Purpose**: Precisely describe what you built.
**Structure**:
- System/algorithm overview diagram (mandatory for systems papers)
- Component-by-component description
- Pseudocode for any algorithm (even if simple)
- Design choices and justification for each choice
- How your method integrates with existing FL infrastructure (e.g., this paper's framework)

**Common mistakes**:
- Missing implementation details that prevent reproducibility
- Overclaiming what your method does vs. what it assumes
- No diagram for systems papers — reviewers need the architecture

**Reviewer expectation**: Another researcher should be able to reimplement your method from this section alone.

---

## Theory (if applicable)
**Purpose**: Provide formal guarantees (convergence, privacy, complexity) for your method.
**Include**:
- Assumptions (state them explicitly — weaker assumptions = stronger paper)
- Main theorem or privacy claim
- Proof sketch (full proof in appendix)
- Comparison to prior theoretical results

**Common mistakes**:
- Proving a theorem under assumptions that are never satisfied in practice
- Burying assumptions in the appendix
- Not connecting the theory to the experiments

---

## Experiments
**Purpose**: Provide empirical evidence that your method works.
**Structure**:
- Experimental setup: datasets, model architectures, hyperparameters, hardware
- Baselines: include the approach described in this paper as a primary baseline
- Main results table
- Ablation study (required for systems papers)
- Failure analysis: when does your method fail?

**Common mistakes**:
- Choosing baselines that are easily beaten (cherry-picking)
- Reporting only the best hyperparameter setting without ablation
- Simulation results without any real-device validation (for FL systems papers, at least Flower or TFF simulation is expected)
- Not reporting variance (multiple runs with different seeds)

**Reviewer expectation**: At least one of your baselines must be competitive; your improvement must be statistically significant.

---

## Discussion
**Purpose**: Interpret results beyond what the tables show.
**Include**:
- Why your method works (mechanism explanation)
- Surprising results and what they mean
- Comparison to theory (do experiments match predictions?)
- Broader applicability beyond your evaluation setting

---

## Limitations
**Purpose**: Demonstrate awareness of your method's boundaries.
**Include**:
- What your method cannot do
- Conditions under which it fails or degrades
- Practical deployment challenges not addressed

**Reviewer expectation**: A clear limitations section increases reviewer trust — hiding limitations does the opposite.

---

## Conclusion
**Purpose**: One concise summary paragraph + two-sentence future work.
**Common mistakes**:
- Repeating the abstract verbatim
- Making promises about future work that won't happen
- Over-expanding conclusions beyond what results support

---

## References
- Cite this paper as: Bonawitz et al., 2019, "Towards Federated Learning at Scale: A System Design"
- All FL systems papers must cite McMahan et al. 2017 (FedAvg)
- All privacy-in-FL papers must cite Bonawitz et al. 2017 (SecAgg) and McMahan et al. 2018 (DP-FL)

---

# 13. Publication Strategy Guide

## Suitable Venues

| Type | Examples | Fit |
|---|---|---|
| **Top ML Conferences** | NeurIPS, ICML, ICLR | Suitable if algorithmic contribution or strong theoretical result is included |
| **Systems Conferences** | MLSys, OSDI, SOSP, EuroSys | Best fit for pure systems contributions; this paper's primary venue type |
| **Security/Privacy Conferences** | CCS, IEEE S&P, USENIX Security | Suitable if SecAgg scaling or formal DP accounting is the primary contribution |
| **FL Workshops** | FL@NeurIPS, FL@ICML | Lower bar; useful for position papers or early-stage work |
| **Applied ML Venues** | KDD, WWW, AAAI | Suitable for applications of FL (healthcare, NLP, recommendation) |

## Required Baseline Expectations

For any paper claiming to improve upon Bonawitz 2019 or FL systems:
- Must implement or simulate the round-based protocol described in this paper as a baseline
- Must use Flower, TFF, or OpenFL (open-source FL frameworks) as the implementation substrate
- Must report results at meaningful scale (at minimum hundreds of simulated devices; thousands preferred)
- For privacy claims: must include formal DP accounting or SecAgg protocol analysis

## Experimental Rigor Level

| Venue Type | Rigor Level |
|---|---|
| MLSys / OSDI | High — requires real hardware measurements, microbenchmarks, and system throughput numbers |
| NeurIPS / ICML | High — requires convergence results, ablations, statistical significance |
| FL Workshops | Moderate — simulation acceptable; novel idea + preliminary results sufficient |

## Common Rejection Reasons for FL Systems Papers

1. **Weak motivation**: "Existing FL systems are slow" without quantifying how slow, on what workload, and for what application
2. **No real-device experiment**: For systems papers, simulation-only results are often insufficient
3. **Missing comparison to this paper**: Reviewers familiar with Bonawitz 2019 will immediately ask "how does this compare?"
4. **No ablation**: Multiple components in your system but no evidence each one contributes
5. **Overclaimed privacy guarantee**: Formal privacy analysis of the full system is harder than claiming "we use SecAgg"
6. **No failure analysis**: System papers that only show success cases are viewed with suspicion

## Increment Needed for Acceptance

| Target Venue | Minimum Increment |
|---|---|
| NeurIPS / ICML / ICLR | New algorithm + convergence theory + strong empirical results at scale |
| MLSys | New system component with throughput/latency numbers + real deployment or large-scale simulation |
| USENIX Security | Formal security/privacy proof + efficient implementation + real-world feasibility demonstration |
| FL Workshop | One clearly novel idea + preliminary experimental support |

---

# 14. Researcher Quick Reference Tables

## 14.1 Key Terminology Table

| Term | Definition | First Appears |
|---|---|---|
| FL Population | Group of devices working on the same learning problem | Section 2.1 |
| FL Task | A specific computation (training or evaluation) for an FL population | Section 2.1 |
| FL Round | One iteration of the three-phase protocol (Selection → Configuration → Reporting) | Section 2.2 |
| FL Plan | Serialized TF graph + instructions for device + server side execution | Section 2.1 |
| FL Checkpoint | Serialized TF session state (global model weights) | Section 2.1 |
| Pace Steering | Server-side flow control that tells devices when to reconnect | Section 2.3 |
| Coordinator | Top-level persistent actor managing one FL population | Section 4.2 |
| Selector | Actor accepting/forwarding device connections | Section 4.2 |
| Master Aggregator | Ephemeral actor managing rounds of one FL task | Section 4.2 |
| Aggregator | Ephemeral actor collecting updates from a device subset | Section 4.2 |
| Example Store | On-device data repository (implements FL runtime API) | Section 3 |
| Straggler | Device that fails to report within the round's reporting window | Section 2.2 |
| Secure Aggregation (SecAgg) | MPC protocol making individual updates uninspectable to server | Section 6 |
| Diurnal Oscillation | Daily cycle in device availability (more devices at night when charging) | Section 2.3 |
| Thundering Herd | Problem where all devices check in simultaneously, overwhelming the server | Section 2.3 |
| Reservoir Sampling | Uniform random sampling algorithm used for device selection | Section 2.2 |
| Attestation | Android SafetyNet verification that a device is genuine | Section 3 |
| Versioned FL Plan | Plan variant compiled for compatibility with a specific TF runtime version | Section 7.3 |

---

## 14.2 Algorithm Flow Summary

| Step | Actor | Action |
|---|---|---|
| 1 | Device | JobScheduler detects idle + charging + WiFi conditions |
| 2 | Device | FL runtime opens bidirectional stream with Selector |
| 3 | Selector | Accepts or rejects connection; sends reconnect time if rejected |
| 4 | Coordinator | Instructs Selectors on how many devices to forward to Aggregators |
| 5 | Selector | Forwards accepted devices to Aggregators |
| 6 | Master Aggregator | Sends FL Plan + FL Checkpoint (global model) to each device |
| 7 | Device | Queries example store, runs local training per plan |
| 8 | Device | Sends weight delta back to Aggregator |
| 9 | Aggregator | Collects deltas; optionally runs SecAgg over its device group |
| 10 | Master Aggregator | Aggregates intermediate sums from all Aggregators |
| 11 | Coordinator | Writes updated global model to persistent storage |
| 12 | Coordinator | Begins Selection phase of next round (pipelined) |

---

## 14.3 Important Equations Summary

| Equation | Purpose | Key Variable |
|---|---|---|
| $\bar{n}_t = \sum_k n_k^t$ | Total training samples in round $t$ | $n_k^t$ = samples on device $k$ |
| $\Delta_t = \frac{1}{\bar{n}_t} \sum_k n_k^t \Delta_k^t$ | Weighted average update (FedAvg aggregation) | $\Delta_k^t$ = device $k$'s weight delta |
| $w_{t+1} = w_t + \Delta_t$ | Global model update | $w_t$ = current global weights |
| $w \leftarrow w - \eta \nabla L(w; b)$ | Local SGD step on device | $\eta$ = learning rate |
| SecAgg cost $\propto O(n^2)$ | Quadratic server-side computation for Secure Aggregation | $n$ = devices per SecAgg group |

---

## 14.4 Parameter Meaning Table

| Parameter | Role | Typical Value |
|---|---|---|
| Round target count $K$ | Number of device reports needed to complete a round | "A few hundred" |
| Selection count | Number of devices initially selected | $1.3 \times K$ |
| Drop-out rate | Fraction of selected devices that fail to report | 6%–10% |
| $k$ (SecAgg minimum) | Minimum group size for secure aggregation to run | Defined per FL task |
| Round timeout | Maximum time to wait for device reports before declaring round success/fail | Statically configured |
| Local epochs $E$ | Number of full passes over local data per round | Small number |
| Reconnect time window | Suggested reconnect interval given to rejected devices | Calculated by pace steering |

---

# 15. One-Page Master Summary Card

## Problem

How do you train machine learning models on the private data of millions of mobile phone users — without ever moving that data off their devices — while maintaining model quality, system reliability, privacy, and usability at production scale?

## Idea

Build a complete engineering system around synchronous round-based Federated Learning:
- Devices self-report availability when idle, charging, and on WiFi
- Server selects a representative subset, sends them the model and computation plan
- Devices train locally and report only weight updates (not data)
- Server aggregates updates with cryptographic protection (Secure Aggregation) so even it cannot see individual updates
- Process repeats until the global model converges

## Method

Five interlocking system components:
1. **Protocol**: 3-phase round (Selection → Configuration → Reporting) with pace steering for flow control
2. **Device**: Android FL runtime triggered by JobScheduler; data stays in example store; attestation prevents fake devices
3. **Server**: Actor-based architecture (Coordinator → Selector → Master Aggregator → Aggregator); fully ephemeral in-memory state; round pipelining
4. **Privacy**: Secure Aggregation integrated at Aggregator level (per group of ~hundreds); differential privacy hooks implemented
5. **Tooling**: FL Plans (serialized TF graphs with versioning); simulation environment; automated safety testing before deployment

## Results

- Deployed on ~10M daily active devices; up to 10,000 simultaneous participants
- Gboard next-word prediction: FL model matches server-trained RNN quality; outperforms both in live A/B experiments
- 75% of device sessions complete successfully; 6–10% dropout rate managed by 130% overselection buffer
- 4× diurnal variation in device availability managed by pace steering

## Weakness

- Hard eligibility criteria (charging + WiFi) creates systematic demographic bias
- ~22% of device training effort is wasted due to static round timeout windows
- Device selection is uniform (reservoir sampling) — does not account for data quality or diversity
- SecAgg quadratic scaling limits global privacy — hierarchical compromise needed

## Research Opportunity

- **Bias-aware device selection**: privacy-preserving quality signals for device prioritization
- **Adaptive round timeouts**: online learning from per-device historical report-time distributions to reduce wasted computation
- **Hierarchical SecAgg**: tree-structured MPC or lightweight HE for upper aggregation levels to achieve global privacy guarantees
- **Federated Analytics**: compute aggregate device health statistics without any data leaving devices
- **Federated unlearning**: mechanism to remove a user's contribution post-training

## Publishable Extension

*Adaptive Device Scheduling for Production Federated Learning*: Build a learned scheduler that replaces the static round timeout and uniform reservoir sampling with an online model that predicts each device's report-time distribution and data quality proxy. Show in simulation (Flower) that this reduces wasted computation (currently ~22% of training rounds) by at least 30% while maintaining or improving model convergence rate. Target: MLSys or FL@NeurIPS.

---

*End of Research Companion — 21_Bonawitz2019_SystemDesign_CS2.md*
