# SINDy: Sparse Identification of Nonlinear Dynamics — Research Companion

**Paper:** Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems.* PNAS, 113(15), 3932–3937.

---

## 0. Quick Paper Identity Card

| Attribute | Value |
|---|---|
| Problem Domain | Data-driven discovery of governing equations for nonlinear dynamical systems |
| Paper Type | Algorithmic / Method (with mathematical framing and applied demonstrations) |
| Core Contribution | A sparse regression framework (SINDy) that recovers the minimal set of nonlinear terms describing the time-evolution of a system from measured trajectories |
| Key Idea (1–2 lines) | Most physical dynamics are sparse in a well-chosen function basis; therefore use sparse regression against a library of candidate nonlinear terms to recover the equations of motion from data |
| Required Background | Ordinary/partial differential equations, linear regression, LASSO/ℓ1 sparse regression, numerical differentiation, basics of dynamical systems, POD / dimensionality reduction |
| Primary Baseline | Symbolic regression via genetic programming (Bongard & Lipson 2007; Schmidt & Lipson 2009) |
| Main Innovation Type | Reformulation + algorithmic (converts combinatorial model search into convex sparse regression) |
| Difficulty Level | Intermediate (concept is simple; subtleties live in basis selection, noise handling, derivative estimation) |
| Reproducibility Level | High (open code released; demos on canonical systems easy to reimplement) |

---

## 1. Research Context & Core Problem

### Exact Problem Formulation
- Given time-series measurements of a system's state `x(t) ∈ R^n`, recover the vector field `f` such that `dx/dt = f(x)`.
- The target is not just prediction, but an **interpretable symbolic model** whose terms can be read off and interpreted physically.

### Why the Problem Exists
- Many scientific domains (climate, neuroscience, ecology, epidemiology, finance) are data-rich but equation-poor.
- Standard machine learning fits correlations on an attractor but does not produce a compact dynamic law that extrapolates beyond observed regimes.
- Physical laws are typically simple (few terms), but the space of possible nonlinear terms is combinatorially huge.

### Historical and Theoretical Gap
- Symbolic regression (genetic programming) can find such equations but is expensive, scales poorly, and can overfit unless a Pareto front is maintained.
- Equation-free and empirical dynamic modeling methods can forecast but do not produce closed-form, interpretable equations.
- Compressed sensing proved that sparsity + convex optimization can replace combinatorial search, but had not been broadly applied to discovering dynamics until this work.

### Limitations of Previous Approaches
- **Symbolic regression / genetic programming:** combinatorial, heuristic, hard to scale, fragile with noise.
- **Equation-free / empirical dynamic modeling:** non-symbolic; does not yield interpretable equations.
- **Pure black-box ML (neural regression on dx/dt):** non-sparse, non-interpretable, over-parameterised.

### Contribution Category
- **Algorithmic** (new identification procedure)
- **Conceptual reframing** (dynamical-systems discovery as a sparse linear problem in a nonlinear feature space)
- **Empirical insight** (demonstration that a single convex sparse regression suffices for canonical systems)

### Why This Paper Matters
- It converts an open, decades-old problem (automated model discovery) into a scalable convex procedure.
- It rediscovers the non-trivial mean-field / slow-manifold model for cylinder wake — a result that took experts ~30 years.
- It seeded an entire sub-field: SINDy variants now exist for PDEs, control, stochastic systems, implicit dynamics, hybrid systems, and neural-network basis learning.

### Remaining Open Problems
- Automatic selection of measurement coordinates and function basis.
- Noise robustness when derivatives must be estimated numerically.
- Scaling the candidate library beyond polynomial/trigonometric hand-designs.
- Discovery for very high-dimensional PDE systems without first applying POD/DMD.
- Stochastic dynamics, non-Markovian memory terms, and regime-switching systems.
- Theoretical guarantees (recovery conditions in the style of RIP) for general nonlinear libraries.

---

## 2. Minimum Background Concepts

### 2.1 Dynamical system
- **Plain definition:** a rule telling how a state changes over time, `dx/dt = f(x)`.
- **Role here:** `f` is the unknown object the paper learns from data.
- **Why authors need it:** entire framework targets continuous-time (or iterated-map) dynamical systems.

### 2.2 Sparsity
- **Plain definition:** most entries of a solution vector are exactly zero.
- **Role here:** coefficients of most candidate terms should be zero because real physics uses few terms.
- **Why authors need it:** it is the central prior that makes the problem well-posed and interpretable.

### 2.3 Compressed sensing / LASSO
- **Plain definition:** solving underdetermined linear systems by preferring sparse solutions via ℓ1 penalty.
- **Role here:** justifies that exact sparse recovery is possible with convex optimization.
- **Why authors need it:** it supplies the algorithmic machinery and the theoretical intuition for recovery.

### 2.4 Feature library (candidate function basis)
- **Plain definition:** a hand-chosen set of nonlinear functions (constants, polynomials, trigs, exponentials) evaluated at the data.
- **Role here:** columns of the matrix `Θ(X)`; the answer is a sparse selection among them.
- **Why authors need it:** turns the nonlinear discovery problem into a *linear* regression over nonlinear features.

### 2.5 Numerical differentiation and total-variation regularisation
- **Plain definition:** estimating `dx/dt` from sampled `x(t)` while suppressing noise.
- **Role here:** required whenever derivatives are not directly measured.
- **Why authors need it:** sparse regression is sensitive to noisy targets.

### 2.6 Proper Orthogonal Decomposition (POD)
- **Plain definition:** an SVD-based way of extracting dominant spatial modes from high-dimensional simulation/experiment data.
- **Role here:** reduces fluid state (hundreds of thousands of grid points) to a handful of coordinates before applying SINDy.
- **Why authors need it:** SINDy's library grows combinatorially in state dimension.

### 2.7 Normal forms and bifurcations
- **Plain definition:** canonical reduced equations describing qualitative behaviour near parameter-triggered transitions (e.g., Hopf bifurcation).
- **Role here:** SINDy is extended to recover these by appending the bifurcation parameter as an extra coordinate.
- **Why authors need it:** real systems depend on control parameters; normal-form recovery is a stringent test.

---

## 3. Mathematical / Theoretical Understanding Layer

### 3.1 Dynamics Formulation

**Equation:** `dx(t)/dt = f(x(t))`

- **Intuition:** state's instantaneous velocity is a function of the current state; find that function.
- **Problem it solves:** gives a compact representation that predicts the system anywhere in state space.
- **Variables:**

| Symbol | Meaning |
|---|---|
| `x(t) ∈ R^n` | state vector (position, velocity, concentrations, modal amplitudes, …) |
| `f: R^n → R^n` | unknown vector field to be recovered |
| `n` | state dimension |

- **Assumption:** dynamics are deterministic and (mostly) Markovian in `x`.
- **Limitation:** excludes hidden states, strong delays, and non-Markov memory.

### 3.2 Data Matrices

Stacked state and derivative snapshots:
```
X  = [ x(t1) ; x(t2) ; … ; x(tm) ]         (m × n)
Ẋ  = [ ẋ(t1) ; ẋ(t2) ; … ; ẋ(tm) ]         (m × n)
```

- **Intuition:** tabulate measurements at many times.
- **Why:** convert a continuous ODE into rows of a regression problem.
- **Limitation:** `Ẋ` often must be estimated numerically and is noisy.

### 3.3 Candidate Function Library

`Θ(X) = [ 1, X, X^P2, X^P3, …, sin(X), cos(X), exp(X), … ]`

- **Intuition:** lay out every nonlinear term you suspect could appear.
- **What it solves:** linearises the parameter estimation — the unknown becomes a matrix of coefficients, not a symbolic expression tree.
- **Variables:**

| Symbol | Meaning |
|---|---|
| `Θ(X) ∈ R^{m×p}` | feature matrix of `p` candidate functions evaluated at the `m` snapshots |
| `X^Pk` | all degree-`k` monomials of state components |

- **Assumption:** the true vector field is sparse in this library.
- **Limitation:** library must be rich enough to contain the true terms but small enough for `m ≫ p`.

### 3.4 Core Sparse Regression Problem

`Ẋ = Θ(X) · Ξ + Z`

- **Intuition:** each column of `Ξ` picks the few features that reconstruct one component of `f`.
- **Variables:**

| Symbol | Meaning |
|---|---|
| `Ξ ∈ R^{p×n}` | coefficient matrix (sparse) |
| `ξ_k` | k-th column — active terms in `ẋ_k = f_k(x)` |
| `Z` | i.i.d. Gaussian noise matrix |

- **Reconstructed model:** `ẋ_k = Θ(x^T) · ξ_k`, where `Θ(x^T)` is the *symbolic* library for a single state.
- **Solution techniques:** LASSO (`min ||Ẋ − ΘΞ||₂² + λ||Ξ||₁`) or Sequentially Thresholded Least Squares (STLS) given in SI.
- **Assumption:** `m ≫ p`; noise magnitude modest relative to signal; library adequate.
- **Limitation:** no formal RIP-style guarantees because columns of `Θ(X)` are correlated (monomials).

### 3.5 Sequentially Thresholded Least Squares (the practical workhorse)

Simple iterative procedure:
1. Solve ordinary least squares `Ξ = Θ\Ẋ`.
2. Zero-out coefficients with magnitude below a threshold `λ`.
3. Refit least squares over the remaining (non-zero) columns.
4. Repeat steps 2–3 until the support stabilises.

- **Intuition:** a cheap, non-convex surrogate for LASSO that works well in practice for this problem.
- **Advantage:** faster than LASSO on big libraries; produces exact zeros.
- **Limitation:** threshold `λ` is a hyperparameter; sensitive to scaling of library columns.

### 3.6 Parameterised Dynamics

Append the bifurcation parameter as an extra "state":
```
ẋ = f(x ; μ)
μ̇ = 0
```
- **Intuition:** treat `μ` as a slowly-varying state; the library includes functions of `(x, μ)`.
- **Why:** allows normal-form discovery across a sweep of parameter values in one regression.
- **Limitation:** requires data at multiple `μ`-slices; bifurcation diagram must be adequately sampled.

### 3.7 Forcing/Control Generalisation

`ẋ = f(x, u(t), t)` — include actuation `u` as additional library inputs.

### Mathematical Insight Box
**The idea a researcher must remember:** Any unknown nonlinear dynamical law can be encoded as a **linear combination of nonlinear features**. Once written that way, the central prior "nature uses few terms" turns equation discovery into a convex (or near-convex) sparse regression, not a combinatorial symbolic search.

---

## 4. Proposed Method / Framework (MOST IMPORTANT)

### 4.1 Overall Pipeline

```
Measurements x(t)  ──▶  (optional) denoise / reduce via POD
                   ──▶  Estimate derivatives ẋ  (TV-regularised differentiation)
                   ──▶  Build feature library Θ(X)
                   ──▶  Solve sparse regression:  Ẋ ≈ Θ(X) Ξ
                   ──▶  Reconstruct symbolic model  ẋ_k = Θ(x) ξ_k
                   ──▶  Validate by forward simulation
```

### 4.2 Components and Flow

**Component A — Data acquisition**
- Record the state trajectory at uniform (or irregular) samples.
- *Why authors did this:* trajectory sampling is the minimum the method requires.
- *Weakness:* trajectories that explore only the attractor may miss transient manifolds (as noted for cylinder wake).
- *Improvement seed:* inject controlled perturbations to sample off-attractor regions; design optimal excitation experiments.

**Component B — Derivative estimation**
- Total Variation (TV) regularised derivative to denoise finite differences.
- *Why:* differentiation amplifies noise, and SINDy uses `ẋ` as regression target.
- *Weakness:* edge artefacts; hyperparameter tuning for the TV weight.
- *Improvement seed:* replace with Gaussian process derivatives; use implicit-form SINDy that avoids explicit derivatives; use weak/integral formulation (Weak-SINDy).

**Component C — Dimensionality reduction (for PDEs)**
- Project high-dimensional snapshots onto POD modes + a "shift mode" to capture transients.
- *Why:* prevents `p` from exploding and makes `m ≫ p` feasible.
- *Weakness:* POD is linear, may miss nonlinear manifolds.
- *Improvement seed:* use autoencoders or diffusion maps to learn nonlinear coordinates in which dynamics are sparse (precursor to SINDy-Autoencoder).

**Component D — Library construction `Θ(X)`**
- Hand-designed: polynomials up to order 5, optionally trigs/exps.
- *Why:* encodes physicist's prior about admissible terms.
- *Weakness:* if true dynamics are not sparse in this basis, SINDy fails silently (as in the glycolytic oscillator SI example).
- *Improvement seed:* adaptive/learned libraries; kernel-based libraries; symbolic discovery of a sparsifying basis.

**Component E — Sparse regression**
- STLS (default) or LASSO.
- *Why:* convex/near-convex, scalable, noise tolerant.
- *Weakness:* highly correlated feature columns break recovery guarantees; λ must be chosen.
- *Improvement seed:* Bayesian SINDy for uncertainty over supports; stability selection; ensemble SINDy.

**Component F — Model validation**
- Forward-simulate identified ODE; compare trajectory and attractor morphology.
- *Weakness:* chaotic systems diverge in trajectory metrics; paper emphasises attractor topology instead.
- *Improvement seed:* physics-informed validation (conservation laws, Lyapunov spectra, invariants).

### 4.3 Pseudocode-Style Summary

```
Input : time-series X, optional derivatives Ẋ
Output: sparse coefficient matrix Ξ

1. If Ẋ not given:  Ẋ ← TV_regularised_derivative(X)
2. Θ ← build_library(X, {1, poly_k, sin, cos, ...})
3. (optional) normalise columns of Θ
4. Ξ ← least_squares(Θ, Ẋ)                  # initial dense fit
5. repeat                                   # STLS loop
       small = |Ξ| < λ
       Ξ[small] = 0
       for each column k of Ẋ:
           active = indices where Ξ[:, k] ≠ 0
           Ξ[active, k] = least_squares(Θ[:, active], Ẋ[:, k])
   until support stable
6. return Ξ
```

### 4.4 Design Choices and Rejected Alternatives

| Choice | Alternative considered | Why adopted |
|---|---|---|
| Library of polynomials | Full symbolic regression | Far cheaper; convex; scalable |
| STLS | LASSO | STLS gives exact zeros, faster; LASSO falls back for big data |
| POD before SINDy | Direct fit in pixel space | Library would be infeasible on 10^5 grid points |
| TV-regularised differentiation | Finite differences | Heavy noise amplification otherwise |
| Parameter as extra state | Independent fits per μ | Enables normal-form discovery and smooth bifurcation curves |

---

## 5. Experimental Setup / Evaluation Design

### 5.1 Systems Tested
| System | Type | Purpose |
|---|---|---|
| Linear/nonlinear oscillators | ODE, low-dim | Sanity check |
| Lorenz-63 | Chaotic ODE | Noise robustness, attractor recovery |
| 2D Navier–Stokes, cylinder wake (Re=100) | PDE, high-dim | Demonstrates POD + SINDy on real fluids, recovers mean-field model |
| Logistic map with stochastic forcing | Discrete, parameterised | Bifurcation diagram recovery |
| Hopf normal form | ODE, parameterised | Normal-form recovery from noisy data |

### 5.2 Protocol
- Generate/collect trajectories via simulation.
- Add Gaussian measurement noise in several experiments; use TV-regularised derivative when derivatives are hidden.
- Fit SINDy; report identified coefficients and simulate forward from the identified model.
- Compare attractor geometry and coefficient values against ground truth.

### 5.3 Metrics and Rationale
- **Coefficient error** (e.g., within 0.03% on Lorenz, 0.1% on logistic): tests *symbolic* accuracy.
- **Attractor overlay**: visual/topological comparison — appropriate for chaos where point-wise trajectory error is meaningless.
- **Bifurcation diagram fidelity**: tests model behaviour across parameter regimes unseen in training.

### 5.4 Baselines
- Implicit comparison to symbolic regression (GP) by citation and runtime arguments; no head-to-head timing plots in the main paper.

### 5.5 Hyperparameters
- Polynomial order (≤5 for Lorenz), threshold `λ`, TV regularisation strength, POD truncation rank (3 for cylinder wake).

### Experimental Reliability Analysis

**Trustworthy**
- Lorenz and logistic map: ground-truth equations known; coefficient recovery gives a clean yardstick.
- Cylinder wake: independent literature derivation (Noack 2003) for the mean-field model provides an oracle.

**Questionable / Fragile**
- Absence of a quantitative baseline against genetic-programming symbolic regression.
- Limited noise-sweep analysis in the main text (lives in SI).
- Reliance on a "fortunate" choice of basis — not benchmarked against adversarially-chosen coordinates in the main body.
- No uncertainty quantification on `Ξ`.

---

## 6. Results & Findings Interpretation

### 6.1 Main Outcomes
- Lorenz equations reconstructed with coefficient error ≤0.03% even with substantial noise.
- Cylinder wake: SINDy recovers quadratic nonlinearities of Navier–Stokes plus the parabolic slow manifold — reproducing a ~30-year-in-the-making physical model automatically.
- Logistic map and Hopf normal form: parameterised dynamics and the full bifurcation diagram recovered within 0.1%.

### 6.2 Performance Trends
- More noise ⇒ wider STLS threshold; correct support still recovered within reported ranges.
- Training data that only sample the attractor ⇒ method may choose a wrong (cubic) fit for the cylinder wake; including transients to the attractor is essential.

### 6.3 Failure Cases (acknowledged)
- Lorenz in nonlinear coordinate transform — dynamics no longer sparse.
- Glycolytic oscillator — true model has divisions and rationals not in the chosen polynomial library.

### 6.4 Unexpected Observations
- Forecasting individual chaotic trajectories is intrinsically hopeless; attractor reproduction is the right success metric.
- "Failure" is itself useful — it signals that the current coordinates/basis are wrong, providing diagnostic feedback.

### 6.5 Statistical Meaning
- No error bars / repeated trials in main text; coefficient-agreement percentages suggest deterministic evaluation.

### Publishability Strength Check

| Result | Publication-grade? | Needs stronger validation? |
|---|---|---|
| Lorenz symbolic recovery | Yes | — |
| Cylinder wake mean-field recovery | Yes (flagship result) | Broader Re-sweep would strengthen |
| Logistic map bifurcation | Yes | — |
| Hopf normal form | Yes | Noise-level ablation missing from main text |
| Claim of general applicability | Partial | Needs systematic stress-tests on poorly-chosen bases |

---

## 7. Strengths – Weaknesses – Assumptions

### 7.1 Technical Strengths
| # | Strength |
|---|---|
| 1 | Converts combinatorial symbolic search into a convex-like linear algebra problem |
| 2 | Scales with Moore's-law-compatible solvers |
| 3 | Produces interpretable, simulatable symbolic models |
| 4 | Generalises to parameterised, forced, and PDE systems |
| 5 | Failure modes are diagnostic rather than silent when paired with validation |
| 6 | Minimal compute relative to genetic-programming baselines |

### 7.2 Explicit Weaknesses
| # | Weakness |
|---|---|
| 1 | Requires a good guess at the function library |
| 2 | Requires good measurement coordinates; fails in "bad" bases |
| 3 | Sensitive to noise in derivative estimates |
| 4 | Does not handle hidden variables / partial observability |
| 5 | No formal recovery guarantees (correlated library columns) |
| 6 | Hyperparameter `λ` set heuristically |

### 7.3 Hidden Assumptions
| # | Assumption |
|---|---|
| 1 | Dynamics are Markovian in measured `x` |
| 2 | True `f` is sparse in the chosen basis |
| 3 | Noise is Gaussian, additive, state-independent |
| 4 | `m ≫ p` — enough snapshots to over-determine the regression |
| 5 | Sampling covers enough of state space (including transients for accurate manifold recovery) |
| 6 | Underlying system is (approximately) autonomous on the data horizon |

---

## 8. Weakness → Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| Library design is manual | Prior on term types is human-chosen | Automated library expansion / pruning | Reinforcement learning over basis sets; dictionary learning |
| Bad coordinate system ⇒ no sparsity | Nonlinear phenomena may not be sparse in raw states | Learn coordinates in which dynamics become sparse | SINDy-Autoencoders; contrastive manifold learning; Koopman eigenfunction search |
| Noise-sensitive numerical differentiation | Finite differences amplify high-freq noise | Avoid derivatives entirely | Weak/Integral-form SINDy; Gaussian-process derivatives; neural-ODE-style loss |
| No uncertainty on coefficients | Point estimate via STLS | Probabilistic equation discovery | Bayesian SINDy with spike-and-slab priors; bootstrap / stability selection |
| Fails for rational / implicit dynamics | Regression assumes explicit ODE form | Implicit-form discovery | SINDy-PI; algebraic-geometry-based null-space methods |
| Hyperparameter `λ` ad hoc | No principled selection rule | Cross-validated / information-criterion selection | AIC/BIC weighted SINDy; knee-point detection |
| Partial observability | Assumes full state | Latent-state equation discovery | Delay-coordinate embedding + SINDy; Hankel-DMD fusion |
| Scales poorly in dimension | Library grows combinatorially | Scalable discovery for high-dim systems | Tensor-structured libraries; randomised sketching |
| Stationarity assumed | Real systems drift | Non-stationary / streaming SINDy | Online sparse filters; concept-drift detection |
| No theory guarantees | Library columns correlated | Recovery-condition theory | Coherence bounds for polynomial dictionaries |

---

## 9. Novel Contribution Extraction

### Template Claims You Could Adapt

1. "We propose **Weak-form Bayesian SINDy**, which improves **robustness to measurement noise and uncertainty in sparsity pattern** by **replacing point-estimate STLS with variational spike-and-slab inference over a weak-form regression**."
2. "We propose **SINDy-Koopman**, which improves **basis selection** by **learning Koopman-invariant coordinates via a neural encoder, then applying sparse regression in the linearised space**."
3. "We propose **Adaptive-Library SINDy** that improves **library completeness** via **online expansion/pruning guided by residual analysis and information-theoretic scoring**."
4. "We propose **SINDy-for-Stochastic-Control**, extending equation discovery to **drift–diffusion systems under active feedback** using **generalised Fokker–Planck regression with ℓ1 priors**."
5. "We propose **Implicit-Graph-SINDy**, which improves **discovery for rational and coupled-ODE networks** by **combining implicit null-space recovery with sparse message passing over a candidate interaction graph**."

### Explicit Novel Contribution Framing (use in abstracts)
- "We reformulate the identification of governing equations as **X-regularised regression over a Y-augmented library**, removing the need for **Z**, and demonstrate **A% improvement in B on benchmarks C and D**."

---

## 10. Future Research Expansion Map

### Author-Suggested Future Work
- Automated selection of coordinates and basis.
- Integration with dimensionality reduction and machine-learning feature extraction.
- Extension to time-delay coordinates for partial observations.
- Inclusion of external forcing / control signals.

### Missing Directions (not explicitly covered)
- Formal recovery theory for correlated nonlinear dictionaries.
- Uncertainty quantification on discovered equations.
- Online / streaming / non-stationary variants.
- Discovery of conservation laws, symmetries, and Hamiltonians directly.
- Causal SINDy that distinguishes cause from correlation.

### Modern Extensions Already in the Literature (seeds for your own variants)
- **SINDy-PI** (Implicit): rational / implicit ODEs.
- **Weak-SINDy** (Reinbold et al.): noise-robust weak-form derivatives.
- **SINDy-Autoencoder** (Champion et al.): jointly learn coordinates and equations.
- **SINDyC** (Brunton, Proctor, Kutz): dynamics with control.
- **PDE-FIND** (Rudy et al.): discovery of PDEs.

### LLM-Era / Emerging Extensions
- LLM-guided symbolic library proposal (LLM suggests which functions to include).
- Neural-Symbolic hybrids: neural-ODE posterior + symbolic sparsification head.
- Foundation-model surrogates for physical reasoning feeding a SINDy distillation step.
- RAG-based discovery: retrieve analogous prior equations from a paper corpus and seed the library.

### Cross-Domain Combinations
- SINDy + graph neural networks for multi-agent/ecological networks.
- SINDy + reinforcement learning for model-based control that discovers its own plant model.
- SINDy in neuroscience for neural-mass models; in epidemiology for compartmental-model discovery.

---

## 11. How to Write a NEW Paper From This Work

### Reusable Elements
- The **pipeline pattern** (data → library → sparse regression → validate).
- The **attractor-fidelity** evaluation style for chaotic systems.
- The **coefficient-error percentage** benchmark against known systems.
- The **"discover a classical model"** rhetorical arc (e.g., rediscover a textbook equation).
- The **parameterised-system trick** (append `μ` as extra state).

### What MUST NOT Be Copied
- Identical canonical benchmark set (Lorenz + cylinder wake + logistic + Hopf) without adding a new system or twist.
- The exact STLS algorithm verbatim — your method must differ algorithmically or theoretically.
- The wording and figures of the cylinder-wake narrative.
- Unmodified hand-designed polynomial library without justification.

### How to Design a Novel Extension
1. **Pick a weakness** from §7.2 or §8.
2. **Design one concrete mechanism** that removes it (weak form, Bayesian, learned basis, graph prior, control input, …).
3. **Define a failure mode in original SINDy** your method fixes, and construct a benchmark that exposes it.
4. **Prove one theoretical property** (identifiability, consistency, sample complexity, or noise bound).
5. **Show one real dataset** beyond synthetic — fluids, neural recordings, epidemic curves, climate reanalysis, or robotics sensor logs.
6. **Compare quantitatively** to (i) vanilla SINDy, (ii) a neural baseline, (iii) the most recent SINDy variant in your niche.

### Minimum Publishable Contribution Checklist
- [ ] One clearly-named algorithmic contribution.
- [ ] One theoretical result (even a mild bound).
- [ ] ≥3 benchmark systems, at least one real / experimental.
- [ ] Head-to-head against ≥2 prior SINDy-family baselines.
- [ ] Ablations on: noise, library size, sample count, hyperparameter.
- [ ] Open-source code and dataset release.
- [ ] Discussion of failure modes and limitations.

---

## 12. Complete Paper Writing Template

### 12.1 Abstract (≤250 words)
- **Purpose:** one-paragraph pitch.
- **Include:** problem, gap, mechanism, key result, significance.
- **Common mistakes:** listing methods without a result; vague novelty.
- **Reviewer expectation:** understands novelty and main number within 60 seconds.

### 12.2 Introduction
- **Purpose:** motivate problem and contribution.
- **Include:** data-rich / equation-poor domains, limitations of current equation discovery, teaser of your improvement, bullet-list of contributions, paper roadmap.
- **Common mistakes:** history-dump without landing on the gap; no explicit contributions list.
- **Reviewer expectation:** clear "before / after" your method.

### 12.3 Related Work
- **Purpose:** position inside SINDy/equation-discovery family.
- **Include:** symbolic regression, compressed sensing, SINDy variants (Weak, PI, Autoencoder, control), neural ODEs.
- **Common mistakes:** chronological listing; missing nearest competitors.
- **Reviewer expectation:** a **matrix** that places your work along method axes.

### 12.4 Preliminaries / Background
- **Purpose:** make paper self-contained.
- **Include:** dynamical systems, sparse regression, any specialised tool (Koopman, weak form, GP derivatives).
- **Common mistakes:** textbook-length expositions.
- **Reviewer expectation:** notation fixed once and reused throughout.

### 12.5 Method
- **Purpose:** describe your algorithm.
- **Include:** pipeline figure, objective function, algorithm box, hyperparameter table, complexity discussion.
- **Common mistakes:** skipping the "why" behind design choices.
- **Reviewer expectation:** reproducible from the text alone.

### 12.6 Theory (optional but strong)
- **Purpose:** guarantee correctness or sample complexity.
- **Include:** assumptions, theorem statement, proof sketch, corollaries.
- **Common mistakes:** assumptions no real system satisfies.
- **Reviewer expectation:** plausible assumptions + clean statement.

### 12.7 Experiments
- **Purpose:** empirical validation.
- **Include:** benchmarks, baselines, metrics, ablations, compute budget.
- **Common mistakes:** cherry-picked seeds; missing variance.
- **Reviewer expectation:** error bars, fair baselines, reproducible configs.

### 12.8 Discussion
- **Purpose:** interpret results.
- **Include:** when the method works, when it doesn't, diagnostic behaviour.
- **Common mistakes:** over-claiming generality.

### 12.9 Limitations
- **Purpose:** transparent failure modes.
- **Include:** dependency on basis, noise ceiling, dimension limits.
- **Common mistakes:** hiding limitations in supplementary only.

### 12.10 Conclusion
- **Purpose:** crisp restatement + forward pointer.
- **Include:** 1-paragraph recap, 1-paragraph open questions.

### 12.11 References
- Canonical SINDy lineage + the specific method you extend + at least one venue-aligned paper (fluids → JFM, ML → NeurIPS, applied math → SIAM).

---

## 13. Publication Strategy Guide

### Suitable Venues
| Tier | Venue | Angle |
|---|---|---|
| Flagship-science | PNAS, Nature Comm. | Applied discovery on a real dataset with novel science |
| Applied math | SIAM J. Applied Dyn. Syst., J. Comp. Physics, Physica D | Theoretical extensions, PDE discovery |
| ML | NeurIPS, ICML, ICLR, AAAI | Learned-basis or probabilistic SINDy |
| Fluids / physics | J. Fluid Mech., Phys. Rev. Fluids | Turbulence / wake applications |
| Controls | IEEE TAC, Automatica, L4DC | SINDy with control / identification |
| Bio / medical | PLOS Comp. Biol., Chaos | Neural / epidemic / physiological dynamics |

### Required Baseline Expectations
- Vanilla SINDy (this paper).
- ≥1 modern SINDy variant in your niche (Weak-SINDy, SINDy-PI, SINDy-Autoencoder).
- ≥1 neural baseline (Neural ODE, PINN, or DMD-style).

### Experimental Rigor Level
- Minimum: 3 synthetic + 1 real dataset, noise sweeps, seed variance reported, ablations.
- Strong: theorem with sample-complexity bound, failure-mode analysis, compute comparison.

### Common Rejection Reasons
- Incremental: "just another SINDy variant" with no new benchmark or theory.
- Weak baselines: omitting recent SINDy descendants.
- No real-world demonstration.
- Overly narrow benchmark set (Lorenz-only is no longer publishable as standalone).
- Missing code/data release.

### Increment Needed for Acceptance
- Typically **one of**: (i) a new problem class SINDy didn't cover, (ii) a theoretical property previously absent, (iii) a meaningful real-world discovery, (iv) a substantial robustness/efficiency improvement with clear quantitative wins.

---

## 14. Researcher Quick Reference Tables

### 14.1 Key Terminology
| Term | Meaning |
|---|---|
| SINDy | Sparse Identification of Nonlinear Dynamics |
| `Θ(X)` | Feature library matrix |
| `Ξ` | Sparse coefficient matrix |
| STLS | Sequentially Thresholded Least Squares |
| POD | Proper Orthogonal Decomposition |
| TV derivative | Total-variation-regularised numerical differentiation |
| Normal form | Minimal canonical equation near a bifurcation |
| Slow manifold | Low-dimensional attracting surface where transient modes equilibrate fast |
| Mean-field model | Reduced model separating fast and slow scales (cylinder wake) |

### 14.2 Important Equations Summary
| Eq. | Symbolic Form | Role |
|---|---|---|
| Dynamics | `ẋ = f(x)` | Object to discover |
| Regression | `Ẋ = Θ(X) Ξ + Z` | Linear problem in nonlinear features |
| Recovered model | `ẋ_k = Θ(x^T) ξ_k` | Symbolic ODE for each state |
| Parameterised | `ẋ = f(x ; μ)`, `μ̇ = 0` | Normal-form/bifurcation discovery |
| Forced | `ẋ = f(x, u, t)` | Control/forcing extension |
| Mean-field (cylinder) | `ẋ = μx − ωy + Axz`, `ẏ = ωx + μy + Ayz`, `ż = −λ(z − x²−y²)` | Target to reverse-engineer |

### 14.3 Parameter Meaning Table
| Parameter | Meaning | Typical choice |
|---|---|---|
| `p` | # candidate functions | 10²–10³ |
| `m` | # snapshots | ≫ `p` |
| `λ` | STLS sparsity threshold | grid-search or CV |
| `poly_order` | Max polynomial degree | 2–5 |
| `r` (POD rank) | Reduced dimension | 3–20 depending on flow |
| `η` | Noise magnitude in `Z` | problem-specific |

### 14.4 Algorithm Flow Summary
```
1. Collect X, estimate Ẋ
2. Build Θ(X)
3. Solve Ξ with STLS / LASSO
4. Extract symbolic ODE
5. Simulate forward and validate
6. If failure → revisit coordinates / basis (diagnostic loop)
```

---

## 15. One-Page Master Summary Card

| Field | Summary |
|---|---|
| **Problem** | Recover governing ODE/PDE of a nonlinear system from measured time-series |
| **Idea** | Most dynamics are sparse in a well-chosen nonlinear feature basis |
| **Method** | Build a library Θ(X) of candidate terms; solve a sparse regression `Ẋ = Θ(X) Ξ` with STLS or LASSO |
| **Results** | Exact recovery of Lorenz, logistic, Hopf, and cylinder-wake mean-field model with tiny coefficient error and strong noise robustness |
| **Weakness** | Needs a good coordinate system and function basis; sensitive to noisy derivatives; no formal recovery theory |
| **Research Opportunity** | Learn the basis/coordinates automatically; weak-form derivatives; Bayesian uncertainty; implicit, stochastic, controlled, or streaming SINDy; LLM-guided library proposal |
| **Publishable Extension** | Provide an algorithmic twist that demonstrably removes one SINDy limitation, back it with a theorem or ablation, and validate on one real-world system beyond the canonical benchmarks |
