# 34 — The Cognitive Cost of Convenience: A Review on the Decline of Human Intelligence in the Age of AI

> Research Companion (Learning + Writing + Publication Blueprint)
> Paper classification: **Survey / Conceptual (Interdisciplinary Literature Review)**
> Authors: Sowmya Sreepadacharya (Vijaya College, Bengaluru City University). Published in IJFMR via AIMAR25 National Conference.

---

## 0. Quick Paper Identity Card

| Attribute | Value |
|---|---|
| Problem Domain | Human–AI Interaction · Cognitive Science · Educational Technology · AI Ethics |
| Paper Type | Narrative / Interdisciplinary Review (conceptual synthesis, not empirical) |
| Core Contribution | Synthesises scattered psychology, neuroscience, education and ethics evidence to argue that heavy AI use causes measurable cognitive offloading, skill atrophy, and dependency |
| Key Idea (1–2 lines) | Convenience from AI is not free — it trades short-term efficiency for long-term decline in memory, creativity, critical thinking, and decision vigilance, unless integration is designed ethically |
| Required Background | Basics of cognitive psychology (working memory, executive function), neuroplasticity, generative AI (LLMs), educational assessment, AI ethics |
| Primary Baseline | Classical cognitive offloading literature (Sparrow et al. 2011 "Google Effect") and recent AI-in-education EEG studies (MIT Media Lab 2025, Kosmyna et al. 2025) |
| Main Innovation Type | Conceptual framing + interdisciplinary consolidation (no new dataset, no new algorithm) |
| Difficulty Level | Low-to-Medium (conceptual); High if extended to rigorous empirical follow-up |
| Reproducibility Level | Low — no data, no code, no protocol; claims rest on cited third-party studies |

---

## 1. Research Context & Core Problem

### 1.1 Exact Problem Formulation
- Does routine reliance on AI tools (search engines, generative models like ChatGPT, decision-support systems) cause a measurable decline in core human cognitive capacities?
- The paper poses this as a **review question**, not a hypothesis test, and asks which mechanisms (offloading, atrophy, dependency) mediate the effect.

### 1.2 Why the Problem Exists
- AI is now embedded in education, workplaces, and personal life.
- Users increasingly delegate thinking tasks (recall, drafting, analysis) to machines.
- Cognitive effects accumulate slowly and are invisible in short benchmarks, so they go unmonitored.

### 1.3 Historical / Theoretical Gap
- Earlier work (Sparrow 2011) showed only a memory-offloading effect for web search.
- The rise of generative AI changes the picture: AI now offloads not just *storage* but *composition, reasoning, and judgment*.
- No unified framework had been proposed linking neuroscience findings, classroom observations, and ethical risk into one coherent picture.

### 1.4 Limitations of Previous Approaches
- Siloed research: psychology papers ignore neuroscience, neuroscience ignores education, ethics ignores both.
- Most prior studies are cross-sectional and short; no longitudinal cognitive trajectories.
- Pre-LLM literature no longer applies cleanly to ChatGPT-style *generative* offloading.

### 1.5 Contribution Category
- **Conceptual / Survey** contribution.
- Provides taxonomy (offloading → skill decay → dependency).
- Supplies interdisciplinary argument scaffolding for future empirical work.

### Why This Paper Matters
- Offers a compact, citable framing that educators, policy-makers, and HCI designers can reference.
- Names a phenomenon ("cognitive cost of convenience") that can structure future studies.
- Bridges disciplines — useful as an entry point for researchers planning empirical follow-ups.

### Remaining Open Problems
- No quantitative effect sizes aggregated across studies (no meta-analysis).
- No age-group stratification (children vs. adults vs. elderly).
- No dose–response curve: how much AI use is harmful vs. beneficial?
- No task-type taxonomy: which tasks degrade skills, which preserve them?
- No reversal / recovery study: is the decline permanent or retrainable?

---

## 2. Minimum Background Concepts

| Concept | Plain Definition | Role in Paper | Why Authors Needed It |
|---|---|---|---|
| Cognitive Offloading | Pushing a mental task onto an external aid (notebook, calculator, AI) | Central mechanism linking AI use → skill loss | Explains *how* convenience translates into decline |
| Skill Atrophy / Skill Decay | Loss of a trained ability when it is not exercised | Outcome variable | Names the long-term cost |
| Working Memory | Short-term holding + manipulation of information | Cited as weakened by AI reliance | Connects behaviour to brain-level measurement |
| Executive Function | Planning, inhibition, flexible reasoning (prefrontal cortex) | EEG evidence tied here | Provides neuroscience anchor |
| Neuroplasticity | Brain's ability to rewire with use / disuse | Justifies why behaviour can physically change the brain | Supports "use it or lose it" argument |
| Metacognition | Thinking about one's own thinking (monitoring, reflection) | Claimed to be bypassed when AI drafts for the user | Explains educational harm |
| Google Effect | Tendency to remember *where* to find information rather than the information itself | Historical precedent | Shows the phenomenon pre-dates LLMs |
| Generative AI | AI that produces new text / images (e.g., ChatGPT) | Newest source of offloading | Differentiates current era from search-era studies |

---

## 3. Mathematical / Theoretical Understanding Layer

> This paper is **not math-heavy**. It contains no equations, no theorems, and no formal models. The "theory" layer is therefore a **conceptual model** that the reader can extract and formalise.

### 3.1 Extracted Conceptual Model (our formalisation — useful for follow-up work)

Let:
- `U` = frequency/intensity of AI use on a task class
- `C` = native cognitive competence on that task class
- `k_offload` = offloading rate (how much of the task is handed to AI)
- `k_atrophy` = rate of decay when skill is unused
- `k_practice` = rate of gain when skill is practised

Implied informal dynamic:

  dC/dt  ≈  k_practice · (1 − k_offload · U)  −  k_atrophy · k_offload · U

### Mathematical Insight Box
The paper's core claim can be turned into a **deliberate-practice law**: *competence grows with engaged practice and decays with delegated practice.* A researcher who wants to publish empirical follow-up should measure `k_offload`, `k_atrophy`, and `k_practice` directly — the paper does not, which is the biggest empirical gap.

### 3.2 Assumptions Hidden in the Argument
- Cognitive skills behave like muscles (practice–atrophy symmetry). Not universally true — some skills (language, riding a bike) are highly durable.
- Short-term EEG changes predict long-term cognitive outcomes. Open question.
- "Using AI" is a single construct. In reality, *how* one uses AI (passive accept vs. critique-and-revise) may matter more than *whether* one uses it.

---

## 4. Proposed Method / Framework

Because this is a review, the "method" is the **argument pipeline**, not an algorithm. We reconstruct it as a flow.

### 4.1 Overall Pipeline (reconstructed)

1. Define **Human Intelligence** (fluid reasoning, memory, creativity, emotional insight).
2. Define **AI** (speed + scale, but lacking context and ethics).
3. Introduce **Cognitive Offloading** as the bridge mechanism.
4. Survey evidence in four domains:
   - Cognitive psychology (Google Effect, skill decay)
   - Education (shallow learning, soulless essays)
   - Workplace decision-making (skill fade in analysts, clinicians)
   - Neuroscience (EEG / fMRI correlates)
5. Aggregate into **Key Findings** (8 bullet claims).
6. Extract **Ethical Implications** (autonomy, creativity, bias).
7. Offer **Counterarguments** (AI as scaffold, neurodiverse support).
8. Close with **Recommendations** (education, design, resilience programs).

### 4.2 Step-by-Step Breakdown

**Step A — Conceptual Setup**
- Why authors did this: readers span disciplines; shared vocabulary prevents ambiguity.
- Weakness: definitions are brief and informal (e.g., "human intelligence" reduced to four traits).
- Research-idea seed: build a *measurable operationalisation* of each trait with a validated psychometric instrument.

**Step B — Evidence Aggregation (narrative, not systematic)**
- Why: narrative reviews are faster and better for framing a field.
- Weakness: no PRISMA-style search, no inclusion/exclusion criteria, no risk-of-bias rating.
- Research-idea seed: run a **systematic review + meta-analysis** on "AI use and cognitive outcomes" with effect-size aggregation.

**Step C — Mechanism Labelling**
- Why: gives readers a causal vocabulary (offloading → atrophy → dependency).
- Weakness: mechanisms are asserted, not tested.
- Research-idea seed: design **mediation analyses** — does offloading mediate the link between AI exposure and skill decline?

**Step D — Ethical Extension**
- Why: broadens appeal beyond cognitive science.
- Weakness: ethical claims are high-level and unranked.
- Research-idea seed: build an **Ethical Risk Index** for AI-in-education that prioritises which risks matter most by domain.

**Step E — Recommendations**
- Why: gives practitioners takeaways.
- Weakness: recommendations are aspirational, not tested interventions.
- Research-idea seed: **randomised controlled trials** comparing "AI literacy curriculum" vs. "standard curriculum" on long-term cognitive outcomes.

### 4.3 Pseudocode-style Argument Logic
```
INPUT: claim "AI may cause cognitive decline"
FOR domain in {psychology, education, workplace, neuroscience}:
    fetch 1–3 representative studies
    paraphrase study → link to mechanism (offloading | atrophy | dependency)
    accumulate evidence
SYNTHESISE → 8 key findings
RAISE counterarguments (scaffolding, neurodiversity, democratisation)
DERIVE recommendations (education, design, resilience)
OUTPUT: call for ethical, balanced integration
```

### 4.4 Why Alternatives Were Rejected (implicit)
- Systematic review was not chosen — likely due to scope + time of a conference review.
- Empirical study was not chosen — requires IRB, participants, longitudinal data.
- Position paper alone was not chosen — weaker than evidence-backed synthesis.

---

## 5. Experimental Setup / Evaluation Design

The paper is a literature review and **does not run experiments**. What follows is a reconstruction of the *review protocol*, with its weaknesses, so that a new researcher can either replicate or upgrade it.

### 5.1 De-facto Protocol
- Sources: named studies (Sparrow 2011; Bostrom & Yudkowsky 2014; MIT Media Lab 2025; Kosmyna et al. 2025; Edinburgh 2024; OECD 2023; Small et al. 2023; Sternberg 2024; Westfall 2024).
- Selection logic: not disclosed.
- Synthesis: narrative; grouped by domain.
- Metrics: none — claims are qualitative.
- Baselines: none — no comparator against which to judge "decline".

### 5.2 What a Proper Evaluation Would Look Like
- **Longitudinal cohort**: 6–24 month AI-use logs + quarterly cognitive testing (working memory n-back, Raven matrices, creativity AUT).
- **Neural correlates**: EEG pre/post, ideally MRI for cortical-thickness tracking.
- **Dose–response**: hours/week of generative-AI use as the independent variable.
- **Randomisation**: AI-assisted vs. AI-free group on the same curriculum.

### Experimental Reliability Analysis
**Trustworthy elements in the paper**
- The mechanism ladder (offloading → atrophy → dependency) is intuitive and consistent with mainstream cognitive psychology.
- The MIT EEG reference, if independently replicated, strongly supports the narrative.

**Questionable elements**
- Heavy reliance on a single, not-yet-peer-reviewed MIT report ("Unpublished Technical Report").
- Forbes article and OECD PISA used as evidence (mixed source quality).
- No effect sizes, no confidence intervals, no number-needed-to-train statistics.
- Conclusions are universal in tone ("Students exhibit…") but the evidence base is narrow and recent.

---

## 6. Results & Findings Interpretation

### 6.1 Main Outcomes (paraphrased)
- Frequent AI use is correlated with reduced memory retention and lower engagement.
- Generative-AI users show reduced EEG activity in creativity- and semantics-related regions.
- AI-written work is described by instructors as lacking originality.
- Professionals using AI decision-support detect fewer anomalies in high-stakes tasks.
- Neural engagement is strongest in AI-free conditions.

### 6.2 Performance Trends
- Across domains, the direction of effect is uniform (AI use → weaker native performance), which the author treats as convergent evidence. A meta-analytic lens would ask whether effect sizes are small-but-consistent or large-and-heterogeneous.

### 6.3 Failure Cases / Unexpected Observations
- The paper acknowledges a counter-trend: AI as a *cognitive scaffold* supports neurodiverse learners and democratises access. This is the most under-developed part of the review — a strong publishable extension.

### 6.4 Statistical Meaning
- No statistical analysis is performed by the author; therefore inferred significance comes entirely from the cited primary studies.

### Publishability Strength Check
| Finding | Publication-grade? | What is needed to harden it |
|---|---|---|
| Offloading reduces memory | Yes (well-established since 2011) | Replication in the LLM era |
| LLMs reduce creativity EEG | Only if MIT study is peer-reviewed | Independent replication + effect size |
| AI causes skill fade at work | Partially | Controlled workplace study with baseline skills |
| AI reduces empathy / emotional engagement | Weak | Direct emotional-engagement measures (HRV, SAM) |
| Ethical recommendations | Editorial | Convert to tested intervention |

---

## 7. Strengths – Weaknesses – Assumptions

### 7.1 Technical Strengths
| # | Strength | Why it Matters |
|---|---|---|
| 1 | Multi-disciplinary scope | Makes the review attractive to education, HCI, and ethics venues |
| 2 | Clear mechanism vocabulary | Gives the field reusable terminology |
| 3 | Balanced tone (includes counterarguments) | Reduces "alarmist" framing |
| 4 | Actionable recommendations | Useful for policy citation |

### 7.2 Explicit Weaknesses
| # | Weakness | Consequence |
|---|---|---|
| 1 | No systematic search / PRISMA | Possible selection bias |
| 2 | No quantitative synthesis | No effect-size summary |
| 3 | Heavy dependence on one unpublished MIT study | Citation fragility |
| 4 | No operational definitions of "intelligence" components | Hard to replicate or falsify |
| 5 | Missing longitudinal evidence | Can't speak to long-term trajectory |
| 6 | Thin on age-, gender-, and domain-stratified analysis | Generalisations are too broad |
| 7 | No proposed measurement instrument | Future work has no protocol handed down |

### 7.3 Hidden Assumptions
| # | Hidden Assumption | Risk if False |
|---|---|---|
| 1 | EEG connectivity changes translate to real-world cognitive decline | Overstates harm |
| 2 | AI use is a unitary behaviour | Misses nuance (active vs. passive use) |
| 3 | Populations react uniformly | Masks subgroup effects |
| 4 | Short-term lab findings predict longitudinal outcomes | Premature generalisation |
| 5 | "Deep learning" in humans is better operationalised than it actually is | Circular reasoning |
| 6 | Recommendations are implementable in current education systems | Policy naïveté |

---

## 8. Weakness → Research Direction Mapping

| Identified Weakness | Why it Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| No systematic search | Narrative review format | PRISMA-compliant systematic review of "generative-AI use and cognitive outcomes (2020–2026)" | Database search (PubMed, Scopus, ERIC) + dual screening + risk-of-bias rating |
| No effect sizes | No meta-analysis performed | Meta-analysis with subgroup moderators (age, task type, exposure hours) | Random-effects models, meta-regression |
| Unpublished primary source | Emerging-field immaturity | Direct replication of MIT EEG study with pre-registration | EEG protocol with power analysis, open data |
| Mechanism asserted not tested | Out-of-scope for review | Mediation / moderation study: does offloading mediate AI use → skill decline? | Structural equation modelling on cohort data |
| No longitudinal data | Time-constrained | 12–24 month cohort tracking | Repeated measures with diary sampling |
| No dose–response | Unknown field | Design "AI-hours-per-week" × cognitive outcome curve | Dose–response regression, change-point analysis |
| Single-construct AI | Simplification | Taxonomy of AI use: passive-consume / co-pilot / critique-and-revise | Behavioural clustering + outcome analysis |
| Missing subgroups | Scope limit | Age-, neurotype-, discipline-stratified analysis | Factorial designs |
| Recommendations untested | Policy-level advice | RCT of AI-literacy curriculum | Cluster-randomised trial in schools |
| Scaffold / benefit side under-studied | Reviewer bias toward harm | Study of when AI *enhances* cognition | Within-subject A/B on guided vs. unguided AI |
| No reversibility data | Not discussed | Detraining study: does skill return after AI abstinence? | Pre–deprivation–post design |
| No neurodiversity focus | Narrow sample | Evaluate AI as assistive tech for ADHD / dyslexia / ASD | Field studies with validated measures |

---

## 9. Novel Contribution Extraction

### 9.1 Novel Claim Templates (inspired by this paper)

1. **"We propose a dose–response framework that quantifies how hours of generative-AI use per week predict longitudinal decline in working-memory span, thereby replacing the qualitative 'cognitive cost of convenience' thesis with a measurable law."**
2. **"We propose a Use-Mode Taxonomy (Passive / Co-pilot / Critique) that improves predictions of AI-related skill atrophy by showing that outcome depends more on interaction style than on exposure volume."**
3. **"We propose the Cognitive Scaffold Index, a measurement tool that distinguishes harmful offloading from beneficial augmentation, enabling educators to certify AI-enhanced curricula."**
4. **"We propose the first PRISMA-compliant meta-analysis of AI-induced cognitive effects (2020–2026) that aggregates effect sizes across 40+ studies and identifies moderators (age, task, dose)."**
5. **"We propose an RCT-validated AI Literacy curriculum that reduces metacognitive bypass in undergraduate students, improving essay originality scores by X% over a 12-week period."**

### 9.2 Recommended Framing
- Keep the phrase "cognitive cost of convenience" as a **citation hook** — it is memorable and likely to become canonical.
- Position new work as *operationalising* what Sreepadacharya *described*.

---

## 10. Future Research Expansion Map

### 10.1 Author-Suggested Future Work
- Longitudinal effects of AI on cognition, especially for youth and neurodiverse populations.
- Differentiating passive vs. active, generative vs. assistive AI interaction.

### 10.2 Missing Directions (our additions)
- **Biomarkers** of cognitive offloading (eye-tracking entropy, EEG theta/beta ratios).
- **Cultural moderation**: are effects the same across collectivist vs. individualist education systems?
- **Elderly populations**: AI as potential protective scaffold against age-related decline.
- **Professional skill loss**: pilots, surgeons, radiologists — domains already seeing AI assist.

### 10.3 Modern Extensions
- LLM-specific extensions:
  - Does *chain-of-thought prompting* preserve user reasoning while still gaining help?
  - Do *agentic AIs* (multi-step tool-use) amplify or reduce dependency vs. single-shot ChatGPT?
  - Retrieval-augmented systems — do they cause more or less memory offloading than vanilla LLMs?
- Neurotech extensions:
  - Pair EEG/fNIRS with real-time LLM interaction logs for millisecond-level offloading detection.

### 10.4 Cross-Domain Combinations
| Partner Discipline | New Angle |
|---|---|
| Behavioural Economics | Is AI offloading a form of present-bias / hyperbolic discounting? |
| HCI | UI patterns that *require* user reflection before accepting AI output |
| Neuroergonomics | Workload curves in AI-assisted vs. AI-free tasks |
| Philosophy of Mind | Extended-mind thesis vs. atrophy thesis — are they really opposed? |
| Public Policy | AI labelling / "nutrition facts" for cognitive impact |

---

## 11. How to Write a NEW Paper From This Work

### 11.1 Reusable Elements
- The **three-level mechanism ladder** (offloading → atrophy → dependency).
- The **interdisciplinary sweep** (psychology + neuroscience + education + ethics).
- The **counter-balanced framing** (harm + scaffold) that protects against reviewer pushback.
- The **recommendation trio** (educational reform, ethical design, resilience programs) as a policy-section template.

### 11.2 What MUST NOT Be Copied
- Do not reuse direct sentences or phrases (plagiarism risk; paper is open-access but not CC-BY in all venues).
- Do not recycle the same four domain sections as-is — restructure by *mechanism* instead of *domain* for novelty.
- Do not lean on the unpublished MIT report as a central citation — secure the peer-reviewed version or replace it.
- Avoid reusing the table of "Key Findings" — reframe as hypotheses to test.

### 11.3 How to Design a Novel Extension
1. Pick one mechanism (e.g., cognitive offloading) and operationalise it.
2. Choose one population (e.g., STEM undergraduates).
3. Design a 3-arm study: AI-free vs. AI-copilot vs. AI-generator.
4. Use validated instruments: N-back, Raven, Alternate Uses Task, AUT-creativity, metacognitive awareness inventory.
5. Add EEG or pupillometry as an objective correlate.
6. Pre-register, open-data, 12-week longitudinal.
7. Report effect sizes, not just p-values.

### 11.4 Minimum Publishable Contribution Checklist
- [ ] A clearly named construct or framework (e.g., "Use-Mode Taxonomy").
- [ ] At least one measurable hypothesis.
- [ ] A dataset (collected or secondary) with N ≥ statistical-power threshold.
- [ ] At least one validated cognitive instrument.
- [ ] Pre-registration link.
- [ ] Code / analysis scripts in a public repository.
- [ ] A limitations section addressing generalisability.
- [ ] Ethical review (IRB) statement.

---

## 12. Complete Paper Writing Template

Section-by-section scaffold for a new paper in this space.

### 12.1 Abstract
- **Purpose**: advertise the problem, method, and headline result.
- **Include**: 1 line problem, 1 line gap, 1 line method, 1 line result, 1 line implication.
- **Common mistakes**: vague claims ("AI affects cognition"); no numbers.
- **Reviewer expectation**: a quantitative headline (effect size or % change).

### 12.2 Introduction
- Purpose: motivate urgency + place paper in discourse.
- Include: societal hook, research gap, contribution list (bulleted), roadmap.
- Common mistakes: over-long story, buried contribution.
- Reviewer expectation: explicit "Our contributions are: (i)…, (ii)…, (iii)…".

### 12.3 Related Work
- Purpose: show mastery, differentiate from prior work.
- Include: Google Effect line, modern LLM-era line, neuroscience line.
- Common mistakes: listing papers without synthesis.
- Reviewer expectation: comparison table vs. closest 3–5 papers.

### 12.4 Method / Framework
- Purpose: make the study auditable.
- Include: population, tasks, interventions, measures, protocol timeline.
- Common mistakes: missing exclusion criteria; un-validated instruments.
- Reviewer expectation: pre-registration ID + IRB number.

### 12.5 Theory / Conceptual Model
- Purpose: give the reader a formal hook (see Section 3.1 above).
- Include: variables, assumptions, hypotheses (H1…Hk).
- Common mistakes: no falsifiable statement.
- Reviewer expectation: hypotheses must map 1-to-1 with experiments.

### 12.6 Experiments / Data Analysis
- Purpose: test hypotheses.
- Include: power calculation, pre-analysis plan, statistical models, robustness checks.
- Common mistakes: only reporting p-values; no effect size; multiple-comparisons not corrected.
- Reviewer expectation: open data + Bayesian or frequentist rigor.

### 12.7 Discussion
- Purpose: interpret, not repeat.
- Include: mechanistic explanation, surprises, practical implications.
- Common mistakes: over-claiming causation from correlation.
- Reviewer expectation: explicit list of alternative explanations.

### 12.8 Limitations
- Purpose: pre-empt reviewers.
- Include: sampling limits, instrument limits, generalisation limits, causal limits.
- Common mistakes: trivial limitations only.
- Reviewer expectation: at least one limitation that could overturn the main claim.

### 12.9 Conclusion
- Purpose: restate contribution in sharpest form.
- Include: 3–4 sentences maximum.
- Common mistakes: adding new content here.
- Reviewer expectation: pointer to future work.

### 12.10 References
- Purpose: credibility.
- Include: peer-reviewed primary sources > press articles.
- Common mistakes: relying on unpublished reports (as this paper partly does).
- Reviewer expectation: ≥ 60% of citations peer-reviewed and < 5 years old.

---

## 13. Publication Strategy Guide

### 13.1 Suitable Venues

| Venue Type | Examples | Why |
|---|---|---|
| HCI Conferences | CHI, CSCW, UIST | Human–AI interaction angle |
| Cognitive Science | Cognition, Topics in Cognitive Science | Cognition + AI overlap |
| Education | Computers & Education, BJET | Educational impact focus |
| AI Ethics | AIES, FAccT | Ethical implications strand |
| Interdisciplinary | Nature Human Behaviour, PNAS | Cross-cutting meta-analyses |
| Neuroscience | NeuroImage, Cerebral Cortex | EEG / fMRI correlates |
| Regional / Conference | IJFMR, AIMAR (as this paper) | Fast conceptual outlet |

### 13.2 Required Baseline Expectations
- Tier 1 venues (CHI, FAccT, Nature HB): pre-registration, N ≥ 200, effect sizes, open data.
- Tier 2 venues (BJET, C&E): validated instruments, justified sampling, moderate N.
- Tier 3 venues (IJFMR-style): conceptual rigor, clear contribution, good structure.

### 13.3 Experimental Rigor Level (desired upgrade)
- From narrative → systematic review: **PRISMA 2020** checklist, PROSPERO registration.
- From narrative → empirical: power analysis, IRB, pre-registration (OSF), open materials.

### 13.4 Common Rejection Reasons
- "Incremental" (pure review without new data).
- Over-claiming causation.
- Weak primary-source quality (press citations).
- Missing effect sizes.
- No operational definitions of "intelligence".
- No falsifiable hypothesis.

### 13.5 Increment Needed for Acceptance
- Tier 3 → Tier 2: add a systematic review + meta-analysis.
- Tier 2 → Tier 1: add original empirical data, pre-registered, with neural / behavioural measures.
- Any tier: clarify *which* cognitive function is targeted and use a validated measurement.

---

## 14. Researcher Quick Reference Tables

### 14.1 Key Terminology Table
| Term | Meaning in this Paper |
|---|---|
| Cognitive Offloading | Transferring a mental task to an external aid |
| Skill Atrophy | Loss of a skill through disuse |
| Cognitive Dependency | Inability to perform without the aid |
| Deep Learning (human sense) | Effortful, reflective understanding |
| Cognitive Scaffold | A supportive aid that boosts without replacing |
| Metacognition | Monitoring and controlling one's own thinking |
| Skill Fade | Workplace-specific term for declining professional acuity |
| Intellectual Ownership | Authorship of one's own ideas |

### 14.2 Important Equations Summary
| Equation | Present in Paper? | Notes |
|---|---|---|
| Any formal model | No | Reader-reconstructed model given in §3.1 |

### 14.3 Parameter Meaning Table (for the follow-up framework in §3.1)
| Symbol | Meaning | How to Measure |
|---|---|---|
| `U` | AI-use intensity | Hours/week, session count |
| `C` | Native competence | N-back, Raven, AUT, domain tests |
| `k_offload` | Offloading rate | Fraction of task delegated |
| `k_atrophy` | Decay rate | Pre–post deprivation test |
| `k_practice` | Practice gain rate | Pre–post practice test |

### 14.4 Algorithm Flow Summary
1. Define cognitive target (memory / creativity / reasoning).
2. Select instrument.
3. Vary AI-use dose.
4. Measure behavioural + neural outcome.
5. Repeat longitudinally.
6. Fit dose–response curve.
7. Extract policy thresholds.

---

## 15. One-Page Master Summary Card

| Dimension | Summary |
|---|---|
| Problem | Does heavy AI reliance cause decline in human cognitive capacity? |
| Idea | Convenience has a cost mediated by offloading → atrophy → dependency; evidence aggregated from four disciplines |
| Method | Interdisciplinary narrative literature review — no data, no meta-analysis |
| Results | Convergent qualitative evidence of reduced memory, creativity, vigilance, and neural engagement under AI-heavy use; counter-evidence that ethically designed AI can scaffold cognition |
| Weakness | No PRISMA search, no quantitative synthesis, dependence on unpublished MIT report, no operational definitions, no longitudinal or dose–response data |
| Research Opportunity | Convert the conceptual mechanism ladder into a measurable, pre-registered, longitudinal empirical programme with a Use-Mode Taxonomy and Cognitive Scaffold Index |
| Publishable Extension | (i) PRISMA meta-analysis of AI-cognition studies 2020–2026; (ii) RCT of AI-literacy curriculum on metacognitive outcomes; (iii) EEG dose–response study of generative-AI use intensity; (iv) AI-as-assistive-technology evaluation in neurodiverse populations |

---

*End of companion document for Paper 34 — "The Cognitive Cost of Convenience".*
