# Research Companion: GPTs are GPTs — An Early Look at the Labor Market Impact Potential of Large Language Models
**Eloundou, Manning, Mishkin & Rock (2023) | OpenAI / University of Pennsylvania**

---

# 0. Quick Paper Identity Card

| Field | Details |
|---|---|
| **Problem Domain** | Labor Economics + AI Policy + Human-Computer Interaction |
| **Paper Type** | Empirical + Conceptual (Survey-style measurement study) |
| **Core Contribution** | First large-scale rubric-based exposure measurement of LLM impact on U.S. occupations using both human annotators and GPT-4 |
| **Key Idea** | LLMs (like GPTs) qualify as general-purpose technologies — they affect nearly all wage levels, with higher-skilled jobs surprisingly more exposed |
| **Required Background** | Basic understanding of labor economics, LLMs, O*NET occupational data, task-based automation theory |
| **Primary Baseline** | Prior automation exposure studies: Brynjolfsson et al. (SML), Frey & Osborne (2017), Webb (2020), Felten et al. (2018), Acemoglu & Autor (2011) |
| **Main Innovation Type** | New measurement rubric + novel use of LLM-as-annotator at scale |
| **Difficulty Level** | Moderate (no heavy math; requires economics & policy literacy) |
| **Reproducibility Level** | Medium (rubric is published; O*NET data is public; GPT-4 version used is pinned but not fully accessible) |

---

# 1. Research Context & Core Problem

## 1.1 Exact Problem Formulation

The central question: **Which U.S. worker tasks and occupations are exposed to LLMs, and to what degree?**

"Exposed" here means: would an LLM or LLM-powered software reduce the time to complete a task by at least 50% while maintaining equivalent quality?

The authors deliberately avoid predicting actual job loss or displacement. They measure *technical feasibility of impact*, not economic outcomes.

## 1.2 Why This Problem Exists

- LLMs (GPT-3.5, GPT-4) achieved dramatic capability jumps in 2022–2023.
- Society lacks systematic frameworks for understanding *which* jobs are at risk and *why*.
- Prior automation frameworks (Frey & Osborne, SML) were built for narrower machine learning — not for the broad language capabilities of GPT-class models.
- Policymakers, educators, and workers need advance knowledge to prepare.

## 1.3 Historical / Theoretical Gap

- Classic automation research (Autor et al., 2003) focused on *routine vs. non-routine* tasks.
- Previous AI exposure indices measured generic ML, not language-specific capabilities.
- No prior work specifically measured GPT-type model capabilities at task-level scale across all U.S. occupations.
- The role of *complementary software* built on top of LLMs was unexamined.

## 1.4 Limitations of Previous Approaches

| Prior Method | Limitation |
|---|---|
| Frey & Osborne (2017) | Expert-labeled subset only; no task-level resolution |
| Webb (2020) patent mapping | Patent language ≠ actual LLM capabilities |
| Felten et al. (2018) AI Occupational Exposure | Maps AI to human *abilities*, not *tasks*; predates LLMs |
| Brynjolfsson et al. SML | Designed for traditional ML, not generative text/code models |
| Acemoglu & Autor (2011) RTI | Focuses on routine/manual, misses cognitive language tasks |

## 1.5 Contribution Category

- **Empirical insight** (large-scale occupation exposure measurement)
- **Methodological** (new rubric + LLM-as-annotator pipeline)
- **Conceptual** (GPTs as general-purpose technologies argument)

---

### Why This Paper Matters

This is the first study to quantify, at task level, the overlap between LLM capabilities and real-world occupation tasks across the entire U.S. labor market. It establishes that LLMs' impact is not limited to low-wage routine work — it hits highly educated, high-wage, cognitive workers hardest. That inversion of prior expectations reshapes how AI policy must be designed.

---

### Remaining Open Problems

- How do exposure rates change as LLM capabilities advance (GPT-4 → GPT-5+)?
- What is the actual adoption rate and timeline across different industries?
- How do non-U.S. economies, informal workers, and self-employed workers differ?
- What is the actual productivity gain (not just theoretical feasibility)?
- How do workers adapt their tasks to remain relevant after exposure?
- What new tasks and jobs do LLMs create (task-reinstatement)?

---

# 2. Minimum Background Concepts

## 2.1 O*NET Database

- **What it is**: A U.S. government database cataloging 1,016 occupations with their associated tasks and Detailed Work Activities (DWAs).
- **Role in paper**: Primary source of task descriptions used for exposure labeling.
- **Why needed**: Provides standardized, government-recognized task definitions that can be systematically evaluated.

**Key terms**:
- **Task**: An occupation-specific unit of work (e.g., "Monitor system operation to detect potential problems").
- **DWA (Detailed Work Activity)**: A more granular action that is part of completing a task (e.g., "Monitor computer system performance").
- **Core task**: O*NET-designated primary tasks (weighted double in aggregation).
- **Supplemental task**: Secondary tasks (weighted half as much as core tasks).

## 2.2 Exposure Rubric (Central Concept)

The authors define **exposure** as a proxy for potential economic impact — whether LLM access would reduce task time by ≥50% at equivalent quality. Three levels:

| Label | Name | Meaning |
|---|---|---|
| E0 | No Exposure | LLM provides no or minimal time reduction; or reduces quality |
| E1 | Direct Exposure | Bare ChatGPT/OpenAI Playground alone cuts time ≥50% |
| E2 | LLM+ Exposure | Extra software built on LLMs needed to cut time ≥50% |
| E3 | Image-Capability Exposure | Requires LLM + image understanding (merged with E2 in analysis) |

## 2.3 Three Aggregate Exposure Measures

| Symbol | Formula | Interpretation |
|---|---|---|
| α (alpha) | = E1 proportion | Lower bound — only direct LLM capability |
| β (beta) | = E1 + 0.5 × E2 | Middle estimate — complementary tools get half-weight |
| ζ (zeta) | = E1 + E2 | Upper bound — full potential including all LLM-powered tools |

**β is the primary measure used throughout the paper.**

## 2.4 General-Purpose Technology (GPT as GPT)

A **general-purpose technology** (GPT — note the deliberate wordplay) is one that:
1. Improves continuously over time
2. Spreads pervasively across the economy
3. Spawns complementary innovations

Examples: printing press, steam engine, electricity. The authors argue LLMs meet all three criteria.

## 2.5 Task-Based Automation Framework

Standard labor economics conceptualizes jobs as **bundles of tasks** rather than single monolithic roles. This lets automation affect parts of a job without fully replacing it. Key insight: LLMs rarely expose 100% of any job — they expose varying fractions of tasks within jobs.

## 2.6 Skill-Biased vs. Routine-Biased Technological Change

- **Skill-biased**: Technology raises demand for high-skill workers over low-skill ones.
- **Routine-biased**: Technology displaces workers doing repetitive, predictable tasks.
- LLMs break this pattern — they show *higher* exposure for *higher-wage*, *higher-skill* jobs.

## 2.7 Baumol's Cost Disease (Referenced in Paper)

A theory stating that as productivity grows in some sectors (e.g., manufacturing), wages rise economy-wide, making labor-intensive sectors (education, healthcare) relatively more expensive — even if their productivity didn't increase. The paper finds LLM exposure is *uncorrelated* with past productivity growth, suggesting LLMs may not worsen this effect.

---

# 3. Mathematical / Theoretical Understanding Layer

This paper is primarily **empirical and conceptual**, not mathematics-heavy. Key quantitative elements are statistical rather than theoretical derivations.

## 3.1 Exposure Score Aggregation

**Step 1 — Task-level score**: Each task gets a label: E0=0, E1=1, E2=0.5 (for β) or E2=1 (for ζ).

**Step 2 — Occupation-level score**:

$$\beta_{occupation} = \frac{\sum_{i} w_i \cdot score_i}{\sum_i w_i}$$

Where:
- $w_i = 2$ for core tasks, $w_i = 1$ for supplemental tasks
- $score_i \in \{0, 0.5, 1\}$ under β weighting

**Why this matters**: The weighting scheme (core = 2×) prioritizes primary job functions, preventing peripheral tasks from distorting exposure estimates.

## 3.2 Agreement Metrics (Table 2)

**Agreement rate**: How often two annotators assign the same label (E0, E1, or E2) to the same task.

**Pearson's correlation**: Measures linear relationship between continuous exposure scores from two sources (human vs. GPT-4).

| Comparison | β Agreement | β Pearson's r |
|---|---|---|
| GPT-4 Rubric 1 vs. Human | 65.6% | 0.591 |
| GPT-4 Rubric 2 vs. Human | 65.6% | 0.538 |
| GPT-4 Rubric 1 vs. GPT-4 Rubric 2 | 76.0% | 0.705 |

**Interpretation**: Moderate agreement at task level, but stronger correlation at occupation level — individual task scores are noisy; aggregated occupation scores are more consistent.

## 3.3 Regression Validation (Section 5)

OLS regression of new LLM exposure scores on prior measures:

$$\text{LLM Exposure}_j = \alpha + \beta_1 \text{SML}_j + \beta_2 \text{Software}_{Webb,j} + \beta_3 \text{Robot}_{Webb,j} + \ldots + \epsilon_j$$

**Key findings**:
- R² = 60.7% to 72.8% — prior measures explain 60–73% of variance
- **28–40% of variance is new** — LLM exposure captures something prior metrics missed
- Positive associations: Software (Webb), SML, routine cognitive
- Negative associations: Robot (Webb), routine manual, Frey & Osborne automation

### Mathematical Insight Box
> The 28–40% unexplained variance is the most important number in Section 5. It proves that LLM exposure is *not* just a rehash of prior automation risk — it is a genuinely new dimension of technological exposure. This unexplained variance is the **research opportunity**.

## 3.4 Skill Importance Regression (Table 5)

Regression of human-annotated β exposure on normalized O*NET skill importance scores:

| Skill | Coefficient (β exposure) | Direction |
|---|---|---|
| Writing | +0.467 *** | Positive |
| Programming | +0.623 *** | Positive |
| Reading Comprehension | +0.470 *** | Positive |
| Science | −0.230 *** | Negative |
| Critical Thinking | −0.196 *** | Negative |
| Active Learning | −0.065 ** | Negative |
| Mathematics | +0.161 *** | Positive |

**Key insight**: Jobs requiring human judgment, experimentation, and scientific reasoning are *less* exposed. Jobs centered on text production and code are *most* exposed.

---

# 4. Proposed Method / Framework

## 4.1 Overall Pipeline

```
O*NET Database (tasks + DWAs)
        |
        v
Exposure Rubric (E0 / E1 / E2 / E3)
        |
   +----+----+
   |         |
Human      GPT-4
Annotators  (as classifier)
   |         |
   +----+----+
        |
Task-level Exposure Labels
        |
        v
Aggregate to Occupation Level (α, β, ζ)
        |
        v
Merge with BLS Wage & Employment Data
        |
        v
Regression / Cross-tabulation Analysis
        |
        v
Results: Exposure by Wage, Skill, Job Zone, Industry
```

## 4.2 Step-by-Step Breakdown

### Step 1 — Data Collection

**What**: O*NET 27.2 database → 19,265 tasks + 2,087 DWAs across 1,016 occupations.
**Why**: Standardized, government-validated task taxonomy enables systematic labeling.
**Weakness**: O*NET tasks may not fully capture tacit, informal, or physical sub-components of work.
**Improvement idea**: Augment O*NET with real-time job postings (LinkedIn, Indeed) for more dynamic task coverage.

### Step 2 — Rubric Design

**What**: A structured decision guide asking "Would this LLM cut task time ≥50% at equivalent quality?"
**Why**: Creates objective, reproducible classification criteria instead of vague expert judgment.
**Weakness**: 50% threshold is arbitrary; real adoption may cluster at different efficiency thresholds.
**Improvement idea**: Test multiple thresholds (25%, 50%, 75%) to create exposure *curves* rather than binary cutoffs.

### Step 3 — Human Annotation

**What**: Authors + experienced OpenAI alignment annotators labeled DWAs and subsets of tasks.
**Why**: Ground truth anchor for GPT-4 calibration and validation.
**Weakness**: Annotators are OpenAI-adjacent — not occupationally diverse. Biased toward tech-literate perspective.
**Improvement idea**: Use domain experts (nurses, lawyers, teachers) as annotators for their own profession's tasks.

### Step 4 — GPT-4 as Annotator

**What**: GPT-4 labels all 19,265 task/occupation pairs using a prompt version of the rubric.
**Why**: Human annotation at full scale (19k tasks) is infeasible in time/cost.
**Weakness**: GPT-4 self-assessment of LLM capabilities is circular — the model judges its own potential.
**Improvement idea**: Use external benchmarks or third-party capability assessments to anchor the rubric.

### Step 5 — Score Aggregation

**What**: DWA-level → Task-level → Occupation-level scores, weighting core tasks 2× supplemental.
**Why**: Core tasks are more central to occupation identity.
**Weakness**: Equal weight within core/supplemental ignores task *importance* (O*NET has importance ratings that were mostly not used).
**Improvement idea**: Weighted by O*NET Importance scores + frequency × importance product.

### Step 6 — Wage / Employment / Skill Merging

**What**: Link occupation exposure scores to BLS wage data, employment counts, skill profiles.
**Why**: Enables distributional analysis — who bears the exposure burden?
**Weakness**: BLS data covers formal economy only; excludes gig workers, self-employed, informal labor.
**Improvement idea**: Supplement with CPS microdata and platform economy data (Uber, Upwork) for broader coverage.

---

# 5. Experimental Setup / Evaluation Design

## 5.1 Dataset Characteristics

| Dataset | Source | Size | Notes |
|---|---|---|---|
| O*NET Tasks | O*NET 27.2 | 19,265 tasks | Occupation-specific |
| O*NET DWAs | O*NET 27.2 | 2,087 DWAs | Cross-occupation activities |
| Occupations | O*NET / BLS | 1,016 occupations | Full U.S. economy |
| Wage & Employment | BLS OES 2020–2021 | All occupations | Formal economy only |
| Demographics | CPS via BLS | Worker-level | Used for population exposure estimates |

## 5.2 Annotation Protocol

- Human annotators labeled all 2,087 DWAs + subset of tasks (especially tasks without DWAs).
- GPT-4 labeled all 19,265 task/occupation pairs.
- Authors personally labeled a sample to tune GPT-4 prompt for alignment.
- Core tasks weighted 2× supplemental in all occupation-level aggregations.

## 5.3 Metrics Used and Why

| Metric | Why Used |
|---|---|
| Agreement rate (%) | Checks raw label matching between annotators |
| Pearson's r | Measures linear correlation between continuous exposure scores |
| α, β, ζ measures | Provide lower, middle, and upper exposure bounds |
| R² in OLS regression | Shows how much prior measures explain new LLM exposure score |
| Occupation-level mean β | Main exposure index for distributional analysis |

## 5.4 Baseline Selection Logic

Prior measures used as baselines:
- **SML (Brynjolfsson)**: Closest methodological cousin — rubric-based evaluation.
- **Webb (2020)**: Patent-to-task mapping — tests software vs. robot exposure.
- **Frey & Osborne (2017)**: Expert-based, occupation-level — tests against high-level automation risk.
- **Felten et al. (2018)**: AI-to-ability mapping — tests cognitive capability framing.
- **Acemoglu & Autor (2011) RTI**: Routine task intensity — tests manual vs. cognitive dimension.

## 5.5 Experimental Limitations

### Experimental Reliability Analysis

| What is Trustworthy | What is Questionable |
|---|---|
| Direction of findings (higher wages → higher exposure) — consistent across human + GPT-4 | Exact exposure percentages — subjective 50% threshold inflates/deflates estimates |
| Agreement between GPT-4 and human at occupation level (correlation ~0.6–0.65) | Agreement at task level (65.6% for β) — many individual task labels are disputed |
| Negative correlation with physical/manual tasks — physically grounded and robust | Forward-looking claims about LLM-powered software — speculative by design |
| Industry-level heterogeneity (information → high; agriculture → low) | Applying U.S.-based results globally — not validated outside formal U.S. economy |

---

# 6. Results & Findings Interpretation

## 6.1 Main Outcomes

**Headline finding**: About **80% of U.S. workers** are in occupations where at least 10% of tasks are exposed to LLMs (β measure). About **19% of workers** are in occupations where ≥50% of tasks are exposed.

**Task-level finding**: With bare LLMs (α): ~15% of all tasks are directly exposed. With LLM-powered software (ζ): 47–56% of all tasks become exposed. → Complementary software amplifies impact by 3–4×.

## 6.2 Key Exposure Patterns

### By Occupation (Most Exposed — Human β)
1. Survey Researchers — 84.4%
2. Writers and Authors — 82.5%
3. Interpreters and Translators — 82.4%
4. Public Relations Specialists — 80.6%
5. Animal Scientists — 77.8%

### By Occupation (Least Exposed)
34 occupations have zero tasks exposed, all physically intensive:
- Agricultural Equipment Operators, Dishwashers, Stonemasons, Tire Repairers, Roustabouts (Oil & Gas), etc.

### By Wage Level
- **Higher wages → higher exposure** (positive relationship)
- This is the opposite of prior automation studies (which found low-wage routine workers most at risk)
- LLMs target *cognitive* work, not manual/routine work

### By Job Zone (Education Level)
| Job Zone | Preparation | Median β Exposure |
|---|---|---|
| 1 (0–3 months) | No degree | 0.06 |
| 2 (3–12 months) | High school diploma | 0.16 |
| 3 (1–2 years) | Vocational/Associate | 0.26 |
| 4 (2–4 years) | Bachelor's degree | 0.47 |
| 5 (4+ years) | Master's or higher | 0.43 |

Job Zone 4 (Bachelor's degree level) is *most* exposed.

### By Skill Type
- **Positively exposed**: Writing (+0.467), Programming (+0.623), Reading Comprehension (+0.470)
- **Negatively exposed**: Science (−0.230), Critical Thinking (−0.196)

### By Industry
- **High exposure**: Data processing, information services, hospitals, legal services
- **Low exposure**: Manufacturing, agriculture, mining, construction

## 6.3 Validation Against Prior Measures

The new LLM exposure score explains 28–40% of variance not captured by prior AI/automation measures. It positively correlates with software-related measures and negatively correlates with robot/manual routine measures — as theoretically expected.

## 6.4 Unexpected Observations

- **Writers and translators** — intuitively expected to be exposed, confirmed.
- **Mathematicians labeled 100% exposed by GPT-4** — surprising since mathematics seems creative/complex.
- **Barbers appear moderately exposed** — artifact of task-weighting scheme (flagged in paper as curious result).
- **Science skills are negatively correlated** — suggests LLMs struggle with empirical reasoning, experiments, observation-based tasks.

---

### Publishability Strength Check

| Result | Assessment |
|---|---|
| 80% of workers have ≥10% task exposure | Publication-grade: large sample, two annotation methods agree |
| Higher wages → higher exposure | Publication-grade: statistically robust, robust to human vs. GPT-4 |
| Writing/programming skills → positive exposure | Publication-grade: regression with significance |
| Science/critical thinking → negative exposure | Publication-grade: strong significance |
| 28–40% unique variance vs. prior measures | Publication-grade: empirically validated |
| LLMs as general-purpose technologies | Conceptual argument — needs longitudinal economic data to fully confirm |
| Industry exposure heterogeneity | Needs stronger validation — relies on GPT-4 aggregation without detailed breakdown |

---

# 7. Strengths – Weaknesses – Assumptions

## Table 1: Technical Strengths

| Strength | Explanation |
|---|---|
| Task-level granularity | Exposure measured at task (not occupation) level — more precise than prior occupation-level studies |
| Dual annotation source | Both human annotators and GPT-4 used — cross-validation increases confidence |
| Full economy coverage | 1,016 occupations; 19,265 tasks — complete U.S. formal economy coverage |
| Novel LLM-as-annotator method | First demonstration of using GPT-4 to scale occupational exposure measurement |
| Three exposure metrics | α/β/ζ provide lower, middle, upper bounds — captures uncertainty in software development trajectories |
| Validated against prior literature | Regression against 6 prior measures; 60–73% R² confirms consistency |
| Industry + wage + skill analysis | Comprehensive distributional analysis across multiple dimensions |

## Table 2: Explicit Weaknesses

| Weakness | Severity | Impact |
|---|---|---|
| Subjective 50% threshold | High | Arbitrary cutoff affects all exposure estimates |
| Annotators not occupationally diverse | High | Likely biased toward tech-familiar interpretation of tasks |
| GPT-4 self-evaluates its own capabilities | High | Circular reasoning; model may overestimate or underestimate |
| Formal economy only | High | Gig workers, self-employed, informal sector excluded |
| U.S.-only scope | Medium | Results may not generalize internationally |
| Task-based framework validity | Medium | Tacit knowledge, judgment, physical sub-tasks may be missed |
| No adoption timeline | Medium | Technical feasibility ≠ actual deployment or impact |
| Binary exposure (yes/no to 50%) | Medium | Misses continuous efficiency gains below 50% threshold |
| Forward-looking LLM-powered software labels | High | Speculative — software does not yet exist for many E2 labels |

## Table 3: Hidden Assumptions

| Assumption | Where It Appears | Why It's Risky |
|---|---|---|
| Tasks adequately represent jobs | Core methodology | Tacit skills and judgment are not captured in O*NET task lists |
| 50% time reduction = meaningful adoption trigger | Exposure rubric | Actual adoption may trigger at 20% or 80% depending on cost/benefit |
| LLM capabilities are constant across all users | Rubric | "Average worker" assumption hides expertise variation |
| Complementary software will be built | E2 / ζ measure | Investment incentives may delay or prevent software development |
| Equivalent quality is observable | Rubric definition | Quality judgment is itself subjective and task-dependent |
| Formal employment data reflects actual labor market | BLS data | Millions of informal/gig/self-employed workers are excluded |
| GPT-4's self-assessment is calibrated | LLM annotation | No external benchmark validates GPT-4's own capability claims |

---

# 8. Weakness → Research Direction Mapping

| Identified Weakness | Why It Exists | Research Opportunity | Possible Method |
|---|---|---|---|
| Arbitrary 50% threshold | Chosen for annotator interpretability, not economic validity | Study how productivity adoption varies with efficiency gain thresholds | Survey firms on LLM adoption triggers; model adoption curves at different efficiency levels |
| Annotators not occupationally diverse | Practical constraint (OpenAI team) | Validated exposure labeling by domain experts | Crowdsource rubric via domain professionals (nurses, lawyers, teachers) on their own task lists |
| GPT-4 self-evaluation bias | Model cannot objectively assess its own limits | Use external benchmarks to evaluate LLM capability per task type | Map standardized capability benchmarks (MMLU, HumanEval) to O*NET task categories |
| Formal economy only | BLS/CPS data limitation | Gig/informal economy exposure study | Use platform API data (Upwork, Fiverr) + task descriptions to estimate freelancer exposure |
| U.S.-only | Dataset availability | Cross-country comparative exposure study | Extend rubric to EU ISCO, Indian NCS occupational databases |
| No adoption timeline | Outside paper's scope | LLM adoption speed as a function of exposure level + industry characteristics | Time-series analysis of LLM adoption data (job posting changes, API usage) |
| Task-based framework validity | O*NET limitation | Sub-task decomposition study | Fine-grained task video annotation + LLM step evaluation |
| Binary exposure measure | Simplification choice | Continuous productivity gain estimation per task | A/B experiment measuring actual time savings with and without LLM |
| E2 is speculative | Software doesn't exist yet | Track whether predicted E2 software emerges over time | Longitudinal study: match 2023 E2 tasks to software products released 2024–2026 |
| No wage/inequality modeling | Beyond scope | Model distributional income effects of LLM adoption under different scenarios | Computable general equilibrium (CGE) model with LLM exposure as input |

---

# 9. Novel Contribution Extraction

## 9.1 What the Authors Explicitly Contribute

1. A new exposure rubric for measuring LLM impact on worker tasks.
2. A large-scale annotated dataset combining human + GPT-4 labeling.
3. Empirical evidence that higher-wage jobs face higher LLM exposure (inverts prior assumption).
4. Evidence that LLMs qualify as general-purpose technologies.
5. Demonstration that LLM-powered software multiplies direct LLM impact by 3–4×.

## 9.2 Novel Claim Templates for Your Research

- "We propose **[domain-specific exposure rubric]** that improves upon Eloundou et al.'s task-level LLM exposure measurement by incorporating **[tacit knowledge / sub-task decomposition / multimodal context]** to better capture **[jobs requiring physical judgment / emotional labor / dynamic decision-making]**."

- "We propose **[longitudinal exposure tracking framework]** that extends static rubric-based measurement by monitoring **[actual software deployment versus predicted E2 exposure]** to measure the **[speed of complementary innovation adoption]**."

- "We propose **[cross-national LLM exposure index]** that adapts the Eloundou et al. rubric to **[EU ISCO / Indian NCS / OECD occupational taxonomies]** to quantify **[how labor market structure and wage distribution shape differential LLM impact across nations]**."

- "We propose **[expert-annotated domain-specific exposure dataset]** that replaces generalist annotators with **[nurses / lawyers / software engineers annotating their own profession's tasks]** to eliminate **[tech-familiarity bias in LLM exposure estimates]**."

- "We propose **[continuous exposure curve methodology]** that replaces the binary 50% threshold with **[marginal productivity gain models]** to provide **[adoption-realistic estimates of LLM impact at different efficiency levels]**."

---

# 10. Future Research Expansion Map

## 10.1 Author-Suggested Future Work

- Extend to non-U.S. labor markets (different industrial structures, regulatory environments).
- Study actual LLM adoption patterns by sector and occupation.
- Study real model capabilities vs. theoretical task-level claims (beyond exposure scores).
- Account for GPT-4 vision capabilities in direct exposure (α) measurement.
- Track how emerging model capabilities change exposure classifications over time.

## 10.2 Missing Directions Not Mentioned by Authors

- **Worker adaptation study**: How do workers in high-exposure occupations restructure their task portfolios to avoid displacement?
- **Firm-level adoption heterogeneity**: Why do some firms in high-exposure industries adopt LLMs faster than others?
- **Quality measurement**: Does LLM-assisted output actually achieve "equivalent quality" as the rubric assumes?
- **New task creation**: What new tasks (task-reinstatement) do LLMs generate that didn't exist before?
- **Socioeconomic group analysis**: How does exposure map across race, gender, and disability status within high-exposure occupations?

## 10.3 Modern Extensions (Post-2023)

- **Multimodal LLMs** (GPT-4V, Gemini): How does vision capability shift E0 → E1 for tasks requiring image reading or diagram analysis?
- **Agentic LLMs** (Claude, GPT with tools): Autonomous task execution changes E2 classification — more tasks become directly automatable.
- **Fine-tuned domain LLMs** (legal, medical, coding): Does domain-specific LLM training shift exposure estimates for specialist occupations?
- **LLM-in-the-loop workflows**: How does the human-AI collaboration model change labor demand in partially exposed occupations?

## 10.4 Cross-Domain Combinations

- **Education research**: Which educational credentials will retain value if LLMs expose high-credential occupations?
- **Public health**: If medical coding, clinical documentation, and data entry are exposed, how does this affect healthcare system costs?
- **Legal studies**: If legal research and document drafting are highly exposed, how should law school curricula adapt?
- **Organizational behavior**: How do managers change task assignment across human and LLM workers in exposed occupations?

---

# 11. How to Write a NEW Paper From This Work

## 11.1 Reusable Elements

- **The exposure rubric design pattern**: Threshold-based binary classification → adaptable to any technology capability assessment.
- **LLM-as-annotator methodology**: Using GPT-4 to scale human annotation — replicable for any new capability or new occupation database.
- **α / β / ζ tripartite exposure bounding**: A clean way to report lower/central/upper estimates when future software is uncertain.
- **Regression against prior measures**: Standard approach to validate new exposure measures against established baselines.
- **Job Zone / Education Level cross-tabulation**: Clean way to show distributional impacts.

## 11.2 What MUST NOT Be Copied

- The exact rubric wording (published and copyrightable by OpenAI/authors).
- The specific O*NET data labels generated by this team.
- The exact regression tables without proper citation.
- The "GPTs are GPTs" framing without attribution.

## 11.3 How to Design a Novel Extension

**Option A — Better Annotation**:
Replace generalist annotators with domain-expert annotators (nurses rating healthcare tasks; lawyers rating legal tasks). Measure how bias shifts when annotators have deep domain knowledge.

**Option B — Longitudinal**:
Re-run the rubric on the same O*NET tasks in 2025–2026 with GPT-4o or Claude-3 capabilities. Measure how many E0 tasks became E1, how many E2 tasks became E1. Track the velocity of exposure increase.

**Option C — New Geography**:
Apply the rubric to European ISCO occupational taxonomy and PIAAC skills data. Compare U.S. vs. EU exposure distributions. Test whether stronger labor protections in the EU dampen actual adoption despite similar technical exposure.

**Option D — Deeper Measurement**:
Replace the binary rubric with a continuous time-reduction experiment. Actually measure how long workers take with/without LLM assistance on O*NET tasks. Convert to actual productivity gain estimates instead of theoretical possibility estimates.

**Option E — Supply Side**:
Instead of measuring *which tasks* LLMs can perform, measure *which LLM capabilities* are demanded — using job postings, hiring patterns, and firm investment data to detect revealed adoption preferences.

## 11.4 Minimum Publishable Contribution Checklist

- [ ] A new dataset, annotation, or extension to an existing database
- [ ] At least one methodological improvement over Eloundou et al.
- [ ] Validation against the original paper's findings (can confirm, extend, or challenge)
- [ ] At least one new distributional result not present in the original paper
- [ ] Discussion of policy implications beyond U.S. formal economy
- [ ] Statistical robustness checks (multiple specifications, sensitivity to threshold choice)
- [ ] Comparison to at least 3 prior automation exposure measures

---

# 12. Complete Paper Writing Template

## Abstract
**Purpose**: State the exact research gap, the new method, the key finding, and the implication in 150–250 words.
**Include**: Research question → dataset/method → one quantitative headline result → what it means for labor/policy.
**Common mistake**: Burying the key finding at the end. Lead with the most surprising result.
**Reviewer expectation**: Clear gap statement, distinct contribution, quantitative claim, broad implication.

---

## 1. Introduction
**Purpose**: Motivate the research, state the gap, preview contributions, and provide paper roadmap.
**Include**:
- Concrete evidence of LLM capability growth (capability jump examples)
- What we don't know yet (the gap)
- Your paper's specific contributions (3–5 bullet points)
- Brief description of data and method
- Paper structure overview
**Common mistake**: Overpromising results or claiming to "solve" labor displacement.
**Reviewer expectation**: Measurable contribution clearly distinct from Eloundou et al. and Brynjolfsson et al.

---

## 2. Literature Review
**Purpose**: Situate paper within three streams: LLM capability research, labor automation research, task-based framework.
**Include**:
- Prior exposure measurement methods (Table-style comparison is effective)
- What each prior approach can and cannot capture
- Where your method fits relative to prior work
**Common mistake**: Being too broad (reviewing all of AI history). Stay focused on: task-based automation + LLM-specific studies.
**Reviewer expectation**: Clear articulation of *why* existing measures are insufficient for your specific research question.

---

## 3. Data and Methods
**Purpose**: Describe data sources, annotation protocol, aggregation method, and exposure measure construction.
**Include**:
- Data sources with sizes, dates, coverage limitations
- Exact rubric or adaptation of rubric
- Annotation procedure (who labeled what, how disagreements were resolved)
- Exposure measure formulas with variable definitions
- Aggregation scheme (task → occupation → industry)
**Common mistake**: Insufficient detail on annotation quality and disagreement resolution.
**Reviewer expectation**: Reproducibility — another researcher should be able to replicate the measurement.

---

## 4. Results
**Purpose**: Present main findings with tables, figures, and statistical summaries.
**Include**:
- Summary statistics (occupation-level and task-level exposure distributions)
- Wage/employment distributional results
- Skill importance regressions
- Job zone or education level breakdowns
- Industry-level analysis
**Common mistake**: Reporting numbers without interpreting them. Always explain *what the result means*.
**Reviewer expectation**: Clean tables matching text, with appropriate significance indicators. No hidden multiple testing.

---

## 5. Validation
**Purpose**: Show your new measure is valid by comparing it to prior established measures.
**Include**:
- Regression of new measure on prior measures
- R² and residual variance (how much is new)
- Agreement rates between annotation sources
- Robustness to threshold variation
**Common mistake**: Treating high R² as success. In this field, 28–40% *unexplained* variance is the contribution.
**Reviewer expectation**: Validation against at least 3–4 prior measures; honest discussion of where your measure agrees and disagrees.

---

## 6. Discussion
**Purpose**: Interpret findings through theoretical lens; connect to policy implications.
**Include**:
- Does the technology qualify as a general-purpose technology?
- What does the wage/education distribution of exposure mean for inequality?
- What policy interventions are implied?
- What the findings do NOT mean (avoid overreach)
**Common mistake**: Making causal claims from correlational data (exposure ≠ displacement).
**Reviewer expectation**: Measured interpretation; explicit acknowledgment of what cannot be inferred.

---

## 7. Limitations
**Purpose**: Honest accounting of what the methodology cannot capture.
**Include**:
- Scope limitations (U.S. only, formal economy, specific LLM version)
- Annotation limitations (subjectivity, annotator diversity, rubric sensitivity)
- Temporal limitations (snapshot; capabilities change)
- Framework limitations (task-based decomposition may miss tacit work)
**Common mistake**: Listing only minor limitations while hiding major ones.
**Reviewer expectation**: Major methodological limitations must be disclosed, not just minor ones.

---

## 8. Conclusion
**Purpose**: Restate the key finding, contribution, and call for future work.
**Include**:
- One-sentence statement of main finding
- Two-sentence summary of methodological contribution
- Three specific directions for future research (matching your identified weaknesses)
**Common mistake**: Repeating the abstract verbatim.
**Reviewer expectation**: Synthesis with forward momentum — what should the field do next?

---

## References
**Purpose**: Credit prior work; demonstrate command of literature.
**Include**: All works cited, using the venue's citation style.
**Common mistake**: Missing key prior works (especially Brynjolfsson et al. SML, Frey & Osborne, Webb — reviewers will notice).
**Reviewer expectation**: 40–80 references for a full empirical paper; bias toward recent (2018–2024) LLM-era work.

---

# 13. Publication Strategy Guide

## 13.1 Suitable Venue Types

| Venue Type | Examples | Fit Criteria |
|---|---|---|
| Labor Economics Journals | Journal of Labor Economics, ILR Review, Labour Economics | Strong if main contribution is the exposure dataset and wage distributional analysis |
| AI/Society Conferences | FAccT, AIES, NeurIPS Social Impact Track | Strong if focusing on bias in annotation, inequality implications |
| Economics Working Paper Series | NBER, SSRN | Good for early dissemination before peer review |
| Science/Nature family | Science, Nature Human Behaviour | Only if result is genuinely surprising at population scale with robust causal claims |
| Policy Journals | Journal of Policy Analysis and Management | Strong if policy recommendations are the core output |
| Management Science | Management Science, Organization Science | Strong if firm-level adoption and organizational design is the focus |

## 13.2 Required Baseline Expectations

For publication in a top labor economics or AI policy venue:
- Must compare against at least: SML (Brynjolfsson), Webb (2020), Felten et al. — the standard prior work in this domain.
- Must use O*NET or comparable standardized occupational taxonomy.
- Must include at least two annotation sources or validation datasets.
- Must report Pearson's r or equivalent agreement metric between annotation sources.

## 13.3 Experimental Rigor Level

- **Two independent annotation sources** (human + model, or two human groups)
- **OLS regression validation** against prior exposure measures
- **Multiple robustness checks** (threshold sensitivity, weighting scheme sensitivity, prompt wording sensitivity for LLM annotator)
- **Population-weighted exposure estimates** (not just occupation averages)

## 13.4 Common Rejection Reasons

1. "The threshold of 50% is arbitrary and sensitivity analysis is missing."
2. "Annotators are not representative of the occupations they labeled."
3. "GPT-4 self-assessment cannot be considered objective."
4. "Results are for U.S. only — generalizability is not discussed."
5. "The distinction between exposure and actual displacement is blurred in the discussion."
6. "No causal mechanism links task exposure to wage or employment outcomes."
7. "The E2 category is entirely speculative — software that doesn't exist cannot be evaluated."

## 13.5 Increment Needed for Acceptance

To publish an extension of this paper, you need **at least one** of:
- New geography (non-U.S. occupational taxonomy) with comparative analysis
- New model version (GPT-4o, Claude-3) showing how exposure has changed since 2023
- Domain expert annotation that corrects tech-literacy bias in original labels
- Actual measured productivity data (not theoretical exposure)
- Longitudinal tracking of exposure scores over multiple years
- Cross-sectional wage/employment impact study using exposure as an instrument

---

# 14. Researcher Quick Reference Tables

## 14.1 Key Terminology Table

| Term | Plain Definition |
|---|---|
| Exposure | Whether LLM access would cut task time ≥50% at equal quality |
| E0 | No exposure — LLM doesn't help meaningfully |
| E1 | Direct exposure — bare ChatGPT can cut time ≥50% |
| E2 | LLM+ exposure — needs software built on LLM to cut time ≥50% |
| α (alpha) | Share of E1-labeled tasks in an occupation (lower bound) |
| β (beta) | E1 + 0.5×E2 — primary measure (middle estimate) |
| ζ (zeta) | E1 + E2 — full potential including all LLM-powered software (upper bound) |
| DWA | Detailed Work Activity — granular action from O*NET |
| O*NET | U.S. government occupational database with tasks and skills for 1,016 jobs |
| Job Zone | O*NET classification of jobs by preparation time (1=lowest, 5=highest) |
| General-Purpose Technology (GPT) | Technology that spreads widely, improves continuously, spawns complements |
| SML | Suitability for Machine Learning — Brynjolfsson's prior rubric measure |
| RTI | Routine Task Intensity — Acemoglu & Autor's automation index |
| Skill-Biased TC | Technology that raises demand for skilled over unskilled workers |
| Routine-Biased TC | Technology that displaces workers doing repetitive tasks |
| Baumol's Cost Disease | Rising service costs when productivity gains are uneven across sectors |
| Task Reinstatement | New tasks created by automation (opposite of displacement) |

## 14.2 Important Equations Summary

| Equation | What It Computes |
|---|---|
| β_occ = Σ(w_i × score_i) / Σw_i | Occupation-level exposure score using weighted task average |
| w_i = 2 (core), 1 (supplemental) | O*NET task weighting scheme used in all aggregations |
| score_i(α) = E1 | Binary 0/1 for direct LLM exposure |
| score_i(β) = E1 + 0.5×E2 | Fractional exposure including complementary tools at half weight |
| score_i(ζ) = E1 + E2 | Full exposure including all LLM-powered software |
| LLM_exp = α + Σβ_k × prior_k + ε | OLS validation regression (Table 9) |

## 14.3 Parameter Meaning Table

| Symbol | Meaning | Value/Range |
|---|---|---|
| α | Direct LLM exposure share | 0.14–0.15 (mean, human/GPT-4) |
| β | Primary exposure measure (direct + half of LLM+) | 0.30–0.34 (mean) |
| ζ | Full potential exposure (direct + all LLM-powered) | 0.46–0.55 (mean) |
| R² | Variance explained by prior measures in OLS | 60.7% – 72.8% |
| Agreement | Exact label match between two annotator groups | 65.6% – 91.1% (varies by measure) |
| Pearson's r | Correlation between annotator groups' continuous scores | 0.221 – 0.705 (varies by measure/level) |

## 14.4 Algorithm Flow Summary

```
INPUT: O*NET Task Description + Occupation Name
         |
         v
RUBRIC CHECK: Can bare ChatGPT cut this task time by ≥50%?
         |
    YES──────────────────────── E1 (Direct)
         |
    NO → Can software built on LLM cut this task time by ≥50%?
         |
    YES─────────────────────── E2 (LLM+)
         |
    NO ──────────────────────── E0 (No Exposure)
         |
AGGREGATE: DWA level → Task level → Occupation level
           (core tasks = 2× weight, supplemental = 1× weight)
         |
         v
MERGE: Occupation exposure score + BLS wage/employment data
         |
         v
ANALYZE: Distribution by wage, skill, job zone, industry
```

---

# 15. One-Page Master Summary Card

## Problem
Which U.S. occupations and tasks are most exposed to LLMs — and does this map differently than prior automation risk?

## Idea
Build a new rubric defining "exposure" as LLM-enabled ≥50% time reduction at equivalent quality. Apply it to all O*NET tasks using human annotators + GPT-4 at scale.

## Method
1. O*NET database: 1,016 occupations, 19,265 tasks, 2,087 DWAs
2. Rubric with E0/E1/E2 labels → aggregated to α, β, ζ exposure measures
3. Human annotation + GPT-4 annotation → cross-validated
4. Merged with BLS wage/employment data
5. Regression validation against 6 prior automation exposure measures

## Results

| Finding | Number |
|---|---|
| Workers with ≥10% tasks exposed (β) | 80% of U.S. workforce |
| Workers with ≥50% tasks exposed (β) | 19% of U.S. workforce |
| Tasks directly exposed (α) | ~15% |
| Tasks exposed including LLM-powered software (ζ) | 47–56% |
| Variance unexplained by prior measures | 28–40% |
| Unique variance = new insight about LLMs | Yes |

**Direction of wage effect**: Higher wages → higher exposure (inverts prior automation literature).
**Most exposed skills**: Writing, Programming, Reading Comprehension.
**Least exposed skills**: Science, Critical Thinking.
**Conclusion**: GPTs are general-purpose technologies.

## Weakness
- Arbitrary 50% threshold
- Annotators not occupationally diverse (tech-literate bias)
- GPT-4 self-evaluates its own capabilities (circular)
- U.S. formal economy only
- E2 (LLM-powered software) is speculative — software doesn't yet exist

## Research Opportunity
- Expert annotators for domain-specific tasks
- Cross-national replication (EU, India, Global South)
- Longitudinal tracking as LLM capabilities advance
- Continuous productivity gain measurement (replace binary threshold)
- Actual adoption data vs. theoretical exposure scores

## Publishable Extension
**"Domain-expert validated LLM exposure measurement: correcting annotator bias in the Eloundou et al. framework using occupational insiders"** — Recruit lawyers, nurses, teachers to re-annotate tasks in their own professions. Measure how expert-corrected exposure scores differ from generalist scores. Publish as a methods correction + updated exposure dataset.
