# Intelligent Candidate Discovery & Ranking — Submission README

**Repository:** [Intelligent-Candidate-Discovery](https://github.com/sunoy2004/Intelligent-Candidate-Discovery)  
**Challenge:** Redrob Intelligent Candidate Discovery & Ranking Challenge  
**Target role:** Senior AI Engineer — Founding Team

This document describes our solution, reproduction steps, methodology, and the analysis that informed our ranking system. It is intended for hackathon organizers and reviewers evaluating our submission at Stages 3–5.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Solution Overview](#solution-overview)
3. [Reproduction Instructions](#reproduction-instructions)
4. [Ranking Methodology](#ranking-methodology)
5. [Dataset & Challenge Analysis](#dataset--challenge-analysis)
6. [Job Description Interpretation](#job-description-interpretation)
7. [Feature Engineering Design](#feature-engineering-design)
8. [Behavioral Signal Integration](#behavioral-signal-integration)
9. [Submission Compliance](#submission-compliance)
10. [Risk Mitigation & Design Decisions](#risk-mitigation--design-decisions)

---

## Executive Summary

We built a modular, CPU-only candidate ranking engine that identifies the top 100 candidates from a pool of 100,000 profiles for the released Senior AI Engineer job description.

Our approach combines:

- **Semantic relevance** via chunked HashingVectorizer + TF-IDF on career text (full pool)
- **Structured scoring** across skills, experience, behavior, startup fit, and career growth
- **Trap detection** for keyword stuffers, honeypots, and inconsistent profiles
- **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) on the top 500 candidates
- **Hallucination-safe reasoning** generated from actual profile fields

The final submission is produced at `project/outputs/submission.csv` and passes the official `validate_submission.py` checks. Ranking on the full pool completes in under 3 minutes on CPU after offline precomputation (~6–8 minutes).

**Representative top-ranked profiles:** Lead AI Engineer, Senior AI Engineer, Staff Machine Learning Engineer, Senior NLP Engineer — predominantly India-based, aligned with the JD.

---

## Solution Overview

Our implementation lives under `project/` and follows a two-phase pipeline:

| Phase | Command | Purpose |
|-------|---------|---------|
| Precompute (offline) | `python main.py precompute` | Feature extraction, semantic scores, component scores |
| Rank (≤5 min CPU) | `python main.py rank` | Score fusion, rerank top 500 → final 100, write CSV |

### Architecture

| Stage | Module | Our implementation |
|-------|--------|-------------------|
| 1 | `jd_analysis` | Rule-based JD feature extraction + spaCy supplement |
| 2 | `parser` + `feature_engineering` | Stream JSONL → `candidate_features.parquet` |
| 3 | `embeddings` | Chunked TF-IDF semantic scoring (career-weighted text) |
| 4 | Scorers | Skill ontology, title taxonomy, behavior, startup, growth, traps |
| 5 | `ranking` | Weighted fusion with availability multiplier → top 500 |
| 6 | `reranking` | Cross-encoder rerank → top 100 |
| 7 | `explanations` + `submission` | CSV + validator integration |

### Repository layout

```
project/
├── configs/
├── data/
├── outputs/
├── src/
├── main.py
├── rank.py
├── submission_metadata.yaml
└── requirements.txt
```

### Compute compliance

| Constraint | Our compliance |
|------------|----------------|
| ≤ 5 min ranking runtime | ~3 min on 100K after precompute |
| ≤ 16 GB RAM | Parquet + numpy artifacts |
| CPU only | No GPU during ranking |
| No network during ranking | All models cached locally |

---

## Reproduction Instructions

### Environment setup

```bash
cd project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Phase 1 — Precompute (offline, ~6–8 min on 100K)

```bash
python main.py precompute
```

Produces: `job_features.json`, `candidate_features.parquet`, `semantic_scores.npy`, `component_scores.npz`

Smoke test: `python main.py precompute --limit 500`

### Phase 2 — Rank (≤5 min CPU, no network)

```bash
python main.py rank
```

Reproduction command:

```bash
python rank.py --candidates "../[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" --out outputs/submission.csv
```

### Validation

```bash
python "../[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" outputs/submission.csv
```

---

## Ranking Methodology

We designed the ranker around the JD's stated intent: **match what the role means, not keyword density on skills**.

```
base = 0.30×semantic + 0.20×skill + 0.15×experience + 0.15×behavior + 0.10×startup + 0.10×growth
final = (base − trap_penalty) × availability_modifier
```

Top 500 are reranked with a cross-encoder (`0.70×final + 0.30×rerank_score`).

| Priority | Our approach |
|----------|--------------|
| P0 — Role/title fit | Title taxonomy (ML/IR/engineering vs operational) |
| P0 — Career evidence | Semantic match on career descriptions |
| P0 — Availability | Behavioral multiplier on engagement signals |
| Anti — Traps | Honeypot rules, keyword-stuffer penalties |

---

## Dataset & Challenge Analysis

We analyzed the released candidate pool and challenge materials before building the ranker.

## 1. Repository Structure

### Directory Tree

```
E:\H2S\
├── .gitignore                          # Ignores template PPTX and challenge bundle
├── Idea Submission Template _ Redrob.pptx   # Hackathon idea deck (not challenge data)
└── [PUB] India_runs_data_and_ai_challenge\
    ├── __MACOSX\                       # macOS archive metadata (ignore)
    └── [PUB] India_runs_data_and_ai_challenge\
        └── India_runs_data_and_ai_challenge\   # Core challenge bundle
            ├── README.docx             # Participant onboarding guide
            ├── job_description.docx    # Target role JD (Senior AI Engineer)
            ├── submission_spec.docx    # Rules, format, evaluation pipeline
            ├── redrob_signals_doc.docx # Behavioral signal reference
            ├── candidate_schema.json   # JSON Schema for candidate records
            ├── candidates.jsonl        # Full pool: 100,000 candidates (~465 MB)
            ├── sample_candidates.json  # First 50 candidates (pretty-printed)
            ├── sample_submission.csv   # CSV format reference only (NOT a good ranking)
            ├── submission_metadata_template.yaml  # Portal/repo metadata template
            └── validate_submission.py  # Local CSV format validator
```

**Note:** The README references `candidates.jsonl.gz` and `.md` versions of docs; this bundle ships uncompressed `.jsonl` and `.docx` equivalents.

### Purpose of Each File

| File | Purpose | Critical? |
|------|---------|-----------|
| `job_description.docx` | Defines the role candidates are ranked against | **Yes** |
| `submission_spec.docx` | Submission format, compute limits, scoring weights, stages | **Yes** |
| `redrob_signals_doc.docx` | Explains 23 behavioral signals and trap context | **Yes** |
| `candidates.jsonl` | Full ranking pool (100K records) | **Yes** |
| `candidate_schema.json` | Field definitions, types, enums, constraints | **Yes** |
| `validate_submission.py` | Pre-upload format validation | **Yes** |
| `sample_candidates.json` | Quick schema inspection (50 records) | High |
| `sample_submission.csv` | CSV structure example only | Medium |
| `submission_metadata_template.yaml` | Team/repo/sandbox metadata for Stage 3+ | High |
| `README.docx` | Getting-started guide | Medium |


### Critical Files for the Challenge

1. **`job_description.docx`** — What "fit" means.
2. **`candidates.jsonl`** — the ranking pool.
3. **`submission_spec.docx`** — How you submit and scoring methodology.
4. **`redrob_signals_doc.docx`** — Availability and trap avoidance.
5. **`candidate_schema.json`** — Data contract.
6. **`validate_submission.py`** — Format gate before upload.

---

## 2. Challenge Understanding

### Business Problem

Redrob AI is a Series A talent-intelligence platform. Recruiters search large candidate pools and need **high-quality, hireable matches** for a specific role — not keyword matches on a skills list.

The hackathon simulates building the **intelligence layer** (ranking/retrieval/matching) for a real hiring scenario: finding the top 100 candidates for a **Senior AI Engineer — Founding Team** role from a pool of 100,000 synthetic but realistic profiles.

### Challenge Requirements

- A **CSV** with exactly the **top 100** candidates, best-fit first.
- A **reproducible CPU-only ranker** (≤5 min, ≤16 GB RAM, no network at ranking time).
- **GitHub repo** + **sandbox demo** + **metadata YAML**.
- Rankings that reflect **JD intent**, not surface keyword overlap.
- Incorporation of **behavioral signals** (availability, engagement).
- Avoidance of **traps**: keyword stuffers, honeypots, stale/unavailable candidates.
- Honest, specific **reasoning** per row (strongly recommended for Stage 4).

### Input

| Input | Description |
|-------|-------------|
| `candidates.jsonl` | 100,000 JSON objects, one per line |
| `job_description.docx` | Senior AI Engineer JD with hackathon-specific guidance |
| Signal documentation | How to interpret `redrob_signals` |

Each candidate record includes: `candidate_id`, `profile`, `career_history`, `education`, `skills`, optional `certifications`/`languages`, and `redrob_signals`.

### Output

| Column | Type | Description |
|--------|------|-------------|
| `candidate_id` | `CAND_XXXXXXX` | From dataset |
| `rank` | 1–100 (unique) | Best fit = 1 |
| `score` | float | Non-increasing with rank |
| `reasoning` | string | 1–2 sentence justification (optional but strongly recommended) |

Plus portal metadata and code repository per `submission_spec.docx` Section 10.

### Our Success Criteria

**Quantitative (Stage 2):**

```
Final = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

Scored against **hidden ground truth** relevance tiers (not public during competition).

**Qualitative / gates (Stages 3–5):**

- Reproduces within compute budget.
- Honeypot rate ≤ 10% in top 100.
- Reasoning passes manual review (specific, honest, non-hallucinated).
- Can defend architecture in interview.

**Our target for top-10 quality:** Top 10 should contain genuinely hireable Senior AI Engineers in India (or relocatable), with production ML/IR experience — not marketing managers with 9 AI keywords.

---

## 3. Dataset Scale Summary

| Metric | Value |
|--------|-------|
| Total candidates | 100,000 |
| ID range | `CAND_0000001` – `CAND_0100000` |
| Duplicate IDs | 0 |
| File size (uncompressed) | ~487 MB |
| Countries | 8 (India dominant: 75,113 / 75.1%) |
| Unique skill names | 133 |
| Avg skills per candidate | ~9.6 |
| Avg career history entries | ~3.0 |
| Years of experience range | 1.0 – 16.9 (avg 7.17) |
| Honeypots (documented) | ~80 |
| Certifications empty | 75,019 (75%) |
| GitHub not linked (`github_activity_score = -1`) | 64,637 (65%) |
| No offer history (`offer_acceptance_rate = -1`) | 59,554 (60%) |
| `open_to_work_flag = true` | 35,339 (35%) |
| Template summary text (AI-curious boilerplate) | 63,304 (63%) |

### Title Distribution (Top 15)

Dominated by **non-AI operational roles** (~5,500–5,800 each): Business Analyst, HR Manager, Mechanical Engineer, Accountant, Project Manager, Customer Support, Operations Manager, Content Writer, Sales Executive, Civil Engineer, Graphic Designer, Marketing Manager.

AI-adjacent titles are **rare**:

| Title | Count |
|-------|-------|
| Software Engineer | 3,450 |
| Full Stack Developer | 2,873 |
| Cloud Engineer | 2,836 |
| ML Engineer | 167 |
| Data Scientist | 145 |
| AI Engineer | 21 |
| Senior AI Engineer | 4 |

**Implication:** The pool is intentionally sparse in obvious matches. Precision at top ranks matters enormously.

### Geographic Distribution

- **India:** 75,113 candidates; Pune/Noida combined ~8,469.
- **International:** USA (9,978), Australia, Canada, UK, Germany, Singapore, UAE (~2,400–2,600 each).
- JD prefers **Pune/Noida** and Tier-1 Indian cities — location is a meaningful filter.

---

## 4. Key Observations That Informed Our Design

1. **Keyword matching alone will fail** — explicitly stated in JD and reinforced by traps.
2. **Career history and role descriptions** carry more signal than `skills[]` for many candidates.
3. **63% of summaries** use identical AI-curious template text — summary alone is weak/noisy.
4. **Sample submission is a format trap** — reasoning text does not match actual candidate profiles (verified by cross-check).
5. **No live leaderboard** — optimize locally; max 3 submissions.
6. **Hidden ground truth** uses relevance tiers including tier 0 for honeypots.

---

## _REMOVE_DOC_MAP

| Document | Focus |
|----------|-------|
| `JOB_ANALYSIS.md` | JD requirements, disqualifiers, ideal profile |
| `FEATURE_ANALYSIS.md` | Schema fields, feature categories, data quality |
| `SIGNAL_ANALYSIS.md` | 23 Redrob signals, behavioral traps |
| `SUBMISSION_REQUIREMENTS.md` | CSV rules, compute, evaluation stages |
| `RISKS_AND_INSIGHTS.md` | Traps, evaluation inference, ranking opportunities |

---

## Job Description Interpretation

We parsed the Senior AI Engineer JD and encoded requirements into our scoring pipeline.

**Source:** `job_description.docx`  
**Company:** Redrob AI (Series A, AI-native talent intelligence)  
**Role:** Senior AI Engineer — Founding Team  
**Location:** Pune/Noida, India (Hybrid)  
**Experience band (stated):** 5–9 years (flexible with strong signals)

---

## 1. Role Summary

Own the **intelligence layer** of Redrob's product: ranking, retrieval, and matching systems that determine what recruiters and candidates see. First 90 days: audit existing BM25 + rules stack → ship v2 ranking with embeddings/hybrid retrieval/LLM re-ranking → build evaluation infrastructure (offline benchmarks, A/B tests, feedback loops).

This is a **founding-team, shipper-leaning** role — not pure research, not big-tech ladder climbing.

---

## 2. Structured Requirement Summary

### 2.1 Required Skills (Hard Requirements)

| Category | Requirement | Evidence to Look For |
|----------|-------------|----------------------|
| **Retrieval / IR** | Production embeddings-based retrieval (sentence-transformers, OpenAI embeddings, BGE, E5, etc.) | Career descriptions mentioning embeddings, vector search, index refresh, retrieval quality |
| **Vector / hybrid search** | Operational experience with Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, etc. | Named tools in skills **plus** production context in career history |
| **Python** | Strong production Python | Backend/ML engineering titles, pipeline work |
| **Ranking evaluation** | NDCG, MRR, MAP, offline-to-online correlation, A/B interpretation | Explicit eval framework experience in descriptions |
| **Production ML** | Shipped systems to real users | Product-company tenure, end-to-end ownership language |

### 2.2 Preferred Skills (Nice-to-Have)

| Skill | Weight (inferred) |
|-------|-------------------|
| LLM fine-tuning (LoRA, QLoRA, PEFT) | Medium |
| Learning-to-rank (XGBoost, neural LTR) | Medium |
| HR-tech / recruiting / marketplace | Low–Medium |
| Distributed systems / inference optimization | Medium |
| Open-source AI/ML contributions | Medium |

### 2.3 Experience Requirements

| Dimension | Stated | Practical Interpretation |
|-----------|--------|--------------------------|
| Total experience | 5–9 years | Ideal ~6–8; outliers accepted with strong signals |
| Applied ML tenure | ~4–5 years at product companies | Not pure services/consulting entire career |
| Recent coding | Must have written production code in last 18 months | Penalize pure "architecture only" profiles |
| Pre-LLM ML depth | Required if only recent LangChain/OpenAI work | ≥12 months of shallow LLM-wrapper experience is a negative |

### 2.4 Behavioral / Culture Expectations

| Trait | Signal in Data |
|-------|----------------|
| Async-first, writes a lot | Hard to score directly; reasoning quality in submission |
| Shipper > researcher | Career shows shipped products, not papers-only |
| 3+ year tenure intent | Penalize title-chasing (frequent short stints) |
| Comfortable with ambiguity | Startup/product company background |
| Active / reachable candidate | `open_to_work_flag`, `last_active_date`, `recruiter_response_rate` |

### 2.5 Explicit Disqualifiers

| Disqualifier | Detection Strategy |
|--------------|-------------------|
| Pure research, no production deployment | Academic-only career, no shipping language |
| AI experience = LangChain + OpenAI only (<12 months) | Skills heavy on LangChain, thin ML core |
| Senior who hasn't coded in 18 months | Tech lead/architect titles without recent hands-on signals |
| **Title-chasers** | Many short tenures (<1.5 years), title inflation |
| **Framework enthusiasts** | GitHub/blog-driven profile; tutorials without systems thinking |
| **Consulting-only career** (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) | All roles at SI companies with no product company break |
| **CV/speech/robotics only** | No NLP/IR exposure |
| **Closed-source only 5+ years** | No external validation (OSS, talks, papers) |

### 2.6 Logistics / Hidden Filters

| Factor | JD Statement | Ranking Implication |
|--------|--------------|---------------------|
| Location | Pune/Noida preferred; Hyderabad, Mumbai, Delhi NCR welcome | Boost India + target cities |
| Notice period | Sub-30 ideal; buyout up to 30; 30+ raises the bar | Penalize `notice_period_days` > 30–60 |
| Relocation | Open from Tier-1 Indian cities | `willing_to_relocate` matters for non-Pune/Noida |
| International | Case-by-case, no visa sponsorship | We down-rank non-India unless exceptional |
| Platform activity | "Active on Redrob or clear job-market signal" | Behavioral signals are first-class |

---

## 3. Important Keywords

### Must-Align (Semantic, not just keyword hit)

```
embeddings, retrieval, ranking, vector search, hybrid search,
LLM, fine-tuning, RAG, re-ranking, NDCG, MRR, MAP,
production, shipped, A/B test, offline evaluation,
Python, MLOps, index refresh, embedding drift
```

### Tools (Secondary — operational context matters more)

```
Milvus, Pinecone, Weaviate, Qdrant, FAISS, OpenSearch, Elasticsearch,
sentence-transformers, BGE, E5, LoRA, QLoRA, XGBoost,
Spark, Airflow, Kafka (data infra adjacent)
```

### Anti-Keywords (Presence alone is suspicious)

```
LangChain (without deeper ML stack),
generic "AI tools", "ChatGPT", keyword-stuffed skill lists
```

---

## 4. Domain Expectations

| Domain | Relevance |
|--------|-----------|
| **Information Retrieval / Search** | Core — this is the product |
| **Applied ML / MLOps** | Core — ranking at scale |
| **NLP / LLMs** | High — re-ranking, fine-tuning |
| **HR-tech / recruiting** | Bonus — domain familiarity |
| **Data engineering** | Adjacent — pipeline/feature infra |
| **Computer vision / speech** | Low fit unless NLP/IR crossover |
| **Pure frontend / HR ops / accounting** | Poor fit regardless of AI skills listed |

---

## 5. Hidden Requirements (Reading Between the Lines)

### 5.1 "What the JD Says" vs "What the JD Means"

| Surface Text | Deeper Meaning |
|--------------|----------------|
| "5–9 years" | Judgment and production maturity, not tenure alone |
| "embeddings-based retrieval" | Has debugged real retrieval quality regressions |
| "evaluation frameworks" | Understands ranking metrics — meta requirement for hackathon too |
| "ship in a week" | Pragmatic engineering; aligns with 5-minute CPU constraint |
| "Tier 5 candidates" (hackathon note) | Plain-language profiles with strong career evidence beat keyword profiles |
| "Marketing Manager + perfect AI skills" | **Explicit trap** — must not rank highly |

### 5.2 Ideal Candidate Profile (Organizer-Stated)

- 6–8 years total experience
- 4–5 years applied ML at **product companies** (not pure services)
- Shipped end-to-end ranking/search/recommendation at meaningful scale
- Strong opinions on hybrid retrieval, eval methodology, fine-tune vs prompt
- Located in or willing to relocate to **Noida or Pune**
- Active on platform / job market

**Organizers expect very few perfect matches in 100K** — precision over recall at top ranks.

### 5.3 Hackathon-Specific JD Guidance

> The right answer is not "most AI keywords in skills."  
> Reason about gap between what JD says and what JD means.  
> Weigh behavioral signals — perfect-on-paper but inactive = not hireable.

---

## 6. Requirement Priority Matrix (Inferred)

| Priority | Dimension |
|----------|-----------|
| P0 | Role/title + career trajectory aligned with applied ML/IR engineering |
| P0 | Production retrieval/ranking/search evidence in career history |
| P0 | Availability (active, responds to recruiters, reasonable notice) |
| P1 | 5–9 YOE band, India location / relocatable |
| P1 | Product company experience (not consulting-only) |
| P1 | Python + eval framework familiarity |
| P2 | Preferred skills (LoRA, LTR, OSS) |
| P2 | Education tier (tier_1/tier_2 bonus, not sole signal) |
| P3 | Raw skill keyword overlap |
| Anti | Keyword stuffing, honeypots, consulting-only, CV-only, stale profiles |

---

## 7. JD-to-Dataset Gap Analysis

| JD Need | Dataset Reality |
|---------|-----------------|
| Senior AI Engineers | Only ~691 ML/DS/AI-titled candidates total |
| India + 5–9 YOE | ~25,884 candidates |
| Pune/Noida | ~8,469 candidates |
| Product company exp | ~57,450 (broad heuristic) |
| Consulting-only | ~8,253 candidates (explicit anti-fit) |
| LangChain without ML core | ~2,151 candidates |

**Conclusion:** A strong submission will use **multi-signal scoring** with heavy weight on career narrative, title coherence, and behavioral modifiers — not filter-then-sort on `skills[]`.

---

## Feature Engineering Design

We profiled all 100,000 records and engineered features for ranking.

**Sources:** `candidate_schema.json`, `candidates.jsonl` (100,000 records), `sample_candidates.json` (50 records)

---

## 1. Dataset Inventory

| Dataset | Records | Format | Purpose |
|---------|---------|--------|---------|
| `candidates.jsonl` | 100,000 | JSONL | Full ranking pool |
| `sample_candidates.json` | 50 | JSON array | Schema exploration |
| `sample_submission.csv` | 100 | CSV | Output format reference only |

---

## 2. Top-Level Record Structure

Each candidate is a JSON object with **required** top-level keys:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `candidate_id` | string | Yes | `CAND_XXXXXXX` (7 digits) |
| `profile` | object | Yes | Static profile summary |
| `career_history` | array[1–10] | Yes | Employment timeline |
| `education` | array[0–5] | Yes | Academic background |
| `skills` | array | Yes | Skill proficiency + endorsements |
| `redrob_signals` | object | Yes | 23 behavioral/platform signals |
| `certifications` | array | No | Often empty (75% of records) |
| `languages` | array | No | Always populated in full dataset |

---

## 3. Field-by-Field Analysis

### 3.1 `profile` Object

| Field | Type | Missing | Notes |
|-------|------|---------|-------|
| `anonymized_name` | string | 0% | Not useful for ranking |
| `headline` | string | 0% | Short title + tagline; moderate signal |
| `summary` | string | 0% | **Noisy** — 63% share identical AI-curious template |
| `location` | string | 0% | City, region; critical for JD location fit |
| `country` | string | 0% | 75% India |
| `years_of_experience` | float (0–50) | 0% | Range 1.0–16.9, avg 7.17; JD wants 5–9 |
| `current_title` | string | 0% | **High signal** — primary role classification |
| `current_company` | string | 0% | Consulting detection (TCS, Wipro, etc.) |
| `current_company_size` | enum | 0% | 8 buckets from `1-10` to `10001+` |
| `current_industry` | string | 0% | IT Services dominant (29,881) |

**Redundant with career_history:** `current_title`, `current_company`, `current_industry`, `current_company_size` duplicate the `is_current=true` entry (when consistent).

**Potentially useful engineered features:**
- JD location match (Pune/Noida/Delhi NCR/Mumbai/Hyderabad)
- YOE band fit (5–9 sweet spot)
- Title taxonomy (ML/IR/engineering vs operational)
- Consulting-firm flag

### 3.2 `career_history` Array (avg 3.0 entries)

| Field | Type | Description |
|-------|------|-------------|
| `company` | string | Employer name |
| `title` | string | Role title at that company |
| `start_date` | date | ISO `YYYY-MM-DD` |
| `end_date` | date/null | null if current |
| `duration_months` | int | Tenure length |
| `is_current` | bool | Current role flag |
| `industry` | string | Sector |
| `company_size` | enum | Same buckets as profile |
| `description` | string | **Richest text field** — responsibilities, tech, outcomes |

**Critical insight:** Descriptions often contain **different role narratives than `current_title`** (synthetic inconsistency by design). Example: Operations Manager title with Marketing Manager / mechanical engineering descriptions in history. **Career descriptions > title for true fit assessment.**

**Honeypot signal:** Timeline inconsistencies (e.g., total tenure >> stated YOE) — ~23 candidates flagged by simple heuristic; organizers cite ~80 honeypots total.

**Useful features:**
- Count of product-company roles (Software, Fintech, E-commerce, EdTech)
- Retrieval/ranking/embedding mentions in descriptions
- Title progression / job hopping frequency
- Consulting-only career flag (~8,253 candidates)
- Tenure stability (avg duration, short-stint count)

### 3.3 `education` Array

| Field | Type | Distribution / Notes |
|-------|------|----------------------|
| `institution` | string | Varied; includes real and generic names |
| `degree` | string | B.E., B.Tech, M.E., MBA, etc. |
| `field_of_study` | string | CS, IT, Engineering, etc. |
| `start_year` | int | 1970–2030 |
| `end_year` | int | 1970–2035 |
| `grade` | string/null | GPA/percentage; 0% missing in dataset |
| `tier` | enum | **tier_1** (6,852), **tier_2** (27,821), **tier_3** (53,220), **tier_4** (51,885) |

**No empty education arrays** in full dataset.

**Tier interpretation:** Internal prestige proxy. JD mentions "Tier 5 plain language" candidates — likely strong career evidence without buzzwords, not necessarily `tier_5` enum (which doesn't exist in schema; tiers are 1–4 + unknown).

### 3.4 `skills` Array (avg ~9.6 skills)

| Field | Type | Distribution |
|-------|------|--------------|
| `name` | string | 133 unique skills |
| `proficiency` | enum | beginner (379K), intermediate (470K), advanced (110K), expert (1,311) |
| `endorsements` | int | 0+ |
| `duration_months` | int | Months using skill; **trust signal** vs keyword stuffing |

**Proficiency distribution is heavily beginner/intermediate** — "expert" is rare (1,311 total skill entries).

**Top skills by frequency (~12K each):** HTML, Databricks, Redux, Terraform, Angular, etc. — suggests **synthetic uniform sprinkling** across pool; raw frequency is not discriminative.

**AI-relevant skills (moderate frequency ~4.7K–5.2K):** LangChain, LLMs, RAG, Fine-tuning LLMs, MLOps, Milvus, Feature Engineering, etc.

**Redundant/noisy:**
- Skills list often **contradicts title** (Accountant with RAG, LoRA, GANs)
- High endorsement count with beginner proficiency
- Expert proficiency with 0 `duration_months` (84 instances)

**High-value features:**
- AI core skill count **weighted by** `duration_months` × proficiency
- Endorsement-to-proficiency trust ratio
- Skill assessment cross-check (see signals)
- JD-specific skill groups (retrieval, vector DB, eval metrics)

### 3.5 `certifications` (optional)

- Empty in **75,019** records (75%)
- When present: `name`, `issuer`, `year`
- Low discriminative power due to sparsity

### 3.6 `languages`

- Populated for all records in full dataset
- `language` + `proficiency` (basic → native)
- English proficiency useful; Hindi common for India pool

---

## 4. Feature Categories

### Profile Features
- `headline`, `location`, `country`, `years_of_experience`
- `current_title`, `current_company`, `current_company_size`, `current_industry`
- `summary` (use with caution — templated)

### Skills
- Skill names, proficiency, endorsements, duration_months
- Derived: JD skill coverage, trust-weighted skill score

### Experience
- `career_history[]` descriptions, industries, tenures
- Title progression, product vs services mix

### Education
- Degree, field, institution tier
- tier_1/tier_2 as mild positive signal

### Behavioral Signals
- Full `redrob_signals` object (see `SIGNAL_ANALYSIS.md`)

### Career Progression
- Number of roles, tenure lengths, title escalation
- Job hopping / title-chasing detection

### Activity Signals
- `last_active_date`, `profile_views_received_30d`, `search_appearance_30d`
- `applications_submitted_30d`, `saved_by_recruiters_30d`

### Availability Signals
- `open_to_work_flag`, `notice_period_days`, `recruiter_response_rate`
- `interview_completion_rate`, `preferred_work_mode`, `willing_to_relocate`

---

## 5. Data Quality Assessment

### Missing Data

| Field | Missing Rate |
|-------|--------------|
| Core required fields | 0% |
| `certifications` | 75% empty arrays |
| `skill_assessment_scores` | 75.8% empty dict |
| `github_activity_score = -1` | 64.6% (no GitHub) |
| `offer_acceptance_rate = -1` | 59.6% (no offer history) |

### Inconsistencies (Intentional Traps)

| Issue | Scale | Example |
|-------|-------|---------|
| Summary vs title mismatch | ~58,185 summaries mention "marketing manager" for non-marketing titles | Template-driven |
| Skills vs title mismatch | ~1,888 non-tech titles with 7+ AI keywords | Keyword stuffer trap |
| Career description vs title | Common in samples | Ops Manager with marketing/mechanical descriptions |
| Assessment vs claimed proficiency | ~3,008 candidates | Advanced skill, assessment < 30 |
| Impossible timelines | ~23+ (heuristic) | YOE inconsistent with total tenure |
| Expert + 0 duration | 84 skill entries | Honeypot indicator |

### Duplicates
- **0 duplicate `candidate_id`** values

### Outliers
- YOE max 16.9 (below schema max 50)
- `notice_period_days` up to 150 (JD prefers <30)
- Very low `recruiter_response_rate` (4,221 below 0.1)

### Potential Data Leakage
- **No explicit relevance labels** in released data (ground truth is hidden)
- `saved_by_recruiters_30d` and `search_appearance_30d` could partially correlate with hidden relevance if synthetically constructed — treat as signals, not labels
- Sample submission IDs/reasoning do **not** reflect ground truth

### Noisy Features
- `summary` (63% templated)
- Raw skill presence (uniform synthetic distribution)
- `endorsements` without duration/proficiency context
- Fictional companies (Dunder Mifflin, Stark Industries, Acme Corp)

---

## 6. Strong vs Weak Features (Design)

### Strong Ranking Signals
1. **`career_history[].description`** — production retrieval/ranking/embedding language
2. **`current_title`** + title taxonomy — engineering/ML vs operational roles
3. **`redrob_signals`** — availability and engagement (modifier)
4. **`years_of_experience`** — band fit 5–9
5. **`location` + `country`** — India, Pune/Noida
6. **Product company industries** in career history
7. **`skill_assessment_scores`** — when present, validates claimed skills
8. **Skill `duration_months` + proficiency** — anti-stuffing

### Weak Signals
1. **`summary`** — heavily templated
2. **Raw skill count / keyword overlap** — trap-prone
3. **`certifications`** — sparse
4. **`anonymized_name`, `headline`** — minimal JD signal
5. **Uniform high-frequency skills** (HTML, Databricks, etc.)

### Potential Engineered Features
- JD embedding similarity to career descriptions (not skills alone)
- Retrieval/ranking keyword density in career text (weighted)
- Consulting-only penalty
- Title–skill coherence score
- Behavioral availability multiplier
- Notice period penalty curve
- Education tier bonus (small)
- Honeypot/consistency checker (timeline, expert+0mo)
- Title-chasing / job-hop detector from `career_history`

### Potential Semantic Features
- Embedding similarity: JD ↔ career descriptions
- Embedding similarity: JD ↔ full profile (secondary)
- NER/extraction: companies, technologies, metrics from descriptions
- Role classification from text (ML engineer vs marketing)

### Potential Behavioral Features
- Composite engagement score (views, saves, search appearances)
- Response reliability (`recruiter_response_rate`, `avg_response_time_hours`)
- Recency (`last_active_date` decay)
- `open_to_work_flag` × engagement interaction
- Interview/offer completion rates (when not -1)

---

## 7. Schema Constraints Worth Enforcing Locally

- `candidate_id` pattern: `^CAND_[0-9]{7}$`
- `career_history`: 1–10 items
- `education`: 0–5 items
- `company_size` enums fixed
- `proficiency` enums: beginner → expert
- `github_activity_score`: -1 or 0–100
- `offer_acceptance_rate`: -1 or 0–1

We validate candidate records against `candidate_schema.json` during preprocessing.

---

## Behavioral Signal Integration

We integrated all 23 Redrob signals as an availability modifier on fit scoring.

**Source:** `redrob_signals_doc.docx`, `candidate_schema.json`, `candidates.jsonl` (100K statistical profile)

---

## 1. Why Behavioral Signals Exist

In real recruiting platforms, **observed behavior** often predicts hireability better than static profile text:

- Will the candidate respond to outreach?
- Are they actively looking?
- Do recruiters save their profile?
- Do they complete assessments and interviews?

The JD explicitly states:

> A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available.

Signals should act as a **multiplier or modifier** on skill/JD-match scoring — not replace it.

---

## 2. The 23 Signals — Reference Table

| # | Signal | Type / Range | What It Measures | Inferred Importance |
|---|--------|--------------|------------------|---------------------|
| 1 | `profile_completeness_score` | 0–100 | Profile fill completeness | Medium — proxy for seriousness |
| 2 | `signup_date` | date | Platform tenure | Low alone; context for activity |
| 3 | `last_active_date` | date | Last login | **High** — recency / availability |
| 4 | `open_to_work_flag` | bool | Self-reported availability | **High** — only 35% true in pool |
| 5 | `profile_views_received_30d` | int ≥ 0 | Recruiter views (30d) | Medium — market interest |
| 6 | `applications_submitted_30d` | int ≥ 0 | Recent applications | Medium — job-seeking intensity |
| 7 | `recruiter_response_rate` | 0.0–1.0 | Reply rate to recruiters | **High** — reachability |
| 8 | `avg_response_time_hours` | float ≥ 0 | Median response latency | Medium — responsiveness |
| 9 | `skill_assessment_scores` | dict[str, 0–100] | Platform skill tests | **High when present** — validation |
| 10 | `connection_count` | int ≥ 0 | Network size | Low–Medium |
| 11 | `endorsements_received` | int ≥ 0 | Total endorsements | Low alone (can mislead) |
| 12 | `notice_period_days` | 0–180 | Stated notice | **High** — JD wants <30 ideal |
| 13 | `expected_salary_range_inr_lpa` | {min, max} | Salary expectations (INR LPA) | Medium — budget fit (unstated in JD) |
| 14 | `preferred_work_mode` | onsite/hybrid/remote/flexible | Work mode preference | Medium — JD is hybrid |
| 15 | `willing_to_relocate` | bool | Relocation openness | **High** for non-Pune/Noida India |
| 16 | `github_activity_score` | -1 or 0–100 | GitHub activity (12mo) | Medium — -1 = 64.6% of pool |
| 17 | `search_appearance_30d` | int ≥ 0 | Recruiter search impressions | Medium — discoverability |
| 18 | `saved_by_recruiters_30d` | int ≥ 0 | Recruiter bookmarks (30d) | **High** — strong social proof |
| 19 | `interview_completion_rate` | 0.0–1.0 | Attended / scheduled interviews | Medium–High |
| 20 | `offer_acceptance_rate` | -1 or 0–1 | Historical offer acceptance | Low–Medium (-1 = 59.6%) |
| 21 | `verified_email` | bool | Email verified | Low — table stakes |
| 22 | `verified_phone` | bool | Phone verified | Low–Medium |
| 23 | `linkedin_connected` | bool | LinkedIn linked | Low–Medium |

---

## 3. Signal Deep-Dives

### 3.1 Availability Cluster (P0)

**Signals:** `open_to_work_flag`, `last_active_date`, `recruiter_response_rate`, `notice_period_days`

| Statistic | Value |
|-----------|-------|
| `open_to_work_flag = true` | 35,339 (35.3%) |
| Stale (>6 months inactive as of Jun 2026) | 28,615 (28.6%) |
| `recruiter_response_rate` < 0.1 | 4,221 (4.2%) |
| Avg notice period | 87.4 days |
| Notice range | 0–150 days |

**Ranking impact:** We down-rank high skill-match candidates with stale activity, low response rate, or 90+ day notice unless other signals are exceptional. JD states 30+ day notice "raises the bar."

**Trap:** "Behavioral twins" (per README) — candidates with identical skill profiles but divergent behavioral signals. The reachable twin should rank higher.

### 3.2 Skill Validation Cluster (P0)

**Signals:** `skill_assessment_scores`, `endorsements_received`, `github_activity_score`

| Statistic | Value |
|-----------|-------|
| Non-empty `skill_assessment_scores` | 24,244 (24.2%) |
| `github_activity_score = -1` | 64,637 (64.6%) |
| Assessment vs proficiency mismatch | ~3,008 candidates |

**Top assessed skills:** YOLO, Feature Engineering, CNN, Weights & Biases, MLOps, BentoML, etc.

**Ranking impact:** When assessments exist, use them to **validate or penalize** claimed `skills[]`. Low assessment + advanced proficiency = red flag (keyword stuffer). GitHub score supports engineering credibility when not -1.

### 3.3 Market Interest Cluster (P1)

**Signals:** `profile_views_received_30d`, `search_appearance_30d`, `saved_by_recruiters_30d`

**Ranking impact:** `saved_by_recruiters_30d` is strongest "recruiter crowd wisdom" signal. High saves + high JD match = confidence boost. Views/searches alone are weaker (passive exposure).

### 3.4 Professionalism Cluster (P2)

**Signals:** `interview_completion_rate`, `offer_acceptance_rate`, `verified_*`, `linkedin_connected`

| Statistic | Value |
|-----------|-------|
| All three verified (email + phone + LinkedIn) | 16,387 (16.4%) |
| `offer_acceptance_rate = -1` | 59,554 (59.6%) |

**Ranking impact:** Low `interview_completion_rate` suggests flakiness. Treat `offer_acceptance_rate = -1` as missing, not negative.

### 3.5 Logistics Cluster (P1)

**Signals:** `preferred_work_mode`, `willing_to_relocate`, `expected_salary_range_inr_lpa`

| Work mode distribution | ~25K each (balanced) |
|------------------------|----------------------|

**Ranking impact:** Hybrid/flexible preferred for JD. Relocation flag critical for Bangalore/Hyderabad candidates targeting Pune/Noida.

---

## 4. Hidden Signals & Trap Patterns

### 4.1 Documented Trap Types (README + submission_spec)

| Trap Type | Description | Detection Hints |
|-----------|-------------|-----------------|
| **Keyword stuffers** | Many AI skills, wrong title | Non-tech title + 7–9 AI skills + low assessment |
| **Plain-language Tier 5s** | No buzzwords but strong career | Ranking/search/rec in career text without RAG/Pinecone keywords |
| **Behavioral twins** | Same skills, different engagement | Compare signal envelopes pairwise |
| **Honeypots (~80)** | Impossible profiles (tier 0 GT) | Timeline math, expert+0mo skills, tenure at pre-founding companies |
| **Sample submission** | Format-only; reasoning is fabricated | We do not mimic the sample scoring logic |

### 4.2 Keyword Stuffer Profile (Observed)

- **~894** candidates with 8+ AI-keyword skills
- **~1,888** non-tech titles with 7+ AI keywords
- Example pattern: `Accountant` / `HR Manager` / `Content Writer` + RAG, LLMs, LoRA, GANs
- Often paired with **high `recruiter_response_rate` in sample CSV reasoning** but wrong titles in actual data

**Organizer intent:** Systems that count "AI core skills" without title/career coherence will rank traps highly and fail honeypot + NDCG checks.

### 4.3 Honeypot Heuristics (Partial — ~80 total)

Simple rules detect subset (~23 timeline anomalies; 84 expert+0mo):

- Total `duration_months` >> `years_of_experience` × 12
- "Expert" proficiency with 0 `duration_months`
- Experience at companies for longer than plausible org age (per spec narrative)
- 10 expert skills with 0 years used (spec example)

**Stage 3 gate:** >10% honeypots in top 100 → **disqualified**.

### 4.4 Assessment–Skill Divergence

When `skill_assessment_scores[NLP] = 38.8` but skill listed as `advanced`, trust assessment over self-report. Sample candidate CAND_0000001 demonstrates this pattern.

### 4.5 Sentiment Mismatch in Sample CSV

Sample submission ranks CAND_0004989 #1 with reasoning "HR Manager, 6.1 yrs" but actual profile is **Project Manager, 12.6 yrs**. This teaches:

- Reasoning must match profile facts (Stage 4 hallucination check)
- Sample scoring logic is **intentionally wrong**

---

## 5. Recommended Signal Usage Model (Conceptual)

```
final_score = jd_fit_score × availability_modifier × trust_modifier

jd_fit_score     ← career text, title, YOE, location, product exp
availability_mod ← f(last_active, open_to_work, response_rate, notice)
trust_modifier   ← f(assessments, skill duration, honeypot checks)
```

**Do not** let behavioral signals alone promote wrong-role candidates.  
**Do** let behavioral signals demote right-role but unreachable candidates.

---

## 6. Signal Importance Ranking (Inferred)

| Tier | Signals |
|------|---------|
| **Critical** | `last_active_date`, `recruiter_response_rate`, `open_to_work_flag`, `notice_period_days`, `skill_assessment_scores` |
| **High** | `saved_by_recruiters_30d`, `interview_completion_rate`, `willing_to_relocate` |
| **Medium** | `profile_views_received_30d`, `search_appearance_30d`, `github_activity_score`, `avg_response_time_hours`, `preferred_work_mode` |
| **Low** | `connection_count`, `endorsements_received`, `signup_date`, `verified_*`, `profile_completeness_score` |
| **Conditional** | `offer_acceptance_rate` (only when ≠ -1), `applications_submitted_30d` |

---

## 7. Assumptions

1. Signals are **synthetically correlated** with hidden relevance but not perfect labels.
2. Behavioral twins exist though not auto-detected in this analysis — expect near-duplicate skill vectors with divergent `redrob_signals`.
3. Ground truth tier 0 maps to honeypots; tiers 3+ likely "relevant" per P@10 definition.
4. `-1` sentinel values mean **missing**, not bad.

---

## Submission Compliance

Our submission meets the released format, compute, and evaluation requirements.

**Source:** `submission_spec.docx`, `submission_metadata_template.yaml`, `validate_submission.py`, `sample_submission.csv`

---

## 1. Deliverables Overview

Submission has **three required parts**:

| Part | Description |
|------|-------------|
| **10.1 CSV file** | Top-100 ranked candidates |
| **10.2 Portal metadata** | Team, contacts, GitHub, sandbox, AI declaration |
| **10.3 Code repository** | Reproducible ranker + README + `submission_metadata.yaml` |

---

## 2. CSV Format Specification

### 2.1 File Requirements

| Requirement | Value |
|-------------|-------|
| Filename | Registered participant ID, e.g. `team_xxx.csv` |
| Encoding | UTF-8 |
| Extension | `.csv` only (not `.xlsx`, `.json`) |
| Header row | 1 |
| Data rows | **Exactly 100** (rows 2–101) |

### 2.2 Columns (Strict Order)

```csv
candidate_id,rank,score,reasoning
```

| Column | Type | Rules |
|--------|------|-------|
| `candidate_id` | string | Must match `^CAND_[0-9]{7}$`; must exist in `candidates.jsonl`; unique |
| `rank` | int | 1–100, each integer exactly once |
| `score` | float | Non-increasing as rank increases (rank 1 ≥ rank 2 ≥ … ≥ rank 100) |
| `reasoning` | string | Optional but **strongly recommended** |

### 2.3 Tie-Breaking Rules

- Tied scores allowed
- Ranks must still be unique 1–100
- Equal scores: break ties deterministically (secondary model signal or **`candidate_id` ascending**)

### 2.4 Validator Checks (`validate_submission.py`)

The local validator enforces:

1. `.csv` extension
2. Exact header match
3. Exactly 100 non-empty data rows
4. 4 columns per row
5. Valid `candidate_id` format, no duplicates
6. Ranks 1–100, no duplicates
7. Scores are floats
8. Scores non-increasing by rank
9. Equal-score tie-break: lower `candidate_id` must have lower rank

**Our pre-upload validation:**
```bash
python validate_submission.py team_xxx.csv
```

---

## 3. Ranking Constraints

| Constraint | Limit |
|------------|-------|
| Candidates ranked | Top **100 only** (not full 100K) |
| Pool source | Released `candidates.jsonl` only |
| Submissions allowed | **3 max**; last valid submission counts |
| Live leaderboard | **None** — scores revealed after close |

---

## 4. Compute Constraints (Stage 3 Reproduction)

| Resource | Limit |
|----------|-------|
| Wall-clock runtime | ≤ **5 minutes** |
| Memory | ≤ **16 GB RAM** |
| Compute | **CPU only** (no GPU during ranking) |
| Network | **Off** — no API calls (OpenAI, Anthropic, Cohere, Gemini, etc.) |
| Disk (intermediate) | ≤ **5 GB** |

### Allowed vs Forbidden

| Allowed | Forbidden |
|---------|-----------|
| Pre-computed embeddings/indexes (offline) | Hosted LLM API calls at ranking time |
| Local CPU models | GPU inference during ranking |
| Rule-based + feature scoring | Per-candidate cloud inference |
| Compact local models | 100K × LLM calls |

**Pre-computation:** May exceed 5 minutes if documented; **ranking step** producing CSV must finish within budget.

**Template reproduce command:**
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

---

## 5. Score Constraints

- `score` is a float (no fixed range mandated, but sample uses 0.9920 down to 0.2000)
- Must be **monotonically non-increasing** with rank
- All identical scores → flagged as "model isn't differentiating"
- Scores increasing with rank → **rejected**

Sample score pattern: decrement by 0.008 per rank (format illustration only).

---

## 6. Reasoning Requirements (Stage 4)

For top-N submissions, **10 random rows** reviewed against:

| Check | Requirement |
|-------|-------------|
| Specific facts | YOE, title, named skills, signal values from actual profile |
| JD connection | Links to specific JD requirements |
| Honest concerns | Acknowledges gaps (notice period, consulting background, etc.) |
| No hallucination | Every claim exists in candidate record |
| Variation | Not templated across rows |
| Rank consistency | Tone matches rank (rank 5 ≠ harsh criticism; rank 95 ≠ glowing) |

### Penalized Patterns
- Empty reasoning
- Identical strings for all rows
- Template with swapped name only
- Skills/employers not in profile
- Reasoning contradicting rank

### Sample Submission Warning

Cross-validation shows sample `reasoning` column contains **fabricated titles and YOE** that do not match `candidates.jsonl`. We use the sample file for CSV structure validation only.

---

## 7. Evaluation Scoring (Stage 2)

### 7.1 Metrics & Weights

| Metric | Weight | Measures |
|--------|--------|----------|
| **NDCG@10** | 0.50 | Top-10 ranking quality |
| **NDCG@50** | 0.30 | Top-50 quality |
| **MAP** | 0.15 | Precision across relevance levels |
| **P@10** | 0.05 | Fraction of top-10 that are relevant (tier 3+) |

```
Composite = 0.50×NDCG@10 + 0.30×NDCG@50 + 0.15×MAP + 0.05×P@10
```

### 7.2 Tiebreaks (identical composite)
1. Higher P@5
2. Higher P@10
3. Earlier submission timestamp

### 7.3 Ground Truth
- Hidden relevance tiers (not released)
- Honeypots forced to **tier 0**
- Scored once after submissions close

---

## 8. Evaluation Pipeline (Stages)

| Stage | Activity | Elimination Triggers |
|-------|----------|-------------------|
| **1. Format validation** | Auto-validator | Any Section 3 violation |
| **2. Scoring** | Composite vs hidden GT | Below cutoff for Stage 3 |
| **3. Reproduction + honeypot** | Docker sandbox, code review | Can't reproduce; honeypot >10% in top 100; bad/missing repo |
| **4. Manual review** | Reasoning, methodology, git history | Bad reasoning; LLM-only codebase; flat git history |
| **5. Interview** | 30-min defend-your-work | Can't explain architecture; didn't build it |

---

## 9. Portal Metadata (`submission_metadata_template.yaml`)

### Required Fields

| Field | Notes |
|-------|-------|
| `team_name` | Leaderboard display |
| `primary_contact` | name, email, phone |
| `team_members[]` | name, email, role |
| `github_repo` | Must be reachable; private OK with Stage 3 access |
| `sandbox_link` | HuggingFace Spaces, Streamlit Cloud, Replit, Colab, Docker, Binder |
| `reproduce_command` | Single end-to-end command |
| `compute.*` | platform, CPU, RAM, Python version, OS |
| `uses_gpu_for_inference` | **Must be false** |
| `has_network_during_ranking` | **Must be false** |
| `ai_tools_used` | Transparency only — not penalized |
| `declarations.*` | Spec read, original work, no collusion, reproduction tested |

### Optional but Recommended
- `methodology_summary` (≤200 words)
- `honeypot_check_done` flag

### Acceptable Sandbox Platforms
- HuggingFace Spaces (free tier OK)
- Streamlit Cloud
- Replit (public)
- Google Colab (end-to-end notebook link)
- `docker pull` + `docker run` (public registry)
- Binder

Sandbox must:
- Accept ≤100 candidate sample
- Run end-to-end → ranked CSV
- Complete within CPU budget

---

## 10. Repository Requirements

Must include:
- `README.md` with setup + reproduce instructions
- Full source code (no hidden manual steps)
- Pre-computed artifacts OR scripts to generate them
- `requirements.txt` / `pyproject.toml` with pinned versions
- `submission_metadata.yaml` at repo root

---

## 11. Common Rejection Checklist

- [ ] 99 or 101 data rows (not exactly 100)
- [ ] Ranks starting at 0
- [ ] Duplicate `candidate_id` or `rank`
- [ ] Invalid/typo `candidate_id`
- [ ] All scores identical
- [ ] Scores increasing with rank
- [ ] Wrong file format (.xlsx, .json)
- [ ] Missing sandbox link
- [ ] Ranking uses network/GPU
- [ ] Honeypot rate >10% in top 100
- [ ] Reasoning hallucinations at Stage 4

---

## 12. Candidate Count Requirements Summary

| Requirement | Value |
|-------------|-------|
| Input pool size | 100,000 |
| Output ranked | **100** (exactly) |
| Rank range | 1–100 (inclusive, complete permutation) |
| Unique candidates | 100 distinct IDs |
| Must exist in pool | All 100 |

---

## Risk Mitigation & Design Decisions

We identified adversarial patterns and built explicit mitigations into the pipeline.

**Perspective:** Pre-implementation research for Redrob Intelligent Candidate Discovery & Ranking Challenge  
**Assumption:** All inference about ground truth and honeypot placement is speculative unless stated in official docs.

---

## 1. Executive Summary

This challenge rewards **JD-informed, multi-signal ranking** over **keyword retrieval**. The dataset is adversarial by design: sparse true positives (~hundreds of plausible ML/IR engineers), dense keyword-stuffed decoys, behavioral twins, ~80 honeypots, and noisy templated text. Success requires ranking **role coherence + career evidence + availability** — within strict CPU/no-network constraints — while producing auditable reasoning.

---

## 2. Evaluation Inference

### 2.1 Likely Ground Truth Structure (Assumed)

| Tier | Likely Meaning | Evidence |
|------|----------------|----------|
| 0 | Honeypots, impossible profiles | submission_spec §7 |
| 1–2 | Irrelevant / wrong role | Keyword stuffers, consulting-only, wrong domain |
| 3+ | Relevant (for P@10) | submission_spec: "tier 3+" = relevant |
| 4–5 (inferred) | Strong / ideal fit | JD ideal profile: product ML, retrieval, India, available |

**NDCG@10 weighted 50%** → top-10 precision is the primary battlefield. A single honeypot or HR Manager at rank 3 disproportionately hurts score.

### 2.2 Possible Hidden Scoring Elements

| Element | Likelihood | Rationale |
|---------|------------|-----------|
| Graded relevance (not binary) | High | NDCG + MAP imply multi-level labels |
| Behavioral availability in labels | High | JD + signals doc emphasize hireability |
| Title/career coherence in labels | High | Explicit anti-keyword-stuffer guidance |
| Location weighting | Medium | JD stresses Pune/Noida |
| Education tier bonus | Low–Medium | Minor signal in synthetic data |
| Reasoning quality | Stage 4 only | Not in composite formula but gates finalists |

### 2.3 What Probably Does NOT Score Well

- Pure BM25 / keyword overlap on `skills[]`
- Counting "AI core skills" (sample submission pattern — intentionally bad)
- Ignoring `career_history[].description`
- Promoting high-skill but inactive candidates
- Copying sample submission reasoning or score decay pattern

---

## 3. Trap Candidates — Risk Catalog

### 3.1 Keyword Stuffers (~1,900 high-risk)

**Profile:** Non-tech `current_title` (HR Manager, Accountant, Content Writer) + 7–9 AI skills (RAG, LLMs, LoRA, etc.)

**Why dangerous:** High skill overlap with JD keywords → naive rankers put them in top 10.

**Mitigation:** Title taxonomy gate; weight skills by `duration_months` × proficiency; cross-check assessments.

### 3.2 Honeypots (~80)

**Profile:** Subtle impossibilities — timeline contradictions, expert skills with 0 duration, tenure exceeding company age.

**Gate:** >10% in top 100 → **Stage 3 disqualification** (~11+ honeypots).

**Mitigation:** Consistency rules on YOE vs tenure sum, expert+0mo, founding-date checks if company metadata available.

### 3.3 Plain-Language Strong Fits (Tier 5 — Opportunity)

**Profile:** No RAG/Pinecone buzzwords; career descriptions mention recommendation systems, search, ranking at product companies.

**Risk:** Underserved by keyword systems — **rank too low**.

**Mitigation:** Semantic matching on career text; retrieval/ranking phrase detection in descriptions.

### 3.4 Behavioral Twins (Documented)

**Profile:** Near-identical skills/profile; divergent `redrob_signals`.

**Risk:** Picking unreachable twin.

**Mitigation:** Availability modifier on final score.

### 3.5 Templated Summaries (63,304 candidates)

**Profile:** Identical "AI-curious" paragraph regardless of role.

**Risk:** Summary-based embeddings pollute similarity.

**Mitigation:** Down-weight `summary`; prioritize `career_history.description`.

### 3.6 Consulting-Only Careers (~8,253)

**Profile:** Entire history at TCS/Wipro/Infosys/Accenture/Cognizant/etc.

**JD:** Explicit soft disqualifier.

**Mitigation:** Penalty unless strong product-company break elsewhere.

### 3.7 LangChain-Only (~2,151)

**Profile:** LangChain present without PyTorch/TF/ML/MLOps/RAG core.

**JD:** Matches "recent LLM wrapper" anti-pattern.

**Mitigation:** Require depth skills or career evidence of pre-LLM ML.

### 3.8 Sample Submission Mirage

**Verified:** Rank 1 reasoning says "HR Manager, 6.1 yrs" but `CAND_0004989` is **Project Manager, 12.6 yrs**.

**Risk:** Teams copy sample logic → bad rankings + Stage 4 hallucination flags.

---

## 4. Leaderboard & Competition Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| No live feedback | Can't A/B via portal | Local validation harness; manual spot checks |
| 3 submission cap | Low — last wins | Validate thoroughly before submit |
| Hidden GT | Unknown true performance | Diverse sanity checks on top 100 |
| Overfit to public docs | Medium | Docs describe traps explicitly — trust them |
| Stage 3 reproduction fail | Disqualify despite good score | Test on 16GB CPU box early |
| Reasoning audit fail | Eliminate at Stage 4 | Generate reasoning from same features used to rank |
| Interview fail | Top X only | Architecture walkthrough in interview |

---

## 5. Data Quality Risks

| Issue | Scale | Impact |
|-------|-------|--------|
| Summary/template noise | 63% | Mislead text rankers |
| Skills uniform distribution | ~12K per common skill | Poor discriminative power |
| Title ≠ career narrative | Widespread | Title-only filtering fails |
| Missing assessments | 75.8% | Validation gap |
| Missing GitHub | 64.6% | Neutral, not negative |
| Fictional employers | Common | Can't verify company founding dates easily |
| Assessment-skill mismatch | ~3,008 | Trust assessments when present |

---

## 6. Ranking Opportunities

### 6.1 Strong Signals to Exploit

1. **Career description NLP** for retrieval, ranking, embeddings, vector, recommendation language
2. **Title taxonomy** — ML Engineer, AI Engineer, Backend Engineer with ML adjacency
3. **YOE band filter** 5–9 with soft edges
4. **India + Pune/Noida/Delhi NCR/Mumbai/Hyderabad** location scoring
5. **Product industry tenure** (Software, Fintech, E-commerce)
6. **Behavioral availability stack** — active, open to work, response rate, notice ≤30–60
7. **`skill_assessment_scores`** as skill trust anchor
8. **`saved_by_recruiters_30d`** as recruiter interest proxy
9. **Anti-trap consistency checks** — cheap honeypot filter

### 6.2 Weak Signals (Use Carefully)

- Raw `skills[]` count
- `summary` text
- `endorsements` without duration
- High-frequency generic skills (HTML, Excel, Agile)
- `profile_completeness_score` alone

### 6.3 Potential Engineered Features

| Feature | Formula Concept |
|---------|-----------------|
| `jd_career_sim` | TF-IDF or precomputed embedding sim(JD, career text) |
| `title_fit` | ML/IR/engineering title classifier score |
| `coherence_penalty` | title vs skills vs career mismatch |
| `product_ratio` | months in product industries / total months |
| `consulting_penalty` | 1.0 if consulting-only else 0 |
| `availability_mod` | sigmoid(response_rate) × recency_decay(last_active) × open_to_work |
| `notice_penalty` | max(0, (notice_days - 30) / 120) |
| `trust_skill_score` | Σ skill_weight × min(assessment, proficiency_map) |
| `honeypot_flag` | timeline + expert-zero rules |

### 6.4 Potential Semantic Features

- Precomputed JD embedding vs career chunk embeddings (offline)
- Skill taxonomy mapping to JD requirement groups (retrieval, vector DB, eval, Python)
- Role entity extraction from descriptions (what they actually did)
- Buzzword density penalty for skills without career support

### 6.5 Potential Behavioral Features

- `engagement_index` = normalize(views + 2×saves + search_appearances)
- `reachability` = response_rate × (1 - stale_penalty)
- `hire_ready` = open_to_work × (notice < 45) × reachability
- `assessment_coverage` = assessed JD-skills / claimed JD-skills

---

## 7. Architectural Implications

Given **5 min / 16GB / CPU / no network**:

| Approach | Feasibility |
|----------|-------------|
| Precomputed embeddings + dot product | ✅ High |
| Inverted index + BM25 on career text | ✅ High |
| XGBoost/LTR on engineered features | ✅ High |
| Multi-stage: filter 100K → 5K → rank 100 | ✅ Recommended |
| Per-candidate LLM reasoning at runtime | ❌ Forbidden / infeasible |
| Real-time API embedding calls | ❌ Forbidden |
| GPU inference | ❌ Forbidden |

**Recommended conceptual pipeline:**
1. Hard filters: honeypot flags, extreme incoherence
2. Soft scoring: JD-career similarity + title fit + YOE/location
3. Behavioral multiplier
4. Trust adjustment (assessments, skill duration)
5. Deterministic tie-break for CSV compliance

---

## 8. Key Assumptions Log

| # | Assumption | Confidence |
|---|------------|------------|
| A1 | Ground truth uses ≥3 relevance levels | High |
| A2 | Honeypots are exactly ~80 with tier 0 | High (documented) |
| A3 | Top relevant candidates are sparse (<1% of pool) | High |
| A4 | Career text more reliable than skills | High (verified in samples) |
| A5 | Sample submission is anti-pattern for ranking | High (verified) |
| A6 | Behavioral signals correlate with GT hireability | Medium–High |
| A7 | International candidates rarely in top 10 | Medium (JD India focus) |
| A8 | `saved_by_recruiters_30d` correlates with positive labels | Medium |
| A9 | Reasoning quality doesn't affect composite but affects finalist selection | High |
| A10 | Pre-computed embeddings allowed if ranking step stays within budget | High |

---

## 9. Design Checklist

- [ ] Read all `.docx` docs (done in this analysis)
- [ ] Parse full `candidates.jsonl` schema
- [ ] Build local spot-check set: known traps + heuristic strong fits
- [ ] Design honeypot detector (timeline, expert-zero, coherence)
- [ ] Define title taxonomy aligned with JD
- [ ] Plan offline pre-computation vs online ranking split
- [ ] Test `validate_submission.py` on output
- [ ] Draft reasoning generator tied to scoring features (anti-hallucination)
- [ ] Prepare sandbox + metadata YAML early (Stage 1 flags missing sandbox)

---

## 10. Bottom Line

The challenge is a **precision-oriented, adversarial ranking problem** disguised as a straightforward "match skills to JD" task. The documentation repeatedly warns that keyword matching fails. The data confirms this: keyword stuffers are abundant, true Senior AI Engineers are rare, summaries are templated, and the sample CSV teaches format while modeling **what not to do**.

**Our ranking strategy:** Rank like a senior recruiter with an IR engineering background — role first, career evidence second, skills third (validated), availability fourth — and never let a Marketing Manager with nine AI skills reach rank 1.

---

## Outputs

| Artifact | Location |
|----------|----------|
| Submission CSV | `project/outputs/submission.csv` |
| Reasoning | `project/outputs/explanations.json` |
| Metadata | `project/submission_metadata.yaml` |

---

*Submitted for the Redrob Intelligent Candidate Discovery & Ranking Challenge.*
