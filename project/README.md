# LLMarg — Utterance Choice with LLMs

This project tests whether LLMs replicate human behaviour in two complementary
experiments on argumentative language use. Both experiments use stimuli from the
human study reported in:

> *"What guides utterance choice in argumentative language use?"*
> CogSci 2026, Franke, Carcassi et al. — full paper in `projectWM/paper_CogSci-2026/`

The listener-side experiment is documented in
`listener_side_experiment_writeup/listener_side_experiment_writeup.tex`.

---

## The sentence template

Both experiments share the same utterance space. Sentences follow the template:

```
[OQ] of the students got [IQ] of the answers [ADJ]
```

- **OQ** (outer quantifier) ∈ {none, some, most, all}
- **IQ** (inner quantifier) ∈ {none, some, most, all}
- **ADJ** ∈ {right, wrong}

This yields **32 possible sentences**. The exam results are shown as a table
of students and how many of *N* questions each answered correctly.

---

## Experiment 1 — Speaker side (utterance production)

**Task:** Given an exam-results table, choose a sentence that truthfully
describes the results while framing them strategically:

- **High framing** (Green Valley) — make it sound like students had a *high* success rate
- **Low framing** (Riverside) — make it sound like students had a *low* success rate

**Stimuli:** `speaker_stimuli.json` — 4 matrix-size conditions × 20 exam-result
arrays = 80 unique states. Each state is tried under both framings.

| Condition | Students | Questions |
|-----------|----------|-----------|
| `wideShort` | 5 | 12 |
| `wideLong`  | 11 | 12 |
| `narrowShort` | 5 | 6 |
| `narrowLong`  | 11 | 6 |

**Human baseline:** N = 186 participants (between-subjects on condition,
within-subjects on high/low framing, 20 trials each).

---

## Experiment 2 — Listener side (speaker-type attribution)

**Task:** Given an exam-results table *and* a sentence, decide which type of
speaker most likely produced that sentence:

| Role label | Speaker goal |
|------------|--------------|
| `high` — Teacher | make the exam sound *easy* (high success rate) |
| `low` — Student | make the exam sound *hard* (low success rate) |
| `info` — Examiner | describe the results *objectively*, no framing |

This is a forced 3-way choice.

**Stimuli:** `listener_stimuli.csv` — 30 items (10 per speaker type), each
specifying the utterance (Q1, Q2, ADJ), the exam-results state (observation),
the condition (ground-truth speaker type), and the experimental list (1 or 2).
Each participant saw one list (15 items).

Items were selected model-driven: utterance–state pairs that are maximally
diagnostic of a single speaker type according to the posterior of the RSA
speaker models from Experiment 1.

**Human baseline:** collected; results in writeup.

---

## Repository layout

```
projectWM/
├── README.md                              ← this file
│
├── stimuli/
│   ├── speaker_stimuli.json               ← Exp 1 stimuli (all conditions & arrays)
│   └── listener_stimuli.csv               ← Exp 2 stimuli (30 items, 3 conditions)
│
├── results/                               ← TODO: raw LLM response CSVs go here
├── analysis/                              ← TODO: analysis scripts/notebooks go here
│
├── App.vue                                ← original Vue/magpie frontend for Exp 1
├── paper_CogSci-2026/                     ← full paper (LaTeX + PDF)
└── listener_side_experiment_writeup/      ← Exp 2 design & results (LaTeX + PDF)
```

---

## Pipeline overview

**Target models:** OpenAI (GPT-4o), Claude (Anthropic), DeepSeek, Gemini (Google)

```
speaker_stimuli.json ──► [Job A] LLM speaker experiment ──► results/speaker_<model>.csv ──┐
                                                                                            ├──► [Job C] Analysis
listener_stimuli.csv ──► [Job B] LLM listener experiment ──► results/listener_<model>.csv ─┘
                                                                    × 4 models
```

---

## TODOs / Jobs

### Job A — Run the LLM speaker experiment

**Goal:** Replicate Experiment 1 with LLMs instead of humans.

- [ ] Implement the LLM API call in `run_llm_experiment.py` (`call_llm()` stub)
- [ ] Design and pilot the prompt (the key choices: include/exclude the Green Valley
      backstory; show the table as plain text or as ✓/✗ symbols; try chain-of-thought)
- [ ] Run all 4 matrix-size conditions for ≥ 20 runs per stimulus × framing cell;
      save one CSV per model per condition to `results/`
- [ ] Cover all four target models: **OpenAI** (GPT-4o), **Claude** (Anthropic), **DeepSeek**, **Gemini** (Google)

**Inputs:** `speaker_stimuli.json`  
**Outputs:** `results/speaker_<model>_<condition>.csv`
  — columns: condition, stimulus_idx, students_array, n_questions, framing,
    utterance, oq, iq, adj, run, raw_response

---

### Job B — Run the LLM listener experiment

**Goal:** Replicate Experiment 2 — can LLMs infer which type of speaker
produced a given utterance?

- [ ] Write a prompt that presents: the exam-results table + the utterance +
      a description of the three speaker roles, and asks the LLM to pick one
- [ ] Run all 30 items for ≥ 20 runs each; save to `results/`
- [ ] Cover all four target models: **OpenAI**, **Claude**, **DeepSeek**, **Gemini**

**Inputs:** `listener_stimuli.csv`  
**Outputs:** `results/listener_<model>.csv`
  — columns: itemID, list, condition, utterance, observation, chosen_speaker,
    run, raw_response

---

### Job C — Analysis & comparison

**Goal:** Compare LLM choice distributions to human data and RSA model predictions.

- [ ] For Exp 1: plot choice proportions by framing and matrix condition (match
      Figure in the paper); compute correlation with human proportions
- [ ] For Exp 2: compute proportion of "match" responses per condition; compare
      to human match rates from the writeup; test whether LLMs, like humans,
      perform better on `high`/`low` than `info`
- [ ] Compare performance across models (Job A/B outputs)
- [ ] Check whether LLM choices align with any of the RSA model variants
      (vanilla, lr-argstrength, maximin, model-free) — predictions may be
      available from the paper authors

---

### Job D — Open questions to resolve (discuss with supervisor)

- [ ] Should the LLM be given the framing backstory (Green Valley / Riverside)
      or a more neutral instruction? — affects comparability to human experiment
- [ ] Temperature setting: use temperature = 0 for deterministic output, or
      temperature > 0 to sample a distribution over choices?
- [ ] For the listener experiment: should the model see *both* lists or be
      restricted to one (as humans were)?
- [ ] How to handle parse errors (model does not return a valid choice)?
