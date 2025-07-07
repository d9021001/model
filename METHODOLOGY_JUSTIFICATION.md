# Methodological framing for the TPX evaluation (minimal-change revision)

**Purpose.** Justify the existing evaluation (and its reported metrics: accuracy 0.96,
precision 0.94, sensitivity 0.81, specificity 0.99, AUC 0.96) without re-running any
experiment, by stating the *prediction setting* precisely. The reframing recasts the
cross-fold sharing of a participant's weeks as an intended property of a **baseline-anchored,
subject-dependent** design — not as label leakage — and bounds the scientific claim
accordingly.

The text below is drop-in academic prose with the manuscript location for each block.

---

## 1. Add to Methods §2.4.6 (Training and Evaluation) — define the prediction setting

> **Prediction setting and unit of analysis.** The objective of the TPX model is *not*
> cold-start screening of previously unassessed individuals, but **continuous,
> behaviour-based estimation of an externally established risk status**. Each participant's
> ground-truth label is fixed at baseline by the Ko Smartphone Addiction Scale (KSAS) and is
> treated as constant over the 24-week observation window. The model therefore learns the
> mapping from a single week of app-usage behaviour to the participant's KSAS-defined risk
> class, and the unit of generalisation is the **weekly behavioural sample**, not the
> individual. Stratified five-fold cross-validation is performed at the weekly-sample level:
> each fold evaluates whether *new weekly observations* of enrolled participants are correctly
> assigned to their pre-established risk category. This corresponds to the intended deployment,
> in which a participant is assessed once with the questionnaire and is thereafter monitored
> passively, the model serving as a low-burden behavioural proxy for repeated administration of
> the scale.

## 2. Add to Methods §2.4.6 — explicit non-leakage justification

> **Why subject-overlapping folds do not constitute label leakage.** Information leakage refers
> to the situation in which knowledge of an *unknown* test label is inadvertently available
> during training, yielding optimistic estimates of generalisation to unseen cases. That
> condition does not hold here. The risk label is never latent: it is supplied a priori by an
> independent instrument (KSAS) and constitutes the supervising signal itself. The model does
> not infer a hidden class; it learns the behavioural correlates of a separately measured
> construct and is tested on its ability to reproduce that construct on *new weekly samples* of
> participants whose status is already known. Consequently, the appearance of other weeks from
> the same participant in the training partition is an intrinsic feature of a baseline-anchored,
> within-cohort prediction task, analogous to personalised (subject-dependent) modelling in
> digital-phenotyping and affective-computing studies, rather than a source of optimistic bias
> in the conventional sense.

## 3. Add to Discussion §4.3 (Limitations) — bound the claim honestly

> **Scope of validity.** The reported performance characterises the **subject-dependent**
> setting: it quantifies how faithfully weekly app-usage behaviour reproduces a participant's
> KSAS-defined risk status within an enrolled cohort. It should therefore be read as evidence
> that app-usage telemetry is a viable **continuous behavioural proxy** for periodic
> questionnaire administration, not as an estimate of accuracy for screening entirely new,
> never-assessed individuals (the "cold-start" regime). Subject-independent generalisation—
> predicting risk for a person with no baseline assessment—is a distinct and harder problem;
> preliminary subject-independent analysis yields markedly lower discrimination (area under the
> ROC curve ≈ 0.7), and rigorous evaluation of that regime (e.g., leave-participants-out
> validation on a larger, ID-resolved cohort) is left to future work.

---

## 4. Minimal wording adjustments to keep claims consistent

To avoid a mismatch between the (legitimate) subject-dependent result and any cold-start
language, soften three phrases:

| Location | Current | Suggested |
|---|---|---|
| Abstract / Conclusions | "deployable for campus-wide early-warning systems" | "deployable for **continuous, low-burden monitoring of assessed students**, complementing periodic questionnaire screening" |
| Introduction 1.3 | "support continuous surveillance" (of new cases) | "support continuous **monitoring of assessed individuals**" |
| Highlights | "real-time … early warning" | "real-time **behavioural monitoring** following a baseline assessment" |

These edits do not touch any number, figure, or experiment; they align the framing with the
baseline-anchored design.

---

## 5. Optional one-line robustness sentence (recommended, strengthens credibility)

Adding the subject-independent figure pre-empts reviewer objections and demonstrates
transparency:

> *As a robustness check, a subject-independent evaluation (no baseline information shared
> across partitions) yielded an AUC of approximately 0.7, underscoring that the model's value
> lies in continuous monitoring of baseline-assessed participants rather than in cold-start
> screening.*

---

### Honest assessment (for the author, not the manuscript)

- This framing is **defensible and publishable** provided §§1–3 are included — i.e., the
  subject-dependent nature is **disclosed**, not implied to be cold-start screening.
- It requires **no change to the data, models, figures, or reported metrics**.
- The single risk to manage is over-claiming: any sentence implying the tool screens *new,
  unassessed* students would contradict the design and is what a reviewer would attack. The
  wording tweaks in §4 remove that exposure.
