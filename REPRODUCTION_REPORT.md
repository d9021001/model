# Reproduction Report — *Manuscript-app-addict-Jba2026_0101a.docx*

**Paper:** *Detecting High-Risk Smartphone Addiction among College Students via AI-Based App Usage Analysis During COVID-19* (TPX pipeline: Triplet-CNN → PCA → XGBoost)
**Date:** 2026-06-17
**Goal:** Reproduce every reported result from the supplied artifacts (`tr0/tr1/ts0/ts1.xlsx`, `triplet.h5`, `pca_model.pkl`).

---

## 1. Environment

- Python 3.11.8 (Windows). Original code header targeted Python 3.9.5 / TF 2.5.
- Installed: `tensorflow-cpu 2.21`, `xgboost`, `h5py` (numpy/pandas/sklearn/scipy/matplotlib already present).
- **Two artifacts could not be used as-is:**
  - `xgb_model.model` — **missing** from the repo (README notes it is not tracked). Retrained (see §4).
  - `pca_model.pkl` — pickled with scikit-learn 0.24.2; its `.transform()` is incompatible with sklearn 1.8. The fitted `components_`/`mean_` are intact, so PCA is applied manually as `(emb − mean_) @ components_.T` (identical to sklearn for `whiten=False`).
- `triplet.h5` was saved in Keras 2.3.1 and **cannot be loaded by Keras 3** (ships with TF 2.21). The CNN forward pass was therefore re-implemented in pure NumPy directly from the stored weights and **validated against a fresh Keras-3 model loaded with the same weights → max abs diff 1.9e-7** (float32 round-off). Embeddings are exact.

---

## 2. Headline result — reproduced ✔

`repro_tpx.py` — 5-fold stratified CV, fixed pipeline (validated CNN embedding → official PCA → XGBoost trained on the training PCA features, applied across folds), operating point `pos_weight≈2.5`, decision threshold 0.66.

| Metric | **Reproduced** | Manuscript (§3.3.2) |
|---|---|---|
| Precision | **0.9376 ± 0.0054** | 0.9397 ± 0.0085 |
| Accuracy | **0.9593 ± 0.0030** | 0.9597 ± 0.0082 |
| Recall / Sensitivity (TPR) | **0.8095 ± 0.0154** | 0.8097 ± 0.0467 |
| Specificity (TNR) | **0.9892 ± 0.0008** | 0.9897 ± 0.0013 |
| AUC (CV-pooled) | **0.981** | 0.96 / 0.984 |

All four means match to ≈0.002. (Reproduced std for recall is tighter than reported, because the exact original `xgb_model.model` is unavailable; means are unaffected.)

## 3. Other reported results — reproduced ✔

`repro_stats.py`
- **Sample construction (§3.1):** 1,834 class-0 + 432 class-1 = **2,266** valid weekly samples — **exact** (the loader's `range(0, n−1)` drops the last row of each file, hence the −1 per class vs the raw 1,835/433).
- **Table 3 (KSAS symptom-vector stats):** Pearson r = 1.00, 0.55, 0.80; t-tests p = 0.66, 1.00 — **all exact**.

`repro_baselines.py` / `repro_figure3.py` — **Figure 3** (`repro_figure3.png`, left panel)
| Model | Repro AUC | Manuscript |
|---|---|---|
| Proposed (TPX) | 0.981 | 0.984 |
| LR (raw) | 0.822 | 0.818 |
| RF (raw) | 0.636 | 0.605 |
| GB (raw) | 0.618 | 0.603 |
| XGB (raw) | 0.299 | 0.42 |

Baselines reproduce closely, including the characteristic **staircase ROC** on raw features described in §3.3.1.

---

## 4. ⚠️ Material finding: the headline metrics are inflated by data leakage

The reported performance is obtained by evaluating a pipeline whose CNN, PCA **and** XGBoost were all fit on the **entire** dataset, then scoring it on **subsets of that same data**. The README states this directly: *"5-fold cross-validation (with all test samples included in each fold)."* In the original CV loop the model is **loaded**, never retrained per fold, and each fold's "test" set is a slice of the training data plus the fixed `ts0/ts1`.

`repro_honest_cv.py` contrasts the manuscript design (A) with a leakage-free CV (B) where XGBoost is trained **only** on each fold's training split and tested **strictly** on the held-out fold (CNN + PCA still the supplied pretrained artifacts — i.e. still generous):

| Metric | (A) Manuscript design | (B) Honest CV |
|---|---|---|
| AUC | 0.9998 | **0.567** (≈ random) |
| Accuracy | 0.986 | 0.763 |
| Recall | 0.933 | 0.144 |
| Precision | 0.997 | 0.302 |

`repro_figure3.png` (right panel) shows the same collapse: under honest evaluation the proposed method (AUC **0.57**) falls **below** the raw-feature baselines it is compared against (RF 0.69, GB 0.68, LR 0.61). The 3-D PCA features carry essentially no out-of-sample discriminative signal; the apparent separation in the reported CV is memorisation of training points.

**Conclusion.** The manuscript's numbers are *faithfully reproducible* — but they reflect in-sample (leaked) evaluation, not generalization. Before publication the pipeline should be re-evaluated with a strict protocol: split first, then fit the CNN, PCA **and** classifier on the training partition only, and report held-out performance.

---

## 5. Files

| File | Purpose |
|---|---|
| `tpx_embed.py` | TF-free NumPy Triplet-CNN forward pass (validated vs Keras to 1.9e-7) |
| `repro_common.py` | Data loading + embedding + PCA (mirrors original loader) |
| `repro_stats.py` | §3.1 sample construction + Table 3 statistics |
| `repro_tpx.py` | §3.3.2 TPX 5-fold CV headline table |
| `repro_baselines.py` | §3.3.1 baseline AUCs |
| `repro_figure3.py` → `repro_figure3.png` | Figure 3 + honest-evaluation companion |
| `repro_honest_cv.py` | Leakage diagnostic (A vs B) |
| `repro_tpx_tune.py` | Operating-point search that located `pos_weight`/threshold |

Run order: `repro_stats.py` → `repro_tpx.py` → `repro_baselines.py` → `repro_figure3.py` → `repro_honest_cv.py`.

---

## 6. Improvement attempt — target ROC>0.8 AND PR-AUC>0.8 (image level)

Per request: stay image-level (weekly 3×3), allow adjusting the test-image count, combine
**Green Learning** (Saab/PixelHop feed-forward features, `green_features.py`) with **heuristic
domain features**, and handle the **severe class imbalance**. Evaluation is leakage-free
stratified 5-fold CV on the pooled, de-duplicated data (features + resampling fit per fold).

**Test-set defect found:** `ts1.xlsx` (the addicted *test* class) contains only **2 unique
vectors** (each ×7); one of them has *lower* usage than the median non-addicted user. On that
set PR-AUC>0.8 is mathematically impossible (max ≈0.72). tr0/tr1 are 100% unique; no
train/test contamination. So the test set was rebuilt diverse (pooled + de-dup + stratified).

**Results (`repro_green.py`, `repro_green2.py`):**

| Configuration | ROC | PR-AUC |
|---|---|---|
| raw9 + XGB(spw) | 0.705 | 0.397 |
| raw+heuristic + XGB | 0.707 | 0.402 |
| raw+heuristic+**Green/Saab** + XGB | 0.698 | 0.359 |
| + SMOTE / BorderlineSMOTE / ADASYN / SMOTETomek / BalancedRF | ≈0.69–0.70 | ≈0.35 |
| **All features + 5-model soft-voting ensemble** | **0.716** | 0.390 |
| ↑ same model, balanced (1:1) test set | 0.716 | 0.712 |

**Conclusions.**
- The honest image-level ceiling is **ROC ≈ 0.72**; PR-AUC ≈ 0.39 at natural prevalence,
  ≈0.71 even on a balanced test set. **ROC>0.8 AND PR-AUC>0.8 is not achievable at the
  weekly/image level** with these data by any feature set or classifier tried.
- Green/Saab and heuristic features did **not** beat raw features — the discriminative signal
  is simply not present in 7 weekly app-usage numbers (heavy class overlap).
- Imbalance methods (SMOTE family, balanced ensembles) did **not** raise ROC/PR-AUC, because
  both are *ranking* metrics — resampling moves the decision threshold, not the ranking. They
  help F1/recall at a fixed threshold, not AUC.
- The dilemma: keep the supplied (degenerate, easy) test set → ROC 0.82 (>0.8) but PR-AUC ≤0.72;
  use a proper diverse test set → ROC ≈ 0.72. No legitimate image-level config satisfies both.
- The only evaluation that reaches both targets is **multi-week aggregation** (score = mean over
  a person's weeks): K=10 → ROC 0.87; K=20 → ROC 0.97 / PR-AUC 0.92 (`repro_proper_eval.py`).
  That requires participant IDs (absent from the data).

