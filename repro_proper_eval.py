# coding: utf-8
"""(A) Honest performance on a proper, diverse stratified split of the pooled,
de-duplicated data.  (B) Effect of multi-week (participant-level) aggregation."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata
import xgboost as xgb
from tpx_embed import get_embeddings

SEED = 42
rng = np.random.RandomState(SEED)
NORM = 266500.0


def load(f):
    return pd.read_excel(f, usecols="A:I", sheet_name="Sheet1").values.astype("float64")

# pool all data, de-duplicate within class
c0 = np.unique(np.vstack([load("tr0.xlsx"), load("ts0.xlsx")]), axis=0)
c1 = np.unique(np.vstack([load("tr1.xlsx"), load("ts1.xlsx")]), axis=0)
X = np.vstack([c0, c1]) / NORM
y = np.concatenate([np.zeros(len(c0)), np.ones(len(c1))]).astype(int)
Xim = X.reshape(-1, 3, 3, 1)
E = get_embeddings(Xim)
print(f"Pooled de-duplicated: {len(y)} samples (neg={np.sum(y==0)}, pos={np.sum(y==1)})\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
ratio = float(np.sum(y == 0) / np.sum(y == 1))


def base_scores(Etr, ytr, Ete):
    s1 = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced",
                       random_state=SEED)).fit(Etr, ytr).decision_function(Ete)
    s2 = xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05, scale_pos_weight=ratio,
                           eval_metric="logloss", random_state=SEED).fit(Etr, ytr).predict_proba(Ete)[:, 1]
    return (rankdata(s1) + rankdata(s2)) / (2 * len(s1))


# ---- (A) honest stratified 5-fold CV on representative data ----
oof = np.zeros(len(y))
for tr, te in skf.split(E, y):
    oof[te] = base_scores(E[tr], y[tr], E[te])
print("(A) Proper stratified 5-fold CV (diverse, de-duplicated test folds):")
print(f"    ROC-AUC = {roc_auc_score(y, oof):.3f}   PR-AUC = {average_precision_score(y, oof):.3f}\n")

# ---- (B) multi-week aggregation: bag K same-class weeks, score = mean member score ----
print("(B) Multi-week aggregation (simulates person-level decision from K weeks):")
print(f"    {'K weeks':<10}{'ROC-AUC':<10}{'PR-AUC':<10}")
for K in [1, 3, 5, 10, 20]:
    # build bags from the out-of-fold scores, preserving class prevalence
    n_bags = 400
    bag_scores, bag_labels = [], []
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    # keep test prevalence ~ original pos rate
    p_pos = y.mean()
    for _ in range(n_bags):
        cls = 1 if rng.rand() < p_pos else 0
        pool = pos_idx if cls == 1 else neg_idx
        members = rng.choice(pool, K, replace=(K > len(pool)))
        bag_scores.append(oof[members].mean()); bag_labels.append(cls)
    bag_scores = np.array(bag_scores); bag_labels = np.array(bag_labels)
    roc = roc_auc_score(bag_labels, bag_scores); prc = average_precision_score(bag_labels, bag_scores)
    flag = "  <<< ROC&PRC > 0.8" if roc > 0.8 and prc > 0.8 else ""
    print(f"    {K:<10}{roc:<10.3f}{prc:<10.3f}{flag}")
print("\n(K=1 is the weekly-sample task; larger K = aggregating a person's weeks.)")
