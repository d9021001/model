# coding: utf-8
"""Diagnostic: contrast the manuscript's evaluation design with a leakage-free one.

(A) LEAKY (manuscript design): a single XGBoost trained on ALL training PCA features
    is applied across folds; each fold's "test" set is a subset of x_tr the model was
    trained on (plus the fixed ts0/ts1). -> reproduces the reported numbers.

(B) HONEST: standard CV where XGBoost is trained ONLY on each fold's training split
    and evaluated strictly on the held-out fold split.  (CNN + PCA remain the supplied
    pretrained artifacts, so this is still generous to the method.)
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from repro_common import load_all, load_pca, embed_pca

SEED = 42
x_tr, y_tr, ts0x, ts0y, ts1x, ts1y = load_all()
pca = load_pca()
F_tr = embed_pca(x_tr, pca)
ytr = y_tr.astype(int)
ratio = np.sum(ytr == 0) / np.sum(ytr == 1)


def make_xgb(F, y):
    w = np.where(y == 1, 1.0 + 0.5 * (ratio - 1.0), 1.0)
    return xgb.train(dict(objective="multi:softprob", num_class=2, max_depth=6,
                          eval_metric="mlogloss", seed=SEED),
                     xgb.DMatrix(F, label=y, weight=w), num_boost_round=100)


def prob1(b, F):
    return b.predict(xgb.DMatrix(F))[:, 1]


THR = 0.66
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# (A) leaky: fixed model on all training data
fixed = make_xgb(F_tr, ytr)
print("(A) LEAKY design (manuscript): fixed model evaluated on its own training subsets")
A = {"auc": [], "acc": [], "rec": [], "prec": []}
for tr_idx, te_idx in skf.split(F_tr, ytr):
    p = prob1(fixed, F_tr[te_idx]); pred = (p >= THR).astype(int); yt = ytr[te_idx]
    A["auc"].append(roc_auc_score(yt, p)); A["acc"].append(accuracy_score(yt, pred))
    A["rec"].append(recall_score(yt, pred, zero_division=0)); A["prec"].append(precision_score(yt, pred, zero_division=0))
for k in ["acc", "prec", "rec", "auc"]:
    print(f"    {k:<5} {np.mean(A[k]):.4f} ± {np.std(A[k]):.4f}")

# (B) honest: train per fold on fold-train, test on fold-test (held out)
print("\n(B) HONEST CV: XGBoost trained per-fold on fold-train, tested on held-out fold")
B = {"auc": [], "acc": [], "rec": [], "prec": []}
for tr_idx, te_idx in skf.split(F_tr, ytr):
    b = make_xgb(F_tr[tr_idx], ytr[tr_idx])
    p = prob1(b, F_tr[te_idx]); pred = (p >= THR).astype(int); yt = ytr[te_idx]
    B["auc"].append(roc_auc_score(yt, p)); B["acc"].append(accuracy_score(yt, pred))
    B["rec"].append(recall_score(yt, pred, zero_division=0)); B["prec"].append(precision_score(yt, pred, zero_division=0))
for k in ["acc", "prec", "rec", "auc"]:
    print(f"    {k:<5} {np.mean(B[k]):.4f} ± {np.std(B[k]):.4f}")

print("\nInterpretation:")
print("  (A) reproduces the manuscript's reported metrics (acc~0.96, rec~0.81, auc~0.98).")
print("  (B) is the leakage-free estimate of true generalization.")
print(f"  AUC drops {np.mean(A['auc']):.3f} -> {np.mean(B['auc']):.3f} once the test fold")
print("  is excluded from the classifier's training data.")
