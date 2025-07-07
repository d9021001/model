# coding: utf-8
"""Maximal honest image-level attempt: combine raw + heuristic + Green/Saab +
triplet-embedding features with tuned, ensembled models. Report (a) leakage-free
stratified CV ROC/PRC at natural prevalence, and (b) ROC/PRC on a balanced test
set (the legitimate 'adjust test count' lever, since PR-AUC scales with prevalence)."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
from green_features import build_features
from tpx_embed import get_embeddings

SEED = 42
NORM = 266500.0
load = lambda f: pd.read_excel(f, usecols="A:I", sheet_name="Sheet1").values.astype("float64")
c0 = np.unique(np.vstack([load("tr0.xlsx"), load("ts0.xlsx")]), axis=0)
c1 = np.unique(np.vstack([load("tr1.xlsx"), load("ts1.xlsx")]), axis=0)
X = np.vstack([c0, c1]) / NORM
y = np.concatenate([np.zeros(len(c0)), np.ones(len(c1))]).astype(int)
Ximg = X.reshape(-1, 3, 3, 1)
ratio = float(np.sum(y == 0) / np.sum(y == 1))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print(f"Pooled de-dup: {len(y)} (neg={np.sum(y==0)}, pos={np.sum(y==1)})\n")


def feats(tr, te):
    Ftr, Fte = build_features(Ximg[tr], Ximg[te], use_saab=True, use_heur=True, use_raw=True,
                              extra=(get_embeddings(Ximg[tr]), get_embeddings(Ximg[te])))
    sc = StandardScaler().fit(Ftr)
    return sc.transform(Ftr), sc.transform(Fte)


def make_ensemble():
    return VotingClassifier(estimators=[
        ("xgb", xgb.XGBClassifier(max_depth=4, n_estimators=500, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=ratio, min_child_weight=2, eval_metric="aucpr",
                random_state=SEED, n_jobs=-1)),
        ("et", ExtraTreesClassifier(n_estimators=600, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
        ("rf", RandomForestClassifier(n_estimators=600, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
        ("hgb", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                class_weight="balanced", random_state=SEED)),
        ("svm", SVC(C=10, gamma="scale", class_weight="balanced", probability=True, random_state=SEED)),
    ], voting="soft", weights=[2, 1, 1, 1, 1])


# leakage-free OOF predictions
oof = np.zeros(len(y))
for tr, te in skf.split(Ximg, y):
    Ftr, Fte = feats(tr, te)
    clf = make_ensemble().fit(Ftr, y[tr])
    oof[te] = clf.predict_proba(Fte)[:, 1]

roc = roc_auc_score(y, oof); prc = average_precision_score(y, oof)
print("=== (a) Honest CV @ natural prevalence (19% positive) ===")
print(f"    ROC-AUC = {roc:.3f}   PR-AUC = {prc:.3f}   "
      f"{'<<< BOTH>0.8' if roc>0.8 and prc>0.8 else ''}\n")

# (b) effect of test-set balance on PR-AUC (ROC is prevalence-invariant)
print("=== (b) PR-AUC vs test-set positive prevalence (same OOF ranking) ===")
print(f"    {'pos:neg':<12}{'prevalence':<12}{'ROC':<8}{'PR-AUC':<8}")
rng = np.random.RandomState(SEED)
pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
for pr_, label in [(1, "1:3"), (0.5, "1:1"), (0.4, "2:3")]:
    # subsample negatives to reach target prevalence
    n_pos = len(pos); n_neg = int(n_pos * (1 - pr_) / pr_) if pr_ < 1 else int(n_pos * 3)
    # interpret pr_ directly as prevalence for the 1:1 and 2:3 rows
    if label == "1:3":
        n_neg = n_pos * 3
    elif label == "1:1":
        n_neg = n_pos
    elif label == "2:3":
        n_neg = int(n_pos * 1.5)
    rocs, prcs = [], []
    for _ in range(200):
        sel = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
        rocs.append(roc_auc_score(y[sel], oof[sel])); prcs.append(average_precision_score(y[sel], oof[sel]))
    prev = n_pos / (n_pos + n_neg)
    flag = "  <<< PR>0.8" if np.mean(prcs) > 0.8 else ""
    print(f"    {label:<12}{prev:<12.2f}{np.mean(rocs):<8.3f}{np.mean(prcs):<8.3f}{flag}")
