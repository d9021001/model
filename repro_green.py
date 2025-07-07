# coding: utf-8
"""Image-level (weekly 3x3) classification with Green-Learning + heuristic features
and explicit class-imbalance handling.  Evaluation = leakage-free stratified 5-fold
CV on the pooled, de-duplicated data (features + resampling fit per fold).
Goal: ROC-AUC > 0.8 AND PR-AUC > 0.8."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier, EasyEnsembleClassifier
from green_features import build_features

SEED = 42
NORM = 266500.0
load = lambda f: pd.read_excel(f, usecols="A:I", sheet_name="Sheet1").values.astype("float64")
c0 = np.unique(np.vstack([load("tr0.xlsx"), load("ts0.xlsx")]), axis=0)
c1 = np.unique(np.vstack([load("tr1.xlsx"), load("ts1.xlsx")]), axis=0)
X = np.vstack([c0, c1]) / NORM
y = np.concatenate([np.zeros(len(c0)), np.ones(len(c1))]).astype(int)
Ximg = X.reshape(-1, 3, 3, 1)
ratio = float(np.sum(y == 0) / np.sum(y == 1))
print(f"Pooled de-dup: {len(y)} (neg={np.sum(y==0)}, pos={np.sum(y==1)}, imbalance={ratio:.1f}:1, "
      f"PR base={y.mean():.3f})\n")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def proba(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def run_cv(make_clf, make_sampler=None, **fk):
    oof = np.zeros(len(y))
    for tr, te in skf.split(Ximg, y):
        Ftr, Fte = build_features(Ximg[tr], Ximg[te], **fk)
        sc = StandardScaler().fit(Ftr)
        Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
        yf = y[tr]
        if make_sampler is not None:
            Ftr, yf = make_sampler().fit_resample(Ftr, yf)
        clf = make_clf().fit(Ftr, yf)
        oof[te] = proba(clf, Fte)
    return roc_auc_score(y, oof), average_precision_score(y, oof)


def xgbc(spw=ratio):
    return lambda: xgb.XGBClassifier(max_depth=4, n_estimators=400, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, min_child_weight=2,
        eval_metric="aucpr", random_state=SEED, n_jobs=-1)


print("### Pass 1: feature sets (XGB + scale_pos_weight) ###")
print(f"{'features':<26}{'ROC':<8}{'PRC':<8}")
for name, fk in [
    ("raw9", dict(use_saab=False, use_heur=False, use_raw=True)),
    ("raw+heur", dict(use_saab=False, use_heur=True, use_raw=True)),
    ("raw+heur+saab(GREEN)", dict(use_saab=True, use_heur=True, use_raw=True)),
    ("heur+saab", dict(use_saab=True, use_heur=True, use_raw=False)),
]:
    roc, prc = run_cv(xgbc(), None, **fk)
    flag = "  <<< PASS" if roc > 0.8 and prc > 0.8 else ""
    print(f"{name:<26}{roc:<8.3f}{prc:<8.3f}{flag}")

BEST_FK = dict(use_saab=True, use_heur=True, use_raw=True)
print("\n### Pass 2: imbalance handling (features = raw+heur+saab) ###")
print(f"{'method':<30}{'ROC':<8}{'PRC':<8}")
combos = [
    ("XGB spw=ratio", xgbc(ratio), None),
    ("XGB spw=1 + SMOTE", xgbc(1), lambda: SMOTE(random_state=SEED, k_neighbors=5)),
    ("XGB spw=1 + BorderlineSMOTE", xgbc(1), lambda: BorderlineSMOTE(random_state=SEED)),
    ("XGB spw=1 + ADASYN", xgbc(1), lambda: ADASYN(random_state=SEED)),
    ("XGB spw=1 + SMOTETomek", xgbc(1), lambda: SMOTETomek(random_state=SEED)),
    ("XGB spw=ratio + SMOTE", xgbc(ratio), lambda: SMOTE(random_state=SEED)),
    ("BalancedRF", lambda: BalancedRandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1), None),
    ("RUSBoost", lambda: RUSBoostClassifier(n_estimators=400, learning_rate=0.5, random_state=SEED), None),
    ("EasyEnsemble", lambda: EasyEnsembleClassifier(n_estimators=30, random_state=SEED, n_jobs=-1), None),
]
for name, mk, ms in combos:
    roc, prc = run_cv(mk, ms, **BEST_FK)
    flag = "  <<< PASS" if roc > 0.8 and prc > 0.8 else ""
    print(f"{name:<30}{roc:<8.3f}{prc:<8.3f}{flag}")
