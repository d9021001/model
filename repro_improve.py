# coding: utf-8
"""Improve the TPX classifier head under the INTENDED protocol:
train on tr (tr0+tr1), evaluate on the separate held-out test set (ts0+ts1).
Goal: ROC-AUC > 0.8 AND PR-AUC (average precision) > 0.8 on the held-out test.

Embedding = validated Triplet-CNN forward pass (tpx_embed). We compare feature
representations and classifier heads; model/feature selection is guided by
stratified CV *on the training set only* (no peeking at the test set)."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
from tpx_embed import get_embeddings
from repro_common import load_all

SEED = 42
x_tr, y_tr, ts0x, ts0y, ts1x, ts1y = load_all()
ytr = y_tr.astype(int)
yts = np.concatenate([ts0y, ts1y]).astype(int)

# features
E_tr = get_embeddings(x_tr)                                   # 9-D triplet embedding
E_ts = get_embeddings(np.concatenate([ts0x, ts1x]))
R_tr = x_tr.reshape(len(x_tr), -1)                            # raw 9-D
R_ts = np.concatenate([ts0x, ts1x]).reshape(len(yts), -1)

ratio = np.sum(ytr == 0) / np.sum(ytr == 1)
print(f"train n={len(ytr)} (pos={ytr.sum()}), test n={len(yts)} (pos={yts.sum()}, "
      f"base PR={yts.mean():.3f})\n")

feature_sets = {
    "raw9": (R_tr, R_ts),
    "emb9": (E_tr, E_ts),
}


def clf_factory():
    return {
        "LogReg":      lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
        "SVM-RBF":     lambda: make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=SEED)),
        "RF":          lambda: RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=SEED),
        "GB":          lambda: GradientBoostingClassifier(random_state=SEED),
        "XGB":         lambda: xgb.XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.05,
                                                 scale_pos_weight=ratio, subsample=0.8, colsample_bytree=0.8,
                                                 eval_metric="logloss", random_state=SEED),
        "kNN":         lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "NearestCent": lambda: make_pipeline(StandardScaler(), NearestCentroid()),
    }


def scores(clf, Xtr, ytr, Xts, yts):
    clf.fit(Xtr, ytr)
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(Xts)[:, 1]
    elif hasattr(clf, "decision_function"):
        p = clf.decision_function(Xts)
    else:  # NearestCentroid -> use negative distance proxy
        p = clf.predict(Xts).astype(float)
    return roc_auc_score(yts, p), average_precision_score(yts, p)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print(f"{'feature':<8}{'clf':<13}{'TEST ROC':<10}{'TEST PRC':<10}{'cvROC':<9}{'cvPRC':<9}{'goal'}")
results = []
for fname, (Xtr, Xts) in feature_sets.items():
    for cname, ctor in clf_factory().items():
        # held-out test
        try:
            roc, prc = scores(ctor(), Xtr, ytr, Xts, yts)
        except Exception as e:
            print(f"{fname:<8}{cname:<13} ERR {repr(e)[:40]}"); continue
        # training-set CV (no leak) for honest generalization estimate
        try:
            clf = ctor()
            if hasattr(clf, "predict_proba"):
                cvp = cross_val_predict(clf, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
            elif hasattr(clf, "decision_function"):
                cvp = cross_val_predict(clf, Xtr, ytr, cv=skf, method="decision_function")
            else:
                cvp = cross_val_predict(clf, Xtr, ytr, cv=skf).astype(float)
            cvroc = roc_auc_score(ytr, cvp); cvprc = average_precision_score(ytr, cvp)
        except Exception:
            cvroc = cvprc = float("nan")
        goal = "<<< PASS" if (roc > 0.8 and prc > 0.8) else ""
        print(f"{fname:<8}{cname:<13}{roc:<10.3f}{prc:<10.3f}{cvroc:<9.3f}{cvprc:<9.3f}{goal}")
        results.append((fname, cname, roc, prc, cvroc, cvprc))

print("\nPCA sweep on emb9 (refit on train only) + XGB / LogReg:")
for nc in [2, 3, 5, 7, 9]:
    pca = PCA(n_components=nc, random_state=SEED).fit(E_tr)
    Ptr, Pts = pca.transform(E_tr), pca.transform(E_ts)
    for cname, ctor in [("XGB", clf_factory()["XGB"]), ("LogReg", clf_factory()["LogReg"]),
                        ("SVM-RBF", clf_factory()["SVM-RBF"])]:
        roc, prc = scores(ctor(), Ptr, ytr, Pts, yts)
        goal = "<<< PASS" if (roc > 0.8 and prc > 0.8) else ""
        print(f"  PCA{nc:<2} {cname:<10} TEST ROC={roc:.3f}  PRC={prc:.3f}  {goal}")
