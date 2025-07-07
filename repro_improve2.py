# coding: utf-8
"""Robustly tune the classifier head on the Triplet-CNN embedding.
Selection by stratified CV on the TRAINING set (scoring=average_precision);
final report on the held-out test set with seed-stability + bootstrap CI."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
from tpx_embed import get_embeddings
from repro_common import load_all

SEED = 42
rng = np.random.RandomState(SEED)
x_tr, y_tr, ts0x, ts0y, ts1x, ts1y = load_all()
ytr = y_tr.astype(int)
yts = np.concatenate([ts0y, ts1y]).astype(int)
E_tr = get_embeddings(x_tr)
E_ts = get_embeddings(np.concatenate([ts0x, ts1x]))
ratio = float(np.sum(ytr == 0) / np.sum(ytr == 1))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def boot_ci(yt, p, n=2000):
    rocs, prcs = [], []
    idx_pos = np.where(yt == 1)[0]; idx_neg = np.where(yt == 0)[0]
    for _ in range(n):
        bp = rng.choice(idx_pos, len(idx_pos), replace=True)
        bn = rng.choice(idx_neg, len(idx_neg), replace=True)
        bi = np.concatenate([bp, bn])
        rocs.append(roc_auc_score(yt[bi], p[bi])); prcs.append(average_precision_score(yt[bi], p[bi]))
    f = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
    return f(rocs), f(prcs)


def evaluate(name, build_proba, Xtr, Xts):
    # training CV (honest selection metric)
    cvp = build_proba(cv=True, Xtr=Xtr, ytr=ytr, Xts=None)
    cvroc, cvprc = roc_auc_score(ytr, cvp), average_precision_score(ytr, cvp)
    # held-out test, averaged over model seeds for stability
    rocs, prcs, last_p = [], [], None
    for s in range(5):
        p = build_proba(cv=False, Xtr=Xtr, ytr=ytr, Xts=Xts, seed=s)
        rocs.append(roc_auc_score(yts, p)); prcs.append(average_precision_score(yts, p)); last_p = p
    (rl, rh), (pl, ph) = boot_ci(yts, last_p)
    flag = "PASS" if (np.mean(rocs) > 0.8 and np.mean(prcs) > 0.8) else ""
    print(f"{name:<26} cv[ROC={cvroc:.3f} PRC={cvprc:.3f}]  "
          f"TEST ROC={np.mean(rocs):.3f}±{np.std(rocs):.3f} PRC={np.mean(prcs):.3f}±{np.std(prcs):.3f}  "
          f"95%CI ROC[{rl:.2f},{rh:.2f}] PRC[{pl:.2f},{ph:.2f}]  {flag}")
    return np.mean(rocs), np.mean(prcs)


# --- candidate 1: PCA-k (refit on train) + XGB ---
def make_pca_xgb(k):
    def f(cv, Xtr, ytr, Xts, seed=0):
        pca = PCA(n_components=k, random_state=SEED)
        clf = xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05,
                                scale_pos_weight=ratio, subsample=0.9, colsample_bytree=0.9,
                                eval_metric="logloss", random_state=seed)
        pipe = make_pipeline(pca, clf)
        if cv:
            return cross_val_predict(pipe, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
        pipe.fit(Xtr, ytr); return pipe.predict_proba(Xts)[:, 1]
    return f


# --- candidate 2: (Std) PCA-k + SVM-RBF tuned ---
def make_pca_svm(k):
    def f(cv, Xtr, ytr, Xts, seed=0):
        pipe = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=SEED),
                             SVC(C=10, gamma="scale", probability=True,
                                 class_weight="balanced", random_state=seed))
        if cv:
            return cross_val_predict(pipe, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
        pipe.fit(Xtr, ytr); return pipe.predict_proba(Xts)[:, 1]
    return f


print("=== PCA-k + XGB (emb9) ===")
for k in [2, 3, 4]:
    evaluate(f"PCA{k}+XGB", make_pca_xgb(k), E_tr, E_ts)
print("\n=== PCA-k + SVM-RBF (emb9) ===")
for k in [2, 3, 5, 9]:
    evaluate(f"PCA{k}+SVM", make_pca_svm(k), E_tr, E_ts)

# --- candidate 3: GridSearched SVM on emb9, selected by AP on training CV ---
print("\n=== GridSearchCV SVM-RBF on emb9 (select by AP on train) ===")
gs = GridSearchCV(make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=SEED)),
                  {"svc__C": [1, 3, 10, 30, 100], "svc__gamma": ["scale", 0.1, 0.3, 1.0]},
                  scoring="average_precision", cv=skf, n_jobs=-1)
gs.fit(E_tr, ytr)
print("best params:", gs.best_params_, "best train-CV AP=%.3f" % gs.best_score_)
p = gs.predict_proba(E_ts)[:, 1]
(rl, rh), (pl, ph) = boot_ci(yts, p)
print(f"  TEST ROC={roc_auc_score(yts,p):.3f} PRC={average_precision_score(yts,p):.3f}  "
      f"95%CI ROC[{rl:.2f},{rh:.2f}] PRC[{pl:.2f},{ph:.2f}]")
