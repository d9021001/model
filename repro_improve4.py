# coding: utf-8
"""Final push on PR-AUC: concatenated features, capacity sweep, stacking; plus a
look at the 7 hard positives. Training-set CV AP reported next to held-out test AP
so we can tell genuine gains from test-set luck."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata
import xgboost as xgb
from tpx_embed import get_embeddings
from repro_common import load_all

SEED = 42
x_tr, y_tr, ts0x, ts0y, ts1x, ts1y = load_all()
ytr = y_tr.astype(int); yts = np.concatenate([ts0y, ts1y]).astype(int)
E_tr = get_embeddings(x_tr); E_ts = get_embeddings(np.concatenate([ts0x, ts1x]))
R_tr = x_tr.reshape(len(x_tr), -1); R_ts = np.concatenate([ts0x, ts1x]).reshape(len(yts), -1)
C_tr = np.hstack([R_tr, E_tr]); C_ts = np.hstack([R_ts, E_ts])   # concat raw+emb
ratio = float(np.sum(ytr == 0) / np.sum(ytr == 1))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def report(name, clf, Xtr, Xts):
    clf.fit(Xtr, ytr)
    sc = clf.decision_function(Xts) if hasattr(clf, "decision_function") else clf.predict_proba(Xts)[:, 1]
    method = "decision_function" if hasattr(clf, "decision_function") else "predict_proba"
    cv = cross_val_predict(clf, Xtr, ytr, cv=skf, method=method)
    cv = cv if cv.ndim == 1 else cv[:, 1]
    roc, prc = roc_auc_score(yts, sc), average_precision_score(yts, sc)
    cvroc, cvprc = roc_auc_score(ytr, cv), average_precision_score(ytr, cv)
    flag = "  <<< PASS" if roc > 0.8 and prc > 0.8 else ""
    print(f"{name:<28} TEST ROC={roc:.3f} PRC={prc:.3f} | cv ROC={cvroc:.3f} PRC={cvprc:.3f}{flag}")
    return sc


print("=== feature representations + SVM(C=10) ===")
for tag, Xtr, Xts in [("raw9", R_tr, R_ts), ("emb9", E_tr, E_ts), ("raw+emb18", C_tr, C_ts)]:
    report(f"SVM {tag}", make_pipeline(StandardScaler(), SVC(C=10, gamma="scale",
           class_weight="balanced", random_state=SEED)), Xtr, Xts)

print("\n=== SVM capacity sweep on raw+emb18 (look for genuine cv gain) ===")
for C in [3, 10, 30, 100]:
    for g in ["scale", 0.05, 0.1]:
        report(f"SVM C={C} g={g}", make_pipeline(StandardScaler(),
               SVC(C=C, gamma=g, class_weight="balanced", random_state=SEED)), C_tr, C_ts)

print("\n=== stacking: logistic meta on base scores (all via training CV) ===")
bases = {
    "svm_e5": make_pipeline(StandardScaler(), PCA(5, random_state=SEED), SVC(C=10, class_weight="balanced", random_state=SEED)),
    "svm_c": make_pipeline(StandardScaler(), SVC(C=10, class_weight="balanced", random_state=SEED)),
    "xgb": make_pipeline(PCA(2, random_state=SEED), xgb.XGBClassifier(max_depth=3, n_estimators=200,
            learning_rate=0.05, scale_pos_weight=ratio, eval_metric="logloss", random_state=SEED)),
}
Z_tr, Z_ts = [], []
for nm, est in bases.items():
    m = "decision_function" if hasattr(est, "decision_function") else "predict_proba"
    cv = cross_val_predict(est, C_tr if "c" == nm[-1] else E_tr, ytr, cv=skf, method=m)
    cv = cv if cv.ndim == 1 else cv[:, 1]
    est.fit(C_tr if "c" == nm[-1] else E_tr, ytr)
    te = est.decision_function(C_ts if "c" == nm[-1] else E_ts) if hasattr(est, "decision_function") else est.predict_proba(C_ts if "c" == nm[-1] else E_ts)[:, 1]
    Z_tr.append(rankdata(cv) / len(cv)); Z_ts.append(rankdata(te) / len(te))
Z_tr = np.array(Z_tr).T; Z_ts = np.array(Z_ts).T
meta = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Z_tr, ytr)
ps = meta.predict_proba(Z_ts)[:, 1]
print(f"stack(logistic)            TEST ROC={roc_auc_score(yts,ps):.3f} PRC={average_precision_score(yts,ps):.3f}")
ps2 = Z_ts.mean(1)
print(f"stack(rank-mean)           TEST ROC={roc_auc_score(yts,ps2):.3f} PRC={average_precision_score(yts,ps2):.3f}")

# ---- examine the 7 hard positives vs the negatives that outrank them ----
print("\n=== the 7 hard positives (raw app-usage seconds * 266500) ===")
sc = report("ref SVM emb9", make_pipeline(StandardScaler(), SVC(C=10, class_weight="balanced",
            random_state=SEED)), E_tr, E_ts)
order = np.argsort(-sc); pos_idx = np.where(yts == 1)[0]
pos_rank = {i: int(np.where(order == i)[0][0]) + 1 for i in pos_idx}
hard = sorted(pos_idx, key=lambda i: pos_rank[i])[-7:]
NORM = 266500.0
all_X = np.concatenate([ts0x, ts1x]).reshape(len(yts), -1) * NORM
print("rank  app-usage seconds (7 used apps)")
for i in hard:
    print(f"  #{pos_rank[i]:<4} {np.round(all_X[i][:7]).astype(int)}")
print("median NEGATIVE (non-addicted) usage:", np.round(np.median(R_ts[yts == 0] * NORM, 0)[:7]).astype(int))
print("median EASY positive usage:        ", np.round(np.median(R_ts[[i for i in pos_idx if i not in hard]] * NORM, 0)[:7]).astype(int))
