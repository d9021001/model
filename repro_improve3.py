# coding: utf-8
"""Push PR-AUC up: diagnose positive ranking, then try rank-ensembling and an
end-to-end supervised net (imbalance-aware). Train on tr, evaluate on held-out ts."""
import os, sys, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
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
ratio = float(np.sum(ytr == 0) / np.sum(ytr == 1))


def rep(name, p):
    roc, prc = roc_auc_score(yts, p), average_precision_score(yts, p)
    flag = "  <<< PASS" if roc > 0.8 and prc > 0.8 else ""
    print(f"{name:<30} ROC={roc:.3f}  PRC={prc:.3f}{flag}")
    return roc, prc, p


# ---- base rankers (decision scores) ----
def svm_score(k):
    pipe = make_pipeline(StandardScaler(), PCA(k, random_state=SEED),
                         SVC(C=10, gamma="scale", class_weight="balanced", random_state=SEED))
    pipe.fit(E_tr, ytr); return pipe.decision_function(E_ts)

def xgb_score(k):
    pipe = make_pipeline(PCA(k, random_state=SEED),
                         xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05,
                                           scale_pos_weight=ratio, subsample=0.9, colsample_bytree=0.9,
                                           eval_metric="logloss", random_state=SEED))
    pipe.fit(E_tr, ytr); return pipe.predict_proba(E_ts)[:, 1]

print("=== base rankers ===")
s3 = rep("PCA3+SVM", svm_score(3))[2]
s5 = rep("PCA5+SVM", svm_score(5))[2]
s2x = rep("PCA2+XGB", xgb_score(2))[2]
semb = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced", random_state=SEED))
semb.fit(E_tr, ytr); sE = rep("emb9+SVM", semb.decision_function(E_ts))[2]

# ---- diagnostic: how do the 13 positives rank under PCA5+SVM? ----
order = np.argsort(-s5)
ranks_of_pos = np.where(yts[order] == 1)[0] + 1   # 1 = top
print(f"\nRanks of the 13 test positives (1=highest score, out of {len(yts)}):")
print("  ", sorted(ranks_of_pos.tolist()))
print(f"  positives in top-13: {np.sum(ranks_of_pos <= 13)}/13;  in top-26: {np.sum(ranks_of_pos <= 26)}/13")

# ---- rank-average ensemble ----
print("\n=== rank-average ensemble ===")
def rnk(p): return rankdata(p) / len(p)
ens = (rnk(s3) + rnk(s5) + rnk(s2x) + rnk(sE)) / 4
rep("ens(SVM3,SVM5,XGB2,SVMemb)", ens)
ens2 = (rnk(s5) + rnk(s2x)) / 2
rep("ens(SVM5,XGB2)", ens2)

# ---- end-to-end supervised net (imbalance-aware), train on tr / test on ts ----
print("\n=== end-to-end supervised MLP (class-weighted) ===")
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
tf.random.set_seed(SEED)

def make_mlp(indim):
    m = models.Sequential([
        layers.Input((indim,)), layers.Normalization(),
        layers.Dense(64, activation="relu"), layers.Dropout(0.3),
        layers.Dense(32, activation="relu"), layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m

for tag, Xtr, Xts in [("raw9", R_tr, R_ts), ("emb9", E_tr, E_ts)]:
    norm = StandardScaler().fit(Xtr)
    Xtr_s, Xts_s = norm.transform(Xtr), norm.transform(Xts)
    cw = {0: 1.0, 1: ratio}
    preds = []
    for s in range(5):
        tf.random.set_seed(s)
        m = make_mlp(Xtr.shape[1])
        m.fit(Xtr_s, ytr, epochs=60, batch_size=64, class_weight=cw, verbose=0)
        preds.append(m.predict(Xts_s, verbose=0).ravel())
    p = np.mean(preds, axis=0)
    rep(f"MLP({tag}) 5-seed avg", p)
