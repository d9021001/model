# coding: utf-8
"""Regenerate Figure 3 (single-panel, publication style) from the reproducible
pipeline: Proposed (Triplet+PCA+XGB, CV-pooled subject-dependent) vs raw-feature
baselines (LR/RF/GB/XGB on the held-out test). Prints exact AUCs for text sync."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from repro_common import load_all, load_pca, embed_pca

SEED = 42
x_tr, y_tr, ts0x, ts0y, ts1x, ts1y = load_all()
pca = load_pca()
ytr = y_tr.astype(int)
ratio = np.sum(ytr == 0) / np.sum(ytr == 1)
F_tr = embed_pca(x_tr, pca)
F_ts0 = embed_pca(ts0x, pca); F_ts1 = embed_pca(ts1x, pca)

# Proposed: fixed model, CV-pooled (subject-dependent) ROC
w = np.where(ytr == 1, 1.0 + 0.5 * (ratio - 1.0), 1.0)
fixed = xgb.train(dict(objective="multi:softprob", num_class=2, max_depth=6, seed=SEED),
                  xgb.DMatrix(F_tr, label=ytr, weight=w), num_boost_round=107)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pp, py = [], []
for tr_idx, te_idx in skf.split(F_tr, ytr):
    F_val = np.concatenate([F_tr[te_idx], F_ts0, F_ts1])
    y_val = np.concatenate([ytr[te_idx], ts0y, ts1y]).astype(int)
    pp.append(fixed.predict(xgb.DMatrix(F_val))[:, 1]); py.append(y_val)
pp = np.concatenate(pp); py = np.concatenate(py)

# Baselines on raw 9-dim, held-out test
Xr_tr = x_tr.reshape(len(x_tr), -1)
Xr_ts = np.concatenate([ts0x, ts1x]).reshape(len(ts0x) + len(ts1x), -1)
y_ts = np.concatenate([ts0y, ts1y]).astype(int)
base = {}
base["LR"] = LogisticRegression(max_iter=2000).fit(Xr_tr, ytr).predict_proba(Xr_ts)[:, 1]
base["RF"] = RandomForestClassifier(random_state=SEED).fit(Xr_tr, ytr).predict_proba(Xr_ts)[:, 1]
base["GB"] = GradientBoostingClassifier(random_state=SEED).fit(Xr_tr, ytr).predict_proba(Xr_ts)[:, 1]
bx = xgb.train(dict(objective="binary:logistic", max_depth=6, seed=SEED),
               xgb.DMatrix(Xr_tr, label=ytr), num_boost_round=100)
base["XGB"] = bx.predict(xgb.DMatrix(Xr_ts))

# ---- draw ----
plt.figure(figsize=(7, 7))
auc_p = roc_auc_score(py, pp)
fpr, tpr, _ = roc_curve(py, pp)
plt.plot(fpr, tpr, color="black", lw=3, label=f"Proposed (Triplet+PCA+XGB) AUC={auc_p:.3f}")
colors = {"LR": "tab:blue", "RF": "tab:orange", "GB": "tab:green", "XGB": "tab:red"}
aucs = {"Proposed": auc_p}
for name in ["LR", "RF", "GB", "XGB"]:
    a = roc_auc_score(y_ts, base[name]); aucs[name] = a
    fpr, tpr, _ = roc_curve(y_ts, base[name])
    plt.plot(fpr, tpr, color=colors[name], lw=1.6, drawstyle="steps-post",
             label=f"{name} (raw) AUC={a:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
plt.xlim(-0.02, 1.02); plt.ylim(-0.02, 1.02)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Comparison: Proposed vs Multiple Raw Baselines")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("Figure3_regenerated.png", dpi=150)
print("Saved Figure3_regenerated.png")
print("EXACT AUCs:")
for k in ["Proposed", "LR", "RF", "GB", "XGB"]:
    print(f"  {k:<10} {aucs[k]:.3f}")
