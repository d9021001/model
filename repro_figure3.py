# coding: utf-8
"""Reproduce Figure 3 (ROC comparison: proposed TPX vs raw-feature baselines) and
add an honest leakage-free companion panel."""
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

# ---------- LEFT PANEL: manuscript-style Figure 3 ----------
# Proposed curve = CV-pooled predictions of the fixed pipeline (manuscript design).
w = np.where(ytr == 1, 1.0 + 0.5 * (ratio - 1.0), 1.0)
fixed = xgb.train(dict(objective="multi:softprob", num_class=2, max_depth=6, seed=SEED),
                  xgb.DMatrix(F_tr, label=ytr, weight=w), num_boost_round=107)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pooled_p, pooled_y = [], []
for tr_idx, te_idx in skf.split(F_tr, ytr):
    F_val = np.concatenate([F_tr[te_idx], F_ts0, F_ts1])
    y_val = np.concatenate([ytr[te_idx], ts0y, ts1y]).astype(int)
    pooled_p.append(fixed.predict(xgb.DMatrix(F_val))[:, 1]); pooled_y.append(y_val)
pooled_p = np.concatenate(pooled_p); pooled_y = np.concatenate(pooled_y)

# Baselines on raw 9-dim vectors, evaluated on the fixed held-out test set ts0+ts1.
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

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax = axes[0]
fpr, tpr, _ = roc_curve(pooled_y, pooled_p)
ax.plot(fpr, tpr, lw=2.5, color="crimson",
        label=f"Proposed (TPX)  AUC={roc_auc_score(pooled_y, pooled_p):.3f}")
colors = {"LR": "tab:blue", "RF": "tab:green", "GB": "tab:orange", "XGB": "tab:purple"}
for name in ["LR", "RF", "GB", "XGB"]:
    fpr, tpr, _ = roc_curve(y_ts, base[name])
    ax.plot(fpr, tpr, lw=1.6, color=colors[name], drawstyle="steps-post",
            label=f"{name} (raw)  AUC={roc_auc_score(y_ts, base[name]):.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Fig. 3 reproduction: Proposed vs raw baselines\n(manuscript evaluation design)")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)

# ---------- RIGHT PANEL: honest leakage-free evaluation ----------
ax = axes[1]
# Honest CV for proposed: train XGB per fold on fold-train PCA, test on held-out fold.
hp, hy = [], []
for tr_idx, te_idx in skf.split(F_tr, ytr):
    wf = np.where(ytr[tr_idx] == 1, 1.0 + 0.5 * (ratio - 1.0), 1.0)
    b = xgb.train(dict(objective="multi:softprob", num_class=2, max_depth=6, seed=SEED),
                  xgb.DMatrix(F_tr[tr_idx], label=ytr[tr_idx], weight=wf), num_boost_round=107)
    hp.append(b.predict(xgb.DMatrix(F_tr[te_idx]))[:, 1]); hy.append(ytr[te_idx])
hp = np.concatenate(hp); hy = np.concatenate(hy)
fpr, tpr, _ = roc_curve(hy, hp)
ax.plot(fpr, tpr, lw=2.5, color="crimson",
        label=f"Proposed (TPX), honest CV  AUC={roc_auc_score(hy, hp):.3f}")
# honest baselines: pooled out-of-fold on raw features
for name, ctor in [("LR", lambda: LogisticRegression(max_iter=2000)),
                   ("RF", lambda: RandomForestClassifier(random_state=SEED)),
                   ("GB", lambda: GradientBoostingClassifier(random_state=SEED))]:
    bp_, by_ = [], []
    for tr_idx, te_idx in skf.split(Xr_tr, ytr):
        m = ctor().fit(Xr_tr[tr_idx], ytr[tr_idx])
        bp_.append(m.predict_proba(Xr_tr[te_idx])[:, 1]); by_.append(ytr[te_idx])
    bp_ = np.concatenate(bp_); by_ = np.concatenate(by_)
    fpr, tpr, _ = roc_curve(by_, bp_)
    ax.plot(fpr, tpr, lw=1.6, color=colors[name], label=f"{name} (raw), honest CV  AUC={roc_auc_score(by_, bp_):.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Leakage-free evaluation\n(test fold excluded from classifier training)")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("repro_figure3.png", dpi=150)
print("Saved repro_figure3.png")
print(f"Proposed (manuscript design) pooled AUC = {roc_auc_score(pooled_y, pooled_p):.3f}")
print(f"Proposed (honest CV)         pooled AUC = {roc_auc_score(hy, hp):.3f}")
