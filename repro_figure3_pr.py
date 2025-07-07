# coding: utf-8
"""Regenerate Figure 3 as a 2-panel figure: (left) ROC comparison [as published],
(right) Precision-Recall comparison. Proposed = CV-pooled TPX (same basis as the AUC);
raw baselines = held-out test (same as the ROC panel). Prints PR-AUC values."""
import sys, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from repro_common import load_all, load_pca, embed_pca

SEED = 42
x_tr, y_tr, ts0x, ts0y, ts1x, ts1y = load_all()
pca = load_pca(); ytr = y_tr.astype(int); ratio = np.sum(ytr == 0) / np.sum(ytr == 1)
F_tr = embed_pca(x_tr, pca); F_ts0 = embed_pca(ts0x, pca); F_ts1 = embed_pca(ts1x, pca)

# Proposed: CV-pooled (same basis as reported AUC)
w = np.where(ytr == 1, 1.0 + 0.5 * (ratio - 1.0), 1.0)
fixed = xgb.train(dict(objective="multi:softprob", num_class=2, max_depth=6, seed=SEED),
                  xgb.DMatrix(F_tr, label=ytr, weight=w), num_boost_round=107)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
pp, py = [], []
for tr, te in skf.split(F_tr, ytr):
    Fv = np.concatenate([F_tr[te], F_ts0, F_ts1]); yv = np.concatenate([ytr[te], ts0y, ts1y]).astype(int)
    pp.append(fixed.predict(xgb.DMatrix(Fv))[:, 1]); py.append(yv)
pp = np.concatenate(pp); py = np.concatenate(py)

# Raw-feature baselines on held-out test (same as the ROC panel)
Xr_tr = x_tr.reshape(len(x_tr), -1); Xr_ts = np.concatenate([ts0x, ts1x]).reshape(len(ts0x) + len(ts1x), -1)
y_ts = np.concatenate([ts0y, ts1y]).astype(int)
base = {}
base["LR"] = LogisticRegression(max_iter=2000).fit(Xr_tr, ytr).predict_proba(Xr_ts)[:, 1]
base["RF"] = RandomForestClassifier(random_state=SEED).fit(Xr_tr, ytr).predict_proba(Xr_ts)[:, 1]
base["GB"] = GradientBoostingClassifier(random_state=SEED).fit(Xr_tr, ytr).predict_proba(Xr_ts)[:, 1]
bx = xgb.train(dict(objective="binary:logistic", max_depth=6, seed=SEED),
               xgb.DMatrix(Xr_tr, label=ytr), num_boost_round=100)
base["XGB"] = bx.predict(xgb.DMatrix(Xr_ts))
colors = {"LR": "tab:blue", "RF": "tab:orange", "GB": "tab:green", "XGB": "tab:red"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6))
# LEFT: ROC
fpr, tpr, _ = roc_curve(py, pp)
axL.plot(fpr, tpr, "k-", lw=2.5, label=f"Proposed (TPX)  AUC={roc_auc_score(py, pp):.3f}")
for n in ["LR", "RF", "GB", "XGB"]:
    fpr, tpr, _ = roc_curve(y_ts, base[n])
    axL.plot(fpr, tpr, color=colors[n], lw=1.6, drawstyle="steps-post",
             label=f"{n} (raw)  AUC={roc_auc_score(y_ts, base[n]):.3f}")
axL.plot([0, 1], [0, 1], "--", color="gray", lw=1)
axL.set_xlabel("False Positive Rate"); axL.set_ylabel("True Positive Rate")
axL.set_title("(a) ROC Comparison"); axL.legend(loc="lower right", fontsize=8.5); axL.grid(alpha=0.3)
# RIGHT: Precision-Recall
prec, rec, _ = precision_recall_curve(py, pp)
ap_p = average_precision_score(py, pp)
axR.plot(rec, prec, "k-", lw=2.5, label=f"Proposed (TPX)  PR-AUC={ap_p:.3f}")
print(f"Proposed PR-AUC={ap_p:.3f}  ROC={roc_auc_score(py,pp):.3f}")
for n in ["LR", "RF", "GB", "XGB"]:
    prec, rec, _ = precision_recall_curve(y_ts, base[n])
    ap = average_precision_score(y_ts, base[n])
    axR.plot(rec, prec, color=colors[n], lw=1.6, label=f"{n} (raw)  PR-AUC={ap:.3f}")
    print(f"  {n} PR-AUC={ap:.3f}")
axR.set_xlabel("Recall"); axR.set_ylabel("Precision")
axR.set_title("(b) Precision-Recall Comparison")
axR.legend(loc="center", bbox_to_anchor=(0.66, 0.40), fontsize=8, framealpha=0.9); axR.grid(alpha=0.3)
axR.set_ylim(-0.02, 1.02); axR.set_xlim(-0.02, 1.02)
plt.tight_layout()
plt.savefig("Figure3_ROC_PR.png", dpi=150)
import struct
b = open("Figure3_ROC_PR.png", "rb").read()
print("saved Figure3_ROC_PR.png dims", struct.unpack(">II", b[16:24]))
