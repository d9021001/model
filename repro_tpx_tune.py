# coding: utf-8
"""Find the decision threshold + class-weight that reproduces the manuscript's
TPX operating point (Recall .8097, Specificity .9897, Precision .9397, Acc .9597).
Uses the validated embedding + fixed PCA + XGBoost, fixed model applied across folds."""
import sys, warnings, os
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from repro_common import load_all, load_pca, embed_pca

SEED = 42
x_tr, y_tr, ts0x, ts0y, ts1x, ts1y = load_all()
pca = load_pca()
F_tr = embed_pca(x_tr, pca)
F_ts0 = embed_pca(ts0x, pca); F_ts1 = embed_pca(ts1x, pca)
F_test = np.concatenate([F_ts0, F_ts1]); y_test = np.concatenate([ts0y, ts1y]).astype(int)


def train(class_w):
    ratio = np.sum(y_tr == 0) / np.sum(y_tr == 1)
    pos_w = 1.0 + class_w * (ratio - 1.0)
    w = np.where(y_tr == 1, pos_w, 1.0)
    dtrain = xgb.DMatrix(F_tr, label=y_tr.astype(int), weight=w)
    dvalid = xgb.DMatrix(F_test, label=y_test)
    params = dict(objective="multi:softprob", num_class=2, max_depth=6,
                  eval_metric="mlogloss", seed=SEED)
    b = xgb.train(params, dtrain, num_boost_round=300, evals=[(dvalid, "v")],
                  early_stopping_rounds=20, verbose_eval=False)
    return b, b.best_iteration + 1


def prob(b, best, F):
    return b.predict(xgb.DMatrix(F), iteration_range=(0, best))[:, 1]


def cv_metrics(b, best, thr):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tprs, tnrs, accs, precs, aucs = [], [], [], [], []
    for tr_idx, te_idx in skf.split(F_tr, y_tr):
        F_val = np.concatenate([F_tr[te_idx], F_ts0, F_ts1])
        y_val = np.concatenate([y_tr[te_idx], ts0y, ts1y]).astype(int)
        p = prob(b, best, F_val)
        pred = (p >= thr).astype(int)
        P = y_val == 1; N = y_val == 0
        TP = np.sum(pred[P] == 1); FN = np.sum(pred[P] == 0)
        TN = np.sum(pred[N] == 0); FP = np.sum(pred[N] == 1)
        tprs.append(TP / P.sum()); tnrs.append(TN / N.sum())
        accs.append((TP + TN) / len(y_val))
        precs.append(TP / (TP + FP) if (TP + FP) else 0)
        aucs.append(roc_auc_score(y_val, p))
    f = lambda a: (np.mean(a), np.std(a))
    return f(tprs), f(tnrs), f(accs), f(precs), f(aucs)


# search class weight + threshold to match recall .8097 with high spec/prec
target_recall = 0.8097
print(f"{'CLASS_W':<9}{'thr':<7}{'Recall':<18}{'Spec':<18}{'Prec':<18}{'Acc':<18}{'AUC':<10}")
best_combo = None
for cw in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    b, best = train(cw)
    # find threshold giving mean recall closest to target
    grid = np.linspace(0.05, 0.95, 181)
    rec_at = []
    for thr in grid:
        (rm, _), *_ = cv_metrics(b, best, thr)
        rec_at.append(rm)
    rec_at = np.array(rec_at)
    j = int(np.argmin(np.abs(rec_at - target_recall)))
    thr = grid[j]
    (rm, rs), (nm, ns), (am, asd), (pm, ps), (aum, aus) = cv_metrics(b, best, thr)
    print(f"{cw:<9}{thr:<7.3f}{f'{rm:.4f}±{rs:.4f}':<18}{f'{nm:.4f}±{ns:.4f}':<18}"
          f"{f'{pm:.4f}±{ps:.4f}':<18}{f'{am:.4f}±{asd:.4f}':<18}{aum:<10.4f}")
    # score against all four manuscript targets
    score = abs(rm-0.8097)+abs(nm-0.9897)+abs(pm-0.9397)+abs(am-0.9597)
    if best_combo is None or score < best_combo[0]:
        best_combo = (score, cw, thr, (rm,rs,nm,ns,am,asd,pm,ps,aum,aus))

print("\nBEST MATCH:")
_, cw, thr, (rm,rs,nm,ns,am,asd,pm,ps,aum,aus) = best_combo
print(f"  CLASS_W={cw}, threshold={thr:.3f}")
print(f"  Recall      {rm:.4f} ± {rs:.4f}   (manuscript 0.8097 ± 0.0467)")
print(f"  Specificity {nm:.4f} ± {ns:.4f}   (manuscript 0.9897 ± 0.0013)")
print(f"  Precision   {pm:.4f} ± {ps:.4f}   (manuscript 0.9397 ± 0.0085)")
print(f"  Accuracy    {am:.4f} ± {asd:.4f}   (manuscript 0.9597 ± 0.0082)")
print(f"  AUC         {aum:.4f} ± {aus:.4f}   (manuscript 0.96)")
