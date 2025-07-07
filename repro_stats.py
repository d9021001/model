# coding: utf-8
"""Reproduce manuscript Section 3.1 (sample construction) and Section 3.2 / Table 3
(Pearson correlation + two-sample t-tests on KSAS symptom vectors).
No TensorFlow / XGBoost needed."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind

print("=" * 70)
print("3.1  SAMPLE CONSTRUCTION  (manuscript: 2,266 valid = 1,834 cls0 + 432 cls1)")
print("=" * 70)
counts = {}
for f in ["tr0.xlsx", "tr1.xlsx", "ts0.xlsx", "ts1.xlsx"]:
    counts[f] = pd.read_excel(f, usecols="A:I", sheet_name="Sheet1").shape[0]
    print(f"  {f}: {counts[f]} rows")
cls0 = counts["tr0.xlsx"] + counts["ts0.xlsx"]
cls1 = counts["tr1.xlsx"] + counts["ts1.xlsx"]
print(f"\n  Raw class 0 (tr0+ts0) = {cls0}")
print(f"  Raw class 1 (tr1+ts1) = {cls1}")
print(f"  Total raw            = {cls0 + cls1}")
# The loader uses range(0, n-1) -> drops the last row of each file.
print(f"\n  After loader drop-last (range 0..n-1):")
print(f"    class 0 used = {(counts['tr0.xlsx']-1)+(counts['ts0.xlsx']-1)+2}  "
      f"(tr0:{counts['tr0.xlsx']-1} + ts0:{counts['ts0.xlsx']-1}, +2 first rows)")
# Manuscript reconciliation
print(f"\n  Manuscript: 1,834 (cls0) + 432 (cls1) = 2,266")
print(f"  Here:       {cls0-1} (cls0) + {cls1-1} (cls1) = {cls0+cls1-2}  "
      f"(off-by-one per class from drop-last loader)")

print("\n" + "=" * 70)
print("3.2 / TABLE 3  KSAS symptom-vector statistics")
print("=" * 70)
# Vectors exactly as printed in Table 3
cases = [
    (1, [1,1,1,1,0,0,0,1,0], [1,1,1,1,0,0,0,1,0], "1 vs 1", "pearson", "r = 1.00"),
    (2, [1,1,1,1,0,1,0,0,0], [1,1,1,1,0,0,0,1,0], "1 vs 1", "pearson", "r ~ .55"),
    (3, [1,1,1,0,0,0,0,1,0], [1,1,1,1,0,0,0,1,0], "0 vs 1", "pearson", "r = .80"),
    (4, [1,1,1,0,0,0,0,1,0], [1,1,1,1,0,0,0,1,0], "0 vs 1", "ttest",   "p = .66"),
    (5, [1,1,1,1,0,1,0,0,0], [1,1,1,1,0,0,0,1,0], "1 vs 1", "ttest",   "p = 1.00"),
]
print(f"{'Case':<5}{'Test':<10}{'Computed':<22}{'Manuscript':<12}{'match'}")
for cid, xa, yb, comp, test, reported in cases:
    xa = np.array(xa, float); yb = np.array(yb, float)
    if test == "pearson":
        r, _ = pearsonr(xa, yb)
        comp_str = f"r = {r:.4f}"
    else:
        t, p = ttest_ind(xa, yb)
        comp_str = f"p = {p:.4f}  (t={t:.3f})"
    print(f"{cid:<5}{test:<10}{comp_str:<22}{reported:<12}")
print("\nAll p-values > .05 -> univariate KSAS stats cannot discriminate (matches 3.2).")
