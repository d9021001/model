# coding: utf-8
"""Organize the labeled cohort (use_time_data, 2019 cohort) into ONE tidy CSV with
BOTH app usage time AND usage frequency (access count), plus the per-(user, week)
addiction label, for the second paper.

Source of truth:
  work_app_time_num_ai/use_time_data/*.csv  — per-(user, day) rows:
      App name, Usage time (H:MM:SS), Access count, Date, User(=user{ID:03d})
  work_app_time_num_ai/115位user四周成癮狀態(修改檔)2.xlsx — per-(user, week) label (0/1),
      col -1 = user ID, cols 0..24 = weekly addiction status.
App category is added best-effort from the clean_data 'Type' column (109 cohort).

Output: app_usage_time_freq_labeled.csv  (one row per user x week x app)
"""
import sys, os, glob, re, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from datetime import date

BASE = "work_app_time_num_ai"
UTD = os.path.join(BASE, "use_time_data")
START = date(2019, 12, 24)            # week 0 begins here (earliest date in cohort)


def hms_to_sec(s):
    try:
        parts = str(s).strip().split(":")
        parts = [int(float(p)) for p in parts]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 1:
            return parts[0]
    except Exception:
        pass
    return np.nan


def uid_of(u):
    d = re.sub(r"\D", "", str(u))
    return int(d) if d else np.nan


# ---- 1) build app -> category lookup from clean_data (best effort) ----
print("Building app->category lookup from clean_data ...")
cat = {}
clean_files = glob.glob(os.path.join(BASE, "109", "*", "clean_data", "*.csv"))
for f in clean_files[:1500]:
    try:
        d = pd.read_csv(f, usecols=["App name", "Type"])
        for a, ty in zip(d["App name"], d["Type"]):
            if pd.notna(a) and pd.notna(ty) and str(ty) not in ("None", "nan"):
                cat.setdefault(str(a), {}).setdefault(str(ty), 0)
                cat[str(a)][str(ty)] += 1
    except Exception:
        pass
app_cat = {a: max(c, key=c.get) for a, c in cat.items()}
print(f"  category lookup covers {len(app_cat)} app names")

# ---- 2) read all use_time_data daily files ----
files = glob.glob(os.path.join(UTD, "*.csv"))
print(f"Reading {len(files)} daily files from use_time_data ...")
frames = []
for i, f in enumerate(files):
    try:
        d = pd.read_csv(f, usecols=["App name", "Usage time", "Access count", "Date", "User"])
        frames.append(d)
    except Exception:
        pass
    if (i + 1) % 4000 == 0:
        print(f"  ... {i+1}/{len(files)}")
raw = pd.concat(frames, ignore_index=True)
print(f"  raw daily app-rows: {len(raw)}")

# ---- 3) clean / derive ----
raw["user_id"] = raw["User"].map(uid_of)
raw["usage_seconds"] = raw["Usage time"].map(hms_to_sec)
raw["access_count"] = pd.to_numeric(raw["Access count"], errors="coerce")
raw["d"] = pd.to_datetime(raw["Date"], errors="coerce").dt.date
raw = raw.dropna(subset=["user_id", "d", "App name"])
raw["user_id"] = raw["user_id"].astype(int)
raw["week"] = raw["d"].map(lambda x: (x - START).days // 7)
raw = raw[raw["week"] >= 0]
raw["app_name"] = raw["App name"].astype(str)

# ---- 4) aggregate per (user, week, app): sum time + sum count + active days ----
agg = (raw.groupby(["user_id", "week", "app_name"])
          .agg(usage_seconds=("usage_seconds", "sum"),
               access_count=("access_count", "sum"),
               active_days=("d", "nunique"))
          .reset_index())
agg["usage_seconds"] = agg["usage_seconds"].round().astype("Int64")
agg["access_count"] = agg["access_count"].round().astype("Int64")
agg["app_category"] = agg["app_name"].map(app_cat).fillna("")

# ---- 5) join per-(user, week) addiction label ----
lab = pd.read_excel(os.path.join(BASE, "115位user四周成癮狀態(修改檔)2.xlsx"))
lab = lab.rename(columns={lab.columns[0]: "user_id"})
week_cols = [c for c in lab.columns if isinstance(c, (int, np.integer)) and c >= 0]
long_lab = lab.melt(id_vars=["user_id"], value_vars=week_cols,
                    var_name="week", value_name="addicted")
long_lab["week"] = long_lab["week"].astype(int)
long_lab["user_id"] = long_lab["user_id"].astype(int)
out = agg.merge(long_lab, on=["user_id", "week"], how="left")
out["addicted"] = out["addicted"].astype("Int64")

# ---- 6) order + write ----
out = out.sort_values(["user_id", "week", "usage_seconds"],
                      ascending=[True, True, False]).reset_index(drop=True)
out = out[["user_id", "week", "app_name", "app_category",
           "usage_seconds", "access_count", "active_days", "addicted"]]
out.to_csv("app_usage_time_freq_labeled.csv", index=False, encoding="utf-8-sig")

print("\n=== app_usage_time_freq_labeled.csv written ===")
print(f"rows={len(out)}, users={out.user_id.nunique()}, weeks={sorted(out.week.unique())[:3]}..{out.week.max()}")
print(f"rows WITH a label (week 0-24): {out.addicted.notna().sum()}  "
      f"(labeled users: {out[out.addicted.notna()].user_id.nunique()})")
print(f"addicted=1 rows: {(out.addicted==1).sum()}, =0 rows: {(out.addicted==0).sum()}")
print("\nsample:")
print(out.head(10).to_string())
