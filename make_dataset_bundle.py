# coding: utf-8
"""Bundle everything tied to app_usage_time_freq_labeled.csv into a self-contained,
portable folder (+zip), so work_app_time_num_ai can be moved out of the project.

Bundle layout:
  app_usage_time_freq_dataset/
    app_usage_time_freq_labeled.csv          (the dataset)
    app_usage_time_freq_labeled_README.md    (data dictionary)
    regenerate_csv.py                         (self-contained rebuild from ./source)
    build_dataset_csv.py                      (original build script, for reference)
    source/
      use_time_data/                          (20,129 daily per-user CSVs = the raw source)
      115位user四周成癮狀態(修改檔)2.xlsx       (per-user-week addiction labels)
      20210721_..._各題分數ryan.xlsx           (KSAS item scores; provenance of labels)
      app_category_lookup.csv                 (frozen app->category map; replaces 109/clean_data)
"""
import os, sys, glob, shutil, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

BASE = "work_app_time_num_ai"
BUNDLE = "app_usage_time_freq_dataset"
SRC = os.path.join(BUNDLE, "source")
LABEL = "115位user四周成癮狀態(修改檔)2.xlsx"
SCORE = "20210721_AI計畫子計劃四_手機成癮量表_各題分數ryan.xlsx"

if os.path.exists(BUNDLE):
    shutil.rmtree(BUNDLE)
os.makedirs(SRC)

# 1) freeze app -> category lookup from 109/clean_data (the source used in the build)
print("Freezing app->category lookup from 109/clean_data ...")
cat = {}
for f in glob.glob(os.path.join(BASE, "109", "*", "clean_data", "*.csv"))[:3000]:
    try:
        d = pd.read_csv(f, usecols=["App name", "Type"])
        for a, ty in zip(d["App name"], d["Type"]):
            if pd.notna(a) and pd.notna(ty) and str(ty) not in ("None", "nan"):
                cat.setdefault(str(a), {}).setdefault(str(ty), 0)
                cat[str(a)][str(ty)] += 1
    except Exception:
        pass
look = pd.DataFrame(
    [(a, max(c, key=c.get)) for a, c in cat.items()],
    columns=["app_name", "app_category"]).sort_values("app_name")
look.to_csv(os.path.join(SRC, "app_category_lookup.csv"), index=False, encoding="utf-8-sig")
print(f"  {len(look)} app->category entries frozen")

# 2) copy raw source (use_time_data) + labels + score file
print("Copying use_time_data (this is the bulk) ...")
shutil.copytree(os.path.join(BASE, "use_time_data"), os.path.join(SRC, "use_time_data"))
shutil.copy2(os.path.join(BASE, LABEL), os.path.join(SRC, LABEL))
shutil.copy2(os.path.join(BASE, SCORE), os.path.join(SRC, SCORE))

# 3) copy the deliverable + docs + scripts
for f in ["app_usage_time_freq_labeled.csv", "app_usage_time_freq_labeled_README.md",
          "build_dataset_csv.py"]:
    if os.path.exists(f):
        shutil.copy2(f, os.path.join(BUNDLE, f))

# 4) write a self-contained regeneration script (reads ./source, frozen lookup)
regen = r'''# coding: utf-8
"""Self-contained rebuild of app_usage_time_freq_labeled.csv from ./source/ ONLY.
No dependency on the original work_app_time_num_ai folder."""
import os, glob, re, warnings, sys
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from datetime import date
SRC = "source"; START = date(2019, 12, 24)
LABEL = "115位user四周成癮狀態(修改檔)2.xlsx"

def hms(s):
    try:
        p = [int(float(x)) for x in str(s).strip().split(":")]
        return p[0]*3600 + p[1]*60 + p[2] if len(p) == 3 else (p[0]*60 + p[1] if len(p) == 2 else p[0])
    except Exception:
        return np.nan
def uid(u):
    d = re.sub(r"\D", "", str(u)); return int(d) if d else np.nan

app_cat = dict(pd.read_csv(os.path.join(SRC, "app_category_lookup.csv")).dropna().values)
files = glob.glob(os.path.join(SRC, "use_time_data", "*.csv"))
raw = pd.concat([pd.read_csv(f, usecols=["App name","Usage time","Access count","Date","User"])
                 for f in files], ignore_index=True)
raw["user_id"] = raw["User"].map(uid); raw["usage_seconds"] = raw["Usage time"].map(hms)
raw["access_count"] = pd.to_numeric(raw["Access count"], errors="coerce")
raw["d"] = pd.to_datetime(raw["Date"], errors="coerce").dt.date
raw = raw.dropna(subset=["user_id","d","App name"]); raw["user_id"] = raw["user_id"].astype(int)
raw["week"] = raw["d"].map(lambda x: (x - START).days // 7); raw = raw[raw["week"] >= 0]
raw["app_name"] = raw["App name"].astype(str)
agg = (raw.groupby(["user_id","week","app_name"])
          .agg(usage_seconds=("usage_seconds","sum"), access_count=("access_count","sum"),
               active_days=("d","nunique")).reset_index())
agg["usage_seconds"] = agg["usage_seconds"].round().astype("Int64")
agg["access_count"] = agg["access_count"].round().astype("Int64")
agg["app_category"] = agg["app_name"].map(app_cat).fillna("")
lab = pd.read_excel(os.path.join(SRC, LABEL)).rename(columns={ "Unnamed: 0": "user_id"})
lab = lab.rename(columns={lab.columns[0]: "user_id"})
wk = [c for c in lab.columns if isinstance(c,(int,np.integer)) and c >= 0]
ll = lab.melt(id_vars=["user_id"], value_vars=wk, var_name="week", value_name="addicted")
ll["week"] = ll["week"].astype(int); ll["user_id"] = ll["user_id"].astype(int)
out = agg.merge(ll, on=["user_id","week"], how="left")
out["addicted"] = out["addicted"].astype("Int64")
out = out.sort_values(["user_id","week","usage_seconds"], ascending=[True,True,False])
out = out[["user_id","week","app_name","app_category","usage_seconds","access_count","active_days","addicted"]]
out.to_csv("app_usage_time_freq_labeled.csv", index=False, encoding="utf-8-sig")
print("rebuilt:", len(out), "rows")
'''
open(os.path.join(BUNDLE, "regenerate_csv.py"), "w", encoding="utf-8").write(regen)

# 5) bundle README
bundle_readme = """# app_usage_time_freq_dataset — portable bundle

Self-contained copy of everything needed for **app_usage_time_freq_labeled.csv**
(app usage **time + frequency** + per-(user,week) addiction label), so the original
`work_app_time_num_ai/` folder can be moved out of the project.

## Contents
- `app_usage_time_freq_labeled.csv` — the dataset (one row per user x week x app).
- `app_usage_time_freq_labeled_README.md` — full data dictionary.
- `regenerate_csv.py` — rebuild the CSV from `source/` alone:  `python regenerate_csv.py`
- `build_dataset_csv.py` — original build script (referenced the full work_app_time_num_ai tree).
- `source/use_time_data/` — raw per-(user, day) telemetry (App name, Usage time, Access count, Date, User). THE source.
- `source/115位user四周成癮狀態(修改檔)2.xlsx` — per-(user, week) KSAS addiction labels.
- `source/20210721_..._各題分數ryan.xlsx` — KSAS per-item scores (label provenance / cohort map).
- `source/app_category_lookup.csv` — frozen app->category map (originally derived from the
  109-cohort `clean_data` `Type` column; frozen here so the large `109/` tree is not required).

## Note
This bundle contains ONLY what feeds this CSV. Other materials in the original folder
(SCL-90-R depression scores, gaming-addiction scores, alternate label versions, model
artifacts, etc.) are NOT included — they are unrelated to this particular dataset.
"""
open(os.path.join(BUNDLE, "BUNDLE_README.md"), "w", encoding="utf-8").write(bundle_readme)

# 6) zip
print("Zipping ...")
shutil.make_archive(BUNDLE, "zip", BUNDLE)
nfiles = sum(len(fs) for _, _, fs in os.walk(BUNDLE))
print(f"\nDONE: {BUNDLE}/ ({nfiles} files) and {BUNDLE}.zip "
      f"({os.path.getsize(BUNDLE + '.zip')/1e6:.1f} MB)")
