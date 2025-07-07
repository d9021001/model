# Dataset: `app_usage_time_freq_labeled.csv`

Tidy, analysis-ready dataset for the **second paper** — combines, for the labeled
cohort, **both app usage time and usage frequency** with the per-week addiction label.

## Source & cohort resolution
- **Data**: `work_app_time_num_ai/use_time_data/*.csv` — the **2019 labeled cohort**
  (20,129 per-user-per-day files; `User = user{ID:03d}`, e.g. `user002` → id 2).
  Each daily row carries `App name`, `Usage time` (H:MM:SS), `Access count` (= launches/frequency).
- **Labels**: `work_app_time_num_ai/115位user四周成癮狀態(修改檔)2.xlsx` — per-(user, week)
  KSAS addiction status (0/1), weeks 0–24. (Verified: user 1 = weeks 0–7 addicted then 0 ✔.)
- **Category**: best-effort `app_name → app_category` lookup built from the `Type` column of
  the `109/*/clean_data` files (covers ~408 app names; blank if unmatched).
- Note: the `109/clean_data` folders are a **different (2020) cohort** — only ~18 overlap
  the labels, so they were **not** used as the source. `use_time_data` is the labeled cohort
  (all 115 label IDs present).

## Grain & columns (one row per user × week × app)
| column | meaning |
|---|---|
| `user_id` | participant id (int; = label id, e.g. `user002` → 2) |
| `week` | 0-based study week (week 0 starts 2019-12-24; matches label-file week columns) |
| `app_name` | application name |
| `app_category` | functional category (通訊/社交/影音… ; best-effort, may be blank) |
| `usage_seconds` | **usage time** that week (Σ daily foreground seconds) |
| `access_count` | **usage frequency** that week (Σ daily launch/access counts) |
| `active_days` | # days that week with data for this app (0–7) |
| `addicted` | per-(user,week) KSAS label: 1=addicted, 0=non-addicted, blank if week>24 |

## Size
- 132,726 rows · 125 users · weeks 0–33 (labels for weeks 0–24).
- Labeled rows: 122,210 (115 users); addicted=1: 24,032, addicted=0: 98,178.
- ~42 apps per user-week on average (no app-level filtering applied — filter as needed).

## Deriving the first-paper "top-7" input
```python
import pandas as pd
df = pd.read_csv("app_usage_time_freq_labeled.csv")
top7 = (df.sort_values("usage_seconds", ascending=False)
          .groupby(["user_id","week"]).head(7))   # 7 most-used apps per user-week
# each row keeps both usage_seconds (time) and access_count (frequency) + addicted
```

Rebuild with `build_dataset_csv.py`.
