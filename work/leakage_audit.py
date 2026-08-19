"""
Leakage audit and corrected model run.

Background
----------
The Week 3 and Week 6 leakage checks looked for forbidden columns BY NAME and
both passed. They could not detect that the label was still reconstructible
from columns that are individually allowed.

docs/data-dictionary.md defines the label as:

    trend_pct       = (impressions_last_30d - impressions_prev_30d)
                      / impressions_prev_30d * 100
    trend_direction = "down" when trend_pct < -20
    label           = 1 when trend_direction == "down"

Both ingredients were in the feature set, so excluding `trend_pct` by name left
the label fully available to the model.

This script:
  0. reproduces the published w05 result on all 26 features;
  1. audits the leak three ways (formula, reconstruction probe, single-feature AUC);
  2. retrains without the six recent-window columns;
  3. repeats the w06 row-based vs client-grouped comparison on both feature sets.

Run from the repository root:

    python work/leakage_audit.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42

CANDIDATE_PATHS = [
    "data/raw/content_refresh_anonymized.csv",
    "../data/raw/content_refresh_anonymized.csv",
    "/content/data/raw/content_refresh_anonymized.csv",
]
CSV_URL = (
    "https://raw.githubusercontent.com/Nayab-khalid/FlyRank-AI-Internship/"
    "main/data/raw/content_refresh_anonymized.csv"
)

# The original 26-feature vector from w05_model.ipynb.
NUMERIC_FULL = [
    "impressions_90d", "clicks_90d", "sessions_90d", "users_90d",
    "engaged_sessions_90d", "ai_sessions_90d", "days_with_impressions",
    "days_with_sessions", "impressions_last_30d", "clicks_last_30d",
    "sessions_last_30d", "impressions_prev_30d", "clicks_prev_30d",
    "sessions_prev_30d", "word_count", "char_count", "content_age_days",
    "days_since_last_update", "ctr", "avg_position", "engagement_rate",
    "scroll_rate", "ai_traffic_pct",
]
CATEGORICAL = ["content_type", "main_intent", "competition_level"]

# The label's own ingredients, plus the same two windows measured on clicks and
# sessions: not the label by definition, but close enough to approximate it.
RECENT_WINDOW = [
    "impressions_last_30d", "impressions_prev_30d",
    "clicks_last_30d", "clicks_prev_30d",
    "sessions_last_30d", "sessions_prev_30d",
]
NUMERIC_CLEAN = [c for c in NUMERIC_FULL if c not in RECENT_WINDOW]


def load():
    path = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)
    df = pd.read_csv(path if path else CSV_URL)
    df["is_declining_label"] = (
        df["trend_direction"].astype(str).str.lower() == "down"
    ).astype(int)
    return df


def build_model(numeric):
    pre = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocessor", pre),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED)),
    ])


def precision_at_k(y_true, scores, k=50):
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    k = min(k, len(scores))
    return y_true[np.argsort(-scores)[:k]].mean()


def week4_baseline(test_rows):
    """The Week 4 rule, recreated exactly as in w05_model.ipynb."""
    b = test_rows.copy()
    imp = pd.to_numeric(b["impressions_90d"], errors="coerce").fillna(0)
    stale_days = pd.to_numeric(b["days_since_last_update"], errors="coerce").fillna(0)
    staleness = np.select([stale_days >= 180, stale_days >= 90], [2, 1], default=0)
    visibility = np.select([imp >= 3000, imp >= 500, imp >= 100], [3, 2, 1], default=0)
    score = pd.Series(staleness + visibility, index=b.index).astype(float)
    # Deterministic tie-break on impressions. Does not use the label.
    return score + imp.rank(method="average") / (len(b) * 1_000_000)


def evaluate(df, numeric, split="group"):
    X = df[numeric + CATEGORICAL]
    y = df["is_declining_label"]
    if split == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        train_idx, test_idx = next(splitter.split(X, y, groups=df["client_id"]))
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        train_idx, test_idx = next(splitter.split(X, y))

    model = build_model(numeric).fit(X.iloc[train_idx], y.iloc[train_idx])
    p = model.predict_proba(X.iloc[test_idx])[:, 1]
    b = week4_baseline(df.iloc[test_idx])
    y_test = y.iloc[test_idx]

    return {
        "test_base_rate": y_test.mean(),
        "model_ap": average_precision_score(y_test, p),
        "model_auc": roc_auc_score(y_test, p),
        "model_p50": precision_at_k(y_test, p, 50),
        "base_ap": average_precision_score(y_test, b),
        "base_auc": roc_auc_score(y_test, b),
        "base_p50": precision_at_k(y_test, b, 50),
    }


def report(title, r):
    print("\n--- %s ---" % title)
    print("  test base rate: %.4f" % r["test_base_rate"])
    print("  BASELINE  AP=%.6f  AUC=%.6f  P@50=%.2f"
          % (r["base_ap"], r["base_auc"], r["base_p50"]))
    print("  MODEL     AP=%.6f  AUC=%.6f  P@50=%.2f"
          % (r["model_ap"], r["model_auc"], r["model_p50"]))


def audit(df):
    """The two checks the name-based audit was missing."""
    y = df["is_declining_label"]

    # Check 1: apply the documented label formula to the feature set.
    prev = pd.to_numeric(df["impressions_prev_30d"], errors="coerce")
    last = pd.to_numeric(df["impressions_last_30d"], errors="coerce")
    rule = (((last - prev) / prev.replace(0, np.nan)) * 100 < -20).fillna(False).astype(int)
    print("\n  documented rule vs label agreement: %.6f" % (rule == y).mean())
    print("  positives predicted: %d    actual: %d" % (rule.sum(), y.sum()))

    # Check 2: can a probe rebuild the label from the features alone?
    for name, cols in (("23 numeric (original)", NUMERIC_FULL),
                       ("17 numeric (corrected)", NUMERIC_CLEAN)):
        acc = cross_val_score(
            RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
            df[cols].fillna(0), y, cv=3, scoring="accuracy").mean()
        print("  reconstruction accuracy, %-22s: %.4f" % (name, acc))
    print("  (base rate %.4f — a probe near 1.0 is reading the label, not learning it)"
          % y.mean())

    # Check 3: any single feature that is the label in disguise.
    flagged = []
    for c in NUMERIC_FULL:
        v = pd.to_numeric(df[c], errors="coerce")
        a = roc_auc_score(y, v.fillna(v.median()))
        if max(a, 1 - a) > 0.95:
            flagged.append((c, a))
    print("  single-feature AUC > 0.95: %s"
          % (flagged if flagged else "none (this leak is a ratio, not one column)"))


def main():
    df = load()
    print("=" * 68)
    print("STEP 0 - REPRODUCE THE PUBLISHED w05 RESULT (26 features)")
    print("=" * 68)
    leaky = evaluate(df, NUMERIC_FULL, "group")
    report("26 features, client-grouped", leaky)
    print("\n  published: model AP=0.871540 AUC=0.849590 P@50=1.00")

    print("\n" + "=" * 68)
    print("STEP 1 - LEAKAGE AUDIT")
    print("=" * 68)
    audit(df)

    print("\n" + "=" * 68)
    print("STEP 2 - RETRAIN WITHOUT THE LABEL INGREDIENTS (20 features)")
    print("=" * 68)
    print("\n  dropped: %s" % ", ".join(RECENT_WINDOW))
    clean = evaluate(df, NUMERIC_CLEAN, "group")
    report("20 features, client-grouped (reported result)", clean)

    print("\n" + "=" * 68)
    print("STEP 3 - VALIDATION DESIGN AUDIT (w06)")
    print("=" * 68)
    report("26 features, row-based", evaluate(df, NUMERIC_FULL, "row"))
    report("20 features, row-based", evaluate(df, NUMERIC_CLEAN, "row"))


if __name__ == "__main__":
    main()
