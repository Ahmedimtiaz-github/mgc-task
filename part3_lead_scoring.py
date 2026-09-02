"""
Baseline lead-scoring model for MGC Developments.

Run standalone to clean leads.csv, train a baseline classifier, print the
chosen metric, and save the fitted pipeline to model.pkl. Also exposes
`score_lead(lead_dict)` for reuse by app.py.

--------------------------------------------------------------------------
Data-cleaning decisions (see also README.md)
--------------------------------------------------------------------------
DROP token_amount_received_pkr:
    Verified it is effectively target leakage: every converted=1 row has a
    nonzero token amount, and the overwhelming majority of converted=0 rows
    have exactly 0 (a handful of nonzero-token, non-converted rows are
    presumably bookings that were later cancelled). In production this field
    would not exist yet at scoring time (it's an outcome of the sale, not a
    predictor of it), so training on it would be cheating and useless.

DROP lead_id, crm_record_hash:
    Pure identifiers, no predictive signal. Keeping them risks the model
    memorizing individual rows (or, worse, learning something spurious from
    the hash's numeric value) rather than generalizing.

DROP created_at (as a raw field):
    Kept simple per the brief; not derived further here. A day-of-week or
    seasonality feature would be a reasonable next step (see README).

NORMALIZE city:
    21 distinct raw strings collapse to ~9 real cities once case and known
    abbreviations are folded together (e.g. "ISB"/"ISLAMABAD" -> "Islamabad",
    "Rwp" -> "Rawalpindi", "khi" -> "Karachi"). Left un-normalized, the model
    would treat "Islamabad" and "ISB" as unrelated categories and dilute the
    signal a city genuinely carries.

MISSING VALUES:
    - Numeric columns (budget_pkr_lac, bedrooms, first_response_minutes,
      agent_experience_years): median imputation. Medians are robust to the
      outliers common in budget/experience figures.
    - Categorical columns (area): an explicit "Unknown" category rather than
      dropping rows or imputing a mode. Missingness itself may carry signal
      (e.g. a lead whose first response time is missing may correlate with a
      slow/negligent follow-up), so we keep it visible to the model as its
      own value instead of silently smoothing it away.
    - bedrooms is property_type-dependent (commercial shops and plots
      legitimately have no bedroom count -- confirmed: 0 non-null bedrooms
      for both types in this data) so its ~3600 missing values are not messy
      data, they are structurally absent; median imputation still applies
      because the model needs a numeric value, but the missingness is
      expected, not an error to chase down.

--------------------------------------------------------------------------
Metric choice
--------------------------------------------------------------------------
converted is ~6.9% positive. Accuracy is misleading here: a model that always
predicts "not converted" scores ~93% accuracy while being useless. We report
PR-AUC (average precision) as the primary metric because, with this much
class imbalance, it focuses on how well the model ranks the rare positive
class specifically (precision/recall trade-off) rather than being dominated
by the large negative class the way ROC-AUC's false-positive-rate axis can
be. ROC-AUC is also printed for reference.
"""

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "leads.csv"
MODEL_PATH = "model.pkl"

CATEGORICAL_FEATURES = ["source", "city", "area", "property_type"]
NUMERIC_FEATURES = [
    "budget_pkr_lac",
    "bedrooms",
    "first_response_minutes",
    "calls_made",
    "total_call_seconds",
    "whatsapp_replies",
    "site_visits",
    "agent_experience_years",
]
BOOLEAN_FEATURES = [
    "is_overseas",
    "referred_by_existing_client",
    "has_financing_approved",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOLEAN_FEATURES
TARGET = "converted"

CITY_ALIASES = {
    "islamabad": "Islamabad",
    "isb": "Islamabad",
    "rawalpindi": "Rawalpindi",
    "rwp": "Rawalpindi",
    "lahore": "Lahore",
    "karachi": "Karachi",
    "khi": "Karachi",
    "peshawar": "Peshawar",
    "faisalabad": "Faisalabad",
    "multan": "Multan",
    "gujranwala": "Gujranwala",
    "abbottabad": "Abbottabad",
}


def normalize_city(city) -> str:
    if pd.isna(city):
        return "Unknown"
    key = str(city).strip().lower()
    return CITY_ALIASES.get(key, str(city).strip().title())


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["city"] = df["city"].apply(normalize_city)
    df["area"] = df["area"].fillna("Unknown")

    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(bool)

    df = df.drop(
        columns=["token_amount_received_pkr", "lead_id", "crm_record_hash", "created_at"]
    )
    return df


def build_pipeline() -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ]
    )
    # RandomForest: no scaling needed, handles the mix of one-hot categoricals
    # and raw numerics fine, and is a reasonable no-tuning baseline.
    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, class_weight="balanced", random_state=42
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_and_evaluate(path: str = DATA_PATH):
    df = load_and_clean(path)
    X = df[ALL_FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)}")
    print(f"Positive class rate (test set): {y_test.mean():.4f}")
    print(f"PR-AUC (average precision) [PRIMARY METRIC]: {pr_auc:.4f}")
    print(f"ROC-AUC (reference): {roc_auc:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved fitted pipeline to {MODEL_PATH}")

    return pipeline, pr_auc, roc_auc


_pipeline_cache = None


def _get_pipeline():
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = joblib.load(MODEL_PATH)
    return _pipeline_cache


def score_lead(lead_dict: dict) -> float:
    """Return P(converted=1) for a single lead described by `lead_dict`."""
    pipeline = _get_pipeline()

    row = {}
    for col in CATEGORICAL_FEATURES:
        if col == "city":
            row[col] = normalize_city(lead_dict.get("city"))
        else:
            val = lead_dict.get(col)
            row[col] = "Unknown" if val in (None, "") else val
    for col in NUMERIC_FEATURES:
        val = lead_dict.get(col)
        row[col] = np.nan if val in (None, "") else float(val)
    for col in BOOLEAN_FEATURES:
        row[col] = bool(lead_dict.get(col, False))

    X = pd.DataFrame([row])[ALL_FEATURES]
    proba = pipeline.predict_proba(X)[0, 1]
    return float(proba)


if __name__ == "__main__":
    train_and_evaluate()
    if len(sys.argv) > 1:
        print("(model.pkl trained and saved; score_lead() is ready for use by app.py)")
