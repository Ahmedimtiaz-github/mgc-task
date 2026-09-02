# MGC Developments — Take-Home Build

Four parts: a grounded document Q&A assistant, a lead schema + queries, a lead-scoring
baseline model, and a minimal Flask page tying the assistant and the scorer together.

## Setup & run (Windows)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set your key:

```
GEMINI_API_KEY=your_actual_key_here
```

Get a free key (no credit card) at https://aistudio.google.com/apikey.

`model.pkl` (the fitted preprocessing + model pipeline) is committed to the repo, so
`app.py` works out of the box straight after `pip install` — no training step required.

To retrain from scratch (e.g. if `leads.csv` changes), regenerate it with:

```
python part3_lead_scoring.py
```

This overwrites `model.pkl` with a freshly trained pipeline.

Try Part 1 standalone (no args runs all 5 example questions from the brief; with an
argument it answers just that question):

```
python part1_doc_assistant.py
python part1_doc_assistant.py "What's the transfer fee?"
```

Run the web app:

```
python app.py
```

Then open http://127.0.0.1:5000 in a browser.

## Part 1 — model note

Uses `gemini-3.6-flash`. `gemini-2.5-flash` returned a 404 from the live API for this
key/tier ("no longer available to new users") and the error itself named
`gemini-3.6-flash` as the replacement, so that's what's wired in.

## Part 1 — why full-document context instead of a vector DB

The corpus is three short markdown files (a few pages total) — small enough to pass in
full on every query, which is simpler to reason about and easier 
than a chunking/embedding/retrieval pipeline that buys nothing at this size.

## Part 3 — data-cleaning decisions & metric

- **Dropped `token_amount_received_pkr`** — verified it's effectively target leakage:
  every `converted=1` row has a nonzero token amount, and almost every `converted=0`
  row is exactly 0. In production this field isn't known before the outcome, so
  training on it would be cheating and wouldn't generalize.
- **Dropped `lead_id` and `crm_record_hash`** — pure identifiers, no predictive signal,
  risk of the model memorizing rows instead of generalizing.
- **Dropped `created_at`** as a raw field (kept simple, per the brief) — a
  day-of-week/seasonality feature would be a reasonable next step.
- **Normalized `city`** — 21 raw values (`ISB`, `Rwp`, `khi`, upper/lower/mixed case,
  etc.) collapse to 9 real cities; left alone, the model would treat `"Islamabad"` and
  `"ISB"` as unrelated categories.
- **Missing values**: median imputation for numeric columns (`budget_pkr_lac`,
  `bedrooms`, `first_response_minutes`, `agent_experience_years`); an explicit
  `"Unknown"` category for the categorical `area`. Missingness is kept visible rather
  than smoothed away — it may itself carry signal (e.g. a missing first-response time
  could correlate with a neglected lead). `bedrooms` is missing for ~3,602 rows because
  commercial shops and plots legitimately have no bedroom count (verified: 0 non-null
  bedrooms for both property types), not because the data is broken.
- **Metric: PR-AUC (average precision)**, reported as the primary metric, with
  ROC-AUC printed for reference. `converted` is ~6.9% positive, so accuracy is
  misleading (predicting "never converts" scores ~93% and is useless); PR-AUC focuses
  on how well the model ranks the rare positive class, which is what matters for
  prioritizing which leads to call.
- Model: `RandomForestClassifier` (no scaling needed, handles the mix of one-hot
  categoricals and raw numerics without tuning), inside a single `sklearn.Pipeline`
  saved whole to `model.pkl` via `joblib` so `app.py` never has to retrain.

Actual local run:

```
Train rows: 7328, test rows: 1832
Positive class rate (test set): 0.0693
PR-AUC (average precision) [PRIMARY METRIC]: 0.3109
ROC-AUC (reference): 0.8175
```

## Part 2 — schema notes

`crm_record_hash`, not `lead_id`, is the real identity of a lead: 160 `crm_record_hash`
values appear twice in the raw CSV, each pair sharing every attribute but with two
different `lead_id`s (one suffixed `-B`) — the same lead re-entered by a second agent.
`schema.sql` puts a `UNIQUE` constraint on `crm_record_hash` (the fix that would have
stopped this at write time) and uses `lead_id` only as a non-unique CRM audit-trail
column. `queries.sql` has the two required queries with comments.

## What's rough, and what I'd do with more time

- **No retry/backoff on the Gemini call in Part 1** — a transient network error or rate
  limit fails that single request. With more time I'd add exponential backoff and a
  small response cache for repeated questions.
- **No hyperparameter tuning on the Part 3 baseline** (per the brief's "no tuning
  needed"). I'd next try calibrating probabilities (`CalibratedClassifierCV`, since raw
  RandomForest probabilities aren't perfectly calibrated) and add a
  `created_at`-derived day-of-week/season feature.
- **No automated test suite** — verified by running each script standalone against
  real data and the real API, and exercising the web UI directly, rather than by
  regression tests. I'd add unit tests for `normalize_city`, `score_lead`, and the
  JSON-parsing fallback in Part 1 next.
- **Flask app does basic type coercion, not full input validation** — fine for an
  internal sales tool, not hardened for untrusted input.
- **Styling is a light pass, not a design system** — a navy/gold theme inspired by
  MGC's own branding, added after the initial bare-HTML delivery; no responsive layout
  or component library.

