# POI Intelligence & Ranking System

ML pipeline that ranks Points of Interest for travelers based on their preferences, trip context, and past behavior. Meant to feed into a downstream itinerary planner.

## Setup

```bash
pip install -r requirements.txt
python run_pipeline.py
```

That's it. The script generates synthetic data on first run, trains a model, runs three demo scenarios, and prints evaluation metrics. Results go to `output/`.

Needs Python 3.9+ with numpy, pandas, scikit-learn, scipy.

## What it does

```
POI data + traveler profile + interaction history
  -> data cleaning & feature engineering
  -> candidate generation (500 -> ~100 POIs)
  -> personalized ranking (gradient boosted trees)
  -> context scoring (budget, mobility, availability)
  -> ranked POIs with scores and explanations
```

## Project layout

```
run_pipeline.py              <- run this
src/
  data_generator.py          <- synthetic data (500 POIs, 50 travelers, 5k interactions)
  data_preparation.py        <- cleaning, normalization, dedup
  poi_representation.py      <- POI features (categories, tags, tfidf, behavioral)
  traveler_representation.py <- traveler profiles (explicit + implicit preferences)
  candidate_generation.py    <- multi-signal filtering with diversity sampling
  ranking_model.py           <- GBT pointwise ranker
  context_scoring.py         <- budget/mobility/availability scoring
  explainability.py          <- human-readable recommendation reasons
  evaluation.py              <- ndcg, precision, personalization, coverage, etc
  pipeline.py                <- orchestrates everything
data/                        <- generated CSVs
output/                      <- scenario results + evaluation report
TECHNICAL_DOCUMENTATION.md   <- detailed writeup
```

## Demo scenarios

Three traveler profiles to show the system produces different rankings for different people (not just a popularity list):

1. **Local food explorer** — interested in restaurants, neighborhoods, local spots. Medium budget. Avoids touristy places.
2. **History & culture** — museums, architecture, historical sites. High budget. Okay with famous landmarks.
3. **Family trip** — parks, family activities, nature. Medium budget. Needs child-friendly options.

The personalization metric comes out near 1.0, meaning each traveler gets almost entirely different recommendations.

## Output format

```json
{
  "poi_id": "P0316",
  "score": 0.7967,
  "preference_weight": 0.7309,
  "context_compatibility": 0.7924,
  "confidence": 0.9632
}
```

Downstream system can use `score` for ordering, `preference_weight` for relevance, `context_compatibility` for feasibility, `confidence` for how much to trust the recommendation.

See [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) for the full writeup on model choice, feature engineering, cold start, evaluation, and production considerations.
