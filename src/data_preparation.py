"""
cleans and normalizes the raw POI data before feeding it into the pipeline.

main things this handles:
- missing values (fill with sane defaults)
- duplicates (same name + neighborhood = probably same place)
- normalize ratings, prices, etc to 0-1
- parse the json tags column
- compute a bayesian quality score so 5-star POIs with 2 reviews
  dont beat 4.5-star ones with 500 reviews
- dampen popularity to give long-tail POIs a better shot
"""

import numpy as np
import pandas as pd
import json


class DataPreparer:
    def __init__(self, pop_dampening=0.7):
        # pop_dampening < 1 compresses the popularity distribution
        # so niche places get more visibility. 0.7 felt right after some testing
        self.pop_dampening = pop_dampening
        self._norm_stats = {}

    def run(self, df):
        df = df.copy()
        df = self._fill_missing(df)
        df = self._dedup(df)
        df = self._check_geo(df)
        df = self._normalize_nums(df)
        df = self._parse_tags(df)
        df = self._quality_score(df)
        df = self._dampen_pop(df)
        return df

    def _fill_missing(self, df):
        defaults = {
            "rating": df["rating"].median() if "rating" in df else 3.0,
            "review_count": 0, "popularity": 0.0, "price_level": 2,
            "expected_duration": 60, "opening_hours": "09:00-18:00",
            "reservation_required": False, "accessibility": "partially-accessible",
            "description": "", "tags": "[]",
        }
        for col, val in defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        return df

    def _dedup(self, df):
        # keep the one with more reviews (probably more reliable)
        df = df.sort_values("review_count", ascending=False)
        df = df.drop_duplicates(subset=["name", "neighborhood"], keep="first")
        return df.reset_index(drop=True)

    def _check_geo(self, df):
        # rough bounding box for paris
        ok_lat = df["latitude"].between(48.80, 48.95)
        ok_lon = df["longitude"].between(2.20, 2.50)
        df["geo_valid"] = ok_lat & ok_lon
        bad = (~df["geo_valid"]).sum()
        if bad:
            print(f"  warning: {bad} POIs outside paris bounds")
        return df

    def _normalize_nums(self, df):
        # log transform review count because the distribution is super skewed
        df["review_count_log"] = np.log1p(df["review_count"])

        for col, out in [("rating", "rating_norm"), ("review_count_log", "review_count_norm"),
                         ("price_level", "price_level_norm"), ("expected_duration", "duration_norm")]:
            lo, hi = df[col].min(), df[col].max()
            self._norm_stats[col] = (lo, hi)
            df[out] = (df[col] - lo) / (hi - lo) if hi > lo else 0.5
        return df

    def _parse_tags(self, df):
        df["tags_list"] = df["tags"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        
        # pull out some useful binary features from tags
        df["is_touristy"] = df["tags_list"].apply(lambda t: int("touristy" in t))
        df["is_local_favorite"] = df["tags_list"].apply(
            lambda t: int(any(x in t for x in ["local-favorite", "hidden-gem", "off-beaten-path"])))
        df["is_family_friendly"] = df["tags_list"].apply(
            lambda t: int(any(x in t for x in ["family-friendly", "kid-friendly", "stroller-friendly"])))
        df["is_accessible"] = df["tags_list"].apply(lambda t: int("wheelchair-accessible" in t))
        df["is_free"] = df["tags_list"].apply(lambda t: int("free-entry" in t))
        df["is_outdoor"] = df["tags_list"].apply(lambda t: int("outdoor-only" in t))
        return df

    def _quality_score(self, df):
        """
        bayesian average rating.
        prevents low-review POIs with perfect ratings from floating to the top.
        C = global mean, m = 25th percentile review count
        """
        C = df["rating"].mean()
        m = df["review_count"].quantile(0.25)
        df["quality_score"] = (df["review_count"] * df["rating"] + m * C) / (df["review_count"] + m)
        lo, hi = df["quality_score"].min(), df["quality_score"].max()
        df["quality_score"] = (df["quality_score"] - lo) / (hi - lo) if hi > lo else 0.5
        return df

    def _dampen_pop(self, df):
        # pop^0.7 compresses the range:
        #   pop=1.0 -> 1.0,  pop=0.5 -> 0.62,  pop=0.1 -> 0.20
        # gives long-tail POIs more breathing room in the rankings
        df["popularity_dampened"] = np.power(df["popularity"].clip(lower=1e-6), self.pop_dampening)
        return df
