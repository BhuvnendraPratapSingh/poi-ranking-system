"""
narrows down the full POI catalog to a reasonable candidate set
before we do the expensive per-item ranking.

the key trick here is the diverse candidate pool:
  - 60% from interest matching (the obvious ones)
  - 20% from popularity (stuff people expect to see)
  - 20% from diversity sampling (prevents us from killing off niche POIs)

without that diversity slice, the candidate stage would just
return the same popular stuff for everyone, which defeats the
whole point of personalization.
"""

import numpy as np
import pandas as pd
import json


class CandidateGenerator:
    def __init__(self, max_candidates=100, budget_tolerance=1):
        self.max_cand = max_candidates
        self.budget_tol = budget_tolerance
        # how much of the pool comes from each source
        self.interest_pct = 0.60
        self.pop_pct = 0.20
        self.diverse_pct = 0.20

    def generate(self, profile, poi_df, seed=None):
        rng = np.random.default_rng(seed)
        df = poi_df.copy()

        # hard filters first -- get rid of stuff that clearly won't work
        df = self._hard_filter(df, profile)
        if len(df) == 0:
            return pd.DataFrame(), {}

        # score everything by interest match
        df["_interest_score"] = df.apply(
            lambda r: self._interest_score(r, profile), axis=1)

        # build the candidate pool from three sources
        n_int = int(self.max_cand * self.interest_pct)
        n_pop = int(self.max_cand * self.pop_pct)
        n_div = self.max_cand - n_int - n_pop

        picked = set()

        # top by interest
        by_interest = df.nlargest(n_int * 2, "_interest_score")
        picked.update(by_interest.head(n_int)["poi_id"].tolist())

        # top by popularity
        pop_col = "popularity_dampened" if "popularity_dampened" in df.columns else "popularity"
        by_pop = df.nlargest(n_pop * 2, pop_col)
        remaining_pop = by_pop[~by_pop["poi_id"].isin(picked)]
        picked.update(remaining_pop.head(n_pop)["poi_id"].tolist())

        # diverse sample from whatever's left
        leftover = df[~df["poi_id"].isin(picked)]
        if len(leftover) > 0 and n_div > 0:
            diverse_ids = self._diversity_sample(leftover, profile, n_div, rng)
            picked.update(diverse_ids)

        result = df[df["poi_id"].isin(picked)].copy()
        scores = {r["poi_id"]: r["_interest_score"] for _, r in result.iterrows()}
        return result, scores

    def _hard_filter(self, df, profile):
        """remove stuff that definitely won't work"""
        budget_map = {"low": 1, "medium": 2, "high": 3}
        budget_num = budget_map.get(profile["trip_context"]["budget"], 2)
        max_price = min(budget_num + self.budget_tol, 4)
        df = df[df["price_level"] <= max_price]

        # no nightclubs for families
        if profile["party_type"] == "family":
            df = df[df["category"] != "Nightlife"]

        return df

    def _interest_score(self, poi_row, profile):
        from .traveler_representation import ALL_CATS
        score = 0.0

        cat = poi_row["category"]
        if cat in ALL_CATS:
            idx = ALL_CATS.index(cat)
            score += profile["combined_category_affinity"][idx] * 5.0

        # tag overlap with implicit preferences
        itags = profile.get("implicit_tag_preferences", {})
        ptags = poi_row.get("tags_list", [])
        if isinstance(ptags, list) and itags:
            for t in ptags:
                if t in itags:
                    score += itags[t] * 0.5

        # preference flag bonuses/penalties
        flags = profile.get("preference_flags", {})
        if flags.get("avoid_touristy") and poi_row.get("is_local_favorite", 0):
            score += 0.5
        if flags.get("avoid_touristy") and poi_row.get("is_touristy", 0):
            score -= 0.3
        if flags.get("prefer_famous") and poi_row.get("is_touristy", 0):
            score += 0.3
        if flags.get("child_friendly") and poi_row.get("is_family_friendly", 0):
            score += 0.5

        score += poi_row.get("quality_score", 0) * 0.3
        return max(score, 0.0)

    def _diversity_sample(self, leftover, profile, n, rng):
        """
        sample from underrepresented categories.
        prefer higher quality within each category so we
        don't surface genuinely bad POIs.
        """
        from .traveler_representation import ALL_CATS
        aff = profile["combined_category_affinity"]

        sampled = []
        per_cat = max(1, n // len(ALL_CATS))

        for i, cat in enumerate(ALL_CATS):
            cat_df = leftover[leftover["category"] == cat]
            if len(cat_df) == 0:
                continue

            w = cat_df["quality_score"].values + 0.1
            w = w / w.sum()
            k = min(max(1, int(per_cat * (1 + aff[i] * 3))), len(cat_df))
            picks = rng.choice(cat_df["poi_id"].values, size=k, replace=False, p=w)
            sampled.extend(picks)

        if len(sampled) > n:
            sampled = list(rng.choice(sampled, size=n, replace=False))
        return sampled
