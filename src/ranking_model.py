"""
personalized ranking using gradient boosted trees.

why GBT instead of something fancier:
- works great on tabular data with mixed feature types
- handles interactions between features naturally (budget x category etc)
- fast inference (important if this were real-time)
- feature importances give us some transparency
- neural nets would be overkill for this dataset size

i'm using a pointwise approach (predict relevance per item) rather than
pairwise/listwise because it's simpler to set up and debug, and the
context scoring layer downstream handles the list-level adjustments anyway.

training target: interaction_weight normalized to [0, 1]
  visit=0.875, booking=1.0, save=0.5, view=0.125, dismiss=0.0
negative samples: POIs the traveler didn't interact with -> target 0
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import json


class Ranker:
    def __init__(self, n_trees=200, max_depth=5, lr=0.1, neg_ratio=2.0, seed=42):
        self.model = GradientBoostingRegressor(
            n_estimators=n_trees, max_depth=max_depth,
            learning_rate=lr, subsample=0.8, random_state=seed)
        self.neg_ratio = neg_ratio
        self.seed = seed
        self.feat_names = []
        self.fitted = False

    def build_training_data(self, interactions, pois, travelers, trav_builder,
                            poi_feats, poi_feat_names):
        """construct (traveler_features, poi_features, cross_features) -> relevance pairs"""
        rng = np.random.default_rng(self.seed)

        poi_idx_map = {pid: i for i, pid in enumerate(pois["poi_id"].values)}
        all_pois = set(pois["poi_id"].values)

        # build profiles for everyone
        profiles = {}
        for _, tr in travelers.iterrows():
            profiles[tr["traveler_id"]] = trav_builder.build(tr, interactions, pois)

        Xs, ys = [], []
        grouped = interactions.groupby("traveler_id")

        for tid, grp in grouped:
            if tid not in profiles:
                continue
            prof = profiles[tid]
            tvec, tnames = trav_builder.to_feature_vector(prof)

            seen = set()
            for _, row in grp.iterrows():
                pid = row["poi_id"]
                if pid not in poi_idx_map:
                    continue

                pvec = poi_feats[poi_idx_map[pid]]
                cvec, cnames = self._cross_features(prof, pois.iloc[poi_idx_map[pid]])

                Xs.append(np.concatenate([tvec, pvec, cvec]))
                ys.append(max(row["interaction_weight"], 0.0) / 8.0)
                seen.add(pid)

            # sample negatives
            unseen = list(all_pois - seen)
            n_neg = min(int(len(seen) * self.neg_ratio), len(unseen))
            if n_neg > 0:
                neg_ids = rng.choice(unseen, size=n_neg, replace=False)
                for nid in neg_ids:
                    if nid not in poi_idx_map:
                        continue
                    pvec = poi_feats[poi_idx_map[nid]]
                    cvec, cnames = self._cross_features(prof, pois.iloc[poi_idx_map[nid]])
                    Xs.append(np.concatenate([tvec, pvec, cvec]))
                    ys.append(0.0)

        X = np.array(Xs)
        y = np.array(ys)

        # figure out feature names
        _, tnames = trav_builder.to_feature_vector(list(profiles.values())[0])
        _, cnames = self._cross_features(list(profiles.values())[0], pois.iloc[0])
        self.feat_names = tnames + [f"poi_{n}" for n in poi_feat_names] + cnames

        return X, y, self.feat_names

    def train(self, X, y):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2,
                                                      random_state=self.seed)
        self.model.fit(X_tr, y_tr)
        self.fitted = True

        tr_r2 = self.model.score(X_tr, y_tr)
        val_r2 = self.model.score(X_val, y_val)
        print(f"  ranking model: train R²={tr_r2:.4f}, val R²={val_r2:.4f}")
        return {"train_r2": tr_r2, "val_r2": val_r2}

    def predict(self, profile, candidates, poi_feats, poi_idx_map, trav_builder):
        """score candidate POIs for a specific traveler"""
        assert self.fitted, "model not trained yet"
        tvec, _ = trav_builder.to_feature_vector(profile)

        rows, pids = [], []
        for _, poi in candidates.iterrows():
            pid = poi["poi_id"]
            if pid not in poi_idx_map:
                continue
            pvec = poi_feats[poi_idx_map[pid]]
            cvec, _ = self._cross_features(profile, poi)
            rows.append(np.concatenate([tvec, pvec, cvec]))
            pids.append(pid)

        if not rows:
            return {}

        preds = self.model.predict(np.array(rows))
        preds = np.clip(preds, 0.0, 1.0)
        return dict(zip(pids, preds.tolist()))

    def top_features(self, n=20):
        if not self.fitted:
            return {}
        imp = self.model.feature_importances_
        idx = np.argsort(imp)[::-1][:n]
        return {self.feat_names[i]: float(imp[i]) for i in idx if i < len(self.feat_names)}

    def _cross_features(self, profile, poi_row):
        """
        interaction features between traveler and POI.
        this is where the personalization magic happens --
        same POI gets different scores for different travelers.
        """
        from .traveler_representation import ALL_CATS
        feats, names = [], []

        # how much does the traveler like this category?
        cat = poi_row["category"]
        cat_aff = 0.0
        if cat in ALL_CATS:
            cat_aff = profile["combined_category_affinity"][ALL_CATS.index(cat)]
        feats.append(cat_aff)
        names.append("cross_cat_aff")

        # budget match
        budget_compat = 1.0 - abs(profile["budget_level"] - poi_row.get("price_level_norm", 0.5))
        feats.append(budget_compat)
        names.append("cross_budget")

        # touristy alignment
        avoid = profile["preference_flags"].get("avoid_touristy", False)
        is_tour = poi_row.get("is_touristy", 0)
        is_local = poi_row.get("is_local_favorite", 0)
        feats.append((is_local - is_tour) if avoid else 0.0)
        names.append("cross_touristy")

        # family match
        is_fam = profile["party_type"] == "family"
        feats.append(float(poi_row.get("is_family_friendly", 0)) if is_fam else 0.5)
        names.append("cross_family")

        # quality * affinity -- high quality in a liked category is extra good
        q = poi_row.get("quality_score", 0.5)
        feats.append(q * cat_aff)
        names.append("cross_q_aff")

        return np.array(feats), names
