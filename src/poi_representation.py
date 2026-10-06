"""
builds feature vectors for each POI.

combines a bunch of different signal types:
- one-hot category/subcategory
- numerical stuff (rating, price, quality, popularity)
- geographic (lat/lon, distance from center)
- binary tag features (touristy, local, family-friendly etc)
- tfidf on description text
- behavioral aggregates from interaction history

the idea is that the representation should be rich enough to tell apart
a "huge tourist trap" from a "tiny local gem" even if they're in the
same category. that distinction is what makes personalization possible.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


class POIFeatureBuilder:
    def __init__(self, n_tfidf=50):
        self.n_tfidf = n_tfidf
        self._cat_enc = None
        self._tfidf = None
        self.feature_names = []

    def fit_transform(self, poi_df, interaction_df=None):
        """build features for all POIs. returns (feature_matrix, feature_names)."""
        parts = []
        names = []

        # categories (one-hot)
        self._cat_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cat_feats = self._cat_enc.fit_transform(poi_df[["category", "subcategory"]].values)
        cat_names = [f"cat_{c}" for c in self._cat_enc.get_feature_names_out()]
        parts.append(cat_feats)
        names += cat_names

        # numerical features
        num_cols = ["rating_norm", "review_count_norm", "price_level_norm",
                    "duration_norm", "quality_score", "popularity_dampened"]
        avail = [c for c in num_cols if c in poi_df.columns]
        parts.append(poi_df[avail].values.astype(float))
        names += avail

        # geographic
        geo, geo_n = self._geo_features(poi_df)
        parts.append(geo)
        names += geo_n

        # tag binary features
        tag_cols = ["is_touristy", "is_local_favorite", "is_family_friendly",
                    "is_accessible", "is_free", "is_outdoor"]
        avail_tags = [c for c in tag_cols if c in poi_df.columns]
        parts.append(poi_df[avail_tags].values.astype(float))
        names += avail_tags

        # tfidf on descriptions
        self._tfidf = TfidfVectorizer(max_features=self.n_tfidf, stop_words="english",
                                      ngram_range=(1, 2))
        txt = self._tfidf.fit_transform(poi_df["description"].fillna("")).toarray()
        parts.append(txt)
        names += [f"tfidf_{w}" for w in self._tfidf.get_feature_names_out()]

        # behavioral features (if we have interactions)
        if interaction_df is not None:
            beh, beh_n = self._behavioral_features(poi_df, interaction_df)
            parts.append(beh)
            names += beh_n

        self.feature_names = names
        return np.hstack(parts), names

    def transform(self, poi_df, interaction_df=None):
        """use fitted encoders on new data"""
        parts = []
        
        cat_feats = self._cat_enc.transform(poi_df[["category", "subcategory"]].values)
        if hasattr(cat_feats, "toarray"):
            cat_feats = cat_feats.toarray()
        parts.append(cat_feats)

        num_cols = ["rating_norm", "review_count_norm", "price_level_norm",
                    "duration_norm", "quality_score", "popularity_dampened"]
        avail = [c for c in num_cols if c in poi_df.columns]
        parts.append(poi_df[avail].values.astype(float))

        geo, _ = self._geo_features(poi_df)
        parts.append(geo)

        tag_cols = ["is_touristy", "is_local_favorite", "is_family_friendly",
                    "is_accessible", "is_free", "is_outdoor"]
        avail_tags = [c for c in tag_cols if c in poi_df.columns]
        parts.append(poi_df[avail_tags].values.astype(float))

        txt = self._tfidf.transform(poi_df["description"].fillna("")).toarray()
        parts.append(txt)

        if interaction_df is not None:
            beh, _ = self._behavioral_features(poi_df, interaction_df)
            parts.append(beh)

        return np.hstack(parts)

    def _geo_features(self, df):
        # distance from paris center + normalized coords
        clat, clon = 48.8566, 2.3522
        dist = np.sqrt((df["latitude"] - clat)**2 + (df["longitude"] - clon)**2)
        mx = dist.max()
        dist_norm = dist / mx if mx > 0 else dist * 0

        lat_lo, lat_hi = df["latitude"].min(), df["latitude"].max()
        lon_lo, lon_hi = df["longitude"].min(), df["longitude"].max()
        lat_n = (df["latitude"] - lat_lo) / max(lat_hi - lat_lo, 1e-6)
        lon_n = (df["longitude"] - lon_lo) / max(lon_hi - lon_lo, 1e-6)

        feats = np.column_stack([lat_n.values, lon_n.values, dist_norm.values])
        return feats, ["geo_lat_norm", "geo_lon_norm", "geo_dist_center"]

    def _behavioral_features(self, poi_df, idf):
        """aggregate interaction stats per POI (visit rate, save rate, etc)"""
        stats = idf.groupby("poi_id").agg(
            n_interactions=("interaction_type", "count"),
            avg_w=("interaction_weight", "mean"),
            visits=("interaction_type", lambda x: (x == "visit").sum()),
            saves=("interaction_type", lambda x: (x == "save").sum()),
            dismisses=("interaction_type", lambda x: (x == "dismiss").sum()),
            bookings=("interaction_type", lambda x: (x == "booking").sum()),
            uniq_users=("traveler_id", "nunique"),
        ).reset_index()

        merged = pd.DataFrame({"poi_id": poi_df["poi_id"].values}).merge(stats, on="poi_id", how="left")
        merged = merged.fillna(0)

        total = merged["n_interactions"].clip(lower=1)
        merged["visit_rate"] = merged["visits"] / total
        merged["save_rate"] = merged["saves"] / total
        merged["dismiss_rate"] = merged["dismisses"] / total
        merged["booking_rate"] = merged["bookings"] / total

        for c in ["n_interactions", "avg_w", "uniq_users"]:
            mx = merged[c].max()
            merged[c + "_n"] = merged[c] / mx if mx > 0 else 0.0

        cols = ["n_interactions_n", "avg_w_n", "uniq_users_n",
                "visit_rate", "save_rate", "dismiss_rate", "booking_rate"]
        return merged[cols].values.astype(float), [f"behav_{c}" for c in cols]
