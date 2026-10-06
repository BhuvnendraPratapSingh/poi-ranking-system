"""
builds a profile for each traveler.

two types of preference signals:
  1. explicit - what they told us ("i like local food")
  2. implicit - what they actually do (keeps visiting small restaurants, saves neighborhood stuff)

when we have interaction history, we blend both signals.
when we don't (cold start), we just use whatever they told us.
"""

import numpy as np
import pandas as pd
import json

# maps user interests -> POI categories
INTEREST_TO_CAT = {
    "local food": ["Restaurant"], "fine dining": ["Restaurant"],
    "street food": ["Restaurant"], "wine": ["Restaurant", "Nightlife"],
    "cooking": ["Activity", "Restaurant"],
    "history": ["Historical Site", "Museum"], "architecture": ["Historical Site"],
    "museums": ["Museum"], "art": ["Museum", "Activity"],
    "contemporary art": ["Museum"],
    "nature": ["Nature/Outdoor"], "parks": ["Nature/Outdoor", "Family"],
    "walking": ["Nature/Outdoor", "Activity", "Neighborhood"],
    "cycling": ["Activity", "Nature/Outdoor"],
    "neighborhoods": ["Neighborhood"],
    "local experiences": ["Neighborhood", "Activity", "Event"],
    "shopping": ["Shopping"], "markets": ["Shopping"],
    "nightlife": ["Nightlife"], "music": ["Nightlife", "Event"],
    "theater": ["Nightlife", "Event"],
    "photography": ["Nature/Outdoor", "Historical Site", "Neighborhood"],
    "family activities": ["Family"], "festivals": ["Event"],
    "sports": ["Activity"], "wellness": ["Activity"],
}

ALL_CATS = ["Restaurant", "Museum", "Historical Site", "Nature/Outdoor",
            "Shopping", "Activity", "Neighborhood", "Nightlife", "Family", "Event"]


class TravelerProfileBuilder:
    def __init__(self, explicit_w=0.6, implicit_w=0.4):
        self.explicit_w = explicit_w
        self.implicit_w = implicit_w

    def build(self, trav_row, interaction_df=None, poi_df=None):
        profile = {}

        # explicit preferences -> category affinity vector
        expl = self._explicit_affinity(trav_row)
        profile["explicit_category_affinity"] = expl

        # trip context
        profile["trip_context"] = {
            "destination": trav_row.get("destination", ""),
            "trip_duration": trav_row.get("trip_duration", 3),
            "travel_start": trav_row.get("travel_start", ""),
            "travel_end": trav_row.get("travel_end", ""),
            "party_type": trav_row.get("party_type", "solo"),
            "mobility": trav_row.get("mobility", "walking"),
            "budget": trav_row.get("budget", "medium"),
        }

        # implicit preferences from behavior
        impl = np.zeros(len(ALL_CATS))
        impl_tags = {}
        has_history = False

        if interaction_df is not None and poi_df is not None:
            my_ints = interaction_df[interaction_df["traveler_id"] == trav_row["traveler_id"]]
            if len(my_ints) > 0:
                has_history = True
                impl, impl_tags = self._implicit_affinity(my_ints, poi_df)

        profile["implicit_category_affinity"] = impl
        profile["implicit_tag_preferences"] = impl_tags

        # blend explicit + implicit
        if has_history:
            combined = self.explicit_w * expl + self.implicit_w * impl
        else:
            combined = expl  # cold start - just use what they told us

        if combined.sum() > 0:
            combined = combined / combined.sum()
        profile["combined_category_affinity"] = combined

        # other stuff
        profile["budget_level"] = {"low": 0.0, "medium": 0.5, "high": 1.0}.get(
            trav_row["budget"], 0.5)
        profile["mobility_type"] = trav_row["mobility"]
        profile["party_type"] = trav_row["party_type"]
        profile["preference_flags"] = self._parse_pref_flags(
            trav_row.get("explicit_preferences", ""))

        return profile

    def _explicit_affinity(self, trav_row):
        """convert stated interests into a category affinity vector"""
        interests = json.loads(trav_row["interests"])
        aff = np.zeros(len(ALL_CATS))
        for interest in interests:
            for cat in INTEREST_TO_CAT.get(interest.lower(), []):
                if cat in ALL_CATS:
                    aff[ALL_CATS.index(cat)] += 1.0
        if aff.sum() > 0:
            aff /= aff.sum()
        return aff

    def _implicit_affinity(self, my_interactions, poi_df):
        """figure out what the user actually likes from their behavior"""
        merged = my_interactions.merge(
            poi_df[["poi_id", "category", "tags_list"]], on="poi_id", how="left")

        aff = np.zeros(len(ALL_CATS))
        for _, row in merged.iterrows():
            cat = row.get("category")
            w = row.get("interaction_weight", 1.0)
            if cat in ALL_CATS:
                aff[ALL_CATS.index(cat)] += w

        aff = aff - aff.min()  # shift so negatives become zero
        if aff.sum() > 0:
            aff /= aff.sum()

        # also figure out which tags they gravitate toward
        tag_counts = {}
        pos = merged[merged["interaction_weight"] > 0]
        for _, row in pos.iterrows():
            tags = row.get("tags_list", [])
            if isinstance(tags, list):
                w = row.get("interaction_weight", 1.0)
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + w

        if tag_counts:
            mx = max(tag_counts.values())
            tag_counts = {k: v / mx for k, v in tag_counts.items()}

        return aff, tag_counts

    def _parse_pref_flags(self, text):
        """extract actionable flags from the free-text preferences"""
        if not isinstance(text, str):
            return {}
        t = text.lower()
        return {
            "avoid_touristy": any(w in t for w in ["less touristy", "avoid crowd", "off-beaten", "authentic", "local"]),
            "prefer_famous": any(w in t for w in ["famous", "landmark", "well-known", "must-see", "iconic"]),
            "child_friendly": any(w in t for w in ["child", "kid", "family", "interactive"]),
            "prefer_walkable": "walkable" in t or "walking distance" in t,
            "prefer_morning": "morning" in t,
            "avoid_crowded": any(w in t for w in ["avoid crowd", "less crowded", "quiet"]),
        }

    def to_feature_vector(self, profile):
        """flatten a profile into a numeric vector for the model"""
        feats = []
        names = []

        # category affinity (10)
        feats.extend(profile["combined_category_affinity"])
        names += [f"trav_aff_{c}" for c in ALL_CATS]

        # budget (1)
        feats.append(profile["budget_level"])
        names.append("trav_budget")

        # trip duration normalized (1)
        dur = min(profile["trip_context"]["trip_duration"] / 14.0, 1.0)
        feats.append(dur)
        names.append("trav_dur_norm")

        # party type one-hot (5)
        for pt in ["solo", "couple", "family", "friends", "business"]:
            feats.append(1.0 if profile["party_type"] == pt else 0.0)
            names.append(f"trav_party_{pt}")

        # mobility one-hot (4)
        for mt in ["walking", "public_transport", "car", "bicycle"]:
            feats.append(1.0 if profile["mobility_type"] == mt else 0.0)
            names.append(f"trav_mob_{mt}")

        # preference flags (5)
        for fk in ["avoid_touristy", "prefer_famous", "child_friendly",
                    "prefer_walkable", "avoid_crowded"]:
            feats.append(1.0 if profile["preference_flags"].get(fk) else 0.0)
            names.append(f"trav_fl_{fk}")

        return np.array(feats), names
