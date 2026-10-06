"""
generates human-readable explanations for why a POI was recommended.

pretty straightforward rule-based approach - looks at which scoring
factors contributed most and translates them to plain english.

in production you'd probably want SHAP values or something more
principled, but for a prototype this gets the job done.
"""

import json


class Explainer:
    def explain(self, result, profile, poi_row):
        reasons = []
        name = result.get("name", poi_row.get("name", "?"))

        # preference match
        if result["preference_weight"] > 0.5:
            cat = result.get("category", poi_row.get("category", ""))
            matching = self._matching_interests(cat)
            if matching:
                reasons.append(f"Strong match with your interest in {', '.join(matching)}")
            else:
                reasons.append("Matches your preference profile")

        # behavioral similarity
        itags = profile.get("implicit_tag_preferences", {})
        ptags = poi_row.get("tags_list", [])
        if isinstance(ptags, list):
            hits = [t for t in ptags if t in itags]
            if hits:
                reasons.append(f"Similar to previously enjoyed experiences ({', '.join(hits[:3])})")

        # budget
        if result["budget_compat"] > 0.8:
            reasons.append("Within your budget range")
        elif result["budget_compat"] < 0.4:
            reasons.append("Note: may be above your typical budget")

        # mobility
        if result["mobility_compat"] > 0.8:
            mob = profile["trip_context"]["mobility"]
            labels = {"walking": "walking distance", "public_transport": "public transport",
                      "car": "car", "bicycle": "bicycle"}
            reasons.append(f"Easily accessible by {labels.get(mob, mob)}")

        # party type
        if result["party_compat"] > 0.8:
            if profile["party_type"] == "family":
                reasons.append("Family and child-friendly")
            elif profile["party_type"] == "couple":
                reasons.append("Great for couples")

        # local vs touristy
        flags = profile.get("preference_flags", {})
        if flags.get("avoid_touristy") and poi_row.get("is_local_favorite", 0):
            reasons.append("Local favorite — less touristy atmosphere")
        elif flags.get("prefer_famous") and poi_row.get("is_touristy", 0):
            reasons.append("Well-known attraction popular with visitors")

        # quality
        if poi_row.get("quality_score", 0) > 0.7:
            r = poi_row.get("rating", 0)
            reasons.append(f"Highly rated ({r:.1f}/5.0)")

        # misc
        if poi_row.get("is_free", 0):
            reasons.append("Free entry")
        if poi_row.get("is_outdoor", 0):
            reasons.append("Outdoor experience")

        if result["confidence"] < 0.5:
            reasons.append("Note: confidence is moderate — worth checking reviews first")

        return {
            "poi_name": name, "poi_id": result["poi_id"],
            "category": result.get("category", ""),
            "overall_score": result["score"],
            "reasons": reasons or ["Matches your overall travel profile"],
            "confidence": result["confidence"],
        }

    def _matching_interests(self, category):
        """figure out which interests map to this category"""
        from .traveler_representation import INTEREST_TO_CAT
        return [interest for interest, cats in INTEREST_TO_CAT.items() if category in cats][:3]

    def format(self, expl):
        lines = [
            f"POI: {expl['poi_name']} ({expl['category']})",
            f"Score: {expl['overall_score']:.2f} | Confidence: {expl['confidence']:.2f}",
            "Reasons:",
        ]
        for r in expl["reasons"]:
            lines.append(f"  - {r}")
        return "\n".join(lines)
