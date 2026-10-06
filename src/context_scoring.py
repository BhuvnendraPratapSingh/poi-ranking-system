"""
context-aware scoring.

the ranking model tells us how much a traveler *wants* a POI.
this module checks whether it actually *makes sense* for their trip.

example: a museum might score 0.95 on preference but:
  - budget compatibility: 0.20 (too expensive)
  - mobility: 0.30 (hard to reach on foot)
  - availability: 0.00 (closed on mondays)

so we need to combine preference with practical reality.

final utility formula:
  utility = 0.35 * preference + 0.15 * behavioral + 0.20 * context
          + 0.10 * geographic + 0.20 * practical

the weights were set based on intuition about what matters most.
in production you'd probably tune these with A/B tests.
"""

import numpy as np


class ContextScorer:
    def __init__(self, w_pref=0.35, w_behav=0.15, w_ctx=0.20, w_geo=0.10, w_pract=0.20):
        total = w_pref + w_behav + w_ctx + w_geo + w_pract
        self.w_pref = w_pref / total
        self.w_behav = w_behav / total
        self.w_ctx = w_ctx / total
        self.w_geo = w_geo / total
        self.w_pract = w_pract / total

    def score(self, candidates, profile, pref_scores, behav_scores=None):
        results = []
        for _, poi in candidates.iterrows():
            pid = poi["poi_id"]

            pref = pref_scores.get(pid, 0.0)
            behav = behav_scores.get(pid, 0.0) if behav_scores else 0.0

            bgt = self._budget_score(poi, profile)
            mob = self._mobility_score(poi, profile)
            avail = self._availability_score(poi)
            party = self._party_score(poi, profile)

            ctx = 0.30 * bgt + 0.25 * mob + 0.25 * avail + 0.20 * party
            geo = self._geo_score(poi)

            # geometric mean so a zero in any dimension really hurts
            pract = float(np.power(np.prod([bgt, mob, avail, party]), 0.25))

            utility = (self.w_pref * pref + self.w_behav * behav +
                       self.w_ctx * ctx + self.w_geo * geo + self.w_pract * pract)

            # confidence: high when all signals agree, low when they disagree
            vals = [pref, ctx, pract]
            conf = 1.0 - np.std(vals) / max(np.mean(vals), 0.01)
            conf = float(np.clip(conf, 0.0, 1.0))

            results.append({
                "poi_id": pid, "name": poi["name"],
                "category": poi["category"],
                "subcategory": poi.get("subcategory", ""),
                "score": round(utility, 4),
                "preference_weight": round(float(pref), 4),
                "context_compatibility": round(float(ctx), 4),
                "practical_compatibility": round(float(pract), 4),
                "geographic_score": round(float(geo), 4),
                "confidence": round(conf, 4),
                "budget_compat": round(float(bgt), 4),
                "mobility_compat": round(float(mob), 4),
                "availability": round(float(avail), 4),
                "party_compat": round(float(party), 4),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _budget_score(self, poi, profile):
        budget = profile["trip_context"]["budget"]
        price = poi.get("price_level", 2)
        ceiling = {"low": 1.5, "medium": 2.5, "high": 4.0}.get(budget, 2.5)
        if price <= ceiling:
            return 1.0
        return max(0.0, 1.0 - (price - ceiling) / 4.0 * 1.5)

    def _mobility_score(self, poi, profile):
        mob = profile["trip_context"]["mobility"]
        acc = poi.get("accessibility", "partially-accessible")
        dist = poi.get("dist_from_center_norm", 0.5)

        base = 0.8
        if mob == "walking":
            base = 1.0 if acc == "fully-accessible" else (0.4 if acc == "limited-access" else 0.8)
            base *= max(0.3, 1.0 - dist * 0.8)
        elif mob == "public_transport":
            base = 0.6 if acc == "limited-access" else 0.9
        elif mob == "car":
            base = 0.95
        elif mob == "bicycle":
            base = 0.9 if acc == "fully-accessible" else 0.6
            base *= max(0.4, 1.0 - dist * 0.5)
        return base

    def _availability_score(self, poi):
        hours = poi.get("opening_hours", "09:00-18:00")
        try:
            parts = hours.split("-")
            oh = int(parts[0].split(":")[0])
            ch = int(parts[1].split(":")[0])
        except (ValueError, IndexError):
            return 0.7
        
        hrs = (24 - oh + ch) if ch < oh else (ch - oh)
        score = min(hrs / 12.0, 1.0)
        if poi.get("reservation_required", False):
            score *= 0.85  # reservation adds friction
        return score

    def _party_score(self, poi, profile):
        party = profile["party_type"]
        fam_friendly = poi.get("is_family_friendly", 0)
        cat = poi.get("category", "")

        if party == "family":
            if fam_friendly: return 1.0
            if cat == "Nightlife": return 0.1
            if cat in ("Restaurant", "Nature/Outdoor", "Activity"): return 0.7
            return 0.5
        elif party == "couple":
            return 0.9 if cat in ("Restaurant", "Nightlife", "Nature/Outdoor") else 0.7
        elif party == "business":
            if cat in ("Restaurant", "Museum", "Historical Site"): return 0.9
            if cat in ("Family", "Activity"): return 0.4
            return 0.6
        return 0.8  # solo or friends - most things work

    def _geo_score(self, poi):
        dist = poi.get("dist_from_center_norm", 0.5)
        return max(0.2, 1.0 - dist * 0.6)
