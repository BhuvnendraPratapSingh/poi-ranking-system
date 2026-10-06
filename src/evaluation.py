"""
evaluation metrics.

standard ranking stuff:
- precision@K, recall@K, NDCG@K

plus some beyond-accuracy metrics that are arguably more important
for a real recommendation system:
- personalization (are different users getting different results?)
- coverage (how much of the catalog are we actually recommending?)
- long-tail discovery (do niche POIs show up?)
- diversity (category variety in the results)
- constraint compatibility (are recommendations actually feasible?)
- calibration (do our scores mean anything?)
"""

import numpy as np
from collections import Counter


class Evaluator:
    def __init__(self, ks=None):
        self.ks = ks or [5, 10, 20]

    # --- core ranking metrics ---

    def ranking_metrics(self, ranked, ground_truth, threshold=0.3):
        metrics = {}
        ranked_ids = [r["poi_id"] for r in ranked]
        relevant = {pid for pid, s in ground_truth.items() if s >= threshold}

        for k in self.ks:
            topk = ranked_ids[:k]
            hits = sum(1 for p in topk if p in relevant)

            metrics[f"precision@{k}"] = hits / k if k else 0
            metrics[f"recall@{k}"] = hits / len(relevant) if relevant else 0
            metrics[f"ndcg@{k}"] = self._ndcg(ranked_ids[:k], ground_truth, k)
        return metrics

    def _ndcg(self, ranked_ids, gt, k):
        dcg = sum(gt.get(pid, 0) / np.log2(i + 2) for i, pid in enumerate(ranked_ids[:k]))
        ideal = sorted(gt.values(), reverse=True)[:k]
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    # --- beyond-accuracy metrics ---

    def personalization(self, all_rankings):
        """pairwise jaccard distance between top-K lists. higher = more personalized."""
        out = {}
        for k in self.ks:
            sets = [set(r["poi_id"] for r in ranking[:k]) for ranking in all_rankings]
            dists = []
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    inter = len(sets[i] & sets[j])
                    union = len(sets[i] | sets[j])
                    if union > 0:
                        dists.append(1.0 - inter / union)
            out[f"personalization@{k}"] = round(np.mean(dists), 4) if dists else 0
        return out

    def coverage(self, all_rankings, total_pois, k=20):
        all_rec = set()
        for ranking in all_rankings:
            for r in ranking[:k]:
                all_rec.add(r["poi_id"])
        return {
            "catalog_coverage": round(len(all_rec) / total_pois, 4) if total_pois else 0,
            "unique_pois_recommended": len(all_rec),
            "total_catalog_size": total_pois,
        }

    def long_tail(self, all_rankings, poi_df, pop_threshold=0.2, k=20):
        lt_pois = set(poi_df[poi_df["popularity"] < pop_threshold]["poi_id"])
        ratios = []
        for ranking in all_rankings:
            topk = [r["poi_id"] for r in ranking[:k]]
            ratios.append(sum(1 for p in topk if p in lt_pois) / k if k else 0)
        return {
            "long_tail_ratio_mean": round(np.mean(ratios), 4),
            "long_tail_ratio_std": round(np.std(ratios), 4),
            "long_tail_pois_in_catalog": len(lt_pois),
        }

    def diversity(self, all_rankings, k=20):
        """normalized category entropy. 1.0 = perfectly uniform."""
        entropies = []
        for ranking in all_rankings:
            cats = [r["category"] for r in ranking[:k]]
            if not cats:
                continue
            counts = Counter(cats)
            total = len(cats)
            probs = [c / total for c in counts.values()]
            ent = -sum(p * np.log2(p) for p in probs if p > 0)
            max_ent = np.log2(len(counts)) if len(counts) > 1 else 1.0
            entropies.append(ent / max_ent if max_ent > 0 else 0)
        return {
            "diversity_mean": round(np.mean(entropies), 4) if entropies else 0,
            "diversity_std": round(np.std(entropies), 4) if entropies else 0,
        }

    def constraint_compat(self, all_rankings, k=20):
        scores = [r.get("practical_compatibility", 0)
                  for ranking in all_rankings for r in ranking[:k]]
        return {
            "practical_compat_mean": round(np.mean(scores), 4) if scores else 0,
            "practical_compat_std": round(np.std(scores), 4) if scores else 0,
            "high_compat_ratio": round(np.mean([1 if s > 0.7 else 0 for s in scores]), 4) if scores else 0,
        }

    def calibration(self, all_rankings, all_gts, k=20):
        """spearman correlation between predicted scores and actual relevance"""
        from scipy.stats import spearmanr

        pred, true = [], []
        for ranking, gt in zip(all_rankings, all_gts):
            for r in ranking[:k]:
                pred.append(r["score"])
                true.append(gt.get(r["poi_id"], 0.0))

        if len(pred) < 3:
            return {"calibration_spearman": 0.0, "calibration_p_value": 1.0}

        corr, pval = spearmanr(pred, true)
        return {
            "calibration_spearman": round(float(corr), 4),
            "calibration_p_value": round(float(pval), 6),
        }

    # --- report ---

    def report(self, ranking_metrics, personalization, coverage, long_tail,
               diversity, compat, calibration):
        lines = []
        lines.append("=" * 65)
        lines.append("EVALUATION REPORT")
        lines.append("=" * 65)

        lines.append("\n1. Ranking Metrics (averaged)")
        lines.append("-" * 45)
        for m, v in sorted(ranking_metrics.items()):
            lines.append(f"  {m:25s}: {v:.4f}")

        lines.append("\n2. Personalization")
        lines.append("-" * 45)
        for m, v in personalization.items():
            lines.append(f"  {m:25s}: {v:.4f}")
        lines.append("  (higher = more personalized)")

        lines.append("\n3. Coverage")
        lines.append("-" * 45)
        for m, v in coverage.items():
            lines.append(f"  {m:25s}: {v}")

        lines.append("\n4. Long-Tail Discovery")
        lines.append("-" * 45)
        for m, v in long_tail.items():
            lines.append(f"  {m:25s}: {v}")

        lines.append("\n5. Diversity")
        lines.append("-" * 45)
        for m, v in diversity.items():
            lines.append(f"  {m:25s}: {v:.4f}")

        lines.append("\n6. Practical Compatibility")
        lines.append("-" * 45)
        for m, v in compat.items():
            lines.append(f"  {m:25s}: {v}")

        lines.append("\n7. Calibration")
        lines.append("-" * 45)
        for m, v in calibration.items():
            lines.append(f"  {m:25s}: {v}")
        lines.append("  (spearman > 0.3 = meaningful correlation)")

        lines.append("\n" + "=" * 65)
        return "\n".join(lines)
