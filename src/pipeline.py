"""
ties everything together.

data -> prep -> features -> train -> rank -> score -> evaluate

nothing fancy, just runs the components in order.
"""

import numpy as np
import pandas as pd
import json, os

from .data_generator import generate_all
from .data_preparation import DataPreparer
from .poi_representation import POIFeatureBuilder
from .traveler_representation import TravelerProfileBuilder
from .candidate_generation import CandidateGenerator
from .ranking_model import Ranker
from .context_scoring import ContextScorer
from .explainability import Explainer
from .evaluation import Evaluator


class Pipeline:
    def __init__(self, data_dir="data", seed=42):
        self.data_dir = data_dir
        self.seed = seed

        self.preparer = DataPreparer()
        self.poi_feats = POIFeatureBuilder()
        self.trav_builder = TravelerProfileBuilder()
        self.candidates = CandidateGenerator()
        self.ranker = Ranker(seed=seed)
        self.scorer = ContextScorer()
        self.explainer = Explainer()
        self.evaluator = Evaluator()

        self.pois = None
        self.travelers = None
        self.interactions = None
        self.poi_features = None
        self.poi_feat_names = None
        self.poi_idx_map = None

    def run(self):
        print("=" * 65)
        print("POI INTELLIGENCE & RANKING SYSTEM")
        print("=" * 65)

        self._load_data()
        self._prep_data()
        self._build_features()
        self._train()
        results = self._run_scenarios()
        report = self._evaluate(results)
        self._show_results(results, report)
        return results, report

    def _load_data(self):
        print("\n[1/6] loading data")
        print("-" * 45)

        paths = [os.path.join(self.data_dir, f) for f in ["pois.csv", "travelers.csv", "interactions.csv"]]
        if all(os.path.exists(p) for p in paths):
            print("  found existing csvs, loading...")
            self.pois = pd.read_csv(paths[0])
            self.travelers = pd.read_csv(paths[1])
            self.interactions = pd.read_csv(paths[2])
        else:
            print("  generating synthetic data...")
            self.pois, self.travelers, self.interactions = generate_all(
                out_dir=self.data_dir, seed=self.seed)

        print(f"  {len(self.pois)} POIs, {len(self.travelers)} travelers, "
              f"{len(self.interactions)} interactions")

    def _prep_data(self):
        print("\n[2/6] preparing data")
        print("-" * 45)
        self.pois = self.preparer.run(self.pois)
        print(f"  {len(self.pois)} POIs after cleaning")
        print(f"  {self.pois['category'].nunique()} categories")

    def _build_features(self):
        print("\n[3/6] building features")
        print("-" * 45)
        self.poi_features, self.poi_feat_names = self.poi_feats.fit_transform(
            self.pois, self.interactions)
        self.poi_idx_map = {pid: i for i, pid in enumerate(self.pois["poi_id"].values)}
        print(f"  POI feature matrix: {self.poi_features.shape}")

    def _train(self):
        print("\n[4/6] training ranker")
        print("-" * 45)
        X, y, fnames = self.ranker.build_training_data(
            self.interactions, self.pois, self.travelers,
            self.trav_builder, self.poi_features, self.poi_feat_names)
        print(f"  {len(X)} training samples, {X.shape[1]} features")
        print(f"  {(y > 0).mean():.1%} positive")
        self.ranker.train(X, y)

        print("\n  top features:")
        for name, imp in list(self.ranker.top_features(10).items()):
            print(f"    {name:35s} {imp:.4f}")

    def _run_scenarios(self):
        print("\n[5/6] running scenarios")
        print("-" * 45)

        scenarios = [
            {"name": "Scenario 1 — Local Food Explorer", "idx": 0,
             "desc": "local food, neighborhoods | medium budget | less touristy"},
            {"name": "Scenario 2 — History & Culture Buff", "idx": 1,
             "desc": "history, architecture, museums | high budget | famous landmarks"},
            {"name": "Scenario 3 — Family Trip", "idx": 2,
             "desc": "family activities, parks, nature | medium budget | family party"},
        ]

        results = []
        for sc in scenarios:
            trav = self.travelers.iloc[sc["idx"]]
            r = self._run_one(trav, sc)
            results.append(r)
        return results

    def _run_one(self, trav_row, scenario):
        tid = trav_row["traveler_id"]

        profile = self.trav_builder.build(trav_row, self.interactions, self.pois)

        cands, cand_scores = self.candidates.generate(profile, self.pois, seed=self.seed)

        pref_scores = self.ranker.predict(
            profile, cands, self.poi_features, self.poi_idx_map, self.trav_builder)

        behav = self._behavioral_scores(tid, cands)

        # add geo distance for context scoring
        cands = cands.copy()
        dist = np.sqrt((cands["latitude"] - 48.8566)**2 + (cands["longitude"] - 2.3522)**2)
        mx = dist.max()
        cands["dist_from_center_norm"] = dist / mx if mx > 0 else 0

        ranked = self.scorer.score(cands, profile, pref_scores, behav)

        # explanations for top 10
        explanations = []
        for r in ranked[:10]:
            poi = self.pois[self.pois["poi_id"] == r["poi_id"]].iloc[0]
            explanations.append(self.explainer.explain(r, profile, poi))

        # ground truth from interactions
        my_ints = self.interactions[self.interactions["traveler_id"] == tid]
        gt = {}
        for _, row in my_ints.iterrows():
            pid = row["poi_id"]
            w = max(row["interaction_weight"], 0) / 8.0
            gt[pid] = max(gt.get(pid, 0), w)

        return {
            "scenario": scenario, "traveler_id": tid, "profile": profile,
            "ranked": ranked, "explanations": explanations,
            "ground_truth": gt, "n_candidates": len(cands),
        }

    def _behavioral_scores(self, tid, cands):
        stats = self.interactions.groupby("poi_id").agg(
            avg_w=("interaction_weight", "mean"),
        ).to_dict("index")
        mx = max((s["avg_w"] for s in stats.values()), default=1.0)
        return {pid: max(stats.get(pid, {"avg_w": 0})["avg_w"], 0) / max(mx, 1)
                for pid in cands["poi_id"].values}

    def _evaluate(self, results):
        print("\n[6/6] evaluating")
        print("-" * 45)

        all_ranked = [r["ranked"] for r in results]
        all_gt = [r["ground_truth"] for r in results]

        # per-scenario ranking metrics, then average
        all_metrics = {}
        for r in results:
            m = self.evaluator.ranking_metrics(r["ranked"], r["ground_truth"])
            for k, v in m.items():
                all_metrics.setdefault(k, []).append(v)
        avg = {k: np.mean(v) for k, v in all_metrics.items()}

        pers = self.evaluator.personalization(all_ranked)
        cov = self.evaluator.coverage(all_ranked, len(self.pois))
        lt = self.evaluator.long_tail(all_ranked, self.pois)
        div = self.evaluator.diversity(all_ranked)
        compat = self.evaluator.constraint_compat(all_ranked)
        cal = self.evaluator.calibration(all_ranked, all_gt)

        return self.evaluator.report(avg, pers, cov, lt, div, compat, cal)

    def _show_results(self, results, report):
        print("\n")
        print("=" * 65)
        print("RESULTS")
        print("=" * 65)

        for res in results:
            sc = res["scenario"]
            print(f"\n{'─' * 65}")
            print(f"  {sc['name']}")
            print(f"  {sc['desc']}")
            print(f"  traveler: {res['traveler_id']} | candidates: {res['n_candidates']}")
            print(f"{'─' * 65}")

            print(f"\n  {'#':<4} {'POI':<28} {'Category':<18} {'Score':<7} "
                  f"{'Pref':<7} {'Ctx':<7} {'Conf':<7}")
            print(f"  {'─'*4} {'─'*28} {'─'*18} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

            for i, r in enumerate(res["ranked"][:10]):
                print(f"  {i+1:<4} {r['name'][:27]:<28} {r['category'][:17]:<18} "
                      f"{r['score']:<7.3f} {r['preference_weight']:<7.3f} "
                      f"{r['context_compatibility']:<7.3f} {r['confidence']:<7.3f}")

            print("\n  why these? (top 5)")
            for expl in res["explanations"][:5]:
                print(f"\n  {self.explainer.format(expl)}")

        # show json output sample
        print(f"\n{'─' * 65}")
        print("  sample json output (top 3, scenario 1):")
        print(f"{'─' * 65}")
        sample = [{
            "poi_id": r["poi_id"], "score": r["score"],
            "preference_weight": r["preference_weight"],
            "context_compatibility": r["context_compatibility"],
            "confidence": r["confidence"],
        } for r in results[0]["ranked"][:3]]
        print(json.dumps(sample, indent=2))

        print("\n")
        print(report)
