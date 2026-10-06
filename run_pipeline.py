#!/usr/bin/env python3
"""
entry point for the POI ranking system.
generates data, trains model, runs scenarios, evaluates.
"""

import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pipeline import Pipeline


def main():
    pipe = Pipeline(data_dir="data", seed=42)
    results, report = pipe.run()

    # save outputs
    os.makedirs("output", exist_ok=True)

    with open("output/evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("\nsaved evaluation report to output/evaluation_report.txt")

    for res in results:
        sname = res["scenario"]["name"].replace(" ", "_").replace("—", "-")
        out = {
            "scenario": res["scenario"],
            "traveler_id": res["traveler_id"],
            "n_candidates": res["n_candidates"],
            "top_20": [{
                "rank": i + 1,
                "poi_id": r["poi_id"], "name": r["name"],
                "category": r["category"], "score": r["score"],
                "preference_weight": r["preference_weight"],
                "context_compatibility": r["context_compatibility"],
                "practical_compatibility": r["practical_compatibility"],
                "confidence": r["confidence"],
            } for i, r in enumerate(res["ranked"][:20])],
            "explanations": res["explanations"][:10],
        }
        fname = f"output/{sname}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"saved {fname}")


if __name__ == "__main__":
    main()
