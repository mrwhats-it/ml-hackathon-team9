"""Build candidate/feature tables for all trips in train and student_test.

Usage:  python src/build_features.py
Writes: data/processed/{partition}_candidates.csv.gz, {partition}_points.csv.gz,
        results/preprocessing_summary.csv
"""
import os
import time
from multiprocessing import Pool

import pandas as pd

from data_loader import list_trips, load_trip
from features import TripGeometry

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")


def process(args):
    part, tid = args
    t0 = time.time()
    geo = TripGeometry(load_trip(part, tid))
    cands, points = geo.build()
    summary = {"partition": part, "trip_id": tid, **geo.clean_stats,
               "n_candidates": len(cands),
               "cands_per_point": len(cands) / max(len(points), 1),
               "candidate_miss_rate": 1 - points.true_in_cands.mean() if "true_in_cands" in points else None,
               "unreliable_label_frac": 1 - points.label_reliable.mean() if "label_reliable" in points else None,
               "mean_align_dist_m": points.align_dist.mean() if "align_dist" in points else None,
               "seconds": round(time.time() - t0, 2)}
    return part, cands, points, summary


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    jobs = [(p, t) for p in ("train", "student_test") for t in list_trips(p)]
    with Pool(min(6, os.cpu_count())) as pool:
        results = pool.map(process, jobs, chunksize=1)

    summaries = []
    for part in ("train", "student_test"):
        rs = [r for r in results if r[0] == part]
        cands = pd.concat([r[1] for r in rs], ignore_index=True)
        points = pd.concat([r[2] for r in rs], ignore_index=True)
        cands.to_csv(os.path.join(OUT, f"{part}_candidates.csv.gz"), index=False)
        points.to_csv(os.path.join(OUT, f"{part}_points.csv.gz"), index=False)
        print(f"{part}: {len(rs)} trips, {len(points)} points, {len(cands)} candidate rows, "
              f"positive rate {cands.label.mean():.3f}, "
              f"candidate miss rate {1 - points.true_in_cands.mean():.3%} "
              f"(on reliable-label points {1 - points[points.label_reliable == 1].true_in_cands.mean():.3%}), "
              f"unreliable-label points {1 - points.label_reliable.mean():.2%}")
        summaries += [r[3] for r in rs]
    pd.DataFrame(summaries).to_csv(os.path.join(RES, "preprocessing_summary.csv"), index=False)


if __name__ == "__main__":
    main()
