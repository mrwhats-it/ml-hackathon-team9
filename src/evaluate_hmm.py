"""Compare classifier-only Top-1 predictions with classifier + HMM/Viterbi on the student-test trips.

Usage:  python src/evaluate_hmm.py
Writes: results/hmm/hmm_vs_classifier.csv, results/hmm/hmm_vs_classifier.png,
        results/maps/predicted_<trip>.html (GPS, road network, reference and predicted route)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import list_trips
from pipeline import load_model, predict_trip, route_arcs
from viz import trip_map

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "hmm")
MAPS = os.path.join(ROOT, "results", "maps")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MAPS, exist_ok=True)
    model, name = load_model()
    rows = []
    for tid in list_trips("student_test"):
        r = predict_trip(model, "student_test", tid)
        rows.append({"trip_id": tid, "gps_points": len(r["points"]),
                     **{k: r[k] for k in ("clf_top1", "hmm_top1", "clf_continuity", "hmm_continuity",
                                          "route_precision", "route_recall", "seconds")}})
        print(rows[-1], flush=True)
        trip_map(r["trip"], route_arcs(r["hmm_arcs"])).save(os.path.join(MAPS, f"predicted_{tid}.html"))
    df = pd.DataFrame(rows)
    # point-weighted overall
    w = df.gps_points
    overall = {"trip_id": "ALL (point-weighted)", "gps_points": int(w.sum()),
               **{c: float((df[c] * w).sum() / w.sum()) for c in df.columns if c not in ("trip_id", "gps_points")}}
    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
    df.to_csv(os.path.join(OUT, "hmm_vs_classifier.csv"), index=False)
    print(f"model: {name}\n", df.round(4).to_string())

    per = df.iloc[:-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = range(len(per))
    for a, (c1, c2, title) in zip(ax, (("clf_top1", "hmm_top1", "Top-1 accuracy"),
                                       ("clf_continuity", "hmm_continuity", "Route continuity"))):
        a.bar([i - .2 for i in x], per[c1], .4, label="classifier only")
        a.bar([i + .2 for i in x], per[c2], .4, label="classifier + HMM")
        a.set_xticks(list(x), per.trip_id, rotation=45)
        a.set(title=f"{title} per student-test trip", ylim=(0, 1.05))
        a.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hmm_vs_classifier.png"), dpi=130)


if __name__ == "__main__":
    main()
