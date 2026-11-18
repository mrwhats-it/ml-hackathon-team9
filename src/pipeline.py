"""End-to-end inference for one trip: raw files -> candidates/features -> classifier -> HMM/Viterbi.

Used by evaluate_hmm.py and by the API.
"""
import os
import time

import joblib
import numpy as np

from data_loader import load_trip
from features import FEATURES, TripGeometry
from hmm import viterbi

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")


def load_model():
    for name in ("final_model.joblib", "XGBoost.joblib"):
        path = os.path.join(MODELS, name)
        if os.path.exists(path):
            return joblib.load(path), name
    raise FileNotFoundError("no trained model in models/ - run train_models.py or tune.py")


def continuity(arc_seq, u, v):
    seq = np.asarray([a for a in arc_seq if a >= 0])
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq
    if len(seq) < 2:
        return 1.0
    return float(np.mean(v[seq[:-1]] == u[seq[1:]]))


def predict_trip(model, partition, trip_id):
    t0 = time.time()
    trip = load_trip(partition, trip_id)
    geo = TripGeometry(trip)
    cands, points = geo.build()
    proba = model.predict_proba(cands[FEATURES].to_numpy(float))[:, 1]

    best = cands.assign(proba=proba).sort_values("proba").groupby("point_idx").arc_id.last()
    clf_arcs = best.reindex(range(len(points))).fillna(-1).astype(int).to_numpy()
    hmm_arcs = viterbi(geo, cands, proba)

    res = {"trip": trip, "geo": geo, "points": points, "clf_arcs": clf_arcs, "hmm_arcs": hmm_arcs,
           "seconds": time.time() - t0,
           "clf_continuity": continuity(clf_arcs, geo.u, geo.v),
           "hmm_continuity": continuity(hmm_arcs, geo.u, geo.v)}
    if "true_arc" in points:
        ok = points.label_reliable.to_numpy() == 1
        truth = points.true_arc.to_numpy()
        res["clf_top1"] = float((clf_arcs[ok] == truth[ok]).mean())
        res["hmm_top1"] = float((hmm_arcs[ok] == truth[ok]).mean())
        route = set(trip.route.tolist())
        dec = set(hmm_arcs[hmm_arcs >= 0].tolist())
        res["route_precision"] = len(dec & route) / max(len(dec), 1)
        res["route_recall"] = len(dec & route) / max(len(route), 1)
    return res


def route_arcs(arc_seq):
    """Collapse per-point arcs into the ordered list of distinct arcs travelled."""
    seq = [int(a) for a in arc_seq if a >= 0]
    return [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
