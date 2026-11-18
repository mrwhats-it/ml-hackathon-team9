"""Train and compare candidate classifiers.

Task: binary classification of (GPS point, candidate arc) rows -> is this the true arc?
For each GPS point the candidate with the highest predicted probability is the Top-1 prediction.

Usage:  python src/train_models.py [ModelName ...]   # train the listed models (default: all)
        python src/train_models.py --report          # combine saved runs into tables and plots
Reads:  data/processed/*_candidates.csv.gz, *_points.csv.gz  (from build_features.py)
Writes: results/model_comparison.csv, results/cv_fold_scores.csv, results/plots/*.png,
        results/predictions/student_test_<model>.csv, models/<model>.joblib
"""
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loader import load_trip
from features import FEATURES

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(RES, "plots")
PRED = os.path.join(RES, "predictions")
RUNS = os.path.join(RES, "model_runs")
MODELS = os.path.join(ROOT, "models")
SEED = 42
TRAIN_POINT_STRIDE = 2   # 1 Hz GPS is highly redundant; use every 2nd point of each training trip
KNN_MAX_ROWS = 150_000   # KNN prediction cost grows with training size

MODEL_ZOO = {
    "LogisticRegression": ("Linear", make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))),
    "GaussianNB": ("Probabilistic", make_pipeline(StandardScaler(), GaussianNB())),
    "KNN": ("Instance-based", make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=1))),
    "RandomForest": ("Ensemble (bagging)", RandomForestClassifier(
        n_estimators=200, max_depth=16, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
    "XGBoost": ("Boosting", XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", n_jobs=-1, random_state=SEED)),
    "MLP": ("Neural network", make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
        max_iter=40, early_stopping=True, validation_fraction=0.1, random_state=SEED))),
}


def load(part):
    c = pd.read_csv(os.path.join(PROC, f"{part}_candidates.csv.gz"), dtype={"trip_id": str})
    p = pd.read_csv(os.path.join(PROC, f"{part}_points.csv.gz"), dtype={"trip_id": str})
    return c, p


def fit_rows(model_name, X, y):
    if model_name == "KNN" and len(X) > KNN_MAX_ROWS:
        idx = np.random.default_rng(SEED).choice(len(X), KNN_MAX_ROWS, replace=False)
        return X[idx], y[idx]
    return X, y


def top1_predictions(cands: pd.DataFrame, proba: np.ndarray, points: pd.DataFrame) -> pd.DataFrame:
    """Highest-probability candidate per GPS point (-1 when a point has no candidates)."""
    c = cands[["trip_id", "point_idx", "arc_id"]].assign(proba=proba)
    best = c.loc[c.groupby(["trip_id", "point_idx"]).proba.idxmax()]
    out = points[["trip_id", "point_idx", "true_arc"]].merge(
        best[["trip_id", "point_idx", "arc_id", "proba"]], on=["trip_id", "point_idx"], how="left")
    out["pred_arc"] = out.arc_id.fillna(-1).astype(int)
    return out.drop(columns="arc_id")


def route_continuity(pred: pd.DataFrame, arcs_by_trip: dict) -> float:
    """Fraction of transitions between consecutive (distinct) predicted arcs that are connected
    in the directed road graph (head of previous arc == tail of next arc)."""
    ok = total = 0
    for tid, g in pred.groupby("trip_id"):
        seq = g.sort_values("point_idx").pred_arc.to_numpy()
        seq = seq[seq >= 0]
        seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
        if len(seq) < 2:
            continue
        u, v = arcs_by_trip[tid]
        ok += int((v[seq[:-1]] == u[seq[1:]]).sum())
        total += len(seq) - 1
    return ok / max(total, 1)


def point_metrics(cands, proba, points, arcs_by_trip):
    y = cands.label.to_numpy()
    pred = top1_predictions(cands, proba, points)
    return {
        "accuracy": accuracy_score(y, proba >= .5),
        "precision": precision_score(y, proba >= .5, zero_division=0),
        "recall": recall_score(y, proba >= .5, zero_division=0),
        "macro_f1": f1_score(y, proba >= .5, average="macro"),
        "log_loss": log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7)),
        "top1_accuracy": float((pred.pred_arc == pred.true_arc).mean()),
        "route_continuity": route_continuity(pred, arcs_by_trip),
    }, pred


def main():
    for d in (PLOTS, PRED, MODELS, RUNS):
        os.makedirs(d, exist_ok=True)
    tr_c, tr_p = load("train")
    te_c, te_p = load("student_test")
    # keep only points whose reference label is trustworthy (within RADIUS_M of the reference route)
    tr_c = tr_c[(tr_c.point_idx % TRAIN_POINT_STRIDE == 0) & (tr_c.label_reliable == 1)].reset_index(drop=True)
    tr_p = tr_p[(tr_p.point_idx % TRAIN_POINT_STRIDE == 0) & (tr_p.label_reliable == 1)].reset_index(drop=True)
    te_c = te_c[te_c.label_reliable == 1].reset_index(drop=True)
    te_p = te_p[te_p.label_reliable == 1].reset_index(drop=True)
    print(f"train rows {len(tr_c)} ({tr_p.trip_id.nunique()} trips) | test rows {len(te_c)}")

    arcs_by_trip = {}
    for part, df in (("train", tr_p), ("student_test", te_p)):
        for tid in df.trip_id.unique():
            a = load_trip(part, tid).arcs
            arcs_by_trip[tid] = (a.u.to_numpy(), a.v.to_numpy())

    X, y, groups = tr_c[FEATURES].to_numpy(float), tr_c.label.to_numpy(), tr_c.trip_id.to_numpy()
    Xte, yte = te_c[FEATURES].to_numpy(float), te_c.label.to_numpy()

    # nearest-road baseline (no learning): candidate with smallest distance wins
    base_m, _ = point_metrics(te_c, (te_c.dist_rank == 0).astype(float).to_numpy(), te_p, arcs_by_trip)
    print("Baseline nearest-road:", {k: round(v, 4) for k, v in base_m.items()})
    with open(os.path.join(RUNS, "baseline.json"), "w") as fh:
        json.dump({"row": {"model": "NearestRoad (baseline)", "family": "Rule",
                           **{f"test_{k}": v for k, v in base_m.items()}}, "folds": []}, fh)
    selected = sys.argv[1:] or list(MODEL_ZOO)

    gkf = GroupKFold(n_splits=5)
    for name in selected:
        family, model = MODEL_ZOO[name]
        print(f"\n=== {name}", flush=True)
        fold_f1, fold_top1, fold_rows = [], [], []
        for k, (tr_i, va_i) in enumerate(gkf.split(X, y, groups)):
            m = clone(model)
            m.fit(*fit_rows(name, X[tr_i], y[tr_i]))
            pv = m.predict_proba(X[va_i])[:, 1]
            va_c = tr_c.iloc[va_i]
            va_p = tr_p[tr_p.trip_id.isin(va_c.trip_id.unique())]
            f1 = f1_score(y[va_i], pv >= .5, average="macro")
            top1 = (top1_predictions(va_c, pv, va_p).pipe(lambda d: (d.pred_arc == d.true_arc).mean()))
            fold_f1.append(f1)
            fold_top1.append(top1)
            fold_rows.append({"model": name, "fold": k, "macro_f1": f1, "top1_accuracy": top1})
            print(f"  fold {k}: macro F1 {f1:.4f}  top-1 {top1:.4f}")

        m = clone(model)
        t0 = time.time()
        m.fit(*fit_rows(name, X, y))
        train_time = time.time() - t0
        t0 = time.time()
        pte = m.predict_proba(Xte)[:, 1]
        infer_ms = (time.time() - t0) / te_p.shape[0] * 1000
        met, pred = point_metrics(te_c, pte, te_p, arcs_by_trip)
        pred.to_csv(os.path.join(PRED, f"student_test_{name}.csv"), index=False)
        joblib.dump(m, os.path.join(MODELS, f"{name}.joblib"))
        row = ({"model": name, "family": family,
                     "config": json.dumps({k: v for k, v in m.get_params().items()
                                           if not hasattr(v, "get_params") and k.count("__") <= 1 and
                                           k.split("__")[-1] in ("C", "n_neighbors", "weights", "n_estimators",
                                                                  "max_depth", "min_samples_leaf",
                                                                  "learning_rate", "hidden_layer_sizes", "alpha")}),
                     "cv_macro_f1_mean": np.mean(fold_f1), "cv_macro_f1_std": np.std(fold_f1),
                     "cv_top1_mean": np.mean(fold_top1),
                     **{f"test_{k}": v for k, v in met.items()},
                     "train_time_s": train_time, "inference_ms_per_point": infer_ms})
        with open(os.path.join(RUNS, f"{name}.json"), "w") as fh:
            json.dump({"row": row, "folds": fold_rows}, fh)
        print("  student_test:", {k: round(v, 4) for k, v in met.items()}, f"train {train_time:.1f}s")

        ConfusionMatrixDisplay.from_predictions(yte, pte >= .5, display_labels=["wrong arc", "true arc"],
                                                normalize="true", values_format=".2f")
        plt.title(f"{name} (student_test, row level)")
        plt.savefig(os.path.join(PLOTS, f"confusion_{name}.png"), dpi=120, bbox_inches="tight")
        plt.close("all")
        save_training_curves(name, m, X, y, groups)



def report():
    order = list(MODEL_ZOO) + ["baseline"]
    runs = [json.load(open(os.path.join(RUNS, f"{n}.json"))) for n in order
            if os.path.exists(os.path.join(RUNS, f"{n}.json"))]
    res = pd.DataFrame([r["row"] for r in runs])
    folds = pd.DataFrame([f for r in runs for f in r["folds"]])
    res.to_csv(os.path.join(RES, "model_comparison.csv"), index=False)
    folds.to_csv(os.path.join(RES, "cv_fold_scores.csv"), index=False)
    comparison_plots(res, folds)
    print(res.drop(columns="config", errors="ignore").round(4).to_string())


def save_training_curves(name, m, X, y, groups):
    """Epoch-wise curves only for models that actually have them (MLP, XGBoost)."""
    if name == "MLP":
        mlp = m[-1]
        fig, ax1 = plt.subplots()
        ax1.plot(mlp.loss_curve_, label="training loss")
        ax1.set(xlabel="epoch", ylabel="log loss")
        ax2 = ax1.twinx()
        ax2.plot(mlp.validation_scores_, color="tab:orange", label="validation accuracy")
        ax2.set_ylabel("validation accuracy")
        fig.legend(loc="upper right")
        plt.title("MLP training history")
    elif name == "XGBoost":
        tr_i, va_i = next(GroupKFold(n_splits=5).split(X, y, groups))
        mm = clone(m).fit(X[tr_i], y[tr_i], eval_set=[(X[tr_i], y[tr_i]), (X[va_i], y[va_i])], verbose=False)
        ev = mm.evals_result()
        plt.figure()
        plt.plot(ev["validation_0"]["logloss"], label="train")
        plt.plot(ev["validation_1"]["logloss"], label="validation (held-out trips)")
        plt.xlabel("boosting round")
        plt.ylabel("log loss")
        plt.legend()
        plt.title("XGBoost training history")
    else:
        return
    plt.savefig(os.path.join(PLOTS, f"training_curve_{name}.png"), dpi=120, bbox_inches="tight")
    plt.close("all")


def comparison_plots(res, folds):
    ml = res[res.family != "Rule"]
    fig, ax = plt.subplots(figsize=(8, 4))
    folds.boxplot(column="macro_f1", by="model", ax=ax)
    ax.set(title="5-fold GroupKFold CV macro F1 (train trips)", xlabel="", ylabel="macro F1")
    plt.suptitle("")
    plt.xticks(rotation=20)
    plt.savefig(os.path.join(PLOTS, "cv_macro_f1.png"), dpi=130, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(res))
    ax.bar(x - .2, res.test_top1_accuracy, .4, label="Top-1 accuracy")
    ax.bar(x + .2, res.test_macro_f1, .4, label="macro F1")
    ax.set_xticks(x, res.model, rotation=20)
    ax.set(ylim=(0, 1), title="Student-test performance")
    ax.legend()
    plt.savefig(os.path.join(PLOTS, "test_top1_f1.png"), dpi=130, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ml.model, ml.test_log_loss)
    ax.set(title="Student-test log loss (lower is better)")
    plt.xticks(rotation=20)
    plt.savefig(os.path.join(PLOTS, "test_log_loss.png"), dpi=130, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(res.model, res.test_route_continuity)
    ax.set(ylim=(0, 1), title="Route continuity of Top-1 predictions (student-test)")
    plt.xticks(rotation=20)
    plt.savefig(os.path.join(PLOTS, "test_route_continuity.png"), dpi=130, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    report() if sys.argv[1:] == ["--report"] else main()
