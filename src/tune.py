"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model: XGBoost.

Objective: mean observation-level Top-1 accuracy over 3-fold GroupKFold (grouped by trip)
on the 80 training trips. The student-test trips are only used after tuning, to compare
the tuned model with the untuned one.

Usage:  python src/tune.py [n_trials]
Writes: results/tuning/optuna_trials.csv, best_params.json, tuned_vs_untuned.csv, plots,
        models/final_model.joblib
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
import optuna
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from features import FEATURES
from train_models import MODELS, RES, SEED, load, load_trip, point_metrics, top1_predictions

OUT = os.path.join(RES, "tuning")
TUNE_POINT_STRIDE = 4   # every 4th GPS point per training trip during the search (speed)
FINAL_POINT_STRIDE = 2  # same as train_models.py for the final refit


def make_model(params):
    return XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=SEED, **params)


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    os.makedirs(OUT, exist_ok=True)
    tr_c, tr_p = load("train")
    te_c, te_p = load("student_test")
    tr_c = tr_c[tr_c.label_reliable == 1]
    tr_p = tr_p[tr_p.label_reliable == 1]
    te_c = te_c[te_c.label_reliable == 1].reset_index(drop=True)
    te_p = te_p[te_p.label_reliable == 1].reset_index(drop=True)

    s_c = tr_c[tr_c.point_idx % TUNE_POINT_STRIDE == 0].reset_index(drop=True)
    s_p = tr_p[tr_p.point_idx % TUNE_POINT_STRIDE == 0].reset_index(drop=True)
    X, y, groups = s_c[FEATURES].to_numpy(float), s_c.label.to_numpy(), s_c.trip_id.to_numpy()
    folds = list(GroupKFold(n_splits=3).split(X, y, groups))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
        }
        top1s, f1s, lls = [], [], []
        for tr_i, va_i in folds:
            m = make_model(params).fit(X[tr_i], y[tr_i])
            pv = m.predict_proba(X[va_i])[:, 1]
            va_c = s_c.iloc[va_i]
            va_p = s_p[s_p.trip_id.isin(va_c.trip_id.unique())]
            pred = top1_predictions(va_c, pv, va_p)
            top1s.append((pred.pred_arc == pred.true_arc).mean())
            f1s.append(f1_score(y[va_i], pv >= .5, average="macro"))
            lls.append(log_loss(y[va_i], np.clip(pv, 1e-7, 1 - 1e-7)))
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1s)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        return float(np.mean(top1s))

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    search_s = time.time() - t0
    study.trials_dataframe().to_csv(os.path.join(OUT, "optuna_trials.csv"), index=False)
    best = study.best_params
    json.dump({"best_params": best, "best_cv_top1": study.best_value, "n_trials": n_trials,
               "search_seconds": search_s}, open(os.path.join(OUT, "best_params.json"), "w"), indent=2)
    print("best", best, "cv top-1", round(study.best_value, 4), f"search {search_s:.0f}s")

    # refit untuned (default config from train_models) and tuned on the same data, compare on student_test
    f_c = tr_c[tr_c.point_idx % FINAL_POINT_STRIDE == 0].reset_index(drop=True)
    Xf, yf = f_c[FEATURES].to_numpy(float), f_c.label.to_numpy()
    arcs_by_trip = {t: (lambda a: (a.u.to_numpy(), a.v.to_numpy()))(load_trip("student_test", t).arcs)
                    for t in te_p.trip_id.unique()}
    untuned = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}
    untuned_trial = study.trials[0].user_attrs  # not the same config; CV for untuned computed below
    rows = []
    for name, params in (("XGBoost untuned", untuned), ("XGBoost tuned", best)):
        cv_top1, cv_f1, cv_ll = [], [], []
        for tr_i, va_i in folds:
            m = make_model(params).fit(X[tr_i], y[tr_i])
            pv = m.predict_proba(X[va_i])[:, 1]
            va_c = s_c.iloc[va_i]
            pred = top1_predictions(va_c, pv, s_p[s_p.trip_id.isin(va_c.trip_id.unique())])
            cv_top1.append((pred.pred_arc == pred.true_arc).mean())
            cv_f1.append(f1_score(y[va_i], pv >= .5, average="macro"))
            cv_ll.append(log_loss(y[va_i], np.clip(pv, 1e-7, 1 - 1e-7)))
        t0 = time.time()
        m = make_model(params).fit(Xf, yf)
        train_s = time.time() - t0
        t0 = time.time()
        pte = m.predict_proba(te_c[FEATURES].to_numpy(float))[:, 1]
        infer_ms = (time.time() - t0) / len(te_p) * 1000
        met, _ = point_metrics(te_c, pte, te_p, arcs_by_trip)
        rows.append({"model": name, "params": json.dumps(params), "cv_top1": np.mean(cv_top1),
                     "cv_macro_f1": np.mean(cv_f1), "cv_log_loss": np.mean(cv_ll),
                     **{f"test_{k}": v for k, v in met.items()},
                     "train_time_s": train_s, "inference_ms_per_point": infer_ms})
        if name == "XGBoost tuned":
            joblib.dump(m, os.path.join(MODELS, "final_model.joblib"))
    del untuned_trial
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(os.path.join(OUT, "tuned_vs_untuned.csv"), index=False)
    print(cmp_df.drop(columns="params").round(4).to_string())

    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(os.path.join(OUT, "optimization_history.png"), dpi=120, bbox_inches="tight")
    try:
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(os.path.join(OUT, "param_importances.png"), dpi=120, bbox_inches="tight")
    except Exception as e:  # importance needs enough completed trials
        print("param importance plot skipped:", e)
    plt.close("all")


if __name__ == "__main__":
    main()
