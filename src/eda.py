"""Exploratory data analysis of the raw PathSnap dataset.

Usage:  python src/eda.py
Writes: results/eda/dataset_inventory.csv, results/eda/*.png, results/maps/eda_trip_<id>.html
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from data_loader import DATA_DIR, LocalProjection, list_trips, load_trip
from viz import trip_map

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "eda")
MAPS = os.path.join(ROOT, "results", "maps")


def inspect_trip(part, tid):
    trip = load_trip(part, tid)
    tr, nodes, arcs = trip.track, trip.nodes, trip.arcs
    base = os.path.join(DATA_DIR, part, tid, tid)
    dt = np.diff(tr.t.to_numpy())
    proj = LocalProjection(tr.lon.mean(), tr.lat.mean())
    xy = proj(tr.lon, tr.lat)
    step = np.hypot(*np.diff(xy, axis=0).T)

    n = len(nodes)
    g = coo_matrix((np.ones(len(arcs)), (arcs.u, arcs.v)), shape=(n, n))
    n_comp, comp = connected_components(g, directed=True, connection="weak")
    degree = np.bincount(arcs.u, minlength=n) + np.bincount(arcs.v, minlength=n)
    pair = set(zip(arcs.u, arcs.v))
    row = {
        "partition": part, "trip_id": tid,
        "n_gps": len(tr), "n_nodes": n, "n_arcs": len(arcs),
        "route_len": None if trip.route is None else len(trip.route),
        "track_kb": os.path.getsize(base + ".track") / 1024,
        "missing_values": int(tr.isna().sum().sum()),
        "dup_timestamps": int(tr.t.duplicated().sum()),
        "unsorted_timestamps": int((dt < 0).sum()),
        "duration_s": tr.t.iloc[-1] - tr.t.iloc[0],
        "dt_median": np.median(dt), "dt_max": dt.max(),
        "frac_dt_1s": np.mean(np.isclose(dt, 1.0)),
        "gaps_over_5s": int((dt > 5).sum()),
        "max_step_m": step.max(), "mean_speed_mps": step.sum() / max(dt.sum(), 1),
        "extent_ew_km": np.ptp(xy[:, 0]) / 1000, "extent_ns_km": np.ptp(xy[:, 1]) / 1000,
        "lon_min": tr.lon.min(), "lon_max": tr.lon.max(), "lat_min": tr.lat.min(), "lat_max": tr.lat.max(),
        "network_components": n_comp,
        "largest_component_frac": np.bincount(comp).max() / n,
        "isolated_nodes": int((degree == 0).sum()),
        "frac_two_way_arcs": np.mean([(v, u) in pair for u, v in pair]),
    }
    if trip.route is not None:
        r = trip.route
        row["route_connected_frac"] = np.mean(arcs.v.to_numpy()[r[:-1]] == arcs.u.to_numpy()[r[1:]])
        row["route_valid_ids"] = bool((r >= 0).all() and (r < len(arcs)).all())
    return row, dt, step


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MAPS, exist_ok=True)
    rows, dts, steps = [], [], []
    for part in ("train", "student_test"):
        for tid in list_trips(part):
            row, dt, step = inspect_trip(part, tid)
            rows.append(row)
            dts.append(dt)
            steps.append(step)
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(OUT, "dataset_inventory.csv"), index=False)

    print("Trips per partition:\n", inv.partition.value_counts().to_string())
    cols = ["n_gps", "n_nodes", "n_arcs", "route_len", "duration_s", "dt_max", "gaps_over_5s",
            "max_step_m", "mean_speed_mps", "network_components", "isolated_nodes", "route_connected_frac"]
    desc = inv[cols].describe().T
    desc.to_csv(os.path.join(OUT, "summary_stats.csv"))
    print(desc.round(2).to_string())
    print("Total missing values:", inv.missing_values.sum(), "| duplicate timestamps:", inv.dup_timestamps.sum(),
          "| unsorted timestamps:", inv.unsorted_timestamps.sum())
    all_dt = np.concatenate(dts)
    print(f"Sampling: {np.mean(np.isclose(all_dt, 1)):.1%} of steps are 1 s, {np.mean(all_dt > 5):.2%} are > 5 s")

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        ax[0, 0].hist(inv[inv.partition == part].n_gps, bins=20, alpha=.6, label=part, color=c)
    ax[0, 0].set(title="GPS points per trip", xlabel="points", ylabel="trips")
    ax[0, 0].legend()
    ax[0, 1].hist(np.clip(all_dt, 0, 20), bins=40, log=True)
    ax[0, 1].set(title="Time gap between consecutive GPS points (clipped at 20 s)", xlabel="seconds")
    ax[1, 0].hist(np.clip(np.concatenate(steps), 0, 100), bins=50, log=True)
    ax[1, 0].set(title="Distance between consecutive GPS points (clipped at 100 m)", xlabel="metres")
    ax[1, 1].scatter(inv.n_arcs, inv.n_gps, s=12)
    ax[1, 1].set(title="Road network size vs track length", xlabel="arcs in network", ylabel="GPS points", xscale="log")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "eda_overview.png"), dpi=130)

    # one trip: raw GPS vs road network vs route in projected metres
    tid = list_trips("train")[0]
    trip = load_trip("train", tid)
    proj = LocalProjection(trip.track.lon.mean(), trip.track.lat.mean())
    pts = proj(trip.track.lon, trip.track.lat)
    nxy = proj(trip.nodes.lon, trip.nodes.lat)
    a, b = nxy[trip.arcs.u], nxy[trip.arcs.v]
    lo, hi = pts.min(0) - 300, pts.max(0) + 300
    keep = np.all((a > lo) & (a < hi), axis=1)
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in np.flatnonzero(keep):
        ax.plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], color="lightgrey", lw=.6)
    for i in trip.route:
        ax.plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], color="green", lw=2.5)
    ax.scatter(pts[:, 0], pts[:, 1], s=4, color="blue", zorder=3)
    ax.set(title=f"Train trip {tid}: GPS (blue), route (green), roads (grey)", xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal")
    fig.savefig(os.path.join(OUT, f"trajectory_trip_{tid}.png"), dpi=130)

    for part, tid in (("train", list_trips("train")[0]), ("student_test", list_trips("student_test")[0])):
        trip_map(load_trip(part, tid)).save(os.path.join(MAPS, f"eda_{part}_{tid}.html"))
    print("Saved plots to", OUT, "and maps to", MAPS)


if __name__ == "__main__":
    main()
