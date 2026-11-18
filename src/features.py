"""Candidate generation, label alignment and feature engineering.

Pipeline for one trip:
  1. clean the GPS track and project everything to local metres
  2. for each GPS point, find candidate road segments (arcs) within RADIUS_M, keep the K nearest
  3. align GPS points to the ground-truth route (monotonic DP) to get a true arc per point
  4. build one feature row per (GPS point, candidate arc) with label = candidate is the true arc
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data_loader import LocalProjection, Trip, bearing, clean_track, point_segment_distance

RADIUS_M = 50.0      # candidate search radius
MAX_CANDS = 10       # keep at most this many nearest candidates per point
SAMPLE_STEP_M = 10.0  # arc densification step for the KD-tree prefilter
MIN_MOVE_M = 1.0     # below this displacement GPS heading is unreliable

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio",
    "heading_diff", "heading_valid", "heading_rank",
    "speed", "dt_prev", "arc_len", "proj_t",
    "dist_prev_pt", "dist_next_pt", "n_cands",
]


class TripGeometry:
    """Projected track and road network for one trip."""

    def __init__(self, trip: Trip):
        self.trip = trip
        self.track, self.clean_stats = clean_track(trip.track)
        self.proj = LocalProjection(self.track.lon.mean(), self.track.lat.mean())
        self.pts = self.proj(self.track.lon, self.track.lat)
        self.node_xy = self.proj(trip.nodes.lon, trip.nodes.lat)
        self.u = trip.arcs.u.to_numpy()
        self.v = trip.arcs.v.to_numpy()
        self.a = self.node_xy[self.u]
        self.b = self.node_xy[self.v]
        d = self.b - self.a
        self.arc_len = np.hypot(d[:, 0], d[:, 1])
        self.arc_bearing = bearing(d[:, 0], d[:, 1])
        self.gps_heading, self.heading_valid, self.speed, self.dt_prev = self._kinematics()

    def _kinematics(self):
        p, t = self.pts, self.track.t.to_numpy()
        n = len(p)
        # central difference heading (forward/backward at the ends)
        prev_i = np.maximum(np.arange(n) - 1, 0)
        next_i = np.minimum(np.arange(n) + 1, n - 1)
        d = p[next_i] - p[prev_i]
        move = np.hypot(d[:, 0], d[:, 1])
        dt = np.maximum(t[next_i] - t[prev_i], 1e-6)
        heading = bearing(d[:, 0], d[:, 1])
        valid = move >= MIN_MOVE_M
        dt_prev = np.concatenate([[0.0], np.diff(t)])
        return heading, valid, move / dt, dt_prev

    def candidates(self) -> pd.DataFrame:
        """(point_idx, arc_id, dist, proj_t) for the K nearest arcs within RADIUS_M."""
        lo = self.pts.min(0) - RADIUS_M * 2
        hi = self.pts.max(0) + RADIUS_M * 2
        mid = (self.a + self.b) / 2
        half = self.arc_len / 2 + RADIUS_M * 2
        near = np.all((mid > lo - half[:, None]) & (mid < hi + half[:, None]), axis=1)
        arc_ids = np.flatnonzero(near)

        # densify arcs into sample points so a KD-tree can find nearby segments
        n_s = np.ceil(self.arc_len[arc_ids] / SAMPLE_STEP_M).astype(int) + 1
        owner = np.repeat(arc_ids, n_s)
        frac = np.concatenate([np.linspace(0, 1, k) for k in n_s])
        samples = self.a[owner] + (self.b[owner] - self.a[owner]) * frac[:, None]
        tree = cKDTree(samples)
        hits = tree.query_ball_point(self.pts, r=RADIUS_M + SAMPLE_STEP_M / 2)

        pi = np.concatenate([np.full(len(h), i) for i, h in enumerate(hits)]).astype(np.int64)
        ai = owner[np.concatenate([np.asarray(h, dtype=np.int64) for h in hits])]
        pairs = np.unique(np.column_stack([pi, ai]), axis=0)
        dist, t = point_segment_distance(self.pts[pairs[:, 0]], self.a[pairs[:, 1]], self.b[pairs[:, 1]])
        df = pd.DataFrame({"point_idx": pairs[:, 0], "arc_id": pairs[:, 1], "dist": dist, "proj_t": t})
        df = df[df.dist <= RADIUS_M].sort_values(["point_idx", "dist"], kind="stable")
        df["dist_rank"] = df.groupby("point_idx").cumcount()
        return df[df.dist_rank < MAX_CANDS].reset_index(drop=True)

    def align_route(self) -> tuple[np.ndarray, np.ndarray]:
        """Assign each GPS point to one arc of the ground-truth route.

        The route has no timestamps, so we solve a monotonic alignment by dynamic programming:
        point n is matched to route position m(n) with m non-decreasing over time, minimising
        sum of point-to-arc distance (+ a penalty when the GPS heading opposes the arc).
        Returns (true_arc per point, alignment distance per point).
        """
        route = self.trip.route
        n, m = len(self.pts), len(route)
        P = np.repeat(self.pts, m, axis=0)
        A = np.tile(self.a[route], (n, 1))
        B = np.tile(self.b[route], (n, 1))
        dist = point_segment_distance(P, A, B)[0].reshape(n, m)
        hd = np.abs((self.gps_heading[:, None] - self.arc_bearing[route][None, :] + 180) % 360 - 180)
        cost = dist + 20.0 * ((hd > 90) & self.heading_valid[:, None])

        back = np.zeros((n, m), dtype=np.int32)
        acc = cost[0].copy()
        idx = np.arange(m)
        for i in range(1, n):
            # best predecessor position <= m (running argmin)
            run_min = np.minimum.accumulate(acc)
            is_new = np.concatenate([[True], acc[1:] < run_min[:-1]])
            back[i] = np.maximum.accumulate(np.where(is_new, idx, 0))
            acc = cost[i] + run_min
        pos = np.empty(n, dtype=np.int64)
        pos[-1] = int(np.argmin(acc))
        for i in range(n - 1, 0, -1):
            pos[i - 1] = back[i, pos[i]]
        return route[pos], dist[np.arange(n), pos]

    def build(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (candidate feature rows, per-point table)."""
        c = self.candidates()
        p = c.point_idx.to_numpy()
        arc = c.arc_id.to_numpy()

        c["dist_minus_min"] = c.dist - c.groupby("point_idx").dist.transform("min")
        c["dist_ratio"] = c.dist / (c.groupby("point_idx").dist.transform("min") + 1.0)
        hd = np.abs((self.gps_heading[p] - self.arc_bearing[arc] + 180) % 360 - 180)
        valid = self.heading_valid[p]
        c["heading_diff"] = np.where(valid, hd, 90.0)   # neutral value when stationary
        c["heading_valid"] = valid.astype(int)
        c["heading_rank"] = c.groupby("point_idx").heading_diff.rank(method="min") - 1
        c["speed"] = self.speed[p]
        c["dt_prev"] = self.dt_prev[p]
        c["arc_len"] = self.arc_len[arc]
        n = len(self.pts)
        for col, off in (("dist_prev_pt", -1), ("dist_next_pt", 1)):
            q = np.clip(p + off, 0, n - 1)
            c[col] = np.minimum(point_segment_distance(self.pts[q], self.a[arc], self.b[arc])[0], 2 * RADIUS_M)
        c["n_cands"] = c.groupby("point_idx").arc_id.transform("size")

        points = pd.DataFrame({"point_idx": np.arange(n), "t": self.track.t.to_numpy(),
                               "lon": self.track.lon.to_numpy(), "lat": self.track.lat.to_numpy()})
        if self.trip.route is not None:
            true_arc, align_d = self.align_route()
            points["true_arc"] = true_arc
            points["align_dist"] = align_d
            c["label"] = (arc == true_arc[p]).astype(int)
            hit = c.groupby("point_idx").label.max()
            points["true_in_cands"] = points.point_idx.map(hit).fillna(0).astype(int)
            # points far from the whole reference route: the route does not cover them -> unreliable label
            points["label_reliable"] = (align_d <= RADIUS_M).astype(int)
            c["label_reliable"] = points.label_reliable.to_numpy()[p]
        c.insert(0, "trip_id", self.trip.trip_id)
        points.insert(0, "trip_id", self.trip.trip_id)
        return c, points
