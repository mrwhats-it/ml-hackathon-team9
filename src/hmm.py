"""HMM + Viterbi route reconstruction.

Hidden states   : at each GPS point, its candidate arcs (from features.py).
Emission        : log of the classifier probability that the candidate is the true arc.
Transition      : compares the distance driven along the road network between two candidates
                  (shortest path, directed graph) with the straight-line distance between the two
                  GPS points:  log T = -|d_route - d_gps| / BETA_M  (Newson & Krumm style).
                  Unreachable pairs get a large finite penalty, so the decoder can still "jump"
                  across holes in the network or long GPS gaps instead of failing.
Decoding        : Viterbi (dynamic programming over points, argmax backtracking).
Points without candidates are skipped and get prediction -1; the chain continues across them.
"""
import heapq

import numpy as np
import pandas as pd

BETA_M = 15.0            # tolerance of route-vs-GPS distance mismatch (metres)
UNREACHABLE_LOGP = -30.0  # penalty when no path is found within the search limit
SEARCH_SLACK_M = 150.0   # Dijkstra limit = 2 * GPS distance + slack
EMISSION_FLOOR = 1e-6


class RoadGraph:
    """Directed road graph (node -> [(next_node, length)]) with cached bounded Dijkstra."""

    def __init__(self, u, v, arc_len):
        order = np.argsort(u, kind="stable")
        self.dst, self.w = v[order], arc_len[order]
        self.ptr = np.searchsorted(u[order], np.arange(max(u.max(), v.max()) + 2))
        self.cache = {}

    def dist_from(self, src, limit):
        hit = self.cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, np.inf) or d > limit:
                continue
            for k in range(self.ptr[n], self.ptr[n + 1]):
                nd = d + self.w[k]
                m = self.dst[k]
                if nd <= limit and nd < dist.get(m, np.inf):
                    dist[m] = nd
                    heapq.heappush(heap, (nd, m))
        self.cache[src] = (limit, dist)
        return dist


def viterbi(geo, cands: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    """Return the decoded arc id per GPS point (-1 where a point has no candidates).

    geo: features.TripGeometry of the trip; cands: its candidate rows (point_idx, arc_id, proj_t).
    """
    n = len(geo.pts)
    graph = RoadGraph(geo.u, geo.v, geo.arc_len)
    c = cands.assign(proba=proba).sort_values(["point_idx", "dist_rank"])
    groups = {p: g for p, g in c.groupby("point_idx")}
    steps = [p for p in range(n) if p in groups]
    out = np.full(n, -1, dtype=np.int64)
    if not steps:
        return out

    arcs_l, t_l, score_l, back_l = [], [], [], []
    for i, p in enumerate(steps):
        g = groups[p]
        arcs = g.arc_id.to_numpy()
        t = g.proj_t.to_numpy()
        emis = np.log(np.clip(g.proba.to_numpy(), EMISSION_FLOOR, 1.0))
        if i == 0:
            score, back = emis, np.full(len(arcs), -1)
        else:
            pa, pt, ps = arcs_l[-1], t_l[-1], score_l[-1]
            q = steps[i - 1]
            d_gps = float(np.hypot(*(geo.pts[p] - geo.pts[q])))
            limit = 2 * d_gps + SEARCH_SLACK_M
            trans = np.full((len(pa), len(arcs)), UNREACHABLE_LOGP)
            for a_i, a in enumerate(pa):
                la = geo.arc_len[a]
                reach = None
                for b_i, b in enumerate(arcs):
                    if a == b:
                        d_route = abs(t[b_i] - pt[a_i]) * la
                        if t[b_i] < pt[a_i]:          # moving backwards on a directed arc
                            d_route += 2 * BETA_M
                    else:
                        if reach is None:
                            reach = graph.dist_from(int(geo.v[a]), limit)
                        d_mid = reach.get(int(geo.u[b]))
                        if d_mid is None:
                            continue
                        d_route = (1 - pt[a_i]) * la + d_mid + t[b_i] * geo.arc_len[b]
                    trans[a_i, b_i] = -abs(d_route - d_gps) / BETA_M
            total = ps[:, None] + trans
            back = total.argmax(0)
            score = total.max(0) + emis
        arcs_l.append(arcs)
        t_l.append(t)
        score_l.append(score)
        back_l.append(back)

    k = int(np.argmax(score_l[-1]))
    for i in range(len(steps) - 1, -1, -1):
        out[steps[i]] = arcs_l[i][k]
        k = int(back_l[i][k])
    return out
