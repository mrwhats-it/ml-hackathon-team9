"""Loading raw PathSnap trips and basic geometry helpers.

Each trip folder <part>/<id>/ holds <id>.track, .nodes, .arcs and (optionally) .route.
All files are whitespace separated with no header.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get(
    "PATHSNAP_DATA",
    os.path.join(os.path.dirname(__file__), "..", "Student folder"),
)
EARTH_R = 6_371_000.0  # metres


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame          # lon, lat, t
    nodes: pd.DataFrame          # lon, lat   (row index = node_id)
    arcs: pd.DataFrame           # u, v       (row index = arc_id), directed u -> v
    route: np.ndarray | None     # ground-truth arc_ids in order


def list_trips(partition: str) -> list[str]:
    return sorted(os.listdir(os.path.join(DATA_DIR, partition)))


def load_trip(partition: str, trip_id: str) -> Trip:
    base = os.path.join(DATA_DIR, partition, trip_id, trip_id)
    track = pd.read_csv(base + ".track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(base + ".nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(base + ".arcs", sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64)
    route = None
    if os.path.exists(base + ".route"):
        route = pd.read_csv(base + ".route", header=None).iloc[:, 0].to_numpy(np.int64)
    return Trip(trip_id, partition, track, nodes, arcs, route)


def clean_track(track: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop NaNs, implausible coordinates and duplicate timestamps; sort by time."""
    n0 = len(track)
    t = track.dropna()
    n_nan = n0 - len(t)
    ok = t.lon.between(-180, 180) & t.lat.between(-90, 90)
    n_bad = int((~ok).sum())
    t = t[ok].sort_values("t", kind="stable")
    n_dup = int(t.duplicated("t").sum())
    t = t.drop_duplicates("t").reset_index(drop=True)
    return t, {"n_raw": n0, "n_nan": n_nan, "n_bad_coord": n_bad, "n_dup_time": n_dup, "n_clean": len(t)}


class LocalProjection:
    """Equirectangular projection to metres around a reference point.

    At city scale (< ~50 km) the error versus a true geodesic is well below GPS noise,
    and it lets us use plain Euclidean geometry for point-to-segment distances.
    """

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = EARTH_R * np.cos(np.radians(lat0)) * np.pi / 180
        self.ky = EARTH_R * np.pi / 180

    def __call__(self, lon, lat) -> np.ndarray:
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        return np.column_stack([(lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky])


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Row-wise distance from points p to segments a-b (all shape (k, 2)).

    Returns (distance, t) where t in [0, 1] is the projection position along the segment.
    """
    ab = b - a
    len2 = np.einsum("ij,ij->i", ab, ab)
    t = np.where(len2 > 0, np.einsum("ij,ij->i", p - a, ab) / np.where(len2 > 0, len2, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + ab * t[:, None]
    return np.linalg.norm(p - proj, axis=1), t


def bearing(dx, dy):
    """Compass-style bearing in degrees [0, 360) of a vector in projected metres."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
