"""PathSnap API (Laptop 1): loads the trained model and serves route reconstruction.

Run:   uvicorn api.server:app --host 0.0.0.0 --port 8000
Env:   PATHSNAP_DATA       dataset root (default: ./Student folder)
       PATHSNAP_PARTITION  folder with the 10 evaluation trips (default: student_test)

GET /trips               -> list of trip ids in the evaluation partition
GET /predict/{trip_id}   -> GPS points, predicted route (classifier + HMM), reference route if present, metrics
"""
import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import list_trips  # noqa: E402
from pipeline import load_model, predict_trip, route_arcs  # noqa: E402

PARTITION = os.environ.get("PATHSNAP_PARTITION", "student_test")
MODEL, MODEL_NAME = load_model()
print(f"[PathSnap] model loaded: {MODEL_NAME} | partition: {PARTITION}", flush=True)

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def arcs_to_lines(arcs, trip):
    lat, lon = trip.nodes.lat.to_numpy(), trip.nodes.lon.to_numpy()
    u, v = trip.arcs.u.to_numpy(), trip.arcs.v.to_numpy()
    return [[[float(lat[u[a]]), float(lon[u[a]])], [float(lat[v[a]]), float(lon[v[a]])]] for a in arcs]


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME, "partition": PARTITION}


@app.get("/trips")
def trips():
    return {"partition": PARTITION, "trips": list_trips(PARTITION)}


@app.get("/predict/{trip_id}")
def predict(trip_id: str):
    if trip_id not in list_trips(PARTITION):
        raise HTTPException(404, f"unknown trip {trip_id}")
    t0 = time.time()
    r = predict_trip(MODEL, PARTITION, trip_id)
    trip, pts = r["trip"], r["points"]
    pred_route = route_arcs(r["hmm_arcs"])
    metrics = {k: round(r[k], 4) for k in ("clf_continuity", "hmm_continuity", "clf_top1", "hmm_top1",
                                           "route_precision", "route_recall") if k in r}
    metrics.update({"gps_points": len(pts), "predicted_arcs": len(pred_route),
                    "seconds": round(time.time() - t0, 2)})
    print(f"[PathSnap] trip {trip_id}: {metrics}", flush=True)
    print(f"[PathSnap] predicted arc sequence (first 30): {pred_route[:30]}", flush=True)
    return {
        "trip_id": trip_id,
        "model": MODEL_NAME,
        "metrics": metrics,
        "gps": pts[["lat", "lon"]].round(6).values.tolist(),
        "predicted_arcs": pred_route,
        "predicted_lines": arcs_to_lines(pred_route, trip),
        "reference_lines": arcs_to_lines(trip.route, trip) if trip.route is not None else None,
    }
