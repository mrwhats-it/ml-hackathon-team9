# PathSnap – Map-Matching & Route Reconstruction (Team 9)

Match noisy GPS observations to a road network: for every GPS point, choose the road segment (arc)
the vehicle was actually on, then reconstruct a connected route.

## Repository layout

```
src/
  data_loader.py      load raw trips, cleaning, local metric projection, geometry helpers
  eda.py              dataset inventory, statistics, plots, Folium maps
  features.py         candidate generation, route alignment (labels), feature engineering
  build_features.py   runs features.py over all trips -> data/processed/
  train_models.py     trains/compares 6 classifiers with GroupKFold CV + student-test evaluation
  viz.py              Folium map builder (GPS / road network / reference route / predicted route)
results/
  eda/                dataset_inventory.csv, summary_stats.csv, EDA plots
  maps/               interactive Folium HTML maps
  plots/              model comparison plots, confusion matrices, training curves
  preprocessing_summary.csv, model_comparison.csv, cv_fold_scores.csv, train_log.txt
  predictions/        student-test Top-1 predictions per model
models/               trained models (joblib)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset at `Student folder/{train,student_test}/<trip_id>/` (or set `PATHSNAP_DATA`).
The dataset is not committed.

## Run

```bash
python src/eda.py             # EDA outputs -> results/eda, results/maps
python src/build_features.py  # candidates + features -> data/processed (~1.5 min)
python src/train_models.py    # model comparison -> results/, models/
```

## Method summary

**Preprocessing.** Tracks are checked for missing values, invalid coordinates, duplicate and unsorted
timestamps (none found). Coordinates are projected to local metres with an equirectangular projection
centred on each trip (error far below GPS noise at city scale), so distances are in metres, not degrees.

**Candidate generation.** Arcs are densified every 10 m into a KD-tree; for each GPS point all arcs within
50 m are found and the 10 nearest kept. Arcs are directed, so most roads appear as two arcs (one per direction).

**Labels.** `.route` lists the true arcs in order but has no timestamps. Each GPS point is assigned to a route arc
by a monotonic dynamic-programming alignment (points move forward along the route) minimising distance, with a
penalty if the GPS heading opposes the arc direction. Points > 50 m from the whole reference route are flagged as
unreliable labels (the reference route does not cover them) and excluded from training/scoring.

**Features** (per GPS point × candidate arc): distance, distance rank, distance minus nearest, distance ratio,
heading difference between GPS movement and arc direction (+ validity flag and rank), speed, time gap,
arc length, projection position along the arc, distance from the previous/next GPS point to the same arc,
number of candidates.

**Models.** Logistic Regression, Gaussian Naive Bayes, KNN, Random Forest, XGBoost, MLP — binary classifiers
whose probabilities rank candidates. Evaluated with 5-fold GroupKFold by trip (no trip in both train and
validation) and on the 10 student-test trips. Metrics: accuracy, precision, recall, macro F1, log loss,
observation-level Top-1 accuracy, candidate miss rate, route continuity, training/inference time.

## Hyperparameter tuning, HMM and evaluation

```bash
python src/tune.py 20         # Optuna (TPE) Bayesian search on XGBoost -> results/tuning/, models/final_model.joblib
python src/evaluate_hmm.py    # classifier-only vs classifier+HMM on student_test -> results/hmm/, results/maps/predicted_*.html
```

**HMM / Viterbi** (`src/hmm.py`): hidden states are each GPS point's candidate arcs; emission = log classifier
probability; transition = `-|network distance - GPS distance| / 15 m`, where network distance is the shortest
directed path between the two candidate positions (bounded Dijkstra). Unreachable pairs get a large finite
penalty (-30) so decoding survives network holes and GPS gaps; points with no candidates are skipped.

## Deployment (two laptops on the same Wi-Fi)

**Laptop 1 – model + API**
```bash
source .venv/bin/activate
# evaluation trips go in "Student folder/<partition>/<trip_id>/" (same layout as student_test)
export PATHSNAP_PARTITION=student_test        # or the evaluator partition folder name
uvicorn api.server:app --host 0.0.0.0 --port 8000
```
Find Laptop 1's IP (`ip addr` on Linux, `ipconfig` on Windows) and allow port 8000 through the firewall.
The terminal prints the metrics and predicted arc sequence for every request.

**Laptop 2 – UI**
Open `ui/index.html` in a browser (needs internet for map tiles), enter `http://<laptop-1-ip>:8000`,
click Connect, choose a trip, click Reconstruct route.

API endpoints: `GET /trips`, `GET /predict/{trip_id}` (docs at `http://<ip>:8000/docs`).
