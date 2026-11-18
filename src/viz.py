"""Folium maps: raw GPS, road network, reference route and predicted route as toggleable layers."""
import folium
import numpy as np

from data_loader import Trip


def trip_map(trip: Trip, predicted_arcs=None, network_margin_deg=0.003, max_network_arcs=6000) -> folium.Map:
    tr, nodes, arcs = trip.track, trip.nodes, trip.arcs
    lat = nodes.lat.to_numpy()
    lon = nodes.lon.to_numpy()
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=14, tiles="OpenStreetMap")

    # road network near the trajectory
    lo_lon, hi_lon = tr.lon.min() - network_margin_deg, tr.lon.max() + network_margin_deg
    lo_lat, hi_lat = tr.lat.min() - network_margin_deg, tr.lat.max() + network_margin_deg
    u, v = arcs.u.to_numpy(), arcs.v.to_numpy()
    inside = (lon[u] > lo_lon) & (lon[u] < hi_lon) & (lat[u] > lo_lat) & (lat[u] < hi_lat)
    idx = np.flatnonzero(inside)[:max_network_arcs]
    net = folium.FeatureGroup(name=f"Road network ({len(idx)} arcs)", show=True)
    folium.PolyLine([[[lat[u[i]], lon[u[i]]], [lat[v[i]], lon[v[i]]]] for i in idx],
                    color="#999999", weight=1, opacity=.6).add_to(net)
    net.add_to(m)

    def arc_lines(arc_ids, color, name, weight):
        fg = folium.FeatureGroup(name=name)
        folium.PolyLine([[[lat[u[i]], lon[u[i]]], [lat[v[i]], lon[v[i]]]] for i in arc_ids],
                        color=color, weight=weight, opacity=.85).add_to(fg)
        fg.add_to(m)

    if trip.route is not None:
        arc_lines(trip.route, "#2ca02c", "Reference route (ground truth)", 7)
    if predicted_arcs is not None:
        pred = [a for a in predicted_arcs if a >= 0]
        arc_lines(pred, "#d62728", "Predicted route", 3)

    gps = folium.FeatureGroup(name=f"Raw GPS ({len(tr)} points)")
    step = max(1, len(tr) // 1500)
    for r in tr.iloc[::step].itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=2, color="#1f77b4", fill=True, fill_opacity=.8,
                            weight=0, tooltip=f"t={r.t:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker([tr.lat.iloc[0], tr.lon.iloc[0]], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([tr.lat.iloc[-1], tr.lon.iloc[-1]], tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    return m
