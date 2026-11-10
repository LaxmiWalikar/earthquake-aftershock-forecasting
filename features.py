"""Step 2: grid, stress features and labels for every earthquake.

IMPORTANT: stress_features() is a PROXY (double-couple-like pattern around a strike-slip fault).
Fault strike is estimated from the aftershock cloud and length from magnitude.
For a real model, replace this with Coulomb stress change from an Okada slip model and use all 6 tensor components.
"""
import os, json
import numpy as np, pandas as pd
from config import *

FEATURES = ["f1", "f2", "f3", "u", "v", "logr"]

def stress_features(x, y, strike_deg, length_km):
    a, h = np.radians(strike_deg), length_km / 2
    u = (x * np.sin(a) + y * np.cos(a)) / h        # along the fault (normalised)
    v = (x * np.cos(a) - y * np.sin(a)) / h        # across the fault (normalised)
    r2 = u**2 + v**2 + 0.15
    f1 = np.clip(2 * u * v / r2**1.5, -2, 2)        # shear-like lobes (also the baseline score)
    return np.column_stack([f1, (u**2 - v**2) / r2**2, 1 / r2, u, v, 0.5 * np.log(r2)])

def estimate_strike(ax, ay):
    """Main direction of the aftershock cloud = approximate fault strike (degrees from north, 0-180)."""
    pts = np.column_stack([ax - ax.mean(), ay - ay.mean()])
    vx, vy = np.linalg.eigh(np.cov(pts.T))[1][:, -1]
    return float(np.degrees(np.arctan2(vx, vy)) % 180)

def build(row):
    cat = pd.read_csv(f"data/raw/{row.id}.csv")
    ax, ay = to_local_km(cat.latitude, cat.longitude, row.latitude, row.longitude)
    L = fault_length_km(row.mag); E = extent_km(L); strike = estimate_strike(ax, ay)
    edges = np.linspace(-E, E, GRID_N + 1); c = (edges[:-1] + edges[1:]) / 2
    X, Y = np.meshgrid(c, c)
    df = pd.DataFrame(stress_features(X.ravel(), Y.ravel(), strike, L), columns=FEATURES)
    df["x"], df["y"], df["event"] = X.ravel(), Y.ravel(), row.id
    H, _, _ = np.histogram2d(ay, ax, bins=[edges, edges])
    df["label"] = (H.ravel() > 0).astype(int)
    return df, dict(name=f"M{row.mag:.1f} {row.place} ({str(row.time)[:4]})", mag=float(row.mag),
                    length_km=L, extent=E, strike=strike, n_aftershocks=len(cat))

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    events, meta = pd.read_csv("data/events.csv"), {}
    for row in events.itertuples():
        df, meta[row.id] = build(row); df.to_csv(f"data/processed/{row.id}.csv", index=False)
    json.dump(meta, open("data/meta.json", "w"), indent=1)
    print(f"Built features for {len(meta)} earthquakes ({GRID_N}x{GRID_N} cells each)")
