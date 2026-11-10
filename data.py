"""Step 1: pick the N largest earthquakes from USGS and download their aftershocks.
Needs internet. For an offline test run:  python data.py --synthetic
"""
import argparse, io, os, time
import numpy as np, pandas as pd, requests
from config import *
from features import stress_features

URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

def usgs(**p):
    for _ in range(3):
        try:
            r = requests.get(URL, params=dict(format="csv", **p), timeout=90); r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:
            print("  retrying after error:", e); time.sleep(3)
    return pd.DataFrame()

def real():
    cand = usgs(starttime=START, endtime=END, minmagnitude=MIN_MAINSHOCK_MAG, orderby="magnitude", limit=1000)
    cand = cand[["id", "time", "latitude", "longitude", "mag", "place"]]
    print(f"{len(cand)} candidate earthquakes, using the largest {N_EVENTS} with enough aftershocks")
    keep = []
    for row in cand.itertuples():
        if len(keep) >= N_EVENTS: break
        path = f"data/raw/{row.id}.csv"
        if not os.path.exists(path):
            t0 = pd.to_datetime(row.time)
            R = extent_km(fault_length_km(row.mag)) * 1.5
            df = usgs(starttime=t0.strftime("%Y-%m-%dT%H:%M:%S"), endtime=(t0 + pd.Timedelta(days=DAYS)).strftime("%Y-%m-%dT%H:%M:%S"),
                      latitude=row.latitude, longitude=row.longitude, maxradiuskm=R, minmagnitude=AFTER_MIN_MAG, limit=20000)
            if df.empty: continue
            df[df.id != row.id][["time", "latitude", "longitude", "mag"]].to_csv(path, index=False)
        n = len(pd.read_csv(path))
        if n >= MIN_AFTERSHOCKS:
            keep.append(row); print(f"[{len(keep)}/{N_EVENTS}] {row.place} M{row.mag}: {n} aftershocks")
    return pd.DataFrame(keep).drop(columns="Index")

def synthetic(seed=1):
    rng, rows = np.random.default_rng(seed), []
    for i in range(N_EVENTS):
        mag = round(7.0 + rng.exponential(0.35), 1); lat, lon, strike = rng.uniform(-50, 50), rng.uniform(-170, 170), rng.uniform(0, 180)
        L = fault_length_km(mag); E = extent_km(L); ph = rng.uniform(0, 6, 2)
        x, y = rng.uniform(-E, E, (2, 6000)); f = stress_features(x, y, strike, L)[:, 0]
        p = 1 / (1 + np.exp(-(2.2 * (f + 0.5 * np.sin(x / 40 + ph[0]) * np.cos(y / 40 + ph[1])) - 1.8)))
        k = np.where(rng.random(6000) < p)[0][:300]
        eid = f"syn{i:03d}"
        pd.DataFrame(dict(time="", latitude=lat + y[k] / 110.57, longitude=lon + x[k] / (111.32 * np.cos(np.radians(lat))), mag=4.0)).to_csv(f"data/raw/{eid}.csv", index=False)
        rows.append(dict(id=eid, time=f"{2000 + i % 25}-01-01", latitude=lat, longitude=lon, mag=mag, place=f"Synthetic region {i}"))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", action="store_true"); a = ap.parse_args()
    os.makedirs("data/raw", exist_ok=True)
    ev = synthetic() if a.synthetic else real()
    ev.to_csv("data/events.csv", index=False); print(f"Saved {len(ev)} earthquakes to data/events.csv")
