import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import GRID_N

def _fault(ax, m):
    a, h = np.radians(m["strike"]), m["length_km"] / 2
    ax.plot([-h * np.sin(a), h * np.sin(a)], [-h * np.cos(a), h * np.cos(a)], color="white", lw=3)

def predicted_map(df, m, col):
    z = df[col].rank(pct=True).values.reshape(GRID_N, GRID_N)        # brightness = rank inside this area
    E = m["extent"]; fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(z, extent=[-E, E, -E, E], origin="lower", cmap="inferno", vmin=0, vmax=1)
    _fault(ax, m); ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, ticks=[0.05, 0.95]); cb.ax.set_yticklabels(["Less likely", "More likely"])
    fig.tight_layout(); return fig

def actual_map(df, m):
    z = df["label"].values.reshape(GRID_N, GRID_N); E = m["extent"]; fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(z, extent=[-E, E, -E, E], origin="lower", cmap=matplotlib.colors.ListedColormap(["#0B1F3A", "#FFD60A"]))
    _fault(ax, m); ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
    fig.tight_layout(); return fig
