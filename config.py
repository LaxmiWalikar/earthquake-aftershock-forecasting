"""All settings in one place."""
import numpy as np

N_EVENTS = 118            # how many major earthquakes to use
MIN_MAINSHOCK_MAG = 7.0   # candidates are M7.0+ ...
START, END = "2000-01-01", "2024-12-31"   # ... between these dates, largest first
DAYS = 30                 # aftershock window after each mainshock
AFTER_MIN_MAG = 4.0       # USGS is only complete for smaller quakes inside some countries
MIN_AFTERSHOCKS = 20      # skip mainshocks with too few aftershocks to learn from
GRID_N = 40               # each earthquake is split into GRID_N x GRID_N cells

def fault_length_km(mag):
    """Wells & Coppersmith (1994) style estimate of rupture length from magnitude (approximation)."""
    return 10 ** (-2.44 + 0.59 * mag)

def extent_km(length_km):
    return max(80.0, 2.0 * length_km)      # map covers +/- this many km around the epicentre

def to_local_km(lat, lon, lat0, lon0):
    x = (np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0))   # east
    y = (np.asarray(lat) - lat0) * 110.57                              # north
    return x, y
