"""Shared config + feature building for train.py and update.py."""
import json
import urllib.request

import numpy as np
import pandas as pd

HOME = dict(name="Belleville, IL", lat=38.52, lon=-89.98)
TRAIN_STATION = dict(id="BLV", lat=38.545, lon=-89.835)  # Scott AFB / MidAmerica ASOS

# HRRR left out: only 48h ahead, so it would be missing for days 3-7. It drives Futurecast instead.
MODELS = ["gfs_seamless", "ecmwf_ifs025", "ecmwf_aifs025_single", "icon_seamless",
          "gem_seamless", "ukmo_seamless", "ncep_nbm_conus"]
VARS = ["temperature_2m", "dew_point_2m", "wind_speed_10m", "precipitation", "cloud_cover"]
UNITS = "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"

# target column in obs -> model variable it corrects
TARGETS = {"temp": "temperature_2m", "dewpt": "dew_point_2m", "wind": "wind_speed_10m"}


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "stl-weather-hobby"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def model_frame(d):
    """Open-Meteo multi-model response -> DataFrame indexed by UTC hour."""
    h = pd.DataFrame(d["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    return h.set_index("time")


def features(h, var):
    """Every model's value for `var`, their mean/spread, plus time-of-day/season and cloud/precip context."""
    cols = [f"{var}_{m}" for m in MODELS]
    X = h.reindex(columns=cols).copy()
    X["mean"] = X[cols].mean(axis=1)
    X["spread"] = X[cols].std(axis=1)
    X["cloud"] = h.reindex(columns=[f"cloud_cover_{m}" for m in MODELS]).mean(axis=1)
    X["precip"] = h.reindex(columns=[f"precipitation_{m}" for m in MODELS]).mean(axis=1)
    # local solar-ish hour (UTC-6) so the model learns day/night effects
    hr = (h.index.hour - 6) % 24
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    doy = h.index.dayofyear
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return X


def precip_features(h):
    X = h.reindex(columns=[f"precipitation_{m}" for m in MODELS]).copy()
    X["frac_wet"] = (X >= 0.01).sum(axis=1) / X.notna().sum(axis=1)
    X["mean"] = X.iloc[:, :len(MODELS)].mean(axis=1)
    X["cloud"] = h.reindex(columns=[f"cloud_cover_{m}" for m in MODELS]).mean(axis=1)
    doy = h.index.dayofyear
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return X
