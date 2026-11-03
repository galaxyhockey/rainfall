"""Hourly job: live model runs -> your blend, plus observations and map grids -> site/data.js."""
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common import HOME, MODELS, TARGETS, UNITS, VARS, features, get_json, model_frame, precip_features

TZ = "America/Chicago"
BBOX = (37.4, -92.4, 39.6, -88.3)  # south, west, north, east: Rolla to Effingham
# Places tab: raw model forecasts (the trained blend only covers Belleville). Edit freely.
PLACES = [("St. Louis", 38.627, -90.199), ("O'Fallon, IL", 38.592, -89.911), ("Edwardsville", 38.811, -89.953),
          ("St. Charles", 38.784, -90.481), ("Columbia, IL", 38.444, -90.201)]
GRID_STEP = 0.2  # ponytail: ~250 points keeps us well under Open-Meteo's free daily limit; 0.1 for smoother maps

with open("models.pkl", "rb") as f:
    M = pickle.load(f)


def blend_forecast():
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={HOME['lat']}&longitude={HOME['lon']}"
           f"&hourly={','.join(VARS + ['cape', 'wind_gusts_10m'])}&models={','.join(MODELS)}&forecast_days=9{UNITS}")
    h = model_frame(get_json(url))
    h = h[h.index >= pd.Timestamp.now("UTC").tz_localize(None).floor("h")]
    out = pd.DataFrame(index=h.index)
    for tgt, var in TARGETS.items():
        out[tgt] = M[tgt].predict(features(h, var))
    out["rain"] = M["wet"].predict_proba(precip_features(h))[:, 1]
    out["cloud"] = h.reindex(columns=[f"cloud_cover_{m}" for m in MODELS]).mean(axis=1)
    out["cape"] = h.reindex(columns=[f"cape_{m}" for m in MODELS]).max(axis=1)
    out["gust"] = h.reindex(columns=[f"wind_gusts_10m_{m}" for m in MODELS]).mean(axis=1)
    out["precip"] = h.reindex(columns=[f"precipitation_{m}" for m in MODELS]).mean(axis=1)
    out.index = out.index.tz_localize("UTC").tz_convert(TZ)
    return out


def icon(rain, cloud, temp, cape):
    if rain >= 0.4:
        if temp <= 32:
            return "snow"
        return "storm" if cape >= 500 else "rain"
    if cloud >= 75:
        return "cloudy"
    return "partly" if cloud >= 30 else "sun"


def humidity(t_f, td_f):
    t, td = (t_f - 32) * 5 / 9, (td_f - 32) * 5 / 9
    return round(min(100, 100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))))


def summarize(f):
    hourly = [dict(t=t.isoformat(), temp=round(r.temp), dewpt=round(r.dewpt), wind=round(r.wind), gust=round(r.gust),
                   rh=humidity(r.temp, r.dewpt), rain=int(round(r.rain * 10) * 10), icon=icon(r.rain, r.cloud, r.temp, r.cape))
              for t, r in f.iloc[:24].iterrows()]
    daily = []
    for d, g in f.groupby(f.index.date):
        day = g.between_time("07:00", "19:00")
        if d == f.index[0].date() and len(day) < 10:  # today: whatever's left of it
            day = g
        # like NWS/TV: the low is the night that follows (7pm -> 9am next morning)
        start = pd.Timestamp(d).tz_localize(TZ)
        night = f[(f.index >= start + pd.Timedelta(hours=19)) & (f.index <= start + pd.Timedelta(hours=33))]
        if len(day) == 0 or len(night) < 12:  # skip the partial day at the end
            continue
        # ponytail: daily chance = wettest hour's chance; understates long drizzly days. Train a daily model if it matters.
        rain = g.rain.max()
        daily.append(dict(date=str(d), hi=round(day.temp.max()), lo=round(night.temp.min()),
                          rain=int(round(rain * 10) * 10), precip=round(float(g.precip.sum()), 2), gust=round(g.gust.max()),
                          icon=icon(rain, day.cloud.mean(), day.temp.mean(), g.cape.max())))
    return hourly, daily[:7]


def observations():
    s, w, n, e = BBOX
    obs = get_json(f"https://aviationweather.gov/api/data/metar?bbox={s},{w},{n},{e}&format=json")
    out = []
    for o in obs:
        if o.get("temp") is None:
            continue
        out.append(dict(id=o["icaoId"], name=(o.get("name") or o["icaoId"]).split(",")[0], lat=o["lat"], lon=o["lon"],
                        temp=round(o["temp"] * 9 / 5 + 32), wdir=o.get("wdir"),
                        rh=humidity(o["temp"] * 9 / 5 + 32, o["dewp"] * 9 / 5 + 32) if o.get("dewp") is not None else None,
                        wind=round((o.get("wspd") or 0) * 1.15078) if isinstance(o.get("wspd"), (int, float)) else 0,
                        gust=round(o["wgst"] * 1.15078) if o.get("wgst") else None))
    return out


def grid():
    """HRRR on a lat/lon grid: current temp/wind + next 18h precip for Futurecast."""
    s, w, n, e = BBOX
    lats, lons = np.arange(s, n + 1e-6, GRID_STEP), np.arange(w, e + 1e-6, GRID_STEP)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={','.join(f'{x:.2f}' for x in LA.ravel())}"
           f"&longitude={','.join(f'{x:.2f}' for x in LO.ravel())}"
           f"&hourly=temperature_2m,wind_speed_10m,precipitation,weather_code&models=gfs_hrrr&forecast_hours=18{UNITS}")
    pts = get_json(url)
    times = pts[0]["hourly"]["time"]
    now_i = 0
    field = lambda k: np.array([p["hourly"][k] for p in pts], dtype=float).T.reshape(len(times), len(lats), len(lons))
    temp, wind, precip, code = (field(k) for k in ["temperature_2m", "wind_speed_10m", "precipitation", "weather_code"])
    # precip type from WMO weather code: 66/67/56/57 freezing, 71-77/85/86 snow
    ptype = np.where(np.isin(code, [56, 57, 66, 67]), 2, np.where(np.isin(code, [71, 73, 75, 77, 85, 86]), 3, 1))
    r = lambda a: np.round(np.nan_to_num(a), 2).tolist()
    return dict(bbox=[round(float(x), 3) for x in (lats[0], lons[0], lats[-1], lons[-1])], rows=len(lats), cols=len(lons),
                times=[pd.Timestamp(t).tz_localize("UTC").tz_convert(TZ).isoformat() for t in times],
                temp=r(temp[now_i]), wind=r(wind[now_i]), precip=r(precip), ptype=ptype.astype(int).tolist())


def nowcast():
    """Next 3 hours of rain in 15-minute steps, plus today's sunrise/sunset."""
    d = get_json(f"https://api.open-meteo.com/v1/forecast?latitude={HOME['lat']}&longitude={HOME['lon']}"
                 f"&minutely_15=precipitation&forecast_minutely_15=20&daily=sunrise,sunset&forecast_days=1"
                 f"&timezone={TZ}&precipitation_unit=inch")
    return dict(times=d["minutely_15"]["time"], precip=d["minutely_15"]["precipitation"],
                sunrise=d["daily"]["sunrise"][0], sunset=d["daily"]["sunset"][0])


def places():
    lat, lon = ",".join(str(p[1]) for p in PLACES), ",".join(str(p[2]) for p in PLACES)
    pts = get_json(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                   f"&current=temperature_2m,weather_code,is_day&daily=temperature_2m_max,temperature_2m_min"
                   f"&forecast_days=1&timezone={TZ}{UNITS}")
    return [dict(name=p[0], temp=round(d["current"]["temperature_2m"]), code=d["current"]["weather_code"],
                 is_day=d["current"]["is_day"], hi=round(d["daily"]["temperature_2m_max"][0]),
                 lo=round(d["daily"]["temperature_2m_min"][0])) for p, d in zip(PLACES, pts)]


def nws():
    """NWS 7-day for the same point, shown side by side and logged so you can score yourself against it later."""
    p = get_json(f"https://api.weather.gov/points/{HOME['lat']},{HOME['lon']}")
    periods = get_json(p["properties"]["forecast"])["properties"]["periods"]
    with open("data/nws_log.jsonl", "a") as f:
        f.write(json.dumps(dict(fetched=datetime.now(timezone.utc).isoformat(), periods=periods)) + "\n")
    days = {}
    for q in periods:
        d = days.setdefault(q["startTime"][:10], {})
        d["hi" if q["isDaytime"] else "lo"] = q["temperature"]
        d["rain"] = max(d.get("rain", 0), (q.get("probabilityOfPrecipitation") or {}).get("value") or 0)
    return days


def safe(fn):
    try:
        return fn()
    except Exception as e:  # one feed being down shouldn't blank the whole page
        print(f"{fn.__name__} failed: {e}")
        return None


def main():
    os.makedirs("site", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    f = blend_forecast()
    hourly, daily = summarize(f)
    data = dict(place=HOME["name"], lat=HOME["lat"], lon=HOME["lon"],
                updated=datetime.now().astimezone().isoformat(timespec="minutes"),
                hourly=hourly, daily=daily, obs=safe(observations), grid=safe(grid), nws=safe(nws),
                nowcast=safe(nowcast), places=safe(places))
    with open("site/data.js", "w") as fh:
        fh.write("window.WX = " + json.dumps(data) + ";\n")
    print(f"wrote site/data.js: now {hourly[0]['temp']}F, {len(daily)} days, "
          f"{len(data['obs'] or [])} stations, grid {'ok' if data['grid'] else 'missing'}")


if __name__ == "__main__":
    main()
