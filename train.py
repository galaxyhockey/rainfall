"""Train the KBLV blender: past model forecasts -> what Scott AFB actually observed.

Run occasionally (monthly is plenty). Downloads are cached in data/.
"""
import io
import os
import pickle
import urllib.request
from datetime import date, timedelta

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from common import MODELS, TARGETS, TRAIN_STATION, UNITS, VARS, features, get_json, model_frame, precip_features

START = "2024-02-03"  # first day ECMWF history is available
END = (date.today() - timedelta(days=2)).isoformat()
TEST_FROM = (date.today() - timedelta(days=365)).isoformat()  # hold out the last year for scoring
os.makedirs("data", exist_ok=True)


def load_forecasts():
    path = "data/forecasts.pkl"
    if os.path.exists(path):
        return pd.read_pickle(path)
    s = TRAIN_STATION
    # ponytail: historical-forecast API stitches the freshest run (~0-24h lead), so the blender
    # learns short-range errors and applies them at every lead. Previous-Runs API if day-5 skill matters.
    parts = []
    for a in pd.date_range(START, END, freq="90D"):  # one big request times out, so ~quarterly chunks
        b = min(a + pd.Timedelta(days=89), pd.Timestamp(END))
        print(f"  forecasts {a.date()} .. {b.date()}")
        url = (f"https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={s['lat']}&longitude={s['lon']}"
               f"&start_date={a.date()}&end_date={b.date()}&hourly={','.join(VARS)}&models={','.join(MODELS)}{UNITS}")
        parts.append(model_frame(get_json(url)))
    h = pd.concat(parts)
    h.to_pickle(path)
    return h


def load_obs():
    path = "data/obs.pkl"
    if os.path.exists(path):
        return pd.read_pickle(path)
    a, b = date.fromisoformat(START), date.fromisoformat(END)
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=" + TRAIN_STATION["id"] +
           "&data=tmpf&data=dwpf&data=sknt&data=p01i&tz=Etc/UTC&format=onlycomma&latlon=no&missing=empty"
           f"&trace=0.0001&report_type=3&year1={a.year}&month1={a.month}&day1={a.day}"
           f"&year2={b.year}&month2={b.month}&day2={b.day}")
    with urllib.request.urlopen(url, timeout=300) as r:
        o = pd.read_csv(io.StringIO(r.read().decode()))
    # hourly METARs land at :55 -> credit them to the top of the next hour, like model output
    o["time"] = pd.to_datetime(o["valid"]).dt.ceil("h")
    o = o.groupby("time").last()
    o = pd.DataFrame({"temp": o["tmpf"], "dewpt": o["dwpf"], "wind": o["sknt"] * 1.15078,
                      "wet": (o["p01i"] >= 0.01).where(o["p01i"].notna())})
    o.to_pickle(path)
    return o


def main():
    h, o = load_forecasts(), load_obs()
    df = h.join(o, how="inner")
    test = df.index >= TEST_FROM
    print(f"{len(df):,} hours of matched data, scoring on {test.sum():,} held-out hours since {TEST_FROM}\n")

    models = {}
    for tgt, var in TARGETS.items():
        X, y = features(df, var), df[tgt]
        ok = y.notna()
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05)
        m.fit(X[ok & ~test], y[ok & ~test])

        t = ok & test
        score = {f"raw {c}": (X.loc[t, f"{var}_{c}"] - y[t]).abs().mean() for c in MODELS}
        score["raw average of all"] = (X.loc[t, "mean"] - y[t]).abs().mean()
        score["YOUR BLEND"] = (pd.Series(m.predict(X[t]), index=y[t].index) - y[t]).abs().mean()
        print(f"{tgt} -- mean absolute error (lower is better)")
        for k, v in sorted(score.items(), key=lambda kv: kv[1]):
            print(f"  {v:5.2f}  {k}")
        print()
        models[tgt] = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05).fit(X[ok], y[ok])

    X, y = precip_features(df), df["wet"]
    ok = y.notna()
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05)
    clf.fit(X[ok & ~test], y[ok & ~test].astype(int))
    t = ok & test
    yt = y[t].astype(float)
    raw = ((X.loc[t, "frac_wet"] - yt) ** 2).mean()
    blend = ((clf.predict_proba(X[t])[:, 1] - yt) ** 2).mean()
    print("rain chance per hour -- Brier score (lower is better)")
    print(f"  {blend:.4f}  YOUR BLEND\n  {raw:.4f}  raw: share of models showing rain\n")
    models["wet"] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05).fit(X[ok], y[ok].astype(int))

    with open("models.pkl", "wb") as f:
        pickle.dump(models, f)
    print("saved models.pkl (retrained on all data)")


if __name__ == "__main__":
    main()
