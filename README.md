# Rainfall

Belleville, IL weather: a learned blend of GFS, ECMWF, ECMWF-AIFS, ICON, GEM, UKMO and NBM,
trained on Scott AFB (KBLV) observations, with radar, futurecast, hourly and 7-day views.

- `train.py` downloads past forecasts + observations and trains `models.pkl` (also runs monthly in Actions).
- `update.py` builds `site/data.js` from live model runs (runs hourly in Actions, deployed to GitHub Pages).
- Locally: `Rainfall.bat` refreshes, serves and opens the app.
