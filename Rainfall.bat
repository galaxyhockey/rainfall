@echo off
rem Refresh the forecast, start the local server, open the app.
cd /d "%~dp0"
python update.py
start "Rainfall server" /min python -m http.server 8765 --directory site
start "" http://localhost:8765/
