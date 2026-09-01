#!/usr/bin/env python3
"""
app.py — MLB Schedule & Probable Pitchers web app (Flask).

A deployable front end around mlb_schedule.generate_page(). It serves the same
rich, multi-source page (records, advanced pitcher metrics, bullpen fatigue,
BvP, and the win-probability model) with an on-page date picker.

Freshness strategy:
  - Today's page is pre-built on startup and refreshed every REFRESH_SECONDS
    in a background thread, so visitors get an instant, cached page.
  - Any other date builds on first request, then is cached (CACHE_TTL).

Run locally:
    python3 app.py                       # http://localhost:8000
    PORT=5000 python3 app.py

Production (what the Procfile uses):
    gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT

Keep it to ONE worker: the in-memory cache and the background refresher live in
the process, so a single worker with threads avoids duplicate rebuilds.
"""

import os
import time
import threading
from datetime import date

from flask import Flask, request, Response, redirect

from mlb_schedule import generate_page, valid_day

app = Flask(__name__)

REFRESH_SECONDS = int(os.environ.get("MLB_REFRESH_SECONDS", "900"))  # 15 min
_prewarm_started = False
_prewarm_lock = threading.Lock()


def _prewarm_loop():
    """Continuously keep today's page warm in the cache."""
    while True:
        today = date.today().isoformat()
        try:
            print(f"[prewarm] rebuilding {today} ...", flush=True)
            generate_page(today, force=True)
            print(f"[prewarm] {today} ready.", flush=True)
        except Exception as e:  # noqa: BLE001 - keep the loop alive
            print(f"[prewarm] error: {e}", flush=True)
        time.sleep(REFRESH_SECONDS)


def _start_prewarm():
    """Start the background refresher once (idempotent)."""
    global _prewarm_started
    with _prewarm_lock:
        if _prewarm_started:
            return
        _prewarm_started = True
        threading.Thread(target=_prewarm_loop, daemon=True).start()


@app.before_request
def _ensure_prewarm():
    # Kick off the refresher on the first real request rather than at import,
    # so merely importing the module has no side effects.
    if not _prewarm_started:
        _start_prewarm()


@app.route("/")
def index():
    day = request.args.get("date") or date.today().isoformat()
    if not valid_day(day):
        return redirect("/")
    force = request.args.get("refresh", "") in ("1", "true", "yes")
    try:
        html = generate_page(day, force=force)
    except Exception as e:  # noqa: BLE001
        return Response(f"<h1>Error building {day}</h1><pre>{e}</pre>",
                        status=500, mimetype="text/html")
    return Response(html, mimetype="text/html")


@app.route("/healthz")
def healthz():
    return Response("ok", mimetype="text/plain")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Serving on http://localhost:{port}  (Ctrl+C to stop)")
    _start_prewarm()
    app.run(host="0.0.0.0", port=port)
