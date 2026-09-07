#!/usr/bin/env python3
"""
track.py — build the model's track record from graded final results.

The prediction model is deterministic given point-in-time data, so its record
is reproducible rather than something we must log live. This reads the
reconstructed prediction cache (predictions_cache.json — each entry carries the
model's pre-game P(home win) and the ACTUAL final score from MLB), grades every
game, and writes a committed record.json the website displays.

    python3 track.py                 # rebuild record.json from the cache
    python3 track.py --extend A B    # reconstruct+grade new dates A..B, then rebuild

record.json is durable (committed to git); the site reads it read-only.
"""

import sys
import json
import math
from datetime import datetime

CACHE = "predictions_cache.json"
OUT = "record.json"


def grade():
    try:
        cache = json.load(open(CACHE))
    except FileNotFoundError:
        print(f"{CACHE} not found — run the backtest first to populate it.")
        return None

    rows = []  # (date, p_home, home_win, brier_ok)
    for k, v in cache.items():
        parts = k.split("|")
        if len(parts) != 5:
            continue
        date, _hid, _aid, hs, as_ = parts
        try:
            hs, as_ = int(hs), int(as_)
        except ValueError:
            continue
        p_home = float(v[0])
        home_win = 1 if hs > as_ else 0
        rows.append((date, p_home, home_win))
    if not rows:
        return None

    n = len(rows)
    wins = sum(1 for _, p, y in rows if (p >= 0.5) == (y == 1))
    brier = sum((p - y) ** 2 for _, p, y in rows) / n
    home_rate = sum(y for _, _, y in rows) / n

    # Calibration buckets
    bins = {}
    for _, p, y in rows:
        b = min(9, int(p * 10))
        bins.setdefault(b, []).append((p, y))
    calib = []
    for b in sorted(bins):
        arr = bins[b]
        if len(arr) >= 10:
            calib.append({"bucket": f"{b*10}-{b*10+10}%", "n": len(arr),
                          "pred": round(sum(p for p, _ in arr) / len(arr) * 100, 1),
                          "actual": round(sum(y for _, y in arr) / len(arr) * 100, 1)})

    # Monthly
    months = {}
    for date, p, y in rows:
        m = date[:7]
        months.setdefault(m, []).append((p, y))
    monthly = []
    for m in sorted(months):
        arr = months[m]
        w = sum(1 for p, y in arr if (p >= 0.5) == (y == 1))
        monthly.append({"month": m, "games": len(arr),
                        "record": f"{w}-{len(arr)-w}",
                        "acc": round(w / len(arr) * 100, 1)})

    dates = sorted({d for d, _, _ in rows})
    rec = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "span": f"{dates[0]} .. {dates[-1]}",
        "games": n, "wins": wins, "losses": n - wins,
        "accuracy": round(wins / n * 100, 1),
        "brier": round(brier, 4),
        "home_win_rate": round(home_rate * 100, 1),
        "pick_home_baseline": round(max(home_rate, 1 - home_rate) * 100, 1),
        "calibration": calib, "monthly": monthly,
    }
    json.dump(rec, open(OUT, "w"), indent=2)
    print(f"Wrote {OUT}: {rec['wins']}-{rec['losses']} ({rec['accuracy']}%) "
          f"on {n} games, Brier {rec['brier']} [{rec['span']}]")
    return rec


def extend(start, end):
    """Reconstruct + grade new dates via the backtest, then rebuild record.json."""
    import bet_backtest
    print(f"Reconstructing predictions {start}..{end} (this is the slow part) ...")
    bet_backtest.get_predictions(start, end)  # appends to predictions_cache.json
    grade()


def main():
    a = sys.argv[1:]
    if a and a[0] == "--extend" and len(a) >= 3:
        extend(a[1], a[2])
    else:
        grade()


if __name__ == "__main__":
    main()
