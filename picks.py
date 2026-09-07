#!/usr/bin/env python3
"""
picks.py — maintain the daily-picks record (picks_log.json).

The website already logs its picks and grades them while building pages, but on
an ephemeral host (e.g. Render free tier) that file resets on redeploy. Run this
locally or from a scheduled job to keep a DURABLE, git-committed record:

    python3 picks.py            # log today's picks + grade everything pending
    python3 picks.py 2025-08-15 # log a specific day's picks + grade
    python3 picks.py grade      # only grade already-logged games now Final

Then commit picks_log.json so the deployed site shows the up-to-date record:
    git add picks_log.json && git commit -m "update picks" && git push
"""

import sys
from datetime import date

import mlb_schedule as M


def main():
    a = sys.argv[1:]
    if a and a[0] == "grade":
        log = M.grade_pending_picks()
        graded = [g for e in log.values() for g in e["games"] if g["result"]]
        w = sum(1 for g in graded if g["result"] == "win")
        print(f"Graded {len(graded)} games — record {w}-{len(graded)-w}")
        return
    day = a[0] if a else date.today().isoformat()
    print(f"Logging + grading picks for {day} ...")
    M.generate_page(day)   # side effect: logs the day's picks and grades finals
    log = M._load_picks()
    graded = [g for e in log.values() for g in e["games"] if g["result"]]
    w = sum(1 for g in graded if g["result"] == "win")
    print(f"Done. Overall graded record: {w}-{len(graded)-w} "
          f"across {len(log)} day(s). Commit picks_log.json to persist.")


if __name__ == "__main__":
    main()
