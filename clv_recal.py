#!/usr/bin/env python3
"""
clv_recal.py — two model-honesty tools, run on the real-odds backtest set.

  (1) Closing-line value (CLV): the fast-converging edge signal. For every bet
      we would have placed (positive edge at the OPENING consensus price), we
      compare the price we got to the CLOSING consensus price. Beating the close
      predicts long-run profit in far fewer bets than raw ROI.

  (2) Edge recalibration: the model's value picks are, by selection, where it is
      most overconfident (winner's curse). We fit a Platt calibrator on the
      TRAIN half (p_cal = sigmoid(a*logit(p)+b)) and check, out-of-sample on the
      TEST half, whether calibrated probabilities are (a) better calibrated
      (Brier) and (b) better bets (ROI + CLV). A simple edge SHADE knob is shown
      alongside as a blunter, more conservative alternative.

Pure stdlib. Usage:
    python3 clv_recal.py --odds ~/Downloads/mlb_odds_dataset.json \
        --start 2025-06-01 --end 2025-08-16 --split 2025-07-21 --out clv_recal.txt
"""

import json
import math
import argparse

import bet_backtest as B


# ---------- small math helpers (no numpy) ------------------------------------
def _clip(p, lo=1e-6, hi=1 - 1e-6):
    return max(lo, min(hi, p))


def logit(p):
    p = _clip(p)
    return math.log(p / (1 - p))


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def platt_fit(ps, ys, iters=4000, lr=0.05):
    """Fit p_cal = sigmoid(a*logit(p)+b) by gradient descent on log-loss."""
    xs = [logit(p) for p in ps]
    a, b = 1.0, 0.0
    n = len(xs)
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in zip(xs, ys):
            pc = sigmoid(a * x + b)
            d = pc - y                # dLoss/dz
            ga += d * x
            gb += d
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


def apply_platt(p, ab):
    a, b = ab
    return sigmoid(a * logit(p) + b)


def brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)


def reliability(ps, ys, edges=(0.5, 0.55, 0.6, 0.65, 0.7, 1.01)):
    """Bin by predicted prob; return (lo, hi, n, mean_pred, emp_rate)."""
    lo = 0.5
    out = []
    for hi in edges:
        idx = [i for i, p in enumerate(ps) if lo <= p < hi]
        if idx:
            mp = sum(ps[i] for i in idx) / len(idx)
            er = sum(ys[i] for i in idx) / len(idx)
            out.append((lo, hi, len(idx), mp, er))
        lo = hi
    return out


# ---------- CLV --------------------------------------------------------------
def clv_report(bets, label, p_key="p_pick", edge_min=0.0):
    """CLV on bets we'd place: positive edge at OPEN cons, measured vs CLOSE cons."""
    got = beat = 0
    clv_prob_sum = clv_ret_sum = 0.0
    for b in bets:
        od = b["prices"][("open", "cons")]
        cd = b["prices"][("close", "cons")]
        if not od or not cd:
            continue
        if b[p_key] - (1.0 / od) < edge_min:   # only bets we'd actually make
            continue
        got += 1
        if od > cd:                            # we got longer odds than close
            beat += 1
        clv_prob_sum += (1.0 / cd) - (1.0 / od)   # probability points gained
        clv_ret_sum += (od / cd) - 1.0            # return CLV
    if not got:
        return f"  {label:26} no qualifying bets"
    return (f"  {label:26} {got:4d} bets  beat-close {beat}/{got} "
            f"({100*beat/got:4.1f}%)  avg CLV {100*clv_prob_sum/got:+.2f}pp "
            f"/ {100*clv_ret_sum/got:+.2f}% return")


# ---------- betting eval with an arbitrary probability ------------------------
def eval_prob(bets, ab, shade, line, price, edge_min=0.0):
    """Recompute edge with (optionally) calibrated + shaded prob, then bet."""
    n = wins = 0
    units = 0.0
    for b in bets:
        dec = b["prices"][(line, price)]
        if not dec:
            continue
        p = b["p_pick"] if ab is None else _fold(b, ab)
        implied = 1.0 / dec
        edge = (p - implied) * shade
        if edge < edge_min:
            continue
        n += 1
        if b["won"]:
            wins += 1
            units += dec - 1
        else:
            units -= 1
    roi = units / n if n else 0.0
    return {"n": n, "wins": wins, "units": units, "roi": roi}


def _fold(b, ab):
    """Calibrate on the HOME prob then fold back to the pick side."""
    p_home = b["p_pick"] if b["pick_home"] else 1 - b["p_pick"]
    p_home_cal = apply_platt(p_home, ab)
    return p_home_cal if b["pick_home"] else 1 - p_home_cal


def fmt(r, label):
    return (f"  {label:34} {r['n']:4d} bets  "
            f"W {r['wins']}/{r['n']} ({100*r['wins']/max(r['n'],1):4.1f}%)  "
            f"units {r['units']:+7.2f}  ROI {r['roi']*100:+5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--split", default="2025-07-21")
    ap.add_argument("--shade", type=float, default=0.5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print("Loading predictions + odds ...", flush=True)
    preds = B.get_predictions(args.start, args.end)
    odds = json.load(open(args.odds))
    bets = B.build_bets(preds, odds, B.team_abbr_map())
    train = [b for b in bets if b["date"] < args.split]
    test = [b for b in bets if b["date"] >= args.split]

    L = []
    def out(s=""):
        print(s, flush=True)
        L.append(s)

    out("=" * 82)
    out(f" CLV + RECALIBRATION   ({len(bets)} bets; train {len(train)} | "
        f"test {len(test)}; split {args.split})")
    out("=" * 82)

    # ---- (1) CLV -----------------------------------------------------------
    out("\n(1) CLOSING-LINE VALUE  — bet at OPEN cons where edge>0, compare to CLOSE cons")
    out("    beat-close >50%% and positive avg CLV = genuine, fast-converging edge signal.")
    out(clv_report(bets, "All value picks (full)"))
    out(clv_report(train, "  train half"))
    out(clv_report(test, "  test half (OOS)"))
    out(clv_report(bets, "Stronger picks (edge>=3%)", edge_min=0.03))

    # ---- (2) recalibration -------------------------------------------------
    out("\n(2) EDGE RECALIBRATION  — Platt fit on TRAIN home-prob, checked OOS on TEST")
    ph_tr = [b["p_pick"] if b["pick_home"] else 1 - b["p_pick"] for b in train]
    yh_tr = [1 if b["home_win"] else 0 for b in train]
    ph_te = [b["p_pick"] if b["pick_home"] else 1 - b["p_pick"] for b in test]
    yh_te = [1 if b["home_win"] else 0 for b in test]
    ab = platt_fit(ph_tr, yh_tr)
    out(f"    fitted:  p_cal = sigmoid({ab[0]:.3f}*logit(p) + {ab[1]:+.3f})   "
        f"(a<1 => shrink toward 50%, i.e. model is overconfident)")

    ph_te_cal = [apply_platt(p, ab) for p in ph_te]
    out(f"    Brier on TEST   raw {brier(ph_te, yh_te):.4f}  ->  "
        f"calibrated {brier(ph_te_cal, yh_te):.4f}  (lower = better)")

    out("\n    Reliability on TEST (predicted vs actual home-win rate):")
    out("      bin            n   pred%   RAW actual%   CAL actual-vs-pred")
    rel_raw = reliability(ph_te, yh_te)
    for lo, hi, n, mp, er in rel_raw:
        cp = sum(apply_platt(p, ab) for p in ph_te if lo <= p < hi) / n
        out(f"      {lo:.2f}-{hi:.2f}  {n:4d}  {100*mp:5.1f}   {100*er:6.1f}"
            f"        cal pred {100*cp:5.1f}")

    # ---- (2b) does it bet better? -----------------------------------------
    out("\n    Effect on VALUE BETTING (open cons, edge>0), out-of-sample TEST half:")
    out(fmt(eval_prob(test, None, 1.0, "open", "cons"), "raw prob (baseline)"))
    out(fmt(eval_prob(test, ab, 1.0, "open", "cons"), "calibrated prob"))
    out(fmt(eval_prob(test, None, args.shade, "open", "cons"),
            f"raw + {args.shade:g}x edge shade"))
    out(fmt(eval_prob(test, ab, args.shade, "open", "cons"),
            f"calibrated + {args.shade:g}x shade"))

    out("\n    Same, FULL range (bigger sample, in+out of sample mixed):")
    out(fmt(eval_prob(bets, None, 1.0, "open", "cons"), "raw prob (baseline)"))
    out(fmt(eval_prob(bets, ab, 1.0, "open", "cons"), "calibrated prob"))
    out(fmt(eval_prob(bets, ab, args.shade, "open", "cons"),
            f"calibrated + {args.shade:g}x shade"))

    # CLV under calibration (does calibrating change WHICH bets we take, for the better?)
    out("\n    CLV of calibrated value picks (open cons, cal edge>0):")
    def clv_cal(subset, label, ab, edge_min=0.0):
        got = beat = 0; cp = 0.0
        for b in subset:
            od = b["prices"][("open", "cons")]; cd = b["prices"][("close", "cons")]
            if not od or not cd:
                continue
            p = _fold(b, ab)
            if p - 1.0/od < edge_min:
                continue
            got += 1
            if od > cd: beat += 1
            cp += (1.0/cd) - (1.0/od)
        if not got:
            return f"  {label:26} none"
        return (f"  {label:26} {got:4d} bets  beat-close {beat}/{got} "
                f"({100*beat/got:4.1f}%)  avg CLV {100*cp/got:+.2f}pp")
    out(clv_cal(bets, "calibrated (full)", ab))
    out(clv_cal(test, "calibrated (test OOS)", ab))

    out("\n" + "=" * 82)
    out(" READING IT: CLV beat-close% > 50 and positive avg CLV is the real signal;")
    out(" it stabilises in ~100s of bets where ROI needs 1000s. If calibrated ROI/CLV")
    out(" >= raw, ship the calibrator. Fitted (a,b) go into odds.py. Not betting advice.")
    out("=" * 82)

    # persist fitted constants for the live site to import
    json.dump({"platt_a": ab[0], "platt_b": ab[1], "shade": args.shade,
               "fit_train_n": len(train), "fit_span": [args.start, args.split]},
              open("calibration.json", "w"), indent=2)
    out("\n wrote calibration.json (platt_a, platt_b, shade) for odds.py to load.")

    if args.out:
        open(args.out, "w").write("\n".join(L) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
