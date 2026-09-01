#!/usr/bin/env python3
"""
model.py — MLB game prediction model (shared by the live app and the backtest).

Design goals addressed here:
  1. Handedness-adjusted offense (team vs RHP / vs LHP), not just runs/game.
  2. Offensive QUALITY via wOBA / ISO / OPS (wRC+ estimate), not raw R/G.
  3. Bullpen modeled as QUALITY x AVAILABILITY (fatigue-weighted true talent),
     not saves/holds as the quality proxy.
  4. Expected starter innings: a 6.5-IP arm shifts more weight onto the
     starter and less onto the bullpen than a 4.5-IP arm.
  5. Park factors (Coors != Oracle).
  6. A real run DISTRIBUTION (negative binomial, overdispersed) rather than a
     single deterministic expected-runs number -> win prob + score dist.

Everything here is pure/deterministic given its inputs, so the backtest can feed
it as-of-date features and reproduce exactly what the live app would have shown.
"""

import math

# --- League run-scoring park factors (100 = neutral), keyed by home team id ---
# Approximate multi-year run park factors. Coors is the extreme outlier; pitcher
# parks (Oracle, Petco, T-Mobile) suppress. Neutral fallback = 100.
PARK_FACTORS = {
    115: 112,  # COL Coors
    111: 108,  # BOS Fenway
    113: 105,  # CIN Great American
    118: 104,  # KC Kauffman
    143: 104,  # PHI Citizens Bank
    108: 103,  # LAA Angel Stadium
    109: 102,  # ARI Chase
    140: 101,  # TEX Globe Life
    110: 101,  # BAL Camden
    141: 101,  # TOR Rogers Centre
    142: 101,  # MIN Target
    120: 101,  # WSH Nationals Park
    144: 101,  # ATL Truist
    112: 101,  # CHC Wrigley
    117: 100,  # HOU Minute Maid
    147: 100,  # NYY Yankee Stadium
    145: 100,  # CWS Rate Field
    158: 100,  # MIL American Family
    119:  99,  # LAD Dodger Stadium
    116:  98,  # DET Comerica
    138:  99,  # STL Busch
    134:  99,  # PIT PNC
    114:  99,  # CLE Progressive
    121:  97,  # NYM Citi Field
    146:  97,  # MIA loanDepot
    139:  97,  # TB (Tropicana / Steinbrenner)
    133:  97,  # ATH / OAK
    135:  96,  # SD Petco
    136:  93,  # SEA T-Mobile
    137:  93,  # SF Oracle
}

# --- wOBA linear weights (approx. recent-seasons values) ---
WOBA_W = {"bb": 0.69, "hbp": 0.72, "b1": 0.88, "b2": 1.25, "b3": 1.58, "hr": 2.03}
WOBA_SCALE = 1.15

# --- Model configuration (tunable; the backtest can sweep these) ---
DEFAULT_CFG = {
    "hfa_runs": 0.18,        # home-field edge, in runs added to the home mean
    "dispersion": 1.9,       # variance/mean ratio for team runs (MLB ~1.8-2.2)
    "sp_weight_cap": (3.5, 7.0),  # clamp projected starter innings
    "max_runs": 26,          # support of the run distribution
    "park_strength": 1.0,    # 0..1 scaling of park effect
    "clamp": (0.03, 0.97),   # win-prob clamp
}


def park_factor(home_team_id):
    return PARK_FACTORS.get(home_team_id, 100)


# ---------------------------------------------------------------------------
# Offense: wOBA / ISO from a counting-stat block
# ---------------------------------------------------------------------------

def _num(d, key):
    try:
        return float(d.get(key))
    except (TypeError, ValueError):
        return 0.0


def calc_woba(stat):
    """wOBA from a hitting stat block (StatsAPI hitting split/season)."""
    ab = _num(stat, "atBats")
    bb = _num(stat, "baseOnBalls")
    ibb = _num(stat, "intentionalWalks")
    hbp = _num(stat, "hitByPitch")
    sf = _num(stat, "sacFlies")
    h = _num(stat, "hits")
    d2 = _num(stat, "doubles")
    t3 = _num(stat, "triples")
    hr = _num(stat, "homeRuns")
    b1 = max(0.0, h - d2 - t3 - hr)
    ubb = max(0.0, bb - ibb)
    denom = ab + ubb + sf + hbp
    if denom <= 0:
        return None
    num = (WOBA_W["bb"] * ubb + WOBA_W["hbp"] * hbp + WOBA_W["b1"] * b1
           + WOBA_W["b2"] * d2 + WOBA_W["b3"] * t3 + WOBA_W["hr"] * hr)
    return num / denom


def ip_to_float(ip):
    """Convert baseball innings '68.1'/'68.2' (thirds) to a true float."""
    try:
        s = str(ip)
        if "." in s:
            whole, frac = s.split(".")
            return int(whole) + {"0": 0, "1": 1 / 3, "2": 2 / 3}.get(frac, 0)
        return float(ip)
    except (TypeError, ValueError):
        return 0.0


def calc_fip(stat, fip_constant=3.10):
    """FIP from counting stats — used to reconstruct as-of-date pitcher talent."""
    ip = ip_to_float(stat.get("inningsPitched"))
    if ip <= 0:
        return None
    hr = _num(stat, "homeRuns")
    bb = _num(stat, "baseOnBalls")
    hbp = _num(stat, "hitByPitch")
    k = _num(stat, "strikeOuts")
    return round((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + fip_constant, 2)


def calc_iso(stat):
    slg = stat.get("slg")
    avg = stat.get("avg")
    try:
        return round(float(slg) - float(avg), 3)
    except (TypeError, ValueError):
        return None


def wrc_plus_est(woba, lg_woba, pf, park_strength=1.0):
    """A wRC+ ESTIMATE (100 = league avg). Park-adjusted, league-relative."""
    if not woba or not lg_woba:
        return None
    # runs-above-average per PA, normalized to league, then park-adjusted.
    wraa_pa = (woba - lg_woba) / WOBA_SCALE
    base = lg_woba / WOBA_SCALE  # rough runs/PA scale anchor
    pf_adj = 1 + (pf / 100.0 - 1) * park_strength
    val = (base + wraa_pa) / (base * pf_adj)
    return round(val * 100)


# ---------------------------------------------------------------------------
# Pitching: starter true-talent + bullpen (quality x availability)
# ---------------------------------------------------------------------------

# Availability weight by fatigue rank (1=rested .. 4=down). Availability decides
# how much an arm's QUALITY counts — this is the quality/availability split.
FAT_WEIGHT = {1: 1.0, 2: 0.72, 3: 0.42, 4: 0.15, 0: 1.0}


def _blend(pairs, default):
    num = den = 0.0
    for v, w in pairs:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        num += fv * w
        den += w
    return num / den if den else default


def pitcher_true_talent(era, fip, xfip, xera, lg_era):
    """Run-prevention talent per 9, weighting predictive (expected) metrics more."""
    return _blend([(era, 0.20), (fip, 0.30), (xfip, 0.25), (xera, 0.25)], lg_era)


def bullpen_run_prevention(arms, lg_era):
    """
    arms: list of {"talent": ra_per9, "rank": fatigue_rank}. Returns the
    AVAILABILITY-WEIGHTED average QUALITY of the pen (quality x availability),
    with no saves/holds involved.
    """
    if not arms:
        return lg_era + 0.20
    num = den = 0.0
    for a in arms:
        w = FAT_WEIGHT.get(a.get("rank", 0), 0.5)
        t = a.get("talent")
        if t is None:
            continue
        num += t * w
        den += w
    return num / den if den else lg_era + 0.20


# ---------------------------------------------------------------------------
# Negative-binomial run distribution
# ---------------------------------------------------------------------------

def _nb_pmf(mean, dispersion, kmax):
    """Return a list p[0..kmax] for NB with given mean and var = dispersion*mean.

    Falls back to Poisson when dispersion<=1. Overdispersion (dispersion>1)
    captures the fat tail of real baseball scoring.
    """
    mean = max(0.05, float(mean))
    if dispersion <= 1.0 + 1e-9:
        # Poisson
        p = []
        logm = math.log(mean)
        for k in range(kmax + 1):
            p.append(math.exp(k * logm - mean - math.lgamma(k + 1)))
    else:
        var = dispersion * mean
        r = mean * mean / (var - mean)          # NB "size"
        prob = r / (r + mean)                    # success prob
        p = []
        for k in range(kmax + 1):
            logpmf = (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
                      + r * math.log(prob) + k * math.log(1 - prob))
            p.append(math.exp(logpmf))
    s = sum(p)
    return [x / s for x in p] if s else p


def _winprob_from_dists(ph, pa, tie_home_edge):
    """P(home wins), P(away wins), P(extra innings) from two run pmfs."""
    p_home = p_away = p_tie = 0.0
    for i, phi in enumerate(ph):
        for j, paj in enumerate(pa):
            m = phi * paj
            if i > j:
                p_home += m
            elif j > i:
                p_away += m
            else:
                p_tie += m
    # Ties go to extra innings; split by a small edge toward the stronger team.
    p_home += p_tie * tie_home_edge
    p_away += p_tie * (1 - tie_home_edge)
    return p_home, p_away, p_tie


# ---------------------------------------------------------------------------
# Feature -> lambda -> prediction
# ---------------------------------------------------------------------------

def _project_ip(ip, gs, cfg):
    lo, hi = cfg["sp_weight_cap"]
    try:
        ip = float(ip)
        gs = float(gs)
    except (TypeError, ValueError):
        return 4.5
    if gs <= 0:
        return 4.5
    return min(hi, max(lo, ip / gs))


def team_def_ra(feat, lg_r, cfg):
    """Runs-allowed rate per 9 for a team this game: starter+bullpen, IP-weighted."""
    sp_ra = feat.get("sp_ra") or (lg_r + 0.25)
    pen_ra = feat.get("pen_ra") or (lg_r + 0.20)
    proj_ip = feat.get("proj_ip") or 4.5
    sp_frac = min(1.0, proj_ip / 9.0)
    pen_frac = 1.0 - sp_frac
    pitch = sp_frac * sp_ra + pen_frac * pen_ra
    # Ground it 60/40 in the team's own season runs-allowed for stability.
    rapg = feat.get("rapg") or lg_r
    return 0.6 * pitch + 0.4 * rapg


def predict(home, away, ctx, cfg=None):
    """
    home/away feature dicts:
      off_woba      : team wOBA vs the OPPOSING starter's hand
      lg_off_woba   : league wOBA vs that hand
      sp_ra, proj_ip, pen_ra, rapg : run-prevention pieces (per 9 / innings)
    ctx: {lg_r, park_factor, ...}
    Returns win probs, projected runs, and a small score distribution.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    lg_r = ctx.get("lg_r") or 4.4
    pf = ctx.get("park_factor", 100)
    pf_mult = 1 + (pf / 100.0 - 1) * cfg["park_strength"]

    def off_mult(feat):
        w = feat.get("off_woba")
        lw = feat.get("lg_off_woba") or ctx.get("lg_woba")
        if w and lw:
            # wOBA ratio, softened so extreme splits don't explode.
            return max(0.6, min(1.6, 1 + 0.9 * (w / lw - 1)))
        # fall back to team RS/G ratio if wOBA unavailable
        rspg = feat.get("rspg")
        return max(0.6, min(1.6, (rspg / lg_r))) if rspg else 1.0

    def_home = team_def_ra(home, lg_r, cfg)
    def_away = team_def_ra(away, lg_r, cfg)

    # Expected runs = league base * own offense * opponent defense * park.
    lam_home = lg_r * off_mult(home) * (def_away / lg_r) * pf_mult
    lam_away = lg_r * off_mult(away) * (def_home / lg_r) * pf_mult
    lam_home = max(1.2, lam_home) + cfg["hfa_runs"]
    lam_away = max(1.2, lam_away)

    kmax = cfg["max_runs"]
    disp = cfg["dispersion"]
    ph = _nb_pmf(lam_home, disp, kmax)
    pa = _nb_pmf(lam_away, disp, kmax)

    tie_edge = lam_home / (lam_home + lam_away)
    p_home, p_away, p_tie = _winprob_from_dists(ph, pa, tie_edge)

    lo, hi = cfg["clamp"]
    p_home = min(hi, max(lo, p_home))
    p_away = 1 - p_home

    gap = abs(p_home - 0.5)
    conf = ("Toss-up" if gap < 0.06 else "Lean" if gap < 0.14
            else "Edge" if gap < 0.24 else "Strong")

    # Distribution of total runs (for over/under context).
    total_dist = [0.0] * (2 * kmax + 1)
    for i, phi in enumerate(ph):
        for j, paj in enumerate(pa):
            total_dist[i + j] += phi * paj
    exp_total = sum(t * p for t, p in enumerate(total_dist))

    return {
        "p_home": round(p_home * 100), "p_away": round(p_away * 100),
        "e_home": round(lam_home, 2), "e_away": round(lam_away, 2),
        "exp_total": round(exp_total, 1),
        "p_tie": round(p_tie * 100, 1), "conf": conf,
        "fav_home": p_home >= 0.5,
        "park_factor": pf,
        "def_home": round(def_home, 2), "def_away": round(def_away, 2),
        "off_home": round(off_mult(home), 3), "off_away": round(off_mult(away), 3),
        "p_home_raw": p_home,   # unrounded, for scoring/backtest
    }
