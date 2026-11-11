"""
pre_ufc.py -- Features from a fighter's bouts outside the UFC: the Contender Series
(with stats, first half of this module) and the rest of the pro career (results only,
second half).

Source: raw_data/espn_dwcs_fights.csv + raw_data/espn_fighter_map.csv (scripts/scrape_espn.py).
DWCS stats come from the same official feed as UFCStats (identical numbers on UFC
fights), so the per-fight columns mean the same thing as fight_stats.

Kept separate from the UFC career stats (splm, sapm, ...): DWCS opponents are
prospects, not UFC fighters, and the model should learn how much a Contender Series
performance is worth instead of having it silently averaged into the UFC record.

Rates are shrunk toward frozen league values (UFCStats 2015+) with
RATE_PSEUDO_MIN pseudo-minutes, so a fighter without DWCS bouts sits at the league
average and a pair of non-DWCS fighters gives a 0 diff.

Training and inference share prefight_features() / features_before(): the only
input is the bout list and a cutoff date (bouts strictly before it).
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DWCS_CSV = ROOT_DIR / "raw_data" / "espn_dwcs_fights.csv"
MAP_CSV = ROOT_DIR / "raw_data" / "espn_fighter_map.csv"

# UFCStats 2015+ league rates, frozen so training and inference shrink toward the same values
LEAGUE_SLPM = 3.83
LEAGUE_STR_ACC = 0.459
LEAGUE_TD15 = 1.40
LEAGUE_CTRL_SHARE = 0.19
RATE_PSEUDO_MIN = 15.0   # one 3-round fight
ACC_PSEUDO_ATT = 30.0

COUNT_FEATURES = ("dwcs_fights", "dwcs_wins", "dwcs_finish_wins")
RATE_FEATURES = ("dwcs_splm", "dwcs_sapm", "dwcs_td_avg", "dwcs_str_acc", "dwcs_ctrl_share")
FEATURES = COUNT_FEATURES + RATE_FEATURES

DEFAULTS = {
    "dwcs_fights": 0.0, "dwcs_wins": 0.0, "dwcs_finish_wins": 0.0,
    "dwcs_splm": LEAGUE_SLPM, "dwcs_sapm": LEAGUE_SLPM, "dwcs_td_avg": LEAGUE_TD15,
    "dwcs_str_acc": LEAGUE_STR_ACC, "dwcs_ctrl_share": LEAGUE_CTRL_SHARE,
}


def load_bouts(dwcs_csv: Path = DWCS_CSV, map_csv: Path = MAP_CSV) -> pd.DataFrame:
    """One row per (fighter, DWCS bout) with the UFCStats fighter_id and the opponent's strikes.

    Fighters who never reached the UFC have no UFCStats id and are dropped -- they can't
    appear in a training row or a prediction. Empty frame if the CSVs are missing.
    """
    cols = ["fighter_id", "date", "result", "method", "dur", "sl", "sa", "tl", "ctrl", "o_sl"]
    if not dwcs_csv.exists() or not map_csv.exists():
        return pd.DataFrame(columns=cols)
    ids = {"espn_id": str, "opp_espn_id": str, "competition_id": str}
    d = pd.read_csv(dwcs_csv, dtype=ids)
    fmap = pd.read_csv(map_csv, dtype={"espn_id": str})
    opp = d[["competition_id", "espn_id", "sig_str_landed"]].rename(
        columns={"espn_id": "opp_espn_id", "sig_str_landed": "o_sl"})
    d = d.merge(opp, on=["competition_id", "opp_espn_id"], how="left")
    d = d.merge(fmap[["espn_id", "fighter_id"]], on="espn_id", how="inner")
    out = pd.DataFrame({
        "fighter_id": d["fighter_id"], "date": d["date"].astype(str).str[:10],
        "result": d["result"], "method": d["method"].fillna(""),
        "dur": ((d["finish_round"].astype(float) - 1) * 300 + d["match_time_sec"].astype(float)) / 60.0,
        "sl": d["sig_str_landed"], "sa": d["sig_str_atmpted"], "tl": d["td_landed"],
        "ctrl": d["ctrl"], "o_sl": d["o_sl"],
    })
    num = ["dur", "sl", "sa", "tl", "ctrl", "o_sl"]
    out[num] = out[num].astype(float).fillna(0.0)
    return out.sort_values(["fighter_id", "date"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _cached_bouts() -> pd.DataFrame:
    return load_bouts()


def _aggregate(b: pd.DataFrame) -> dict[str, float]:
    if b.empty:
        return dict(DEFAULTS)
    wins = b["result"] == "W"
    finish = wins & b["method"].str.contains("KO|Submission", case=False, regex=True)
    mins = b["dur"].sum()
    return {
        "dwcs_fights": float(len(b)),
        "dwcs_wins": float(wins.sum()),
        "dwcs_finish_wins": float(finish.sum()),
        "dwcs_splm": (b["sl"].sum() + RATE_PSEUDO_MIN * LEAGUE_SLPM) / (mins + RATE_PSEUDO_MIN),
        "dwcs_sapm": (b["o_sl"].sum() + RATE_PSEUDO_MIN * LEAGUE_SLPM) / (mins + RATE_PSEUDO_MIN),
        "dwcs_td_avg": (b["tl"].sum() * 15 + RATE_PSEUDO_MIN * LEAGUE_TD15) / (mins + RATE_PSEUDO_MIN),
        "dwcs_str_acc": (b["sl"].sum() + ACC_PSEUDO_ATT * LEAGUE_STR_ACC) / (b["sa"].sum() + ACC_PSEUDO_ATT),
        "dwcs_ctrl_share": (b["ctrl"].sum() / 60 + RATE_PSEUDO_MIN * LEAGUE_CTRL_SHARE) / (mins + RATE_PSEUDO_MIN),
    }


def features_before(fighter_id: str | None, before_date: str | None,
                    bouts: pd.DataFrame | None = None) -> dict[str, float]:
    """DWCS features going into a fight on *before_date* (bouts strictly before it)."""
    b = _cached_bouts() if bouts is None else bouts
    if not fighter_id or b.empty:
        return dict(DEFAULTS)
    sub = b[b["fighter_id"] == fighter_id]
    if before_date:
        sub = sub[sub["date"] < str(before_date)[:10]]
    return _aggregate(sub)


def prefight_features(pairs: pd.DataFrame, bouts: pd.DataFrame | None = None) -> pd.DataFrame:
    """features_before() for every (fighter_id, date) row of *pairs*; returns pairs + FEATURES."""
    b = _cached_bouts() if bouts is None else bouts
    by_fighter = {fid: g for fid, g in b.groupby("fighter_id")}
    empty = b.iloc[0:0]
    rows = []
    for fid, d in zip(pairs["fighter_id"], pairs["date"].astype(str).str[:10]):
        g = by_fighter.get(fid, empty)
        rows.append(_aggregate(g[g["date"] < d]))
    return pd.concat([pairs.reset_index(drop=True), pd.DataFrame(rows, columns=list(FEATURES))], axis=1)


# ==============================================================================
# Pro career outside the UFC (raw_data/espn_career_fights.csv, scripts/scrape_espn_careers.py)
# ==============================================================================
#
# Results only (ESPN has no stats outside the UFC). TUF house bouts are exhibitions and
# no contests count as neither win nor loss. Rates are shrunk toward frozen priors
# (CAREER_PSEUDO_FIGHTS pseudo-fights) so a 3-0 prospect isn't read as a 100% fighter.
# Opponent quality is NOT measured here: ESPN only exposes opponents' *current*
# records, which would leak; that needs the opponents' own dated careers (phase 2b).

CAREER_CSV = ROOT_DIR / "raw_data" / "espn_career_fights.csv"

CAREER_WIN_PRIOR = 0.83        # pooled non-UFC record of UFC fighters (ESPN, 2026-11)
CAREER_FINISH_PRIOR = 0.63     # finishing wins per fight, same pool
CAREER_PSEUDO_FIGHTS = 3.0
NO_FIGHT_DAYS = 3650.0         # "days since last bout" when there is none

CAREER_FEATURES = (
    "ext_fights", "ext_wins", "ext_losses", "ext_win_rate", "ext_finish_rate", "ext_ko_losses",
    "ext_win_streak", "ext_named_promo_fights", "pro_years", "pro_days_since_last",
)
_EXHIBITION = r"ultimate fighter"


def load_career(career_csv: Path = CAREER_CSV) -> pd.DataFrame:
    """Non-UFC pro bouts per UFCStats fighter_id; empty frame if the CSV is missing."""
    cols = ["fighter_id", "date", "league", "result", "method"]
    if not career_csv.exists():
        return pd.DataFrame(columns=cols)
    d = pd.read_csv(career_csv, dtype={"espn_id": str, "opp_espn_id": str, "competition_id": str})
    d = d[~d["event_name"].fillna("").str.contains(_EXHIBITION, case=False, regex=True)]
    d = d[d["date"].astype(str).str.len() >= 10]
    out = d[cols].copy()
    out["date"] = out["date"].astype(str).str[:10]
    out["method"] = out["method"].fillna("")
    return out.sort_values(["fighter_id", "date"]).reset_index(drop=True)


def _career_aggregate(ext: pd.DataFrame, all_dates: list[str], date: str) -> dict[str, float]:
    """*ext*: this fighter's non-UFC bouts before *date*; *all_dates*: every pro bout date
    (non-UFC, DWCS, UFC) before *date*."""
    res = ext["result"]
    wins, losses = int((res == "W").sum()), int((res == "L").sum())
    n = wins + losses + int((res == "D").sum())
    finish = (res == "W") & ext["method"].str.contains(r"KO|Submission", case=False, regex=True)
    ko_loss = (res == "L") & ext["method"].str.contains(r"KO", case=False, regex=True)
    streak = 0
    for r in reversed(res.tolist()):
        if r == "W":
            streak += 1
        elif r in ("L", "D"):
            break
    d0 = pd.Timestamp(date)
    first = pd.Timestamp(min(all_dates)) if all_dates else None
    last = pd.Timestamp(max(all_dates)) if all_dates else None
    return {
        "ext_fights": float(n),
        "ext_wins": float(wins),
        "ext_losses": float(losses),
        "ext_win_rate": (wins + CAREER_PSEUDO_FIGHTS * CAREER_WIN_PRIOR) / (n + CAREER_PSEUDO_FIGHTS),
        "ext_finish_rate": (finish.sum() + CAREER_PSEUDO_FIGHTS * CAREER_FINISH_PRIOR) / (n + CAREER_PSEUDO_FIGHTS),
        "ext_ko_losses": float(ko_loss.sum()),
        "ext_win_streak": float(streak),
        "ext_named_promo_fights": float((ext["league"] != "other").sum()),
        "pro_years": (d0 - first).days / 365.25 if first is not None else 0.0,
        "pro_days_since_last": float((d0 - last).days) if last is not None else NO_FIGHT_DAYS,
    }


def prefight_career(pairs: pd.DataFrame, career: pd.DataFrame | None = None,
                    other_bouts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Career features for every (fighter_id, date) row of *pairs* (bouts strictly before the date).

    *other_bouts* (fighter_id, date): the fighter's UFC and DWCS bouts, which count
    for pro_years / pro_days_since_last but not for the ext_* record.
    """
    career = load_career() if career is None else career
    other = other_bouts if other_bouts is not None else pd.DataFrame(columns=["fighter_id", "date"])
    need = set(pairs["fighter_id"].dropna())   # grouping every fighter dominated the live call
    career, other = career[career["fighter_id"].isin(need)], other[other["fighter_id"].isin(need)]
    ext_by = {fid: g for fid, g in career.groupby("fighter_id")}
    dates_by = {fid: sorted(set(g["date"].astype(str).str[:10]))
                for fid, g in pd.concat([career[["fighter_id", "date"]], other[["fighter_id", "date"]]]).groupby("fighter_id")}
    empty = career.iloc[0:0]
    rows = []
    for fid, d in zip(pairs["fighter_id"], pairs["date"].astype(str).str[:10]):
        g = ext_by.get(fid, empty)
        rows.append(_career_aggregate(g[g["date"] < d], [x for x in dates_by.get(fid, []) if x < d], d))
    return pd.concat([pairs.reset_index(drop=True), pd.DataFrame(rows, columns=list(CAREER_FEATURES))], axis=1)


# ==============================================================================
# Opponent quality: one rating replayed over every known pro bout (phase 2b)
# ==============================================================================
#
# Graph = the UFC fighters' non-UFC bouts (espn_career_fights.csv) + their regional
# opponents' own bouts up to their last meeting with a UFC fighter
# (espn_opponent_bouts.csv) + DWCS + UFC bouts (UFCStats). Beating opponents who beat
# good opponents is worth more -- the rating does the cross-promotion weighting, no
# hand-set promotion tiers. Nodes are UFCStats ids where the ESPN id is mapped, else
# "espn:<id>". Every value is the rating *before* the bout (no leakage).

OPP_BOUTS_CSV = ROOT_DIR / "raw_data" / "espn_opponent_bouts.csv"

MMA_ELO_START = 1400.0          # same constants as ml/ELO_calculator.py
MMA_K_PROVISIONAL, MMA_K_NORMAL, MMA_PROVISIONAL_LIMIT = 90.0, 32.0, 3
OPP_PSEUDO_FIGHTS = 3.0

OPP_FEATURES = ("ext_opp_elo", "ext_win_opp_elo", "ext_opp_win_rate", "mma_elo")


def graph_bouts(career: pd.DataFrame | None = None, ufc_bouts: pd.DataFrame | None = None,
                opp_csv: Path = OPP_BOUTS_CSV, career_csv: Path = CAREER_CSV,
                dwcs_csv: Path = DWCS_CSV, map_csv: Path = MAP_CSV) -> pd.DataFrame:
    """Every bout as (bout_id, date, a, b, winner) with winner = a / b / '' (draw; NCs dropped).

    *ufc_bouts* (fight_id, date, r_fighter_id, b_fighter_id, winner_id) come from UFCStats.
    """
    fmap = (dict(zip(*pd.read_csv(map_csv, dtype=str)[["espn_id", "fighter_id"]].T.values))
            if map_csv.exists() else {})
    node = lambda e: fmap.get(e, f"espn:{e}")  # noqa: E731
    parts = []
    if career_csv.exists():
        c = pd.read_csv(career_csv, dtype=str)
        c = c[~c["event_name"].fillna("").str.contains(_EXHIBITION, case=False, regex=True)]
        c = c[c["result"].isin(["W", "L", "D"]) & c["opp_espn_id"].notna()]
        a, b = c["espn_id"].map(node), c["opp_espn_id"].map(node)
        parts.append(pd.DataFrame({"bout_id": "c" + c["competition_id"], "date": c["date"].str[:10], "a": a, "b": b,
                                   "winner": a.where(c["result"] == "W", b.where(c["result"] == "L", ""))}))
    if opp_csv.exists():
        o = pd.read_csv(opp_csv, dtype=str)
        o = o[o["winner"] != "NC"]
        a, b = o["a_espn_id"].map(node), o["b_espn_id"].map(node)
        w = o["winner"].map(lambda x: node(x) if x not in ("D", "NC") else "")
        parts.append(pd.DataFrame({"bout_id": "c" + o["competition_id"], "date": o["date"].str[:10],
                                   "a": a, "b": b, "winner": w}))
    if dwcs_csv.exists():
        d = pd.read_csv(dwcs_csv, dtype=str)
        d = d[d["result"].isin(["W", "L", "D"])].drop_duplicates("competition_id")
        a, b = d["espn_id"].map(node), d["opp_espn_id"].map(node)
        parts.append(pd.DataFrame({"bout_id": "c" + d["competition_id"], "date": d["date"].str[:10], "a": a, "b": b,
                                   "winner": a.where(d["result"] == "W", b.where(d["result"] == "L", ""))}))
    if ufc_bouts is not None and len(ufc_bouts):
        u = ufc_bouts
        w = u["winner_id"].fillna("")
        w = w.where(w.isin(u["r_fighter_id"]) | w.isin(u["b_fighter_id"]), "")
        if "method" in u.columns:   # no winner: a decision is a draw, anything else (CNC, Overturned) a NC
            keep = (w != "") | u["method"].fillna("").str.startswith("Decision")
            u, w = u[keep], w[keep]
        parts.append(pd.DataFrame({"bout_id": "u" + u["fight_id"].astype(str), "date": u["date"].astype(str).str[:10],
                                   "a": u["r_fighter_id"], "b": u["b_fighter_id"], "winner": w}))
    g = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["bout_id", "date", "a", "b", "winner"])
    g = g[g["date"].str.len() == 10].drop_duplicates("bout_id")
    return g.sort_values(["date", "bout_id"]).reset_index(drop=True)


def replay_ratings(bouts: pd.DataFrame, ext_k_scale: float = 1.0) -> pd.DataFrame:
    """Pre-bout rating and record of both sides of every bout, in date order.

    *ext_k_scale* multiplies the K-factor of non-UFC bouts (bout_id 'c...'), so wins
    over a thin regional pool move the rating less than UFC results.

    Returns one row per (bout, side): bout_id, date, node, opp, won (1/0/0.5),
    elo (pre), opp_elo (pre), opp_w / opp_l (opponent's record before the bout), post_elo.
    """
    elo: dict[str, float] = {}
    n: dict[str, int] = {}
    rec: dict[str, list[int]] = {}
    out = []
    for bid, date, a, b, w in bouts[["bout_id", "date", "a", "b", "winner"]].itertuples(index=False):
        ra, rb = elo.get(a, MMA_ELO_START), elo.get(b, MMA_ELO_START)
        sa = 1.0 if w == a else (0.0 if w == b else 0.5)
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        scale = ext_k_scale if bid.startswith("c") else 1.0
        ka = scale * (MMA_K_PROVISIONAL if n.get(a, 0) < MMA_PROVISIONAL_LIMIT else MMA_K_NORMAL)
        kb = scale * (MMA_K_PROVISIONAL if n.get(b, 0) < MMA_PROVISIONAL_LIMIT else MMA_K_NORMAL)
        wa, la = rec.get(a, [0, 0])
        wb, lb = rec.get(b, [0, 0])
        na, nb = ra + ka * (sa - ea), rb + kb * ((1 - sa) - (1 - ea))
        out.append((bid, date, a, b, sa, ra, rb, wb, lb, na))
        out.append((bid, date, b, a, 1 - sa, rb, ra, wa, la, nb))
        elo[a], elo[b] = na, nb
        n[a], n[b] = n.get(a, 0) + 1, n.get(b, 0) + 1
        rec[a] = [wa + (sa == 1.0), la + (sa == 0.0)]
        rec[b] = [wb + (sa == 0.0), lb + (sa == 1.0)]
    return pd.DataFrame(out, columns=["bout_id", "date", "node", "opp", "won", "elo", "opp_elo",
                                      "opp_w", "opp_l", "post_elo"])


def prefight_opp_quality(pairs: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    """Opponent-quality features for every (fighter_id, date) row of *pairs*.

    ext_*: over the fighter's non-UFC bouts (bout_id 'c...': regional + DWCS) strictly
    before the date, shrunk toward MMA_ELO_START / a 50% opponent win rate with
    OPP_PSEUDO_FIGHTS. mma_elo: rating after every bout (UFC included) before the date.
    """
    r = replay[replay["node"].isin(set(pairs["fighter_id"].dropna()))].copy()
    r["ext"] = r["bout_id"].str.startswith("c")
    r["opp_wr"] = (r["opp_w"] + OPP_PSEUDO_FIGHTS * 0.5) / (r["opp_w"] + r["opp_l"] + OPP_PSEUDO_FIGHTS)
    by = {fid: g for fid, g in r.groupby("node")}
    empty = r.iloc[0:0]
    k = OPP_PSEUDO_FIGHTS
    rows = []
    for fid, d in zip(pairs["fighter_id"], pairs["date"].astype(str).str[:10]):
        g = by.get(fid, empty)
        g = g[g["date"] < d]
        e = g[g["ext"]]
        wins = e[e["won"] == 1.0]
        rows.append({
            "ext_opp_elo": (e["opp_elo"].sum() + k * MMA_ELO_START) / (len(e) + k),
            "ext_win_opp_elo": (wins["opp_elo"].sum() + k * MMA_ELO_START) / (len(wins) + k),
            "ext_opp_win_rate": (e["opp_wr"].sum() + k * 0.5) / (len(e) + k),
            "mma_elo": float(g["post_elo"].iloc[-1]) if len(g) else MMA_ELO_START,
        })
    return pd.concat([pairs.reset_index(drop=True), pd.DataFrame(rows, columns=list(OPP_FEATURES))], axis=1)


# ==============================================================================
# Model features: the record + opponent quality + MMA rating, as the models see them
# ==============================================================================
#
# The pre-UFC record says a lot about a debutant and little about a 10-fight UFC
# veteran, so the ext_* values are faded toward their no-data value as UFC fights
# accumulate: neutral + (x - neutral) * k / (k + n_ufc_fights), k = FADE_K (full weight
# at the debut, half after 2 UFC fights). Raw, they helped debut fights but hurt the
# rest; faded, they helped both (9 seeds, 2026-11). pro_years / pro_days_since_last
# are not faded. mma_elo replays every bout with the K of non-UFC bouts scaled by
# MMA_EXT_K_SCALE: at full K, unbeaten regional records over a thin pool inflated it
# (worse on debut fights); 0.25 was neutral there and kept the gain at 2-5 UFC fights.
#
# Training (scripts/add_computed_features_to_csv.py) and inference (predict.py) both
# call prefight_model_features(); at inference the UFCStats connection is restricted
# to fights before the event, and every value uses bouts strictly before the date.

FADE_K = 2.0
MMA_EXT_K_SCALE = 0.25

MODEL_FEATURES = (
    "ext_fights", "ext_wins", "ext_losses", "ext_win_rate", "ext_finish_rate", "ext_ko_losses",
    "ext_win_streak", "ext_named_promo_fights", "pro_years", "pro_days_since_last",
    "ext_opp_elo", "ext_win_opp_elo", "ext_opp_win_rate", "mma_elo",
)
# value of each feature for a fighter with no known bouts (also the fade target)
NO_DATA = {
    "ext_fights": 0.0, "ext_wins": 0.0, "ext_losses": 0.0,
    "ext_win_rate": CAREER_WIN_PRIOR, "ext_finish_rate": CAREER_FINISH_PRIOR,
    "ext_ko_losses": 0.0, "ext_win_streak": 0.0, "ext_named_promo_fights": 0.0,
    "pro_years": 0.0, "pro_days_since_last": NO_FIGHT_DAYS,
    "ext_opp_elo": MMA_ELO_START, "ext_win_opp_elo": MMA_ELO_START, "ext_opp_win_rate": 0.5,
    "mma_elo": MMA_ELO_START,
}
_FADED = tuple(f for f in MODEL_FEATURES if f.startswith("ext_"))


@lru_cache(maxsize=1)
def _cached_offline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(career, DWCS bouts, non-UFC graph bouts) from the ESPN CSVs -- static within a process."""
    return load_career(), load_bouts(), graph_bouts()


def prefight_model_features(pairs: pd.DataFrame, ufc_conn) -> pd.DataFrame:
    """MODEL_FEATURES for every (fighter_id, date) row of *pairs* (UFCStats ids).

    *ufc_conn*: the UFCStats DB (its fights table supplies the UFC bouts). Unknown or
    None fighter ids get NO_DATA, except that mma_elo still counts their UFC bouts.
    """
    career, dwcs, ext_graph = _cached_offline()
    ufc = pd.read_sql_query(
        "SELECT fight_id, date, r_fighter_id, b_fighter_id, winner_id, method FROM fights", ufc_conn)
    ufc["date"] = ufc["date"].astype(str).str[:10]
    pairs = pd.DataFrame({"fighter_id": pairs["fighter_id"].where(pairs["fighter_id"].notna(), None).values,
                          "date": pairs["date"].astype(str).str[:10].values})

    ufc_dates = pd.concat([ufc[["r_fighter_id", "date"]].rename(columns={"r_fighter_id": "fighter_id"}),
                           ufc[["b_fighter_id", "date"]].rename(columns={"b_fighter_id": "fighter_id"})])
    other = pd.concat([ufc_dates, dwcs[["fighter_id", "date"]]], ignore_index=True)
    car = prefight_career(pairs, career, other)

    graph = pd.concat([ext_graph, graph_bouts(ufc_bouts=ufc, opp_csv=Path("-"), career_csv=Path("-"),
                                              dwcs_csv=Path("-"), map_csv=Path("-"))], ignore_index=True)
    graph = graph.drop_duplicates("bout_id").sort_values(["date", "bout_id"]).reset_index(drop=True)
    opp = prefight_opp_quality(pairs, replay_ratings(graph, ext_k_scale=MMA_EXT_K_SCALE))

    need = set(pairs["fighter_id"].dropna())
    by = {fid: sorted(g["date"]) for fid, g in ufc_dates[ufc_dates["fighter_id"].isin(need)].groupby("fighter_id")}
    n_ufc = pd.Series([sum(1 for x in by.get(f, []) if x < d) if f else 0
                       for f, d in zip(pairs["fighter_id"], pairs["date"])], dtype=float)
    out = pairs.copy()
    for f in MODEL_FEATURES:
        x = (car[f] if f in car.columns else opp[f]).astype(float).reset_index(drop=True)
        if f in _FADED:
            x = NO_DATA[f] + (x - NO_DATA[f]) * FADE_K / (FADE_K + n_ufc)
        out[f] = x.values
    return out


def current_model_features(ufc_conn, fighter_ids: list, before_date: str) -> dict:
    """{fighter_id: {feature: value}} going into a fight on *before_date* (predict.py)."""
    pairs = pd.DataFrame({"fighter_id": list(fighter_ids), "date": [before_date] * len(fighter_ids)})
    feats = prefight_model_features(pairs, ufc_conn)
    return {fid: {f: float(row[f]) for f in MODEL_FEATURES}
            for fid, (_, row) in zip(fighter_ids, feats.iterrows())}
