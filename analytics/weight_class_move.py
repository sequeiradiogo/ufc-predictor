"""
analytics/weight_class_move.py -- Does "lighter man skill" hold up? UFC fans
often claim a fighter who moves up a weight class performs better there than
their raw record suggests, because they're bringing skills honed against
faster/technically sharper competition from the lighter division.

This tests it directly: for every fighter who moved up in weight at least
once, compare their win rate across their *entire career in the division
they came from* (up to the point they left it) against their win rate in
their *first three fights in the new, heavier division*.

Data source:
- raw_data/ufc-master.csv -- weight_class (fight-level, the division the
  fight was contracted at) + date + Winner. No division-move data exists
  anywhere else in the repo, so this is the only column that carries it.

Weight-class ordering is fixed by the standard UFC division limits (in lbs),
kept separate for men's and women's divisions since a fighter can't move
between them. "Catch Weight" fights are dropped -- they're one-off,
non-standard bouts with no fixed division to compare against.

A "move up" is any fight where the fighter's weight_class is heavier than
their immediately preceding fight's weight_class. Each move is its own
data point (a fighter who moves up twice in a career contributes two rows).
The "former division" win rate uses only that fighter's fights in the
division they most recently came from -- not their whole career -- so a
fighter who bounced strawweight -> flyweight -> bantamweight isn't credited
with strawweight form when leaving flyweight.

Two significance tests are run:
- Paired (per-mover): each mover's own new-division win rate minus their
  own old-division win rate -- a one-sample Wilcoxon signed-rank test
  against a median difference of 0. This is the right test for "does moving
  up help *this fighter* relative to *their own* prior form," which is what
  the claim is actually about.
- Aggregate (pooled): a two-proportion z-test comparing the total win/loss
  tally across all old-division fights vs. all new-division first-three
  fights, pooled across movers. Included for scale, but overweights movers
  with long old-division careers relative to their first-3-fight sample.

Usage:
    python analytics/weight_class_move.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm, wilcoxon

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

MASTER_CSV = ROOT_DIR / "raw_data" / "ufc-master.csv"
REPORT_PNG = ROOT_DIR / "analytics" / "reports" / "weight_class_move.png"

# Standard UFC division limits (lbs), used purely to order divisions
# light-to-heavy. Men's and women's divisions are kept in separate spaces
# (a fighter can never move between them) by prefixing women's keys.
MEN_ORDER = {
    "flyweight": 125, "bantamweight": 135, "featherweight": 145,
    "lightweight": 155, "welterweight": 170, "middleweight": 185,
    "light heavyweight": 205, "heavyweight": 265,
}
WOMEN_ORDER = {
    "women's strawweight": 115, "women's flyweight": 125,
    "women's bantamweight": 135, "women's featherweight": 145,
}
DIVISION_ORDER = {**MEN_ORDER, **WOMEN_ORDER}

# str.title() mangles "women's" -> "Women'S" (capitalizes after the
# apostrophe); use a display-name map instead.
DIVISION_DISPLAY = {key: key.title().replace("'S", "'s") for key in DIVISION_ORDER}

MIN_OLD_FIGHTS = 5   # minimum career fights in the former division to count
FIRST_N_NEW = 2        # how many fights into the new division to evaluate


def normalize_division(wc) -> str | None:
    if not isinstance(wc, str) or not wc.strip():
        return None
    key = wc.strip().lower()
    return key if key in DIVISION_ORDER else None


def build_long_dataframe() -> pd.DataFrame:
    fights = pd.read_csv(MASTER_CSV)
    fights["division"] = fights["weight_class"].apply(normalize_division)
    fights = fights.dropna(subset=["division"])
    fights = fights[fights["Winner"].isin(["Red", "Blue"])]
    fights["date"] = pd.to_datetime(fights["date"])

    red = pd.DataFrame({
        "name": fights["R_fighter"], "date": fights["date"],
        "division": fights["division"], "won": fights["Winner"] == "Red",
    })
    blue = pd.DataFrame({
        "name": fights["B_fighter"], "date": fights["date"],
        "division": fights["division"], "won": fights["Winner"] == "Blue",
    })
    long_df = pd.concat([red, blue], ignore_index=True)
    long_df = long_df.sort_values(["name", "date"]).reset_index(drop=True)
    return long_df


def find_moves(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per weight-class move-up: old-division career win rate (up to
    the move) vs. new-division win rate over the first FIRST_N_NEW fights."""
    rows = []
    for name, career in long_df.groupby("name", sort=False):
        career = career.reset_index(drop=True)
        divisions = career["division"].tolist()
        won = career["won"].tolist()

        for i in range(1, len(career)):
            old_div, new_div = divisions[i - 1], divisions[i]
            if old_div == new_div:
                continue
            if DIVISION_ORDER[new_div] <= DIVISION_ORDER[old_div]:
                continue  # a move down, not up -- not what we're testing

            # Former-division career: this fighter's fights in old_div,
            # back to (but not past) their last division change into it.
            start = i - 1
            while start > 0 and divisions[start - 1] == old_div:
                start -= 1
            old_fights = won[start:i]
            if len(old_fights) < MIN_OLD_FIGHTS:
                continue

            # New-division: first FIRST_N_NEW fights at new_div starting here.
            new_fights = [won[j] for j in range(i, len(career)) if divisions[j] == new_div][:FIRST_N_NEW]
            if not new_fights:
                continue

            rows.append({
                "name": name, "move_date": career["date"].iloc[i],
                "old_division": DIVISION_DISPLAY[old_div], "new_division": DIVISION_DISPLAY[new_div],
                "old_n": len(old_fights), "old_wins": sum(old_fights),
                "old_win_pct": 100 * sum(old_fights) / len(old_fights),
                "new_n": len(new_fights), "new_wins": sum(new_fights),
                "new_win_pct": 100 * sum(new_fights) / len(new_fights),
            })
    return pd.DataFrame(rows)


def two_proportion_ztest(wins_a, n_a, wins_b, n_b):
    p_a, p_b = wins_a / n_a, wins_b / n_b
    p_pool = (wins_a + wins_b) / (n_a + n_b)
    se = (p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)) ** 0.5
    if se == 0:
        return p_a, p_b, 1.0
    z = (p_a - p_b) / se
    return p_a, p_b, 2 * (1 - norm.cdf(abs(z)))


OLD_COLOR = "#3b6ea5"
NEW_COLOR = "#c7392e"


def plot(moves: pd.DataFrame) -> None:
    diffs = moves["new_win_pct"] - moves["old_win_pct"]
    n_down, n_up, n_flat = (diffs < 0).sum(), (diffs > 0).sum(), (diffs == 0).sum()

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel A: pooled win% bar, old vs new, with sample sizes ---
    old_wins, old_n = moves["old_wins"].sum(), moves["old_n"].sum()
    new_wins, new_n = moves["new_wins"].sum(), moves["new_n"].sum()
    bar_df = pd.DataFrame({
        "which": ["Former division\n(full career there)", f"New division\n(first {FIRST_N_NEW} fights)"],
        "win_pct": [100 * old_wins / old_n, 100 * new_wins / new_n],
        "n": [old_n, new_n],
    })
    ax = axes[0]
    sns.barplot(data=bar_df, x="which", y="win_pct", hue="which", ax=ax,
                palette=[OLD_COLOR, NEW_COLOR], legend=False, width=0.55)
    for container, (_, row) in zip(ax.containers, bar_df.iterrows()):
        ax.bar_label(container, labels=[f"{row.win_pct:.1f}%\n(n={row.n:,})"], padding=4, fontsize=10)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, max(bar_df["win_pct"]) + 20)
    ax.set_xlabel("")
    ax.set_ylabel("Win %")
    ax.set_title("Pooled win rate")

    # --- Panel B: per-mover scatter, old win% vs new win%, diagonal = no change ---
    ax = axes[1]
    below = moves[moves["new_win_pct"] < moves["old_win_pct"]]
    at_or_above = moves[moves["new_win_pct"] >= moves["old_win_pct"]]
    ax.scatter(below["old_win_pct"], below["new_win_pct"], color=NEW_COLOR, alpha=0.6, s=28,
               label=f"Did worse ({len(below)})")
    ax.scatter(at_or_above["old_win_pct"], at_or_above["new_win_pct"], color=OLD_COLOR, alpha=0.6, s=28,
               label=f"Same or better ({len(at_or_above)})")
    ax.plot([0, 100], [0, 100], color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Former-division win %")
    ax.set_ylabel("New-division win %")
    ax.set_title("Each mover: before vs. after")
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    # --- Panel C: histogram of per-mover win% change ---
    ax = axes[2]
    sns.histplot(diffs, bins=14, ax=ax, color=NEW_COLOR, alpha=0.75, edgecolor="white")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(diffs.mean(), color="black", linestyle="-", linewidth=1.3,
               label=f"Mean {diffs.mean():+.1f}pp")
    ax.set_xlabel("Win % change (new - former)")
    ax.set_ylabel("Movers")
    ax.set_title(f"{n_down} did worse, {n_up} did better, {n_flat} unchanged")
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    sns.despine()
    fig.tight_layout()

    REPORT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_PNG, dpi=150, bbox_inches="tight")
    print(f"Chart written to {REPORT_PNG}")


def main() -> None:
    long_df = build_long_dataframe()
    moves = find_moves(long_df)
    print(f"{len(moves)} weight-class-up moves found "
          f"(from {moves['name'].nunique()} unique fighters), "
          f"requiring >= {MIN_OLD_FIGHTS} former-division fights.")

    pd.set_option("display.float_format", lambda v: f"{v:.1f}")
    print(moves[["name", "old_division", "new_division", "old_n", "old_win_pct",
                 "new_n", "new_win_pct"]].to_string(index=False))

    # Aggregate (pooled) two-proportion z-test
    old_wins, old_n = moves["old_wins"].sum(), moves["old_n"].sum()
    new_wins, new_n = moves["new_wins"].sum(), moves["new_n"].sum()
    p_old, p_new, p_value_pooled = two_proportion_ztest(old_wins, old_n, new_wins, new_n)
    print(f"\nPooled: old-division win% = {p_old*100:.1f}% (n={old_n}), "
          f"new-division win% = {p_new*100:.1f}% (n={new_n}), "
          f"diff = {(p_new-p_old)*100:+.1f}pp, p = {p_value_pooled:.4f}")

    # Paired per-mover Wilcoxon signed-rank test
    diffs = moves["new_win_pct"] - moves["old_win_pct"]
    nonzero = diffs[diffs != 0]
    if len(nonzero) >= 1:
        stat, p_value_paired = wilcoxon(nonzero)
    else:
        p_value_paired = float("nan")
    print(f"Paired: mean per-mover diff = {diffs.mean():+.1f}pp, "
          f"median = {diffs.median():+.1f}pp, "
          f"Wilcoxon p = {p_value_paired:.4f} (n={len(moves)} movers)")

    plot(moves)


if __name__ == "__main__":
    main()
