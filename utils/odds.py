"""
odds.py -- Betting odds utilities for the UFC Predictor.

Handles three common formats:
  - American / moneyline  (e.g. -150, +200)
  - Decimal               (e.g. 1.667, 3.00)
  - Fractional            (e.g. '2/3', '2/1')

Functions
---------
american_to_prob(odds)         -> implied probability (float, 0-1)
decimal_to_prob(odds)          -> implied probability
fractional_to_prob(frac_str)   -> implied probability
remove_vig(p_red, p_blue)      -> (fair_p_red, fair_p_blue)  -- vig stripped
value_bet(model_prob, fair_prob, label) -> print edge if significant
format_american(odds)          -> human-readable string
add_odds_columns(db_path)      -> add odds_red / odds_blue to fights table (one-time migration)
"""

import sqlite3
from pathlib import Path


# -- Conversion helpers --------------------------------------------------------

def american_to_prob(odds: float) -> float:
    """
    Convert American (moneyline) odds to implied win probability.

    Examples
    --------
    -150  (favourite) -> 0.600  (60.0 % implied)
    +200  (underdog)  -> 0.333  (33.3 % implied)
    """
    if odds < 0:
        return (-odds) / (-odds + 100)
    else:
        return 100 / (odds + 100)


def decimal_to_prob(odds: float) -> float:
    """
    Convert decimal odds to implied win probability.

    Examples
    --------
    1.667 -> 0.600
    3.00  -> 0.333
    """
    if odds <= 0:
        raise ValueError(f"Decimal odds must be > 0, got {odds}")
    return 1.0 / odds


def fractional_to_prob(frac_str: str) -> float:
    """
    Convert fractional odds string to implied win probability.

    Examples
    --------
    '2/3'  -> 0.600
    '2/1'  -> 0.333
    """
    parts = str(frac_str).split("/")
    if len(parts) != 2:
        raise ValueError(f"Expected 'numerator/denominator', got '{frac_str}'")
    num, den = float(parts[0]), float(parts[1])
    return den / (num + den)


# -- Vig removal ---------------------------------------------------------------

def remove_vig(p_red: float, p_blue: float) -> tuple[float, float]:
    """
    Strip the bookmaker's overround (vig) to get fair probabilities.

    The raw implied probabilities p_red + p_blue > 1.0 due to the vig.
    We normalise by dividing by their sum.

    Parameters
    ----------
    p_red, p_blue : raw implied probabilities from the odds lines

    Returns
    -------
    (fair_p_red, fair_p_blue) summing to exactly 1.0
    """
    total = p_red + p_blue
    if total <= 0:
        raise ValueError("Sum of implied probabilities must be > 0")
    return p_red / total, p_blue / total


# -- Value bet detection -------------------------------------------------------

def compute_edge(model_prob: float, fair_prob: float) -> float:
    """
    Edge = model probability - market fair probability.
    Positive edge -> model thinks fighter is undervalued by the market.
    """
    return model_prob - fair_prob


def kelly_fraction(edge: float, odds_decimal: float, fraction: float = 0.25) -> float:
    """
    Fractional Kelly stake sizing.

    Parameters
    ----------
    edge          : model_prob - fair_prob  (positive = value bet)
    odds_decimal  : decimal odds for the bet
    fraction      : Kelly multiplier (default 0.25 = quarter Kelly, conservative)

    Returns
    -------
    Stake as fraction of bankroll (0.0 -> 1.0). Returns 0 if edge <= 0.
    """
    if edge <= 0 or odds_decimal <= 1:
        return 0.0
    b = odds_decimal - 1  # net odds (profit per unit staked)
    p = edge + (1 / odds_decimal)  # model win probability
    q = 1 - p
    full_kelly = (b * p - q) / b
    return max(0.0, full_kelly * fraction)


def print_value_bet_summary(
    r_name: str,
    b_name: str,
    model_p_red: float,
    model_p_blue: float,
    odds_red_american: float | None = None,
    odds_blue_american: float | None = None,
) -> None:
    """
    Print a value-bet comparison table if odds are provided.

    Parameters
    ----------
    r_name / b_name              : fighter display names
    model_p_red / model_p_blue   : model win probabilities (must sum to ~1)
    odds_red_american            : American moneyline odds for Red  (or None)
    odds_blue_american           : American moneyline odds for Blue (or None)
    """
    if odds_red_american is None or odds_blue_american is None:
        print("\n  Odds: not provided (use --odds-red / --odds-blue to enable value-bet analysis)")
        return

    # Raw implied probabilities
    raw_p_red  = american_to_prob(odds_red_american)
    raw_p_blue = american_to_prob(odds_blue_american)
    vig        = (raw_p_red + raw_p_blue - 1.0) * 100

    # Fair (vig-stripped) probabilities
    fair_p_red, fair_p_blue = remove_vig(raw_p_red, raw_p_blue)

    edge_red  = compute_edge(model_p_red,  fair_p_red)
    edge_blue = compute_edge(model_p_blue, fair_p_blue)

    dec_red  = 1 / raw_p_red  if raw_p_red  > 0 else 0
    dec_blue = 1 / raw_p_blue if raw_p_blue > 0 else 0

    kelly_red  = kelly_fraction(edge_red,  dec_red)
    kelly_blue = kelly_fraction(edge_blue, dec_blue)

    def _fmt_odds(o: float) -> str:
        return f"+{o:.0f}" if o > 0 else f"{o:.0f}"

    print(f"\n{'-' * 56}")
    print(f"  Betting Odds Analysis")
    print(f"{'-' * 56}")
    print(f"  Bookmaker vig: {vig:.1f}%")
    print()
    print(f"  {'':20s}  {'Model':>8}  {'Mkt Fair':>8}  {'Edge':>8}  {'Odds':>7}")
    print(f"  {'-'*56}")

    for name, model_p, fair_p, edge, kelly, dec_odds, am_odds in [
        (r_name,  model_p_red,  fair_p_red,  edge_red,  kelly_red,  dec_red,  odds_red_american),
        (b_name,  model_p_blue, fair_p_blue, edge_blue, kelly_blue, dec_blue, odds_blue_american),
    ]:
        edge_str = f"{edge:+.1%}"
        name_trunc = name[:20]
        print(f"  {name_trunc:20s}  {model_p:>7.1%}  {fair_p:>7.1%}  {edge_str:>8}  {_fmt_odds(am_odds):>7}")

    # Value bet recommendation
    min_edge = 0.03   # 3 pp minimum edge to flag as value
    print()
    found_value = False
    for name, model_p, fair_p, edge, kelly, dec_odds, am_odds in [
        (r_name,  model_p_red,  fair_p_red,  edge_red,  kelly_red,  dec_red,  odds_red_american),
        (b_name,  model_p_blue, fair_p_blue, edge_blue, kelly_blue, dec_blue, odds_blue_american),
    ]:
        if edge >= min_edge:
            found_value = True
            kelly_pct = kelly * 100
            print(f"  [VALUE] {name} -- edge {edge:+.1%} vs market")
            print(f"          Odds: {_fmt_odds(am_odds)} | Fair: {fair_p:.1%} | Model: {model_p:.1%}")
            if kelly > 0:
                print(f"          Quarter-Kelly stake: {kelly_pct:.1f}% of bankroll")

    if not found_value:
        print(f"  No value bets detected (edge < {min_edge:.0%} for both fighters)")


# -- DB migration --------------------------------------------------------------

def add_odds_columns(db_path: Path) -> None:
    """
    One-time migration: add odds_red and odds_blue columns to the fights table.

    Columns store American moneyline odds (float).  NULL = not yet populated.
    This is a safe no-op if the columns already exist.
    """
    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()

    cur.execute("PRAGMA table_info(fights)")
    existing = {row[1] for row in cur.fetchall()}

    for col in ("odds_red", "odds_blue"):
        if col not in existing:
            cur.execute(f"ALTER TABLE fights ADD COLUMN {col} REAL")
            print(f"  Added column '{col}' to fights table.")
        else:
            print(f"  Column '{col}' already exists -- skipped.")

    conn.commit()
    conn.close()
    print("Odds columns ready.")


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    ROOT_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT_DIR))
    from config import DB_PATH

    parser = argparse.ArgumentParser(
        description="Odds utilities -- add DB columns or test conversions.",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Add odds_red / odds_blue columns to the fights table.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run conversion examples.",
    )
    args = parser.parse_args()

    if args.migrate:
        print(f"Migrating database: {DB_PATH}")
        add_odds_columns(DB_PATH)

    if args.test:
        print("Conversion tests:")
        print(f"  -150 American -> {american_to_prob(-150):.3f} implied")
        print(f"  +200 American -> {american_to_prob(+200):.3f} implied")
        print(f"  1.667 decimal -> {decimal_to_prob(1.667):.3f} implied")
        print(f"  '2/3' fraction -> {fractional_to_prob('2/3'):.3f} implied")
        p_r, p_b = remove_vig(0.60, 0.45)
        print(f"  remove_vig(0.60, 0.45) -> ({p_r:.3f}, {p_b:.3f}) -- vig was {(0.60+0.45-1)*100:.1f}%")
