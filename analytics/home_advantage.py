"""
analytics/home_advantage.py -- Do fighters win more often on home soil than
on foreign soil, for American, Brazilian, English (UK), Mexican, Russian,
and Australian fighters -- restricted to fights that went to a judges'
decision (DECISION_ONLY), since that's the outcome type where a scoring-bias
"hometown advantage" could actually operate?

China was tried and dropped: only 12 home-soil decisions resolved for
Chinese fighters, too small a sample to say anything meaningful (the result
swung on a handful of fights and wasn't statistically significant anyway).

Data sources:
- raw_data/fighter_nationality.csv -- fighter hometown/country, scraped from
  ufc.com athlete pages by scrapers/ufc_fighter_nationality.py (neither
  db/ufc_ufcstats.db nor ufc-master.csv carries fighter nationality).
- raw_data/ufc-master.csv -- per-fight event country + Winner (Red/Blue).
  db/ufc_ufcstats.db's own fights.country/location columns are >98% NULL
  (only ~160/8810 rows populated) so ufc-master.csv's Kaggle-sourced country
  field is used instead -- it's populated for 7323/7375 fights.

Fighter nationality is normalized to six buckets via FIGHTER_COUNTRY_CANON --
e.g. a fighter whose ufc.com hometown says "England" is bucketed as "United
Kingdom", since UFC event-country data doesn't distinguish England from
Scotland/Wales/Northern Ireland. Event-country strings are otherwise used
as-is (just whitespace/USA-spelling normalized via EVENT_ALIASES) so away
fights held anywhere in the world are counted, not just in the other target
countries.

"Home soil" is not always just the fighter's own country: HOME_EVENT_COUNTRIES
lets a nationality bucket count more than one event country as home. Russian
(and broader ex-Soviet/Caucasus/Dagestani) fighters fight often in the UAE,
Qatar, and Saudi Arabia as part of the same promotional/cultural axis as
Russia itself, so those three are counted as home soil for the "Russia"
bucket alongside Russia proper -- a deliberate modeling choice, not a data
artifact.

Usage:
    python analytics/home_advantage.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

NATIONALITY_CSV = ROOT_DIR / "raw_data" / "fighter_nationality.csv"
MASTER_CSV = ROOT_DIR / "raw_data" / "ufc-master.csv"
REPORT_PNG = ROOT_DIR / "analytics" / "reports" / "home_advantage.png"

FIGHTER_COUNTRY_CANON = {
    "united states": "USA", "usa": "USA",
    "brazil": "Brazil",
    "england": "United Kingdom", "united kingdom": "United Kingdom",
    "scotland": "United Kingdom", "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    "mexico": "Mexico",
    "russia": "Russia",
    "australia": "Australia",
}
TARGET_COUNTRIES = ["USA", "Brazil", "United Kingdom", "Mexico", "Russia", "Australia"]

# ufc-master.csv's `finish` column; the three judge-scored outcomes. Fights
# decided by KO/TKO, submission, DQ, etc. are excluded when DECISION_ONLY is
# set -- a judged decision is where a scoring-bias "hometown advantage"
# would actually show up, unlike a finish, which the judges don't get a say in.
DECISION_METHODS = {"U-DEC", "S-DEC", "M-DEC"}
DECISION_ONLY = True

# Only variant spellings need aliasing here -- unlike FIGHTER_COUNTRY_CANON,
# event-country strings are otherwise passed through as-is (see module
# docstring) so fights held anywhere count toward the "away" bucket.
EVENT_ALIASES = {"usa": "USA", "united states": "USA"}

# Which event-country values count as "home" for each fighter nationality
# bucket. Defaults to the country matching its own name; Russia is the one
# deliberate exception (see module docstring).
HOME_EVENT_COUNTRIES = {country: {country} for country in TARGET_COUNTRIES}
HOME_EVENT_COUNTRIES["Russia"] |= {"United Arab Emirates", "Qatar", "Saudi Arabia"}

# Matches the home/away palette used in the published home-soil-advantage
# report artifact -- validated colorblind-safe (CVD Delta-E 17.9, normal 26.7)
# via the dataviz skill's scripts/validate_palette.js.
HOME_COLOR = "#c7392e"
AWAY_COLOR = "#3b6ea5"


def canon_fighter_country(country) -> str | None:
    if not isinstance(country, str) or not country.strip():
        return None
    return FIGHTER_COUNTRY_CANON.get(country.strip().lower())


def canon_event_country(country) -> str | None:
    if not isinstance(country, str) or not country.strip():
        return None
    stripped = country.strip()
    return EVENT_ALIASES.get(stripped.lower(), stripped)


def load_fighter_countries() -> dict:
    df = pd.read_csv(NATIONALITY_CSV)
    df["canon"] = df["country"].apply(canon_fighter_country)
    df = df.dropna(subset=["canon"])
    # A few names may repeat (e.g. common names shared by two fighters
    # UFCStats couldn't disambiguate) -- keep the first occurrence, exploratory
    # analysis doesn't need airtight dedup here.
    return dict(zip(df["name"], df["canon"]))


def build_long_dataframe() -> pd.DataFrame:
    fighter_country = load_fighter_countries()
    fights = pd.read_csv(MASTER_CSV)
    fights["event_country"] = fights["country"].apply(canon_event_country)
    fights = fights.dropna(subset=["event_country"])
    fights = fights[fights["Winner"].isin(["Red", "Blue"])]
    if DECISION_ONLY:
        fights = fights[fights["finish"].isin(DECISION_METHODS)]

    red = pd.DataFrame({
        "name": fights["R_fighter"],
        "event_country": fights["event_country"],
        "won": fights["Winner"] == "Red",
    })
    blue = pd.DataFrame({
        "name": fights["B_fighter"],
        "event_country": fights["event_country"],
        "won": fights["Winner"] == "Blue",
    })
    long_df = pd.concat([red, blue], ignore_index=True)
    long_df["country"] = long_df["name"].map(fighter_country)
    long_df = long_df.dropna(subset=["country"])
    long_df = long_df[long_df["country"].isin(TARGET_COUNTRIES)]
    long_df["is_home"] = long_df.apply(
        lambda r: r["event_country"] in HOME_EVENT_COUNTRIES[r["country"]], axis=1)
    return long_df


def two_proportion_ztest(wins_a, n_a, wins_b, n_b):
    """Two-sided z-test for a difference in win proportions. Returns
    (p_home, p_away, p_value); None p_value if either group is empty."""
    if n_a == 0 or n_b == 0:
        return (wins_a / n_a if n_a else float("nan"),
                wins_b / n_b if n_b else float("nan"), None)
    p_a, p_b = wins_a / n_a, wins_b / n_b
    p_pool = (wins_a + wins_b) / (n_a + n_b)
    se = (p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)) ** 0.5
    if se == 0:
        return p_a, p_b, 1.0
    z = (p_a - p_b) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return p_a, p_b, p_value


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country in TARGET_COUNTRIES:
        sub = long_df[long_df["country"] == country]
        home = sub[sub["is_home"]]
        away = sub[~sub["is_home"]]
        p_home, p_away, p_value = two_proportion_ztest(
            home["won"].sum(), len(home), away["won"].sum(), len(away))
        rows.append({
            "country": country,
            "home_fights": len(home), "home_win_pct": p_home * 100,
            "away_fights": len(away), "away_win_pct": p_away * 100,
            "diff_pp": (p_home - p_away) * 100,
            "p_value": p_value,
        })
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame) -> None:
    long_summary = summary.melt(
        id_vars=["country", "home_fights", "away_fights"],
        value_vars=["home_win_pct", "away_win_pct"],
        var_name="soil", value_name="win_pct",
    )
    long_summary["soil"] = long_summary["soil"].map(
        {"home_win_pct": "Home soil", "away_win_pct": "Foreign soil"})
    long_summary["n"] = long_summary.apply(
        lambda r: r["home_fights"] if r["soil"] == "Home soil" else r["away_fights"], axis=1)

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=long_summary, x="country", y="win_pct", hue="soil",
        palette={"Home soil": HOME_COLOR, "Foreign soil": AWAY_COLOR},
        ax=ax,
    )

    for container, soil in zip(ax.containers, ["Home soil", "Foreign soil"]):
        rows = long_summary[long_summary["soil"] == soil].reset_index()
        labels = [f"{row.win_pct:.1f}%\n(n={row.n:,})" for row in rows.itertuples()]
        ax.bar_label(container, labels=labels, padding=3, fontsize=8.5)

    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, max(long_summary["win_pct"]) + 15)
    ax.set_xlabel("")
    ax.set_ylabel("Win %")
    ax.set_title("Fighter win rate: home soil vs. foreign soil")
    ax.legend(title="")
    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    REPORT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_PNG, dpi=150)
    print(f"Chart written to {REPORT_PNG}")


def main() -> None:
    long_df = build_long_dataframe()
    print(f"{len(long_df)} fighter-fight records across the {len(TARGET_COUNTRIES)} target countries "
          f"(from {long_df['name'].nunique()} unique fighters).")

    summary = summarize(long_df)
    pd.set_option("display.float_format", lambda v: f"{v:.1f}")
    print(summary.to_string(index=False))

    plot(summary)


if __name__ == "__main__":
    main()
