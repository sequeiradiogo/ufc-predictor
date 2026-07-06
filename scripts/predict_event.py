"""
predict_event.py -- Scrape the next UFC event and generate predictions.

Usage:
    python scripts/predict_event.py
    python scripts/predict_event.py --model ensemble
    python scripts/predict_event.py --output predictions/my-event.md

Steps:
    1. Scrape the next upcoming event card from UFCStats (Playwright required).
    2. Look up fighters in the v2 DB by UFCStats ID (debut check + name normalisation).
    3. Skip fights where either fighter has no recorded fight history (debut).
    4. Run v1 (mdabbert) prediction for each fight.
    5. Write a Markdown file and a companion interactive HTML file to predictions/.
"""

import argparse
import json
import re
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DB_PATH, DB_V1_PATH, MODELS_V1_DIR, MODELS_V1_PROD_DIR, FINISH_CLASS_NAMES
from predict import compute_prediction
from scrapers.ufcstats import scrape_upcoming_event
from utils.logger import get_logger

log = get_logger(__name__)

PREDICTIONS_DIR = ROOT_DIR / "predictions"

MODEL_LABELS = {
    "xgb":      "XGBoost",
    "lr":       "Logistic Regression",
    "rf":       "Random Forest",
    "lgbm":     "LightGBM",
    "ensemble": "Ensemble (Soft Vote)",
}


# ── DB helpers ────────────────────────────────────────────────────────────────

def lookup_fighter(conn: sqlite3.Connection, fighter_id: str) -> tuple[str, str] | None:
    """Return (fighter_id, name) from DB, or None if not found."""
    row = conn.execute(
        "SELECT fighter_id, name FROM fighters WHERE fighter_id = ?",
        (fighter_id,),
    ).fetchone()
    return row if row else None


def has_fight_history(conn: sqlite3.Connection, fighter_id: str) -> bool:
    """Return True if the fighter has at least one fight_stats row."""
    row = conn.execute(
        "SELECT 1 FROM fight_stats WHERE fighter_id = ? LIMIT 1",
        (fighter_id,),
    ).fetchone()
    return row is not None


# ── Interactive HTML dashboard ────────────────────────────────────────────────

def _fight_data_for_js(results: list[dict]) -> str:
    """Serialise fight stats to a JSON string for embedding in the HTML."""
    fights = []
    for r in results:
        sr = r.get("stats_red",  {})
        sb = r.get("stats_blue", {})
        form_r = r["form_red"]
        form_b = r["form_blue"]

        def pct(v: float) -> float:
            return round(v * 100, 1)

        radar_axes  = ["Str acc", "Str def", "TD acc", "TD def",
                       "Finish rate", "Recent W%"]
        radar_units = ["%", "%", "%", "%", "%", "%"]
        radar_r = [pct(sr.get("str_acc",  0)), pct(sr.get("str_def", 0)),
                   pct(sr.get("td_acc",   0)), pct(sr.get("td_def",  0)),
                   pct(sr.get("ko_rate",  0)), pct(sr.get("win_rate", 0))]
        radar_b = [pct(sb.get("str_acc",  0)), pct(sb.get("str_def", 0)),
                   pct(sb.get("td_acc",   0)), pct(sb.get("td_def",  0)),
                   pct(sb.get("ko_rate",  0)), pct(sb.get("win_rate", 0))]

        # Bars: volume/rate stats only (non-percentage, comparable scales)
        bars_stats = ["Strikes/min", "Str absorbed/min", "TD avg/15min", "Win streak"]
        bars_units = ["/min", "/min", "/15min", ""]
        bars_r = [round(sr.get("splm",        0), 2),
                  round(sr.get("sapm",        0), 2),
                  round(sr.get("td_avg",      0), 2),
                  round(sr.get("win_streak",  0), 0)]
        bars_b = [round(sb.get("splm",        0), 2),
                  round(sb.get("sapm",        0), 2),
                  round(sb.get("td_avg",      0), 2),
                  round(sb.get("win_streak",  0), 0)]

        fights.append({
            "label":      f"{r['red_name']} vs {r['blue_name']}",
            "red_name":   r["red_name"],
            "blue_name":  r["blue_name"],
            "winner":     r["winner"],
            "red_prob":   pct(r["red_prob"]),
            "blue_prob":  pct(r["blue_prob"]),
            "elo_red":    round(r["elo_red"],  0),
            "elo_blue":   round(r["elo_blue"], 0),
            "finish":      r.get("finish_proba") or [],
            "finish_str":  _format_finish(r.get("finish_proba") or []),
            "radar": {
                "axes":  radar_axes,
                "units": radar_units,
                "red":   radar_r,
                "blue":  radar_b,
            },
            "bars": {
                "stats": bars_stats,
                "red":   bars_r,
                "blue":  bars_b,
                "units": bars_units,
            },
            "form": {
                "red_streak":  int(form_r.get("win_streak", 0)),
                "blue_streak": int(form_b.get("win_streak", 0)),
                "red_finish":  pct(form_r.get("recent_finish_rate", 0)),
                "blue_finish": pct(form_b.get("recent_finish_rate", 0)),
                "red_sos":     round(sr.get("sos", 1400), 0),
                "blue_sos":    round(sb.get("sos", 1400), 0),
            },
        })
    return json.dumps(fights, indent=2)


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{event_name}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f0f; color: #e0e0e0; }}
  header {{ background: #1a1a1a; border-bottom: 2px solid #c0392b; padding: 18px 32px; }}
  header h1 {{ font-size: 1.5rem; color: #fff; }}
  header p  {{ font-size: 0.85rem; color: #888; margin-top: 4px; }}
  .tabs {{ display: flex; flex-wrap: wrap; gap: 6px; padding: 16px 32px 0; background: #141414; }}
  .tab-btn {{
    padding: 7px 14px; border: 1px solid #333; border-radius: 4px;
    background: #1e1e1e; color: #bbb; cursor: pointer; font-size: 0.78rem;
    transition: background 0.15s, color 0.15s;
  }}
  .tab-btn:hover  {{ background: #2a2a2a; color: #fff; }}
  .tab-btn.active {{ background: #c0392b; color: #fff; border-color: #c0392b; }}
  .panel {{ display: none; padding: 24px 32px; }}
  .panel.active {{ display: block; }}
  .fight-header {{ display: flex; align-items: center; justify-content: space-between;
                   margin-bottom: 20px; flex-wrap: wrap; gap: 12px; }}
  .fighter-card {{ text-align: center; min-width: 180px; }}
  .fighter-name {{ font-size: 1.3rem; font-weight: bold; }}
  .fighter-elo  {{ font-size: 0.82rem; color: #888; margin-top: 2px; }}
  .win-prob     {{ font-size: 2rem; font-weight: bold; margin-top: 6px; }}
  .red-col  {{ color: #e74c3c; }}
  .blue-col {{ color: #3498db; }}
  .vs-block {{ text-align: center; }}
  .vs-label {{ font-size: 1.1rem; color: #555; }}
  .winner-tag {{ font-size: 0.85rem; color: #2ecc71; margin-top: 6px; font-weight: bold; }}
  .charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 780px) {{ .charts-row {{ grid-template-columns: 1fr; }} }}
  .chart-box {{ background: #1a1a1a; border-radius: 8px; padding: 12px; }}
  .chart-title {{ font-size: 0.82rem; color: #888; text-transform: uppercase;
                  letter-spacing: 0.06em; margin-bottom: 8px; }}
  .meta-row {{ display: flex; gap: 20px; flex-wrap: wrap; margin-top: 20px; }}
  .meta-card {{ background: #1a1a1a; border-radius: 8px; padding: 14px 20px; flex: 1; min-width: 160px; }}
  .meta-card h4 {{ font-size: 0.75rem; color: #666; text-transform: uppercase;
                   letter-spacing: 0.06em; margin-bottom: 8px; }}
  .meta-row-inner {{ display: flex; justify-content: space-between; font-size: 0.9rem; }}
  .meta-val-r {{ color: #e74c3c; }}
  .meta-val-b {{ color: #3498db; }}
</style>
</head>
<body>

<header>
  <h1>{event_name}</h1>
  <p>Model: {model_label} &nbsp;|&nbsp; Generated: {generated} &nbsp;|&nbsp; {n_fights} fights predicted</p>
</header>

<div class="tabs" id="tabs"></div>

<div id="panels"></div>

<script>
const FIGHTS = {fight_data_json};
const FINISH_NAMES = {finish_names_json};

const RED  = '#e74c3c';
const BLUE = '#3498db';

const plotCfg = {{responsive: true, displayModeBar: false}};

function radarLayout() {{
  return {{
    polar: {{
      bgcolor: '#1a1a1a',
      radialaxis: {{ visible: true, range: [0, 100], color: '#444',
                    gridcolor: '#333', tickfont: {{color:'#555', size:9}} }},
      angularaxis: {{ color: '#888', gridcolor: '#2a2a2a', linecolor: '#333' }},
    }},
    paper_bgcolor: '#1a1a1a', plot_bgcolor: '#1a1a1a',
    showlegend: true,
    legend: {{ font: {{color:'#bbb', size:11}}, bgcolor:'#1a1a1a',
               x: 0.5, xanchor:'center', y:-0.15, orientation:'h' }},
    margin: {{t:10, b:40, l:40, r:40}},
  }};
}}

function barLayout() {{
  return {{
    barmode: 'group',
    paper_bgcolor: '#1a1a1a', plot_bgcolor: '#1a1a1a',
    xaxis: {{ color:'#888', gridcolor:'#2a2a2a', zeroline:false }},
    yaxis: {{ color:'#bbb', gridcolor:'#2a2a2a', zeroline:false,
              automargin: true, tickfont: {{size: 11}} }},
    legend: {{ font:{{color:'#bbb', size:11}}, bgcolor:'#1a1a1a',
               x:0.5, xanchor:'center', y:-0.18, orientation:'h' }},
    margin: {{t:10, b:50, l:10, r:20}},
  }};
}}

function buildPanel(f, idx) {{
  const div = document.createElement('div');
  div.className = 'panel';
  div.id = 'panel-' + idx;

  const isRedWinner  = f.winner === f.red_name;
  const isBlueWinner = f.winner === f.blue_name;

  div.innerHTML = `
    <div class="fight-header">
      <div class="fighter-card">
        <div class="fighter-name red-col">${{f.red_name}}</div>
        <div class="fighter-elo">ELO ${{f.elo_red}}</div>
        <div class="win-prob red-col">${{f.red_prob}}%</div>
        ${{isRedWinner ? '<div class="winner-tag">Model pick</div>' : ''}}
      </div>
      <div class="vs-block">
        <div class="vs-label">VS</div>
      </div>
      <div class="fighter-card">
        <div class="fighter-name blue-col">${{f.blue_name}}</div>
        <div class="fighter-elo">ELO ${{f.elo_blue}}</div>
        <div class="win-prob blue-col">${{f.blue_prob}}%</div>
        ${{isBlueWinner ? '<div class="winner-tag">Model pick</div>' : ''}}
      </div>
    </div>

    <div class="charts-row">
      <div class="chart-box">
        <div class="chart-title">Percentage stats (radar)</div>
        <div id="radar-${{idx}}" style="height:340px"></div>
      </div>
      <div class="chart-box">
        <div class="chart-title">Volume / rate stats</div>
        <div id="bars-${{idx}}" style="height:340px"></div>
      </div>
    </div>

    <div class="meta-row">
      <div class="meta-card">
        <h4>Recent form</h4>
        <div class="meta-row-inner">
          <span class="meta-val-r">${{f.red_name.split(' ').pop()}}</span>
          <span class="meta-val-b">${{f.blue_name.split(' ').pop()}}</span>
        </div>
        <div class="meta-row-inner" style="margin-top:6px; font-size:0.82rem; color:#888">
          <span>Win streak: ${{f.form.red_streak}}</span>
          <span>Win streak: ${{f.form.blue_streak}}</span>
        </div>
        <div class="meta-row-inner" style="margin-top:4px; font-size:0.82rem; color:#888">
          <span>Finish rate: ${{f.form.red_finish}}%</span>
          <span>Finish rate: ${{f.form.blue_finish}}%</span>
        </div>
      </div>
      <div class="meta-card">
        <h4>Strength of schedule (ELO)</h4>
        <div class="meta-row-inner">
          <span class="meta-val-r">${{f.red_name.split(' ').pop()}}</span>
          <span class="meta-val-b">${{f.blue_name.split(' ').pop()}}</span>
        </div>
        <div class="meta-row-inner" style="margin-top:6px; font-size:0.82rem; color:#888">
          <span>${{f.form.red_sos}}</span>
          <span>${{f.form.blue_sos}}</span>
        </div>
      </div>
      ${{f.finish.length ? `
      <div class="meta-card">
        <h4>Likely finish</h4>
        ${{FINISH_NAMES.map((n,i) => `
        <div class="meta-row-inner" style="margin-top:${{i?6:0}}px; font-size:0.9rem">
          <span>${{n}}</span>
          <span style="color:#aaa">${{(f.finish[i]*100).toFixed(0)}}%</span>
        </div>`).join('')}}
      </div>` : ''}}
    </div>
  `;

  return div;
}}

function plotRadar(f, idx) {{
  const axes  = f.radar.axes.concat([f.radar.axes[0]]);
  const red_r = f.radar.red.concat([f.radar.red[0]]);
  const blue_r = f.radar.blue.concat([f.radar.blue[0]]);
  const units = f.radar.units;

  const hoverLines = (vals) => f.radar.axes.map((a,i) =>
    `<b>${{a}}</b>: ${{vals[i]}}${{units[i]}}`).join('<br>');

  Plotly.newPlot('radar-' + idx, [
    {{
      type:'scatterpolar', mode:'lines+markers', name: f.red_name,
      r: red_r, theta: axes, fill:'toself',
      line:{{color:RED, width:2}}, fillcolor:'rgba(231,76,60,0.15)',
      marker:{{color:RED, size:5}},
      hovertemplate: hoverLines(f.radar.red) + '<extra>' + f.red_name + '</extra>',
    }},
    {{
      type:'scatterpolar', mode:'lines+markers', name: f.blue_name,
      r: blue_r, theta: axes, fill:'toself',
      line:{{color:BLUE, width:2}}, fillcolor:'rgba(52,152,219,0.15)',
      marker:{{color:BLUE, size:5}},
      hovertemplate: hoverLines(f.radar.blue) + '<extra>' + f.blue_name + '</extra>',
    }},
  ], radarLayout(), plotCfg);
}}

function plotBars(f, idx) {{
  const stats = f.bars.stats;
  const units = f.bars.units;

  Plotly.newPlot('bars-' + idx, [
    {{
      type:'bar', name: f.red_name,
      x: f.bars.red, y: stats, orientation:'h',
      marker:{{color:'rgba(231,76,60,0.8)'}},
      hovertemplate: stats.map((s,i) =>
        `<b>${{s}}</b>: ${{f.bars.red[i]}}${{units[i]}}<extra></extra>`),
    }},
    {{
      type:'bar', name: f.blue_name,
      x: f.bars.blue, y: stats, orientation:'h',
      marker:{{color:'rgba(52,152,219,0.8)'}},
      hovertemplate: stats.map((s,i) =>
        `<b>${{s}}</b>: ${{f.bars.blue[i]}}${{units[i]}}<extra></extra>`),
    }},
  ], barLayout(), plotCfg);
}}

function activate(idx) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelector('#tab-' + idx).classList.add('active');
  document.querySelector('#panel-' + idx).classList.add('active');
  plotRadar(FIGHTS[idx], idx);
  plotBars(FIGHTS[idx],  idx);
}}

const tabsEl   = document.getElementById('tabs');
const panelsEl = document.getElementById('panels');

FIGHTS.forEach((f, i) => {{
  const btn = document.createElement('button');
  btn.className = 'tab-btn';
  btn.id = 'tab-' + i;
  btn.textContent = f.label;
  btn.onclick = () => activate(i);
  tabsEl.appendChild(btn);

  panelsEl.appendChild(buildPanel(f, i));
}});

activate(0);
</script>
</body>
</html>
"""


def generate_html(
    event: dict,
    results: list[dict],
    model_type: str,
    out_path: Path,
) -> None:
    html = _HTML_TEMPLATE.format(
        event_name      = event["name"] + " -- " + _format_date(event["date"]),
        model_label     = MODEL_LABELS.get(model_type, model_type),
        generated       = date.today().isoformat(),
        n_fights        = len(results),
        fight_data_json = _fight_data_for_js(results),
        finish_names_json = json.dumps(FINISH_CLASS_NAMES),
    )
    out_path.write_text(html, encoding="utf-8")


# ── Markdown formatter ────────────────────────────────────────────────────────

def _event_slug(name: str, event_date: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{slug}-{event_date}"


def _format_finish(finish_proba: list[float] | None) -> str:
    if not finish_proba:
        return "N/A"
    pairs = sorted(zip(FINISH_CLASS_NAMES, finish_proba), key=lambda x: -x[1])
    parts = [f"{name} ({p:.0%})" for name, p in pairs[:2]]
    return " / ".join(parts)


def _format_date(iso: str) -> str:
    try:
        d = date.fromisoformat(iso)
        return f"{d.strftime('%B')} {d.day}, {d.year}"
    except Exception:
        return iso


def build_markdown(
    event: dict,
    results: list[dict],
    model_type: str,
    html_name: str | None = None,
) -> str:
    model_label    = MODEL_LABELS.get(model_type, model_type)
    generated      = date.today().isoformat()
    event_date_fmt = _format_date(event["date"])

    lines = [
        f"# {event['name']} -- {event_date_fmt}",
        "",
        f"Model: {model_label} | Generated: {generated}",
        "",
        "Fighters making their UFC debut were excluded (no historical stats in DB).",
        "",
    ]

    if html_name:
        lines += [
            f"> Interactive fighter comparison: [{html_name}](./{html_name})",
            "",
        ]

    lines += [
        "---",
        "",
        "## Predictions",
        "",
        "| Fight | Predicted Winner | Confidence | Likely Method |",
        "|---|---|---|---|",
    ]

    for r in results:
        fight_label = f"{r['red_name']} vs {r['blue_name']}"
        finish_str  = _format_finish(r["finish_proba"])
        lines.append(
            f"| {fight_label} | {r['winner']} | {r['confidence']:.1%} | {finish_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Raw Model Output",
        "",
    ]

    for r in results:
        form_r     = r["form_red"]
        form_b     = r["form_blue"]
        finish_str = _format_finish(r["finish_proba"])

        lines += [
            f"### {r['red_name']} vs {r['blue_name']}",
            f"- ELO: {r['red_name']} {r['elo_red']:.0f} | {r['blue_name']} {r['elo_blue']:.0f}",
            (
                f"- Recent form: "
                f"{r['red_name']} win_rate={form_r['recent_win_rate']:.0%} "
                f"finish_rate={form_r['recent_finish_rate']:.0%} "
                f"streak={int(form_r['win_streak'])} | "
                f"{r['blue_name']} win_rate={form_b['recent_win_rate']:.0%} "
                f"finish_rate={form_b['recent_finish_rate']:.0%} "
                f"streak={int(form_b['win_streak'])}"
            ),
            f"- {model_label}: {r['red_name']} {r['red_prob']:.1%} | {r['blue_name']} {r['blue_prob']:.1%}",
            f"- Finish: {finish_str}",
            "",
        ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the next UFC event card.")
    parser.add_argument("--model", default="ensemble",
                        choices=["xgb", "lr", "rf", "lgbm", "ensemble"],
                        help="Model to use (default: ensemble)")
    parser.add_argument("--output", default=None,
                        help="Output .md path (default: predictions/<slug>.md)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Exit 0 without regenerating if predictions for this event already exist")
    args = parser.parse_args()

    log.info("Scraping next upcoming event from UFCStats...")
    event = scrape_upcoming_event()
    if not event:
        print("[ERROR]  No upcoming events found on UFCStats.")
        sys.exit(1)

    print(f"\nEvent: {event['name']}  ({event['date']})")
    print(f"Fights on card: {len(event['fights'])}\n")

    if not event["fights"]:
        print("[ERROR]  No fights found on the card. The page structure may have changed.")
        sys.exit(1)

    if not DB_V1_PATH.exists() or not MODELS_V1_DIR.exists():
        print("[ERROR]  v1 DB or models not found. Run the v1 training pipeline first.")
        sys.exit(1)

    _models_dir = MODELS_V1_PROD_DIR if MODELS_V1_PROD_DIR.exists() and any(MODELS_V1_PROD_DIR.iterdir()) else MODELS_V1_DIR

    conn    = sqlite3.connect(str(DB_PATH))
    results = []
    skipped = []

    for fight in event["fights"]:
        r_row = lookup_fighter(conn, fight["r_id"])
        b_row = lookup_fighter(conn, fight["b_id"])

        r_db_name = r_row[1] if r_row else None
        b_db_name = b_row[1] if b_row else None

        r_debut = (r_row is None) or (not has_fight_history(conn, fight["r_id"]))
        b_debut = (b_row is None) or (not has_fight_history(conn, fight["b_id"]))

        r_label = r_db_name or fight["r_name"]
        b_label = b_db_name or fight["b_name"]

        if r_debut or b_debut:
            debut_names = [n for n, d in [(r_label, r_debut), (b_label, b_debut)] if d]
            skipped.append(f"{r_label} vs {b_label} (debut: {', '.join(debut_names)})")
            log.info("Skipping %s vs %s -- debut(s): %s", r_label, b_label, debut_names)
            continue

        print(f"  Predicting: {r_db_name} vs {b_db_name} ({fight['division'] or 'unknown div'})")
        try:
            result = compute_prediction(
                red_name=r_db_name,
                blue_name=b_db_name,
                model_type=args.model,
                division=fight["division"] or None,
                title_fight=fight["title_fight"],
                db_path=DB_V1_PATH,
                models_dir=_models_dir,
            )
            results.append(result)
        except SystemExit:
            log.warning("compute_prediction exited for %s vs %s -- skipping", r_db_name, b_db_name)
            skipped.append(f"{r_db_name} vs {b_db_name} (prediction error)")

    conn.close()

    if skipped:
        print(f"\nSkipped {len(skipped)} fight(s):")
        for s in skipped:
            print(f"  - {s}")

    if not results:
        print("\n[WARN]  No fights to predict after filtering.")
        sys.exit(0)

    # Determine paths
    if args.output:
        out_path = Path(args.output)
    else:
        PREDICTIONS_DIR.mkdir(exist_ok=True)
        slug      = _event_slug(event["name"], event["date"])
        event_dir = PREDICTIONS_DIR / slug
        if args.skip_existing and event_dir.exists():
            print(f"Predictions already exist for {slug} -- skipping (--skip-existing).")
            sys.exit(0)
        event_dir.mkdir(exist_ok=True)
        out_path  = event_dir / f"{slug}.md"

    json_path = out_path.with_suffix(".json")
    html_path = out_path.with_suffix(".html")

    # Save raw fight data so the dashboard can be regenerated without re-scraping
    meta = {
        "event":      event["name"],
        "date":       event["date"],
        "event_url":  event.get("url", ""),
        "model_type": args.model,
        "fights":     json.loads(_fight_data_for_js(results)),
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Generate interactive HTML
    generate_html(event, results, args.model, html_path)
    print(f"\nInteractive dashboard: {html_path}")

    # Generate markdown (with link to HTML)
    md = build_markdown(event, results, args.model, html_name=html_path.name)
    out_path.write_text(md, encoding="utf-8")
    print(f"Predictions written to: {out_path}")
    print(f"Fight data JSON:        {json_path}")


if __name__ == "__main__":
    main()
