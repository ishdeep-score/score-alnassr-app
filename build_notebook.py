"""
Generates PostMatchAnalysis.ipynb for the Al Khaleej vs Al Nassr match.
Run with: python build_notebook.py
"""
import json, os

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), 'PostMatchAnalysis.ipynb')

def cell(cell_type, source, outputs=None):
    if cell_type == 'markdown':
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source,
    }

def md(text): return cell('markdown', text)
def code(text): return cell('code', text)

cells = []

# ── Title & Narrative ─────────────────────────────────────────────────────────
cells.append(md("""# Al Khaleej vs Al Nassr — Post-Match Analysis
**Saudi Pro League · Matchday 26 · March 14, 2026**

---

## The Story of the Match

Al Nassr delivered a **dominant 5-0 away victory** over Al Khaleej in one of the most comprehensive performances of the Saudi Pro League season. The result was never in doubt after Abdullah Al Hamdan's 30th-minute opener, but the margin of victory tells a deeper story of tactical superiority and individual brilliance.

### Pre-Match Context
Al Nassr arrived in Matchday 26 needing points to maintain their title challenge, while Al Khaleej were struggling in the lower half of the table. Al Nassr's quality — headlined by João Félix and Kingsley Coman on the flanks — was always likely to be too much for a depleted Al Khaleej side.

### First Half (0–1)
The first half was a study in controlled dominance. Al Nassr immediately established their 4-2-3-1 shape, pressing high with Félix and Coman stretching Al Khaleej's back line. Al Khaleej had only 6 shots all match and barely threatened, while Al Nassr created chance after chance — with Félix hitting the target twice inside the first 5 minutes.

The breakthrough came in the **30th minute**: Abdullah Al Hamdan finished clinically after a slick combination through the lines (xG: 0.27). Al Khaleej's yellow card for Al Hamsal at 40' signalled their growing frustration.

### Second Half (0–4 → 0–5)
The floodgates opened after half-time:
- **53'** — Ayman Yahya controlled and finished to make it 2-0 (xG: 0.35)
- **72'** — João Félix with a composed slot — arguably the goal of the match (xG: 0.60)
- **78'** — Félix again, tap-in after excellent approach play (xG: 0.05 post-shot)
- **90'** — Ângelo with the fifth to cap a brilliant team display (xG: 0.10)

### Why Al Nassr Won So Convincingly
1. **Pressing intensity**: PPDA of **0.54** — allowing just 0.54 passes per defensive action in the opponent's half
2. **xG dominance**: Al Nassr 2.80 vs Al Khaleej 0.51 — 5.5× more expected goals
3. **Pass quality**: 89.1% pass accuracy with 115+ progressive passes breaking lines
4. **Coman as creator**: 14 Shot-Creating Actions — the highest individual figure on the pitch
5. **Al Khaleej's fragility**: Substitutions as early as the 29th minute showed their structure was already breaking
"""))

# ── Setup ─────────────────────────────────────────────────────────────────────
cells.append(code("""\
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
%matplotlib inline

from wyscout_parser import load_match, parse_all, compute_momentum, get_starting_xi
from visuals import *

DATA_PATH = '5758824.json'
PLOT_PATH = '/Users/ishdeepchadha/Documents/Score/Plots/AlNassr'
os.makedirs(PLOT_PATH, exist_ok=True)

data  = load_match(DATA_PATH)
match = parse_all(data)

TEAM_COLORS = {
    'home': '#00A651',   # Al Khaleej — green
    'away': '#C8A400',   # Al Nassr   — gold
}

home_id = match.meta['home_id']
away_id = match.meta['away_id']

print(f"Match: {match.meta['home_name']} {match.meta['home_score']}-{match.meta['away_score']} {match.meta['away_name']}")
print(f"Events: {len(match.events_df)}  |  Players: {len(match.player_lookup)}")
print(f"Home PPDA: {match.ppda.get(home_id, 'N/A'):.2f}  |  Away PPDA: {match.ppda.get(away_id, 'N/A'):.2f}")
"""))

# ── Dashboard 1: Match Story ──────────────────────────────────────────────────
cells.append(md("""\
## Dashboard 1: Match Story

This dashboard captures the full arc of the match.

- **Match Header**: Score, teams, date, venue, competition and referee at a glance.
- **Timeline**: Every goal, card and substitution mapped across 90 minutes. Al Khaleej's early subs (29′, 37′) hint at a team that lost the tactical battle quickly.
- **xG Race**: The cumulative expected goals curve shows Al Nassr pulling away from the first shot. The flatness of Al Khaleej's line tells the full story.
- **Net Momentum**: 5-minute rolling event counts showing who dominated each phase.
- **Key Stats**: Shot counts, pass accuracy, possession, PPDA compared side by side.
- **Rolling Stats**: Four time-series charts — possession %, duel win rate, attacking tempo, and pass accuracy — showing how Al Nassr's dominance evolved over 90 minutes.
"""))
cells.append(code("""\
fig = render_match_story_dashboard(match, data, TEAM_COLORS, figsize=(20, 34))
fig.savefig(f'{PLOT_PATH}/01_dashboard_match_story.png', dpi=150, bbox_inches='tight', facecolor='#010b14')
plt.show()
"""))

# ── Dashboard 2: Shots & Attacking ───────────────────────────────────────────
cells.append(md("""\
## Dashboard 2: Shots & Attacking

This dashboard dissects how each team created and converted their chances.

- **Shot Map**: Both teams on a single pitch — home attacks right, away attacks left. Each shot is numbered and a line extends to the goal mouth. Al Nassr's 20 shots vs Al Khaleej's 6 shows the gulf in attacking output.
- **Shot Breakdown**: Every shot numbered with player, minute, xG, body part and result. All 5 Al Nassr goals are visible in the away column.
- **SCA by Player**: Shot-Creating Actions — the last 2 actions before each shot. Coman's count of 14 is exceptional; he was the engine behind almost every Al Nassr chance.
- **xT by Player**: Expected Threat generated per player, split into pass xT (full color) and carry xT (lighter shade). Shows who moved the ball into dangerous areas most effectively.
"""))
cells.append(code("""\
fig = render_shots_dashboard(match, TEAM_COLORS, figsize=(20, 38))
fig.savefig(f'{PLOT_PATH}/02_dashboard_shots.png', dpi=150, bbox_inches='tight', facecolor='#010b14')
plt.show()
"""))

# ── Dashboard 3: Possession & Passing ────────────────────────────────────────
cells.append(md("""\
## Dashboard 3: Possession & Passing

This dashboard examines how each team structured possession and moved the ball through the pitch.

- **Pass Networks**: One vertical pitch per team showing the full-match passing structure across all players (starters and substitutes). Node size = touches (passes played + received). Line width = cumulative xT through that connection. Square markers indicate substitutes. Al Nassr's network is dense through the double pivot and the wide channels — Coman and Félix constantly in combination. Al Khaleej's network is fragmented by comparison, with limited connections beyond the defensive block.
- **Offensive Action Map**: Two pitches side by side, one per team. All passes and carries plotted at low opacity, with progressive actions, key passes, and box-ending moves highlighted brighter. Corner kicks (★) and free kicks (◆) are marked at their delivery positions.
"""))
cells.append(code("""\
fig = render_possession_dashboard(match, data, TEAM_COLORS, figsize=(20, 28))
fig.savefig(f'{PLOT_PATH}/03_dashboard_possession.png', dpi=150, bbox_inches='tight', facecolor='#010b14')
plt.show()
"""))

# ── Dashboard 4: Defensive ────────────────────────────────────────────────────
cells.append(md("""\
## Dashboard 4: Defensive

This dashboard examines how each team defended and applied pressure.

- **Defensive Actions Map**: Interceptions, clearances and duel wins plotted on the pitch. Al Nassr's defensive actions cluster in the final third — a sign of high pressing, winning the ball in dangerous areas. Al Khaleej's clearances are concentrated in their own box, reflecting a team under sustained siege.
- **Pressing (PPDA) Map**: Density of pressing actions on the pitch. Al Nassr's heat concentrates in Al Khaleej's half — they won the ball high and transitioned to attack immediately. PPDA of **0.54** is elite-level by any standard.
- **PPDA Summary**: Al Nassr 0.54 vs Al Khaleej 5.39 — the press differential tells the tactical story of this match.
"""))
cells.append(code("""\
fig = render_defensive_dashboard(match, TEAM_COLORS, figsize=(20, 24))
fig.savefig(f'{PLOT_PATH}/04_dashboard_defensive.png', dpi=150, bbox_inches='tight', facecolor='#010b14')
plt.show()
"""))

# ── Dashboard 5: Player Reports ───────────────────────────────────────────────
cells.append(md("""\
## Dashboard 5: Player Reports

Individual performance profiles for the match's standout players.

- **Starting XI Formations**: Both teams' initial shapes. Al Nassr's 4-2-3-1 with Félix behind the striker and Coman wide gave them full-pitch width that Al Khaleej's 4-4-2 couldn't match.
- **Player Stats Tables**: Touches, pass accuracy, shots, xG, key passes, progressive passes, duels, interceptions, SCA and xT for every player — home and away.
- **João Félix** *(top xG)*: 9 shots, 2 goals. The action map shows constant movement between the lines. The radar highlights his dominance in xG, shots and SCA within the Al Nassr squad.
- **Kingsley Coman** *(top SCA)*: 14 Shot-Creating Actions. Wide-right-to-inside-channel carry pattern visible on the action map. The radar shows his contribution was creation rather than conversion — the perfect foil for Félix.
"""))
cells.append(code("""\
fig = render_player_dashboard(match, data, TEAM_COLORS, figsize=(20, 46))
fig.savefig(f'{PLOT_PATH}/05_dashboard_players.png', dpi=150, bbox_inches='tight', facecolor='#010b14')
plt.show()
"""))

# ── Tactical Verdict ──────────────────────────────────────────────────────────
cells.append(md("""\
## Tactical Verdict

### Al Nassr's Winning Formula
| Factor | Al Nassr | Al Khaleej |
|--------|----------|------------|
| xG | 2.80 | 0.51 |
| PPDA | **0.54** (elite press) | 5.39 |
| Pass Accuracy | 89.1% | 82.6% |
| Progressive Passes | 115+ | 68 |
| Shots | 20 | 6 |
| SCA leader | Coman — 14 | — |

**Al Nassr's 4-2-3-1 pressed with incredible intensity** (PPDA 0.54 is among the best in world football on any given match), forcing turnovers high up the pitch and creating a succession of high-quality chances. Al Khaleej simply couldn't handle the pace and technical quality of Félix–Coman on the flanks.

**João Félix** was the standout player — 9 shots, 2 goals, multiple key chances created. **Kingsley Coman** was the engine: 14 Shot-Creating Actions, pulling defenders wide and opening lanes through the middle.

For **Al Khaleej**, the early substitutions (from 29') suggested the coaching staff saw the tactical battle was lost. Their lack of structure in the pressing phase allowed Al Nassr to build freely.

> *Result: A comprehensive Al Nassr performance — the xG margin (2.80 vs 0.51) shows this was no fluke.*
"""))

cells.append(code("""\
print("All dashboards saved to:", PLOT_PATH)
for f in sorted(f for f in os.listdir(PLOT_PATH) if f.startswith('0') and f.endswith('.png')):
    sz = os.path.getsize(f'{PLOT_PATH}/{f}') // 1024
    print(f"  {f}  ({sz} KB)")
"""))

# ── Write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "footy", "language": "python", "name": "footy"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written: {NOTEBOOK_PATH}  ({len(cells)} cells)")
