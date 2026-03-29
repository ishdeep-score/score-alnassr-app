"""
visuals.py
==========
All visualization functions for the Wyscout post-match report.
Used by both PostMatchAnalysis.ipynb and app.py.

Each render_* function returns a matplotlib Figure object.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from mplsoccer import Pitch, VerticalPitch
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SORA_DIR = None  # custom font disabled — using matplotlib default

BG         = '#010b14'
PITCH_BG   = '#0a1628'
PANEL_BG   = '#0d1f35'
LINE_COLOR = '#3a5a7a'
TEXT_MAIN  = '#e8e8e8'
TEXT_DIM   = '#7a8a9a'

# Goal zone lookup: (goalY_fraction, goalZ_fraction) of goal face (0-1 scale)
# Goal: 7.32m wide, 2.44m high
# Zones: gb=goal box, gc=goal center, glb=goal left box, gbr=goal right box
# otl=out top left, olb=out left box (off-target approximations)
GOAL_ZONE_COORDS = {
    'gc':  (0.50, 0.45),   # center, mid height — on target
    'gb':  (0.25, 0.30),   # left of center, low — on target
    'gbr': (0.75, 0.30),   # right of center, low — on target
    'glb': (0.15, 0.50),   # far left, mid — on target
    'bc':  (0.50, 0.85),   # center, high — off target (over crossbar)
    'plb': (0.20, 0.85),   # left, high — off target
    'otl': (0.05, 1.10),   # far left, over — off target
    'olb': (-0.10, 0.50),  # wide left — off target
    'obr': (1.10, 0.50),   # wide right — off target
}


def _load_font(style: str = 'Regular') -> fm.FontProperties:
    return fm.FontProperties()


def _pitch(**kwargs) -> Pitch:
    defaults = dict(
        pitch_type='wyscout',
        pitch_color=PITCH_BG,
        line_color=LINE_COLOR,
        line_zorder=2,
    )
    defaults.update(kwargs)
    return Pitch(**defaults)


def _fig_bg(fig: plt.Figure) -> plt.Figure:
    fig.patch.set_facecolor(BG)
    return fig


def _team_color(team_side: str, team_colors: dict) -> str:
    return team_colors.get(team_side, '#e8e8e8')


# ---------------------------------------------------------------------------
# 1. Match Header
# ---------------------------------------------------------------------------

def render_match_header(meta: dict, stats: dict, team_colors: dict,
                        figsize=(18, 5)) -> plt.Figure:
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')
    fp_semi = _load_font('SemiBold')

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    # Background panel
    panel = FancyBboxPatch((0.02, 0.05), 0.96, 0.90,
                            boxstyle='round,pad=0.01',
                            facecolor=PANEL_BG, edgecolor=LINE_COLOR, lw=1.5)
    ax.add_patch(panel)

    # Competition & Date
    ax.text(0.5, 0.90, f"Saudi Pro League  •  Matchday {meta.get('gameweek', '')}  •  {meta.get('date_str', '')}",
            ha='center', va='center', fontproperties=fp_reg, fontsize=15, color=TEXT_DIM, transform=ax.transAxes)

    # Home team name
    ax.text(0.22, 0.62, meta['home_name'],
            ha='center', va='center', fontproperties=fp_bold, fontsize=20,
            color=home_color, transform=ax.transAxes)

    # Score
    score_str = f"{meta['home_score']}  –  {meta['away_score']}"
    ax.text(0.5, 0.60, score_str,
            ha='center', va='center', fontproperties=fp_bold, fontsize=36,
            color=TEXT_MAIN, transform=ax.transAxes)

    # HT score
    ht_str = f"HT: {meta['home_score_ht']} – {meta['away_score_ht']}"
    ax.text(0.5, 0.44, ht_str,
            ha='center', va='center', fontproperties=fp_reg, fontsize=16,
            color=TEXT_DIM, transform=ax.transAxes)

    # Away team name
    ax.text(0.78, 0.62, meta['away_name'],
            ha='center', va='center', fontproperties=fp_bold, fontsize=20,
            color=away_color, transform=ax.transAxes)

    # Formations
    ax.text(0.22, 0.32, meta.get('home_formation_1h', ''),
            ha='center', va='center', fontproperties=fp_semi, fontsize=16,
            color=home_color, alpha=0.9, transform=ax.transAxes)
    ax.text(0.78, 0.32, meta.get('away_formation_1h', ''),
            ha='center', va='center', fontproperties=fp_semi, fontsize=16,
            color=away_color, alpha=0.9, transform=ax.transAxes)

    # Coaches
    home_coach = meta.get('home_coach', '')
    away_coach = meta.get('away_coach', '')
    ax.text(0.22, 0.20, f"Coach: {home_coach}" if home_coach else '',
            ha='center', va='center', fontproperties=fp_reg, fontsize=15,
            color=TEXT_DIM, transform=ax.transAxes)
    ax.text(0.78, 0.20, f"Coach: {away_coach}" if away_coach else '',
            ha='center', va='center', fontproperties=fp_reg, fontsize=15,
            color=TEXT_DIM, transform=ax.transAxes)

    # Venue
    venue = meta.get('venue', '')
    if venue:
        ax.text(0.5, 0.20, f"Venue: {venue}",
                ha='center', va='center', fontproperties=fp_reg, fontsize=15,
                color=TEXT_DIM, transform=ax.transAxes)

    # Divider line
    ax.plot([0.5, 0.5], [0.15, 0.85], color=LINE_COLOR, lw=1, alpha=0.5, transform=ax.transAxes)

    return fig


# ---------------------------------------------------------------------------
# 2. Key Events Timeline
# ---------------------------------------------------------------------------

def render_key_events_timeline(goals_df: pd.DataFrame, cards_df: pd.DataFrame,
                                subs_df: pd.DataFrame, meta: dict,
                                team_colors: dict, figsize=(18, 4)) -> plt.Figure:
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')
    home_id = meta['home_id']
    away_id = meta['away_id']

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-3, 93)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')

    # Title
    ax.text(45, 1.35, 'Match Timeline', ha='center', va='center',
            fontproperties=fp_bold, fontsize=13, color=TEXT_MAIN)

    # Main timeline line
    ax.plot([0, 90], [0, 0], color=LINE_COLOR, lw=3, solid_capstyle='round', zorder=1)

    # Half-time marker
    ax.axvline(45, color=TEXT_DIM, lw=1.5, alpha=0.5, ls='--', zorder=1)
    ax.text(45, -0.35, 'HT', ha='center', fontproperties=fp_reg, fontsize=8, color=TEXT_DIM)

    # Minute markers
    for m in [0, 15, 30, 45, 60, 75, 90]:
        ax.plot([m, m], [-0.1, 0.1], color=LINE_COLOR, lw=1, alpha=0.4, zorder=1)
        ax.text(m, -0.25, f"{m}'", ha='center', fontproperties=fp_reg, fontsize=8, color=TEXT_DIM)

    # Goals
    if not goals_df.empty:
        for _, g in goals_df.iterrows():
            is_home = g['team_id'] == home_id
            color = home_color if is_home else away_color
            ypos = 0.55 if is_home else -0.55
            mm = min(max(g['match_minute'], 0), 90)
            ax.scatter(mm, ypos, s=200, c=color, marker='*', zorder=5, edgecolors='white', lw=0.5)
            ax.text(mm, ypos + (0.25 if is_home else -0.25),
                    f"{g['match_minute']}'\n{g['player_name'].split()[-1]}",
                    ha='center', va='center', fontproperties=fp_reg, fontsize=8,
                    color=color, zorder=6)
            # Vertical connection
            ax.plot([mm, mm], [0, ypos - (0.05 if is_home else -0.05)],
                    color=color, lw=1, alpha=0.5, zorder=2)

    # Cards
    if not cards_df.empty:
        for _, c in cards_df.iterrows():
            is_home = c['team_id'] == home_id
            mm = min(max(c['match_minute'], 0), 90)
            card_color = '#FFD700' if c['card_type'] == 'yellow' else '#FF2020'
            ypos = 0.55 if is_home else -0.55
            card_rect = Rectangle((mm - 0.5, ypos - 0.12), 1, 0.18,
                                    facecolor=card_color, edgecolor='white',
                                    lw=0.5, zorder=5)
            ax.add_patch(card_rect)
            ax.text(mm, ypos - 0.25,
                    f"{c['match_minute']}'\n{c['player_name'].split()[-1]}",
                    ha='center', va='center', fontproperties=fp_reg, fontsize=7,
                    color=TEXT_DIM, zorder=6)

    # Substitutions
    if not subs_df.empty:
        for _, s in subs_df.iterrows():
            is_home = s['team_id'] == home_id
            mm = min(max(s['match_minute'], 0), 90)
            color = home_color if is_home else away_color
            ypos = 0.05 if is_home else -0.05
            ax.scatter(mm, ypos, s=80, c='#4CAF50', marker='^', zorder=4,
                       edgecolors='white', lw=0.4)

    # Team labels
    ax.text(-1, 0.55, meta['home_name'], ha='right', va='center',
            fontproperties=fp_bold, fontsize=10, color=home_color)
    ax.text(-1, -0.55, meta['away_name'], ha='right', va='center',
            fontproperties=fp_bold, fontsize=10, color=away_color)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#e8e8e8', markersize=9, label='Goal', lw=0),
        Rectangle((0, 0), 1, 1, facecolor='#FFD700', label='Yellow Card'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#4CAF50', markersize=7, label='Sub', lw=0),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              facecolor=PANEL_BG, edgecolor=LINE_COLOR, labelcolor=TEXT_DIM,
              prop=fp_reg, framealpha=0.8)

    return fig


# ---------------------------------------------------------------------------
# 3. Key Stats Comparison
# ---------------------------------------------------------------------------

def render_key_stats(stats: dict, meta: dict, team_colors: dict,
                      ppda: dict = None, figsize=(14, 9)) -> plt.Figure:
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')
    hs = stats[home_id]
    aws = stats[away_id]

    stat_labels = [
        ('Possession %',    hs['possession_pct'],       aws['possession_pct'],     100),
        ('Shots',           hs['shots'],                aws['shots'],               None),
        ('Shots on Target', hs['shots_on_target'],      aws['shots_on_target'],     None),
        ('xG',              hs['xg'],                   aws['xg'],                  None),
        ('Passes',          hs['passes'],               aws['passes'],              None),
        ('Pass Accuracy %', hs['pass_accuracy'],        aws['pass_accuracy'],       100),
        ('Key Passes',      hs['key_passes'],           aws['key_passes'],          None),
        ('Progressive Pass',hs['progressive_passes'],  aws['progressive_passes'],  None),
        ('Crosses',         hs['crosses'],              aws['crosses'],             None),
        ('Corners',         hs['corners'],              aws['corners'],             None),
        ('Fouls',           hs['fouls'],                aws['fouls'],               None),
        ('Interceptions',   hs['interceptions'],        aws['interceptions'],       None),
        ('Aerial Duels Won',hs['aerial_duels_won'],     aws['aerial_duels_won'],    None),
    ]
    if ppda:
        # Lower PPDA = better press, so invert for bar length
        stat_labels.append(('PPDA', ppda.get(home_id, 0), ppda.get(away_id, 0), None))

    n = len(stat_labels)
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    row_h = 1.0 / (n + 2)
    y_start = 1.0 - row_h * 1.5

    # Headers
    ax.text(0.22, y_start + row_h * 0.6, meta['home_name'],
            ha='center', va='center', fontproperties=fp_bold, fontsize=12,
            color=home_color, transform=ax.transAxes)
    ax.text(0.78, y_start + row_h * 0.6, meta['away_name'],
            ha='center', va='center', fontproperties=fp_bold, fontsize=12,
            color=away_color, transform=ax.transAxes)
    ax.text(0.5, y_start + row_h * 0.6, 'Stat',
            ha='center', va='center', fontproperties=fp_bold, fontsize=12,
            color=TEXT_DIM, transform=ax.transAxes)

    for i, (label, hval, aval, max_val) in enumerate(stat_labels):
        y = y_start - i * row_h
        total = max_val if max_val else (hval + aval) or 1

        h_frac = hval / total
        a_frac = aval / total

        # Bar width = 0.38, centered at 0.12 (home) and 0.88 (away)
        bar_max = 0.36

        # Home bar (grows from center-left outward)
        h_bar_w = bar_max * h_frac / max(h_frac + a_frac, 0.001) * 2
        h_bar_w = min(h_bar_w, bar_max)
        ax.barh(y, -h_bar_w, left=0.5, height=row_h * 0.5,
                color=home_color, alpha=0.75, transform=ax.transAxes)

        # Away bar
        a_bar_w = bar_max * a_frac / max(h_frac + a_frac, 0.001) * 2
        a_bar_w = min(a_bar_w, bar_max)
        ax.barh(y, a_bar_w, left=0.5, height=row_h * 0.5,
                color=away_color, alpha=0.75, transform=ax.transAxes)

        # Values
        hv_str = f"{hval:.1f}" if isinstance(hval, float) else str(hval)
        av_str = f"{aval:.1f}" if isinstance(aval, float) else str(aval)
        ax.text(0.5 - h_bar_w - 0.02, y, hv_str, ha='right', va='center',
                fontproperties=fp_bold, fontsize=11, color=home_color,
                transform=ax.transAxes)
        ax.text(0.5 + a_bar_w + 0.02, y, av_str, ha='left', va='center',
                fontproperties=fp_bold, fontsize=11, color=away_color,
                transform=ax.transAxes)

        # Label
        ax.text(0.5, y, label, ha='center', va='center',
                fontproperties=fp_reg, fontsize=11, color=TEXT_MAIN,
                transform=ax.transAxes, zorder=5)

        # Separator
        ax.plot([0.05, 0.95], [y - row_h * 0.35, y - row_h * 0.35],
                color=LINE_COLOR, lw=0.4, alpha=0.3, transform=ax.transAxes)

    ax.text(0.5, 0.02, 'PPDA: lower = more aggressive press',
            ha='center', va='bottom', fontproperties=fp_reg, fontsize=10,
            color=TEXT_DIM, transform=ax.transAxes, style='italic')

    return fig


# ---------------------------------------------------------------------------
# 4. xG Race Chart (matplotlib)
# ---------------------------------------------------------------------------

def render_xg_race(xg_timeline: pd.DataFrame, goals_df: pd.DataFrame,
                   meta: dict, team_colors: dict, figsize=(14, 5)) -> plt.Figure:
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(PITCH_BG)
    ax.spines[['top', 'right']].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(LINE_COLOR)

    if not xg_timeline.empty:
        ax.plot(xg_timeline['match_minute'], xg_timeline['home_xg_cumul'],
                color=home_color, lw=2.5, label=meta['home_name'])
        ax.fill_between(xg_timeline['match_minute'], xg_timeline['home_xg_cumul'],
                         alpha=0.15, color=home_color)
        ax.plot(xg_timeline['match_minute'], xg_timeline['away_xg_cumul'],
                color=away_color, lw=2.5, label=meta['away_name'])
        ax.fill_between(xg_timeline['match_minute'], xg_timeline['away_xg_cumul'],
                         alpha=0.15, color=away_color)

    # Goal markers
    if not goals_df.empty:
        home_cum = 0.0
        away_cum = 0.0
        if not xg_timeline.empty:
            for _, g in goals_df.iterrows():
                mm = g['match_minute']
                row = xg_timeline[xg_timeline['match_minute'] <= mm]
                if not row.empty:
                    hc = row.iloc[-1]['home_xg_cumul']
                    ac = row.iloc[-1]['away_xg_cumul']
                else:
                    hc, ac = 0, 0
                is_home = g['team_id'] == home_id
                color = home_color if is_home else away_color
                yval = hc if is_home else ac
                ax.axvline(mm, color=color, lw=1, alpha=0.4, ls=':')
                ax.scatter(mm, yval, s=80, c=color, marker='*', zorder=5,
                           edgecolors='white', lw=0.5)
                ax.text(mm, yval + 0.05, f"{g['player_name'].split()[-1]}",
                        ha='center', fontproperties=fp_reg, fontsize=9,
                        color=color)

    # HT line
    ax.axvline(45, color=TEXT_DIM, lw=1, ls='--', alpha=0.5)
    ax.text(45, ax.get_ylim()[1] * 0.5, 'HT', ha='center',
            fontproperties=fp_reg, fontsize=17, color=TEXT_DIM)

    ax.set_xlim(0, 91)
    ax.set_xlabel("Minute", fontproperties=fp_reg, fontsize=16, color=TEXT_DIM)
    ax.set_ylabel("Cumulative xG", fontproperties=fp_reg, fontsize=16, color=TEXT_DIM)
    ax.tick_params(colors=TEXT_DIM)
    ax.set_title('xG Race', fontproperties=fp_bold, fontsize=16, color=TEXT_MAIN, pad=8)
    ax.legend(facecolor=PANEL_BG, edgecolor=LINE_COLOR, labelcolor=TEXT_MAIN,
              prop=fp_reg, loc='upper left')

    return fig


# ---------------------------------------------------------------------------
# 5. Match Momentum
# ---------------------------------------------------------------------------

def render_momentum(momentum_df: pd.DataFrame, meta: dict,
                    team_colors: dict, figsize=(14, 4)) -> plt.Figure:
    """
    Net momentum chart: net = home_events - away_events per 5-min window.
    Positive (above 0) = home dominance; negative = away dominance.
    Bars are colored by whichever team dominated that window.
    """
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(PITCH_BG)
    for spine in ax.spines.values():
        spine.set_color(LINE_COLOR)
    ax.spines[['top', 'right']].set_visible(False)

    if not momentum_df.empty:
        buckets = momentum_df['bucket'].values
        home_ev = momentum_df['home_events'].values
        away_ev = momentum_df['away_events'].values
        net = home_ev - away_ev
        net_abs_max = max(abs(net).max(), 1)

        bar_width = 4.2
        for bk, nv in zip(buckets, net):
            color = home_color if nv >= 0 else away_color
            alpha = min(0.35 + 0.55 * abs(nv) / net_abs_max, 0.92)
            ax.bar(bk, nv, width=bar_width, color=color, alpha=alpha,
                   align='center', zorder=3)

        # Smooth trend line overlay
        if len(net) >= 3:
            smooth = np.convolve(net.astype(float), np.ones(3) / 3, mode='same')
            ax.plot(buckets, smooth, color='white', lw=1.5, alpha=0.55, zorder=4)

        ax.axhline(0, color=LINE_COLOR, lw=1.5, zorder=2)

    ax.axvline(45, color=TEXT_DIM, lw=1, ls='--', alpha=0.5)
    ax.set_xlim(0, 93)
    ax.set_xlabel("Minute", fontproperties=fp_reg, fontsize=13, color=TEXT_DIM)
    ax.set_title('Match Momentum  (Net events per 5-min window)',
                 fontproperties=fp_bold, fontsize=15, color=TEXT_MAIN, pad=8)
    ax.tick_params(colors=TEXT_DIM, labelsize=11)

    if not momentum_df.empty:
        ymax = max(abs(momentum_df['home_events'] - momentum_df['away_events']).max() + 3, 6)
    else:
        ymax = 10
    ax.set_ylim(-ymax, ymax)
    yticks = [v for v in ax.get_yticks() if -ymax <= v <= ymax]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(abs(int(v))) for v in yticks], color=TEXT_DIM,
                       fontproperties=fp_reg)

    # Team labels inside chart
    ax.text(2, ymax * 0.78, meta['home_name'], ha='left', va='center',
            fontproperties=fp_bold, fontsize=12, color=home_color, alpha=0.85)
    ax.text(2, -ymax * 0.78, meta['away_name'], ha='left', va='center',
            fontproperties=fp_bold, fontsize=12, color=away_color, alpha=0.85)
    ax.text(45, ymax * 0.95, 'HT', ha='center', va='top',
            fontproperties=fp_reg, fontsize=11, color=TEXT_DIM)

    return fig


# ---------------------------------------------------------------------------
# 6. Shot Map
# ---------------------------------------------------------------------------

def render_shot_map(df: pd.DataFrame, meta: dict, team_colors: dict,
                    figsize=(16, 10)) -> plt.Figure:
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    pitch = _pitch()
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PITCH_BG)

    shots = df[df['type_primary'] == 'shot'].copy()
    shots = shots.sort_values(['match_minute', 'second']).reset_index(drop=True)
    # Number per team separately
    shots['shot_num'] = shots.groupby('team_id').cumcount() + 1

    # Goal mouth y-range (wyscout 0-100 scale)
    _GOAL_Y0 = 44.62
    _GOAL_YW = 10.76  # goal width in wyscout units

    for _, s in shots.iterrows():
        # Use mirrored coords: home attacks right (x_plot→100), away attacks left (x_plot→0)
        x, y = s['x_plot'], s['y_plot']
        if pd.isna(x) or pd.isna(y):
            continue

        team_id = s['team_id']
        color = home_color if team_id == home_id else away_color
        xg = s['shot_xg'] or 0
        size = 160 + 520 * xg
        num = str(s['shot_num'])
        is_goal = s['shot_is_goal']
        on_target = s['shot_on_target']
        zone = s.get('shot_goal_zone')
        situation = s.get('shot_situation', 'Open Play') or 'Open Play'
        sit_abbr = {'Corner': 'CK', 'Free Kick': 'FK', 'Penalty': 'PK'}.get(situation, '')

        # Situation edge color: set pieces get gold ring
        edge_col = '#FFD700' if sit_abbr else 'white'
        edge_lw  = 1.8 if sit_abbr else 0.8

        # Trajectory line to goal mouth
        if pd.notna(zone) and zone in GOAL_ZONE_COORDS:
            goal_y_frac, _ = GOAL_ZONE_COORDS[zone]
            if team_id == home_id:
                end_x = 100.0
                end_y = _GOAL_Y0 + goal_y_frac * _GOAL_YW
            else:
                end_x = 0.0
                end_y = _GOAL_Y0 + (1.0 - goal_y_frac) * _GOAL_YW
            ax.plot([x, end_x], [y, end_y],
                    color=color, lw=0.68, alpha=0.35, ls='--', zorder=3)

        if is_goal:
            ax.text(x, y, '⚽', ha='center', va='center', fontsize=16, zorder=7)
        elif on_target:
            ax.scatter(x, y, s=size, c=color, marker='o',
                       edgecolors=edge_col, lw=edge_lw, alpha=0.88, zorder=5)
        elif pd.notna(zone) and zone in ('otl', 'olb', 'obr', 'bc', 'plb'):
            ax.scatter(x, y, s=size * 0.75, c='none', marker='X',
                       edgecolors=color, lw=1.4, alpha=0.6, zorder=4)
        else:
            ax.scatter(x, y, s=size * 0.75, c='none', marker='s',
                       edgecolors=color, lw=1.0, alpha=0.55, zorder=4)

        # Shot number
        num_color = 'white' if not is_goal else '#111111'
        ax.text(x, y + (5 if is_goal else 0), num,
                ha='center', va='center', fontsize=7, fontweight='bold',
                color=num_color, zorder=8,
                path_effects=[pe.withStroke(linewidth=1.5,
                    foreground='black' if num_color == 'white' else 'white')])

        # Situation abbreviation for set pieces
        if sit_abbr:
            ax.text(x, y - 6, sit_abbr, ha='center', va='top', fontsize=6,
                    color='#FFD700', zorder=8,
                    path_effects=[pe.withStroke(linewidth=1.2, foreground=PITCH_BG)])

    # Stats callouts — home shots right (x_plot≈70-100), away shots left (x_plot≈0-30)
    for team_id, color, xpos in [(home_id, home_color, 80), (away_id, away_color, 20)]:
        t_shots = shots[shots['team_id'] == team_id]
        goals = int(t_shots['shot_is_goal'].sum())
        sot = int(t_shots['shot_on_target'].sum())
        xg = t_shots['shot_xg'].sum()
        ax.text(xpos, 102, f"{len(t_shots)} shots | {sot} SoT | xG {xg:.2f} | {goals} G",
                ha='center', va='center', fontproperties=fp_reg, fontsize=12,
                color=color)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=9, label='On Target', lw=0),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='none', markeredgecolor='white', markersize=8, label='Off Target', lw=0),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor='white', markersize=8, label='Blocked', lw=0),
        Line2D([0], [0], color='white', lw=1, ls='--', alpha=0.5, label='Shot trajectory'),
        mpatches.Patch(color=home_color, label=meta['home_name']),
        mpatches.Patch(color=away_color, label=meta['away_name']),
    ]
    ax.legend(handles=legend_elements, loc='lower left', ncol=2, fontsize=9,
              facecolor=PANEL_BG, edgecolor=LINE_COLOR, labelcolor=TEXT_MAIN,
              prop=fp_reg, framealpha=0.9)

    ax.text(2,  -4, '——> Attacking', ha='left',  va='top', fontsize=8,
            color=home_color, fontproperties=fp_reg, style='italic')
    ax.text(98, -4, 'Attacking <——', ha='right', va='top', fontsize=8,
            color=away_color, fontproperties=fp_reg, style='italic')
    return fig


# ---------------------------------------------------------------------------
# 7. Goal Frame
# ---------------------------------------------------------------------------

def render_goal_frame(df: pd.DataFrame, meta: dict, team_colors: dict,
                       figsize=(16, 7)) -> plt.Figure:
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=BG)
    fig.patch.set_facecolor(BG)

    shots = df[df['type_primary'] == 'shot'].copy()

    for ax, team_id, color in [(axes[0], home_id, home_color),
                                 (axes[1], away_id, away_color)]:
        ax.set_facecolor(PITCH_BG)
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.3, 1.2)
        ax.axis('off')

        # Goal frame
        goal_rect = Rectangle((0, 0), 1, 1, facecolor='none',
                               edgecolor=color, lw=2.5, zorder=3)
        ax.add_patch(goal_rect)

        # Crossbar
        ax.plot([0, 1], [1, 1], color=color, lw=2.5, zorder=3)

        # Posts
        ax.plot([0, 0], [0, 1], color=color, lw=2.5, zorder=3)
        ax.plot([1, 1], [0, 1], color=color, lw=2.5, zorder=3)

        # Zone grid lines (light)
        for x in [1/3, 2/3]:
            ax.plot([x, x], [0, 1], color=LINE_COLOR, lw=0.5, alpha=0.4, zorder=2)
        ax.plot([0, 1], [0.5, 0.5], color=LINE_COLOR, lw=0.5, alpha=0.4, zorder=2)

        # Plot shots
        t_shots = shots[shots['team_id'] == team_id]
        for _, s in t_shots.iterrows():
            zone = s.get('shot_goal_zone')
            if not zone or zone not in GOAL_ZONE_COORDS:
                continue
            gz_x, gz_y = GOAL_ZONE_COORDS[zone]
            # Add small jitter to avoid overlap
            jx = np.random.uniform(-0.04, 0.04)
            jy = np.random.uniform(-0.04, 0.04)
            px, py = gz_x + jx, gz_y + jy

            xg = s['shot_xg'] or 0
            size = 80 + 400 * xg

            if s['shot_is_goal']:
                marker, ec, fc = '*', 'white', color
            elif s['shot_on_target']:
                marker, ec, fc = 'o', 'white', color
            else:
                marker, ec, fc = 'X', color, 'none'

            ax.scatter(px, py, s=size, c=fc, marker=marker,
                       edgecolors=ec, lw=0.8, alpha=0.85, zorder=5)

        team_name = meta['home_name'] if team_id == home_id else meta['away_name']
        t_shots_here = t_shots
        goals = int(t_shots_here['shot_is_goal'].sum())
        sot = int(t_shots_here['shot_on_target'].sum())
        xg = t_shots_here['shot_xg'].sum()

        ax.set_title(f"{team_name}", fontproperties=fp_bold, fontsize=15,
                     color=color, pad=8)
        ax.text(0.5, -0.2, f"Shots: {len(t_shots_here)}  |  SoT: {sot}  |  xG: {xg:.2f}  |  Goals: {goals}",
                ha='center', va='center', fontproperties=fp_reg, fontsize=15,
                color=TEXT_DIM, transform=ax.transAxes)

    fig.suptitle('Goal Frame', fontproperties=fp_bold, fontsize=17,
                 color=TEXT_MAIN, y=1.02)
    return fig


# ---------------------------------------------------------------------------
# 8. xG by Player
# ---------------------------------------------------------------------------

def render_xg_by_player(player_stats: pd.DataFrame, meta: dict,
                          team_colors: dict, figsize=(14, 7)) -> plt.Figure:
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax, team_id, color in [(axes[0], home_id, home_color),
                                 (axes[1], away_id, away_color)]:
        ax.set_facecolor(BG)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

        t_players = player_stats[
            (player_stats['team_id'] == team_id) &
            (player_stats['shots'] > 0)
        ].sort_values('xg', ascending=True).tail(8)

        if t_players.empty:
            ax.axis('off')
            continue

        names = t_players['player_name'].tolist()
        xg_vals = t_players['xg'].tolist()
        goals = t_players['goals'].tolist()

        bars = ax.barh(names, xg_vals, color=color, alpha=0.75, height=0.6)

        # Goal markers
        for i, (g, xg) in enumerate(zip(goals, xg_vals)):
            if g > 0:
                ax.scatter(xg + 0.01, i, s=60, c='white', marker='*',
                           zorder=5, edgecolors=color, lw=0.5)
                ax.text(xg + 0.03, i, f" {g}G", va='center',
                        fontproperties=fp_bold, fontsize=17, color='white')

        ax.set_xlabel('xG', fontproperties=fp_reg, fontsize=15, color=TEXT_DIM)
        ax.tick_params(colors=TEXT_DIM)
        ax.xaxis.label.set_color(TEXT_DIM)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(fp_reg)
            tick.set_color(TEXT_DIM)
            tick.set_fontsize(9)

        team_name = meta['home_name'] if team_id == home_id else meta['away_name']
        ax.set_title(team_name, fontproperties=fp_bold, fontsize=15, color=color, pad=8)

    fig.suptitle('xG by Player  (★ = Goal)', fontproperties=fp_bold, fontsize=16,
                 color=TEXT_MAIN, y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8b. SCA by Player
# ---------------------------------------------------------------------------

def render_sca_by_player(sca_df: pd.DataFrame, meta: dict,
                          team_colors: dict, figsize=(14, 7)) -> plt.Figure:
    """Horizontal bar chart: Shot-Creating Actions per player, both teams side by side."""
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax, team_id, color in [(axes[0], home_id, home_color),
                                 (axes[1], away_id, away_color)]:
        ax.set_facecolor(BG)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

        t_players = sca_df[sca_df['team_id'] == team_id].sort_values('sca', ascending=True).tail(10)

        if t_players.empty:
            ax.axis('off')
            continue

        names = t_players['player_name'].tolist()
        sca_vals = t_players['sca'].tolist()

        ax.barh(names, sca_vals, color=color, alpha=0.78, height=0.6)

        for i, v in enumerate(sca_vals):
            ax.text(v + 0.05, i, str(v), va='center',
                    fontproperties=fp_bold, fontsize=12, color='white')

        ax.set_xlabel('SCA', fontproperties=fp_reg, fontsize=13, color=TEXT_DIM)
        ax.tick_params(colors=TEXT_DIM)
        ax.xaxis.label.set_color(TEXT_DIM)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(fp_reg)
            tick.set_color(TEXT_DIM)
            tick.set_fontsize(10)

        team_name = meta['home_name'] if team_id == home_id else meta['away_name']
        ax.set_title(team_name, fontproperties=fp_bold, fontsize=15, color=color, pad=8)

    fig.suptitle('Shot-Creating Actions (SCA) by Player',
                 fontproperties=fp_bold, fontsize=16, color=TEXT_MAIN, y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# xT by Player (pass/carry split)
# ---------------------------------------------------------------------------

def render_xt_by_player(xt_df: pd.DataFrame, meta: dict,
                         team_colors: dict, figsize=(14, 7)) -> plt.Figure:
    """
    Horizontal bar chart: xT generated per player.
    Each bar is split into pass xT (solid) and carry xT (lighter shade).
    Shows top 10 players per team, both teams side by side.
    """
    fp_bold = _load_font('Bold')
    fp_reg  = _load_font('Regular')

    home_id    = meta['home_id'];  away_id   = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax, team_id, color in [(axes[0], home_id, home_color),
                                (axes[1], away_id, away_color)]:
        ax.set_facecolor(BG)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

        t_data = xt_df[xt_df['team_id'] == team_id].copy()
        # Only positive xT contributors
        t_data = t_data[t_data['xT'] > 0].sort_values('xT', ascending=True).tail(10)

        if t_data.empty:
            ax.axis('off')
            continue

        names      = t_data['player_name'].tolist()
        xt_pass    = t_data['xT_pass'].clip(lower=0).tolist()  if 'xT_pass'  in t_data else [0]*len(names)
        xt_carry   = t_data['xT_carry'].clip(lower=0).tolist() if 'xT_carry' in t_data else [0]*len(names)
        xt_total   = t_data['xT'].tolist()

        # Stacked bars: pass + carry
        import matplotlib.colors as mcolors
        carry_color = mcolors.to_rgba(color, alpha=0.45)
        ax.barh(names, xt_pass,  color=color,       alpha=0.85, height=0.6, label='Pass xT')
        ax.barh(names, xt_carry, left=xt_pass,       color=carry_color, height=0.6, label='Carry xT')

        for i, v in enumerate(xt_total):
            ax.text(v + 0.001, i, f"{v:.3f}", va='center',
                    fontproperties=fp_bold, fontsize=10, color='white')

        ax.set_xlabel('xT', fontproperties=fp_reg, fontsize=12, color=TEXT_DIM)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(fp_reg)
            tick.set_color(TEXT_DIM)
            tick.set_fontsize(9)

        team_name = meta['home_name'] if team_id == home_id else meta['away_name']
        ax.set_title(team_name, fontproperties=fp_bold, fontsize=14, color=color, pad=6)
        leg = ax.legend(fontsize=8, framealpha=0, loc='lower right')
        for txt in leg.get_texts():
            txt.set_color(TEXT_DIM)

    fig.suptitle('Expected Threat (xT) by Player  \u00b7  Pass + Carry',
                 fontproperties=fp_bold, fontsize=15, color=TEXT_MAIN, y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Offensive Actions (dual pitch — one per team)
# ---------------------------------------------------------------------------

def render_offensive_actions_dual(df: pd.DataFrame, meta: dict, team_colors: dict,
                                   figsize=(20, 10)) -> plt.Figure:
    """
    Two mplsoccer pitches side by side (home | away).
    All actions plotted at low opacity; progressive passes/carries and box-ending
    actions highlighted in a brighter, thicker style.
    """
    fp_bold = _load_font('Bold')
    fp_reg  = _load_font('Regular')

    home_id    = meta['home_id'];  away_id   = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    pitch = _pitch()
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=figsize)
    fig.patch.set_facecolor(BG)

    BOX_X_MIN = 83.0  # Wyscout penalty box (x_plot)

    for ax, team_id, color in [(axes[0], home_id, home_color),
                                (axes[1], away_id, away_color)]:
        ax.set_facecolor(PITCH_BG)
        tdf = df[df['team_id'] == team_id].copy()

        # ── Passes (all accurate) ────────────────────────────────────────
        passes = tdf[
            tdf['type_primary'].isin(['pass', 'free_kick', 'corner', 'throw_in', 'goal_kick']) &
            (tdf['pass_accurate'] == True) &
            tdf['x_plot'].notna() & tdf['y_plot'].notna() &
            tdf['pass_end_x'].notna() & tdf['pass_end_y'].notna()
        ].copy()

        for _, r in passes.iterrows():
            sx, sy = r['x_plot'], r['y_plot']
            ex, ey = r['pass_end_x'], r['pass_end_y']
            is_prog = bool(r.get('is_progressive_pass', False))
            is_kp   = bool(r.get('is_key_pass', False))
            ends_box = ex >= BOX_X_MIN

            if is_prog or is_kp or ends_box:
                lw    = 1.5;  alpha = 0.75;  c = color
            else:
                lw    = 0.6;  alpha = 0.20;  c = color
            ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle='->', color=c, lw=lw, alpha=alpha))
            if ends_box:
                ax.scatter(ex, ey, s=18, c=color, marker='o',
                           edgecolors='white', lw=0.5, alpha=0.85, zorder=5)

        # ── Carries ──────────────────────────────────────────────────────
        if 'is_carry' in tdf.columns:
            carries = tdf[
                (tdf['is_carry'] == True) &
                tdf['x_plot'].notna() & tdf['y_plot'].notna() &
                tdf['carry_end_x'].notna() & tdf['carry_end_y'].notna()
            ].copy()
            for _, r in carries.iterrows():
                sx, sy = r['x_plot'], r['y_plot']
                ex, ey = r['carry_end_x'], r['carry_end_y']
                is_prog = bool(r.get('is_progressive_carry', False))
                ends_box = ex >= BOX_X_MIN

                if is_prog or ends_box:
                    lw = 1.5;  alpha = 0.7;  ls = '-'
                else:
                    lw = 0.5;  alpha = 0.18; ls = '--'
                ax.plot([sx, ex], [sy, ey], color='#ffffff',
                        lw=lw, alpha=alpha, ls=ls, zorder=4)

        # ── Dead balls: corners and free kicks ───────────────────────────
        corners   = tdf[tdf['type_primary'] == 'corner']
        for _, r in corners.iterrows():
            if pd.notna(r.get('x_plot')) and pd.notna(r.get('y_plot')):
                ax.scatter(r['x_plot'], r['y_plot'], s=60, marker='*',
                           c='#FFD700', edgecolors='white', lw=0.4, zorder=7, alpha=0.9)

        free_kicks = tdf[tdf['type_primary'] == 'free_kick']
        for _, r in free_kicks.iterrows():
            if pd.notna(r.get('x_plot')) and pd.notna(r.get('y_plot')):
                ax.scatter(r['x_plot'], r['y_plot'], s=45, marker='D',
                           c='#FF6B6B', edgecolors='white', lw=0.4, zorder=7, alpha=0.9)

        team_name = meta['home_name'] if team_id == home_id else meta['away_name']
        ax.set_title(team_name, fontproperties=fp_bold, fontsize=14, color=color, pad=6)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=home_color, lw=1.5, alpha=0.8, label='Progressive / Key / Box'),
        Line2D([0], [0], color=home_color, lw=0.6, alpha=0.25, label='Regular pass'),
        Line2D([0], [0], color='white',    lw=1.5, alpha=0.7,  label='Progressive carry'),
        Line2D([0], [0], color='white',    lw=0.5, alpha=0.2,  ls='--', label='Regular carry'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700',
               markersize=10, linestyle='None', label='Corner'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#FF6B6B',
               markersize=8,  linestyle='None', label='Free Kick'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=9,
               facecolor=PANEL_BG, edgecolor=LINE_COLOR, labelcolor=TEXT_MAIN,
               prop=fp_reg, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle('Offensive Action Map  \u00b7  Brighter = Progressive / Key / Box-ending  '
                 '\u00b7  \u2605 = Corner  \u00b7  \u25c6 = Free Kick',
                 fontproperties=fp_bold, fontsize=14, color=TEXT_MAIN, y=1.01)
    return fig


# ---------------------------------------------------------------------------
# 9. Pass Network
# ---------------------------------------------------------------------------

def render_pass_network(df: pd.DataFrame, meta: dict, team_colors: dict,
                         player_lookup: dict = None,
                         min_passes: int = 3, period: str = '1H',
                         min_start: int = None, min_end: int = None,
                         home_min_start: int = None, home_min_end: int = None,
                         away_min_start: int = None, away_min_end: int = None,
                         label: str = '',
                         figsize=(18, 9)) -> plt.Figure:
    from wyscout_parser import compute_pass_network, compute_xt_grid, _xy_to_xt_cell

    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    pitch = _pitch()
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=figsize)
    fig.patch.set_facecolor(BG)

    for ax, team_id, color in [(axes[0], home_id, home_color),
                                 (axes[1], away_id, away_color)]:
        ax.set_facecolor(PITCH_BG)

        # Determine interval for this team
        if team_id == home_id:
            t_min_s = home_min_start if home_min_start is not None else min_start
            t_min_e = home_min_end if home_min_end is not None else min_end
        else:
            t_min_s = away_min_start if away_min_start is not None else min_start
            t_min_e = away_min_end if away_min_end is not None else min_end

        nodes, edges = compute_pass_network(df, team_id, period, min_passes, t_min_s, t_min_e)

        if nodes.empty:
            continue

        # xT KDE background — starting positions of progressive passes weighted by xT gain
        try:
            xt_grid = compute_xt_grid()
            t_filter = df[df['team_id'] == team_id].copy()
            if t_min_s is not None and t_min_e is not None:
                t_filter = t_filter[(t_filter['match_minute'] >= t_min_s) & (t_filter['match_minute'] < t_min_e)]
            prog = t_filter[
                t_filter['type_primary'].isin(['pass', 'free_kick', 'corner', 'throw_in', 'goal_kick'])
            ].copy()
            prog = prog[prog['x_plot'].notna() & prog['y_plot'].notna() &
                        prog['pass_end_x'].notna() & prog['pass_end_y'].notna()]
            if len(prog) > 5:
                xt_xs, xt_ys, xt_ws = [], [], []
                for _, r in prog.iterrows():
                    try:
                        rc, cc = _xy_to_xt_cell(r['x_plot'], r['y_plot'])
                        re_, ce_ = _xy_to_xt_cell(r['pass_end_x'], r['pass_end_y'])
                        gain = xt_grid[re_, ce_] - xt_grid[rc, cc]
                        if gain > 0:
                            xt_xs.append(r['x_plot'])
                            xt_ys.append(r['y_plot'])
                            xt_ws.append(gain)
                    except Exception:
                        pass
                if len(xt_xs) > 5:
                    pitch.kdeplot(np.array(xt_xs), np.array(xt_ys),
                                  ax=ax, fill=True, alpha=0.22,
                                  levels=6, cut=4,
                                  cmap='hot', zorder=1)
        except Exception:
            pass

        # Draw edges
        if not edges.empty:
            max_passes = edges['pass_count'].max()
            for _, e in edges.iterrows():
                lw = 0.5 + 3.5 * (e['pass_count'] / max_passes)
                ax.plot([e['from_x'], e['to_x']], [e['from_y'], e['to_y']],
                        color=color, lw=lw, alpha=0.5, zorder=3)

        # Draw nodes
        max_touches = nodes['touches'].max()
        for _, n in nodes.iterrows():
            size = 200 + 400 * (n['touches'] / max_touches)
            ax.scatter(n['avg_x'], n['avg_y'], s=size,
                       c=color, edgecolors='white', lw=1.5, zorder=5, alpha=0.9)
            # Use jersey number if available, else short name
            p_info = player_lookup.get(int(n['player_id'])) if (player_lookup and n['player_id']) else {}
            shirt = p_info.get('shirtNumber') if p_info else None
            display = str(shirt) if shirt else (str(n['player_name']).split()[-1] if n['player_name'] else '')
            ax.text(n['avg_x'], n['avg_y'] - 4, display,
                    ha='center', va='top', fontproperties=fp_reg, fontsize=16,
                    color='white', zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=PITCH_BG)])

        team_name = meta['home_name'] if team_id == home_id else meta['away_name']
        t_label = f"{t_min_s}'–{t_min_e}'" if (t_min_s is not None) else (label if label else period)
        ax.set_title(f"{team_name}  ({t_label})", fontproperties=fp_bold,
                     fontsize=17, color=color, pad=8)

    fig.suptitle('Pass Network  (node size = touches, line width = passes)',
                 fontproperties=fp_bold, fontsize=16, color=TEXT_MAIN, y=1.01)
    return fig


# ---------------------------------------------------------------------------
# 10. Touch Heatmap
# ---------------------------------------------------------------------------

def render_touch_heatmap(df: pd.DataFrame, meta: dict, team_colors: dict,
                          figsize=(18, 8)) -> plt.Figure:
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    pitch = _pitch()
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=figsize)
    fig.patch.set_facecolor(BG)

    active = df[df['type_primary'] != 'game_interruption']

    for ax, team_id, color in [(axes[0], home_id, home_color),
                                 (axes[1], away_id, away_color)]:
        ax.set_facecolor(PITCH_BG)
        t = active[active['team_id'] == team_id].copy()
        t = t[t['x_plot'].notna() & t['y_plot'].notna()]

        if len(t) < 5:
            continue

        # KDE heatmap using mplsoccer bin_statistic
        bin_stat = pitch.bin_statistic(
            t['x_plot'], t['y_plot'],
            statistic='count', bins=(24, 16)
        )
        bin_stat['statistic'] = gaussian_filter(
            bin_stat['statistic'].astype(float), sigma=1.5
        )

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            'heat', [PITCH_BG, color + '44', color + 'aa', color], N=256
        )
        pitch.heatmap(bin_stat, ax=ax, cmap=cmap, alpha=0.8)

        team_name = meta['home_name'] if team_id == home_id else meta['away_name']
        ax.set_title(f"{team_name}", fontproperties=fp_bold, fontsize=17,
                     color=color, pad=8)

    fig.suptitle('Touch Heatmap', fontproperties=fp_bold, fontsize=16,
                 color=TEXT_MAIN, y=1.01)
    return fig


# ---------------------------------------------------------------------------
# 11. Progressive Actions (passes + carries)
# ---------------------------------------------------------------------------

def render_progressive_actions(df: pd.DataFrame, meta: dict, team_colors: dict,
                                 figsize=(16, 10)) -> plt.Figure:
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    pitch = _pitch()
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PITCH_BG)

    # Progressive passes
    prog_passes = df[df['is_progressive_pass'] == True].copy()
    prog_passes = prog_passes[
        prog_passes['x_plot'].notna() &
        prog_passes['y_plot'].notna() &
        prog_passes['pass_end_x'].notna() &
        prog_passes['pass_end_y'].notna()
    ]

    # Progressive carries
    prog_carries = df[df['is_progressive_carry'] == True].copy()
    prog_carries = prog_carries[
        prog_carries['x_plot'].notna() &
        prog_carries['y_plot'].notna() &
        prog_carries['carry_end_x'].notna() &
        prog_carries['carry_end_y'].notna()
    ]

    for team_id, color in [(home_id, home_color), (away_id, away_color)]:
        t_passes = prog_passes[prog_passes['team_id'] == team_id]
        for _, p in t_passes.iterrows():
            ax.annotate('', xy=(p['pass_end_x'], p['pass_end_y']),
                        xytext=(p['x_plot'], p['y_plot']),
                        arrowprops=dict(
                            arrowstyle='->', color=color,
                            lw=1.2, alpha=0.55,
                            mutation_scale=10,
                        ), zorder=3)

        t_carries = prog_carries[prog_carries['team_id'] == team_id]
        for _, c in t_carries.iterrows():
            ax.annotate('', xy=(c['carry_end_x'], c['carry_end_y']),
                        xytext=(c['x_plot'], c['y_plot']),
                        arrowprops=dict(
                            arrowstyle='->', color=color,
                            lw=2, alpha=0.7, linestyle='dashed',
                            mutation_scale=10,
                        ), zorder=4)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=home_color, lw=1.5, label=f"{meta['home_name']} passes"),
        Line2D([0], [0], color=away_color, lw=1.5, label=f"{meta['away_name']} passes"),
        Line2D([0], [0], color=home_color, lw=2, ls='--', label='Progressive carry'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=17,
              facecolor=PANEL_BG, edgecolor=LINE_COLOR, labelcolor=TEXT_MAIN,
              prop=fp_reg, framealpha=0.9)

    # Counts
    hp = len(prog_passes[prog_passes['team_id'] == home_id])
    ap = len(prog_passes[prog_passes['team_id'] == away_id])
    hc = len(prog_carries[prog_carries['team_id'] == home_id])
    ac = len(prog_carries[prog_carries['team_id'] == away_id])
    ax.text(10, 105, f"{meta['home_name']}: {hp} prog passes, {hc} prog carries",
            ha='left', fontproperties=fp_reg, fontsize=15, color=home_color)
    ax.text(90, 105, f"{meta['away_name']}: {ap} prog passes, {ac} prog carries",
            ha='right', fontproperties=fp_reg, fontsize=15, color=away_color)

    ax.set_title('Progressive Actions', fontproperties=fp_bold, fontsize=16,
                 color=TEXT_MAIN, pad=8)
    return fig


# ---------------------------------------------------------------------------
# Offensive Actions Pitch Map
# ---------------------------------------------------------------------------

def render_offensive_actions(df: pd.DataFrame, meta: dict, team_colors: dict,
                              action_types: list = None,
                              team_filter: str = 'both',
                              pitch_zone: str = 'full',
                              min_minute: int = 0, max_minute: int = 90,
                              figsize=(18, 10)) -> plt.Figure:
    """
    Configurable offensive actions pitch map.
    action_types: list of strings from ['passes', 'carries', 'dribbles',
                  'progressive_passes', 'progressive_carries', 'key_passes', 'crosses']
    team_filter: 'home', 'away', 'both'
    pitch_zone: 'full', 'final_third', 'penalty_box', 'own_half', 'middle_third'
    min_minute / max_minute: time range filter
    """
    if action_types is None:
        action_types = ['passes']

    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    pitch = _pitch()
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PITCH_BG)

    # Time filter
    fdf = df[(df['match_minute'] >= min_minute) & (df['match_minute'] <= max_minute)].copy()

    # Team filter
    if team_filter == 'home':
        fdf = fdf[fdf['team_id'] == home_id]
    elif team_filter == 'away':
        fdf = fdf[fdf['team_id'] == away_id]

    # Pitch zone x-limits (Wyscout x in plot coords = x_plot)
    zone_xlim = {
        'full': (0, 100),
        'own_half': (0, 50),
        'middle_third': (33, 67),
        'final_third': (66, 100),
        'penalty_box': (83, 100),
    }
    x_min, x_max = zone_xlim.get(pitch_zone, (0, 100))

    # Set axis zoom for zone
    if pitch_zone != 'full':
        ax.set_xlim(x_min - 2, x_max + 2)

    drawn = 0

    for act_type in action_types:
        if act_type == 'passes':
            subset = fdf[
                fdf['type_primary'].isin(['pass', 'free_kick', 'throw_in', 'goal_kick']) &
                ~fdf['is_key_pass'] & ~fdf['is_progressive_pass'] & ~fdf['is_cross']
            ].copy()
            for _, r in subset.iterrows():
                if pd.isna(r['x_plot']) or pd.isna(r['y_plot']): continue
                if not (x_min <= r['x_plot'] <= x_max): continue
                color = home_color if r['team_id'] == home_id else away_color
                alpha = 0.35 if r.get('pass_accurate') else 0.18
                if pd.notna(r.get('pass_end_x')) and pd.notna(r.get('pass_end_y')):
                    ax.annotate('', xy=(r['pass_end_x'], r['pass_end_y']),
                                xytext=(r['x_plot'], r['y_plot']),
                                arrowprops=dict(arrowstyle='->', color=color,
                                                lw=0.7, alpha=alpha))
                else:
                    ax.scatter(r['x_plot'], r['y_plot'], s=10, c=color, alpha=alpha, zorder=3)
                drawn += 1

        elif act_type == 'key_passes':
            subset = fdf[fdf['is_key_pass'] == True].copy()
            for _, r in subset.iterrows():
                if pd.isna(r['x_plot']) or pd.isna(r['y_plot']): continue
                if not (x_min <= r['x_plot'] <= x_max): continue
                color = home_color if r['team_id'] == home_id else away_color
                if pd.notna(r.get('pass_end_x')) and pd.notna(r.get('pass_end_y')):
                    ax.annotate('', xy=(r['pass_end_x'], r['pass_end_y']),
                                xytext=(r['x_plot'], r['y_plot']),
                                arrowprops=dict(arrowstyle='->', color=color,
                                                lw=1.5, alpha=0.8,
                                                connectionstyle='arc3,rad=0.1'))
                ax.scatter(r['x_plot'], r['y_plot'], s=40, c=color, marker='D',
                           edgecolors='white', lw=0.5, alpha=0.9, zorder=5)
                drawn += 1

        elif act_type == 'progressive_passes':
            subset = fdf[fdf['is_progressive_pass'] == True].copy()
            for _, r in subset.iterrows():
                if pd.isna(r['x_plot']) or pd.isna(r['y_plot']): continue
                if not (x_min <= r['x_plot'] <= x_max): continue
                color = home_color if r['team_id'] == home_id else away_color
                if pd.notna(r.get('pass_end_x')) and pd.notna(r.get('pass_end_y')):
                    ax.annotate('', xy=(r['pass_end_x'], r['pass_end_y']),
                                xytext=(r['x_plot'], r['y_plot']),
                                arrowprops=dict(arrowstyle='->', color=color,
                                                lw=1.2, alpha=0.75))
                drawn += 1

        elif act_type == 'crosses':
            subset = fdf[fdf['is_cross'] == True].copy()
            for _, r in subset.iterrows():
                if pd.isna(r['x_plot']) or pd.isna(r['y_plot']): continue
                if not (x_min <= r['x_plot'] <= x_max): continue
                color = home_color if r['team_id'] == home_id else away_color
                if pd.notna(r.get('pass_end_x')) and pd.notna(r.get('pass_end_y')):
                    ax.annotate('', xy=(r['pass_end_x'], r['pass_end_y']),
                                xytext=(r['x_plot'], r['y_plot']),
                                arrowprops=dict(arrowstyle='->', color=color,
                                                lw=1.2, alpha=0.7,
                                                connectionstyle='arc3,rad=0.15'))
                ax.scatter(r['x_plot'], r['y_plot'], s=30, c=color, marker='^',
                           alpha=0.8, zorder=4)
                drawn += 1

        elif act_type == 'carries':
            subset = fdf[fdf['is_carry'] == True].copy()
            for _, r in subset.iterrows():
                if pd.isna(r['x_plot']) or pd.isna(r['y_plot']): continue
                if not (x_min <= r['x_plot'] <= x_max): continue
                color = home_color if r['team_id'] == home_id else away_color
                if pd.notna(r.get('carry_end_x')) and pd.notna(r.get('carry_end_y')):
                    ax.plot([r['x_plot'], r['carry_end_x']],
                            [r['y_plot'], r['carry_end_y']],
                            color=color, lw=1.0, alpha=0.45, ls='dotted', zorder=3)
                ax.scatter(r['x_plot'], r['y_plot'], s=15, c=color, alpha=0.5, zorder=3)
                drawn += 1

        elif act_type == 'progressive_carries':
            subset = fdf[fdf['is_progressive_carry'] == True].copy()
            for _, r in subset.iterrows():
                if pd.isna(r['x_plot']) or pd.isna(r['y_plot']): continue
                if not (x_min <= r['x_plot'] <= x_max): continue
                color = home_color if r['team_id'] == home_id else away_color
                if pd.notna(r.get('carry_end_x')) and pd.notna(r.get('carry_end_y')):
                    ax.annotate('', xy=(r['carry_end_x'], r['carry_end_y']),
                                xytext=(r['x_plot'], r['y_plot']),
                                arrowprops=dict(arrowstyle='->', color=color,
                                                lw=1.3, alpha=0.8, linestyle='dashed'))
                drawn += 1

        elif act_type == 'dribbles':
            subset = fdf[
                fdf['type_primary'] == 'duel'
            ].copy()
            subset = subset[subset['duel_subtype'] == 'offensive_duel']
            for _, r in subset.iterrows():
                if pd.isna(r['x_plot']) or pd.isna(r['y_plot']): continue
                if not (x_min <= r['x_plot'] <= x_max): continue
                color = home_color if r['team_id'] == home_id else away_color
                alpha = 0.85 if r.get('duel_won') else 0.4
                ax.scatter(r['x_plot'], r['y_plot'], s=35, c=color,
                           marker='o', edgecolors='white', lw=0.5,
                           alpha=alpha, zorder=4)
                drawn += 1

        elif act_type == 'corners':
            subset = fdf[fdf['type_primary'] == 'corner'].copy()
            for _, r in subset.iterrows():
                if pd.isna(r['x_plot']) or pd.isna(r['y_plot']): continue
                if not (x_min <= r['x_plot'] <= x_max): continue
                color = home_color if r['team_id'] == home_id else away_color
                if pd.notna(r.get('pass_end_x')) and pd.notna(r.get('pass_end_y')):
                    ax.annotate('', xy=(r['pass_end_x'], r['pass_end_y']),
                                xytext=(r['x_plot'], r['y_plot']),
                                arrowprops=dict(arrowstyle='->', color='#FFD700',
                                                lw=1.2, alpha=0.8,
                                                connectionstyle='arc3,rad=0.15'))
                ax.scatter(r['x_plot'], r['y_plot'], s=55, c='#FFD700', marker='*',
                           edgecolors='white', lw=0.5, alpha=0.9, zorder=5)
                drawn += 1

        elif act_type == 'free_kicks':
            subset = fdf[fdf['type_primary'] == 'free_kick'].copy()
            for _, r in subset.iterrows():
                if pd.isna(r['x_plot']) or pd.isna(r['y_plot']): continue
                if not (x_min <= r['x_plot'] <= x_max): continue
                color = home_color if r['team_id'] == home_id else away_color
                if pd.notna(r.get('pass_end_x')) and pd.notna(r.get('pass_end_y')):
                    ax.annotate('', xy=(r['pass_end_x'], r['pass_end_y']),
                                xytext=(r['x_plot'], r['y_plot']),
                                arrowprops=dict(arrowstyle='->', color='#FF6B6B',
                                                lw=1.2, alpha=0.8))
                ax.scatter(r['x_plot'], r['y_plot'], s=45, c='#FF6B6B', marker='D',
                           edgecolors='white', lw=0.5, alpha=0.9, zorder=5)
                drawn += 1

        elif act_type == 'throw_ins':
            subset = fdf[fdf['type_primary'] == 'throw_in'].copy()
            for _, r in subset.iterrows():
                if pd.isna(r['x_plot']) or pd.isna(r['y_plot']): continue
                if not (x_min <= r['x_plot'] <= x_max): continue
                color = home_color if r['team_id'] == home_id else away_color
                if pd.notna(r.get('pass_end_x')) and pd.notna(r.get('pass_end_y')):
                    ax.annotate('', xy=(r['pass_end_x'], r['pass_end_y']),
                                xytext=(r['x_plot'], r['y_plot']),
                                arrowprops=dict(arrowstyle='->', color=color,
                                                lw=0.9, alpha=0.55, linestyle='dotted'))
                ax.scatter(r['x_plot'], r['y_plot'], s=28, c=color, marker='s',
                           edgecolors='white', lw=0.5, alpha=0.7, zorder=4)
                drawn += 1

    # Zone boundary indicator
    if pitch_zone != 'full':
        from matplotlib.patches import Rectangle as _Rect
        zone_rect = _Rect((x_min, 0), x_max - x_min, 100,
                          linewidth=1.5, edgecolor=TEXT_DIM,
                          facecolor='none', linestyle='--', alpha=0.5, zorder=10)
        ax.add_patch(zone_rect)

    zone_labels = {'full': 'Full Pitch', 'own_half': 'Own Half',
                   'middle_third': 'Middle Third', 'final_third': 'Final Third',
                   'penalty_box': 'Penalty Box'}
    time_label = f"{min_minute}'–{max_minute}'"
    act_label = ', '.join(a.replace('_', ' ').title() for a in action_types)
    team_label = {'home': meta['home_name'], 'away': meta['away_name'], 'both': 'Both Teams'}[team_filter]

    # Attacking direction indicator — home attacks right, away attacks left
    if team_filter == 'away':
        ax.text(98, -4, 'Attacking <——', ha='right', va='top',
                fontsize=8, color=away_color, fontproperties=fp_reg, style='italic')
    elif team_filter == 'home':
        ax.text(2, -4, '——> Attacking', ha='left', va='top',
                fontsize=8, color=home_color, fontproperties=fp_reg, style='italic')
    else:
        ax.text(2, -4, f'——> {meta["home_name"]}', ha='left', va='top',
                fontsize=8, color=home_color, fontproperties=fp_reg, style='italic')
        ax.text(98, -4, f'{meta["away_name"]} <——', ha='right', va='top',
                fontsize=8, color=away_color, fontproperties=fp_reg, style='italic')

    ax.set_title(
        f'{act_label}  |  {zone_labels.get(pitch_zone,"Full Pitch")}  |  {team_label}  |  {time_label}',
        fontproperties=fp_bold, fontsize=12, color=TEXT_MAIN, pad=10
    )

    # Legend
    legend_elements = []
    if team_filter in ('home', 'both'):
        legend_elements.append(mpatches.Patch(color=home_color, label=meta['home_name']))
    if team_filter in ('away', 'both'):
        legend_elements.append(mpatches.Patch(color=away_color, label=meta['away_name']))
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower left', fontsize=10,
                  facecolor=PANEL_BG, edgecolor=LINE_COLOR, labelcolor=TEXT_MAIN,
                  prop=fp_reg, framealpha=0.8)

    return fig


# ---------------------------------------------------------------------------
# 12. Defensive Actions Map
# ---------------------------------------------------------------------------

def render_defensive_actions(df: pd.DataFrame, meta: dict, team_colors: dict,
                               figsize=(20, 10)) -> plt.Figure:
    """
    Zone % heatmap on Juego de Posición positional pitch (6×5 grid).
    Each zone shows % of that team's total defensive actions.
    Colour intensity ∝ percentage.
    """
    fp_bold = _load_font('Bold')
    fp_reg  = _load_font('Regular')

    home_id    = meta['home_id'];   away_id    = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    DEF_TYPES = ['interception', 'clearance', 'duel']

    pitch = Pitch(
        pitch_type='wyscout',
        positional=True, shade_middle=True,
        positional_color='#eadddd', shade_color='#f2f2f2',
        line_color='#888888', line_zorder=2,
    )
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=figsize)
    fig.patch.set_facecolor(BG)

    N_COLS, N_ROWS = 6, 5
    x_edges = np.linspace(0, 100, N_COLS + 1)
    y_edges = np.linspace(0, 100, N_ROWS + 1)

    for ax, team_id, color, team_name in [
        (axes[0], home_id, home_color, meta['home_name']),
        (axes[1], away_id, away_color, meta['away_name']),
    ]:
        tdf = df[
            (df['team_id'] == team_id) &
            df['type_primary'].isin(DEF_TYPES) &
            df['x_plot'].notna() & df['y_plot'].notna()
        ].copy()

        total = max(len(tdf), 1)

        zone_counts = np.zeros((N_COLS, N_ROWS))
        for ci in range(N_COLS):
            for ri in range(N_ROWS):
                x0, x1 = x_edges[ci], x_edges[ci + 1]
                y0, y1 = y_edges[ri], y_edges[ri + 1]
                zone_counts[ci, ri] = int(
                    ((tdf['x_plot'] >= x0) & (tdf['x_plot'] < x1) &
                     (tdf['y_plot'] >= y0) & (tdf['y_plot'] < y1)).sum()
                )
        max_count = max(zone_counts.max(), 1)

        for ci in range(N_COLS):
            for ri in range(N_ROWS):
                x0, x1 = x_edges[ci], x_edges[ci + 1]
                y0, y1 = y_edges[ri], y_edges[ri + 1]
                cnt = zone_counts[ci, ri]
                pct = cnt / total * 100
                alpha = 0.15 + 0.70 * (cnt / max_count)

                rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  linewidth=0.6, edgecolor='white',
                                  facecolor=color, alpha=alpha,
                                  linestyle='-', zorder=3)
                ax.add_patch(rect)

                if pct >= 0.5:
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    ax.text(cx, cy, f"{pct:.0f}%", ha='center', va='center',
                            fontsize=10, fontweight='bold', color='white', zorder=5,
                            path_effects=[pe.withStroke(linewidth=2.0, foreground='#111111')])

        int_n  = len(tdf[tdf['type_primary'] == 'interception'])
        clr_n  = len(tdf[tdf['type_primary'] == 'clearance'])
        duel_n = len(tdf[tdf['type_primary'] == 'duel'])
        ax.set_title(
            f"{team_name}  ·  INT {int_n}  |  CLR {clr_n}  |  Duels {duel_n}",
            fontproperties=fp_bold, fontsize=11, color=color, pad=6)
        ax.text(2, -4, '——> Attacking', ha='left', va='top',
                fontsize=8, color='#555555', fontproperties=fp_reg, style='italic')

    return fig


# ---------------------------------------------------------------------------
# 13. PPDA Pressing Map
# ---------------------------------------------------------------------------

def render_pressing_map(df: pd.DataFrame, meta: dict, team_colors: dict,
                         ppda: dict, figsize=(18, 8)) -> plt.Figure:
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    pitch = _pitch()
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=figsize)
    fig.patch.set_facecolor(BG)

    for ax, team_id, color in [(axes[0], home_id, home_color),
                                 (axes[1], away_id, away_color)]:
        ax.set_facecolor(PITCH_BG)
        t = df[df['team_id'] == team_id]

        # Pressing actions: duels + interceptions in opponent's half (x_plot > 50)
        pressing = t[
            (t['type_primary'].isin(['duel', 'interception'])) &
            (t['x_plot'] > 50) &
            t['x_plot'].notna()
        ]

        if len(pressing) > 5:
            bin_stat = pitch.bin_statistic(
                pressing['x_plot'], pressing['y_plot'],
                statistic='count', bins=(16, 12)
            )
            bin_stat['statistic'] = gaussian_filter(
                bin_stat['statistic'].astype(float), sigma=1.5
            )
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list(
                'press', [PITCH_BG, color + '33', color + '99', color], N=256
            )
            pitch.heatmap(bin_stat, ax=ax, cmap=cmap, alpha=0.75)

        ppda_val = ppda.get(team_id, 0)
        team_name = meta['home_name'] if team_id == home_id else meta['away_name']
        ax.set_title(f"{team_name}  |  PPDA: {ppda_val:.2f}",
                     fontproperties=fp_bold, fontsize=17, color=color, pad=8)

    ax_shared_note = "PPDA = opponent passes allowed per pressing action in opp. defensive 40%"
    fig.text(0.5, 0.01, ax_shared_note, ha='center', fontproperties=fp_reg,
             fontsize=17, color=TEXT_DIM, style='italic')
    fig.suptitle('High Press Map (PPDA)',
                 fontproperties=fp_bold, fontsize=16, color=TEXT_MAIN, y=1.02)
    return fig


# ---------------------------------------------------------------------------
# Duel Zone Map
# ---------------------------------------------------------------------------

def render_duel_map(df: pd.DataFrame, meta: dict, team_colors: dict,
                    figsize=(18, 11)) -> plt.Figure:
    """
    Single pitch divided into 6×5 zones.
    Each zone shows home_won/away_won duel counts.
    Zone coloured by dominant team; brightness ∝ dominance margin.
    Attacking direction labels added at bottom.
    """
    import matplotlib.colors as mcolors
    fp_bold = _load_font('Bold')
    fp_reg  = _load_font('Regular')

    home_id    = meta['home_id'];   away_id    = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')
    home_name  = meta['home_name']; away_name  = meta['away_name']

    pitch = _pitch()
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PITCH_BG)

    N_COLS, N_ROWS = 6, 5
    x_edges = np.linspace(0, 100, N_COLS + 1)
    y_edges = np.linspace(0, 100, N_ROWS + 1)

    duels = df[df['type_primary'] == 'duel'].copy()
    duels = duels[duels['x_plot'].notna() & duels['y_plot'].notna()]
    h_won = duels[(duels['team_id'] == home_id) & (duels['duel_won'] == True)]
    a_won = duels[(duels['team_id'] == away_id) & (duels['duel_won'] == True)]

    for ci in range(N_COLS):
        for ri in range(N_ROWS):
            x0, x1 = x_edges[ci], x_edges[ci + 1]
            y0, y1 = y_edges[ri], y_edges[ri + 1]
            hw = int(((h_won['x_plot'] >= x0) & (h_won['x_plot'] < x1) &
                      (h_won['y_plot'] >= y0) & (h_won['y_plot'] < y1)).sum())
            aw = int(((a_won['x_plot'] >= x0) & (a_won['x_plot'] < x1) &
                      (a_won['y_plot'] >= y0) & (a_won['y_plot'] < y1)).sum())
            total = hw + aw
            if total == 0:
                face, alpha = PITCH_BG, 0.0
            elif hw > aw:
                alpha = 0.30 + 0.55 * (hw / total)
                face  = home_color
            elif aw > hw:
                alpha = 0.30 + 0.55 * (aw / total)
                face  = away_color
            else:
                face, alpha = '#888888', 0.30

            rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                              linewidth=0.8, edgecolor='white',
                              facecolor=face, alpha=alpha,
                              linestyle='--', zorder=2)
            ax.add_patch(rect)

            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(cx, cy, f"{hw}/{aw}", ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white', zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='#000000')])

    # Attacking direction labels
    ax.annotate('', xy=(92, -5), xytext=(8, -5),
                arrowprops=dict(arrowstyle='->', color=home_color, lw=1.8))
    ax.text(2,  -8, f'Attacking Direction ——>',
            ha='left', va='top', fontsize=9, color=home_color,
            fontproperties=fp_reg, style='italic')
    ax.text(98, -8, f'<—— Attacking Direction',
            ha='right', va='top', fontsize=9, color=away_color,
            fontproperties=fp_reg, style='italic')

    # Legend patches
    home_patch = mpatches.Patch(color=home_color, label=f'{home_name} (home/away wins)')
    away_patch = mpatches.Patch(color=away_color, label=away_name)
    ax.legend(handles=[home_patch, away_patch], loc='upper center',
              ncol=2, fontsize=10, facecolor=PANEL_BG, edgecolor=LINE_COLOR,
              labelcolor=TEXT_MAIN, prop=fp_reg, bbox_to_anchor=(0.5, 1.08))

    ax.set_title(f'Duel Zones  ·  {home_name} / {away_name}  ·  format: home won / away won',
                 fontproperties=fp_bold, fontsize=12, color=TEXT_MAIN, pad=18)
    return fig


# ---------------------------------------------------------------------------
# Transitions Map (Offensive & Defensive)
# ---------------------------------------------------------------------------

def _flag_led_to_attack(df: pd.DataFrame, team_id: int,
                         events: pd.DataFrame, window_secs: int = 10) -> list:
    """
    For each event row in `events`, check whether the same team made a pass
    into the opponent final third (x_plot > 66) within `window_secs` seconds.
    Returns a boolean list aligned to events.iterrows().
    """
    df_team = df[df['team_id'] == team_id].copy()
    df_team['abs_sec'] = df_team['match_minute'] * 60 + df_team['second'].fillna(0)
    df_team = df_team.sort_values('abs_sec')

    results = []
    for _, ev in events.iterrows():
        t0 = ev['match_minute'] * 60 + (ev.get('second') or 0)
        window = df_team[
            (df_team['abs_sec'] > t0) &
            (df_team['abs_sec'] <= t0 + window_secs) &
            df_team['type_primary'].isin(['pass', 'free_kick', 'corner']) &
            (df_team['x_plot'] > 66)
        ]
        results.append(len(window) > 0)
    return results


def render_transitions_map(df: pd.DataFrame, meta: dict, team_colors: dict,
                            mode: str = 'both',
                            figsize=(20, 10)) -> plt.Figure:
    """
    Two pitches (home | away).
    mode: 'offensive' — show ball recoveries (interceptions) only, starred if led to attack
          'defensive' — show turnovers (inaccurate passes + duel losses in att. half) only
          'both'      — show both layers
    """
    fp_bold = _load_font('Bold')
    fp_reg  = _load_font('Regular')

    home_id    = meta['home_id'];  away_id   = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    pitch = _pitch()
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=figsize)
    fig.patch.set_facecolor(BG)

    for ax, team_id, color, team_name in [
        (axes[0], home_id, home_color, meta['home_name']),
        (axes[1], away_id, away_color, meta['away_name']),
    ]:
        ax.set_facecolor(PITCH_BG)
        tdf = df[df['team_id'] == team_id].copy()

        recoveries = tdf[
            (tdf['type_primary'] == 'interception') &
            tdf['x_plot'].notna() & tdf['y_plot'].notna()
        ].copy()

        turnovers = tdf[
            (
                (tdf['type_primary'].isin(['pass', 'free_kick']) &
                 (tdf['pass_accurate'] == False)) |
                ((tdf['type_primary'] == 'duel') & (tdf['duel_won'] != True))
            ) &
            (tdf['x_plot'] > 50) &
            tdf['x_plot'].notna() & tdf['y_plot'].notna()
        ].copy()

        if mode in ('offensive', 'both') and not recoveries.empty:
            led = _flag_led_to_attack(df, team_id, recoveries)
            rec_atk  = recoveries[[l for l in led]]
            rec_no   = recoveries[[not l for l in led]]
            # Star = led to attack, circle = did not
            if any(led):
                ax.scatter(recoveries.loc[recoveries.index[led], 'x_plot'],
                           recoveries.loc[recoveries.index[led], 'y_plot'],
                           s=180, c='gold', marker='*', edgecolors=color,
                           lw=1.2, alpha=0.95, zorder=6,
                           label='Recovery → Attack')
            no_led_idx = [not l for l in led]
            if any(no_led_idx):
                ax.scatter(recoveries.loc[recoveries.index[no_led_idx], 'x_plot'],
                           recoveries.loc[recoveries.index[no_led_idx], 'y_plot'],
                           s=130, c=color, marker='o', edgecolors='white',
                           lw=1.2, alpha=0.85, zorder=5,
                           label='Recovery (no att.)')

        if mode in ('defensive', 'both') and not turnovers.empty:
            ax.scatter(turnovers['x_plot'], turnovers['y_plot'], s=90,
                       c='#F44336', marker='X', edgecolors='none',
                       alpha=0.65, zorder=3, label='Turnover (att. half)')

        n_rec  = len(recoveries)
        n_turn = len(turnovers)
        parts = []
        if mode in ('offensive', 'both'):
            led_n = sum(_flag_led_to_attack(df, team_id, recoveries)) if not recoveries.empty else 0
            parts.append(f"{n_rec} recoveries ({led_n} → attack)")
        if mode in ('defensive', 'both'):
            parts.append(f"{n_turn} turnovers")
        ax.set_title(f"{team_name}  ·  " + "  ·  ".join(parts),
                     fontproperties=fp_bold, fontsize=11, color=color, pad=6)
        ax.text(2, -4, '——> Attacking', ha='left', va='top',
                fontsize=8, color=color, fontproperties=fp_reg, style='italic')

    legend_elements = []
    if mode in ('offensive', 'both'):
        legend_elements += [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                   markersize=11, linestyle='None', label='Recovery → led to attack'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=home_color,
                   markersize=9,  linestyle='None', label='Recovery (no attack)'),
        ]
    if mode in ('defensive', 'both'):
        legend_elements.append(
            Line2D([0], [0], marker='X', color='w', markerfacecolor='#F44336',
                   markersize=9, linestyle='None', label='Turnover (attacking half)'))
    if legend_elements:
        fig.legend(handles=legend_elements, loc='lower center',
                   ncol=len(legend_elements), fontsize=10,
                   facecolor=PANEL_BG, edgecolor=LINE_COLOR, labelcolor=TEXT_MAIN,
                   prop=fp_reg, bbox_to_anchor=(0.5, -0.04))
    return fig


def compute_transitions_table(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Build a per-event transitions dataframe for display in the app.
    Returns rows with: Minute, Team, Player, Type, Zone, Led to Attack.
    """
    home_id = meta['home_id']; away_id = meta['away_id']
    rows = []
    for team_id, team_name in [(home_id, meta['home_name']), (away_id, meta['away_name'])]:
        tdf = df[df['team_id'] == team_id].copy()

        recoveries = tdf[
            (tdf['type_primary'] == 'interception') &
            tdf['x_plot'].notna() & tdf['y_plot'].notna()
        ].copy()
        if not recoveries.empty:
            led = _flag_led_to_attack(df, team_id, recoveries)
            for (_, ev), l in zip(recoveries.iterrows(), led):
                xp = ev['x_plot']
                zone = ('Own Third' if xp < 33 else
                        'Middle Third' if xp < 66 else 'Attacking Third')
                rows.append({
                    'Min': f"{int(ev['match_minute'])}'",
                    'Team': team_name,
                    'Player': ev.get('player_name', ''),
                    'Type': 'Recovery',
                    'Zone': zone,
                    'Led to Attack': '⭐ Yes' if l else 'No',
                })

        turnovers = tdf[
            (
                (tdf['type_primary'].isin(['pass', 'free_kick']) &
                 (tdf['pass_accurate'] == False)) |
                ((tdf['type_primary'] == 'duel') & (tdf['duel_won'] != True))
            ) &
            (tdf['x_plot'] > 50) &
            tdf['x_plot'].notna() & tdf['y_plot'].notna()
        ].copy()
        for _, ev in turnovers.iterrows():
            xp = ev['x_plot']
            zone = 'Middle Third' if xp < 66 else 'Attacking Third'
            rows.append({
                'Min': f"{int(ev['match_minute'])}'",
                'Team': team_name,
                'Player': ev.get('player_name', ''),
                'Type': 'Turnover',
                'Zone': zone,
                'Led to Attack': '—',
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values('Min')
    return result


# ---------------------------------------------------------------------------
# 14. Corner Kick Map
# ---------------------------------------------------------------------------

def render_corner_map(df: pd.DataFrame, meta: dict, team_colors: dict,
                       figsize=(16, 10)) -> plt.Figure:
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    pitch = _pitch(half=True)
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PITCH_BG)

    corners = df[df['type_primary'] == 'corner'].copy()

    for team_id, color in [(home_id, home_color), (away_id, away_color)]:
        t_corners = corners[corners['team_id'] == team_id]
        for _, c in t_corners.iterrows():
            if pd.isna(c['x_plot']) or pd.isna(c['y_plot']):
                continue
            if pd.isna(c.get('pass_end_x')) or pd.isna(c.get('pass_end_y')):
                # Just plot origin
                ax.scatter(c['x_plot'], c['y_plot'], s=60, c=color,
                           marker='s', edgecolors='white', lw=0.8, alpha=0.8, zorder=4)
                continue
            # Arrow from corner to delivery
            is_goal = c['possession_with_goal']
            alpha = 0.9 if is_goal else 0.55
            lw = 2.5 if is_goal else 1.2
            ax.annotate('', xy=(c['pass_end_x'], c['pass_end_y']),
                        xytext=(c['x_plot'], c['y_plot']),
                        arrowprops=dict(
                            arrowstyle='->', color=color, lw=lw,
                            alpha=alpha, mutation_scale=12,
                        ), zorder=4)
            ax.scatter(c['x_plot'], c['y_plot'], s=40, c=color,
                       marker='s', edgecolors='white', lw=0.8, alpha=0.8, zorder=5)

    # Stats
    for team_id, color, xpos in [(home_id, home_color, 60), (away_id, away_color, 90)]:
        t_c = corners[corners['team_id'] == team_id]
        sp = get_set_pieces_summary(df, meta)
        ck_xg = round(t_c['possession_xg'].sum() if 'possession_xg' in t_c.columns else 0, 3)
        ax.text(xpos, 102, f"{meta['home_name'] if team_id==home_id else meta['away_name']}: {len(t_c)} corners | xG: {ck_xg:.2f}",
                ha='center', fontproperties=fp_reg, fontsize=15, color=color)

    ax.set_title('Corner Kicks  (thick = goal, ■ = origin)',
                 fontproperties=fp_bold, fontsize=16, color=TEXT_MAIN, pad=8)

    legend_elements = [
        mpatches.Patch(color=home_color, label=meta['home_name']),
        mpatches.Patch(color=away_color, label=meta['away_name']),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=17,
              facecolor=PANEL_BG, edgecolor=LINE_COLOR, labelcolor=TEXT_MAIN,
              prop=fp_reg, framealpha=0.9)

    return fig


def get_set_pieces_summary(df: pd.DataFrame, meta: dict) -> dict:
    """Return basic set pieces stats summary."""
    sp_shots = df[(df['type_primary'] == 'shot') & (df['is_set_piece'] == True)]
    op_shots = df[(df['type_primary'] == 'shot') & (df['is_set_piece'] == False)]
    return {
        'sp_shots': len(sp_shots),
        'op_shots': len(op_shots),
        'sp_xg': round(sp_shots['shot_xg'].sum(), 3),
        'op_xg': round(op_shots['shot_xg'].sum(), 3),
    }


# ---------------------------------------------------------------------------
# 15. Free Kick Map
# ---------------------------------------------------------------------------

def render_free_kick_map(df: pd.DataFrame, meta: dict, team_colors: dict,
                          figsize=(16, 9)) -> plt.Figure:
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    pitch = _pitch()
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PITCH_BG)

    fks = df[df['type_primary'] == 'free_kick'].copy()

    for team_id, color in [(home_id, home_color), (away_id, away_color)]:
        t_fks = fks[fks['team_id'] == team_id]
        for _, fk in t_fks.iterrows():
            if pd.isna(fk['x_plot']) or pd.isna(fk['y_plot']):
                continue

            is_goal_possession = fk.get('possession_with_goal', False)
            size = 100 if is_goal_possession else 50
            marker = '*' if is_goal_possession else 'D'
            ax.scatter(fk['x_plot'], fk['y_plot'], s=size, c=color,
                       marker=marker, edgecolors='white', lw=0.6,
                       alpha=0.85, zorder=4)

            # Draw delivery if pass
            if pd.notna(fk.get('pass_end_x')) and pd.notna(fk.get('pass_end_y')):
                ax.annotate('', xy=(fk['pass_end_x'], fk['pass_end_y']),
                            xytext=(fk['x_plot'], fk['y_plot']),
                            arrowprops=dict(
                                arrowstyle='->', color=color, lw=1,
                                alpha=0.4, mutation_scale=8,
                            ), zorder=3)

    # Stats
    for team_id, color, ypos in [(home_id, home_color, 105), (away_id, away_color, 102)]:
        t_fks = fks[fks['team_id'] == team_id]
        fname = meta['home_name'] if team_id == home_id else meta['away_name']
        ax.text(50, ypos, f"{fname}: {len(t_fks)} free kicks",
                ha='center', fontproperties=fp_reg, fontsize=15, color=color)

    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='white',
               markersize=10, label='Led to Goal', lw=0),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
               markersize=7, label='Free Kick', lw=0),
        mpatches.Patch(color=home_color, label=meta['home_name']),
        mpatches.Patch(color=away_color, label=meta['away_name']),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=17,
              facecolor=PANEL_BG, edgecolor=LINE_COLOR, labelcolor=TEXT_MAIN,
              prop=fp_reg, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    ax.set_title('Free Kicks  (★ = led to goal opportunity)',
                 fontproperties=fp_bold, fontsize=16, color=TEXT_MAIN, pad=8)
    return fig


# ---------------------------------------------------------------------------
# 16. Set Piece xG vs Open Play
# ---------------------------------------------------------------------------

def render_set_piece_xg(stats: dict, meta: dict, team_colors: dict,
                          figsize=(12, 5)) -> plt.Figure:
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax, team_id, color in [(axes[0], home_id, home_color),
                                 (axes[1], away_id, away_color)]:
        ax.set_facecolor(BG)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

        s = stats[team_id]
        categories = ['Open Play', 'Set Pieces']
        values = [s.get('open_play_xg', 0), s.get('set_piece_xg', 0)]
        colors_bar = [color, color + '88']

        bars = ax.bar(categories, values, color=colors_bar, edgecolor='white',
                      lw=0.8, width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom',
                    fontproperties=fp_bold, fontsize=16, color='white')

        # Shot counts
        sh_op = s.get('open_play_shots', 0)
        sh_sp = s.get('set_piece_shots', 0)
        ax.text(0, -0.08, f"{sh_op} shots", ha='center', va='top',
                fontproperties=fp_reg, fontsize=17, color=TEXT_DIM,
                transform=ax.get_xaxis_transform())
        ax.text(1, -0.08, f"{sh_sp} shots", ha='center', va='top',
                fontproperties=fp_reg, fontsize=17, color=TEXT_DIM,
                transform=ax.get_xaxis_transform())

        team_name = meta['home_name'] if team_id == home_id else meta['away_name']
        ax.set_title(team_name, fontproperties=fp_bold, fontsize=17,
                     color=color, pad=8)
        ax.tick_params(colors=TEXT_DIM)
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(fp_reg)
            tick.set_color(TEXT_DIM)
        ax.set_ylabel('xG', fontproperties=fp_reg, fontsize=15, color=TEXT_DIM)

    fig.suptitle('xG: Open Play vs Set Pieces',
                 fontproperties=fp_bold, fontsize=16, color=TEXT_MAIN, y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Flank Attacks — Final Third Zone Map
# ---------------------------------------------------------------------------

def render_flank_attacks(flank_attacks: pd.DataFrame, meta: dict,
                          team_colors: dict, figsize=(14, 10)) -> plt.Figure:
    """
    Attacks per flank shown on a VerticalPitch zoomed into the final third.
    Attacking direction runs bottom → top. Both teams side-by-side.

    On VerticalPitch (wyscout): axes x-axis = wyscout y (0-100, left-right),
    axes y-axis = wyscout x (0-100, own goal → opp goal).
    Final third = wyscout x 66-100 → axes y 66-100 (top portion of pitch).
    Flanks by wyscout y: left=0-33, centre=33-67, right=67-100.
    """
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')
    home_name = meta['home_name']
    away_name = meta['away_name']

    vp = VerticalPitch(
        pitch_type='wyscout',
        pitch_color=PITCH_BG, line_color=LINE_COLOR,
        line_zorder=2, half=True,
    )
    fig, axes = vp.draw(nrows=1, ncols=2, figsize=figsize)
    fig.patch.set_facecolor(BG)

    # Flank zones in VerticalPitch axes coords:
    # Rectangle(xy=(axes_x_start, axes_y_start), width, height)
    # axes_x = wyscout_y, axes_y = wyscout_x
    # Zones cover wyscout x 50-100 (half pitch), wyscout y 0-33/33-67/67-100
    flank_zones = {
        'left':   {'ax_x': 0,  'ax_y': 50, 'w': 33, 'h': 50, 'label': 'Left'},
        'center': {'ax_x': 33, 'ax_y': 50, 'w': 34, 'h': 50, 'label': 'Centre'},
        'right':  {'ax_x': 67, 'ax_y': 50, 'w': 33, 'h': 50, 'label': 'Right'},
    }

    for ax, team_id, color, tname in [
        (axes[0], home_id, home_color, home_name),
        (axes[1], away_id, away_color, away_name),
    ]:
        ax.set_facecolor(PITCH_BG)

        team_data = flank_attacks[flank_attacks['team_id'] == team_id]
        counts = {row['flank']: row['count'] for _, row in team_data.iterrows()}
        max_count = max(counts.values(), default=1)

        for flank, z in flank_zones.items():
            cnt = counts.get(flank, 0)
            alpha = 0.12 + 0.62 * (cnt / max_count) if max_count > 0 else 0.12

            ax.add_patch(Rectangle(
                (z['ax_x'], z['ax_y']), z['w'], z['h'],
                facecolor=color, alpha=alpha,
                edgecolor=color, lw=1.2, zorder=3,
            ))

            cx = z['ax_x'] + z['w'] / 2   # horizontal centre of zone
            cy = z['ax_y'] + z['h'] / 2   # vertical centre of zone

            ax.text(cx, cy + 5, str(cnt),
                    ha='center', va='center', fontproperties=fp_bold,
                    fontsize=22, color='white', zorder=5,
                    path_effects=[pe.withStroke(linewidth=3, foreground='black')])
            ax.text(cx, cy - 5, z['label'],
                    ha='center', va='center', fontproperties=fp_reg,
                    fontsize=12, color=TEXT_MAIN, alpha=0.9, zorder=5)

        # Attacking direction arrow (pointing upward in VerticalPitch)
        ax.annotate('', xy=(50, 98), xytext=(50, 52),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
        ax.text(50, 50, 'Attacking', ha='center', va='top',
                fontproperties=fp_reg, fontsize=9, color=color, style='italic')

        ax.set_title(tname, fontproperties=fp_bold, fontsize=15,
                     color=color, pad=8)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 17a. Player Stats Table
# ---------------------------------------------------------------------------

def render_player_stats_table(player_stats: pd.DataFrame, team_id: int,
                               team_name: str, team_color: str,
                               sca_df: pd.DataFrame = None,
                               xt_df: pd.DataFrame = None,
                               figsize=(16, 8)) -> plt.Figure:
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    t = player_stats[player_stats['team_id'] == team_id].copy()
    if t.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', color='white',
                transform=ax.transAxes, fontproperties=fp_bold, fontsize=17)
        return fig

    # Merge SCA and xT
    if sca_df is not None and not sca_df.empty:
        t = t.merge(sca_df[['player_id', 'sca']], on='player_id', how='left', suffixes=('', '_sca'))
        if 'sca_sca' in t.columns:
            t['sca'] = t['sca_sca'].fillna(t['sca'])
            t.drop(columns=['sca_sca'], inplace=True)
    t['sca'] = t.get('sca', 0).fillna(0).astype(int)

    if xt_df is not None and not xt_df.empty:
        t = t.merge(xt_df[['player_id', 'xT']], on='player_id', how='left', suffixes=('', '_xt'))
        if 'xT_xt' in t.columns:
            t['xT'] = t['xT_xt'].fillna(t.get('xT', 0))
            t.drop(columns=['xT_xt'], inplace=True)
    t['xT'] = t.get('xT', 0).fillna(0)

    t = t.sort_values('touches', ascending=False).head(14)

    columns = ['player_name', 'role', 'minutes_played', 'touches',
               'passes', 'pass_accuracy', 'shots', 'goals',
               'xg', 'xT', 'sca', 'key_passes', 'duels', 'duel_win_pct',
               'interceptions']
    col_headers = ['Player', 'Pos', 'Min', 'Touch',
                   'Pass', 'Pass%', 'Sh', 'G',
                   'xG', 'xT', 'SCA', 'KP', 'Duel', 'Duel%',
                   'INT']

    t_display = t[columns].copy()
    t_display['xg'] = t_display['xg'].round(2)
    t_display['xT'] = t_display['xT'].round(3)
    t_display['pass_accuracy'] = t_display['pass_accuracy'].round(0).astype(int)
    t_display['duel_win_pct'] = t_display['duel_win_pct'].round(0).astype(int)

    cell_data = [t_display.iloc[i].tolist() for i in range(len(t_display))]

    table = ax.table(
        cellText=cell_data,
        colLabels=col_headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Style
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(PANEL_BG if row % 2 == 0 else BG)
        cell.set_edgecolor(LINE_COLOR)
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor(team_color + '44')
            cell.set_text_props(fontproperties=fp_bold, color=team_color, fontsize=17)
        else:
            cell.set_text_props(fontproperties=fp_reg, color=TEXT_MAIN, fontsize=17)

    table.scale(1, 1.6)

    ax.set_title(f"{team_name} — Player Statistics",
                 fontproperties=fp_bold, fontsize=16, color=team_color,
                 pad=12)
    return fig


# ---------------------------------------------------------------------------
# 17b. Starting XI Formation
# ---------------------------------------------------------------------------

# Position to (x, y) on a 100x100 pitch (attacking left to right)
POSITION_COORDS = {
    'gk': (5, 50),
    'rb': (25, 15), 'rcb': (20, 35), 'cb': (20, 50), 'lcb': (20, 65), 'lb': (25, 85),
    'rwb': (38, 10), 'lwb': (38, 90),
    'rdmf': (40, 30), 'dmf': (40, 50), 'ldmf': (40, 70),
    'rcmf': (55, 25), 'cmf': (55, 50), 'lcmf': (55, 75),
    'ramf': (68, 25), 'amf': (68, 50), 'lamf': (68, 75),
    'rw': (75, 12), 'lw': (75, 88),
    'rfw': (80, 30), 'fw': (80, 50), 'lfw': (80, 70),
    'cf': (85, 50),
    'ss': (78, 50),
}

def render_starting_xi(data: dict, meta: dict, team_colors: dict,
                        player_lookup: dict, figsize=(18, 10)) -> plt.Figure:
    from wyscout_parser import get_starting_xi

    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    home_id = meta['home_id']
    away_id = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    pitch = _pitch()
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=figsize)
    fig.patch.set_facecolor(BG)

    for ax, team_id, color in [(axes[0], home_id, home_color),
                                 (axes[1], away_id, away_color)]:
        ax.set_facecolor(PITCH_BG)
        xi = get_starting_xi(data, team_id, '1H')

        for player in xi:
            pid = player['player_id']
            pos = player['position'].lower()
            coords = POSITION_COORDS.get(pos, (50, 50))
            px, py = coords

            # Flip x for away team (they attack right to left visually)
            if team_id == away_id:
                px = 100 - px

            p_info = player_lookup.get(pid, {})
            name = p_info.get('shortName', str(pid))
            role = p_info.get('role_code2', '')

            ax.scatter(px, py, s=250, c=color, edgecolors='white', lw=1.5,
                       zorder=5, alpha=0.9)
            ax.text(px, py, role[:2], ha='center', va='center',
                    fontproperties=fp_bold, fontsize=16, color='white', zorder=6)
            ax.text(px, py - 5, name.split()[-1][:10],
                    ha='center', va='top', fontproperties=fp_reg, fontsize=15,
                    color='white', zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=PITCH_BG)])

        team_name = meta['home_name'] if team_id == home_id else meta['away_name']
        form = meta.get('home_formation_1h' if team_id == home_id else 'away_formation_1h', '')
        ax.set_title(f"{team_name}  {form}", fontproperties=fp_bold,
                     fontsize=17, color=color, pad=8)

    fig.suptitle('Starting XI', fontproperties=fp_bold, fontsize=17,
                 color=TEXT_MAIN, y=1.01)
    return fig


# ---------------------------------------------------------------------------
# 18. Individual Player Action Map
# ---------------------------------------------------------------------------

def render_player_action_map(df: pd.DataFrame, player_id: int,
                              player_name: str, team_color: str,
                              player_stats_row: pd.Series = None,
                              action_filter: list = None,
                              figsize=(12, 8)) -> plt.Figure:
    """
    Plot all actions for a player on a pitch.
    action_filter: list of strings from ['passes', 'shots', 'carries', 'duels', 'defensive']
    """
    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    if action_filter is None:
        action_filter = ['passes', 'shots', 'carries', 'duels', 'defensive']

    pitch = _pitch()
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PITCH_BG)

    p = df[df['player_id'] == player_id].copy()

    legend_elements = []

    # Passes
    if 'passes' in action_filter:
        passes = p[p['type_primary'].isin(['pass', 'free_kick', 'corner'])].copy()
        passes = passes[passes['x_plot'].notna() & passes['pass_end_x'].notna()]
        for _, ev in passes.iterrows():
            color = '#4CAF50' if ev['pass_accurate'] else '#F44336'
            alpha = 0.7 if ev['pass_accurate'] else 0.5
            lw = 2.5 if ev['is_key_pass'] else 1.8
            ax.annotate('', xy=(ev['pass_end_x'], ev['pass_end_y']),
                        xytext=(ev['x_plot'], ev['y_plot']),
                        arrowprops=dict(
                            arrowstyle='->', color=color, lw=lw,
                            alpha=alpha, mutation_scale=12,
                        ), zorder=3)
        legend_elements += [
            Line2D([0], [0], color='#4CAF50', lw=1.5, label='Accurate pass'),
            Line2D([0], [0], color='#F44336', lw=1, label='Inaccurate pass'),
        ]

    # Shots
    if 'shots' in action_filter:
        shots = p[p['type_primary'] == 'shot'].copy()
        shots = shots[shots['x_plot'].notna()]
        for _, s in shots.iterrows():
            xg = s['shot_xg'] or 0
            size = 100 + 400 * xg
            if s['shot_is_goal']:
                ax.scatter(s['x_plot'], s['y_plot'], s=size, c='gold',
                           marker='*', edgecolors='white', lw=1, zorder=6)
            elif s['shot_on_target']:
                ax.scatter(s['x_plot'], s['y_plot'], s=size, c=team_color,
                           marker='o', edgecolors='white', lw=0.8, alpha=0.85, zorder=5)
            else:
                ax.scatter(s['x_plot'], s['y_plot'], s=size * 0.7, c='none',
                           marker='X', edgecolors=team_color, lw=1, alpha=0.6, zorder=4)
        if not shots.empty:
            legend_elements += [
                Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                       markersize=10, label='Goal', lw=0),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=team_color,
                       markersize=8, label='Shot on target', lw=0),
            ]

    # Carries
    if 'carries' in action_filter:
        carries = p[p['is_carry'] == True].copy()
        carries = carries[carries['x_plot'].notna() & carries['carry_end_x'].notna()]
        for _, c in carries.iterrows():
            color = team_color
            lw = 3 if c.get('is_progressive_carry') else 2
            ax.annotate('', xy=(c['carry_end_x'], c['carry_end_y']),
                        xytext=(c['x_plot'], c['y_plot']),
                        arrowprops=dict(
                            arrowstyle='->', color=color, lw=lw,
                            alpha=0.65, linestyle='dashed',
                            mutation_scale=12,
                        ), zorder=3)
        if not carries.empty:
            legend_elements.append(
                Line2D([0], [0], color=team_color, lw=1.5, ls='--', label='Carry')
            )

    # Duels
    if 'duels' in action_filter:
        duels = p[p['duel_type'].notna()].copy()
        duels = duels[duels['x_plot'].notna()]
        duels_won = duels[duels['duel_won'] == True]
        duels_lost = duels[duels['duel_won'] != True]
        ax.scatter(duels_won['x_plot'], duels_won['y_plot'],
                   s=90, c='#4CAF50', marker='^', edgecolors='white',
                   lw=1.2, alpha=0.75, zorder=4)
        ax.scatter(duels_lost['x_plot'], duels_lost['y_plot'],
                   s=90, c='#F44336', marker='v', edgecolors='white',
                   lw=1.2, alpha=0.6, zorder=4)
        if not duels.empty:
            legend_elements += [
                Line2D([0], [0], marker='^', color='w', markerfacecolor='#4CAF50',
                       markersize=7, label='Duel won', lw=0),
                Line2D([0], [0], marker='v', color='w', markerfacecolor='#F44336',
                       markersize=7, label='Duel lost', lw=0),
            ]

    # Defensive (catch-all — shows interceptions + clearances)
    if 'defensive' in action_filter:
        inter = p[(p['type_primary'] == 'interception') & p['x_plot'].notna()]
        clrs  = p[(p['type_primary'] == 'clearance')    & p['x_plot'].notna()]
        ax.scatter(inter['x_plot'], inter['y_plot'],
                   s=110, c='none', marker='D', edgecolors='#FF9800',
                   lw=2.0, alpha=0.85, zorder=5)
        ax.scatter(clrs['x_plot'], clrs['y_plot'],
                   s=110, c='none', marker='P', edgecolors='#9C27B0',
                   lw=2.0, alpha=0.8, zorder=5)
        if len(inter) + len(clrs) > 0:
            legend_elements += [
                Line2D([0], [0], marker='D', color='w', markerfacecolor='none',
                       markeredgecolor='#FF9800', markersize=7, label='Interception', lw=0),
                Line2D([0], [0], marker='P', color='w', markerfacecolor='none',
                       markeredgecolor='#9C27B0', markersize=7, label='Clearance', lw=0),
            ]

    # Granular defensive: Interceptions only
    if 'interceptions' in action_filter:
        inter = p[(p['type_primary'] == 'interception') & p['x_plot'].notna()]
        ax.scatter(inter['x_plot'], inter['y_plot'],
                   s=75, c='none', marker='D', edgecolors='#FF9800',
                   lw=1.3, alpha=0.9, zorder=5)
        if not inter.empty:
            legend_elements.append(
                Line2D([0], [0], marker='D', color='w', markerfacecolor='none',
                       markeredgecolor='#FF9800', markersize=8, label='Interception', lw=0))

    # Granular defensive: Clearances only
    if 'clearances' in action_filter:
        clrs = p[(p['type_primary'] == 'clearance') & p['x_plot'].notna()]
        ax.scatter(clrs['x_plot'], clrs['y_plot'],
                   s=75, c='none', marker='P', edgecolors='#9C27B0',
                   lw=1.3, alpha=0.85, zorder=5)
        if not clrs.empty:
            legend_elements.append(
                Line2D([0], [0], marker='P', color='w', markerfacecolor='none',
                       markeredgecolor='#9C27B0', markersize=8, label='Clearance', lw=0))

    # Granular defensive: Challenges / Tackles
    if 'challenges' in action_filter:
        chl = p[
            (p['type_primary'] == 'duel') &
            p['duel_subtype'].isin(['ground_defending_duel', 'ground_loose_ball_duel']) &
            p['x_plot'].notna()
        ]
        chl_won  = chl[chl['duel_won'] == True]
        chl_lost = chl[chl['duel_won'] != True]
        ax.scatter(chl_won['x_plot'],  chl_won['y_plot'],  s=60, c='#4CAF50',
                   marker='^', edgecolors='white', lw=0.5, alpha=0.85, zorder=4)
        ax.scatter(chl_lost['x_plot'], chl_lost['y_plot'], s=55, c='#F44336',
                   marker='v', edgecolors='white', lw=0.5, alpha=0.65, zorder=4)
        if not chl.empty:
            legend_elements += [
                Line2D([0], [0], marker='^', color='w', markerfacecolor='#4CAF50',
                       markersize=8, label='Challenge won', lw=0),
                Line2D([0], [0], marker='v', color='w', markerfacecolor='#F44336',
                       markersize=8, label='Challenge lost', lw=0),
            ]

    # Granular defensive: Aerial Duels
    if 'aerial_duels' in action_filter:
        aer = p[
            (p['type_primary'] == 'duel') &
            p['duel_subtype'].isin(['head_duel', 'air_duel']) &
            p['x_plot'].notna()
        ]
        aer_won  = aer[aer['duel_won'] == True]
        aer_lost = aer[aer['duel_won'] != True]
        ax.scatter(aer_won['x_plot'],  aer_won['y_plot'],  s=65, c='#2196F3',
                   marker='D', edgecolors='white', lw=0.5, alpha=0.85, zorder=4)
        ax.scatter(aer_lost['x_plot'], aer_lost['y_plot'], s=55, c='none',
                   marker='D', edgecolors='#2196F3', lw=1.3, alpha=0.65, zorder=4)
        if not aer.empty:
            legend_elements += [
                Line2D([0], [0], marker='D', color='w', markerfacecolor='#2196F3',
                       markersize=8, label='Aerial won', lw=0),
                Line2D([0], [0], marker='D', color='w', markerfacecolor='none',
                       markeredgecolor='#2196F3', markersize=8, label='Aerial lost', lw=0),
            ]

    # Stats box
    if player_stats_row is not None:
        stats_text = (
            f"Touches: {int(player_stats_row.get('touches', 0))}  |  "
            f"Passes: {int(player_stats_row.get('passes', 0))} ({player_stats_row.get('pass_accuracy', 0):.0f}%)  |  "
            f"Shots: {int(player_stats_row.get('shots', 0))}  |  "
            f"xG: {player_stats_row.get('xg', 0):.2f}  |  "
            f"xT: {player_stats_row.get('xT', 0):.3f}  |  "
            f"SCA: {int(player_stats_row.get('sca', 0))}"
        )
        ax.text(50, -3, stats_text, ha='center', va='top',
                fontproperties=fp_reg, fontsize=10, color=TEXT_DIM)

    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower left', fontsize=11,
                  facecolor=PANEL_BG, edgecolor=LINE_COLOR, labelcolor=TEXT_MAIN,
                  prop=fp_reg, framealpha=0.9)

    ax.set_title(player_name, fontproperties=fp_bold, fontsize=13,
                 color=team_color, pad=8)
    return fig


# ---------------------------------------------------------------------------
# 19. Player Radar (PyPizza)
# ---------------------------------------------------------------------------

def render_player_radar(player_stats_row: pd.Series, squad_stats: pd.DataFrame,
                          player_name: str, team_color: str,
                          figsize=(8, 8)) -> plt.Figure:
    from mplsoccer import PyPizza

    fp_bold = _load_font('Bold')
    fp_reg = _load_font('Regular')

    metrics = ['touches', 'pass_accuracy', 'xg', 'xT', 'sca',
               'duel_win_pct', 'interceptions']
    labels = ['Touches', 'Pass%', 'xG', 'xT', 'SCA',
              'Duel%', 'INT']

    # Compute percentiles vs squad
    values = []
    for m in metrics:
        val = player_stats_row.get(m, 0) or 0
        squad_vals = squad_stats[m].dropna()
        if len(squad_vals) > 1:
            pct = int(100 * (squad_vals <= val).mean())
        else:
            pct = 50
        values.append(pct)

    baker = PyPizza(
        params=labels,
        straight_line_color=LINE_COLOR,
        straight_line_lw=1,
        last_circle_color=LINE_COLOR,
        last_circle_lw=2,
        other_circle_lw=1,
        other_circle_color=LINE_COLOR,
        inner_circle_size=20,
    )

    fig, ax = baker.make_pizza(
        values,
        figsize=figsize,
        color_blank_space='same',
        slice_colors=[team_color] * len(labels),
        value_colors=[TEXT_MAIN] * len(labels),
        value_bck_colors=[PANEL_BG] * len(labels),
        blank_alpha=0.15,
        kwargs_slices=dict(edgecolor=LINE_COLOR, zorder=2, linewidth=1),
        kwargs_params=dict(color=TEXT_MAIN, fontsize=10,
                          va='center', fontproperties=fp_reg),
        kwargs_values=dict(color=TEXT_MAIN, fontsize=10,
                          zorder=3, fontproperties=fp_bold,
                          bbox=dict(edgecolor=LINE_COLOR, facecolor=PANEL_BG,
                                    boxstyle='round,pad=0.2', lw=1.5)),
    )

    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    fig.text(0.5, 0.97, player_name, ha='center', va='top',
             fontproperties=fp_bold, fontsize=12, color=team_color)
    fig.text(0.5, 0.93, 'Percentile rank vs squad',
             ha='center', va='top', fontproperties=fp_reg, fontsize=10, color=TEXT_DIM)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_img(fig: plt.Figure, dpi: int = 120) -> np.ndarray:
    """Render a Figure to a PNG numpy array and close it."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    return img


def _embed_fig(ax: plt.Axes, fig: plt.Figure, dpi: int = 120) -> None:
    """Embed a sub-figure into an axes panel as an image."""
    ax.imshow(_fig_to_img(fig, dpi), aspect='auto', interpolation='lanczos')
    ax.axis('off')


def _build_sub_intervals(subs_df: pd.DataFrame, team_id: int) -> list:
    """Return [(start, end, label), ...] for a team's lineup intervals."""
    team_subs = subs_df[subs_df['team_id'] == team_id] if not subs_df.empty else pd.DataFrame()
    mins = sorted(set(team_subs['match_minute'].tolist())) if not team_subs.empty else []
    bps  = sorted(set([0] + mins + [90]))
    return [(bps[i], bps[i + 1], f"{bps[i]}\u2032\u2013{bps[i + 1]}\u2032")
            for i in range(len(bps) - 1)]


def _draw_pass_network_on_ax(ax, df, team_id, team_name, color, player_lookup,
                               min_start, min_end, subs_df=None, min_passes=1):
    """
    Draw a single-team pass network directly onto a mplsoccer pitch ax.
    - Node size proportional to passes played + received
    - Line thickness proportional to cumulative xT of that pass pair
    - Square marker for players involved in a substitution
    - No KDE background
    """
    from wyscout_parser import compute_pass_network, compute_xt_grid, _xy_to_xt_cell

    fp_bold = _load_font('Bold')
    fp_reg  = _load_font('Regular')
    pitch   = _pitch()
    pitch.draw(ax=ax)
    ax.set_facecolor(PITCH_BG)

    nodes, edges = compute_pass_network(df, team_id, '1H', min_passes, min_start, min_end)

    if nodes.empty:
        ax.set_title(f"{team_name}\n(no data)", fontproperties=fp_bold,
                     fontsize=11, color=color, pad=4)
        return

    # Pass volume per player (played + received)
    pass_vol = {}
    if not edges.empty:
        for _, e in edges.iterrows():
            pid  = e['player_id']
            rpid = e['pass_recipient_id']
            pass_vol[pid]  = pass_vol.get(pid, 0)  + e['pass_count']
            pass_vol[rpid] = pass_vol.get(rpid, 0) + e['pass_count']
    max_vol = max(pass_vol.values()) if pass_vol else 1

    # xT per edge
    edge_xt = {}
    try:
        xt_grid = compute_xt_grid()
        tf = df[(df['team_id'] == team_id) &
                (df['match_minute'] >= min_start) &
                (df['match_minute'] < min_end) &
                df['type_primary'].isin(['pass', 'free_kick', 'corner', 'throw_in', 'goal_kick']) &
                (df['pass_accurate'] == True) &
                df['pass_recipient_id'].notna()].copy()
        for _, r in tf.iterrows():
            try:
                rc, cc = _xy_to_xt_cell(r['x_plot'], r['y_plot'])
                re, ce = _xy_to_xt_cell(r['pass_end_x'], r['pass_end_y'])
                gain = max(xt_grid[re, ce] - xt_grid[rc, cc], 0)
                key = (r['player_id'], int(r['pass_recipient_id']))
                edge_xt[key] = edge_xt.get(key, 0) + gain
            except Exception:
                pass
    except Exception:
        pass
    max_xt = max(edge_xt.values()) if edge_xt else 1

    # Substitution player IDs → square marker
    sub_pids = set()
    if subs_df is not None and not subs_df.empty:
        for _, s in subs_df[subs_df['team_id'] == team_id].iterrows():
            sub_pids.add(s['player_in_id'])
            sub_pids.add(s['player_out_id'])

    # Edges
    if not edges.empty:
        for _, e in edges.iterrows():
            key   = (e['player_id'], e['pass_recipient_id'])
            xt_v  = edge_xt.get(key, 0)
            lw    = 0.6 + 3.4 * (xt_v / max_xt)
            alpha = 0.30 + 0.50 * (xt_v / max_xt)
            ax.plot([e['from_x'], e['to_x']], [e['from_y'], e['to_y']],
                    color=color, lw=lw, alpha=alpha, zorder=3)

    # Nodes
    for _, n in nodes.iterrows():
        pid    = n['player_id']
        vol    = pass_vol.get(pid, 1)
        sz     = 80 + 320 * (vol / max_vol)
        marker = 's' if pid in sub_pids else 'o'
        ax.scatter(n['avg_x'], n['avg_y'], s=sz, c=color, marker=marker,
                   edgecolors='white', lw=1.2, zorder=5, alpha=0.9)
        p_info = (player_lookup or {}).get(int(pid), {}) if pid else {}
        shirt  = (p_info or {}).get('shirtNumber')
        lbl_t  = (str(shirt) if shirt else
                  (str(n['player_name']).split()[-1] if n['player_name'] else ''))
        ax.text(n['avg_x'], n['avg_y'] - 4, lbl_t,
                ha='center', va='top', fontproperties=fp_reg, fontsize=10,
                color='white', zorder=6,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=PITCH_BG)])

    lbl = f"{min_start}\u2032\u2013{min_end}\u2032" if min_start is not None else 'Full Match'
    ax.set_title(f"{team_name}  \u00b7  {lbl}",
                 fontproperties=fp_bold, fontsize=12, color=color, pad=5)


def _draw_shot_table(ax, shots_df, team_name, team_color):
    """Draw a numbered shot breakdown table directly on ax."""
    fp_bold = _load_font('Bold')
    fp_reg  = _load_font('Regular')
    ax.set_facecolor(PANEL_BG)
    ax.axis('off')

    shots_df = shots_df.reset_index(drop=True)
    ax.text(0.5, 0.97, f"{team_name}  \u2014  {len(shots_df)} Shot{'s' if len(shots_df) != 1 else ''}",
            fontproperties=fp_bold, fontsize=12, color=team_color,
            ha='center', va='top', transform=ax.transAxes)

    headers = ['#', 'Player', "Min'", 'xG', 'Situation', 'Result']
    col_x   = [0.02, 0.10, 0.54, 0.64, 0.74, 0.87]
    hy = 0.85
    for hx, ht in zip(col_x, headers):
        ax.text(hx, hy, ht, fontproperties=fp_bold, fontsize=9, color=TEXT_DIM,
                va='top', transform=ax.transAxes)
    ax.plot([0.02, 0.98], [hy - 0.05, hy - 0.05],
            color=LINE_COLOR, lw=0.8, transform=ax.transAxes)

    max_rows = 22
    usable   = hy - 0.10
    row_h    = usable / max(max_rows, 1)
    for i, (_, row) in enumerate(shots_df.iterrows()):
        if i >= max_rows:
            break
        y       = hy - 0.10 - i * row_h
        is_goal = bool(row.get('shot_is_goal', False))
        is_sot  = bool(row.get('shot_on_target', False))
        result  = 'GOAL' if is_goal else ('On Target' if is_sot else 'Off Target')
        rc      = team_color if is_goal else (TEXT_MAIN if is_sot else TEXT_DIM)
        sit     = str(row.get('shot_situation') or 'Open Play')
        sit_short = {'Open Play': 'Open', 'Free Kick': 'FK', 'Corner': 'CK', 'Penalty': 'PK'}.get(sit, sit)
        items   = [str(i + 1),
                   str(row.get('player_name', ''))[:22],
                   f"{int(row.get('match_minute', 0))}'",
                   f"{(row.get('shot_xg') or 0):.2f}",
                   sit_short]
        for hx, txt in zip(col_x, items):
            ax.text(hx, y, txt, fontproperties=fp_reg, fontsize=8.5,
                    color=TEXT_MAIN, va='top', transform=ax.transAxes)
        ax.text(col_x[5], y, result, fontproperties=fp_reg, fontsize=8.5,
                color=rc, va='top', transform=ax.transAxes)


def render_pass_network_grid(df: pd.DataFrame, meta: dict, team_colors: dict,
                              player_lookup: dict, subs_df: pd.DataFrame,
                              figsize=(20, 14)) -> plt.Figure:
    """Grid of pass networks for every lineup interval — 2 rows (home/away) × N cols."""
    fp_bold    = _load_font('Bold')
    home_id    = meta['home_id'];   away_id   = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')
    home_name  = meta['home_name']; away_name = meta['away_name']

    home_ints = _build_sub_intervals(subs_df, home_id)
    away_ints = _build_sub_intervals(subs_df, away_id)
    max_cols  = max(len(home_ints), len(away_ints), 1)

    fig = plt.figure(figsize=figsize, facecolor=BG)
    gs  = gridspec.GridSpec(2, max_cols, figure=fig, hspace=0.35, wspace=0.08)

    for col, (s, e, _) in enumerate(home_ints):
        ax = fig.add_subplot(gs[0, col])
        _draw_pass_network_on_ax(ax, df, home_id, home_name, home_color, player_lookup, s, e, subs_df=subs_df)

    for col, (s, e, _) in enumerate(away_ints):
        ax = fig.add_subplot(gs[1, col])
        _draw_pass_network_on_ax(ax, df, away_id, away_name, away_color, player_lookup, s, e, subs_df=subs_df)

    # Blank cells when home has fewer intervals than away
    for col in range(len(home_ints), max_cols):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(BG)
        ax.axis('off')

    fig.suptitle(
        'Pass Networks  \u00b7  Each panel = one lineup interval  \u00b7  '
        'Node size = touches  \u00b7  Line width = passes',
        fontproperties=fp_bold, fontsize=14, color=TEXT_MAIN, y=1.01,
    )
    return fig


def render_pass_network_vertical(df: pd.DataFrame, meta: dict, team_colors: dict,
                                  player_lookup: dict, subs_df: pd.DataFrame = None,
                                  figsize=(18, 14)) -> plt.Figure:
    """
    Two vertical pitches (home | away) showing full-match pass network.
    All players (starters + subs), full 90 minutes, min_passes=1.
    Node size = passes played+received. Edge width = xT. Squares for subs.
    """
    from wyscout_parser import compute_pass_network, compute_xt_grid, _xy_to_xt_cell

    fp_bold = _load_font('Bold')
    fp_reg  = _load_font('Regular')

    home_id    = meta['home_id'];  away_id   = meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')
    home_name  = meta['home_name']; away_name = meta['away_name']

    vp = VerticalPitch(
        pitch_type='wyscout',
        pitch_color=PITCH_BG,
        line_color=LINE_COLOR,
        line_zorder=2,
    )
    fig, axes = vp.draw(nrows=1, ncols=2, figsize=figsize)
    fig.patch.set_facecolor(BG)

    for ax, team_id, color, team_name_t in [
        (axes[0], home_id, home_color, home_name),
        (axes[1], away_id, away_color, away_name),
    ]:
        ax.set_facecolor(PITCH_BG)

        nodes, edges = compute_pass_network(df, team_id, '1H', 1, 0, 90)

        if nodes.empty:
            ax.set_title(f"{team_name_t}\n(no data)", fontproperties=fp_bold,
                         fontsize=12, color=color, pad=4)
            continue

        # Pass volume per player (played + received)
        pass_vol = {}
        if not edges.empty:
            for _, e in edges.iterrows():
                pid  = e['player_id']
                rpid = e['pass_recipient_id']
                pass_vol[pid]  = pass_vol.get(pid, 0) + e['pass_count']
                pass_vol[rpid] = pass_vol.get(rpid, 0) + e['pass_count']
        max_vol = max(pass_vol.values()) if pass_vol else 1

        # xT per edge
        edge_xt = {}
        try:
            xt_grid = compute_xt_grid()
            tf = df[
                (df['team_id'] == team_id) &
                df['type_primary'].isin(['pass', 'free_kick', 'corner', 'throw_in', 'goal_kick']) &
                (df['pass_accurate'] == True) &
                df['pass_recipient_id'].notna()
            ].copy()
            for _, r in tf.iterrows():
                try:
                    rc, cc = _xy_to_xt_cell(r['x_plot'], r['y_plot'])
                    re, ce = _xy_to_xt_cell(r['pass_end_x'], r['pass_end_y'])
                    gain = max(xt_grid[re, ce] - xt_grid[rc, cc], 0)
                    key = (r['player_id'], int(r['pass_recipient_id']))
                    edge_xt[key] = edge_xt.get(key, 0) + gain
                except Exception:
                    pass
        except Exception:
            pass
        max_xt = max(edge_xt.values()) if edge_xt else 1

        # Substitution player IDs → square marker
        sub_pids = set()
        if subs_df is not None and not subs_df.empty:
            for _, s in subs_df[subs_df['team_id'] == team_id].iterrows():
                sub_pids.add(s['player_in_id'])
                sub_pids.add(s['player_out_id'])

        # Draw edges — VerticalPitch: plot(y_coord, x_coord)
        if not edges.empty:
            for _, e in edges.iterrows():
                key   = (e['player_id'], e['pass_recipient_id'])
                xt_v  = edge_xt.get(key, 0)
                lw    = 0.6 + 3.4 * (xt_v / max_xt)
                alpha = 0.30 + 0.50 * (xt_v / max_xt)
                ax.plot([e['from_y'], e['to_y']], [e['from_x'], e['to_x']],
                        color=color, lw=lw, alpha=alpha, zorder=3)

        # Draw nodes — VerticalPitch: scatter(y_coord, x_coord)
        for _, n in nodes.iterrows():
            pid    = n['player_id']
            vol    = pass_vol.get(pid, 1)
            sz     = 80 + 320 * (vol / max_vol)
            marker = 's' if pid in sub_pids else 'o'
            ax.scatter(n['avg_y'], n['avg_x'], s=sz, c=color, marker=marker,
                       edgecolors='white', lw=1.2, zorder=5, alpha=0.9)
            p_info = (player_lookup or {}).get(int(pid), {}) if pid else {}
            shirt  = (p_info or {}).get('shirtNumber')
            lbl_t  = (str(shirt) if shirt else
                      (str(n['player_name']).split()[-1] if n['player_name'] else ''))
            ax.text(n['avg_y'], n['avg_x'] - 4, lbl_t,
                    ha='center', va='top', fontproperties=fp_reg, fontsize=10,
                    color='white', zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=PITCH_BG)])

        ax.set_title(team_name_t, fontproperties=fp_bold, fontsize=13, color=color, pad=5)

    fig.suptitle(
        'Pass Networks  \u00b7  Full Match  \u00b7  Node size = touches  '
        '\u00b7  Line width = xT  \u00b7  \u25a1 = substitute',
        fontproperties=fp_bold, fontsize=12, color=TEXT_MAIN, y=1.01,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard 1: Match Story
# ─────────────────────────────────────────────────────────────────────────────

def render_match_story_dashboard(match, data: dict, team_colors: dict,
                                  figsize=(20, 34)) -> plt.Figure:
    """
    Single large figure combining:
      Row 0 — Match header
      Row 1 — Key events timeline
      Row 2 — xG race (L) + Net momentum (R)
      Row 3 — Key stats comparison
      Row 4 — Rolling stats 2x2 grid
    """
    from wyscout_parser import extract_key_events, compute_momentum

    fp_bold = _load_font('Bold')
    fp_reg  = _load_font('Regular')

    key_events = extract_key_events(match.events_df, data, match.meta, match.player_lookup)
    subs_df    = key_events['subs']

    home_id    = match.meta['home_id'];  away_id   = match.meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')
    home_name  = match.meta['home_name']
    away_name  = match.meta['away_name']

    fig = plt.figure(figsize=figsize, facecolor=BG)
    gs  = gridspec.GridSpec(5, 2, figure=fig,
                            height_ratios=[3, 2, 5, 7, 10],
                            hspace=0.12, wspace=0.1)

    # Row 0: Match header (full width)
    ax0 = fig.add_subplot(gs[0, :])
    _embed_fig(ax0, render_match_header(match.meta, match.stats, team_colors, figsize=(20, 3)))

    # Row 1: Timeline (full width)
    ax1 = fig.add_subplot(gs[1, :])
    _embed_fig(ax1, render_key_events_timeline(
        match.goals, match.cards, subs_df, match.meta, team_colors, figsize=(20, 2.5)))

    # Row 2: xG race (left) + Momentum (right)
    ax2l = fig.add_subplot(gs[2, 0])
    _embed_fig(ax2l, render_xg_race(
        match.xg_timeline, match.goals, match.meta, team_colors, figsize=(12, 5)))

    momentum = compute_momentum(match.events_df, home_id, away_id, window=5)
    ax2r = fig.add_subplot(gs[2, 1])
    _embed_fig(ax2r, render_momentum(momentum, match.meta, team_colors, figsize=(9, 5)))

    # Row 3: Key stats (full width)
    ax3 = fig.add_subplot(gs[3, :])
    _embed_fig(ax3, render_key_stats(match.stats, match.meta, team_colors, match.ppda,
                                     figsize=(20, 7)))

    # Row 4: Rolling stats 2×2
    rs = match.rolling_stats
    if rs is not None and not rs.empty:
        gs4 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[4, :],
                                               hspace=0.5, wspace=0.35)
        rolling_panels = [
            ('home_poss_pct',        'away_poss_pct',        'Possession %',    0,    100),
            ('home_duel_win_pct',    'away_duel_win_pct',    'Duel Win %',      0,    100),
            ('home_attacks_per_min', 'away_attacks_per_min', 'Attacks / Min',   None, None),
            ('home_pass_acc',        'away_pass_acc',        'Pass Accuracy %', 0,    100),
        ]
        for idx, (hcol, acol, title, ymin, ymax) in enumerate(rolling_panels):
            r, c = divmod(idx, 2)
            ax = fig.add_subplot(gs4[r, c])
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_DIM, labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor(LINE_COLOR)
            x = rs['bucket']
            if hcol in rs.columns and acol in rs.columns:
                ax.fill_between(x, rs[hcol], alpha=0.18, color=home_color)
                ax.fill_between(x, rs[acol], alpha=0.18, color=away_color)
                ax.plot(x, rs[hcol], color=home_color, lw=2, label=home_name)
                ax.plot(x, rs[acol], color=away_color, lw=2, label=away_name)
            for _, g in match.goals.iterrows():
                ax.axvline(g['match_minute'], color='white', lw=0.8, alpha=0.35, ls='--')
            ax.set_title(title, fontproperties=fp_bold, fontsize=12, color=TEXT_MAIN, pad=4)
            ax.set_xlabel('Minute', fontproperties=fp_reg, fontsize=9, color=TEXT_DIM)
            if ymin is not None:
                ax.set_ylim(ymin, ymax)
            leg = ax.legend(fontsize=8, framealpha=0, loc='upper left')
            for txt in leg.get_texts():
                txt.set_color(TEXT_DIM)
    else:
        ax_blank = fig.add_subplot(gs[4, :])
        ax_blank.set_facecolor(BG)
        ax_blank.axis('off')
        ax_blank.text(0.5, 0.5, 'Rolling stats not available',
                      ha='center', va='center', color=TEXT_DIM,
                      fontproperties=fp_reg, fontsize=13, transform=ax_blank.transAxes)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard 2: Attacking & Possession
# ─────────────────────────────────────────────────────────────────────────────

def render_shots_dashboard(match, team_colors: dict,
                            figsize=(22, 36)) -> plt.Figure:
    """
    Dashboard 2a: Shots & Attacking
      Row 0 — [Home shot table | Shot map pitch | Away shot table]
      Row 1 — SCA by player (L) + xT by player (R)
    """
    home_id    = match.meta['home_id'];  away_id   = match.meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    fig = plt.figure(figsize=figsize, facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            height_ratios=[20, 8],
                            width_ratios=[1, 2.2, 1],
                            hspace=0.06, wspace=0.04)

    # Row 0 col 1: Shot map (centre)
    ax_map = fig.add_subplot(gs[0, 1])
    _embed_fig(ax_map, render_shot_map(match.events_df, match.meta, team_colors, figsize=(12, 18)))

    # Row 0 col 0: Home shot table (left)
    shots = match.events_df[
        (match.events_df['type_primary'] == 'shot') &
        match.events_df['type_secondary'].apply(lambda ts: 'shot_against' not in (ts or []))
    ].sort_values('match_minute')

    ax_home = fig.add_subplot(gs[0, 0])
    _draw_shot_table(ax_home, shots[shots['team_id'] == home_id], match.meta['home_name'], home_color)

    # Row 0 col 2: Away shot table (right)
    ax_away = fig.add_subplot(gs[0, 2])
    _draw_shot_table(ax_away, shots[shots['team_id'] == away_id], match.meta['away_name'], away_color)

    # Row 1: SCA (L) + xT by player (R)
    ax1l = fig.add_subplot(gs[1, :2])
    _embed_fig(ax1l, render_sca_by_player(match.sca_df, match.meta, team_colors, figsize=(14, 7)))

    ax1r = fig.add_subplot(gs[1, 2:])
    _embed_fig(ax1r, render_xt_by_player(match.xt_by_player, match.meta, team_colors, figsize=(10, 7)))

    return fig


def render_possession_dashboard(match, data: dict, team_colors: dict,
                                 figsize=(20, 28)) -> plt.Figure:
    """
    Dashboard 2b: Possession & Passing
      Row 0 — Pass networks (2 vertical pitches, full match, all players)
      Row 1 — Offensive action map (dual pitch, corners + free kicks marked)
    """
    from wyscout_parser import extract_key_events

    key_events = extract_key_events(match.events_df, data, match.meta, match.player_lookup)
    subs_df    = key_events['subs']

    fig = plt.figure(figsize=figsize, facecolor=BG)
    gs  = gridspec.GridSpec(2, 1, figure=fig,
                            height_ratios=[14, 10],
                            hspace=0.06)

    # Row 0: Pass networks — 2 vertical pitches, full match
    ax0 = fig.add_subplot(gs[0])
    _embed_fig(ax0, render_pass_network_vertical(
        match.events_df, match.meta, team_colors, match.player_lookup, subs_df,
        figsize=(18, 14)))

    # Row 1: Offensive action map — dual pitch, corners + FKs marked
    ax1 = fig.add_subplot(gs[1])
    _embed_fig(ax1, render_offensive_actions_dual(
        match.events_df, match.meta, team_colors, figsize=(20, 10)))

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard 3: Defensive
# ─────────────────────────────────────────────────────────────────────────────

def render_defensive_dashboard(match, team_colors: dict,
                                figsize=(20, 24)) -> plt.Figure:
    """
    Single large figure combining:
      Row 0 — Defensive actions pitch map
      Row 1 — Pressing (PPDA) map
      Row 2 — PPDA summary text bar
    """
    fp_bold = _load_font('Bold')
    fp_reg  = _load_font('Regular')
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    fig = plt.figure(figsize=figsize, facecolor=BG)
    gs  = gridspec.GridSpec(3, 1, figure=fig,
                            height_ratios=[10, 10, 2],
                            hspace=0.08)

    # Row 0: Defensive actions
    ax0 = fig.add_subplot(gs[0])
    _embed_fig(ax0, render_defensive_actions(
        match.events_df, match.meta, team_colors, figsize=(20, 10)))

    # Row 1: Pressing map
    ax1 = fig.add_subplot(gs[1])
    _embed_fig(ax1, render_pressing_map(
        match.events_df, match.meta, team_colors, match.ppda, figsize=(20, 10)))

    # Row 2: PPDA summary
    ax2 = fig.add_subplot(gs[2])
    ax2.set_facecolor(PANEL_BG)
    ax2.axis('off')
    home_ppda = match.ppda.get(match.meta['home_id'], 0)
    away_ppda = match.ppda.get(match.meta['away_id'], 0)
    home_label = 'High press' if home_ppda < 5 else ('Moderate press' if home_ppda < 10 else 'Low press')
    away_label = 'High press' if away_ppda < 5 else ('Moderate press' if away_ppda < 10 else 'Low press')
    txt = (f"{match.meta['home_name']}  PPDA: {home_ppda:.2f}  ({home_label})"
           f"          {match.meta['away_name']}  PPDA: {away_ppda:.2f}  ({away_label})"
           f"          (Lower PPDA = more aggressive high press)")
    ax2.text(0.5, 0.5, txt, ha='center', va='center',
             fontproperties=fp_reg, fontsize=12, color=TEXT_DIM,
             transform=ax2.transAxes)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard 4: Set Pieces
# ─────────────────────────────────────────────────────────────────────────────

def render_set_pieces_dashboard(match, team_colors: dict,
                                 figsize=(20, 26)) -> plt.Figure:
    """
    Single large figure combining:
      Row 0 — Corner map (L) + Free kick map (R)
      Row 1 — Set piece xG vs open play
    """
    fig = plt.figure(figsize=figsize, facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[12, 4],
                            hspace=0.08, wspace=0.06)

    # Row 0: Corner map (L) + Free kick map (R)
    ax0l = fig.add_subplot(gs[0, 0])
    _embed_fig(ax0l, render_corner_map(match.events_df, match.meta, team_colors, figsize=(10, 12)))

    ax0r = fig.add_subplot(gs[0, 1])
    _embed_fig(ax0r, render_free_kick_map(match.events_df, match.meta, team_colors, figsize=(10, 12)))

    # Row 1: Set piece xG (full width)
    ax1 = fig.add_subplot(gs[1, :])
    _embed_fig(ax1, render_set_piece_xg(match.stats, match.meta, team_colors, figsize=(20, 4)))

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard 5: Player Reports
# ─────────────────────────────────────────────────────────────────────────────

def render_player_dashboard(match, data: dict, team_colors: dict,
                             figsize=(20, 46)) -> plt.Figure:
    """
    Single large figure combining:
      Row 0 — Starting XI formations
      Row 1 — Home player stats table
      Row 2 — Away player stats table
      Row 3 — Top player 1 action map (L) + radar (R)
      Row 4 — Top player 2 action map (L) + radar (R)
    """
    home_id    = match.meta['home_id'];  away_id   = match.meta['away_id']
    home_color = team_colors.get('home', '#4fc3f7')
    away_color = team_colors.get('away', '#ffb300')

    fig = plt.figure(figsize=figsize, facecolor=BG)
    gs  = gridspec.GridSpec(5, 3, figure=fig,
                            height_ratios=[8, 8, 8, 11, 11],
                            width_ratios=[2, 1, 1],
                            hspace=0.1, wspace=0.08)

    # Row 0: Starting XI (full width)
    ax0 = fig.add_subplot(gs[0, :])
    _embed_fig(ax0, render_starting_xi(data, match.meta, team_colors,
                                        match.player_lookup, figsize=(20, 8)))

    # Row 1: Home player stats table
    ax1 = fig.add_subplot(gs[1, :])
    _embed_fig(ax1, render_player_stats_table(
        match.player_stats, home_id, match.meta['home_name'], home_color,
        match.sca_df, match.xt_by_player, figsize=(20, 7)))

    # Row 2: Away player stats table
    ax2 = fig.add_subplot(gs[2, :])
    _embed_fig(ax2, render_player_stats_table(
        match.player_stats, away_id, match.meta['away_name'], away_color,
        match.sca_df, match.xt_by_player, figsize=(20, 7)))

    # Identify top 2 away players: top-xG and top-SCA
    away_ps = match.player_stats[match.player_stats['team_id'] == away_id].copy()
    top_xg_row  = away_ps.sort_values('xg', ascending=False).iloc[0] if len(away_ps) else None
    top_sca_row = None
    if not match.sca_df.empty:
        away_sca = match.sca_df[match.sca_df['team_id'] == away_id].sort_values('sca', ascending=False)
        if len(away_sca):
            pid = int(away_sca.iloc[0]['player_id'])
            candidates = away_ps[away_ps['player_id'] == pid]
            if len(candidates):
                top_sca_row = candidates.iloc[0]
    # Fallback: second-highest xG player
    if top_sca_row is None and len(away_ps) > 1:
        top_sca_row = away_ps.sort_values('xg', ascending=False).iloc[1]

    away_squad = away_ps.copy()
    for row_idx, (row, color) in enumerate([(top_xg_row, away_color),
                                             (top_sca_row, away_color)]):
        if row is None:
            continue
        pid   = int(row['player_id'])
        pname = str(row['player_name'])

        ax_map = fig.add_subplot(gs[3 + row_idx, 0])
        _embed_fig(ax_map, render_player_action_map(
            match.events_df, pid, pname, color,
            player_stats_row=row,
            action_filter=['passes', 'shots', 'carries', 'duels', 'defensive'],
            figsize=(13, 10)))

        ax_rad = fig.add_subplot(gs[3 + row_idx, 1:])
        _embed_fig(ax_rad, render_player_radar(row, away_squad, pname, color, figsize=(9, 9)))

    return fig
