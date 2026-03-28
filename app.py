"""
app.py — Wyscout Post-Match Report
Streamlit app: upload any Wyscout match JSON to get a full interactive analysis.

Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
from io import BytesIO


def _hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,a)' for Plotly fillcolor."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


@st.cache_data(show_spinner=False)
def _fetch_logo_bytes(team_name: str) -> bytes | None:
    """Fetch team logo from luukhopman GitHub logos repo. Returns PNG bytes or None."""
    base = "https://raw.githubusercontent.com/luukhopman/football-logos/master/logos"
    leagues = ["Saudi Pro League", "Premier League", "La Liga", "Bundesliga",
               "Serie A", "Ligue 1", "Champions League"]
    for league in leagues:
        url = f"{base}/{league}/{team_name}.png"
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return r.content
        except Exception:
            continue
    return None


ZONE_NAMES = {
    'gc':  'Centre',
    'gb':  'Bottom Left',
    'gbr': 'Bottom Right',
    'glb': 'Far Left',
    'bc':  'Over Bar (Centre)',
    'plb': 'Over Bar (Left)',
    'otl': 'Wide Left',
    'olb': 'Left Post',
    'obr': 'Right Post',
}

from wyscout_parser import (
    load_match, parse_all, compute_momentum, get_starting_xi,
    extract_key_events, compute_pass_network,
)
from visuals import (
    render_match_header, render_key_events_timeline, render_key_stats,
    render_xg_race, render_momentum,
    render_shot_map, render_goal_frame, render_xg_by_player,
    render_sca_by_player,
    render_pass_network, render_offensive_actions,
    render_defensive_actions, render_pressing_map,
    render_duel_map, render_transitions_map, compute_transitions_table,
    render_corner_map, render_free_kick_map, render_set_piece_xg,
    render_player_stats_table, render_starting_xi,
    render_player_action_map, render_player_radar,
    render_flank_attacks,
    BG, PANEL_BG, TEXT_MAIN, TEXT_DIM, LINE_COLOR,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Post-Match Report",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #00060d; }
    .stApp { background-color: #00060d; }
    section[data-testid="stSidebar"] { background-color: #060f1e; }
    .stMarkdown, .stText, label, .stSelectbox label, .stCheckbox label {
        color: #e8e8e8 !important;
    }
    .metric-card {
        background: #0d1f35;
        border: 1px solid #3a5a7a;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
        margin: 4px 0;
    }
    .metric-value { font-size: 26px; font-weight: 700; }
    .metric-label { font-size: 11px; color: #7a8a9a; margin-top: 2px; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0d1f35; border-radius: 6px; }
    .stTabs [data-baseweb="tab"] { color: #7a8a9a; background-color: transparent; }
    .stTabs [aria-selected="true"] { color: #e8e8e8 !important; background-color: #1a3050 !important; }
    .stDataFrame { background-color: #0d1f35; }
    h1, h2, h3 { color: #e8e8e8 !important; }
    .stSelectbox > div > div { background-color: #0d1f35; color: #e8e8e8; }
    div[data-testid="stExpander"] { background-color: #0d1f35; border: 1px solid #3a5a7a; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
if 'match_data' not in st.session_state:
    st.session_state.match_data = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Post-Match Report")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Wyscout Match JSON",
        type=["json"],
        help="Upload the Wyscout event data JSON file for any match",
    )

    if uploaded_file is not None:
        with st.spinner("Parsing match data..."):
            try:
                raw_data = load_match(uploaded_file)
                match = parse_all(raw_data)
                st.session_state.match_data = match
                st.session_state.raw_data = raw_data
                st.success("✓ Match loaded successfully!")
            except Exception as e:
                st.error(f"Error parsing file: {e}")
                st.stop()

    if st.session_state.match_data is not None:
        match = st.session_state.match_data
        m = match.meta
        st.markdown(f"**{m['home_name']}** vs **{m['away_name']}**")
        st.markdown(f"Score: **{m['home_score']}–{m['away_score']}**")
        st.markdown(f"Date: {m['date_str']}")
        st.markdown("---")

        # Team color pickers
        st.markdown("### Team Colors")
        home_color = st.color_picker(
            f"{m['home_name']}", "#00A651", key="home_color"
        )
        away_color = st.color_picker(
            f"{m['away_name']}", "#C8A400", key="away_color"
        )
        TEAM_COLORS = {'home': home_color, 'away': away_color}
        st.markdown("---")
        st.markdown("### Quick Stats")
        home_id = m['home_id']
        away_id = m['away_id']
        hs = match.stats.get(home_id, {})
        aws = match.stats.get(away_id, {})
        st.markdown(f"**{m['home_name']}**: {hs.get('shots',0)} shots | xG {hs.get('xg',0):.2f}")
        st.markdown(f"**{m['away_name']}**: {aws.get('shots',0)} shots | xG {aws.get('xg',0):.2f}")
        st.markdown(f"PPDA — {m['home_name']}: `{match.ppda.get(home_id,0):.2f}` | {m['away_name']}: `{match.ppda.get(away_id,0):.2f}`")
    else:
        st.info("Upload a Wyscout JSON file to begin analysis")
        TEAM_COLORS = {'home': '#4fc3f7', 'away': '#ffb300'}


# ─── Main Content ─────────────────────────────────────────────────────────────
if st.session_state.match_data is None:
    # Landing page
    st.markdown("""
    # ⚽ Wyscout Post-Match Report

    Upload a **Wyscout event data JSON** file using the sidebar to generate a complete tactical analysis report.

    ### What's included:
    - **Match Story** — Score header, key events timeline, xG race chart, match momentum
    - **Attacking** — Shot map, goal frame visualization, xG by player, shot breakdown table
    - **Possession** — Pass network, touch heatmap, progressive actions, xT by player
    - **Defensive** — Defensive actions map, PPDA pressing analysis, duels breakdown
    - **Set Pieces** — Corner kick map, free kick map, set piece vs open play xG
    - **Players** — Individual player action maps, radar charts, detailed stats, multi-player comparison

    ### Metrics included:
    - xG (Expected Goals) · xT (Expected Threat) · PPDA · SCA (Shot-Creating Actions)
    - Progressive passes & carries · Pass networks · Touch heatmaps

    > *Supports any Wyscout match JSON format*
    """)
    st.stop()

# ─── Load data ────────────────────────────────────────────────────────────────
match = st.session_state.match_data
raw_data = st.session_state.raw_data
m = match.meta
home_id = m['home_id']
away_id = m['away_id']
home_name = m['home_name']
away_name = m['away_name']
home_color = TEAM_COLORS['home']
away_color = TEAM_COLORS['away']

key_events = extract_key_events(match.events_df, raw_data, m, match.player_lookup)
subs_df = key_events['subs']
momentum_df = compute_momentum(match.events_df, home_id, away_id, window=5)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📖 Match Story",
    "⚽ Attacking & Possession",
    "🛡️ Defensive",
    "👤 Players",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MATCH STORY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Header
    # ── Header with logos ──────────────────────────────────────────────────────
    logo_home_bytes = _fetch_logo_bytes(home_name)
    logo_away_bytes = _fetch_logo_bytes(away_name)

    hdr_cols = st.columns([1, 6, 1])
    with hdr_cols[0]:
        if logo_home_bytes:
            st.image(logo_home_bytes, width=90)
    with hdr_cols[1]:
        fig_header = render_match_header(m, match.stats, TEAM_COLORS, figsize=(18, 5))
        st.pyplot(fig_header, use_container_width=True)
        plt.close(fig_header)
    with hdr_cols[2]:
        if logo_away_bytes:
            st.image(logo_away_bytes, width=90)

    st.markdown("---")

    # ── Key stats summary metrics ──────────────────────────────────────────────
    hs = match.stats[home_id]
    aws = match.stats[away_id]
    cols = st.columns(6)
    metric_pairs = [
        (f"{hs['goals']}–{aws['goals']}", "Goals"),
        (f"{hs['xg']:.2f} – {aws['xg']:.2f}", "xG"),
        (f"{hs['shots']} – {aws['shots']}", "Shots"),
        (f"{hs['possession_pct']:.0f}% – {aws['possession_pct']:.0f}%", "Possession"),
        (f"{hs['pass_accuracy']:.0f}% – {aws['pass_accuracy']:.0f}%", "Pass Accuracy"),
        (f"{match.ppda.get(home_id,0):.1f} – {match.ppda.get(away_id,0):.1f}", "PPDA"),
    ]
    for col, (val, label) in zip(cols, metric_pairs):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Timeline ───────────────────────────────────────────────────────────────
    st.markdown("### Key Events Timeline")
    fig_tl = render_key_events_timeline(match.goals, match.cards, subs_df, m, TEAM_COLORS)
    st.pyplot(fig_tl, use_container_width=True)
    plt.close(fig_tl)

    st.markdown("---")

    # ── xG Race + Momentum ────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### xG Race")
        if not match.xg_timeline.empty:
            fig_xg = go.Figure()
            fig_xg.add_trace(go.Scatter(
                x=match.xg_timeline['match_minute'],
                y=match.xg_timeline['home_xg_cumul'],
                mode='lines', name=home_name,
                line=dict(color=home_color, width=2.5),
                fill='tozeroy', fillcolor=_hex_to_rgba(home_color),
            ))
            fig_xg.add_trace(go.Scatter(
                x=match.xg_timeline['match_minute'],
                y=match.xg_timeline['away_xg_cumul'],
                mode='lines', name=away_name,
                line=dict(color=away_color, width=2.5),
                fill='tozeroy', fillcolor=_hex_to_rgba(away_color),
            ))
            if not match.goals.empty:
                for _, g in match.goals.iterrows():
                    color = home_color if g['team_id'] == home_id else away_color
                    row = match.xg_timeline[match.xg_timeline['match_minute'] <= g['match_minute']]
                    if not row.empty:
                        yval = row.iloc[-1]['home_xg_cumul' if g['team_id'] == home_id else 'away_xg_cumul']
                        fig_xg.add_trace(go.Scatter(
                            x=[g['match_minute']], y=[yval],
                            mode='markers+text',
                            marker=dict(symbol='star', size=14, color=color),
                            text=[g['player_name'].split()[-1]],
                            textposition='top center',
                            textfont=dict(color=color, size=11),
                            showlegend=False,
                        ))
            fig_xg.add_vline(x=45, line_dash='dash', line_color='#7a8a9a', opacity=0.5)
            fig_xg.update_layout(
                plot_bgcolor='#0a1628', paper_bgcolor='#00060d',
                font=dict(color='#e8e8e8', size=13),
                xaxis=dict(title='Minute', gridcolor='#3a5a7a', color='#7a8a9a', title_font_size=13),
                yaxis=dict(title='Cumulative xG', gridcolor='#3a5a7a', color='#7a8a9a', title_font_size=13),
                legend=dict(bgcolor='#0d1f35', bordercolor='#3a5a7a', font_size=13),
                margin=dict(l=40, r=20, t=30, b=40),
                height=380,
            )
            st.plotly_chart(fig_xg, use_container_width=True)

    with col_b:
        st.markdown("### Match Momentum")
        fig_mom = render_momentum(momentum_df, m, TEAM_COLORS, figsize=(10, 4))
        st.pyplot(fig_mom, use_container_width=True)
        plt.close(fig_mom)

    st.markdown("---")

    # ── Time-series line charts ────────────────────────────────────────────────
    st.markdown("### Match Flow — Rolling Stats (5-min windows)")
    if not match.rolling_stats.empty:
        rs = match.rolling_stats
        _plotly_layout = dict(
            plot_bgcolor='#0a1628', paper_bgcolor='#00060d',
            font=dict(color='#e8e8e8', size=13),
            xaxis=dict(title='Minute', gridcolor='#3a5a7a', color='#7a8a9a',
                       title_font_size=13, showgrid=True),
            legend=dict(bgcolor='#0d1f35', bordercolor='#3a5a7a', font_size=13,
                        orientation='h', yanchor='bottom', y=1.02),
            margin=dict(l=40, r=20, t=40, b=40),
            height=280,
        )
        _vline_45 = dict(x=45, line_dash='dash', line_color='#7a8a9a', opacity=0.5)

        chart_cols = st.columns(2)

        # 1) Possession %
        with chart_cols[0]:
            fig_poss = go.Figure()
            fig_poss.add_trace(go.Scatter(
                x=rs['bucket'], y=rs['home_poss_pct'],
                mode='lines+markers', name=home_name,
                line=dict(color=home_color, width=2.5),
                fill='tozeroy', fillcolor=_hex_to_rgba(home_color, 0.15),
                marker=dict(size=6),
            ))
            fig_poss.add_trace(go.Scatter(
                x=rs['bucket'], y=rs['away_poss_pct'],
                mode='lines+markers', name=away_name,
                line=dict(color=away_color, width=2.5),
                fill='tozeroy', fillcolor=_hex_to_rgba(away_color, 0.12),
                marker=dict(size=6),
            ))
            fig_poss.add_hline(y=50, line_dash='dot', line_color='#7a8a9a', opacity=0.4)
            fig_poss.add_vline(**_vline_45)
            fig_poss.update_layout(
                **_plotly_layout,
                yaxis=dict(title='Possession %', gridcolor='#3a5a7a', color='#7a8a9a',
                           title_font_size=13, range=[0, 100]),
                title=dict(text='Possession %', font_size=14, x=0.5),
            )
            st.plotly_chart(fig_poss, use_container_width=True)

        # 2) Duel Win Rate
        with chart_cols[1]:
            fig_duel = go.Figure()
            fig_duel.add_trace(go.Scatter(
                x=rs['bucket'], y=rs['home_duel_win_pct'],
                mode='lines+markers', name=home_name,
                line=dict(color=home_color, width=2.5),
                marker=dict(size=6),
            ))
            fig_duel.add_trace(go.Scatter(
                x=rs['bucket'], y=rs['away_duel_win_pct'],
                mode='lines+markers', name=away_name,
                line=dict(color=away_color, width=2.5),
                marker=dict(size=6),
            ))
            fig_duel.add_hline(y=50, line_dash='dot', line_color='#7a8a9a', opacity=0.4)
            fig_duel.add_vline(**_vline_45)
            fig_duel.update_layout(
                **_plotly_layout,
                yaxis=dict(title='Duel Win %', gridcolor='#3a5a7a', color='#7a8a9a',
                           title_font_size=13, range=[0, 100]),
                title=dict(text='Duel Win Rate %', font_size=14, x=0.5),
            )
            st.plotly_chart(fig_duel, use_container_width=True)

        chart_cols2 = st.columns(2)

        # 3) Attacks per minute
        with chart_cols2[0]:
            fig_atk = go.Figure()
            fig_atk.add_trace(go.Scatter(
                x=rs['bucket'], y=rs['home_attacks_per_min'],
                mode='lines+markers', name=home_name,
                line=dict(color=home_color, width=2.5),
                fill='tozeroy', fillcolor=_hex_to_rgba(home_color, 0.15),
                marker=dict(size=6),
            ))
            fig_atk.add_trace(go.Scatter(
                x=rs['bucket'], y=rs['away_attacks_per_min'],
                mode='lines+markers', name=away_name,
                line=dict(color=away_color, width=2.5),
                fill='tozeroy', fillcolor=_hex_to_rgba(away_color, 0.12),
                marker=dict(size=6),
            ))
            fig_atk.add_vline(**_vline_45)
            fig_atk.update_layout(
                **_plotly_layout,
                yaxis=dict(title='Attacks / min', gridcolor='#3a5a7a', color='#7a8a9a',
                           title_font_size=13),
                title=dict(text='Attacks per Minute', font_size=14, x=0.5),
            )
            st.plotly_chart(fig_atk, use_container_width=True)

        # 4) Rolling PPDA
        with chart_cols2[1]:
            fig_ppda = go.Figure()
            fig_ppda.add_trace(go.Scatter(
                x=rs['bucket'], y=rs['home_ppda'],
                mode='lines+markers', name=f"{home_name} PPDA",
                line=dict(color=home_color, width=2.5),
                marker=dict(size=6),
            ))
            fig_ppda.add_trace(go.Scatter(
                x=rs['bucket'], y=rs['away_ppda'],
                mode='lines+markers', name=f"{away_name} PPDA",
                line=dict(color=away_color, width=2.5),
                marker=dict(size=6),
            ))
            fig_ppda.add_vline(**_vline_45)
            fig_ppda.update_layout(
                **_plotly_layout,
                yaxis=dict(title='PPDA (lower = more pressure)', gridcolor='#3a5a7a',
                           color='#7a8a9a', title_font_size=13, autorange='reversed'),
                title=dict(text='PPDA (Rolling Pressing Intensity)', font_size=14, x=0.5),
            )
            st.plotly_chart(fig_ppda, use_container_width=True)

    st.markdown("---")

    # ── Attacks per Flank ─────────────────────────────────────────────────────
    st.markdown("### Attacks per Flank")
    if not match.flank_attacks.empty:
        fig_flank = render_flank_attacks(match.flank_attacks, m, TEAM_COLORS, figsize=(16, 8))
        st.pyplot(fig_flank, use_container_width=True)
        plt.close(fig_flank)
    else:
        st.info("No flank attack data available in this match file.")

    st.markdown("---")

    # ── Full key stats ─────────────────────────────────────────────────────────
    st.markdown("### Full Match Statistics")
    fig_stats = render_key_stats(match.stats, m, TEAM_COLORS, match.ppda, figsize=(14, 10))
    st.pyplot(fig_stats, use_container_width=True)
    plt.close(fig_stats)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ATTACKING & POSSESSION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    # ── Shot Map ──────────────────────────────────────────────────────────────
    fig_shots = render_shot_map(match.events_df, m, TEAM_COLORS, figsize=(16, 10))
    st.pyplot(fig_shots, use_container_width=True)
    plt.close(fig_shots)

    st.markdown("---")

    # ── Shot Breakdown ────────────────────────────────────────────────────────
    st.markdown("### Shot Breakdown")
    shots_df_all = match.events_df[match.events_df['type_primary'] == 'shot'].copy()
    if not shots_df_all.empty:
        col_sh1, col_sh2 = st.columns(2)
        for col_sh, team_id_, tname in [
            (col_sh1, home_id, home_name),
            (col_sh2, away_id, away_name),
        ]:
            with col_sh:
                t_shots = shots_df_all[shots_df_all['team_id'] == team_id_].copy()
                t_shots = t_shots.sort_values(['match_minute', 'second']).reset_index(drop=True)
                t_shots['#'] = t_shots.index + 1
                tbl = t_shots[['#', 'match_minute', 'player_name', 'shot_body_part',
                               'shot_goal_zone', 'shot_on_target', 'shot_is_goal',
                               'shot_xg', 'shot_post_xg']].copy()
                tbl.columns = ['#', 'Min', 'Player', 'Body Part', 'Zone',
                               'On Target', 'Goal', 'xG', 'PSxG']
                tbl['Zone'] = tbl['Zone'].map(lambda z: ZONE_NAMES.get(z, z) if pd.notna(z) else z)
                tbl['xG'] = tbl['xG'].round(3)
                tbl['PSxG'] = tbl['PSxG'].round(3)
                color = home_color if team_id_ == home_id else away_color
                st.markdown(f"<span style='color:{color};font-weight:700'>{tname}</span>",
                            unsafe_allow_html=True)
                st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── SCA by Player + xG by Player ──────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Shot-Creating Actions")
        if not match.sca_df.empty:
            fig_sca = render_sca_by_player(match.sca_df, m, TEAM_COLORS, figsize=(12, 6))
            st.pyplot(fig_sca, use_container_width=True)
            plt.close(fig_sca)
    with col_b:
        st.markdown("### xG by Player")
        fig_xgp = render_xg_by_player(match.player_stats, m, TEAM_COLORS, figsize=(12, 6))
        st.pyplot(fig_xgp, use_container_width=True)
        plt.close(fig_xgp)

    st.markdown("---")
    st.markdown("## Possession & Passing")

    # ── Pass Network with per-team substitution interval filters ──────────────
    st.markdown("### Pass Network")

    def _build_intervals(subs_df_team):
        """Build (label, start, end) list from a team's sub minutes."""
        team_mins = sorted(set(subs_df_team['match_minute'].tolist())) if not subs_df_team.empty else []
        bps = sorted(set([0] + team_mins + [90]))
        opts = []
        for i in range(len(bps) - 1):
            s, e = bps[i], bps[i + 1]
            lbl = f"{s}' – {e}'" + ("  (Kickoff)" if s == 0 else "")
            opts.append((lbl, s, e))
        return opts

    home_subs = subs_df[subs_df['team_id'] == home_id] if not subs_df.empty else pd.DataFrame()
    away_subs = subs_df[subs_df['team_id'] == away_id] if not subs_df.empty else pd.DataFrame()
    home_intervals = _build_intervals(home_subs)
    away_intervals = _build_intervals(away_subs)

    col_hpn, col_apn, col_mpn = st.columns([2, 2, 1])
    with col_hpn:
        h_idx = st.selectbox(
            f"🏠 {home_name} lineup",
            options=range(len(home_intervals)),
            format_func=lambda i: home_intervals[i][0],
            index=0, key='pn_home_interval',
        )
    with col_apn:
        a_idx = st.selectbox(
            f"✈️ {away_name} lineup",
            options=range(len(away_intervals)),
            format_func=lambda i: away_intervals[i][0],
            index=0, key='pn_away_interval',
        )
    with col_mpn:
        min_passes = st.slider("Min passes", 2, 8, 3, key='pn_min_passes')

    _, h_start, h_end = home_intervals[h_idx]
    _, a_start, a_end = away_intervals[a_idx]

    fig_pn = render_pass_network(
        match.events_df, m, TEAM_COLORS,
        player_lookup=match.player_lookup,
        min_passes=min_passes,
        home_min_start=h_start, home_min_end=h_end,
        away_min_start=a_start, away_min_end=a_end,
    )
    st.pyplot(fig_pn, use_container_width=True)
    plt.close(fig_pn)

    st.markdown("---")
    st.markdown("### Offensive Actions Map")

    _ACTION_OPTIONS = {
        'Passes': 'passes',
        'Key Passes': 'key_passes',
        'Progressive Passes': 'progressive_passes',
        'Crosses': 'crosses',
        'Carries': 'carries',
        'Progressive Carries': 'progressive_carries',
        'Dribbles': 'dribbles',
        'Corners': 'corners',
        'Free Kicks': 'free_kicks',
        'Throw Ins': 'throw_ins',
    }
    _ZONE_OPTIONS = {
        'Full Pitch': 'full',
        'Final Third': 'final_third',
        'Penalty Box': 'penalty_box',
        'Own Half': 'own_half',
        'Middle Third': 'middle_third',
    }

    col_act, col_zone, col_team, col_time = st.columns([3, 2, 2, 2])
    with col_act:
        selected_actions = st.multiselect(
            "Action types",
            options=list(_ACTION_OPTIONS.keys()),
            default=['Passes'],
            key='oa_actions',
        )
    with col_zone:
        selected_zone = st.selectbox(
            "Pitch zone",
            options=list(_ZONE_OPTIONS.keys()),
            index=0,
            key='oa_zone',
        )
    with col_team:
        selected_team = st.selectbox(
            "Team",
            options=[f'Both Teams', home_name, away_name],
            index=0,
            key='oa_team',
        )
    with col_time:
        # Build time bins: Full Match, 1H, 2H, and sub intervals
        _all_sub_mins = sorted(set(subs_df['match_minute'].tolist())) if not subs_df.empty else []
        _time_options = {
            'Full Match (0–90)': (0, 90),
            '1st Half (0–45)': (0, 45),
            '2nd Half (45–90)': (45, 90),
        }
        _all_bps = sorted(set([0] + _all_sub_mins + [90]))
        for _bi in range(len(_all_bps) - 1):
            _bs, _be = _all_bps[_bi], _all_bps[_bi + 1]
            _time_options[f"{_bs}'–{_be}'"] = (_bs, _be)
        selected_time = st.selectbox(
            "Time range",
            options=list(_time_options.keys()),
            index=0,
            key='oa_time',
        )

    _act_codes = [_ACTION_OPTIONS[a] for a in selected_actions] if selected_actions else ['passes']
    _zone_code = _ZONE_OPTIONS[selected_zone]
    _team_code = 'both' if selected_team == 'Both Teams' else ('home' if selected_team == home_name else 'away')
    _t_min, _t_max = _time_options[selected_time]

    fig_oa = render_offensive_actions(
        match.events_df, m, TEAM_COLORS,
        action_types=_act_codes,
        team_filter=_team_code,
        pitch_zone=_zone_code,
        min_minute=_t_min, max_minute=_t_max,
    )
    st.pyplot(fig_oa, use_container_width=True)
    plt.close(fig_oa)

    st.markdown("---")
    st.markdown("### Expected Threat (xT) by Player — Top Contributors")
    if not match.xt_by_player.empty:
        xt_display = match.xt_by_player.copy()
        xt_display = xt_display[xt_display['player_id'].notna()].copy()
        xt_display['team_name_col'] = xt_display['team_id'].map({
            home_id: home_name, away_id: away_name
        })
        xt_display = xt_display.sort_values('xT', ascending=False).head(16)
        fig_xt = px.bar(
            xt_display, x='xT', y='player_name', orientation='h',
            color='team_name_col',
            color_discrete_map={home_name: home_color, away_name: away_color},
            title='xT Generated (Passes + Carries)',
            labels={'xT': 'Expected Threat', 'player_name': '', 'team_name_col': 'Team'},
        )
        fig_xt.update_layout(
            plot_bgcolor='#0a1628', paper_bgcolor='#00060d',
            font_color='#e8e8e8', font=dict(size=13),
            yaxis=dict(autorange='reversed', gridcolor='#3a5a7a', color='#7a8a9a'),
            xaxis=dict(gridcolor='#3a5a7a', color='#7a8a9a', zeroline=True,
                       zerolinecolor='#3a5a7a'),
            legend=dict(bgcolor='#0d1f35', bordercolor='#3a5a7a'),
            height=450,
        )
        st.plotly_chart(fig_xt, use_container_width=True)

    st.markdown("---")
    st.markdown("### Set Pieces Summary")
    sp_cols = st.columns(2)
    for idx, (team_id, color) in enumerate([(home_id, home_color), (away_id, away_color)]):
        tname = home_name if team_id == home_id else away_name
        corners_n = len(match.events_df[
            (match.events_df['type_primary'] == 'corner') &
            (match.events_df['team_id'] == team_id)])
        fks_n = len(match.events_df[
            (match.events_df['type_primary'] == 'free_kick') &
            (match.events_df['team_id'] == team_id)])
        sp_shots_n = match.events_df[
            (match.events_df['type_primary'] == 'shot') &
            (match.events_df['team_id'] == team_id) &
            (match.events_df['shot_situation'].isin(['Corner', 'Free Kick', 'Penalty'])
             if 'shot_situation' in match.events_df.columns else
             match.events_df.get('is_set_piece', False))
        ]
        sp_xg = sp_shots_n['shot_xg'].sum() if not sp_shots_n.empty else 0.0
        with sp_cols[idx]:
            st.markdown(
                f'<div class="metric-card"><b style="color:{color}">{tname}</b><br>'
                f'Corners: <b>{corners_n}</b> &nbsp;|&nbsp; Free Kicks: <b>{fks_n}</b>'
                f' &nbsp;|&nbsp; SP Shots: <b>{len(sp_shots_n)}</b>'
                f' &nbsp;|&nbsp; SP xG: <b>{sp_xg:.2f}</b></div>',
                unsafe_allow_html=True
            )



# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEFENSIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"### {home_name}")
        st.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color:{home_color}">'
            f'{match.ppda.get(home_id, 0):.2f}</div><div class="metric-label">PPDA</div></div>',
            unsafe_allow_html=True
        )
        hs = match.stats[home_id]
        c1, c2, c3 = st.columns(3)
        c1.metric("Interceptions", hs['interceptions'])
        c2.metric("Clearances", hs['clearances'])
        c3.metric("Aerial Won", f"{hs['aerial_duels_won']}/{hs['aerial_duels_total']}")
    with col_b:
        st.markdown(f"### {away_name}")
        st.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color:{away_color}">'
            f'{match.ppda.get(away_id, 0):.2f}</div><div class="metric-label">PPDA</div></div>',
            unsafe_allow_html=True
        )
        aws = match.stats[away_id]
        c1, c2, c3 = st.columns(3)
        c1.metric("Interceptions", aws['interceptions'])
        c2.metric("Clearances", aws['clearances'])
        c3.metric("Aerial Won", f"{aws['aerial_duels_won']}/{aws['aerial_duels_total']}")

    st.markdown("---")
    st.markdown("### Defensive Actions Map")
    fig_def = render_defensive_actions(match.events_df, m, TEAM_COLORS)
    st.pyplot(fig_def, use_container_width=True)
    plt.close(fig_def)

    st.markdown("---")
    st.markdown("### Duel Zones")
    fig_duels_map = render_duel_map(match.events_df, m, TEAM_COLORS)
    st.pyplot(fig_duels_map, use_container_width=True)
    plt.close(fig_duels_map)

    st.markdown("---")
    st.markdown("### Transitions — Ball Recoveries & Losses")
    trans_mode_label = st.radio(
        "Transition type",
        ["Both", "Offensive (Recoveries)", "Defensive (Turnovers)"],
        horizontal=True,
        key="trans_mode",
    )
    _mode_map = {
        "Both": "both",
        "Offensive (Recoveries)": "offensive",
        "Defensive (Turnovers)": "defensive",
    }
    fig_trans = render_transitions_map(
        match.events_df, m, TEAM_COLORS, mode=_mode_map[trans_mode_label]
    )
    st.pyplot(fig_trans, use_container_width=True)
    plt.close(fig_trans)

    trans_tbl = compute_transitions_table(match.events_df, m)
    if not trans_tbl.empty:
        if _mode_map[trans_mode_label] == "offensive":
            trans_tbl = trans_tbl[trans_tbl["Type"] == "Recovery"]
        elif _mode_map[trans_mode_label] == "defensive":
            trans_tbl = trans_tbl[trans_tbl["Type"] == "Turnover"]
        st.dataframe(trans_tbl, use_container_width=True, hide_index=True)

    st.markdown("---")
    # Duels breakdown (Plotly)
    st.markdown("### Duels Breakdown by Player")
    duels_data = match.player_stats[match.player_stats['duels'] > 0].copy()
    duels_data['team_name_col'] = duels_data['team_id'].map({
        home_id: home_name, away_id: away_name
    })
    if not duels_data.empty:
        fig_duels = px.bar(
            duels_data.sort_values('duels', ascending=False).head(20),
            x='player_name', y=['duels_won', 'ground_duels', 'aerial_duels'],
            color_discrete_sequence=[away_color, _hex_to_rgba(home_color, 0.7), _hex_to_rgba(home_color, 0.4)],
            barmode='group',
            title='Duels by Player',
            labels={'value': 'Count', 'player_name': '', 'variable': 'Type'},
        )
        fig_duels.update_layout(
            plot_bgcolor='#0a1628', paper_bgcolor='#010b14',
            font_color='#e8e8e8',
            xaxis=dict(gridcolor='#3a5a7a', color='#7a8a9a', tickangle=-30),
            yaxis=dict(gridcolor='#3a5a7a', color='#7a8a9a'),
            legend=dict(bgcolor='#0d1f35', bordercolor='#3a5a7a'),
            height=400,
        )
        st.plotly_chart(fig_duels, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PLAYERS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    # ── Player selector ───────────────────────────────────────────────────────
    all_players = match.player_stats.copy()
    all_players['display_name'] = (
        all_players['player_name'] + ' (' +
        all_players['team_id'].map({home_id: home_name, away_id: away_name}).fillna('') +
        ')'
    )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        home_options = all_players[all_players['team_id'] == home_id]['display_name'].tolist()
        player1_sel = st.selectbox(f"🟢 {home_name} Player", ['— Select —'] + home_options)
    with col_p2:
        away_options = all_players[all_players['team_id'] == away_id]['display_name'].tolist()
        player2_sel = st.selectbox(f"🟡 {away_name} Player", ['— Select —'] + away_options)

    # Action filter
    st.markdown("**Action types to display:**")
    acols = st.columns(5)
    show_passes  = acols[0].checkbox("Passes",    value=True,  key='pam_passes')
    show_shots   = acols[1].checkbox("Shots",     value=True,  key='pam_shots')
    show_carries = acols[2].checkbox("Carries",   value=True,  key='pam_carries')
    show_duels   = acols[3].checkbox("Duels",     value=False, key='pam_duels')
    show_def     = acols[4].checkbox("Defensive", value=False, key='pam_def')

    action_filter = []
    if show_passes:  action_filter.append('passes')
    if show_shots:   action_filter.append('shots')
    if show_carries: action_filter.append('carries')
    if show_duels:   action_filter.append('duels')

    if show_def:
        st.markdown("**Defensive action types:**")
        dcols = st.columns(4)
        show_int = dcols[0].checkbox("Interceptions",       value=True,  key='def_int')
        show_clr = dcols[1].checkbox("Clearances",          value=True,  key='def_clr')
        show_chl = dcols[2].checkbox("Challenges/Tackles",  value=True,  key='def_chl')
        show_aer = dcols[3].checkbox("Aerial Duels",        value=False, key='def_aer')
        if show_int: action_filter.append('interceptions')
        if show_clr: action_filter.append('clearances')
        if show_chl: action_filter.append('challenges')
        if show_aer: action_filter.append('aerial_duels')
        if not any([show_int, show_clr, show_chl, show_aer]):
            action_filter.append('defensive')

    if not action_filter:
        action_filter = ['passes']

    def _get_player_row(display_name: str):
        rows = all_players[all_players['display_name'] == display_name]
        return rows.iloc[0] if len(rows) else None

    def _render_player_section(prow, team_color, squad_df):
        pid = int(prow['player_id'])
        pname = prow['player_name']

        # Stats metrics row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Touches", int(prow.get('touches', 0)))
        c2.metric("Passes", f"{int(prow.get('passes', 0))} ({prow.get('pass_accuracy', 0):.0f}%)")
        c3.metric("Shots / Goals", f"{int(prow.get('shots', 0))} / {int(prow.get('goals', 0))}")
        c4.metric("xG", f"{prow.get('xg', 0):.3f}")
        c5.metric("xT", f"{prow.get('xT', 0):.4f}")
        c6.metric("SCA", int(prow.get('sca', 0)))

        col_map, col_radar = st.columns([3, 2])
        with col_map:
            fig_am = render_player_action_map(
                match.events_df, pid, pname, team_color,
                player_stats_row=prow,
                action_filter=action_filter,
                figsize=(12, 8)
            )
            st.pyplot(fig_am, use_container_width=True)
            plt.close(fig_am)
        with col_radar:
            if len(squad_df) > 2:
                fig_radar = render_player_radar(prow, squad_df, pname, team_color, figsize=(7, 7))
                st.pyplot(fig_radar, use_container_width=True)
                plt.close(fig_radar)
            else:
                st.info("Need more squad players for radar chart")

    # Render selected players
    p1_selected = player1_sel != '— Select —'
    p2_selected = player2_sel != '— Select —'

    if p1_selected and p2_selected:
        # Side-by-side comparison
        st.markdown("### Player Comparison")
        p1row = _get_player_row(player1_sel)
        p2row = _get_player_row(player2_sel)

        if p1row is not None and p2row is not None:
            # Comparison stats table
            metrics_to_compare = ['touches', 'passes', 'pass_accuracy', 'shots', 'goals',
                                   'xg', 'xT', 'sca', 'key_passes', 'duels', 'duel_win_pct',
                                   'interceptions']
            labels_cmp = ['Touches', 'Passes', 'Pass %', 'Shots', 'Goals',
                          'xG', 'xT', 'SCA', 'Key Passes', 'Duels', 'Duel Win%', 'INT']
            comp_data = {
                'Metric': labels_cmp,
                p1row['player_name']: [round(float(p1row.get(m, 0) or 0), 2) for m in metrics_to_compare],
                p2row['player_name']: [round(float(p2row.get(m, 0) or 0), 2) for m in metrics_to_compare],
            }
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Side-by-side action maps
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown(f"#### {p1row['player_name']}")
                fig_p1 = render_player_action_map(
                    match.events_df, int(p1row['player_id']),
                    p1row['player_name'], home_color,
                    player_stats_row=p1row, action_filter=action_filter,
                    figsize=(10, 7),
                )
                st.pyplot(fig_p1, use_container_width=True)
                plt.close(fig_p1)
            with col_right:
                st.markdown(f"#### {p2row['player_name']}")
                fig_p2 = render_player_action_map(
                    match.events_df, int(p2row['player_id']),
                    p2row['player_name'], away_color,
                    player_stats_row=p2row, action_filter=action_filter,
                    figsize=(10, 7),
                )
                st.pyplot(fig_p2, use_container_width=True)
                plt.close(fig_p2)

    elif p1_selected:
        p1row = _get_player_row(player1_sel)
        if p1row is not None:
            st.markdown(f"### {p1row['player_name']}")
            squad_df = all_players[all_players['team_id'] == home_id]
            _render_player_section(p1row, home_color, squad_df)

    elif p2_selected:
        p2row = _get_player_row(player2_sel)
        if p2row is not None:
            st.markdown(f"### {p2row['player_name']}")
            squad_df = all_players[all_players['team_id'] == away_id]
            _render_player_section(p2row, away_color, squad_df)
    else:
        st.info("Select one or two players to see their individual analysis")

    st.markdown("---")
    st.markdown("### Advanced Statistics — All Players")

    # Build combined table from player_stats (already has xT and sca merged)
    adv = match.player_stats.copy()
    adv['Team'] = adv['team_id'].map({home_id: home_name, away_id: away_name})

    # Merge xT pass/carry split if available
    if not match.xt_by_player.empty:
        xt_cols = ['player_id', 'xT']
        for c in ['xT_pass', 'xT_carry']:
            if c in match.xt_by_player.columns:
                xt_cols.append(c)
        adv = adv.merge(
            match.xt_by_player[xt_cols].rename(columns={'xT': 'xT_total'}),
            on='player_id', how='left'
        )
        if 'xT_total' in adv.columns:
            adv['xT'] = adv['xT_total'].fillna(adv.get('xT', 0))
            adv.drop(columns=['xT_total'], inplace=True)

    # Merge SCA if not already present
    if 'sca' not in adv.columns and not match.sca_df.empty:
        adv = adv.merge(
            match.sca_df[['player_id', 'sca']], on='player_id', how='left'
        )

    # Column selection and display order
    display_cols = {
        'player_name': 'Player',
        'Team': 'Team',
        'touches': 'Touches',
        'passes': 'Passes',
        'pass_accuracy': 'Pass%',
        'key_passes': 'Key Pass',
        'shots': 'Shots',
        'goals': 'Goals',
        'xg': 'xG',
        'xT': 'xT',
        'sca': 'SCA',
        'duels': 'Duels',
        'duel_win_pct': 'Duel%',
        'interceptions': 'INT',
        'clearances': 'Clr',
        'aerial_duels_won': 'Aerial Won',
        'aerial_duels_total': 'Aerial Tot',
    }
    avail = [c for c in display_cols if c in adv.columns]
    tbl = adv[avail].copy()
    tbl = tbl.rename(columns={c: display_cols[c] for c in avail})

    # Sort: home first, then away; within team by touches desc
    tbl.insert(0, '_sort_team', adv['team_id'].map({home_id: 0, away_id: 1}))
    tbl = tbl.sort_values(['_sort_team', 'Touches'], ascending=[True, False])
    tbl.drop(columns=['_sort_team'], inplace=True)

    # Round floats
    for col in ['Pass%', 'xG', 'xT', 'Duel%']:
        if col in tbl.columns:
            tbl[col] = tbl[col].round(2)

    st.dataframe(tbl, use_container_width=True, hide_index=True)
