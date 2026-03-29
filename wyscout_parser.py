"""
wyscout_parser.py
=================
Data processing module for Wyscout event JSON format.
Parses match data, builds DataFrames, and computes derived metrics
(PPDA, SCA, xT) for post-match analysis.

Usage:
    from wyscout_parser import load_match, parse_all
    data = load_match("5758824.json")
    match = parse_all(data)
    # match.meta, match.events_df, match.player_stats, etc.
"""

import json
import io
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MatchData:
    meta: dict
    player_lookup: dict
    team_lookup: dict
    events_df: pd.DataFrame
    substitutions_raw: dict
    goals: pd.DataFrame = field(default_factory=pd.DataFrame)
    cards: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: dict = field(default_factory=dict)
    xg_timeline: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    sca_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    xt_by_player: pd.DataFrame = field(default_factory=pd.DataFrame)
    ppda: dict = field(default_factory=dict)
    flank_attacks: pd.DataFrame = field(default_factory=pd.DataFrame)
    rolling_stats: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_match(path_or_bytes) -> dict:
    """Accept file path (str/Path) or bytes/BytesIO from st.file_uploader."""
    if isinstance(path_or_bytes, (str,)) or hasattr(path_or_bytes, '__fspath__'):
        with open(path_or_bytes, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif isinstance(path_or_bytes, bytes):
        return json.loads(path_or_bytes.decode('utf-8'))
    else:
        # BytesIO or UploadedFile
        return json.loads(path_or_bytes.read())


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def extract_match_meta(data: dict) -> dict:
    """Extract flat match metadata."""
    m = data['match']
    teams_data = m['teamsData']

    home_id = None
    away_id = None
    for tid, td in teams_data.items():
        if td['side'] == 'home':
            home_id = int(tid)
        else:
            away_id = int(tid)

    home_td = teams_data[str(home_id)]
    away_td = teams_data[str(away_id)]

    # Team names from events (teams dict has null names in Wyscout export)
    home_name, away_name = _extract_team_names(data['events'], home_id, away_id)

    # Formations
    home_form_1h = _get_formation(data, str(home_id), '1H')
    away_form_1h = _get_formation(data, str(away_id), '1H')
    home_form_2h = _get_formation(data, str(home_id), '2H')
    away_form_2h = _get_formation(data, str(away_id), '2H')

    # Coaches
    home_coach = _get_coach_name(data, home_id)
    away_coach = _get_coach_name(data, away_id)

    # Referee
    referees = m.get('referees', [])
    main_ref_id = None
    for r in referees:
        if r.get('role') == 'referee':
            main_ref_id = r.get('refereeId')
            break

    return {
        'match_id': m['wyId'],
        'label': m['label'],
        'date_utc': m['dateutc'],
        'date_str': m['dateutc'][:10],
        'competition_id': m['competitionId'],
        'season_id': m['seasonId'],
        'gameweek': m.get('gameweek'),
        'venue': m.get('venue'),
        'duration': m.get('duration', 90),
        'home_id': home_id,
        'away_id': away_id,
        'home_name': home_name,
        'away_name': away_name,
        'home_score': home_td['score'],
        'away_score': away_td['score'],
        'home_score_ht': home_td['scoreHT'],
        'away_score_ht': away_td['scoreHT'],
        'home_formation_1h': home_form_1h,
        'away_formation_1h': away_form_1h,
        'home_formation_2h': home_form_2h,
        'away_formation_2h': away_form_2h,
        'home_coach': home_coach,
        'away_coach': away_coach,
        'referee_id': main_ref_id,
    }


def _extract_team_names(events: list, home_id: int, away_id: int) -> Tuple[str, str]:
    home_name = away_name = None
    for e in events:
        if e is None:
            continue
        team = e.get('team', {}) or {}
        tid = team.get('id')
        tname = team.get('name')
        if tname and tid == home_id and home_name is None:
            home_name = tname
        if tname and tid == away_id and away_name is None:
            away_name = tname
        if home_name and away_name:
            break
    return home_name or f'Team {home_id}', away_name or f'Team {away_id}'


def _get_formation(data: dict, team_id_str: str, period: str) -> Optional[str]:
    formations = data.get('formations', {})
    if team_id_str not in formations:
        return None
    team_forms = formations[team_id_str]
    if period not in team_forms:
        return None
    period_forms = team_forms[period]
    # Get the first time entry (kickoff formation)
    for sec_str in sorted(period_forms.keys(), key=lambda x: int(x)):
        for scheme in period_forms[sec_str]:
            return scheme
    return None


def _get_coach_name(data: dict, team_id: int) -> Optional[str]:
    coaches = data.get('coaches', {})
    entry = coaches.get(str(team_id))
    if entry and isinstance(entry, dict):
        coach = entry.get('coach', {})
        return coach.get('shortName') or coach.get('lastName')
    return None


# ---------------------------------------------------------------------------
# Player & Team lookups
# ---------------------------------------------------------------------------

def build_player_lookup(data: dict) -> Dict[int, dict]:
    """Returns {player_wyId: player_info_dict}."""
    lookup = {}
    players_data = data.get('players', {})
    for team_id_str, entries in players_data.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            p = entry.get('player', {})
            if not p:
                continue
            wid = p.get('wyId')
            if not wid:
                continue
            role = p.get('role', {}) or {}
            lookup[int(wid)] = {
                'wyId': int(wid),
                'shortName': p.get('shortName', ''),
                'firstName': p.get('firstName', ''),
                'lastName': p.get('lastName', ''),
                'height': p.get('height'),
                'weight': p.get('weight'),
                'foot': p.get('foot'),
                'role_name': role.get('name', ''),
                'role_code2': role.get('code2', ''),
                'role_code3': role.get('code3', ''),
                'team_id': int(team_id_str),
                'imageDataURL': p.get('imageDataURL', ''),
                'birthDate': p.get('birthDate'),
                'nationality': (p.get('passportArea') or {}).get('name', ''),
            }
    # Enrich with shirt numbers from formation
    teams_data = data.get('match', {}).get('teamsData', {})
    for team_id_str, td in teams_data.items():
        formation = td.get('formation', {})
        for group in ['lineup', 'bench']:
            for p in formation.get(group, []):
                pid = p.get('playerId')
                shirt = p.get('shirtNumber')
                if pid and shirt is not None and int(pid) in lookup:
                    lookup[int(pid)]['shirtNumber'] = int(shirt)
    return lookup


def build_team_lookup(data: dict, meta: dict) -> Dict[int, dict]:
    """Returns {team_id: team_info_dict}."""
    return {
        meta['home_id']: {
            'id': meta['home_id'],
            'name': meta['home_name'],
            'side': 'home',
            'score': meta['home_score'],
            'score_ht': meta['home_score_ht'],
            'formation_1h': meta['home_formation_1h'],
            'formation_2h': meta['home_formation_2h'],
            'coach': meta['home_coach'],
        },
        meta['away_id']: {
            'id': meta['away_id'],
            'name': meta['away_name'],
            'side': 'away',
            'score': meta['away_score'],
            'score_ht': meta['away_score_ht'],
            'formation_1h': meta['away_formation_1h'],
            'formation_2h': meta['away_formation_2h'],
            'coach': meta['away_coach'],
        },
    }


# ---------------------------------------------------------------------------
# Formations / Starting XI
# ---------------------------------------------------------------------------

def get_starting_xi(data: dict, team_id: int, period: str = '1H') -> List[dict]:
    """Returns list of {player_id, position} for the starting formation."""
    formations = data.get('formations', {})
    tid_str = str(team_id)
    if tid_str not in formations:
        return []
    team_forms = formations[tid_str]
    if period not in team_forms:
        return []
    period_forms = team_forms[period]
    for sec_str in sorted(period_forms.keys(), key=lambda x: int(x)):
        for scheme, form_data in period_forms[sec_str].items():
            players_raw = form_data.get('players', [])
            result = []
            for item in players_raw:
                if isinstance(item, dict):
                    for pid_str, pdata in item.items():
                        result.append({
                            'player_id': int(pdata.get('playerId', pid_str)),
                            'position': pdata.get('position', ''),
                        })
            return result
    return []


# ---------------------------------------------------------------------------
# Events DataFrame
# ---------------------------------------------------------------------------

def build_events_df(data: dict, player_lookup: dict, team_lookup: dict) -> pd.DataFrame:
    """Flatten all events into a DataFrame with normalized columns."""
    rows = []
    events = data.get('events', [])
    home_id = [tid for tid, tinfo in team_lookup.items() if tinfo['side'] == 'home'][0]
    away_id = [tid for tid, tinfo in team_lookup.items() if tinfo['side'] == 'away'][0]

    for e in events:
        if e is None:
            continue

        type_info = e.get('type', {}) or {}
        type_primary = type_info.get('primary', '')
        type_secondary = type_info.get('secondary') or []

        # Skip shot_against (GK duplicate of shot events)
        if type_primary == 'shot_against':
            continue

        loc = e.get('location', {}) or {}
        x = loc.get('x')
        y = loc.get('y')

        team = e.get('team', {}) or {}
        team_id = team.get('id')
        team_name = team.get('name', '')
        team_side = team_lookup.get(team_id, {}).get('side', '') if team_id else ''

        # Mirror x for away team so both teams always attack left→right
        x_plot = (100 - x) if (team_side == 'away' and x is not None) else x
        y_plot = (100 - y) if (team_side == 'away' and y is not None) else y

        player = e.get('player', {}) or {}
        player_id = player.get('id')
        player_name = player.get('name', '')

        # Enrich from player lookup
        p_info = player_lookup.get(int(player_id), {}) if player_id else {}
        player_role = p_info.get('role_code2', player.get('position', ''))

        period = e.get('matchPeriod', '')
        minute = e.get('minute', 0) or 0
        second = e.get('second', 0) or 0
        # match_minute: Wyscout minute is already absolute game time (0-90+), no offset needed
        match_minute = minute

        # Possession
        poss = e.get('possession', {}) or {}
        poss_id = poss.get('id')
        poss_types = poss.get('types') or []
        poss_attack = poss.get('attack') or {}
        poss_with_shot = poss_attack.get('withShot', False)
        poss_with_goal = poss_attack.get('withGoal', False)
        poss_xg = poss_attack.get('xg', 0.0)
        poss_flank = poss_attack.get('flank', None)  # 'left', 'center', 'right'

        # Shot fields
        shot_data = e.get('shot', {}) or {}
        shot_is_goal = shot_data.get('isGoal', False) if shot_data else False
        shot_on_target = shot_data.get('onTarget', False) if shot_data else False
        shot_xg = shot_data.get('xg') if shot_data else None
        shot_post_xg = shot_data.get('postShotXg') if shot_data else None
        shot_body_part = shot_data.get('bodyPart') if shot_data else None
        shot_goal_zone = shot_data.get('goalZone') if shot_data else None
        shot_gk_id = (shot_data.get('goalkeeper') or {}).get('id') if shot_data else None
        # Derive situation from possession types (computed below but poss_types already set)
        _pt = poss_types or []
        if 'corner' in _pt:
            shot_situation = 'Corner'
        elif 'free_kick' in _pt:
            shot_situation = 'Free Kick'
        elif 'penalty' in _pt:
            shot_situation = 'Penalty'
        else:
            shot_situation = 'Open Play'

        # Pass fields
        pass_data = e.get('pass', {}) or {}
        pass_accurate = pass_data.get('accurate') if pass_data else None
        pass_length = pass_data.get('length') if pass_data else None
        pass_angle = pass_data.get('angle') if pass_data else None
        pass_end = (pass_data.get('endLocation') or {}) if pass_data else {}
        pass_end_x = pass_end.get('x')
        pass_end_y = pass_end.get('y')
        recipient = (pass_data.get('recipient') or {}) if pass_data else {}
        pass_recipient_id = recipient.get('id')
        pass_recipient_name = recipient.get('name', '')

        # Mirror pass end for away team
        if team_side == 'away':
            if pass_end_x is not None:
                pass_end_x = 100 - pass_end_x
            if pass_end_y is not None:
                pass_end_y = 100 - pass_end_y

        # Duel fields
        gd = e.get('groundDuel', {}) or {}
        ad = e.get('aerialDuel', {}) or {}
        duel_type = None
        duel_won = None
        if 'aerial_duel' in type_secondary:
            duel_type = 'aerial'
            duel_won = ad.get('firstTouch', False)
        elif any(d in type_secondary for d in ('ground_duel', 'defensive_duel', 'offensive_duel', 'loose_ball_duel')):
            duel_type = 'ground'
            # Won if recovered possession, progressed, or stopped opponent
            duel_won = bool(
                gd.get('recoveredPossession') or
                gd.get('progressedWithBall') or
                gd.get('stoppedProgress')
            )
            # If 'loss' in secondary, override to lost
            if 'loss' in type_secondary:
                duel_won = False

        # Duel subtype
        duel_subtype = None
        for dt in ('defensive_duel', 'offensive_duel', 'loose_ball_duel'):
            if dt in type_secondary:
                duel_subtype = dt
                break

        # Carry fields
        carry_data = e.get('carry', {}) or {}
        carry_end = (carry_data.get('endLocation') or {}) if carry_data else {}
        carry_end_x = carry_end.get('x')
        carry_end_y = carry_end.get('y')
        carry_progression = carry_data.get('progression') if carry_data else None

        if team_side == 'away':
            if carry_end_x is not None:
                carry_end_x = 100 - carry_end_x
            if carry_end_y is not None:
                carry_end_y = 100 - carry_end_y

        # Infraction / card fields
        inf = e.get('infraction', {}) or {}
        yellow_card = inf.get('yellowCard', False)
        red_card = inf.get('redCard', False)
        is_foul = type_primary in ('infraction',)

        rows.append({
            'event_id': e.get('id'),
            'match_id': e.get('matchId'),
            'period': period,
            'minute': minute,
            'second': second,
            'match_minute': match_minute,
            'type_primary': type_primary,
            'type_secondary': type_secondary,
            'x': x,
            'y': y,
            'x_plot': x_plot,
            'y_plot': y_plot,
            'team_id': team_id,
            'team_name': team_name,
            'team_side': team_side,
            'player_id': int(player_id) if player_id else None,
            'player_name': player_name,
            'player_role': player_role,
            'possession_id': poss_id,
            'possession_types': poss_types,
            'possession_with_shot': poss_with_shot,
            'possession_with_goal': poss_with_goal,
            'possession_xg': poss_xg,
            'possession_flank': poss_flank,
            # Shot
            'shot_is_goal': shot_is_goal,
            'shot_on_target': shot_on_target,
            'shot_xg': shot_xg,
            'shot_post_xg': shot_post_xg,
            'shot_body_part': shot_body_part,
            'shot_goal_zone': shot_goal_zone,
            'shot_gk_id': shot_gk_id,
            'shot_situation': shot_situation,
            # Pass
            'pass_accurate': pass_accurate,
            'pass_length': pass_length,
            'pass_angle': pass_angle,
            'pass_end_x': pass_end_x,
            'pass_end_y': pass_end_y,
            'pass_recipient_id': pass_recipient_id,
            'pass_recipient_name': pass_recipient_name,
            # Duel
            'duel_type': duel_type,
            'duel_subtype': duel_subtype,
            'duel_won': duel_won,
            # Carry
            'carry_end_x': carry_end_x,
            'carry_end_y': carry_end_y,
            'carry_progression': carry_progression,
            # Derived booleans
            'is_progressive_pass': 'progressive_pass' in type_secondary,
            'is_key_pass': 'key_pass' in type_secondary,
            'is_cross': 'cross' in type_secondary,
            'is_long_pass': 'long_pass' in type_secondary,
            'is_carry': 'carry' in type_secondary,
            'is_progressive_carry': (
                'carry' in type_secondary and carry_progression is not None and carry_progression > 0
            ),
            # Cards
            'yellow_card': yellow_card,
            'red_card': red_card,
            'is_foul': is_foul,
            # Set piece
            'is_set_piece': any(sp in poss_types for sp in ('set_piece_attack', 'corner', 'free_kick_attack')),
            'is_corner': 'corner' in poss_types,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ensure numeric types
    for col in ['x', 'y', 'x_plot', 'y_plot', 'match_minute', 'minute', 'second',
                'shot_xg', 'shot_post_xg', 'pass_end_x', 'pass_end_y',
                'carry_end_x', 'carry_end_y', 'carry_progression']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ---------------------------------------------------------------------------
# Key events (goals, cards, substitutions)
# ---------------------------------------------------------------------------

def extract_key_events(df: pd.DataFrame, data: dict, meta: dict, player_lookup: dict) -> dict:
    """Returns dict with goals_df, cards_df, subs_df DataFrames."""
    # Goals
    goals = df[df['shot_is_goal'] == True].copy()
    goals_rows = []
    for _, r in goals.iterrows():
        goals_rows.append({
            'minute': r['minute'],
            'match_minute': r['match_minute'],
            'period': r['period'],
            'player_id': r['player_id'],
            'player_name': r['player_name'],
            'team_id': r['team_id'],
            'team_name': r['team_name'],
            'xg': r['shot_xg'],
            'body_part': r['shot_body_part'],
            'goal_zone': r['shot_goal_zone'],
        })
    goals_df = pd.DataFrame(goals_rows)

    # Cards
    cards = df[(df['yellow_card'] == True) | (df['red_card'] == True)].copy()
    cards_rows = []
    for _, r in cards.iterrows():
        cards_rows.append({
            'minute': r['minute'],
            'match_minute': r['match_minute'],
            'period': r['period'],
            'player_id': r['player_id'],
            'player_name': r['player_name'],
            'team_id': r['team_id'],
            'team_name': r['team_name'],
            'card_type': 'red' if r['red_card'] else 'yellow',
        })
    cards_df = pd.DataFrame(cards_rows)

    # Substitutions
    subs_raw = data.get('substitutions', {})
    subs_rows = []
    for team_id_str, periods in subs_raw.items():
        team_id = int(team_id_str)
        team_name = meta['home_name'] if team_id == meta['home_id'] else meta['away_name']
        if not isinstance(periods, dict):
            continue
        for period, times in periods.items():
            if not isinstance(times, dict):
                continue
            for sec_str, sub_data in times.items():
                sec = int(sec_str)
                # Wyscout stores cumulative seconds from kickoff (2H starts ~2700s)
                match_minute = sec // 60
                ins = sub_data.get('in', [])
                outs = sub_data.get('out', [])
                # in/out lists are positionally paired
                for player_in, player_out in zip(ins, outs):
                    pid_in = player_in.get('playerId')
                    pid_out = player_out.get('playerId')
                    p_in_info = player_lookup.get(int(pid_in), {}) if pid_in else {}
                    p_out_info = player_lookup.get(int(pid_out), {}) if pid_out else {}
                    subs_rows.append({
                        'team_id': team_id,
                        'team_name': team_name,
                        'period': period,
                        'match_second': sec,
                        'match_minute': match_minute,
                        'player_in_id': int(pid_in) if pid_in else None,
                        'player_in_name': p_in_info.get('shortName', str(pid_in)),
                        'player_out_id': int(pid_out) if pid_out else None,
                        'player_out_name': p_out_info.get('shortName', str(pid_out)),
                    })
    subs_df = pd.DataFrame(subs_rows)

    return {'goals': goals_df, 'cards': cards_df, 'subs': subs_df}


# ---------------------------------------------------------------------------
# Match stats
# ---------------------------------------------------------------------------

def compute_match_stats(df: pd.DataFrame, home_id: int, away_id: int) -> dict:
    """Compute per-team aggregate stats."""
    stats = {}
    for team_id in [home_id, away_id]:
        t = df[df['team_id'] == team_id]
        shots = t[t['type_primary'] == 'shot']
        passes = t[t['type_primary'].isin(['pass', 'free_kick', 'corner', 'throw_in', 'goal_kick'])]
        acc_passes = passes[passes['pass_accurate'] == True]
        duels = t[t['duel_type'].notna()]
        aerial = duels[duels['duel_type'] == 'aerial']
        ground = duels[duels['duel_type'] == 'ground']

        # Possession: fraction of non-interruption events
        total_events_excl = df[df['type_primary'] != 'game_interruption']
        team_events = total_events_excl[total_events_excl['team_id'] == team_id]
        poss_pct = 100 * len(team_events) / max(len(total_events_excl), 1)

        # Pressing events: defensive duels + interceptions in own defensive half
        # (opponent's side: x_plot > 50 means opponent has ball there; x < 40 means team is pressing in opp half)
        # PPDA pressing zone: x_plot > 60 (pressing in opponent's half, defensive actions)
        pressing = t[
            (t['type_primary'].isin(['duel', 'interception'])) &
            (t['x_plot'] > 60)
        ]

        stats[team_id] = {
            'shots': len(shots),
            'shots_on_target': int(shots['shot_on_target'].sum()),
            'goals': int(shots['shot_is_goal'].sum()),
            'xg': round(shots['shot_xg'].sum(), 3),
            'post_shot_xg': round(shots['shot_post_xg'].sum(), 3),
            'passes': len(passes),
            'accurate_passes': len(acc_passes),
            'pass_accuracy': round(100 * len(acc_passes) / max(len(passes), 1), 1),
            'key_passes': int(t['is_key_pass'].sum()),
            'progressive_passes': int(t['is_progressive_pass'].sum()),
            'crosses': int(t['is_cross'].sum()),
            'long_passes': int(t['is_long_pass'].sum()),
            'corners': int((t['type_primary'] == 'corner').sum()),
            'fouls': int((t['type_primary'] == 'infraction').sum()),
            'yellow_cards': int(t['yellow_card'].sum()),
            'red_cards': int(t['red_card'].sum()),
            'possession_pct': round(poss_pct, 1),
            'aerial_duels_total': len(aerial),
            'aerial_duels_won': int(aerial['duel_won'].sum()),
            'ground_duels_total': len(ground),
            'ground_duels_won': int(ground['duel_won'].sum()),
            'interceptions': int((t['type_primary'] == 'interception').sum()),
            'clearances': int((t['type_primary'] == 'clearance').sum()),
            'set_piece_shots': int(shots['is_set_piece'].sum()),
            'open_play_shots': int((shots['is_set_piece'] == False).sum()),
            'set_piece_xg': round(shots[shots['is_set_piece'] == True]['shot_xg'].sum(), 3),
            'open_play_xg': round(shots[shots['is_set_piece'] == False]['shot_xg'].sum(), 3),
        }

    return stats


# ---------------------------------------------------------------------------
# xG timeline
# ---------------------------------------------------------------------------

def compute_xg_timeline(df: pd.DataFrame, home_id: int, away_id: int) -> pd.DataFrame:
    """Cumulative xG over match time for both teams."""
    shots = df[df['type_primary'] == 'shot'].copy()
    shots = shots[shots['shot_xg'].notna()].sort_values('match_minute')

    home_shots = shots[shots['team_id'] == home_id][['match_minute', 'shot_xg']].copy()
    away_shots = shots[shots['team_id'] == away_id][['match_minute', 'shot_xg']].copy()

    home_shots['home_xg_cumul'] = home_shots['shot_xg'].cumsum()
    away_shots['away_xg_cumul'] = away_shots['shot_xg'].cumsum()

    # Build minute-by-minute timeline
    all_minutes = sorted(set(home_shots['match_minute'].tolist() + away_shots['match_minute'].tolist()))
    if not all_minutes:
        return pd.DataFrame(columns=['match_minute', 'home_xg_cumul', 'away_xg_cumul'])

    rows = [{'match_minute': 0, 'home_xg_cumul': 0.0, 'away_xg_cumul': 0.0}]
    home_cum = 0.0
    away_cum = 0.0

    for _, r in shots.sort_values('match_minute').iterrows():
        if r['team_id'] == home_id:
            home_cum += r['shot_xg']
        else:
            away_cum += r['shot_xg']
        rows.append({
            'match_minute': r['match_minute'],
            'home_xg_cumul': round(home_cum, 4),
            'away_xg_cumul': round(away_cum, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Match momentum (rolling event counts)
# ---------------------------------------------------------------------------

def compute_momentum(df: pd.DataFrame, home_id: int, away_id: int, window: int = 5) -> pd.DataFrame:
    """Rolling event counts per team per 5-minute window."""
    # Exclude interruptions
    active = df[df['type_primary'] != 'game_interruption'].copy()
    active['bucket'] = (active['match_minute'] // window) * window

    home_counts = (
        active[active['team_id'] == home_id]
        .groupby('bucket').size().reset_index(name='home_events')
    )
    away_counts = (
        active[active['team_id'] == away_id]
        .groupby('bucket').size().reset_index(name='away_events')
    )

    all_buckets = pd.DataFrame({'bucket': range(0, 95, window)})
    momentum = all_buckets.merge(home_counts, on='bucket', how='left').merge(
        away_counts, on='bucket', how='left'
    ).fillna(0)

    return momentum


# ---------------------------------------------------------------------------
# Player stats
# ---------------------------------------------------------------------------

def compute_player_stats(df: pd.DataFrame, player_lookup: dict, data: dict, meta: dict) -> pd.DataFrame:
    """Per-player aggregate statistics."""
    rows = []
    player_ids = df['player_id'].dropna().unique()

    # Compute xT per player
    xt_grid = compute_xt_grid()
    xt_series = compute_xt_per_player(df, xt_grid)
    xt_dict = xt_series.set_index('player_id')['xT'].to_dict() if not xt_series.empty else {}

    # SCA per player
    sca_df = compute_sca(df)
    sca_dict = sca_df.set_index('player_id')['sca'].to_dict() if not sca_df.empty else {}

    # Minutes played from subs
    minutes_dict = _compute_minutes_played(data, meta, player_lookup)

    for pid in player_ids:
        if pid is None or np.isnan(pid):
            continue
        pid = int(pid)
        p = df[df['player_id'] == pid]
        p_info = player_lookup.get(pid, {})
        team_id = p['team_id'].iloc[0] if len(p) > 0 else None

        shots = p[p['type_primary'] == 'shot']
        passes = p[p['type_primary'].isin(['pass', 'free_kick', 'corner', 'throw_in', 'goal_kick'])]
        acc_passes = passes[passes['pass_accurate'] == True]
        duels = p[p['duel_type'].notna()]
        aerial = duels[duels['duel_type'] == 'aerial']
        ground = duels[duels['duel_type'] == 'ground']

        rows.append({
            'player_id': pid,
            'player_name': p_info.get('shortName', p['player_name'].iloc[0] if len(p) > 0 else ''),
            'team_id': team_id,
            'team_name': p['team_name'].iloc[0] if len(p) > 0 else '',
            'role': p_info.get('role_code2', p['player_role'].iloc[0] if len(p) > 0 else ''),
            'minutes_played': minutes_dict.get(pid, 90),
            'touches': len(p),
            'passes': len(passes),
            'accurate_passes': len(acc_passes),
            'pass_accuracy': round(100 * len(acc_passes) / max(len(passes), 1), 1) if len(passes) > 0 else 0,
            'shots': len(shots),
            'shots_on_target': int(shots['shot_on_target'].sum()),
            'goals': int(shots['shot_is_goal'].sum()),
            'xg': round(shots['shot_xg'].sum(), 3),
            'key_passes': int(p['is_key_pass'].sum()),
            'progressive_passes': int(p['is_progressive_pass'].sum()),
            'crosses': int(p['is_cross'].sum()),
            'duels': len(duels),
            'duels_won': int(duels['duel_won'].sum()),
            'duel_win_pct': round(100 * int(duels['duel_won'].sum()) / max(len(duels), 1), 1) if len(duels) > 0 else 0,
            'aerial_duels': len(aerial),
            'aerial_duels_won': int(aerial['duel_won'].sum()),
            'ground_duels': len(ground),
            'ground_duels_won': int(ground['duel_won'].sum()),
            'interceptions': int((p['type_primary'] == 'interception').sum()),
            'clearances': int((p['type_primary'] == 'clearance').sum()),
            'xT': round(xt_dict.get(pid, 0.0), 4),
            'sca': int(sca_dict.get(pid, 0)),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(['team_id', 'touches'], ascending=[True, False])
    return result


def _compute_minutes_played(data: dict, meta: dict, player_lookup: dict) -> dict:
    """Approximate minutes played based on starting XI + substitutions."""
    minutes = {}

    # Starting XI players play 90 min unless subbed off
    for team_id in [meta['home_id'], meta['away_id']]:
        xi_1h = get_starting_xi(data, team_id, '1H')
        for player in xi_1h:
            minutes[player['player_id']] = 90

    # Process substitutions
    subs_raw = data.get('substitutions', {})
    for team_id_str, periods in subs_raw.items():
        if not isinstance(periods, dict):
            continue
        for period, times in periods.items():
            if not isinstance(times, dict):
                continue
            for sec_str, sub_data in times.items():
                sec = int(sec_str)
                match_minute = sec // 60

                for player_out in sub_data.get('out', []):
                    pid_out = player_out.get('playerId')
                    if pid_out:
                        minutes[int(pid_out)] = match_minute

                for player_in in sub_data.get('in', []):
                    pid_in = player_in.get('playerId')
                    if pid_in:
                        minutes[int(pid_in)] = 90 - match_minute

    return minutes


# ---------------------------------------------------------------------------
# Pass network
# ---------------------------------------------------------------------------

def compute_pass_network(df: pd.DataFrame, team_id: int, period: str = '1H',
                          min_passes: int = 3,
                          min_start: int = None, min_end: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute average positions (nodes) and pass pairs (edges) for a team.
    If min_start/min_end are provided, filter by match_minute range instead of period."""
    if min_start is not None and min_end is not None:
        t = df[(df['team_id'] == team_id) &
               (df['match_minute'] >= min_start) &
               (df['match_minute'] < min_end)].copy()
    else:
        t = df[(df['team_id'] == team_id) & (df['period'] == period)].copy()

    # Nodes: average position of all events per player
    nodes = (
        t.groupby(['player_id', 'player_name'])
        .agg(avg_x=('x_plot', 'mean'), avg_y=('y_plot', 'mean'), touches=('event_id', 'count'))
        .reset_index()
    )

    # Edges: accurate pass pairs
    passes = t[
        (t['type_primary'].isin(['pass', 'free_kick', 'corner'])) &
        (t['pass_accurate'] == True) &
        (t['pass_recipient_id'].notna())
    ].copy()

    if passes.empty:
        return nodes, pd.DataFrame()

    passes['pass_recipient_id'] = passes['pass_recipient_id'].astype(int)
    edge_counts = (
        passes.groupby(['player_id', 'pass_recipient_id'])
        .size().reset_index(name='pass_count')
    )
    edge_counts = edge_counts[edge_counts['pass_count'] >= min_passes]

    # Join average positions
    node_pos = nodes[['player_id', 'avg_x', 'avg_y', 'player_name']].copy()
    edges = edge_counts.merge(
        node_pos.rename(columns={'player_id': 'player_id', 'avg_x': 'from_x', 'avg_y': 'from_y', 'player_name': 'from_name'}),
        on='player_id', how='left'
    ).merge(
        node_pos.rename(columns={'player_id': 'pass_recipient_id', 'avg_x': 'to_x', 'avg_y': 'to_y', 'player_name': 'to_name'}),
        on='pass_recipient_id', how='left'
    )

    return nodes, edges


# ---------------------------------------------------------------------------
# PPDA (Passes Allowed Per Defensive Action)
# ---------------------------------------------------------------------------

def compute_ppda(df: pd.DataFrame, home_id: int, away_id: int) -> dict:
    """
    PPDA = opponent passes in their own half / own pressing actions in that same half.
    Lower = more aggressive high press.

    Uses the halfway line (x_plot = 50) as the boundary — standard PPDA definition.
    Home attacks right: home presses in x_plot > 50, counts away passes in x_plot < 50.
    Away attacks left:  away presses in x_plot < 50, counts home passes in x_plot > 50.
    """
    ppda = {}

    for team_id in [home_id, away_id]:
        if team_id == home_id:
            opp_id = away_id
            opp_passes = df[
                (df['team_id'] == opp_id) &
                (df['type_primary'].isin(['pass', 'free_kick', 'corner', 'throw_in', 'goal_kick'])) &
                (df['x_plot'] < 50)  # away in their own half
            ]
            pressing_actions = df[
                (df['team_id'] == team_id) &
                (df['type_primary'].isin(['duel', 'interception'])) &
                (df['x_plot'] > 50)  # home pressing in away's half
            ]
        else:
            opp_id = home_id
            opp_passes = df[
                (df['team_id'] == opp_id) &
                (df['type_primary'].isin(['pass', 'free_kick', 'corner', 'throw_in', 'goal_kick'])) &
                (df['x_plot'] > 50)  # home in their own half
            ]
            pressing_actions = df[
                (df['team_id'] == team_id) &
                (df['type_primary'].isin(['duel', 'interception'])) &
                (df['x_plot'] < 50)  # away pressing in home's half
            ]

        n_opp_passes = len(opp_passes)
        n_pressing = len(pressing_actions)
        ppda[team_id] = round(n_opp_passes / max(n_pressing, 1), 2)

    return ppda


# ---------------------------------------------------------------------------
# SCA (Shot-Creating Actions)
# ---------------------------------------------------------------------------

def compute_sca(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each shot, trace last 2 non-shot events in same possession chain.
    Count how many times each player appears as an SCA contributor.
    """
    if df.empty:
        return pd.DataFrame(columns=['player_id', 'player_name', 'team_id', 'sca'])

    shots = df[df['type_primary'] == 'shot'].copy()
    if shots.empty:
        return pd.DataFrame(columns=['player_id', 'player_name', 'team_id', 'sca'])

    # For each possession_id, build ordered list of events
    poss_events = (
        df[df['possession_id'].notna()]
        .sort_values('event_id')
        .groupby('possession_id')
        .apply(lambda x: x[['event_id', 'player_id', 'player_name', 'team_id', 'type_primary']].to_dict('records'),
               include_groups=False)
        .to_dict()
    )

    sca_counts = {}

    for _, shot in shots.iterrows():
        poss_id = shot['possession_id']
        if poss_id is None or poss_id not in poss_events:
            continue

        chain = poss_events[poss_id]
        shot_event_id = shot['event_id']

        # Find shot index in chain
        shot_idx = None
        for i, ev in enumerate(chain):
            if ev['event_id'] == shot_event_id:
                shot_idx = i
                break

        if shot_idx is None:
            continue

        # Take the 2 events before the shot (non-interruption)
        pre_shot = [
            ev for ev in chain[:shot_idx]
            if ev['type_primary'] not in ('game_interruption', 'shot_against')
        ][-2:]

        for ev in pre_shot:
            pid = ev['player_id']
            if pid is None or (isinstance(pid, float) and np.isnan(pid)):
                continue
            pid = int(pid)
            if pid not in sca_counts:
                sca_counts[pid] = {
                    'player_id': pid,
                    'player_name': ev['player_name'],
                    'team_id': ev['team_id'],
                    'sca': 0
                }
            sca_counts[pid]['sca'] += 1

    if not sca_counts:
        return pd.DataFrame(columns=['player_id', 'player_name', 'team_id', 'sca'])

    return pd.DataFrame(list(sca_counts.values())).sort_values('sca', ascending=False)


# ---------------------------------------------------------------------------
# xT (Expected Threat)
# ---------------------------------------------------------------------------

# Standard 12x8 xT grid (rows=y-axis/pitch width, cols=x-axis/pitch length)
# Source: StatsBomb / Karun Singh xT framework
_XT_GRID_12x8 = np.array([
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01169549, 0.01364478, 0.01541983, 0.01705904, 0.02143969, 0.02497175, 0.02884498, 0.03117366],
    [0.00649605, 0.00773627, 0.00861088, 0.01017688, 0.01213768, 0.01408537, 0.01574672, 0.01846799, 0.02258279, 0.02609720, 0.03228473, 0.03849432],
    [0.00712071, 0.00833699, 0.00906736, 0.01091661, 0.01302816, 0.01524075, 0.01786375, 0.02101418, 0.02658265, 0.03329802, 0.04619488, 0.07133662],
    [0.00740017, 0.00851408, 0.00953690, 0.01135451, 0.01340437, 0.01541344, 0.01818782, 0.02188782, 0.02809138, 0.04006699, 0.07084898, 0.20271566],
    [0.00740017, 0.00851408, 0.00953690, 0.01135451, 0.01340437, 0.01541344, 0.01818782, 0.02188782, 0.02809138, 0.04006699, 0.07084898, 0.20271566],
    [0.00712071, 0.00833699, 0.00906736, 0.01091661, 0.01302816, 0.01524075, 0.01786375, 0.02101418, 0.02658265, 0.03329802, 0.04619488, 0.07133662],
    [0.00649605, 0.00773627, 0.00861088, 0.01017688, 0.01213768, 0.01408537, 0.01574672, 0.01846799, 0.02258279, 0.02609720, 0.03228473, 0.03849432],
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01169549, 0.01364478, 0.01541983, 0.01705904, 0.02143969, 0.02497175, 0.02884498, 0.03117366],
])  # shape (8, 12)


def compute_xt_grid() -> np.ndarray:
    """Return the standard 12x8 xT grid."""
    return _XT_GRID_12x8.copy()


def _xy_to_xt_cell(x: float, y: float) -> Tuple[int, int]:
    """Convert Wyscout 0-100 coords to xT grid cell (row, col)."""
    col = min(int(x / 100 * 12), 11)
    row = min(int(y / 100 * 8), 7)
    return row, col


def compute_xt_per_player(df: pd.DataFrame, xt_grid: np.ndarray) -> pd.DataFrame:
    """
    xT gain = xT(end_location) - xT(start_location) for passes and carries.
    Sum per player.
    """
    if df.empty:
        return pd.DataFrame(columns=['player_id', 'player_name', 'team_id', 'xT'])

    # Passes with end location
    passes = df[
        (df['type_primary'].isin(['pass', 'free_kick', 'corner', 'throw_in', 'goal_kick'])) &
        df['x'].notna() & df['y'].notna() &
        df['pass_end_x'].notna() & df['pass_end_y'].notna()
    ].copy()

    # Use original (pre-mirror) coords for xT grid lookup, but mirror end location back
    # Since x_plot and pass_end_x are both mirrored for away, we can use them directly
    xt_gains = []

    for _, r in passes.iterrows():
        try:
            start_x = r['x_plot']
            start_y = r['y_plot']
            end_x = r['pass_end_x']
            end_y = r['pass_end_y']
            if any(v is None or np.isnan(v) for v in [start_x, start_y, end_x, end_y]):
                continue
            r_start, c_start = _xy_to_xt_cell(start_x, start_y)
            r_end, c_end = _xy_to_xt_cell(end_x, end_y)
            xt_gain = xt_grid[r_end, c_end] - xt_grid[r_start, c_start]
            xt_gains.append({
                'player_id': r['player_id'],
                'player_name': r['player_name'],
                'team_id': r['team_id'],
                'xt_gain': xt_gain,
                'action': 'pass',
            })
        except Exception:
            continue

    # Carries with end location
    carries = df[df['is_carry'] == True] if 'is_carry' in df.columns else pd.DataFrame()

    for _, r in carries.iterrows():
        try:
            start_x = r['x_plot']
            start_y = r['y_plot']
            end_x = r['carry_end_x']
            end_y = r['carry_end_y']
            if any(v is None or np.isnan(v) for v in [start_x, start_y, end_x, end_y]):
                continue
            r_start, c_start = _xy_to_xt_cell(start_x, start_y)
            r_end, c_end = _xy_to_xt_cell(end_x, end_y)
            xt_gain = xt_grid[r_end, c_end] - xt_grid[r_start, c_start]
            xt_gains.append({
                'player_id': r['player_id'],
                'player_name': r['player_name'],
                'team_id': r['team_id'],
                'xt_gain': xt_gain,
                'action': 'carry',
            })
        except Exception:
            continue

    if not xt_gains:
        return pd.DataFrame(columns=['player_id', 'player_name', 'team_id', 'xT'])

    xt_df = pd.DataFrame(xt_gains)
    result = (
        xt_df.groupby(['player_id', 'player_name', 'team_id'])
        .apply(lambda g: pd.Series({
            'xT':       g['xt_gain'].sum(),
            'xT_pass':  g.loc[g['action'] == 'pass', 'xt_gain'].sum(),
            'xT_carry': g.loc[g['action'] == 'carry', 'xt_gain'].sum(),
        }))
        .reset_index()
    )
    return result.sort_values('xT', ascending=False)


# ---------------------------------------------------------------------------
# Set pieces
# ---------------------------------------------------------------------------

def get_set_pieces(df: pd.DataFrame) -> dict:
    """Extract corner and free kick events with outcomes."""
    corners = df[df['type_primary'] == 'corner'].copy()
    free_kicks = df[df['type_primary'] == 'free_kick'].copy()

    # Tag outcome of each corner/FK based on whether the possession led to a shot/goal
    for events_df in [corners, free_kicks]:
        if events_df.empty:
            continue

    corner_shots = df[
        (df['type_primary'] == 'shot') &
        (df['is_corner'] == True)
    ]

    # Set piece xG breakdown
    sp_shots = df[
        (df['type_primary'] == 'shot') &
        (df['is_set_piece'] == True)
    ]
    op_shots = df[
        (df['type_primary'] == 'shot') &
        (df['is_set_piece'] == False)
    ]

    return {
        'corners': corners,
        'free_kicks': free_kicks,
        'corner_shots': corner_shots,
        'set_piece_shots': sp_shots,
        'open_play_shots': op_shots,
    }


# ---------------------------------------------------------------------------
# Flank attacks
# ---------------------------------------------------------------------------

def compute_flank_attacks(df: pd.DataFrame, home_id: int, away_id: int) -> pd.DataFrame:
    """
    Count attacks per flank (left/center/right) per team.
    Uses possession_flank column added to events_df.
    Only counts rows where flank is set (attack possessions with shot or goal).
    Returns a DataFrame with columns: team_id, flank, count.
    """
    if df.empty or 'possession_flank' not in df.columns:
        return pd.DataFrame(columns=['team_id', 'flank', 'count'])

    # Use unique possessions that have a flank value
    flank_events = df[df['possession_flank'].notna()].copy()
    if flank_events.empty:
        return pd.DataFrame(columns=['team_id', 'flank', 'count'])

    # Deduplicate by possession_id so each possession is counted once
    flank_poss = flank_events.drop_duplicates(subset=['possession_id', 'team_id'])
    result = (
        flank_poss.groupby(['team_id', 'possession_flank'])
        .size().reset_index(name='count')
        .rename(columns={'possession_flank': 'flank'})
    )
    return result


# ---------------------------------------------------------------------------
# Rolling stats (for time-series charts)
# ---------------------------------------------------------------------------

def compute_rolling_stats(df: pd.DataFrame, home_id: int, away_id: int,
                           window: int = 5) -> pd.DataFrame:
    """
    Compute per-window rolling stats for time-series charts.
    Returns a DataFrame with columns:
    bucket, home_poss_pct, away_poss_pct,
    home_duel_win_pct, away_duel_win_pct,
    home_attacks_per_min, away_attacks_per_min,
    home_ppda, away_ppda
    """
    if df.empty:
        return pd.DataFrame()

    max_min = min(int(df['match_minute'].max()), 90)
    buckets = list(range(0, max_min + window, window))
    rows = []

    for bucket in buckets:
        mask = (df['match_minute'] >= bucket) & (df['match_minute'] < bucket + window)
        window_df = df[mask]
        if window_df.empty:
            continue

        # Possession %: fraction of non-interruption events per team
        non_int = window_df[window_df['type_primary'] != 'game_interruption']
        total_ev = max(len(non_int), 1)
        home_ev = len(non_int[non_int['team_id'] == home_id])
        away_ev = len(non_int[non_int['team_id'] == away_id])
        home_poss_pct = 100 * home_ev / total_ev
        away_poss_pct = 100 * away_ev / total_ev

        # Duel win rate
        duels = window_df[window_df['duel_type'].notna()]
        home_duels = duels[duels['team_id'] == home_id]
        away_duels = duels[duels['team_id'] == away_id]
        home_duel_win = 100 * home_duels['duel_won'].sum() / max(len(home_duels), 1)
        away_duel_win = 100 * away_duels['duel_won'].sum() / max(len(away_duels), 1)

        # Attacks per minute: possessions with attack intent per team
        attack_types = {'set_piece_attack', 'offensive_transition_with_ball', 'offensive_transition'}
        attack_poss_home = window_df[
            (window_df['team_id'] == home_id) & window_df['possession_types'].apply(
                lambda pt: bool(pt) and any(t in attack_types for t in pt)
            )
        ]['possession_id'].nunique()
        attack_poss_away = window_df[
            (window_df['team_id'] == away_id) & window_df['possession_types'].apply(
                lambda pt: bool(pt) and any(t in attack_types for t in pt)
            )
        ]['possession_id'].nunique()
        home_apm = attack_poss_home / window
        away_apm = attack_poss_away / window

        # Rolling PPDA in this window
        # home PPDA: away passes in away defensive zone / home pressing actions there
        away_def_passes = window_df[
            (window_df['team_id'] == away_id) &
            (window_df['type_primary'].isin(['pass', 'free_kick', 'corner', 'throw_in', 'goal_kick'])) &
            (window_df['x_plot'] < 40)
        ]
        home_pressing = window_df[
            (window_df['team_id'] == home_id) &
            (window_df['type_primary'].isin(['duel', 'interception'])) &
            (window_df['x_plot'] > 60)
        ]
        home_ppda = len(away_def_passes) / max(len(home_pressing), 1)

        home_def_passes = window_df[
            (window_df['team_id'] == home_id) &
            (window_df['type_primary'].isin(['pass', 'free_kick', 'corner', 'throw_in', 'goal_kick'])) &
            (window_df['x_plot'] > 60)
        ]
        away_pressing = window_df[
            (window_df['team_id'] == away_id) &
            (window_df['type_primary'].isin(['duel', 'interception'])) &
            (window_df['x_plot'] < 40)
        ]
        away_ppda = len(home_def_passes) / max(len(away_pressing), 1)

        # Pass accuracy
        pass_types = ['pass', 'free_kick', 'corner', 'throw_in', 'goal_kick']
        home_p = window_df[(window_df['team_id'] == home_id) &
                           window_df['type_primary'].isin(pass_types)]
        away_p = window_df[(window_df['team_id'] == away_id) &
                           window_df['type_primary'].isin(pass_types)]
        home_pass_acc = 100 * home_p['pass_accurate'].sum() / max(len(home_p), 1)
        away_pass_acc = 100 * away_p['pass_accurate'].sum() / max(len(away_p), 1)

        rows.append({
            'bucket': bucket + window / 2,  # midpoint of window
            'home_poss_pct': round(home_poss_pct, 1),
            'away_poss_pct': round(away_poss_pct, 1),
            'home_duel_win_pct': round(home_duel_win, 1),
            'away_duel_win_pct': round(away_duel_win, 1),
            'home_attacks_per_min': round(home_apm, 2),
            'away_attacks_per_min': round(away_apm, 2),
            'home_ppda': round(min(home_ppda, 30), 2),
            'away_ppda': round(min(away_ppda, 30), 2),
            'home_pass_acc': round(home_pass_acc, 1),
            'away_pass_acc': round(away_pass_acc, 1),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------

def parse_all(data: dict) -> MatchData:
    """Parse all match data and return a MatchData object."""
    meta = extract_match_meta(data)
    player_lookup = build_player_lookup(data)
    team_lookup = build_team_lookup(data, meta)
    events_df = build_events_df(data, player_lookup, team_lookup)

    home_id = meta['home_id']
    away_id = meta['away_id']

    key_events = extract_key_events(events_df, data, meta, player_lookup)
    stats = compute_match_stats(events_df, home_id, away_id)
    xg_timeline = compute_xg_timeline(events_df, home_id, away_id)
    player_stats = compute_player_stats(events_df, player_lookup, data, meta)
    sca_df = compute_sca(events_df)
    xt_grid = compute_xt_grid()
    xt_by_player = compute_xt_per_player(events_df, xt_grid)
    ppda = compute_ppda(events_df, home_id, away_id)
    flank_attacks = compute_flank_attacks(events_df, home_id, away_id)
    rolling_stats = compute_rolling_stats(events_df, home_id, away_id, window=5)

    match = MatchData(
        meta=meta,
        player_lookup=player_lookup,
        team_lookup=team_lookup,
        events_df=events_df,
        substitutions_raw=data.get('substitutions', {}),
        goals=key_events['goals'],
        cards=key_events['cards'],
        stats=stats,
        xg_timeline=xg_timeline,
        player_stats=player_stats,
        sca_df=sca_df,
        xt_by_player=xt_by_player,
        ppda=ppda,
        flank_attacks=flank_attacks,
        rolling_stats=rolling_stats,
    )
    return match


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import os
    path = os.path.join(os.path.dirname(__file__), '5758824.json')
    data = load_match(path)
    match = parse_all(data)
    m = match.meta
    print(f"Match: {m['home_name']} {m['home_score']}-{m['away_score']} {m['away_name']}")
    print(f"Date: {m['date_str']}  |  GW{m['gameweek']}")
    print(f"Formations: {m['home_name']} {m['home_formation_1h']} vs {m['away_name']} {m['away_formation_1h']}")
    print()

    for team_id, s in match.stats.items():
        tname = match.team_lookup[team_id]['name']
        print(f"{tname}: shots={s['shots']}, goals={s['goals']}, xG={s['xg']}, "
              f"passes={s['passes']}, pass%={s['pass_accuracy']}, poss={s['possession_pct']}%")

    print(f"\nGoals:")
    for _, g in match.goals.iterrows():
        print(f"  {g['match_minute']}' {g['player_name']} ({g['team_name']}) xG={g['xg']:.3f}")

    print(f"\nCards: {len(match.cards)}")
    for _, c in match.cards.iterrows():
        print(f"  {c['minute']}' {c['card_type']} {c['player_name']} ({c['team_name']})")

    print(f"\nPPDA: {match.ppda}")

    print(f"\nTop SCA players:")
    if not match.sca_df.empty:
        print(match.sca_df.head(5).to_string(index=False))

    print(f"\nTop xT generators:")
    if not match.xt_by_player.empty:
        print(match.xt_by_player.head(5).to_string(index=False))
