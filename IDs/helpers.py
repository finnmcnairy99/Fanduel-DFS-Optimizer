from IDs.player_ids import PLAYER_IDS
from IDs.team_ids import TEAM_IDS


def get_player_ids(player_names, team):
    """
    Convert a list of player names into a comma-separated string of IDs.

    Example:
        get_player_ids(["Jayson Tatum", "Payton Pritchard"], "BOS")
        -> "1628369,1630202"
    """
    team = team.upper()

    if team not in PLAYER_IDS:
        raise ValueError(f"Unknown team: {team}")

    team_players = PLAYER_IDS[team]

    ids = []
    for name in player_names:
        if name not in team_players:
            raise ValueError(f"{name} not found on {team}")
        ids.append(str(team_players[name]))

    return ",".join(ids)


def build_wowy_url(
    team,
    players_off,
    table,
    season="2025-26",
    season_type="Regular Season",
    stat_type="Per100Possessions",
    from_margin=-20,
    to_margin=20,
):
    """
    Build a PBPStats WOWY URL automatically.

    table options:
        "Scoring"
        "Rebounds"
        "Assists"
        "Turnovers"
    """
    team = team.upper()

    if team not in TEAM_IDS:
        raise ValueError(f"Unknown team: {team}")

    team_id = TEAM_IDS[team]

    n = len(players_off)
    if n == 0:
        raise ValueError("players_off must contain at least one player")
    if n > 5:
        raise ValueError("pbpstats supports at most 5 ExactlyOffFloor players")

    off_ids = get_player_ids(players_off, team)
    off_param = f"0Exactly{n}OffFloor"

    base = "https://www.pbpstats.com/wowy/nba"

    params = (
        f"{off_param}={off_ids}"
        f"&TeamId={team_id}"
        f"&Season={season}"
        f"&SeasonType={season_type.replace(' ', '+')}"
        f"&Type=Player"
        f"&FromMargin={from_margin}"
        f"&ToMargin={to_margin}"
        f"&Table={table}"
        f"&StatType={stat_type}"
    )

    return f"{base}?{params}"
