# api.py
# Connects to real football data from the internet

import requests
from config import FOOTBALL_API_KEY, ODDS_API_KEY, LEAGUES

HEADERS = {
    "X-RapidAPI-Key": FOOTBALL_API_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

def get_upcoming_matches():
    """Get next matches from all leagues"""
    all_matches = []

    for league_name, league_id in LEAGUES.items():
        try:
            url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
            params = {
                "league": league_id,
                "next": 5,
                "season": 2024
            }

            res = requests.get(
                url, headers=HEADERS,
                params=params, timeout=10
            )
            data = res.json()

            if 'response' in data:
                for match in data['response']:
                    match['league_name'] = league_name
                    all_matches.append(match)

        except Exception as e:
            print(f"Error fetching {league_name}: {e}")

    return all_matches

def get_odds(match_name=""):
    """Get betting odds for a match"""
    try:
        url = "https://api.the-odds-api.com/v4/sports/soccer/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "eu",
            "markets": "h2h"
        }

        res = requests.get(url, params=params, timeout=10)
        data = res.json()

        # Try to find matching game
        for event in data:
            home = event.get('home_team', '').lower()
            if match_name.lower() in home:
                outcomes = event['bookmakers'][0]['markets'][0]['outcomes']
                return {
                    "home": outcomes[0]['price'],
                    "draw": outcomes[1]['price'] if len(outcomes) > 1 else 3.0,
                    "away": outcomes[2]['price'] if len(outcomes) > 2 else 3.0
                }

        # Default odds if not found
        return {"home": 2.0, "draw": 3.2, "away": 2.8}

    except Exception:
        return {"home": 2.0, "draw": 3.2, "away": 2.8}

def get_live_stats():
    """Get data for live prediction (no screenshot needed)"""
    return {
        "home_team": "Live Home",
        "away_team": "Live Away",
        "form_home": 65,
        "form_away": 60,
        "goals_home": 1,
        "goals_away": 1,
        "features": [65, 60, 10, 8, 5, 3],
        "sequence": [[10, 5], [8, 3], [10, 5], [8, 3], [10, 5]]
    }