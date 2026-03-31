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
                url,
                headers=HEADERS,
                params=params,
                timeout=10
            )

            if res.status_code != 200:
                print(f"API error for {league_name}: status {res.status_code}")
                continue

            data = res.json()
            if not isinstance(data, dict):
                continue

            response_data = data.get('response', [])
            if not isinstance(response_data, list):
                response_data = []

            for match in response_data:
                if isinstance(match, dict):
                    match['league_name'] = league_name
                    all_matches.append(match)

        except Exception as e:
            print(f"Error fetching {league_name}: {e}")

    return all_matches


def get_odds(match_name=""):
    """Get betting odds for a match"""
    default_odds = {"home": 2.0, "draw": 3.2, "away": 2.8}

    try:
        url = "https://api.the-odds-api.com/v4/sports/soccer/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "eu",
            "markets": "h2h"
        }

        res = requests.get(url, params=params, timeout=10)

        if res.status_code != 200:
            print(f"Odds API error: status {res.status_code}")
            return default_odds

        data = res.json()
        if not isinstance(data, list):
            return default_odds

        match_name = (match_name or "").lower().strip()

        for event in data:
            if not isinstance(event, dict):
                continue

            home = str(event.get('home_team', '')).lower()
            away = str(event.get('away_team', '')).lower()

            # If match name is empty, just skip matching and return first valid odds found
            is_match = (
                not match_name or
                match_name in home or
                match_name in away
            )

            if not is_match:
                continue

            bookmakers = event.get('bookmakers', [])
            if not bookmakers or not isinstance(bookmakers, list):
                continue

            bookmaker = bookmakers[0] if bookmakers else {}
            markets = bookmaker.get('markets', [])
            if not markets or not isinstance(markets, list):
                continue

            market = markets[0] if markets else {}
            outcomes = market.get('outcomes', [])
            if not outcomes or not isinstance(outcomes, list):
                continue

            home_price = None
            draw_price = None
            away_price = None

            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue

                name = str(outcome.get('name', '')).lower()
                price = outcome.get('price')

                try:
                    price = float(price)
                except Exception:
                    continue

                if 'draw' in name:
                    draw_price = price
                elif home and name == home:
                    home_price = price
                elif away and name == away:
                    away_price = price

            # Fallback by position if names didn't match
            if home_price is None and len(outcomes) > 0:
                try:
                    home_price = float(outcomes[0].get('price', default_odds['home']))
                except Exception:
                    home_price = default_odds['home']

            if draw_price is None:
                if len(outcomes) > 1:
                    try:
                        draw_price = float(outcomes[1].get('price', default_odds['draw']))
                    except Exception:
                        draw_price = default_odds['draw']
                else:
                    draw_price = default_odds['draw']

            if away_price is None:
                if len(outcomes) > 2:
                    try:
                        away_price = float(outcomes[2].get('price', default_odds['away']))
                    except Exception:
                        away_price = default_odds['away']
                else:
                    away_price = default_odds['away']

            return {
                "home": home_price if home_price and home_price > 0 else default_odds["home"],
                "draw": draw_price if draw_price and draw_price > 0 else default_odds["draw"],
                "away": away_price if away_price and away_price > 0 else default_odds["away"]
            }

        return default_odds

    except Exception as e:
        print(f"Odds fetch error: {e}")
        return default_odds


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