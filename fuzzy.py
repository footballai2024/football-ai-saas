# fuzzy.py
# Fuzzy logic - makes decisions like humans do (not just yes/no)

def fuzzy_score(team_a_strength, team_b_strength, home_advantage=5):
    """
    Calculate fuzzy prediction score

    team_a_strength = how strong team A is (0-100)
    team_b_strength = how strong team B is (0-100)
    home_advantage = bonus for playing at home (0-10)

    Returns a number between 0 and 1
    Higher = team A more likely to win
    """
    diff = team_a_strength - team_b_strength
    home_boost = home_advantage * 0.02

    if diff > 15:
        base = 0.85
    elif diff > 10:
        base = 0.75
    elif diff > 5:
        base = 0.65
    elif diff > 0:
        base = 0.55
    elif diff == 0:
        base = 0.50
    elif diff > -5:
        base = 0.45
    elif diff > -10:
        base = 0.35
    elif diff > -15:
        base = 0.25
    else:
        base = 0.15

    return min(base + home_boost, 1.0)