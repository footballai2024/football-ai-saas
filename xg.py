# xg.py
# Calculates Expected Goals (how many goals a team should score)

def calculate_xg(goals, shots, shots_on_target, possession=50):
    """
    Calculate expected goals based on match stats

    goals = actual goals scored
    shots = total shots taken
    shots_on_target = shots that went towards goal
    possession = ball possession percentage
    """
    base = 0.1 * shots + 0.3 * shots_on_target + 0.5 * goals
    possession_bonus = (possession - 50) * 0.02
    return round(base + possession_bonus, 2)