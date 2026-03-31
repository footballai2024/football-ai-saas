import numpy as np
import pandas as pd
import warnings
import joblib

warnings.filterwarnings("ignore")

from ocr import extract_data
from xg import calculate_xg
from fuzzy import fuzzy_score
from api import get_odds, get_live_stats
from config import BET_MIN_ODDS, BET_MAX_ODDS

# ==========================================
# LOAD ALL 5 MODELS
# ==========================================
MODELS_LOADED = False
rf = None
gb = None
xgb_model = None
nn_model = None
scaler = None
voting_model = None
feature_names = None

try:
    rf = joblib.load('models/rf_model.pkl')
    gb = joblib.load('models/gb_model.pkl')
    print("RF and GB loaded!")

    try:
        xgb_model = joblib.load('models/xgb_model.pkl')
        print("XGBoost loaded!")
    except Exception:
        print("XGBoost not available")

    try:
        nn_model = joblib.load('models/nn_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("Neural Network loaded!")
    except Exception:
        print("Neural Network not available")

    try:
        voting_model = joblib.load('models/voting_model.pkl')
        print("Voting Ensemble loaded!")
    except Exception:
        print("Voting not available")

    try:
        feature_names = joblib.load('models/feature_names.pkl')
        print(f"Features loaded: {len(feature_names)}")
    except Exception:
        # Default feature names matching train.py
        feature_names = [
            'HS', 'AS', 'HST', 'AST',
            'shot_diff', 'target_diff',
            'home_accuracy', 'away_accuracy',
            'home_dominance',
            'home_shot_form', 'away_shot_form',
            'home_shot_form_long', 'away_shot_form_long',
            'home_target_form', 'away_target_form',
            'home_target_ratio',
            'home_pressure', 'away_pressure',
            'shot_efficiency',
            'home_attack_power', 'away_attack_power',
            'power_diff', 'total_shot_ratio',
            'home_on_target_pressure',
            'away_on_target_pressure',
            'home_momentum', 'away_momentum'
        ]
        print("Using default feature names")

    MODELS_LOADED = True
    print("All models loaded successfully!")

except Exception as e:
    print(f"Models not loaded: {e}")
    print("Run 'python train.py' first!")


def build_features(data, odds):
    """
    Build ALL features matching train.py exactly
    Including odds features if available
    """

    f = data.get('features', [])
    while len(f) < 6:
        f.append(0)

    # Basic shot values
    hs = float(f[0]) if f[0] else 12.0
    as_ = float(f[1]) if f[1] else 10.0
    hst = float(f[2]) if f[2] else 5.0
    ast = float(f[3]) if f[3] else 3.5

    # Basic features
    shot_diff = hs - as_
    target_diff = hst - ast

    home_accuracy = hst / max(hs, 1)
    away_accuracy = ast / max(as_, 1)

    home_dominance = hs / max(hs + as_, 1)

    # Rolling forms (use current values as estimate)
    home_shot_form = hs
    away_shot_form = as_
    home_shot_form_long = hs
    away_shot_form_long = as_

    # Target forms
    home_target_form = hst
    away_target_form = ast

    home_target_ratio = hst / max(hst + ast, 1)

    home_pressure = hst * home_accuracy
    away_pressure = ast * away_accuracy

    shot_efficiency = (hst - ast) / max(hs + as_, 1)

    home_attack_power = hs * 0.4 + hst * 0.6
    away_attack_power = as_ * 0.4 + ast * 0.6
    power_diff = home_attack_power - away_attack_power

    total_shot_ratio = hs / max(hs + as_, 1)

    home_on_target_pressure = hst
    away_on_target_pressure = ast

    home_momentum = 0.0
    away_momentum = 0.0

    # Build base feature dictionary
    all_values = {
        'HS': hs,
        'AS': as_,
        'HST': hst,
        'AST': ast,
        'shot_diff': shot_diff,
        'target_diff': target_diff,
        'home_accuracy': home_accuracy,
        'away_accuracy': away_accuracy,
        'home_dominance': home_dominance,
        'home_shot_form': home_shot_form,
        'away_shot_form': away_shot_form,
        'home_shot_form_long': home_shot_form_long,
        'away_shot_form_long': away_shot_form_long,
        'home_target_form': home_target_form,
        'away_target_form': away_target_form,
        'home_target_ratio': home_target_ratio,
        'home_pressure': home_pressure,
        'away_pressure': away_pressure,
        'shot_efficiency': shot_efficiency,
        'home_attack_power': home_attack_power,
        'away_attack_power': away_attack_power,
        'power_diff': power_diff,
        'total_shot_ratio': total_shot_ratio,
        'home_on_target_pressure': home_on_target_pressure,
        'away_on_target_pressure': away_on_target_pressure,
        'home_momentum': home_momentum,
        'away_momentum': away_momentum,
    }

    # Add odds features if available (matching train.py)
    home_odds = odds.get('home', 2.0)
    draw_odds = odds.get('draw', 3.2)
    away_odds = odds.get('away', 2.8)

    if home_odds > 0 and draw_odds > 0 and away_odds > 0:
        home_prob = 1 / home_odds
        draw_prob = 1 / draw_odds
        away_prob = 1 / away_odds

        total_prob = home_prob + draw_prob + away_prob

        home_prob_norm = home_prob / total_prob
        away_prob_norm = away_prob / total_prob

        odds_diff = away_odds - home_odds
        home_favourite = 1 if home_odds < away_odds else 0

        all_values['home_prob'] = home_prob
        all_values['draw_prob'] = draw_prob
        all_values['away_prob'] = away_prob
        all_values['home_prob_norm'] = home_prob_norm
        all_values['away_prob_norm'] = away_prob_norm
        all_values['odds_diff'] = odds_diff
        all_values['home_favourite'] = home_favourite

    # Return values in EXACT feature_names order
    result = []
    for name in feature_names:
        result.append(float(all_values.get(name, 0.0)))

    return result


def full_pipeline(file=None, live=False):
    """Main prediction function using all 5 models"""

    # Get data
    if live:
        data = get_live_stats()
    elif file:
        data = extract_data(file)
    else:
        return {"error": "No input"}

    # Get odds first (needed for features)
    odds = get_odds(data.get('home_team', ''))

    # Build features
    all_features = build_features(data, odds)

    # xG calculation
    xg_home = calculate_xg(
        data.get('goals_home', 1),
        all_features[0] if len(all_features) > 0 else 10,
        all_features[2] if len(all_features) > 2 else 5
    )
    xg_away = calculate_xg(
        data.get('goals_away', 1),
        all_features[1] if len(all_features) > 1 else 8,
        all_features[3] if len(all_features) > 3 else 3
    )

    # Fuzzy score
    fuzzy = fuzzy_score(
        data.get('form_home', 60),
        data.get('form_away', 60)
    )

    # Model predictions
    rf_pred = 0.5
    gb_pred = 0.5
    xgb_pred = 0.5
    nn_pred = 0.5
    vote_pred = 0.5

    if MODELS_LOADED:
        # Create feature DataFrame
        feature_df = pd.DataFrame(
            [all_features],
            columns=feature_names
        )

        # Fill any NaN values
        feature_df = feature_df.fillna(0)

        # Random Forest
        try:
            rf_pred = float(
                rf.predict_proba(feature_df)[0][1]
            )
        except Exception:
            rf_pred = 0.5

        # Gradient Boosting
        try:
            gb_pred = float(
                gb.predict_proba(feature_df)[0][1]
            )
        except Exception:
            gb_pred = 0.5

        # XGBoost
        if xgb_model is not None:
            try:
                xgb_pred = float(
                    xgb_model.predict_proba(
                        feature_df
                    )[0][1]
                )
            except Exception:
                xgb_pred = (rf_pred + gb_pred) / 2

        # Neural Network
        if nn_model is not None and scaler is not None:
            try:
                scaled = scaler.transform(feature_df)
                nn_pred = float(
                    nn_model.predict_proba(scaled)[0][1]
                )
            except Exception:
                nn_pred = (rf_pred + gb_pred) / 2

        # Voting Ensemble
        if voting_model is not None:
            try:
                vote_pred = float(
                    voting_model.predict_proba(
                        feature_df
                    )[0][1]
                )
            except Exception:
                vote_pred = (
                    rf_pred + gb_pred + xgb_pred
                ) / 3

    # Combine all models
    # Weights based on accuracy:
    # NN=75.10%, GB=75.05%, XGB=74.81%, Vote=74.95%, RF=74.18%
    if (nn_model is not None and
            xgb_model is not None and
            voting_model is not None):
        final = (
            0.05 * fuzzy +
            0.10 * rf_pred +
            0.20 * gb_pred +
            0.20 * xgb_pred +
            0.25 * nn_pred +
            0.20 * vote_pred
        )
    elif xgb_model is not None:
        final = (
            0.10 * fuzzy +
            0.20 * rf_pred +
            0.30 * gb_pred +
            0.40 * xgb_pred
        )
    else:
        final = (
            0.15 * fuzzy +
            0.40 * rf_pred +
            0.45 * gb_pred
        )

    # Decision
    if final > 0.6:
        prediction = "Home Win"
    elif final < 0.4:
        prediction = "Away Win"
    else:
        prediction = "Draw"

    confidence = round(final * 100, 2)

    # Value bets
    value_home = check_value_bet(
        final, odds.get('home', 2.0)
    )
    value_away = check_value_bet(
        1 - final, odds.get('away', 2.0)
    )

    # Betting advice
    advice = get_betting_advice(final, odds)

    return {
        "home_team": data.get('home_team', 'Home'),
        "away_team": data.get('away_team', 'Away'),
        "prediction": prediction,
        "confidence": confidence,
        "xG_home": xg_home,
        "xG_away": xg_away,
        "fuzzy_score": round(fuzzy, 3),
        "rf_score": round(rf_pred, 3),
        "gb_score": round(gb_pred, 3),
        "lstm_score": round(xgb_pred, 3),
        "odds": odds,
        "value_bet_home": value_home,
        "value_bet_away": value_away,
        "betting_advice": advice
    }


def check_value_bet(prob, odds):
    """Check if bet has value"""
    if odds <= 0:
        return "No data"
    bk = 1 / odds
    if prob > bk and prob > 0.55:
        edge = round((prob - bk) * 100, 1)
        return f"VALUE BET (edge: {edge}%)"
    return "No value"


def get_betting_advice(prob, odds):
    """Give betting recommendation"""
    ho = odds.get('home', 0)
    ao = odds.get('away', 0)

    if prob > 0.65 and BET_MIN_ODDS <= ho <= BET_MAX_ODDS:
        return "BET HOME (Strong signal)"
    elif (prob < 0.35 and
            BET_MIN_ODDS <= ao <= BET_MAX_ODDS):
        return "BET AWAY (Strong signal)"
    elif 0.45 <= prob <= 0.55:
        return "CONSIDER DRAW"
    else:
        return "SKIP - Low confidence"