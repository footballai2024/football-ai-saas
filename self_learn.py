# self_learn.py
# Makes the AI learn from its mistakes and improve over time

import sqlite3
import pandas as pd
import joblib
import os

def check_and_retrain():
    """Check accuracy and retrain if needed"""

    if not os.path.exists("predictions.db"):
        print("No prediction history yet")
        return

    conn = sqlite3.connect("predictions.db")

    try:
        df = pd.read_sql(
            "SELECT * FROM predictions WHERE actual_result != 'pending'",
            conn
        )
    except Exception:
        print("No completed predictions yet")
        conn.close()
        return

    conn.close()

    if len(df) < 20:
        print(f"Only {len(df)} results - need 20+ to retrain")
        return

    # Calculate accuracy
    df['correct'] = (df['prediction'] == df['actual_result']).astype(int)
    accuracy = df['correct'].mean()

    print(f"Current accuracy: {accuracy:.2%}")

    if accuracy < 0.60:
        print("Accuracy below 60% - retraining models...")
        retrain(df)
    else:
        print("Accuracy OK - no retrain needed")


def retrain(df):
    """Retrain models using prediction history"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    result_map = {"Home Win": 1, "Away Win": 0, "Draw": 0}

    X = df[['confidence', 'xg_home', 'xg_away', 'odds_home', 'odds_away']].values
    y = df['actual_result'].map(result_map).fillna(0).values

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, "models/rf_model.pkl")

    gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
    gb.fit(X, y)
    joblib.dump(gb, "models/gb_model.pkl")

    print("Models retrained and saved!")


if __name__ == "__main__":
    check_and_retrain()