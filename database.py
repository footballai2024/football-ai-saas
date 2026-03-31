# database.py
# This file handles saving and reading predictions from database

import sqlite3
from datetime import datetime

def init_db():
    """Creates the database table if it doesnt exist"""
    conn = sqlite3.connect("predictions.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            league TEXT,
            home_team TEXT,
            away_team TEXT,
            prediction TEXT,
            confidence REAL,
            xg_home REAL,
            xg_away REAL,
            odds_home REAL,
            odds_draw REAL,
            odds_away REAL,
            value_bet TEXT,
            actual_result TEXT DEFAULT 'pending'
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(league, home, away, pred, conf, xg_h, xg_a, odds, value):
    """Saves a prediction to database"""
    conn = sqlite3.connect("predictions.db")
    conn.execute("""
        INSERT INTO predictions
        (timestamp, league, home_team, away_team, prediction,
         confidence, xg_home, xg_away, odds_home, odds_draw,
         odds_away, value_bet)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        league, home, away, pred, conf, xg_h, xg_a,
        odds.get('home', 0),
        odds.get('draw', 0),
        odds.get('away', 0),
        value
    ))
    conn.commit()
    conn.close()

def get_all_predictions():
    """Gets last 50 predictions"""
    conn = sqlite3.connect("predictions.db")
    cursor = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 50"
    )
    columns = [desc[0] for desc in cursor.description]
    rows = []
    for row in cursor.fetchall():
        rows.append(dict(zip(columns, row)))
    conn.close()
    return rows

def get_accuracy_stats():
    """Calculates how accurate our predictions are"""
    conn = sqlite3.connect("predictions.db")
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN prediction = actual_result THEN 1 ELSE 0 END) as correct
        FROM predictions
        WHERE actual_result != 'pending'
    """)
    row = cursor.fetchone()
    conn.close()

    total = row[0] if row[0] else 0
    correct = row[1] if row[1] else 0
    accuracy = (correct / total * 100) if total > 0 else 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 2)
    }

# Create database when this file loads
init_db()