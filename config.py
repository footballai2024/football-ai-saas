# config.py
# This file stores all your settings and API keys

import os

# API Keys - we use os.environ.get so it works on server too
FOOTBALL_API_KEY = os.environ.get("FOOTBALL_API_KEY", "paste_your_key_here")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "cffd0302f052af91f7c9ffc0c374f1c5")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "paste_your_token_here")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "paste_your_chat_id_here")

# Football leagues we track
LEAGUES = {
    "EPL": 39,
    "La Liga": 140,
    "Bundesliga": 78,
    "Ligue 1": 61,
    "Serie A": 135
}

# Prediction settings
CONFIDENCE_THRESHOLD = 65
BET_MIN_ODDS = 1.8
BET_MAX_ODDS = 3.5

# How often scheduler runs (seconds)
CHECK_INTERVAL_SECONDS = 600