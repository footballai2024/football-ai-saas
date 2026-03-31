# alerts.py
# Sends prediction alerts to your Telegram

import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_telegram(message):
    """Send a message to your Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

        requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        })

        print("Telegram alert sent!")

    except Exception as e:
        print(f"Telegram alert failed: {e}")


def send_match_alert(result):
    """Send a formatted match prediction alert"""

    message = f"""
⚽ <b>AI MATCH PREDICTION</b>

🏠 {result['home_team']} vs {result['away_team']}

📊 <b>Prediction:</b> {result['prediction']}
🎯 <b>Confidence:</b> {result['confidence']}%

⚽ xG: {result['xG_home']} vs {result['xG_away']}

💰 <b>Odds:</b>
  Home: {result['odds']['home']}
  Draw: {result['odds']['draw']}
  Away: {result['odds']['away']}

🔥 <b>Value Bets:</b>
  Home: {result['value_bet_home']}
  Away: {result['value_bet_away']}

💡 <b>Advice:</b> {result['betting_advice']}
"""

    send_telegram(message)