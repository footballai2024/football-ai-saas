# scheduler.py
# Runs automatically - checks matches and sends alerts

import time
from predictor import full_pipeline
from alerts import send_match_alert
from database import save_prediction
from self_learn import check_and_retrain
from config import CHECK_INTERVAL_SECONDS, CONFIDENCE_THRESHOLD

def run_auto():
    print("=" * 50)
    print("AUTO PREDICTION SYSTEM RUNNING")
    print("Checking every", CHECK_INTERVAL_SECONDS, "seconds")
    print("=" * 50)

    cycle = 0

    while True:
        cycle += 1
        print(f"\n--- Cycle {cycle} ---")

        try:
            # Try to get live data
            from api import get_upcoming_matches
            matches = get_upcoming_matches()
            print(f"Found {len(matches)} matches")

            for match in matches:
                try:
                    home = match['teams']['home']['name']
                    away = match['teams']['away']['name']
                    league = match.get('league_name', 'Unknown')

                    print(f"Analyzing: {home} vs {away}")

                    result = full_pipeline(None, live=True)
                    result['home_team'] = home
                    result['away_team'] = away

                    # Save prediction
                    save_prediction(
                        league, home, away,
                        result['prediction'],
                        result['confidence'],
                        result['xG_home'],
                        result['xG_away'],
                        result['odds'],
                        result.get('betting_advice', 'N/A')
                    )

                    # Alert if strong prediction
                    if result['confidence'] > CONFIDENCE_THRESHOLD:
                        print(f"  STRONG: {result['prediction']} ({result['confidence']}%)")
                        send_match_alert(result)
                    else:
                        print(f"  Weak: {result['confidence']}% - skipped")

                except Exception as e:
                    print(f"  Error: {e}")

            # Self-learn check every 10 cycles
            if cycle % 10 == 0:
                print("\nSelf-learning check...")
                check_and_retrain()

        except Exception as e:
            print(f"System error: {e}")

        print(f"Sleeping {CHECK_INTERVAL_SECONDS}s...")
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_auto()