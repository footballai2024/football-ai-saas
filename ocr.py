# ocr.py
# Reads text from screenshot images

import cv2
import numpy as np
import re

def extract_data(file):
    """
    Takes an uploaded image file
    Extracts match data from it
    Returns structured data for prediction
    """
    try:
        # Read image from uploaded file
        img_bytes = file.read()
        img = cv2.imdecode(
            np.frombuffer(img_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        # Convert to grayscale (black and white)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try to use OCR if available
        try:
            import pytesseract
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            text = pytesseract.image_to_string(thresh)
            return parse_match_text(text)
        except ImportError:
            # If pytesseract not available use defaults
            return get_default_data()

    except Exception as e:
        print(f"OCR Error: {e}")
        return get_default_data()

def parse_match_text(text):
    """Parse extracted text into match data"""

    # Find scores like 2-1 or 3:0
    scores = re.findall(r'(\d+)\s*[-:]\s*(\d+)', text)

    # Count W D L letters for form
    wins = len(re.findall(r'[Ww]', text))
    draws = len(re.findall(r'[Dd]', text))
    losses = len(re.findall(r'[Ll]', text))

    total = max(wins + draws + losses, 1)
    form = ((wins * 3 + draws) / (total * 3)) * 100

    goals_home = sum(int(s[0]) for s in scores) if scores else 2
    goals_away = sum(int(s[1]) for s in scores) if scores else 2

    return {
        "home_team": "Home Team",
        "away_team": "Away Team",
        "form_home": min(form, 100),
        "form_away": min(100 - form, 100),
        "goals_home": goals_home,
        "goals_away": goals_away,
        "features": [
            form, 100 - form,
            goals_home, goals_away,
            wins - losses,
            goals_home - goals_away
        ],
        "sequence": [
            [goals_home, wins],
            [goals_away, draws],
            [goals_home, wins],
            [goals_away, losses],
            [goals_home, wins]
        ]
    }

def get_default_data():
    """Default data when OCR fails"""
    return {
        "home_team": "Home Team",
        "away_team": "Away Team",
        "form_home": 60,
        "form_away": 55,
        "goals_home": 2,
        "goals_away": 1,
        "features": [60, 55, 10, 8, 2, 1],
        "sequence": [[10, 5], [8, 3], [10, 5], [8, 3], [10, 5]]
    }