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
        if file is None:
            return get_default_data()

        img_bytes = file.read()
        if not img_bytes:
            return get_default_data()

        np_arr = np.frombuffer(img_bytes, np.uint8)
        if np_arr is None or len(np_arr) == 0:
            return get_default_data()

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return get_default_data()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        try:
            import pytesseract

            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

            text = pytesseract.image_to_string(thresh)

            if text is None or not isinstance(text, str) or not text.strip():
                return get_default_data()

            parsed = parse_match_text(text)
            if not isinstance(parsed, dict):
                return get_default_data()

            return parsed

        except ImportError:
            print("pytesseract not installed, using default OCR data")
            return get_default_data()

        except Exception as e:
            print(f"Tesseract OCR error: {e}")
            return get_default_data()

    except Exception as e:
        print(f"OCR Error: {e}")
        return get_default_data()


def parse_match_text(text):
    """Parse extracted text into match data"""
    try:
        if text is None:
            text = ""

        if not isinstance(text, str):
            text = str(text)

        scores = re.findall(r'(\d+)\s*[-:]\s*(\d+)', text)

        wins = len(re.findall(r'[Ww]', text))
        draws = len(re.findall(r'[Dd]', text))
        losses = len(re.findall(r'[Ll]', text))

        total = max(wins + draws + losses, 1)
        form = ((wins * 3 + draws) / (total * 3)) * 100

        goals_home = sum(int(s[0]) for s in scores) if scores else 2
        goals_away = sum(int(s[1]) for s in scores) if scores else 1

        data = {
            "home_team": "Home Team",
            "away_team": "Away Team",
            "form_home": min(max(form, 0), 100),
            "form_away": min(max(100 - form, 0), 100),
            "goals_home": goals_home,
            "goals_away": goals_away,
            "features": [
                float(form),
                float(100 - form),
                float(goals_home),
                float(goals_away),
                float(wins - losses),
                float(goals_home - goals_away)
            ],
            "sequence": [
                [goals_home, wins],
                [goals_away, draws],
                [goals_home, wins],
                [goals_away, losses],
                [goals_home, wins]
            ]
        }

        return data

    except Exception as e:
        print(f"Parse match text error: {e}")
        return get_default_data()


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