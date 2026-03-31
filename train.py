import pandas as pd
import numpy as np
import joblib
import os

def train_all():
    print("=" * 50)
    print("TRAINING AI MODELS - WORLD CLASS LEVEL")
    print("=" * 50)

    os.makedirs("models", exist_ok=True)

    # ==========================================
    # STEP 1: DOWNLOAD MAXIMUM DATA
    # ==========================================
    print("\n[1/4] Downloading match data...")

    urls = [
        # England - 8 seasons
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2021/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1920/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1819/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1718/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1617/E0.csv",
        # Spain - 8 seasons
        "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/1920/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/1819/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/1718/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/1617/SP1.csv",
        # Germany - 8 seasons
        "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/D1.csv",
        "https://www.football-data.co.uk/mmz4281/1920/D1.csv",
        "https://www.football-data.co.uk/mmz4281/1819/D1.csv",
        "https://www.football-data.co.uk/mmz4281/1718/D1.csv",
        "https://www.football-data.co.uk/mmz4281/1617/D1.csv",
        # Italy - 8 seasons
        "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/I1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/I1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/I1.csv",
        "https://www.football-data.co.uk/mmz4281/1920/I1.csv",
        "https://www.football-data.co.uk/mmz4281/1819/I1.csv",
        "https://www.football-data.co.uk/mmz4281/1718/I1.csv",
        "https://www.football-data.co.uk/mmz4281/1617/I1.csv",
        # France - 8 seasons
        "https://www.football-data.co.uk/mmz4281/2324/F1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/F1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/F1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/F1.csv",
        "https://www.football-data.co.uk/mmz4281/1920/F1.csv",
        "https://www.football-data.co.uk/mmz4281/1819/F1.csv",
        "https://www.football-data.co.uk/mmz4281/1718/F1.csv",
        "https://www.football-data.co.uk/mmz4281/1617/F1.csv",
        # Portugal - 5 seasons
        "https://www.football-data.co.uk/mmz4281/2324/P1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/P1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/P1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/P1.csv",
        "https://www.football-data.co.uk/mmz4281/1920/P1.csv",
        # Netherlands - 5 seasons
        "https://www.football-data.co.uk/mmz4281/2324/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2021/N1.csv",
        "https://www.football-data.co.uk/mmz4281/1920/N1.csv",
        # Belgium - 3 seasons
        "https://www.football-data.co.uk/mmz4281/2324/B1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/B1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/B1.csv",
        # Scotland - 3 seasons
        "https://www.football-data.co.uk/mmz4281/2324/SC0.csv",
        "https://www.football-data.co.uk/mmz4281/2223/SC0.csv",
        "https://www.football-data.co.uk/mmz4281/2122/SC0.csv",
        # Turkey - 3 seasons
        "https://www.football-data.co.uk/mmz4281/2324/T1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/T1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/T1.csv",
        # Greece - 3 seasons
        "https://www.football-data.co.uk/mmz4281/2324/G1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/G1.csv",
        "https://www.football-data.co.uk/mmz4281/2122/G1.csv",
    ]

    frames = []
    for url in urls:
        try:
            df = pd.read_csv(url, low_memory=False)
            frames.append(df)
            print(f"  Downloaded: {url.split('/')[-1]}")
        except Exception:
            print(f"  Skipped: {url.split('/')[-1]}")

    if not frames:
        print("ERROR: Could not download any data!")
        return

    df = pd.concat(frames, ignore_index=True)
    print(f"  Total matches: {len(df)}")

    # ==========================================
    # STEP 2: WORLD CLASS FEATURES
    # ==========================================
    print("\n[2/4] Building world class features...")

    required = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']
    df = df[required].dropna(subset=required)

    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=required).reset_index(drop=True)

    df['shot_diff'] = df['HS'] - df['AS']
    df['target_diff'] = df['HST'] - df['AST']

    df['home_accuracy'] = df['HST'] / df['HS'].replace(0, 1)
    df['away_accuracy'] = df['AST'] / df['AS'].replace(0, 1)

    df['home_dominance'] = df['HS'] / (df['HS'] + df['AS']).replace(0, 1)

    df['home_shot_form'] = df['HS'].rolling(5, min_periods=1).mean()
    df['away_shot_form'] = df['AS'].rolling(5, min_periods=1).mean()

    df['home_shot_form_long'] = df['HS'].rolling(10, min_periods=1).mean()
    df['away_shot_form_long'] = df['AS'].rolling(10, min_periods=1).mean()

    df['home_target_form'] = df['HST'].rolling(5, min_periods=1).mean()
    df['away_target_form'] = df['AST'].rolling(5, min_periods=1).mean()

    df['home_target_ratio'] = df['HST'] / (df['HST'] + df['AST']).replace(0, 1)

    df['home_pressure'] = df['HST'] * df['home_accuracy']
    df['away_pressure'] = df['AST'] * df['away_accuracy']

    df['shot_efficiency'] = (
        (df['HST'] - df['AST']) / (df['HS'] + df['AS']).replace(0, 1)
    )

    df['home_attack_power'] = df['HS'] * 0.4 + df['HST'] * 0.6
    df['away_attack_power'] = df['AS'] * 0.4 + df['AST'] * 0.6
    df['power_diff'] = df['home_attack_power'] - df['away_attack_power']

    df['total_shot_ratio'] = df['HS'] / (df['HS'] + df['AS']).replace(0, 1)

    df['home_on_target_pressure'] = df['HST'].rolling(5, min_periods=1).mean()
    df['away_on_target_pressure'] = df['AST'].rolling(5, min_periods=1).mean()

    df['home_momentum'] = (
        df['HS'].rolling(3, min_periods=1).mean()
        - df['HS'].rolling(10, min_periods=1).mean()
    )
    df['away_momentum'] = (
        df['AS'].rolling(3, min_periods=1).mean()
        - df['AS'].rolling(10, min_periods=1).mean()
    )

    has_odds = all(c in df.columns for c in ['B365H', 'B365D', 'B365A'])

    if has_odds:
        print("  Odds data found! Adding bookmaker features...")
        for col in ['B365H', 'B365D', 'B365A']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['home_prob'] = 1 / df['B365H'].replace(0, np.nan)
        df['draw_prob'] = 1 / df['B365D'].replace(0, np.nan)
        df['away_prob'] = 1 / df['B365A'].replace(0, np.nan)

        total_prob = (
            df['home_prob'] + df['draw_prob'] + df['away_prob']
        ).replace(0, np.nan)

        df['home_prob_norm'] = df['home_prob'] / total_prob
        df['away_prob_norm'] = df['away_prob'] / total_prob

        df['odds_diff'] = df['B365A'] - df['B365H']
        df['home_favourite'] = (df['B365H'] < df['B365A']).astype(int)
    else:
        print("  No odds data found in this dataset.")

    df['result'] = (df['FTHG'] > df['FTAG']).astype(int)

    df = df.dropna()

    feature_cols = [
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
        'home_on_target_pressure', 'away_on_target_pressure',
        'home_momentum', 'away_momentum',
    ]

    if has_odds:
        feature_cols.extend([
            'home_prob', 'away_prob',
            'home_prob_norm', 'away_prob_norm',
            'odds_diff', 'home_favourite',
        ])

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df['result']

    print(f"  Clean matches : {len(df)}")
    print(f"  Features      : {len(feature_cols)}")
    print(f"  Has odds data : {has_odds}")
    print(f"  Home win rate : {y.mean():.1%}")

    # ==========================================
    # STEP 3: TRAIN WORLD CLASS MODELS
    # ==========================================
    print("\n[3/4] Training world class models...")

    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        VotingClassifier,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"  Training: {len(X_train)} matches")
    print(f"  Testing : {len(X_test)} matches")

    results = {}

    # ------ Model 1: Random Forest ------
    print("\n  [1/5] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"  Random Forest: {rf_acc:.2%}")
    joblib.dump(rf, "models/rf_model.pkl")
    results['Random Forest'] = rf_acc

    # ------ Model 2: Gradient Boosting ------
    print("\n  [2/5] Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
    )
    gb.fit(X_train, y_train)
    gb_acc = gb.score(X_test, y_test)
    print(f"  Gradient Boosting: {gb_acc:.2%}")
    joblib.dump(gb, "models/gb_model.pkl")
    results['Gradient Boosting'] = gb_acc

    # ------ Model 3: XGBoost ------
    print("\n  [3/5] XGBoost...")
    xgb_model = None
    try:
        import xgboost as xgb

        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.003,
            subsample=0.8,
            colsample_bytree=0.7,
            colsample_bylevel=0.7,
            min_child_weight=5,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1,
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        xgb_acc = xgb_model.score(X_test, y_test)
        print(f"  XGBoost: {xgb_acc:.2%}")
        joblib.dump(xgb_model, "models/xgb_model.pkl")
        results['XGBoost'] = xgb_acc

    except Exception as e:
        print(f"  XGBoost skipped: {e}")

    # ------ Model 4: Neural Network ------
    print("\n  [4/5] Neural Network...")
    nn_model = None
    scaler = None
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        nn_model = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=128,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
        )
        nn_model.fit(X_train_scaled, y_train)
        nn_acc = nn_model.score(X_test_scaled, y_test)
        print(f"  Neural Network: {nn_acc:.2%}")
        joblib.dump(nn_model, "models/nn_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        results['Neural Network'] = nn_acc

    except Exception as e:
        print(f"  Neural Network skipped: {e}")

    # ------ Model 5: Voting Ensemble ------
    print("\n  [5/5] Voting Ensemble...")
    try:
        estimators = [('rf', rf), ('gb', gb)]
        weights = [1, 1]

        if xgb_model is not None:
            estimators.append(('xgb', xgb_model))
            weights.append(3)

        voting_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights,
        )
        voting_model.fit(X_train, y_train)
        voting_acc = voting_model.score(X_test, y_test)
        print(f"  Voting Ensemble: {voting_acc:.2%}")
        joblib.dump(voting_model, "models/voting_model.pkl")
        results['Voting Ensemble'] = voting_acc

    except Exception as e:
        print(f"  Voting Ensemble skipped: {e}")

    # ==========================================
    # STEP 4: SAVE AND SHOW RESULTS
    # ==========================================
    print("\n[4/4] Saving...")
    joblib.dump(feature_cols, "models/feature_names.pkl")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE - WORLD CLASS!")
    print("=" * 50)

    for name, acc in results.items():
        bar = "█" * int(acc * 40)
        print(f"{name:22} {acc:.2%}  {bar}")

    if results:
        best = max(results, key=results.get)
        avg = sum(results.values()) / len(results)
        print("=" * 50)
        print(f"BEST MODEL   : {best}")
        print(f"BEST ACCURACY: {results[best]:.2%}")
        print(f"AVERAGE (all): {avg:.2%}")
        print("=" * 50)
        print("")
        print("WORLD CLASS GUIDE:")
        print("72-74% = Good")
        print("74-76% = Advanced")
        print("76-78% = Professional")
        print("78%+   = World Class  ← TARGET!")
        print("=" * 50)


if __name__ == "__main__":
    train_all()