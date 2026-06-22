"""Evaluate the XGBoost models against simple baselines and run a walk-forward
(season-by-season) backtest to check stability over time.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from train_xgboost import FEATURE_COLS, prep_features, temporal_split, train_outcome_classifier

DATA_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = DATA_DIR / "data" / "processed"
REPORTS_DIR = DATA_DIR / "reports"


def baseline_majority_class(train, test):
    majority = train["result"].mode()[0]
    probs = train["result"].value_counts(normalize=True)
    classes = ["away_win", "draw", "home_win"]
    p = np.array([probs.get(c, 1e-6) for c in classes])
    p = p / p.sum()
    y_true = test["result"].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
    proba = np.tile(p, (len(test), 1))
    pred = np.full(len(test), classes.index(majority))
    return {
        "accuracy": accuracy_score(y_true, pred),
        "log_loss": log_loss(y_true, proba, labels=list(range(3))),
    }


def baseline_elo_logistic(train, val, test):
    """Simple logistic-style baseline using only the Elo difference, fit with
    a closed-form approach via the standard Elo expected-score formula
    (no extra training data leakage, just the Elo features already computed).
    """
    classes = ["away_win", "draw", "home_win"]

    def elo_probs(elo_diff):
        p_home_raw = 1 / (1 + 10 ** (-(elo_diff + 65) / 400))  # +65 = generic home advantage in Elo points
        p_away_raw = 1 / (1 + 10 ** ((elo_diff + 65) / 400))
        draw_share = 0.25
        p_home = p_home_raw * (1 - draw_share)
        p_away = p_away_raw * (1 - draw_share)
        p_draw = 1 - p_home - p_away
        return np.stack([p_away, p_draw, p_home], axis=1)

    proba = elo_probs(test["elo_diff"].to_numpy())
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1)
    y_true = test["result"].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "log_loss": log_loss(y_true, proba, labels=list(range(3))),
    }


def walk_forward_backtest(df: pd.DataFrame):
    df = df.sort_values("date").reset_index(drop=True)
    df["season"] = df["date"].dt.year
    seasons = sorted(df["season"].unique())
    results = []

    for test_season in seasons:
        train_df = df[df["season"] < test_season]
        test_df = df[df["season"] == test_season]
        if len(train_df) < 500 or len(test_df) < 30:
            continue

        classes = ["away_win", "draw", "home_win"]
        result_encoder = LabelEncoder().fit(classes)
        y_train = result_encoder.transform(train_df["result"].astype(str))
        y_test = result_encoder.transform(test_df["result"].astype(str))

        X_train, encoders = prep_features(train_df, fit_encoders=True)
        X_test, _ = prep_features(test_df, encoders=encoders)

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
        )
        clf.fit(X_train, y_train, verbose=False)
        proba = clf.predict_proba(X_test)
        pred = clf.predict(X_test)

        results.append(
            {
                "season": int(test_season),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "accuracy": accuracy_score(y_test, pred),
                "log_loss": log_loss(y_test, proba, labels=list(range(3))),
            }
        )
    return pd.DataFrame(results)


def main():
    df = pd.read_parquet(PROCESSED_DIR / "match_features_full.parquet")
    df = df.dropna(subset=FEATURE_COLS + ["result", "home_score", "away_score"])

    train, val, test = temporal_split(df)

    clf, result_encoder, encoders, xgb_metrics = train_outcome_classifier(train, val, test)
    maj_metrics = baseline_majority_class(train, test)
    elo_metrics = baseline_elo_logistic(train, val, test)

    comparison = {
        "xgboost_test": xgb_metrics["test"],
        "majority_class_baseline_test": maj_metrics,
        "elo_only_baseline_test": elo_metrics,
    }
    print(json.dumps(comparison, indent=2))

    backtest = walk_forward_backtest(df)
    print(backtest.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "baseline_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    backtest.to_csv(REPORTS_DIR / "walk_forward_backtest.csv", index=False)


if __name__ == "__main__":
    main()
