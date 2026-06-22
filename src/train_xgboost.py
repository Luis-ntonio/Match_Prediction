"""Train XGBoost models for 1X2 outcome classification and expected-goals regression.

Split is purely temporal (train -> val -> test by date) to avoid leakage from
shuffling future matches into training.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_poisson_deviance
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

DATA_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = DATA_DIR / "data" / "processed"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"

FEATURE_COLS = [
    "elo_diff",
    "elo_home_pre",
    "elo_away_pre",
    "h2h_matches",
    "h2h_home_win_rate",
    "home_form_goals_for_5",
    "home_form_goals_against_5",
    "home_form_points_5",
    "away_form_goals_for_5",
    "away_form_goals_against_5",
    "away_form_points_5",
    "home_form_goals_for_10",
    "home_form_goals_against_10",
    "home_form_points_10",
    "away_form_goals_for_10",
    "away_form_goals_against_10",
    "away_form_points_10",
    "form_points_diff_5",
    "form_goal_diff_5",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "home_matches_played",
    "away_matches_played",
    "home_mcmc_attack_mean",
    "home_mcmc_attack_sd",
    "home_mcmc_defense_mean",
    "home_mcmc_defense_sd",
    "away_mcmc_attack_mean",
    "away_mcmc_attack_sd",
    "away_mcmc_defense_mean",
    "away_mcmc_defense_sd",
    "mcmc_attack_diff",
    "mcmc_defense_diff",
    "mcmc_net_strength_home",
    "mcmc_net_strength_away",
    "neutral",
]
CATEGORICAL_COLS = ["tournament_tier"]


def temporal_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.15):
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    train = df.iloc[: n - n_val - n_test]
    val = df.iloc[n - n_val - n_test : n - n_test]
    test = df.iloc[n - n_test :]
    return train, val, test


def prep_features(df: pd.DataFrame, encoders: dict[str, LabelEncoder] | None = None, fit_encoders=False):
    X = df[FEATURE_COLS].copy()
    X["neutral"] = X["neutral"].astype(int)

    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if fit_encoders:
            enc = LabelEncoder()
            df[f"{col}_enc"] = enc.fit_transform(df[col].astype(str))
            encoders[col] = enc
        else:
            enc = encoders[col]
            known = set(enc.classes_)
            safe_vals = df[col].astype(str).where(df[col].astype(str).isin(known), enc.classes_[0])
            df[f"{col}_enc"] = enc.transform(safe_vals)
        X[f"{col}_enc"] = df[f"{col}_enc"]
    return X, encoders


def train_outcome_classifier(train, val, test):
    result_encoder = LabelEncoder()
    y_train = result_encoder.fit_transform(train["result"].astype(str))
    y_val = result_encoder.transform(val["result"].astype(str))
    y_test = result_encoder.transform(test["result"].astype(str))

    X_train, encoders = prep_features(train, fit_encoders=True)
    X_val, _ = prep_features(val, encoders=encoders)
    X_test, _ = prep_features(test, encoders=encoders)

    clf = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    metrics = {}
    for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        proba = clf.predict_proba(X)
        pred = clf.predict(X)
        metrics[name] = {
            "accuracy": accuracy_score(y, pred),
            "log_loss": log_loss(y, proba, labels=list(range(3))),
        }
        # Brier score per class (one-vs-rest), averaged
        brier = np.mean(
            [brier_score_loss((y == c).astype(int), proba[:, c]) for c in range(3)]
        )
        metrics[name]["brier_score"] = brier

    return clf, result_encoder, encoders, metrics


def train_goals_regressor(train, val, test, target_col: str, encoders: dict):
    X_train, _ = prep_features(train, encoders=encoders)
    X_val, _ = prep_features(val, encoders=encoders)
    X_test, _ = prep_features(test, encoders=encoders)

    y_train, y_val, y_test = train[target_col], val[target_col], test[target_col]

    reg = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        early_stopping_rounds=50,
        random_state=42,
    )
    reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    metrics = {}
    for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        pred = np.clip(reg.predict(X), 1e-6, None)
        metrics[name] = {
            "poisson_deviance": mean_poisson_deviance(y, pred),
            "mae": float(np.mean(np.abs(y - pred))),
        }
    return reg, metrics


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROCESSED_DIR / "match_features_full.parquet")
    df = df.dropna(subset=FEATURE_COLS + ["result", "home_score", "away_score"])

    train, val, test = temporal_split(df)
    print(f"train={len(train)} ({train['date'].min().date()}->{train['date'].max().date()}) "
          f"val={len(val)} ({val['date'].min().date()}->{val['date'].max().date()}) "
          f"test={len(test)} ({test['date'].min().date()}->{test['date'].max().date()})")

    clf, result_encoder, encoders, clf_metrics = train_outcome_classifier(train, val, test)
    print("1X2 classifier:", json.dumps(clf_metrics, indent=2))

    reg_home, home_metrics = train_goals_regressor(train, val, test, "home_score", encoders)
    reg_away, away_metrics = train_goals_regressor(train, val, test, "away_score", encoders)
    print("home goals regressor:", json.dumps(home_metrics, indent=2))
    print("away goals regressor:", json.dumps(away_metrics, indent=2))

    clf.save_model(MODELS_DIR / "xgb_1x2.json")
    reg_home.save_model(MODELS_DIR / "xgb_home_goals.json")
    reg_away.save_model(MODELS_DIR / "xgb_away_goals.json")
    joblib.dump(
        {"result_encoder": result_encoder, "categorical_encoders": encoders},
        MODELS_DIR / "encoders.joblib",
    )

    report = {
        "result_classes": list(result_encoder.classes_),
        "feature_cols": FEATURE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "outcome_metrics": clf_metrics,
        "home_goals_metrics": home_metrics,
        "away_goals_metrics": away_metrics,
    }
    with open(REPORTS_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
