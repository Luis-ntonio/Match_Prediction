"""Predict 1X2 probabilities and expected goals for a hypothetical fixture
between two teams, using each team's most recent known form/Elo/MCMC rating.

Usage:
    python src/predict.py "Argentina" "Brazil" --neutral
    python src/predict.py "France" "England" --tournament "FIFA World Cup"
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from features import add_elo, tournament_tier
from train_xgboost import FEATURE_COLS

DATA_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = DATA_DIR / "data" / "processed"
MODELS_DIR = DATA_DIR / "models"


def load_models():
    calibrated_clf = joblib.load(MODELS_DIR / "xgb_1x2_calibrated.joblib")
    reg_home = XGBRegressor()
    reg_home.load_model(MODELS_DIR / "xgb_home_goals.json")
    reg_away = XGBRegressor()
    reg_away.load_model(MODELS_DIR / "xgb_away_goals.json")
    encoders = joblib.load(MODELS_DIR / "encoders.joblib")
    return calibrated_clf, reg_home, reg_away, encoders


def latest_team_form(results: pd.DataFrame, team: str, window: int) -> dict:
    mask = (results["home_team"] == team) | (results["away_team"] == team)
    team_games = results[mask].sort_values("date")
    games = team_games.tail(window)
    if games.empty:
        return {"goals_for": 0.0, "goals_against": 0.0, "points": 0.0, "rest_days": 30.0, "matches_played": 0}

    goals_for, goals_against, points = [], [], []
    for _, g in games.iterrows():
        is_home = g["home_team"] == team
        gf = g["home_score"] if is_home else g["away_score"]
        ga = g["away_score"] if is_home else g["home_score"]
        goals_for.append(gf)
        goals_against.append(ga)
        points.append(3 if gf > ga else (1 if gf == ga else 0))

    last_date = games["date"].max()
    rest_days = (pd.Timestamp.now().normalize() - last_date).days

    return {
        "goals_for": float(np.mean(goals_for)),
        "goals_against": float(np.mean(goals_against)),
        "points": float(np.mean(points)),
        "rest_days": float(rest_days),
        "matches_played": int(len(team_games)),
    }


def latest_elo(results: pd.DataFrame) -> dict:
    """Final post-match Elo per team, derived from the same chronological
    update rule used in features.add_elo (run on the full match history).
    """
    results_with_elo = add_elo(results)
    elo = {}
    for _, row in results_with_elo.sort_values("date").iterrows():
        elo[row["home_team"]] = row["elo_home_pre"]
        elo[row["away_team"]] = row["elo_away_pre"]
    return elo


def latest_mcmc_rating(ratings: pd.DataFrame, team: str) -> dict:
    team_ratings = ratings[ratings["team"] == team]
    if team_ratings.empty:
        return {"attack_mean": 0.0, "attack_sd": 1.0, "defense_mean": 0.0, "defense_sd": 1.0}
    last = team_ratings.sort_values("season").iloc[-1]
    return {
        "attack_mean": float(last["mcmc_attack_mean"]),
        "attack_sd": float(last["mcmc_attack_sd"]),
        "defense_mean": float(last["mcmc_defense_mean"]),
        "defense_sd": float(last["mcmc_defense_sd"]),
    }


def head_to_head_stats(results: pd.DataFrame, home_team: str, away_team: str) -> dict:
    pair_mask = (
        ((results["home_team"] == home_team) & (results["away_team"] == away_team))
        | ((results["home_team"] == away_team) & (results["away_team"] == home_team))
    )
    past = results[pair_mask]
    if past.empty:
        return {"matches": 0, "home_win_rate": 0.5}
    wins_for_home_team = (
        ((past["home_team"] == home_team) & (past["home_score"] > past["away_score"]))
        | ((past["away_team"] == home_team) & (past["away_score"] > past["home_score"]))
    ).sum()
    return {"matches": int(len(past)), "home_win_rate": float(wins_for_home_team / len(past))}


def build_feature_row(
    home_team: str, away_team: str, tournament: str, neutral: bool,
    results: pd.DataFrame, ratings: pd.DataFrame, elo: dict,
) -> pd.DataFrame:
    home_form5 = latest_team_form(results, home_team, 5)
    home_form10 = latest_team_form(results, home_team, 10)
    away_form5 = latest_team_form(results, away_team, 5)
    away_form10 = latest_team_form(results, away_team, 10)

    home_mcmc = latest_mcmc_rating(ratings, home_team)
    away_mcmc = latest_mcmc_rating(ratings, away_team)
    h2h = head_to_head_stats(results, home_team, away_team)

    elo_home = elo.get(home_team, 1500.0)
    elo_away = elo.get(away_team, 1500.0)

    row = {
        "elo_diff": elo_home - elo_away,
        "elo_home_pre": elo_home,
        "elo_away_pre": elo_away,
        "h2h_matches": h2h["matches"],
        "h2h_home_win_rate": h2h["home_win_rate"],
        "home_form_goals_for_5": home_form5["goals_for"],
        "home_form_goals_against_5": home_form5["goals_against"],
        "home_form_points_5": home_form5["points"],
        "away_form_goals_for_5": away_form5["goals_for"],
        "away_form_goals_against_5": away_form5["goals_against"],
        "away_form_points_5": away_form5["points"],
        "home_form_goals_for_10": home_form10["goals_for"],
        "home_form_goals_against_10": home_form10["goals_against"],
        "home_form_points_10": home_form10["points"],
        "away_form_goals_for_10": away_form10["goals_for"],
        "away_form_goals_against_10": away_form10["goals_against"],
        "away_form_points_10": away_form10["points"],
        "form_points_diff_5": home_form5["points"] - away_form5["points"],
        "form_goal_diff_5": (home_form5["goals_for"] - home_form5["goals_against"])
        - (away_form5["goals_for"] - away_form5["goals_against"]),
        "home_rest_days": home_form5["rest_days"],
        "away_rest_days": away_form5["rest_days"],
        "rest_days_diff": home_form5["rest_days"] - away_form5["rest_days"],
        "home_matches_played": home_form5["matches_played"],
        "away_matches_played": away_form5["matches_played"],
        "home_mcmc_attack_mean": home_mcmc["attack_mean"],
        "home_mcmc_attack_sd": home_mcmc["attack_sd"],
        "home_mcmc_defense_mean": home_mcmc["defense_mean"],
        "home_mcmc_defense_sd": home_mcmc["defense_sd"],
        "away_mcmc_attack_mean": away_mcmc["attack_mean"],
        "away_mcmc_attack_sd": away_mcmc["attack_sd"],
        "away_mcmc_defense_mean": away_mcmc["defense_mean"],
        "away_mcmc_defense_sd": away_mcmc["defense_sd"],
        "mcmc_attack_diff": home_mcmc["attack_mean"] - away_mcmc["attack_mean"],
        "mcmc_defense_diff": home_mcmc["defense_mean"] - away_mcmc["defense_mean"],
        "mcmc_net_strength_home": home_mcmc["attack_mean"] - away_mcmc["defense_mean"],
        "mcmc_net_strength_away": away_mcmc["attack_mean"] - home_mcmc["defense_mean"],
        "neutral": int(neutral),
        "tournament_tier": tournament_tier(tournament),
    }
    return pd.DataFrame([row])


def predict_match(home_team: str, away_team: str, tournament: str = "Friendly", neutral: bool = False):
    results = pd.read_parquet(PROCESSED_DIR / "results_clean.parquet")
    ratings = pd.read_parquet(PROCESSED_DIR / "mcmc_ratings.parquet")
    elo = latest_elo(results)

    calibrated_clf, reg_home, reg_away, encoders = load_models()
    result_encoder = encoders["result_encoder"]
    tier_encoder = encoders["categorical_encoders"]["tournament_tier"]

    row = build_feature_row(home_team, away_team, tournament, neutral, results, ratings, elo)

    tier_value = row["tournament_tier"].iloc[0]
    if tier_value not in set(tier_encoder.classes_):
        tier_value = tier_encoder.classes_[0]
    row["tournament_tier_enc"] = tier_encoder.transform([tier_value])[0]

    X = row[FEATURE_COLS + ["tournament_tier_enc"]].copy()
    X["neutral"] = X["neutral"].astype(int)

    proba = calibrated_clf.predict_proba(X)[0]
    classes = list(result_encoder.classes_)

    exp_home_goals = float(reg_home.predict(X)[0])
    exp_away_goals = float(reg_away.predict(X)[0])

    return {
        "home_team": home_team,
        "away_team": away_team,
        "p_home_win": float(proba[classes.index("home_win")]),
        "p_draw": float(proba[classes.index("draw")]),
        "p_away_win": float(proba[classes.index("away_win")]),
        "expected_home_goals": exp_home_goals,
        "expected_away_goals": exp_away_goals,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--tournament", default="Friendly")
    parser.add_argument("--neutral", action="store_true")
    args = parser.parse_args()

    result = predict_match(args.home_team, args.away_team, args.tournament, args.neutral)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
