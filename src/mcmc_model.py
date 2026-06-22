"""Hierarchical Bayesian (Dixon-Coles style) attack/defense ratings via MCMC (PyMC/NUTS).

Each team has an attack and defense strength that evolves season-to-season as a
random walk, fit jointly across all teams and seasons in one model. Goals are
modeled as Poisson with rate driven by attack/defense/home-advantage.

To stay leakage-safe for downstream ML, we only ever expose to a match the
posterior rating estimated as of the END of the PREVIOUS season (i.e. a
team's strength going into the season, not estimated using games from the
season being predicted).
"""
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

DATA_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = DATA_DIR / "data" / "processed"
MODELS_DIR = DATA_DIR / "models"


def build_model(results: pd.DataFrame, teams: list[str], seasons: list[int]):
    n_teams = len(teams)
    n_seasons = len(seasons)
    team_idx = {t: i for i, t in enumerate(teams)}
    season_idx = {s: i for i, s in enumerate(seasons)}

    home_team = results["home_team"].map(team_idx).to_numpy()
    away_team = results["away_team"].map(team_idx).to_numpy()
    season = results["season"].map(season_idx).to_numpy()
    home_goals = results["home_score"].to_numpy()
    away_goals = results["away_score"].to_numpy()
    is_neutral = results["neutral"].to_numpy().astype(float)

    with pm.Model() as model:
        mu_goals = pm.Normal("mu_goals", mu=0.3, sigma=1)
        home_adv = pm.Normal("home_adv", mu=0.2, sigma=0.5)

        sigma_attack0 = pm.HalfNormal("sigma_attack0", sigma=1)
        sigma_defense0 = pm.HalfNormal("sigma_defense0", sigma=1)
        sigma_attack_rw = pm.HalfNormal("sigma_attack_rw", sigma=0.3)
        sigma_defense_rw = pm.HalfNormal("sigma_defense_rw", sigma=0.3)

        eps_attack = pm.Normal("eps_attack", mu=0, sigma=1, shape=(n_teams, n_seasons))
        eps_defense = pm.Normal("eps_defense", mu=0, sigma=1, shape=(n_teams, n_seasons))

        scale_attack = pt.concatenate(
            [pt.ones((n_teams, 1)) * sigma_attack0, pt.ones((n_teams, n_seasons - 1)) * sigma_attack_rw],
            axis=1,
        )
        scale_defense = pt.concatenate(
            [pt.ones((n_teams, 1)) * sigma_defense0, pt.ones((n_teams, n_seasons - 1)) * sigma_defense_rw],
            axis=1,
        )

        attack_steps = eps_attack * scale_attack
        defense_steps = eps_defense * scale_defense

        attack = pm.Deterministic("attack", pt.cumsum(attack_steps, axis=1))
        defense = pm.Deterministic("defense", pt.cumsum(defense_steps, axis=1))

        log_rate_home = (
            mu_goals + home_adv * (1 - is_neutral) + attack[home_team, season] - defense[away_team, season]
        )
        log_rate_away = mu_goals + attack[away_team, season] - defense[home_team, season]

        pm.Poisson("home_goals_obs", mu=pt.exp(log_rate_home), observed=home_goals)
        pm.Poisson("away_goals_obs", mu=pt.exp(log_rate_away), observed=away_goals)

    return model


def fit(results: pd.DataFrame, draws=600, tune=600, chains=4, target_accept=0.9):
    teams = sorted(set(results["home_team"]) | set(results["away_team"]))
    seasons = sorted(results["season"].unique())

    model = build_model(results, teams, seasons)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=42,
            progressbar=True,
        )
    return idata, teams, seasons


def extract_ratings(idata, teams: list[str], seasons: list[int]) -> pd.DataFrame:
    attack_mean = idata.posterior["attack"].mean(dim=("chain", "draw")).to_numpy()
    attack_sd = idata.posterior["attack"].std(dim=("chain", "draw")).to_numpy()
    defense_mean = idata.posterior["defense"].mean(dim=("chain", "draw")).to_numpy()
    defense_sd = idata.posterior["defense"].std(dim=("chain", "draw")).to_numpy()

    rows = []
    for ti, team in enumerate(teams):
        for si, season in enumerate(seasons):
            rows.append(
                {
                    "team": team,
                    "season": season,
                    "mcmc_attack_mean": attack_mean[ti, si],
                    "mcmc_attack_sd": attack_sd[ti, si],
                    "mcmc_defense_mean": defense_mean[ti, si],
                    "mcmc_defense_sd": defense_sd[ti, si],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_parquet(PROCESSED_DIR / "results_clean.parquet")
    results["season"] = results["date"].dt.year

    idata, teams, seasons = fit(results)

    print(az.summary(idata, var_names=["mu_goals", "home_adv", "sigma_attack0", "sigma_defense0",
                                        "sigma_attack_rw", "sigma_defense_rw"]))

    ratings = extract_ratings(idata, teams, seasons)
    ratings.to_parquet(PROCESSED_DIR / "mcmc_ratings.parquet", index=False)
    idata.to_netcdf(MODELS_DIR / "mcmc_idata.nc")
    print(f"Saved ratings for {len(teams)} teams x {len(seasons)} seasons")


if __name__ == "__main__":
    main()
