"""Attach MCMC attack/defense ratings to each match, using only the PREVIOUS
season's posterior (the team's strength going into the season) to avoid
leaking information from the season being predicted.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = DATA_DIR / "data" / "processed"


def attach_mcmc_features(matches: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    matches["season"] = matches["date"].dt.year
    # ratings row (team, season=S) describes strength as of end of season S;
    # it becomes a feature for matches played in season S+1.
    lookup = ratings.rename(columns={"season": "rating_season"})
    lookup["season"] = lookup["rating_season"] + 1
    lookup = lookup.drop(columns=["rating_season"])

    home_lookup = lookup.rename(columns={"team": "home_team", **{
        c: f"home_{c}" for c in lookup.columns if c not in ("team", "season")
    }})
    away_lookup = lookup.rename(columns={"team": "away_team", **{
        c: f"away_{c}" for c in lookup.columns if c not in ("team", "season")
    }})

    matches = matches.merge(home_lookup, on=["home_team", "season"], how="left")
    matches = matches.merge(away_lookup, on=["away_team", "season"], how="left")

    mcmc_cols = [c for c in matches.columns if c.startswith("home_mcmc_") or c.startswith("away_mcmc_")]
    matches[mcmc_cols] = matches[mcmc_cols].fillna(0.0)

    matches["mcmc_attack_diff"] = matches["home_mcmc_attack_mean"] - matches["away_mcmc_attack_mean"]
    matches["mcmc_defense_diff"] = matches["home_mcmc_defense_mean"] - matches["away_mcmc_defense_mean"]
    matches["mcmc_net_strength_home"] = matches["home_mcmc_attack_mean"] - matches["away_mcmc_defense_mean"]
    matches["mcmc_net_strength_away"] = matches["away_mcmc_attack_mean"] - matches["home_mcmc_defense_mean"]

    return matches


def main() -> None:
    matches = pd.read_parquet(PROCESSED_DIR / "match_features.parquet")
    ratings = pd.read_parquet(PROCESSED_DIR / "mcmc_ratings.parquet")
    out = attach_mcmc_features(matches, ratings)
    out.to_parquet(PROCESSED_DIR / "match_features_full.parquet", index=False)
    print(f"Final feature table: {out.shape}")
    print(out.filter(regex="mcmc_attack_diff|mcmc_defense_diff").describe())


if __name__ == "__main__":
    main()
