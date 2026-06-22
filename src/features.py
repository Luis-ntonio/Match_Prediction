"""Build leakage-safe, pre-match features for every fixture.

All rolling/Elo/head-to-head features are computed using only information
available strictly *before* kickoff (shift(1) / chronological iteration),
so they are safe to use for both training and live inference.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = DATA_DIR / "data" / "processed"

ELO_K = 20
ELO_HOME_ADV = 50
ELO_INITIAL = 1500

TOURNAMENT_TIERS = {
    "FIFA World Cup": "world_cup",
    "FIFA World Cup qualification": "wc_qualifier",
    "UEFA Euro": "continental_final",
    "UEFA Euro qualification": "continental_qualifier",
    "Copa América": "continental_final",
    "African Cup of Nations": "continental_final",
    "African Cup of Nations qualification": "continental_qualifier",
    "AFC Asian Cup": "continental_final",
    "AFC Asian Cup qualification": "continental_qualifier",
    "Gold Cup": "continental_final",
    "UEFA Nations League": "nations_league",
    "CONCACAF Nations League": "nations_league",
    "CONCACAF Nations League qualification": "nations_league",
    "Friendly": "friendly",
}


def tournament_tier(name: str) -> str:
    return TOURNAMENT_TIERS.get(name, "other")


def _to_long(results: pd.DataFrame) -> pd.DataFrame:
    home = results.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_score": "goals_for",
            "away_score": "goals_against",
        }
    ).copy()
    home["is_home"] = 1

    away = results.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_score": "goals_for",
            "home_score": "goals_against",
        }
    ).copy()
    away["is_home"] = 0

    long_df = pd.concat([home, away], ignore_index=True)
    long_df["points"] = np.select(
        [long_df["goals_for"] > long_df["goals_against"], long_df["goals_for"] == long_df["goals_against"]],
        [3, 1],
        default=0,
    )
    long_df = long_df.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    return long_df


def add_rolling_form(long_df: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    grp = long_df.groupby("team", group_keys=False)
    for w in windows:
        long_df[f"form_goals_for_{w}"] = grp["goals_for"].apply(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        long_df[f"form_goals_against_{w}"] = grp["goals_against"].apply(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        long_df[f"form_points_{w}"] = grp["points"].apply(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
    long_df["rest_days"] = grp["date"].diff().dt.days
    long_df["matches_played"] = grp.cumcount()
    return long_df


def add_elo(results: pd.DataFrame) -> pd.DataFrame:
    """Iterate matches chronologically once, maintaining a single Elo dict."""
    elo = {}
    pre_elo_home = np.empty(len(results))
    pre_elo_away = np.empty(len(results))

    ordered = results.sort_values(["date", "match_id"])
    for i, row in ordered.iterrows():
        h, a = row["home_team"], row["away_team"]
        elo_h = elo.get(h, ELO_INITIAL)
        elo_a = elo.get(a, ELO_INITIAL)
        pre_elo_home[i] = elo_h
        pre_elo_away[i] = elo_a

        adj_h = elo_h + (0 if row["neutral"] else ELO_HOME_ADV)
        expected_h = 1 / (1 + 10 ** ((elo_a - adj_h) / 400))

        if row["home_score"] > row["away_score"]:
            score_h = 1.0
        elif row["home_score"] == row["away_score"]:
            score_h = 0.5
        else:
            score_h = 0.0

        margin = abs(row["home_score"] - row["away_score"])
        margin_mult = np.log(margin + 1) + 1  # Dixon-Coles style margin-of-victory boost

        delta = ELO_K * margin_mult * (score_h - expected_h)
        elo[h] = elo_h + delta
        elo[a] = elo_a - delta

    results = results.copy()
    results["elo_home_pre"] = pre_elo_home
    results["elo_away_pre"] = pre_elo_away
    results["elo_diff"] = results["elo_home_pre"] - results["elo_away_pre"]
    return results


ONLINE_LR = 0.02
ONLINE_MU = 0.2
ONLINE_HOME_ADV = 0.25


def _run_online_poisson(results, lr, mu, home_adv):
    """Single chronological pass of the online Poisson attack/defense updates.

    Returns the per-match pre-match state array plus the final att/def dicts
    (state after every match has been seen) so both the feature builder and
    live inference can share one implementation.
    """
    ordered = results.sort_values(["date", "match_id"])
    h = ordered["home_team"].to_numpy()
    a = ordered["away_team"].to_numpy()
    hs = ordered["home_score"].to_numpy().astype(float)
    as_ = ordered["away_score"].to_numpy().astype(float)
    neutral = ordered["neutral"].to_numpy()

    att: dict[str, float] = {}
    dfn: dict[str, float] = {}
    pre = np.zeros((len(ordered), 4))  # h_att, h_def, a_att, a_def (pre-match)

    for i in range(len(ordered)):
        ah = att.get(h[i], 0.0); dh = dfn.get(h[i], 0.0)
        aa = att.get(a[i], 0.0); da = dfn.get(a[i], 0.0)
        pre[i] = [ah, dh, aa, da]

        ha = 0.0 if neutral[i] else home_adv
        lam_h = np.exp(mu + ha + ah - da)
        lam_a = np.exp(mu + aa - dh)
        # gradient of Poisson NLL wrt each linear-predictor term = (lambda - goals)
        g_h = lam_h - hs[i]
        g_a = lam_a - as_[i]
        att[h[i]] = ah - lr * g_h
        dfn[a[i]] = da + lr * g_h
        att[a[i]] = aa - lr * g_a
        dfn[h[i]] = dh + lr * g_a

    return pre, att, dfn, ordered["match_id"].to_numpy()


def latest_online_rating(results, lr=ONLINE_LR, mu=ONLINE_MU, home_adv=ONLINE_HOME_ADV) -> dict:
    """Final (post-all-matches) online att/def per team, for live inference."""
    _, att, dfn, _ = _run_online_poisson(results, lr, mu, home_adv)
    return {"att": att, "def": dfn}


def add_online_poisson_rating(
    results: pd.DataFrame, lr=ONLINE_LR, mu=ONLINE_MU, home_adv=ONLINE_HOME_ADV
) -> pd.DataFrame:
    """Online attack/defense ratings updated every match via one SGD step on the
    Poisson NLL of the observed goals (log-rate = mu + home_adv + att - opp_def).

    Unlike the seasonal MCMC ratings (which freeze a team's strength to the end
    of the *previous* season and so can be up to a year stale), this rating is
    always current: each match only ever sees the pre-match state, so it stays
    leakage-safe while reacting to recent results. It complements MCMC — the
    ablation showed ratings carry essentially all the signal, and a fresher
    rating measurably improves probability calibration (log loss).
    """
    pre, _att, _dfn, match_ids = _run_online_poisson(results, lr, mu, home_adv)

    out = pd.DataFrame(
        {
            "match_id": match_ids,
            "online_att_home": pre[:, 0],
            "online_def_home": pre[:, 1],
            "online_att_away": pre[:, 2],
            "online_def_away": pre[:, 3],
        }
    )
    results = results.merge(out, on="match_id")
    results["online_net_home"] = results["online_att_home"] - results["online_def_away"]
    results["online_net_away"] = results["online_att_away"] - results["online_def_home"]
    results["online_att_diff"] = results["online_att_home"] - results["online_att_away"]
    results["online_def_diff"] = results["online_def_home"] - results["online_def_away"]
    return results


def add_head_to_head(results: pd.DataFrame) -> pd.DataFrame:
    results = results.sort_values(["date", "match_id"]).reset_index(drop=True)
    pair_key = results.apply(lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1)
    results["_pair_key"] = pair_key

    h2h_matches = np.zeros(len(results))
    h2h_home_win_rate = np.full(len(results), 0.5)

    history: dict[tuple, list[tuple[str, str]]] = {}
    for idx, row in results.iterrows():
        key = row["_pair_key"]
        past = history.get(key, [])
        h2h_matches[idx] = len(past)
        if past:
            wins_for_home_team = sum(1 for winner in past if winner == row["home_team"])
            h2h_home_win_rate[idx] = wins_for_home_team / len(past)

        if row["home_score"] > row["away_score"]:
            winner = row["home_team"]
        elif row["home_score"] < row["away_score"]:
            winner = row["away_team"]
        else:
            winner = "draw"
        history.setdefault(key, []).append(winner)

    results["h2h_matches"] = h2h_matches
    results["h2h_home_win_rate"] = h2h_home_win_rate
    results = results.drop(columns=["_pair_key"])
    return results


def build_match_features(results: pd.DataFrame) -> pd.DataFrame:
    results = add_elo(results)
    results = add_online_poisson_rating(results)
    results = add_head_to_head(results)
    results["tournament_tier"] = results["tournament"].map(tournament_tier)

    long_df = _to_long(results)
    long_df = add_rolling_form(long_df)

    feature_cols = [c for c in long_df.columns if c.startswith("form_") or c in ("rest_days", "matches_played")]
    home_feats = long_df[long_df["is_home"] == 1][["match_id", "team"] + feature_cols]
    away_feats = long_df[long_df["is_home"] == 0][["match_id", "team"] + feature_cols]

    home_feats = home_feats.rename(columns={c: f"home_{c}" for c in feature_cols}).drop(columns="team")
    away_feats = away_feats.rename(columns={c: f"away_{c}" for c in feature_cols}).drop(columns="team")

    df = results.merge(home_feats, on="match_id").merge(away_feats, on="match_id")

    df["form_points_diff_5"] = df["home_form_points_5"] - df["away_form_points_5"]
    df["form_goal_diff_5"] = (df["home_form_goals_for_5"] - df["home_form_goals_against_5"]) - (
        df["away_form_goals_for_5"] - df["away_form_goals_against_5"]
    )
    df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]

    return df


def main() -> None:
    results = pd.read_parquet(PROCESSED_DIR / "results_clean.parquet")
    feats = build_match_features(results)
    feats.to_parquet(PROCESSED_DIR / "match_features.parquet", index=False)
    print(f"Built {feats.shape[1]} columns for {len(feats)} matches")
    print(feats.filter(regex="elo_diff|form_points_diff|h2h_matches").describe())


if __name__ == "__main__":
    main()
