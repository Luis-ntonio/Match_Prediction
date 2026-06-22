"""Load and clean the international football results data, scoped to 2015 onward."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = DATA_DIR / "data" / "processed"
START_DATE = "2015-01-01"


def load_former_names() -> pd.DataFrame:
    fn = pd.read_csv(DATA_DIR / "former_names.csv")
    fn["start_date"] = pd.to_datetime(fn["start_date"])
    fn["end_date"] = pd.to_datetime(fn["end_date"])
    return fn


def rename_to_current(df: pd.DataFrame, team_cols: list[str], former_names: pd.DataFrame) -> pd.DataFrame:
    """Map any historical team name still present after START_DATE to its current name.

    Renames are applied for every row regardless of date so a team's full
    history is tracked under a single identity (needed for rolling features).
    """
    rename_map = dict(zip(former_names["former"], former_names["current"]))
    for col in team_cols:
        df[col] = df[col].replace(rename_map)
    return df


def load_results() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "results.csv", encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])

    former_names = load_former_names()
    df = rename_to_current(df, ["home_team", "away_team"], former_names)

    df = df[df["date"] >= START_DATE].copy()

    # Drop fixtures without a played score (future/unplayed matches)
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df = df.drop_duplicates(subset=["date", "home_team", "away_team", "tournament"])
    df = df.sort_values("date").reset_index(drop=True)

    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["result"] = pd.cut(
        df["goal_diff"], bins=[-100, -1, 0, 100], labels=["away_win", "draw", "home_win"]
    )
    df["match_id"] = df.index

    return df


def load_goalscorers(valid_match_dates: pd.Series | None = None) -> pd.DataFrame:
    gs = pd.read_csv(DATA_DIR / "goalscorers.csv", encoding="utf-8")
    gs["date"] = pd.to_datetime(gs["date"])
    former_names = load_former_names()
    gs = rename_to_current(gs, ["home_team", "away_team", "team"], former_names)
    gs = gs[gs["date"] >= START_DATE].copy()
    return gs


def load_shootouts() -> pd.DataFrame:
    so = pd.read_csv(DATA_DIR / "shootouts.csv", encoding="utf-8")
    so["date"] = pd.to_datetime(so["date"])
    former_names = load_former_names()
    so = rename_to_current(so, ["home_team", "away_team", "winner"], former_names)
    so = so[so["date"] >= START_DATE].copy()
    return so


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    goalscorers = load_goalscorers()
    shootouts = load_shootouts()

    results.to_parquet(PROCESSED_DIR / "results_clean.parquet", index=False)
    goalscorers.to_parquet(PROCESSED_DIR / "goalscorers_clean.parquet", index=False)
    shootouts.to_parquet(PROCESSED_DIR / "shootouts_clean.parquet", index=False)

    print(f"results: {len(results)} rows, {results['date'].min().date()} -> {results['date'].max().date()}")
    print(f"teams: {pd.concat([results['home_team'], results['away_team']]).nunique()}")
    print(f"goalscorers: {len(goalscorers)} rows")
    print(f"shootouts: {len(shootouts)} rows")
    print(results["result"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
