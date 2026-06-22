## ML Pipeline: MCMC + XGBoost Match Prediction

This fork adds a prediction pipeline on top of the dataset below. It predicts
1X2 outcomes (home win / draw / away win) and expected goals for each team,
using matches from 2015 onward.

### Approach

1. **Data cleaning** (`src/data_cleaning.py`) — filters to 2015+, normalizes
   historical team names to their current name (via `former_names.csv`),
   drops unplayed/future fixtures, dedupes.
2. **Feature engineering** (`src/features.py`) — for every match, computes
   only pre-match (leakage-safe) features: a simple Elo rating (with a
   margin-of-victory boost), rolling form (last 5/10 matches: goals for/against,
   points), rest days, head-to-head history, and tournament tier.
3. **MCMC ratings** (`src/mcmc_model.py`) — a hierarchical Bayesian model
   (PyMC/NUTS) in the spirit of Dixon-Coles: each team has an attack and
   defense strength per season, evolving as a random walk, with goals modeled
   as Poisson(attack − opponent defense + home advantage). Sampled via NUTS.
   To avoid leakage, a match in season *S* only ever sees the posterior
   rating estimated as of the end of season *S − 1* (`src/merge_mcmc_features.py`).
4. **XGBoost models** (`src/train_xgboost.py`) — a multiclass classifier for
   1X2 (`multi:softprob`) and two Poisson regressors for home/away expected
   goals, trained on a strictly temporal train/val/test split (no shuffling
   across time).
5. **Evaluation** (`src/evaluate.py`) — compares XGBoost against a
   majority-class baseline and an Elo-only baseline, plus a season-by-season
   walk-forward backtest.

### Results (test split, matches from 2024-10 to 2026-06)

| Model | Accuracy | Log loss |
|---|---|---|
| Majority-class baseline | 0.480 | 1.050 |
| Elo-only baseline | 0.584 | 0.903 |
| **XGBoost (MCMC + Elo + form features)** | **0.613** | **0.812** |

Walk-forward backtest accuracy stays in the 0.52–0.65 range across seasons
2016–2026 with no degenerate failures — see `reports/walk_forward_backtest.csv`.

### Running it

```bash
python -m venv .venv && source .venv/Scripts/activate  # or .venv/bin/activate on Linux/Mac
pip install -r requirements.txt

python src/data_cleaning.py
python src/features.py
python src/mcmc_model.py          # fits the MCMC model, ~1 min on this dataset size
python src/merge_mcmc_features.py
python src/train_xgboost.py       # trains + saves models/xgb_*.json
python src/evaluate.py            # baselines + walk-forward backtest
```

Trained XGBoost models are saved in `models/`; the full MCMC posterior
(`models/mcmc_idata.nc`, ~260MB) is gitignored — rerun `src/mcmc_model.py` to
regenerate it. Intermediate parquet files in `data/processed/` are also
gitignored and rebuilt by the scripts above.

---

### Context

Well, what happened was that I was looking for a semi-definitive easy-to-read list of international football matches and couldn't find anything decent. So I took it upon myself to collect it for my own use. I might as well share it.

### Content

This dataset includes **49,398 results of international football matches starting from the very first official match in 1872 up to 2024. The matches range from FIFA World Cup to FIFI Wild Cup to regular friendly matches. The matches are strictly men's full internationals and the data does not include Olympic Games or matches where at least one of the teams was the nation's B-team, U-23 or a league select team.

`results.csv` includes the following columns:

-   `date` - date of the match
-   `home_team` - the name of the home team
-   `away_team` - the name of the away team
-   `home_score` - full-time home team score including extra time, not including penalty-shootouts
-   `away_score` - full-time away team score including extra time, not including penalty-shootouts
-   `tournament` - the name of the tournament
-   `city` - the name of the city/town/administrative unit where the match was played
-   `country` - the name of the country where the match was played
-   `neutral` - TRUE/FALSE column indicating whether the match was played at a neutral venue

`shootouts.csv` includes the following columns:

-   `date` - date of the match
-   `home_team` - the name of the home team
-   `away_team` - the name of the away team
-   `winner` - winner of the penalty-shootout
-   `first_shooter` - the team that went first in the shootout

`goalscorers.csv` includes the following columns:

-   `date` - date of the match
-   `home_team` - the name of the home team
-   `away_team` - the name of the away team
-   `team` - name of the team scoring the goal
-   `scorer` - name of the player scoring the goal
-   `own_goal` - whether the goal was an own-goal
-   `penalty` - whether the goal was a penalty

Note on team and country names: For home and away teams the *current* name of the team has been used. For example, when in 1882 a team who called themselves Ireland played against England, in this dataset, it is called Northern Ireland because the current team of Northern Ireland is the successor of the 1882 Ireland team. This is done so it is easier to track the history and statistics of teams.

For country names, the name of the country *at the time of the match* is used. So when Ghana played in Accra, Gold Coast in the 1950s, even though the names of the home team and the country don't match, it was a home match for Ghana. This is indicated by the neutral column, which says FALSE for those matches, meaning it was **not** at a neutral venue.

### Acknowledgements

The data is gathered from several sources including but not limited to Wikipedia, rsssf.com, and individual football associations' websites.

### Inspiration

Some directions to take when exploring the data:

-   Who is the best team of all time
-   Which teams dominated different eras of football
-   What trends have there been in international football throughout the ages - home advantage, total goals scored, distribution of teams' strength etc
-   Can we say anything about geopolitics from football fixtures - how has the number of countries changed, which teams like to play each other
-   Which countries host the most matches where they themselves are not participating in
-   How much, if at all, does hosting a major tournament help a country's chances in the tournament
-   Which teams are the most active in playing friendlies and friendly tournaments - does it help or hurt them

The world's your oyster, my friend.

### Contribute

If you notice a mistake or the results are not updated fast enough for your liking, you can fix that by submitting a pull request.