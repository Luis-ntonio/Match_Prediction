## ML Pipeline: MCMC + XGBoost Match Prediction

This fork adds a prediction pipeline on top of the dataset below. It predicts
1X2 outcomes (home win / draw / away win) and expected goals for each team,
using matches from 2002 onward (extended from an initial 2015+ cutoff to give
Elo/MCMC ratings a longer warm-up and roughly double the training data).

### Approach

1. **Data cleaning** (`src/data_cleaning.py`) — filters to 2002+, normalizes
   historical team names to their current name (via `former_names.csv`),
   drops unplayed/future fixtures, dedupes.
2. **Feature engineering** (`src/features.py`) — for every match, computes
   only pre-match (leakage-safe) features: a simple Elo rating (with a
   margin-of-victory boost), rolling form (last 5/10 matches: goals for/against,
   points), rest days, head-to-head history, tournament tier, and an **online
   Poisson attack/defense rating** updated every match via one SGD step on the
   goals likelihood. The online rating complements the seasonal MCMC ratings
   below: it reacts to recent results instead of being frozen to the previous
   season, which measurably improves probability calibration (see Results).
3. **MCMC ratings** (`src/mcmc_model.py`) — a hierarchical Bayesian model
   (PyMC/NUTS) in the spirit of Dixon-Coles: each team has an attack and
   defense strength per season, evolving as a random walk, with goals modeled
   as Poisson(attack − opponent defense + home advantage). Sampled via NUTS.
   To avoid leakage, a match in season *S* only ever sees the posterior
   rating estimated as of the end of season *S − 1* (`src/merge_mcmc_features.py`).
   A feature ablation showed these ratings carry essentially all of the model's
   signal — Elo, form and head-to-head add little on top of them.
4. **XGBoost models** (`src/train_xgboost.py`) — a multiclass classifier for
   1X2 (`multi:softprob`) and two Poisson regressors for home/away expected
   goals, trained on a strictly temporal train/val/test split (no shuffling
   across time). Training runs on GPU automatically when a CUDA device is
   available (`device="cuda"`, auto-detected with CPU fallback); models are
   switched back to CPU for clean single-row inference. The shipped classifier
   is probability-calibrated via 5-fold
   `CalibratedClassifierCV` (isotonic) fit on train+val — a single-split
   calibration (fit on val alone) was tried first and *hurt* held-out test
   log-loss/Brier; cross-validated calibration over more data measurably
   helped instead (see Results below). On top of that, the shipped model
   averages predictions from `CALIBRATED_SEED_ENSEMBLE` (5) independent
   refits that differ only in `random_state` (`SeedEnsembleClassifier` in
   `src/seed_ensemble.py`) — a single seed's test accuracy can swing by a
   point or more from training randomness alone, and averaging seeds trades
   that variance away for free.
5. **Evaluation** (`src/evaluate.py`) — compares XGBoost against a
   majority-class baseline and an Elo-only baseline, plus a season-by-season
   walk-forward backtest. **Predict a single fixture** with `src/predict.py`
   (see Usage below).

### Results (test split, matches from 2023-03 to 2026-06, 3,457 matches)

| Model | Accuracy | Log loss | Brier |
|---|---|---|---|
| Majority-class baseline | 0.473 | 1.054 | — |
| Elo-only baseline | 0.596 | 0.895 | — |
| XGBoost, raw (uncalibrated, single seed) | 0.621 | 0.812 | 0.159 |
| **XGBoost, calibrated + 5-seed ensemble (shipped model)** | **0.621** | **0.808** | **0.158** |

The online Poisson rating improved held-out log loss (0.810 → 0.808) and
Brier (0.159 → 0.158) over the previous version; accuracy is unchanged.

### On the accuracy ceiling

Accuracy on this task sits at a genuine plateau of **~0.62** that several
independent levers fail to break, each tested on the same fixed temporal split:

- **Model architecture** — a direct 1X2 classifier, a Dixon-Coles Poisson
  score-grid derived from the goals models, and a blend of the two all land at
  ~0.62; the optimal blend weight on the score grid is ~0.
- **Engineered features** — quality-of-opponent-adjusted form, win/unbeaten
  streaks, Elo momentum, recency-weighted (EWM) form and home/away-split form
  each move accuracy by less than the seed-to-seed noise band; added together
  they slightly *hurt*.
- **Training window** — sweeping the start year from 2002 to 2016 keeps
  accuracy within ~0.623–0.626 (flat).
- **Elo tuning** — grid-searching K / home-advantage / margin improves Elo's
  *own* predictive log loss, but feeding the retuned ratings to XGBoost does
  not move the full model (it already learns the optimal scaling from the raw
  ratings plus the `neutral` flag).
- **Draws are structurally hard** — the model predicts a draw as the most
  likely outcome only ~2% of the time though draws are ~23% of matches; this is
  expected (a draw is rarely any single match's modal outcome) and is the main
  irreducible accuracy loss.

The reason: with only match results as input, the rating features already
extract most of the available signal. We then went further and **tested the
obvious external data sources** — FIFA world ranking, EA FC player talent/value
aggregated by nationality, and Wikipedia tournament-squad composition — and
**all of them proved redundant** with the existing MCMC/Elo ratings (talent →
results → ratings already captures team strength). The only thing known to beat
this ceiling is bookmaker closing odds, which for international matches is
effectively paid data. See [FINDINGS.md](FINDINGS.md) for the full set of
experiments (with numbers), including the ones that failed.

Walk-forward backtest accuracy ranges 0.56–0.66 for seasons with a
meaningful training history (2006 onward; earlier seasons are a cold-start
warm-up for the ratings) — see `reports/walk_forward_backtest.csv`.

### Predicting a single fixture

```bash
python src/predict.py "Argentina" "Brazil" --neutral --tournament "Copa América"
python src/predict.py "France" "England" --tournament "FIFA World Cup"
```

Uses each team's latest known Elo/form/MCMC rating to build the feature row
and the calibrated classifier + goals regressors to predict.

### Running it

```bash
python -m venv .venv && source .venv/Scripts/activate  # or .venv/bin/activate on Linux/Mac
pip install -r requirements.txt

python src/data_cleaning.py
python src/features.py
python src/mcmc_model.py          # fits the MCMC model, ~2-3 min on this dataset size
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