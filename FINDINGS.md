# Accuracy investigation — what moved the needle and what didn't

This documents a focused effort to push the 1X2 model past ~0.62 accuracy,
including experiments that **failed** (kept here so they aren't re-attempted).

All experiments were evaluated on the same strict temporal split (train → val →
test by date) so the numbers are comparable. The shipped test split is matches
from **2023-03 to 2026-06 (~3,457 matches)**.

## TL;DR

- The model is **near a genuine information ceiling (~0.62 accuracy)** for
  predicting 1X2 outcomes from the available data.
- The signal is concentrated in the **results-based ratings** (MCMC attack/
  defense, Elo). An ablation showed form, head-to-head and rest-days add almost
  nothing on top of them.
- **Every external "strength" data source we tested turned out redundant** with
  those ratings, because *talent → results → ratings* already captures it.
- What *did* help (shipped): an **online Poisson rating** (better calibration),
  **GPU training**, a **seed ensemble**, and a **2002+ training window**.

## What was shipped (real improvements)

| Change | Effect |
|---|---|
| Online Poisson attack/defense rating (`features.py`) | log loss 0.810 → 0.808, Brier 0.159 → 0.158; de-stales the seasonal MCMC rating |
| 5-seed calibrated ensemble (`seed_ensemble.py`) | removes ±1pt seed-to-seed accuracy noise |
| Training window extended to 2002+ | doubles data, longer rating warm-up |
| GPU auto-detection (`train_xgboost.py`) | trains on CUDA when present (no speed win at 23k rows, future-proofing); models moved to CPU for clean inference |

Accuracy stayed flat (~0.62) — these changes improved **calibration and
robustness**, not raw accuracy, because accuracy is at its ceiling.

## What did NOT help (external data — all redundant)

| Source | How obtained | Coverage | Result |
|---|---|---|---|
| **FIFA world ranking** | free CSV ([Dato-Futbol](https://github.com/Dato-Futbol/fifa-ranking)), as-of join by date | 95% | acc 0.6225 → 0.6202 (no lift) |
| **Player talent / value** | EA FC ratings (Kaggle `stefanoleone992/ea-sports-fc-24-...`), top-23 mean overall + log value by nationality/year | 27% (2015+) | acc 0.6030 → 0.6030 (no lift) |
| **Squad "% in top-5 leagues"** | parsed Wikipedia tournament-squad articles | 24% of tournament matches | faint +2.7pt blip, did not survive — small-sample noise |

**Why they fail:** FIFA ranking, Elo, MCMC and player-talent all measure the
same underlying team strength. Once one good results-based rating is in the
model, adding more correlated strength measures is redundant — exactly as Elo
was already redundant with MCMC in the internal ablation.

## Other levers tested (in-data) — all flat

- **Model architecture**: direct 1X2 classifier vs Dixon-Coles Poisson
  score-grid vs a blend — all ~0.62; optimal blend weight ≈ 0.
- **Engineered features**: opponent-quality-adjusted form, win/unbeaten
  streaks, Elo momentum, EWM-weighted form, home/away-split form — each within
  seed noise; combined they slightly *hurt*.
- **Training window** (2002 → 2016 start): flat (0.623–0.626).
- **Elo hyperparameter tuning** (K / home-adv / margin): improves Elo's own
  log loss but not the full model (XGBoost relearns the scaling from raw
  ratings + the `neutral` flag).

## Structural insight: tournament matches are harder

Predicting **tournament** matches (World Cup, Copa América, Euro, …) has a
*lower* accuracy ceiling (~0.51–0.56) than the overall figure, because those
games are between qualified, evenly-matched teams. This is irreducible
uncertainty, not missing data — a Brazil–Argentina is genuinely less
predictable than a Brazil–Bolivia friendly. For tournaments the model's value
is in **well-calibrated probabilities**, not high top-1 accuracy.

## What remains (honest)

- The single strongest predictor in football is **bookmaker closing odds**,
  which for international matches is effectively **paid/scraped** data.
- The only free, *orthogonal* (untested) signals left are **match context**:
  venue altitude, travel distance, fixture congestion, weather. These are real
  for specific cases (e.g. altitude in La Paz) but expected to be marginal and
  match-specific, not a broad accuracy lift.
- Everything based on **team/player strength is exhausted** — it is already
  captured by the ratings.
