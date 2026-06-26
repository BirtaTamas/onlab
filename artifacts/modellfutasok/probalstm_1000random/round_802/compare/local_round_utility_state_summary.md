# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-astralis-vs-wildcard-bo3-qSXX__H_dx2QMbEuGWf0Qb/astralis-vs-wildcard-m2-mirage.csv`
- round_num: `5`
- rows: `173`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 173 | 1.000 | 0.607080 | 0.560144 | 0.046936 | 134 | 39 | 0.890173 | 0.294798 |
| active/recent utility | 173 | 1.000 | 0.607080 | 0.560144 | 0.046936 | 134 | 39 | 0.890173 | 0.294798 |
| strong utility action | 160 | 0.925 | 0.615369 | 0.566006 | 0.049363 | 122 | 38 | 0.893750 | 0.318750 |
| utility damage | 10 | 0.058 | 0.891429 | 0.941587 | -0.050158 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 160 | 0.925 | 0.615369 | 0.566006 | 0.049363 | 122 | 38 | 0.893750 | 0.318750 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 173 | 1.000 | 0.607080 | 0.560144 | 0.046936 | 134 | 39 | 0.890173 | 0.294798 |

## Active Smoke/Inferno Intervals

- `6.5s` - `86.0s`, rows `160`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `53.0`, LSTM `0.6574`, XGBoost `0.4855`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.6407`, XGBoost `0.4687`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.6357`, XGBoost `0.4682`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.6493`, XGBoost `0.4850`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.6473`, XGBoost `0.4841`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.6470`, XGBoost `0.4857`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.6412`, XGBoost `0.4845`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.6394`, XGBoost `0.4835`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.6400`, XGBoost `0.4850`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.6399`, XGBoost `0.4857`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
