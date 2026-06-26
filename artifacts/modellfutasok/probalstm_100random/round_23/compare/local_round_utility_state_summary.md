# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-eternal-fire-vs-spirit-bo5-7H36TpK_LYGHtCXpF3Cgdr/eternal-fire-vs-spirit-m3-dust2.csv`
- round_num: `6`
- rows: `217`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 217 | 1.000 | 0.711341 | 0.766666 | -0.055325 | 69 | 148 | 0.953917 | 1.000000 |
| active/recent utility | 217 | 1.000 | 0.711341 | 0.766666 | -0.055325 | 69 | 148 | 0.953917 | 1.000000 |
| strong utility action | 160 | 0.737 | 0.711227 | 0.758451 | -0.047224 | 49 | 111 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.046 | 0.866190 | 0.952072 | -0.085882 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 157 | 0.724 | 0.710629 | 0.759087 | -0.048458 | 46 | 111 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.046 | 0.716236 | 0.719799 | -0.003563 | 4 | 6 | 1.000000 | 1.000000 |
| flash effect present | 217 | 1.000 | 0.711341 | 0.766666 | -0.055325 | 69 | 148 | 0.953917 | 1.000000 |

## Active Smoke/Inferno Intervals

- `3.0s` - `40.5s`, rows `76`
- `51.0s` - `85.5s`, rows `70`
- `95.0s` - `100.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `95.0`, LSTM `0.5039`, XGBoost `0.7669`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.7113`, XGBoost `0.9106`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.5312`, XGBoost `0.7300`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.5`, LSTM `0.5293`, XGBoost `0.7254`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.5397`, XGBoost `0.7323`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.5216`, XGBoost `0.7093`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.5318`, XGBoost `0.7194`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.6438`, XGBoost `0.8313`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.5441`, XGBoost `0.7307`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.5`, LSTM `0.5203`, XGBoost `0.7064`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
