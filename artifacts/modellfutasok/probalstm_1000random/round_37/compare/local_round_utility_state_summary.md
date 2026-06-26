# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-falcons-bo3-Bm3FkXiO5h_cvpKxUnOmaW/eternal-fire-vs-falcons-m1-inferno.csv`
- round_num: `19`
- rows: `213`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 213 | 1.000 | 0.565123 | 0.555836 | 0.009287 | 133 | 80 | 0.901408 | 0.924883 |
| active/recent utility | 213 | 1.000 | 0.565123 | 0.555836 | 0.009287 | 133 | 80 | 0.901408 | 0.924883 |
| strong utility action | 198 | 0.930 | 0.567838 | 0.556999 | 0.010839 | 127 | 71 | 0.914141 | 0.919192 |
| utility damage | 11 | 0.052 | 0.687615 | 0.671781 | 0.015834 | 7 | 4 | 1.000000 | 0.636364 |
| active smoke/inferno | 188 | 0.883 | 0.570957 | 0.557597 | 0.013360 | 127 | 61 | 0.936170 | 0.914894 |
| recent utility last 5s | 20 | 0.094 | 0.511989 | 0.530735 | -0.018746 | 6 | 14 | 0.700000 | 1.000000 |
| flash effect present | 213 | 1.000 | 0.565123 | 0.555836 | 0.009287 | 133 | 80 | 0.901408 | 0.924883 |

## Active Smoke/Inferno Intervals

- `11.0s` - `39.0s`, rows `57`
- `41.0s` - `106.0s`, rows `131`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `104.0`, LSTM `0.8429`, XGBoost `0.9609`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.5`, LSTM `0.8158`, XGBoost `0.9290`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.0`, LSTM `0.8006`, XGBoost `0.9102`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.8029`, XGBoost `0.9106`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.5176`, XGBoost `0.4136`, closer `lstm`, smoke `2`, inferno `4`, utility_damage `25.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.8294`, XGBoost `0.9267`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.7553`, XGBoost `0.6580`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `42.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.5082`, XGBoost `0.4136`, closer `lstm`, smoke `2`, inferno `4`, utility_damage `29.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.8382`, XGBoost `0.9267`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.8381`, XGBoost `0.9262`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
