# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `16`
- rows: `156`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 156 | 1.000 | 0.079234 | 0.072904 | 0.006330 | 97 | 59 | 1.000000 | 1.000000 |
| active/recent utility | 156 | 1.000 | 0.079234 | 0.072904 | 0.006330 | 97 | 59 | 1.000000 | 1.000000 |
| strong utility action | 114 | 0.731 | 0.093510 | 0.084346 | 0.009165 | 70 | 44 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 114 | 0.731 | 0.093510 | 0.084346 | 0.009165 | 70 | 44 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.064 | 0.053569 | 0.086857 | -0.033288 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 156 | 1.000 | 0.079234 | 0.072904 | 0.006330 | 97 | 59 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `3.0s` - `34.0s`, rows `63`
- `44.5s` - `69.5s`, rows `51`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.0`, LSTM `0.3415`, XGBoost `0.2148`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.3325`, XGBoost `0.2128`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.3307`, XGBoost `0.2205`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.3158`, XGBoost `0.2070`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.3218`, XGBoost `0.2182`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.3209`, XGBoost `0.2182`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.3195`, XGBoost `0.2215`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.3141`, XGBoost `0.2182`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.3086`, XGBoost `0.2170`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.3054`, XGBoost `0.2182`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
