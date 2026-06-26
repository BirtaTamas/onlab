# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `7`
- rows: `140`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 140 | 1.000 | 0.523395 | 0.458010 | 0.065385 | 26 | 114 | 0.300000 | 0.585714 |
| active/recent utility | 140 | 1.000 | 0.523395 | 0.458010 | 0.065385 | 26 | 114 | 0.300000 | 0.585714 |
| strong utility action | 107 | 0.764 | 0.509008 | 0.407015 | 0.101993 | 8 | 99 | 0.336449 | 0.710280 |
| utility damage | 20 | 0.143 | 0.503099 | 0.412529 | 0.090569 | 0 | 20 | 0.500000 | 1.000000 |
| active smoke/inferno | 107 | 0.764 | 0.509008 | 0.407015 | 0.101993 | 8 | 99 | 0.336449 | 0.710280 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 140 | 1.000 | 0.523395 | 0.458010 | 0.065385 | 26 | 114 | 0.300000 | 0.585714 |

## Active Smoke/Inferno Intervals

- `6.0s` - `53.5s`, rows `96`
- `54.5s` - `59.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `36.5`, LSTM `0.5173`, XGBoost `0.2754`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.5212`, XGBoost `0.2795`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5609`, XGBoost `0.3201`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.5662`, XGBoost `0.3277`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.5215`, XGBoost `0.2857`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.5135`, XGBoost `0.2798`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.5239`, XGBoost `0.2924`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.5512`, XGBoost `0.3209`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5527`, XGBoost `0.3284`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.4930`, XGBoost `0.2700`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
