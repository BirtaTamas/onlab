# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `7`
- rows: `222`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 222 | 1.000 | 0.499235 | 0.550974 | -0.051739 | 66 | 156 | 0.418919 | 0.545045 |
| active/recent utility | 222 | 1.000 | 0.499235 | 0.550974 | -0.051739 | 66 | 156 | 0.418919 | 0.545045 |
| strong utility action | 162 | 0.730 | 0.540452 | 0.563456 | -0.023004 | 63 | 99 | 0.487654 | 0.629630 |
| utility damage | 10 | 0.045 | 0.509019 | 0.494220 | 0.014799 | 9 | 1 | 0.900000 | 0.000000 |
| active smoke/inferno | 162 | 0.730 | 0.540452 | 0.563456 | -0.023004 | 63 | 99 | 0.487654 | 0.629630 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 222 | 1.000 | 0.499235 | 0.550974 | -0.051739 | 66 | 156 | 0.418919 | 0.545045 |

## Active Smoke/Inferno Intervals

- `6.5s` - `65.0s`, rows `118`
- `67.0s` - `88.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `83.0`, LSTM `0.2756`, XGBoost `0.4389`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.5243`, XGBoost `0.6780`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `9.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.5948`, XGBoost `0.7476`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `9.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.5309`, XGBoost `0.6728`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `9.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.2984`, XGBoost `0.4389`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.5405`, XGBoost `0.6728`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `9.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.3185`, XGBoost `0.4389`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.3191`, XGBoost `0.4389`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.6244`, XGBoost `0.7400`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `3.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.5430`, XGBoost `0.4297`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
