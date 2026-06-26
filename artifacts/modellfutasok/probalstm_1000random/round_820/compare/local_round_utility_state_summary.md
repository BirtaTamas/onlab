# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m2-train.csv`
- round_num: `6`
- rows: `174`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 174 | 1.000 | 0.335056 | 0.387370 | -0.052314 | 50 | 124 | 0.143678 | 0.270115 |
| active/recent utility | 174 | 1.000 | 0.335056 | 0.387370 | -0.052314 | 50 | 124 | 0.143678 | 0.270115 |
| strong utility action | 163 | 0.937 | 0.339994 | 0.388012 | -0.048018 | 50 | 113 | 0.153374 | 0.263804 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 144 | 0.828 | 0.351779 | 0.384734 | -0.032955 | 50 | 94 | 0.173611 | 0.229167 |
| recent utility last 5s | 20 | 0.115 | 0.246737 | 0.406945 | -0.160208 | 0 | 20 | 0.000000 | 0.500000 |
| flash effect present | 174 | 1.000 | 0.335056 | 0.387370 | -0.052314 | 50 | 124 | 0.143678 | 0.270115 |

## Active Smoke/Inferno Intervals

- `7.0s` - `42.0s`, rows `71`
- `50.5s` - `86.5s`, rows `73`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `5.5`, LSTM `0.3068`, XGBoost `0.5399`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.0`, LSTM `0.3039`, XGBoost `0.5369`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.5`, LSTM `0.3106`, XGBoost `0.5267`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `4.0`, LSTM `0.3007`, XGBoost `0.5131`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.5`, LSTM `0.3037`, XGBoost `0.5131`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.3058`, XGBoost `0.5144`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `11.5`, LSTM `0.0941`, XGBoost `0.3014`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `102.0`, recent_utility `0`
- seconds `3.5`, LSTM `0.3075`, XGBoost `0.5144`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.0`, LSTM `0.3250`, XGBoost `0.5267`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.3275`, XGBoost `0.5267`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
