# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `4`
- rows: `309`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 309 | 1.000 | 0.571111 | 0.559518 | 0.011592 | 156 | 153 | 0.288026 | 0.197411 |
| active/recent utility | 309 | 1.000 | 0.571111 | 0.559518 | 0.011592 | 156 | 153 | 0.288026 | 0.197411 |
| strong utility action | 248 | 0.803 | 0.575553 | 0.559412 | 0.016140 | 109 | 139 | 0.181452 | 0.205645 |
| utility damage | 20 | 0.065 | 0.509793 | 0.502554 | 0.007239 | 7 | 13 | 0.250000 | 0.450000 |
| active smoke/inferno | 248 | 0.803 | 0.575553 | 0.559412 | 0.016140 | 109 | 139 | 0.181452 | 0.205645 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 309 | 1.000 | 0.571111 | 0.559518 | 0.011592 | 156 | 153 | 0.288026 | 0.197411 |

## Active Smoke/Inferno Intervals

- `10.5s` - `60.5s`, rows `101`
- `70.5s` - `121.5s`, rows `103`
- `123.0s` - `144.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `138.5`, LSTM `0.6303`, XGBoost `0.2839`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `139.0`, LSTM `0.6551`, XGBoost `0.3693`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `116.0`, LSTM `0.5830`, XGBoost `0.4015`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `113.5`, LSTM `0.4036`, XGBoost `0.2269`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `114.0`, LSTM `0.5232`, XGBoost `0.3536`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `118.5`, LSTM `0.6324`, XGBoost `0.4635`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `115.5`, LSTM `0.5736`, XGBoost `0.4119`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `118.0`, LSTM `0.6095`, XGBoost `0.4522`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `116.5`, LSTM `0.5910`, XGBoost `0.4346`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `138.0`, LSTM `0.7954`, XGBoost `0.6416`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
