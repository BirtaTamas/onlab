# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `33`
- rows: `210`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 210 | 1.000 | 0.430940 | 0.303276 | 0.127664 | 11 | 199 | 0.471429 | 0.852381 |
| active/recent utility | 210 | 1.000 | 0.430940 | 0.303276 | 0.127664 | 11 | 199 | 0.471429 | 0.852381 |
| strong utility action | 188 | 0.895 | 0.440796 | 0.308201 | 0.132595 | 5 | 183 | 0.473404 | 0.877660 |
| utility damage | 31 | 0.148 | 0.322058 | 0.232426 | 0.089632 | 0 | 31 | 0.645161 | 1.000000 |
| active smoke/inferno | 178 | 0.848 | 0.431299 | 0.296707 | 0.134592 | 5 | 173 | 0.500000 | 0.926966 |
| recent utility last 5s | 10 | 0.048 | 0.609848 | 0.512794 | 0.097054 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 210 | 1.000 | 0.430940 | 0.303276 | 0.127664 | 11 | 199 | 0.471429 | 0.852381 |

## Active Smoke/Inferno Intervals

- `9.0s` - `48.5s`, rows `80`
- `51.0s` - `99.5s`, rows `98`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `51.5`, LSTM `0.5930`, XGBoost `0.3408`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.5852`, XGBoost `0.3429`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.5730`, XGBoost `0.3400`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.5502`, XGBoost `0.3263`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.5585`, XGBoost `0.3400`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.5440`, XGBoost `0.3263`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.5262`, XGBoost `0.3112`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `11.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.5535`, XGBoost `0.3400`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.5529`, XGBoost `0.3399`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `26.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.5405`, XGBoost `0.3285`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
