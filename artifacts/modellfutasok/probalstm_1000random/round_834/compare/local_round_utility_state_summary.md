# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-heroic-vs-aurora-bo3-QigxwcikBDdlIOkrYDpY7y/heroic-vs-aurora-m2-dust2.csv`
- round_num: `20`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.590761 | 0.689032 | -0.098271 | 0 | 230 | 0.386957 | 0.626087 |
| active/recent utility | 230 | 1.000 | 0.590761 | 0.689032 | -0.098271 | 0 | 230 | 0.386957 | 0.626087 |
| strong utility action | 167 | 0.726 | 0.490166 | 0.609521 | -0.119355 | 0 | 167 | 0.227545 | 0.502994 |
| utility damage | 10 | 0.043 | 0.936051 | 0.967025 | -0.030975 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 161 | 0.700 | 0.472086 | 0.595204 | -0.123118 | 0 | 161 | 0.198758 | 0.484472 |
| recent utility last 5s | 10 | 0.043 | 0.976205 | 0.993603 | -0.017398 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.590761 | 0.689032 | -0.098271 | 0 | 230 | 0.386957 | 0.626087 |

## Active Smoke/Inferno Intervals

- `6.0s` - `86.0s`, rows `161`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `69.5`, LSTM `0.4288`, XGBoost `0.6725`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.2973`, XGBoost `0.5045`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.2999`, XGBoost `0.5037`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.3015`, XGBoost `0.5037`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.3019`, XGBoost `0.5016`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.3008`, XGBoost `0.5002`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.3008`, XGBoost `0.5002`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.3011`, XGBoost `0.5002`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.4985`, XGBoost `0.6934`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.3078`, XGBoost `0.5002`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
