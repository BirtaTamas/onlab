# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-m80-bo3-mWbCj4SBCT3wH-l62HcQgw/furia-vs-m80-m1-mirage.csv`
- round_num: `3`
- rows: `198`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 198 | 1.000 | 0.940365 | 0.981886 | -0.041521 | 0 | 198 | 1.000000 | 1.000000 |
| active/recent utility | 198 | 1.000 | 0.940365 | 0.981886 | -0.041521 | 0 | 198 | 1.000000 | 1.000000 |
| strong utility action | 150 | 0.758 | 0.934913 | 0.980876 | -0.045962 | 0 | 150 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.101 | 0.924043 | 0.982314 | -0.058271 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 140 | 0.707 | 0.933750 | 0.981859 | -0.048110 | 0 | 140 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.051 | 0.951201 | 0.967103 | -0.015902 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 198 | 1.000 | 0.940365 | 0.981886 | -0.041521 | 0 | 198 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `48.5s`, rows `84`
- `59.0s` - `81.0s`, rows `45`
- `88.5s` - `93.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `29.0`, LSTM `0.8997`, XGBoost `0.9833`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.9030`, XGBoost `0.9834`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.9036`, XGBoost `0.9834`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.9039`, XGBoost `0.9834`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.9046`, XGBoost `0.9834`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.9060`, XGBoost `0.9833`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.9062`, XGBoost `0.9833`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.9066`, XGBoost `0.9834`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.9077`, XGBoost `0.9833`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.9080`, XGBoost `0.9834`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
