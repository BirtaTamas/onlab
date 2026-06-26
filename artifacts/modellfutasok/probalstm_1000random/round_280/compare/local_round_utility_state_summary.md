# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-fluxo-bo3-Kqy3ohBVu1ANumI6Qdn26R/eternal-fire-vs-fluxo-m2-dust2.csv`
- round_num: `3`
- rows: `105`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 105 | 1.000 | 0.951348 | 0.984539 | -0.033191 | 0 | 105 | 1.000000 | 1.000000 |
| active/recent utility | 105 | 1.000 | 0.951348 | 0.984539 | -0.033191 | 0 | 105 | 1.000000 | 1.000000 |
| strong utility action | 74 | 0.705 | 0.945694 | 0.982128 | -0.036434 | 0 | 74 | 1.000000 | 1.000000 |
| utility damage | 30 | 0.286 | 0.940939 | 0.980836 | -0.039897 | 0 | 30 | 1.000000 | 1.000000 |
| active smoke/inferno | 58 | 0.552 | 0.949790 | 0.983261 | -0.033471 | 0 | 58 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 105 | 1.000 | 0.951348 | 0.984539 | -0.033191 | 0 | 105 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `37.5s`, rows `58`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `4.0`, LSTM `0.8880`, XGBoost `0.9677`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `4.5`, LSTM `0.8888`, XGBoost `0.9671`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `5.0`, LSTM `0.8945`, XGBoost `0.9671`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.8969`, XGBoost `0.9673`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `6.0`, LSTM `0.8995`, XGBoost `0.9681`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `5.5`, LSTM `0.8989`, XGBoost `0.9671`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.9041`, XGBoost `0.9673`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.9121`, XGBoost `0.9673`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.9296`, XGBoost `0.9792`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `64.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.9191`, XGBoost `0.9673`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
