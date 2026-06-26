# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-3dmax-vs-m80-bo3-DeIrLPYSKhgd10M8zQmUUV/3dmax-vs-m80-m2-train.csv`
- round_num: `16`
- rows: `148`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 148 | 1.000 | 0.756067 | 0.764753 | -0.008686 | 54 | 94 | 0.979730 | 0.932432 |
| active/recent utility | 148 | 1.000 | 0.756067 | 0.764753 | -0.008686 | 54 | 94 | 0.979730 | 0.932432 |
| strong utility action | 110 | 0.743 | 0.730898 | 0.742371 | -0.011473 | 39 | 71 | 0.972727 | 0.909091 |
| utility damage | 17 | 0.115 | 0.637383 | 0.636549 | 0.000834 | 4 | 13 | 1.000000 | 1.000000 |
| active smoke/inferno | 110 | 0.743 | 0.730898 | 0.742371 | -0.011473 | 39 | 71 | 0.972727 | 0.909091 |
| recent utility last 5s | 10 | 0.068 | 0.564283 | 0.623206 | -0.058924 | 1 | 9 | 1.000000 | 1.000000 |
| flash effect present | 148 | 1.000 | 0.756067 | 0.764753 | -0.008686 | 54 | 94 | 0.979730 | 0.932432 |

## Active Smoke/Inferno Intervals

- `7.5s` - `62.0s`, rows `110`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `38.5`, LSTM `0.5115`, XGBoost `0.3700`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5107`, XGBoost `0.3992`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.5097`, XGBoost `0.3992`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5477`, XGBoost `0.6481`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `36.5`, LSTM `0.5087`, XGBoost `0.4087`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.5065`, XGBoost `0.4067`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.5494`, XGBoost `0.6481`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `36.0`, LSTM `0.5019`, XGBoost `0.4059`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5093`, XGBoost `0.4185`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.5319`, XGBoost `0.6184`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
