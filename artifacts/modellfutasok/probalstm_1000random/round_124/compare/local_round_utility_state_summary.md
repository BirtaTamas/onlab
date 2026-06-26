# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-b8-vs-hotu-bo3-tmCfOETKzYqjV6vSvNp3-F/b8-vs-hotu-m3-ancient.csv`
- round_num: `2`
- rows: `112`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 112 | 1.000 | 0.950437 | 0.984265 | -0.033828 | 0 | 112 | 1.000000 | 1.000000 |
| active/recent utility | 112 | 1.000 | 0.950437 | 0.984265 | -0.033828 | 0 | 112 | 1.000000 | 1.000000 |
| strong utility action | 66 | 0.589 | 0.939746 | 0.981240 | -0.041494 | 0 | 66 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.179 | 0.961000 | 0.986269 | -0.025269 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 66 | 0.589 | 0.939746 | 0.981240 | -0.041494 | 0 | 66 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 112 | 1.000 | 0.950437 | 0.984265 | -0.033828 | 0 | 112 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `39.5s`, rows `66`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.0`, LSTM `0.8686`, XGBoost `0.9792`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.8744`, XGBoost `0.9792`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.8837`, XGBoost `0.9792`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.8863`, XGBoost `0.9792`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.8905`, XGBoost `0.9792`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.8937`, XGBoost `0.9787`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.8946`, XGBoost `0.9792`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.8965`, XGBoost `0.9792`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.9008`, XGBoost `0.9787`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.9084`, XGBoost `0.9787`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
