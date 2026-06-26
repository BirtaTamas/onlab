# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `8`
- rows: `285`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 285 | 1.000 | 0.794706 | 0.814303 | -0.019598 | 69 | 216 | 0.908772 | 1.000000 |
| active/recent utility | 285 | 1.000 | 0.794706 | 0.814303 | -0.019598 | 69 | 216 | 0.908772 | 1.000000 |
| strong utility action | 196 | 0.688 | 0.784370 | 0.809438 | -0.025068 | 41 | 155 | 0.928571 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 196 | 0.688 | 0.784370 | 0.809438 | -0.025068 | 41 | 155 | 0.928571 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 285 | 1.000 | 0.794706 | 0.814303 | -0.019598 | 69 | 216 | 0.908772 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `53.0s`, rows `91`
- `65.0s` - `99.5s`, rows `70`
- `103.5s` - `115.0s`, rows `24`
- `124.0s` - `129.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `107.5`, LSTM `0.8288`, XGBoost `0.9393`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.4600`, XGBoost `0.5697`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `109.0`, LSTM `0.8310`, XGBoost `0.9282`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.4726`, XGBoost `0.5697`, closer `xgboost`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `108.5`, LSTM `0.8325`, XGBoost `0.9282`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `109.5`, LSTM `0.8387`, XGBoost `0.9328`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `110.0`, LSTM `0.8429`, XGBoost `0.9328`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5406`, XGBoost `0.6304`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.4836`, XGBoost `0.5669`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `107.0`, LSTM `0.8567`, XGBoost `0.9393`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
