# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-furia-vs-m80-bo3-mWbCj4SBCT3wH-l62HcQgw/furia-vs-m80-m1-mirage.csv`
- round_num: `2`
- rows: `212`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 212 | 1.000 | 0.677607 | 0.744671 | -0.067065 | 25 | 187 | 1.000000 | 1.000000 |
| active/recent utility | 212 | 1.000 | 0.677607 | 0.744671 | -0.067065 | 25 | 187 | 1.000000 | 1.000000 |
| strong utility action | 178 | 0.840 | 0.656991 | 0.728808 | -0.071817 | 24 | 154 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 166 | 0.783 | 0.663215 | 0.739117 | -0.075902 | 23 | 143 | 1.000000 | 1.000000 |
| recent utility last 5s | 14 | 0.066 | 0.566245 | 0.584706 | -0.018461 | 1 | 13 | 1.000000 | 1.000000 |
| flash effect present | 212 | 1.000 | 0.677607 | 0.744671 | -0.067065 | 25 | 187 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `51.5s`, rows `87`
- `58.0s` - `97.0s`, rows `79`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `40.5`, LSTM `0.5310`, XGBoost `0.6978`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.5375`, XGBoost `0.6978`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.5379`, XGBoost `0.6978`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.5383`, XGBoost `0.6978`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.5339`, XGBoost `0.6933`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5371`, XGBoost `0.6960`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.5334`, XGBoost `0.6917`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.5383`, XGBoost `0.6960`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.5354`, XGBoost `0.6915`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.5553`, XGBoost `0.7113`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
