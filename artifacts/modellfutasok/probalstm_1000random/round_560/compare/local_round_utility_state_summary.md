# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-gentle-mates-bo3-AJh0VVYB1ya_7X1VH9GAqu/g2-vs-gentle-mates-m1-inferno.csv`
- round_num: `8`
- rows: `139`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 139 | 1.000 | 0.354061 | 0.235089 | 0.118972 | 12 | 127 | 0.755396 | 0.805755 |
| active/recent utility | 139 | 1.000 | 0.354061 | 0.235089 | 0.118972 | 12 | 127 | 0.755396 | 0.805755 |
| strong utility action | 129 | 0.928 | 0.330529 | 0.210791 | 0.119738 | 12 | 117 | 0.813953 | 0.868217 |
| utility damage | 24 | 0.173 | 0.321177 | 0.184369 | 0.136807 | 0 | 24 | 0.791667 | 1.000000 |
| active smoke/inferno | 119 | 0.856 | 0.302896 | 0.182103 | 0.120793 | 12 | 107 | 0.882353 | 0.941176 |
| recent utility last 5s | 10 | 0.072 | 0.659364 | 0.552176 | 0.107188 | 0 | 10 | 0.000000 | 0.000000 |
| flash effect present | 139 | 1.000 | 0.354061 | 0.235089 | 0.118972 | 12 | 127 | 0.755396 | 0.805755 |

## Active Smoke/Inferno Intervals

- `10.0s` - `69.0s`, rows `119`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `47.5`, LSTM `0.4844`, XGBoost `0.1481`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.5309`, XGBoost `0.2328`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `45.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.5212`, XGBoost `0.2264`, closer `xgboost`, smoke `2`, inferno `3`, utility_damage `45.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.4460`, XGBoost `0.1524`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.4317`, XGBoost `0.1408`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.4391`, XGBoost `0.1534`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.4401`, XGBoost `0.1554`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.4366`, XGBoost `0.1590`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.4209`, XGBoost `0.1437`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.4154`, XGBoost `0.1422`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
