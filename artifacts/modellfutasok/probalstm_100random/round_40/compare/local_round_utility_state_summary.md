# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m1-inferno.csv`
- round_num: `4`
- rows: `216`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 216 | 1.000 | 0.312204 | 0.249596 | 0.062607 | 100 | 116 | 0.504630 | 1.000000 |
| active/recent utility | 216 | 1.000 | 0.312204 | 0.249596 | 0.062607 | 100 | 116 | 0.504630 | 1.000000 |
| strong utility action | 192 | 0.889 | 0.325665 | 0.257836 | 0.067829 | 85 | 107 | 0.489583 | 1.000000 |
| utility damage | 20 | 0.093 | 0.507747 | 0.377232 | 0.130515 | 1 | 19 | 0.300000 | 1.000000 |
| active smoke/inferno | 182 | 0.843 | 0.315591 | 0.245914 | 0.069677 | 85 | 97 | 0.494505 | 1.000000 |
| recent utility last 5s | 10 | 0.046 | 0.509003 | 0.474813 | 0.034190 | 0 | 10 | 0.400000 | 1.000000 |
| flash effect present | 216 | 1.000 | 0.312204 | 0.249596 | 0.062607 | 100 | 116 | 0.504630 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `58.0s`, rows `98`
- `59.5s` - `101.0s`, rows `84`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `52.5`, LSTM `0.6564`, XGBoost `0.4686`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.6546`, XGBoost `0.4686`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.6542`, XGBoost `0.4686`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.6513`, XGBoost `0.4680`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.6501`, XGBoost `0.4703`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.6470`, XGBoost `0.4707`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.6469`, XGBoost `0.4717`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.6454`, XGBoost `0.4707`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.6420`, XGBoost `0.4673`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.6472`, XGBoost `0.4737`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
