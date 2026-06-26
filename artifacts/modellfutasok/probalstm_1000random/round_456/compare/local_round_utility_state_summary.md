# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `5`
- rows: `251`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 251 | 1.000 | 0.345604 | 0.345108 | 0.000496 | 134 | 117 | 0.649402 | 0.701195 |
| active/recent utility | 251 | 1.000 | 0.345604 | 0.345108 | 0.000496 | 134 | 117 | 0.649402 | 0.701195 |
| strong utility action | 165 | 0.657 | 0.390542 | 0.391421 | -0.000879 | 105 | 60 | 0.581818 | 0.642424 |
| utility damage | 19 | 0.076 | 0.546800 | 0.496838 | 0.049962 | 9 | 10 | 0.000000 | 0.421053 |
| active smoke/inferno | 157 | 0.625 | 0.382892 | 0.391248 | -0.008356 | 105 | 52 | 0.611465 | 0.624204 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 251 | 1.000 | 0.345604 | 0.345108 | 0.000496 | 134 | 117 | 0.649402 | 0.701195 |

## Active Smoke/Inferno Intervals

- `8.0s` - `51.5s`, rows `88`
- `54.0s` - `88.0s`, rows `69`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `125.0`, LSTM `0.5666`, XGBoost `0.3584`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `21.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.3681`, XGBoost `0.1718`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.3664`, XGBoost `0.1717`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.3568`, XGBoost `0.1723`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.3400`, XGBoost `0.1728`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `124.0`, LSTM `0.5582`, XGBoost `0.3924`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `21.0`, recent_utility `0`
- seconds `124.5`, LSTM `0.5478`, XGBoost `0.3856`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `21.0`, recent_utility `0`
- seconds `123.5`, LSTM `0.5393`, XGBoost `0.3870`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `21.0`, recent_utility `0`
- seconds `123.0`, LSTM `0.5324`, XGBoost `0.3877`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `21.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.3142`, XGBoost `0.1747`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
