# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `3`
- rows: `165`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 165 | 1.000 | 0.713829 | 0.729732 | -0.015904 | 45 | 120 | 0.600000 | 0.600000 |
| active/recent utility | 165 | 1.000 | 0.713829 | 0.729732 | -0.015904 | 45 | 120 | 0.600000 | 0.600000 |
| strong utility action | 85 | 0.515 | 0.684391 | 0.707660 | -0.023270 | 21 | 64 | 0.576471 | 0.576471 |
| utility damage | 10 | 0.061 | 0.470108 | 0.461570 | 0.008538 | 8 | 2 | 0.000000 | 0.000000 |
| active smoke/inferno | 85 | 0.515 | 0.684391 | 0.707660 | -0.023270 | 21 | 64 | 0.576471 | 0.576471 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 165 | 1.000 | 0.713829 | 0.729732 | -0.015904 | 45 | 120 | 0.600000 | 0.600000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `20.0s`, rows `22`
- `22.5s` - `29.0s`, rows `14`
- `40.0s` - `64.0s`, rows `49`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `19.0`, LSTM `0.3730`, XGBoost `0.4777`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.3817`, XGBoost `0.4807`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.3849`, XGBoost `0.4807`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.3864`, XGBoost `0.4807`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.3941`, XGBoost `0.4784`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.3987`, XGBoost `0.4777`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.4040`, XGBoost `0.4777`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.8783`, XGBoost `0.9494`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.4098`, XGBoost `0.4784`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.4116`, XGBoost `0.4784`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
