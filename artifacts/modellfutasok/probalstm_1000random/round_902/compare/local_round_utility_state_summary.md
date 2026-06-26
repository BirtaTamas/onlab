# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-inner-circle-vs-gentle-mates-bo3-u31MSfrH-KJtKM4rM-4jj7/inner-circle-vs-gentle-mates-m1-nuke.csv`
- round_num: `4`
- rows: `266`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 266 | 1.000 | 0.338132 | 0.388159 | -0.050027 | 235 | 31 | 0.548872 | 0.582707 |
| active/recent utility | 266 | 1.000 | 0.338132 | 0.388159 | -0.050027 | 235 | 31 | 0.548872 | 0.582707 |
| strong utility action | 217 | 0.816 | 0.367355 | 0.422655 | -0.055299 | 186 | 31 | 0.529954 | 0.571429 |
| utility damage | 24 | 0.090 | 0.637034 | 0.791504 | -0.154470 | 24 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 217 | 0.816 | 0.367355 | 0.422655 | -0.055299 | 186 | 31 | 0.529954 | 0.571429 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 266 | 1.000 | 0.338132 | 0.388159 | -0.050027 | 235 | 31 | 0.548872 | 0.582707 |

## Active Smoke/Inferno Intervals

- `9.0s` - `75.0s`, rows `133`
- `76.0s` - `117.5s`, rows `84`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `29.5`, LSTM `0.5870`, XGBoost `0.7918`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5880`, XGBoost `0.7918`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5999`, XGBoost `0.7937`, closer `lstm`, smoke `6`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.6021`, XGBoost `0.7918`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.6083`, XGBoost `0.7918`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.6117`, XGBoost `0.7918`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.6133`, XGBoost `0.7914`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.6139`, XGBoost `0.7918`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.6214`, XGBoost `0.7974`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.6157`, XGBoost `0.7914`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
