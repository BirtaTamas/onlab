# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `21`
- rows: `161`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 161 | 1.000 | 0.132801 | 0.151736 | -0.018935 | 142 | 19 | 0.807453 | 0.807453 |
| active/recent utility | 161 | 1.000 | 0.132801 | 0.151736 | -0.018935 | 142 | 19 | 0.807453 | 0.807453 |
| strong utility action | 88 | 0.547 | 0.144152 | 0.165221 | -0.021069 | 74 | 14 | 0.818182 | 0.818182 |
| utility damage | 12 | 0.075 | 0.429569 | 0.443382 | -0.013813 | 9 | 3 | 0.250000 | 0.250000 |
| active smoke/inferno | 88 | 0.547 | 0.144152 | 0.165221 | -0.021069 | 74 | 14 | 0.818182 | 0.818182 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 161 | 1.000 | 0.132801 | 0.151736 | -0.018935 | 142 | 19 | 0.807453 | 0.807453 |

## Active Smoke/Inferno Intervals

- `7.5s` - `51.0s`, rows `88`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.5`, LSTM `0.1730`, XGBoost `0.0838`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `60.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.0377`, XGBoost `0.1214`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.0490`, XGBoost `0.1224`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.0522`, XGBoost `0.1224`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.0573`, XGBoost `0.1222`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0682`, XGBoost `0.1214`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.0494`, XGBoost `0.1025`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.0494`, XGBoost `0.1025`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.0491`, XGBoost `0.1020`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.5401`, XGBoost `0.5903`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `60.0`, recent_utility `0`
