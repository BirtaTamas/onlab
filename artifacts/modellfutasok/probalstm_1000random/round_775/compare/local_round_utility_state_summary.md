# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-g2-vs-gamerlegion-bo3-gcs9469UuxWlHi6X2zI5Oy/g2-vs-gamerlegion-m2-ancient.csv`
- round_num: `4`
- rows: `224`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 224 | 1.000 | 0.144422 | 0.184308 | -0.039886 | 195 | 29 | 0.839286 | 0.843750 |
| active/recent utility | 224 | 1.000 | 0.144422 | 0.184308 | -0.039886 | 195 | 29 | 0.839286 | 0.843750 |
| strong utility action | 148 | 0.661 | 0.178871 | 0.236042 | -0.057170 | 129 | 19 | 0.831081 | 0.831081 |
| utility damage | 16 | 0.071 | 0.514810 | 0.509380 | 0.005430 | 7 | 9 | 0.187500 | 0.187500 |
| active smoke/inferno | 148 | 0.661 | 0.178871 | 0.236042 | -0.057170 | 129 | 19 | 0.831081 | 0.831081 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 224 | 1.000 | 0.144422 | 0.184308 | -0.039886 | 195 | 29 | 0.839286 | 0.843750 |

## Active Smoke/Inferno Intervals

- `5.5s` - `79.0s`, rows `148`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.5`, LSTM `0.1213`, XGBoost `0.2632`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.0312`, XGBoost `0.1688`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.1121`, XGBoost `0.2487`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.0326`, XGBoost `0.1688`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.0355`, XGBoost `0.1688`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.1284`, XGBoost `0.2585`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.0365`, XGBoost `0.1666`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0389`, XGBoost `0.1666`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.1294`, XGBoost `0.2568`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.1330`, XGBoost `0.2594`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
