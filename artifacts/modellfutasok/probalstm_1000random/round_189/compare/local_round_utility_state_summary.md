# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `22`
- rows: `103`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 103 | 1.000 | 0.111127 | 0.220995 | -0.109867 | 97 | 6 | 1.000000 | 1.000000 |
| active/recent utility | 103 | 1.000 | 0.111127 | 0.220995 | -0.109867 | 97 | 6 | 1.000000 | 1.000000 |
| strong utility action | 83 | 0.806 | 0.125329 | 0.228798 | -0.103469 | 77 | 6 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.107 | 0.146675 | 0.263965 | -0.117291 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 83 | 0.806 | 0.125329 | 0.228798 | -0.103469 | 77 | 6 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 103 | 1.000 | 0.111127 | 0.220995 | -0.109867 | 97 | 6 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `49.0s`, rows `83`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `35.5`, LSTM `0.0907`, XGBoost `0.2790`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.0949`, XGBoost `0.2768`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.1049`, XGBoost `0.2714`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0616`, XGBoost `0.2243`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.1174`, XGBoost `0.2790`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.1186`, XGBoost `0.2788`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `24.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.0646`, XGBoost `0.2241`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.1146`, XGBoost `0.2729`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.1227`, XGBoost `0.2773`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.1196`, XGBoost `0.2734`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
