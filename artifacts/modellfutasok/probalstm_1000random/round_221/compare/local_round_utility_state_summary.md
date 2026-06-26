# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m2-dust2.csv`
- round_num: `4`
- rows: `178`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 178 | 1.000 | 0.081952 | 0.131845 | -0.049893 | 177 | 1 | 1.000000 | 1.000000 |
| active/recent utility | 178 | 1.000 | 0.081952 | 0.131845 | -0.049893 | 177 | 1 | 1.000000 | 1.000000 |
| strong utility action | 136 | 0.764 | 0.105358 | 0.169004 | -0.063646 | 135 | 1 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.056 | 0.192592 | 0.270285 | -0.077693 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 128 | 0.719 | 0.102881 | 0.167472 | -0.064592 | 127 | 1 | 1.000000 | 1.000000 |
| recent utility last 5s | 24 | 0.135 | 0.191833 | 0.238640 | -0.046806 | 23 | 1 | 1.000000 | 1.000000 |
| flash effect present | 178 | 1.000 | 0.081952 | 0.131845 | -0.049893 | 177 | 1 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `5.0s` - `68.5s`, rows `128`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `36.0`, LSTM `0.1070`, XGBoost `0.2597`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.1103`, XGBoost `0.2597`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.1113`, XGBoost `0.2597`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.1133`, XGBoost `0.2583`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.1147`, XGBoost `0.2597`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.1132`, XGBoost `0.2570`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.1167`, XGBoost `0.2597`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.1171`, XGBoost `0.2597`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.1152`, XGBoost `0.2577`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.1187`, XGBoost `0.2583`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
