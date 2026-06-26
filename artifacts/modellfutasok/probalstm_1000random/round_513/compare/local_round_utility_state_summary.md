# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m2-dust2.csv`
- round_num: `16`
- rows: `170`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 170 | 1.000 | 0.722873 | 0.719100 | 0.003773 | 74 | 96 | 1.000000 | 1.000000 |
| active/recent utility | 170 | 1.000 | 0.722873 | 0.719100 | 0.003773 | 74 | 96 | 1.000000 | 1.000000 |
| strong utility action | 140 | 0.824 | 0.720246 | 0.723118 | -0.002872 | 54 | 86 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.059 | 0.657620 | 0.611835 | 0.045785 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 140 | 0.824 | 0.720246 | 0.723118 | -0.002872 | 54 | 86 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 170 | 1.000 | 0.722873 | 0.719100 | 0.003773 | 74 | 96 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `54.5s`, rows `93`
- `61.5s` - `84.5s`, rows `47`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `81.0`, LSTM `0.5638`, XGBoost `0.6822`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.6069`, XGBoost `0.7080`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.5325`, XGBoost `0.6166`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.5335`, XGBoost `0.6166`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.6873`, XGBoost `0.6042`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.5698`, XGBoost `0.6493`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.6812`, XGBoost `0.6051`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.6755`, XGBoost `0.6007`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.6786`, XGBoost `0.6042`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.5448`, XGBoost `0.6166`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
