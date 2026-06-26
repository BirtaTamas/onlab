# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-wildcard-vs-furia-bo3-u8Kr9GGu18RWnHSjYzEreW/wildcard-vs-furia-m2-inferno.csv`
- round_num: `8`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.719709 | 0.762632 | -0.042923 | 26 | 204 | 0.969565 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.719709 | 0.762632 | -0.042923 | 26 | 204 | 0.969565 | 1.000000 |
| strong utility action | 205 | 0.891 | 0.737660 | 0.785116 | -0.047456 | 15 | 190 | 0.985366 | 1.000000 |
| utility damage | 30 | 0.130 | 0.613549 | 0.697697 | -0.084148 | 2 | 28 | 1.000000 | 1.000000 |
| active smoke/inferno | 202 | 0.878 | 0.739790 | 0.787043 | -0.047252 | 15 | 187 | 0.985149 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.719709 | 0.762632 | -0.042923 | 26 | 204 | 0.969565 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `24.5s`, rows `31`
- `28.0s` - `113.0s`, rows `171`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.5`, LSTM `0.5936`, XGBoost `0.7621`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.6119`, XGBoost `0.7699`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.6146`, XGBoost `0.7621`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.6164`, XGBoost `0.7537`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.6209`, XGBoost `0.7537`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.5330`, XGBoost `0.6608`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5362`, XGBoost `0.6608`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.5363`, XGBoost `0.6608`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5385`, XGBoost `0.6608`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.5401`, XGBoost `0.6608`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `1.0`, recent_utility `0`
