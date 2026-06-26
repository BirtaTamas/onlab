# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj/falcons-vs-virtus-pro-m2-ancient.csv`
- round_num: `15`
- rows: `180`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 180 | 1.000 | 0.418441 | 0.520271 | -0.101830 | 1 | 179 | 0.327778 | 0.466667 |
| active/recent utility | 180 | 1.000 | 0.418441 | 0.520271 | -0.101830 | 1 | 179 | 0.327778 | 0.466667 |
| strong utility action | 120 | 0.667 | 0.295407 | 0.379505 | -0.084098 | 1 | 119 | 0.100000 | 0.308333 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 120 | 0.667 | 0.295407 | 0.379505 | -0.084098 | 1 | 119 | 0.100000 | 0.308333 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 180 | 1.000 | 0.418441 | 0.520271 | -0.101830 | 1 | 179 | 0.327778 | 0.466667 |

## Active Smoke/Inferno Intervals

- `6.0s` - `29.0s`, rows `47`
- `30.0s` - `66.0s`, rows `73`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `6.5`, LSTM `0.0813`, XGBoost `0.3052`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.0`, LSTM `0.0859`, XGBoost `0.3038`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.0937`, XGBoost `0.3052`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.1217`, XGBoost `0.2993`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.1279`, XGBoost `0.3004`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.1539`, XGBoost `0.3059`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1663`, XGBoost `0.3091`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.3929`, XGBoost `0.5350`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.1606`, XGBoost `0.2998`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.3965`, XGBoost `0.5348`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
