# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `5`
- rows: `307`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 307 | 1.000 | 0.460814 | 0.369154 | 0.091661 | 71 | 236 | 0.322476 | 0.723127 |
| active/recent utility | 307 | 1.000 | 0.460814 | 0.369154 | 0.091661 | 71 | 236 | 0.322476 | 0.723127 |
| strong utility action | 219 | 0.713 | 0.548416 | 0.438291 | 0.110125 | 27 | 192 | 0.187215 | 0.694064 |
| utility damage | 10 | 0.033 | 0.678017 | 0.505763 | 0.172254 | 0 | 10 | 0.000000 | 0.000000 |
| active smoke/inferno | 219 | 0.713 | 0.548416 | 0.438291 | 0.110125 | 27 | 192 | 0.187215 | 0.694064 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 307 | 1.000 | 0.460814 | 0.369154 | 0.091661 | 71 | 236 | 0.322476 | 0.723127 |

## Active Smoke/Inferno Intervals

- `12.0s` - `34.0s`, rows `45`
- `37.5s` - `118.5s`, rows `163`
- `119.5s` - `124.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `13.0`, LSTM `0.7185`, XGBoost `0.4972`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.7072`, XGBoost `0.4970`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.7048`, XGBoost `0.4972`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.6891`, XGBoost `0.4816`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.6831`, XGBoost `0.4816`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.6762`, XGBoost `0.4816`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.6955`, XGBoost `0.5010`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.6923`, XGBoost `0.5013`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.6008`, XGBoost `0.4102`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.6698`, XGBoost `0.4816`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `6.0`, recent_utility `0`
