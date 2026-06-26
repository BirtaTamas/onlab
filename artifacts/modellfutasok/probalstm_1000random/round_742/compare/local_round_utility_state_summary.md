# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m3-train.csv`
- round_num: `13`
- rows: `210`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 210 | 1.000 | 0.645541 | 0.750250 | -0.104709 | 18 | 192 | 1.000000 | 0.980952 |
| active/recent utility | 210 | 1.000 | 0.645541 | 0.750250 | -0.104709 | 18 | 192 | 1.000000 | 0.980952 |
| strong utility action | 104 | 0.495 | 0.655314 | 0.779596 | -0.124282 | 14 | 90 | 1.000000 | 0.961538 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 104 | 0.495 | 0.655314 | 0.779596 | -0.124282 | 14 | 90 | 1.000000 | 0.961538 |
| recent utility last 5s | 10 | 0.048 | 0.616875 | 0.832939 | -0.216065 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 210 | 1.000 | 0.645541 | 0.750250 | -0.104709 | 18 | 192 | 1.000000 | 0.980952 |

## Active Smoke/Inferno Intervals

- `25.5s` - `52.0s`, rows `54`
- `71.0s` - `95.5s`, rows `50`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `37.0`, LSTM `0.6795`, XGBoost `0.9420`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `6.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.5841`, XGBoost `0.8329`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.5873`, XGBoost `0.8329`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.5877`, XGBoost `0.8329`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.5894`, XGBoost `0.8329`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.5922`, XGBoost `0.8329`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.6031`, XGBoost `0.8329`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.6083`, XGBoost `0.8329`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `49.5`, LSTM `0.6097`, XGBoost `0.8329`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `40.5`, LSTM `0.6101`, XGBoost `0.8329`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
