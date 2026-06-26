# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `6`
- rows: `197`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 197 | 1.000 | 0.175801 | 0.208892 | -0.033091 | 151 | 46 | 1.000000 | 1.000000 |
| active/recent utility | 197 | 1.000 | 0.175801 | 0.208892 | -0.033091 | 151 | 46 | 1.000000 | 1.000000 |
| strong utility action | 196 | 0.995 | 0.175811 | 0.208667 | -0.032856 | 150 | 46 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.051 | 0.270774 | 0.290348 | -0.019574 | 8 | 2 | 1.000000 | 1.000000 |
| active smoke/inferno | 184 | 0.934 | 0.179252 | 0.205841 | -0.026589 | 138 | 46 | 1.000000 | 1.000000 |
| recent utility last 5s | 14 | 0.071 | 0.120213 | 0.251551 | -0.131339 | 14 | 0 | 1.000000 | 1.000000 |
| flash effect present | 197 | 1.000 | 0.175801 | 0.208892 | -0.033091 | 151 | 46 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `98.0s`, rows `184`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `1.0`, LSTM `0.1025`, XGBoost `0.2529`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `2.0`, LSTM `0.1028`, XGBoost `0.2529`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `4`
- seconds `2.5`, LSTM `0.1053`, XGBoost `0.2545`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `5`
- seconds `6.5`, LSTM `0.1007`, XGBoost `0.2498`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `0.5`, LSTM `0.1074`, XGBoost `0.2529`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `7.0`, LSTM `0.1056`, XGBoost `0.2478`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.0`, LSTM `0.1035`, XGBoost `0.2447`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `1.5`, LSTM `0.1137`, XGBoost `0.2529`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `4`
- seconds `7.5`, LSTM `0.1103`, XGBoost `0.2475`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.1436`, XGBoost `0.2802`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `13.0`, recent_utility `0`
