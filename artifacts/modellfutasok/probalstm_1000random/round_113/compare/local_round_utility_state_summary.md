# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-tyloo-vs-falcons-bo3-MBKGKnSCeuy54EHzS5mmW8/tyloo-vs-falcons-m2-ancient.csv`
- round_num: `2`
- rows: `205`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 205 | 1.000 | 0.035492 | 0.062222 | -0.026730 | 192 | 13 | 1.000000 | 1.000000 |
| active/recent utility | 205 | 1.000 | 0.035492 | 0.062222 | -0.026730 | 192 | 13 | 1.000000 | 1.000000 |
| strong utility action | 125 | 0.610 | 0.043876 | 0.075835 | -0.031959 | 113 | 12 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.049 | 0.120775 | 0.158838 | -0.038063 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 115 | 0.561 | 0.024784 | 0.060899 | -0.036114 | 109 | 6 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.049 | 0.263432 | 0.247606 | 0.015827 | 4 | 6 | 1.000000 | 1.000000 |
| flash effect present | 205 | 1.000 | 0.035492 | 0.062222 | -0.026730 | 192 | 13 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `37.0s`, rows `58`
- `50.0s` - `78.0s`, rows `57`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `12.5`, LSTM `0.1209`, XGBoost `0.1926`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `48.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.0841`, XGBoost `0.1532`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `3.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.1249`, XGBoost `0.1871`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `45.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.0041`, XGBoost `0.0653`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.0117`, XGBoost `0.0687`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.0042`, XGBoost `0.0610`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.0032`, XGBoost `0.0598`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.0038`, XGBoost `0.0603`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.0034`, XGBoost `0.0598`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.0068`, XGBoost `0.0631`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
