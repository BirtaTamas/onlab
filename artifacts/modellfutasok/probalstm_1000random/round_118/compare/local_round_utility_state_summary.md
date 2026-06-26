# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `6`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.627951 | 0.635139 | -0.007187 | 80 | 150 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.627951 | 0.635139 | -0.007187 | 80 | 150 | 1.000000 | 1.000000 |
| strong utility action | 193 | 0.839 | 0.621308 | 0.635109 | -0.013800 | 50 | 143 | 1.000000 | 1.000000 |
| utility damage | 30 | 0.130 | 0.565508 | 0.562156 | 0.003352 | 12 | 18 | 1.000000 | 1.000000 |
| active smoke/inferno | 193 | 0.839 | 0.621308 | 0.635109 | -0.013800 | 50 | 143 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.043 | 0.532797 | 0.546662 | -0.013865 | 1 | 9 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.627951 | 0.635139 | -0.007187 | 80 | 150 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `31.5s`, rows `44`
- `33.0s` - `81.5s`, rows `98`
- `85.5s` - `110.5s`, rows `51`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `86.0`, LSTM `0.5815`, XGBoost `0.7305`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.6050`, XGBoost `0.7345`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.7823`, XGBoost `0.8877`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.6292`, XGBoost `0.7284`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.6313`, XGBoost `0.7259`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.0`, LSTM `0.6396`, XGBoost `0.7259`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `1.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.8020`, XGBoost `0.8845`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.8062`, XGBoost `0.8865`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.5`, LSTM `0.8099`, XGBoost `0.8867`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `89.0`, LSTM `0.6456`, XGBoost `0.7163`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `14.0`, recent_utility `0`
