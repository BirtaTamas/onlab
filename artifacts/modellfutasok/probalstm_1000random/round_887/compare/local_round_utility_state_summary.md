# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m1-train.csv`
- round_num: `5`
- rows: `238`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 238 | 1.000 | 0.171809 | 0.168044 | 0.003764 | 187 | 51 | 0.768908 | 0.785714 |
| active/recent utility | 238 | 1.000 | 0.171809 | 0.168044 | 0.003764 | 187 | 51 | 0.768908 | 0.785714 |
| strong utility action | 124 | 0.521 | 0.258191 | 0.250681 | 0.007510 | 87 | 37 | 0.677419 | 0.709677 |
| utility damage | 10 | 0.042 | 0.510092 | 0.433730 | 0.076362 | 0 | 10 | 0.500000 | 0.500000 |
| active smoke/inferno | 124 | 0.521 | 0.258191 | 0.250681 | 0.007510 | 87 | 37 | 0.677419 | 0.709677 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 238 | 1.000 | 0.171809 | 0.168044 | 0.003764 | 187 | 51 | 0.768908 | 0.785714 |

## Active Smoke/Inferno Intervals

- `7.5s` - `47.0s`, rows `80`
- `64.5s` - `86.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `17.5`, LSTM `0.5130`, XGBoost `0.3200`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.5199`, XGBoost `0.3282`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.5046`, XGBoost `0.3200`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.5044`, XGBoost `0.3200`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.4657`, XGBoost `0.2865`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.4954`, XGBoost `0.3203`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.4933`, XGBoost `0.3203`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.4881`, XGBoost `0.3182`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.4897`, XGBoost `0.3203`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.4821`, XGBoost `0.3282`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
