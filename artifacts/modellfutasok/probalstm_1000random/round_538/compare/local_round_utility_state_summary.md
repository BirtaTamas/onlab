# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m3-mirage.csv`
- round_num: `4`
- rows: `157`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 157 | 1.000 | 0.332929 | 0.324211 | 0.008718 | 81 | 76 | 0.840764 | 1.000000 |
| active/recent utility | 157 | 1.000 | 0.332929 | 0.324211 | 0.008718 | 81 | 76 | 0.840764 | 1.000000 |
| strong utility action | 140 | 0.892 | 0.335332 | 0.322474 | 0.012859 | 67 | 73 | 0.821429 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 140 | 0.892 | 0.335332 | 0.322474 | 0.012859 | 67 | 73 | 0.821429 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 157 | 1.000 | 0.332929 | 0.324211 | 0.008718 | 81 | 76 | 0.840764 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `37.5s`, rows `64`
- `39.5s` - `77.0s`, rows `76`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.5`, LSTM `0.5861`, XGBoost `0.3915`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5839`, XGBoost `0.3986`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5840`, XGBoost `0.3988`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.5770`, XGBoost `0.3926`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5794`, XGBoost `0.3988`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5784`, XGBoost `0.3986`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5674`, XGBoost `0.3926`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.5657`, XGBoost `0.3926`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5616`, XGBoost `0.3988`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.5576`, XGBoost `0.3988`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
