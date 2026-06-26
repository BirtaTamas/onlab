# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-heroic-vs-natus-vincere-bo3-P_vZ7pAIyzYcLTUjDHhSUR/heroic-vs-natus-vincere-m2-ancient.csv`
- round_num: `6`
- rows: `173`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 173 | 1.000 | 0.792533 | 0.791089 | 0.001444 | 80 | 93 | 0.919075 | 1.000000 |
| active/recent utility | 173 | 1.000 | 0.792533 | 0.791089 | 0.001444 | 80 | 93 | 0.919075 | 1.000000 |
| strong utility action | 92 | 0.532 | 0.771203 | 0.758435 | 0.012768 | 58 | 34 | 0.869565 | 1.000000 |
| utility damage | 12 | 0.069 | 0.731861 | 0.755827 | -0.023965 | 1 | 11 | 1.000000 | 1.000000 |
| active smoke/inferno | 92 | 0.532 | 0.771203 | 0.758435 | 0.012768 | 58 | 34 | 0.869565 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 173 | 1.000 | 0.792533 | 0.791089 | 0.001444 | 80 | 93 | 0.919075 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `46.0s`, rows `81`
- `81.0s` - `86.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `29.5`, LSTM `0.8708`, XGBoost `0.7769`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.4735`, XGBoost `0.5669`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.4708`, XGBoost `0.5639`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.6984`, XGBoost `0.6060`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.8691`, XGBoost `0.7767`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.4712`, XGBoost `0.5621`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.4790`, XGBoost `0.5673`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.8633`, XGBoost `0.7769`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.8608`, XGBoost `0.7764`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.8593`, XGBoost `0.7767`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
