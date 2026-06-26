# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-furia-vs-g2-bo3-QMek4tXQesgbTlulfGKOmD/furia-vs-g2-m1-inferno.csv`
- round_num: `12`
- rows: `167`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 167 | 1.000 | 0.494195 | 0.491607 | 0.002587 | 81 | 86 | 0.365269 | 0.365269 |
| active/recent utility | 167 | 1.000 | 0.494195 | 0.491607 | 0.002587 | 81 | 86 | 0.365269 | 0.365269 |
| strong utility action | 148 | 0.886 | 0.464122 | 0.463068 | 0.001054 | 80 | 68 | 0.412162 | 0.412162 |
| utility damage | 30 | 0.180 | 0.517419 | 0.512962 | 0.004457 | 16 | 14 | 0.333333 | 0.333333 |
| active smoke/inferno | 148 | 0.886 | 0.464122 | 0.463068 | 0.001054 | 80 | 68 | 0.412162 | 0.412162 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 167 | 1.000 | 0.494195 | 0.491607 | 0.002587 | 81 | 86 | 0.365269 | 0.365269 |

## Active Smoke/Inferno Intervals

- `9.5s` - `83.0s`, rows `148`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `71.5`, LSTM `0.2467`, XGBoost `0.1240`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.2247`, XGBoost `0.1237`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.2246`, XGBoost `0.1240`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.2208`, XGBoost `0.1229`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.2175`, XGBoost `0.1240`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.6616`, XGBoost `0.7541`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.2132`, XGBoost `0.1229`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.2141`, XGBoost `0.1240`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.6699`, XGBoost `0.7534`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.6807`, XGBoost `0.7596`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
