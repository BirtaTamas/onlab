# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62/tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `5`
- rows: `149`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 149 | 1.000 | 0.200702 | 0.180341 | 0.020361 | 91 | 58 | 0.711409 | 0.718121 |
| active/recent utility | 149 | 1.000 | 0.200702 | 0.180341 | 0.020361 | 91 | 58 | 0.711409 | 0.718121 |
| strong utility action | 79 | 0.530 | 0.214276 | 0.193300 | 0.020975 | 43 | 36 | 0.696203 | 0.708861 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 79 | 0.530 | 0.214276 | 0.193300 | 0.020975 | 43 | 36 | 0.696203 | 0.708861 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 149 | 1.000 | 0.200702 | 0.180341 | 0.020361 | 91 | 58 | 0.711409 | 0.718121 |

## Active Smoke/Inferno Intervals

- `9.5s` - `48.5s`, rows `79`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `21.0`, LSTM `0.5470`, XGBoost `0.3809`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.7161`, XGBoost `0.5996`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.7253`, XGBoost `0.6091`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.7022`, XGBoost `0.5996`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.7052`, XGBoost `0.6059`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.6926`, XGBoost `0.5994`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.6936`, XGBoost `0.6035`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6869`, XGBoost `0.5996`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.6916`, XGBoost `0.6083`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.6907`, XGBoost `0.6101`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
