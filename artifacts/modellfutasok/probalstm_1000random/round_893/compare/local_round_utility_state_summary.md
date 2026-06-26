# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-nrg-bo3-WMQcRUwgyUmu57EEkX9f3P/falcons-vs-nrg-m1-train.csv`
- round_num: `5`
- rows: `238`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 238 | 1.000 | 0.295106 | 0.326562 | -0.031456 | 184 | 54 | 0.605042 | 0.441176 |
| active/recent utility | 238 | 1.000 | 0.295106 | 0.326562 | -0.031456 | 184 | 54 | 0.605042 | 0.441176 |
| strong utility action | 177 | 0.744 | 0.336797 | 0.375108 | -0.038311 | 129 | 48 | 0.542373 | 0.367232 |
| utility damage | 10 | 0.042 | 0.423419 | 0.510837 | -0.087418 | 10 | 0 | 1.000000 | 0.000000 |
| active smoke/inferno | 177 | 0.744 | 0.336797 | 0.375108 | -0.038311 | 129 | 48 | 0.542373 | 0.367232 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 238 | 1.000 | 0.295106 | 0.326562 | -0.031456 | 184 | 54 | 0.605042 | 0.441176 |

## Active Smoke/Inferno Intervals

- `8.5s` - `48.5s`, rows `81`
- `51.0s` - `98.5s`, rows `96`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `71.0`, LSTM `0.0434`, XGBoost `0.3328`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.0506`, XGBoost `0.3321`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.0494`, XGBoost `0.3307`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.0540`, XGBoost `0.3337`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.0540`, XGBoost `0.3337`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.0606`, XGBoost `0.3307`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.0671`, XGBoost `0.3300`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.0637`, XGBoost `0.3246`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.0742`, XGBoost `0.3314`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.0687`, XGBoost `0.3251`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
