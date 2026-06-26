# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-virtuspro-bo3-8Z0L17IYJlstHvIADVy9G9/vitality-vs-virtus-pro-m3-mirage.csv`
- round_num: `3`
- rows: `120`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 120 | 1.000 | 0.717229 | 0.664785 | 0.052443 | 76 | 44 | 0.966667 | 0.850000 |
| active/recent utility | 120 | 1.000 | 0.717229 | 0.664785 | 0.052443 | 76 | 44 | 0.966667 | 0.850000 |
| strong utility action | 107 | 0.892 | 0.727229 | 0.686106 | 0.041123 | 63 | 44 | 0.962617 | 0.953271 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 107 | 0.892 | 0.727229 | 0.686106 | 0.041123 | 63 | 44 | 0.962617 | 0.953271 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 120 | 1.000 | 0.717229 | 0.664785 | 0.052443 | 76 | 44 | 0.966667 | 0.850000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `59.5s`, rows `107`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `12.5`, LSTM `0.6766`, XGBoost `0.5086`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.6752`, XGBoost `0.5087`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.6576`, XGBoost `0.4928`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.6678`, XGBoost `0.5057`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.6532`, XGBoost `0.4928`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6641`, XGBoost `0.5055`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.6673`, XGBoost `0.5087`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.6657`, XGBoost `0.5087`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6618`, XGBoost `0.5055`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.6579`, XGBoost `0.5087`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
