# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-nrg-bo3-WMQcRUwgyUmu57EEkX9f3P/falcons-vs-nrg-m1-train.csv`
- round_num: `18`
- rows: `184`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 184 | 1.000 | 0.265257 | 0.337227 | -0.071970 | 164 | 20 | 1.000000 | 1.000000 |
| active/recent utility | 184 | 1.000 | 0.265257 | 0.337227 | -0.071970 | 164 | 20 | 1.000000 | 1.000000 |
| strong utility action | 163 | 0.886 | 0.263665 | 0.338556 | -0.074892 | 150 | 13 | 1.000000 | 1.000000 |
| utility damage | 8 | 0.043 | 0.178327 | 0.253099 | -0.074771 | 8 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 146 | 0.793 | 0.275502 | 0.344867 | -0.069365 | 133 | 13 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.054 | 0.148050 | 0.306326 | -0.158276 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 184 | 1.000 | 0.265257 | 0.337227 | -0.071970 | 164 | 20 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `72.0s`, rows `128`
- `74.5s` - `81.0s`, rows `14`
- `86.5s` - `88.0s`, rows `4`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.5`, LSTM `0.1206`, XGBoost `0.3468`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.1353`, XGBoost `0.3498`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.1445`, XGBoost `0.3560`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.1374`, XGBoost `0.3468`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.1460`, XGBoost `0.3441`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.1562`, XGBoost `0.3453`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.1697`, XGBoost `0.3588`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.1594`, XGBoost `0.3468`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `1.5`, LSTM `0.1499`, XGBoost `0.3358`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.1499`, XGBoost `0.3311`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
