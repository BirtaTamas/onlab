# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-faze-vs-aurora-bo3-ZssSxRC3p7Nn5A_BOLQ-lD/faze-vs-aurora-m2-mirage.csv`
- round_num: `9`
- rows: `161`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 161 | 1.000 | 0.720501 | 0.673503 | 0.046998 | 101 | 60 | 0.962733 | 0.987578 |
| active/recent utility | 161 | 1.000 | 0.720501 | 0.673503 | 0.046998 | 101 | 60 | 0.962733 | 0.987578 |
| strong utility action | 117 | 0.727 | 0.685660 | 0.629668 | 0.055992 | 81 | 36 | 0.948718 | 0.982906 |
| utility damage | 18 | 0.112 | 0.688828 | 0.747237 | -0.058409 | 0 | 18 | 1.000000 | 1.000000 |
| active smoke/inferno | 114 | 0.708 | 0.687090 | 0.627317 | 0.059773 | 81 | 33 | 0.947368 | 0.982456 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 161 | 1.000 | 0.720501 | 0.673503 | 0.046998 | 101 | 60 | 0.962733 | 0.987578 |

## Active Smoke/Inferno Intervals

- `6.0s` - `30.5s`, rows `50`
- `33.0s` - `59.0s`, rows `53`
- `60.5s` - `65.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.5`, LSTM `0.7541`, XGBoost `0.5846`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.7462`, XGBoost `0.5818`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.7408`, XGBoost `0.5818`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.7445`, XGBoost `0.5862`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.7387`, XGBoost `0.5841`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.7374`, XGBoost `0.5839`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.7343`, XGBoost `0.5818`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.7405`, XGBoost `0.5880`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.7392`, XGBoost `0.5880`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.7320`, XGBoost `0.5818`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
