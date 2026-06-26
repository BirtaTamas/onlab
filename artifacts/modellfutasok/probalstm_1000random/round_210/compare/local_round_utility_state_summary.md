# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-nrg-bo3-WMQcRUwgyUmu57EEkX9f3P/falcons-vs-nrg-m1-train.csv`
- round_num: `7`
- rows: `144`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 144 | 1.000 | 0.048574 | 0.099093 | -0.050519 | 137 | 7 | 1.000000 | 1.000000 |
| active/recent utility | 144 | 1.000 | 0.048574 | 0.099093 | -0.050519 | 137 | 7 | 1.000000 | 1.000000 |
| strong utility action | 61 | 0.424 | 0.088403 | 0.158066 | -0.069663 | 54 | 7 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 61 | 0.424 | 0.088403 | 0.158066 | -0.069663 | 54 | 7 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 144 | 1.000 | 0.048574 | 0.099093 | -0.050519 | 137 | 7 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `31.0s`, rows `46`
- `54.0s` - `61.0s`, rows `15`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.5`, LSTM `0.0515`, XGBoost `0.2160`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0557`, XGBoost `0.2157`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.0595`, XGBoost `0.2160`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.0603`, XGBoost `0.2160`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.0609`, XGBoost `0.2160`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.0619`, XGBoost `0.2157`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.0623`, XGBoost `0.2157`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0663`, XGBoost `0.2157`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.0613`, XGBoost `0.2086`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.0625`, XGBoost `0.2086`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
